use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, OnceLock};

use bytes::Bytes;
use tracing::instrument;
use url::Url;

use crate::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
#[cfg(test)]
use crate::actions::Add;
use crate::actions::{BackReference, ADD_NAME};
use crate::content_tree::reader::ContentTreeNodeEntryVisitor;
#[cfg(test)]
use crate::content_tree::stats::delta_json_stats_to_content_stats;
use crate::content_tree::stats::{self, aggregate_content_stats};
use crate::content_tree::writer::ContentTreeNodeWriter;
#[cfg(test)]
use crate::content_tree::ManifestInfo;
use crate::content_tree::{
    metadata_entry_to_scalars, ContentTreeNode, ContentTreeNodeEntry, ContentTreeNodeEntryBuilder,
    DataContentType, DeletionVectorInfo, TrackingInfo, TrackingStatus, CONTENT_STATS_FIELD_NAME,
    CONTENT_TYPE, DELTA_STATS_MAX_VALUES, DELTA_STATS_MIN_VALUES, DELTA_STATS_NULL_COUNT,
    DELTA_STATS_NUM_RECORDS, DELTA_STATS_TIGHT_BOUNDS, DV_INFO, FILE_FORMAT, FILE_SIZE_IN_BYTES,
    LOCATION, PARTITION, PARTITION_SPEC_ID, RECORD_COUNT, SORT_ORDER_ID, TAGS, TRACKING,
};
use crate::engine_data::{FilteredRowVisitor, GetData, RowVisitor, TypedGetData as _};
use crate::expressions::{ArrayData, Expression, Predicate, Scalar, Transform};
use crate::log_replay::{ActionsBatch, FileActionKey, LogReplayProcessor};
use crate::row_tracking::CursorRowIdAllocator;
use crate::scan::data_skipping::DataSkippingFilter;
use crate::scan::log_replay::{
    DEFAULT_ROW_COMMIT_VERSION_NAME, FILE_CONSTANT_VALUES_NAME, STATS_PARSED_NAME,
};
use crate::schema::{
    column_name, ArrayType, ColumnMetadataKey, ColumnName, ColumnNamesAndTypes, DataType, MapType,
    MetadataValue, Schema, SchemaRef, StructField, StructType,
};
use crate::utils::require;
#[cfg(test)]
use crate::utils::try_parse_uri;
use crate::{
    DeltaResult, Engine, EngineData, Error, ExpressionEvaluator, FilteredEngineData, Version,
};

/// Magic number for the Roaring bitmap portable format, stored as big-endian bytes.
const ROARING_BITMAP_PORTABLE_MAGIC_BYTES: [u8; 4] = 1681511377u32.to_be_bytes();
const ROARING_BITMAP_PORTABLE_MAGIC_LEN: usize = ROARING_BITMAP_PORTABLE_MAGIC_BYTES.len();

/// Helper function to serialize a RoaringTreemap with the portable magic number prefix.
fn serialize_roaring_treemap(treemap: &roaring::RoaringTreemap) -> DeltaResult<Bytes> {
    let mut serialized =
        Vec::with_capacity(ROARING_BITMAP_PORTABLE_MAGIC_LEN + treemap.serialized_size());
    serialized.extend_from_slice(&ROARING_BITMAP_PORTABLE_MAGIC_BYTES);
    treemap.serialize_into(&mut serialized).map_err(|e| {
        Error::generic(format!("Failed to serialize deletion vector bitmap: {}", e))
    })?;
    Ok(Bytes::from(serialized))
}

/// Helper function to deserialize a RoaringTreemap from bytes with magic number prefix.
fn deserialize_roaring_treemap(bytes: &Bytes) -> DeltaResult<roaring::RoaringTreemap> {
    if bytes.len() < ROARING_BITMAP_PORTABLE_MAGIC_LEN {
        return Err(Error::generic(format!(
            "Invalid manifest DV: bytes too small (less than {} bytes)",
            ROARING_BITMAP_PORTABLE_MAGIC_LEN
        )));
    }
    roaring::RoaringTreemap::deserialize_from(&bytes[ROARING_BITMAP_PORTABLE_MAGIC_LEN..]).map_err(
        |e| {
            Error::generic(format!(
                "Failed to deserialize deletion vector bitmap: {}",
                e
            ))
        },
    )
}

/// Builds the partition struct type for the AMT partition field from a table's partition columns
/// and logical schema. Field names are logical column names (matching Delta's `partitionValues`
/// map keys), and each field carries its `PARQUET:field_id` metadata from the source column.
/// This follows the Iceberg convention where partition struct field IDs correspond to partition
/// spec field IDs (in Delta's case, the source column's field ID since all transforms are
/// identity). All fields are nullable since `MapToStruct` map lookups can return null. Returns
/// `None` if `partition_columns` is empty.
///
/// The returned type is self-contained and requires no further schema fixup. Column mapping
/// metadata (physicalName, id) is intentionally not propagated because the partition struct is
/// written by kernel and always uses logical names directly as Parquet field names.
pub(crate) fn build_partition_type(
    partition_columns: &[String],
    logical_schema: &StructType,
) -> Option<StructType> {
    if partition_columns.is_empty() {
        return None;
    }
    let fields: Vec<StructField> = partition_columns
        .iter()
        .filter_map(|col_name| {
            let field = logical_schema.field(col_name)?;
            let mut partition_field = StructField::nullable(col_name, field.data_type().clone());
            if let Some(MetadataValue::Number(id)) = field
                .metadata
                .get(ColumnMetadataKey::ParquetFieldId.as_ref())
            {
                partition_field = partition_field
                    .add_metadata([(ColumnMetadataKey::ParquetFieldId.as_ref(), *id)]);
            }
            Some(partition_field)
        })
        .collect();
    if fields.is_empty() {
        return None;
    }
    Some(StructType::new_unchecked(fields))
}

/// Extracts deletion vector content from a DeletionVectorDescriptor.
///
/// This function decodes the `path_or_inline_dv` field based on the storage type:
///
/// - `PersistedRelative`: The format is `<random prefix - optional><base85 encoded uuid>`. The UUID
///   is 20 characters (base85 encoded), and any characters before that are the optional random
///   prefix. The function reconstructs the absolute path to the DV file.
///
/// - `PersistedAbsolute`: The `path_or_inline_dv` contains the absolute path to the DV file.
///
/// - `Inline`: Currently not supported - returns an error. Inline DVs would need to be persisted
///   first before being added to metadata.
///
/// # Format Differences: Delta vs Iceberg
///
/// Both Delta and Iceberg use the Roaring bitmap Portable format for deletion vectors:
/// <https://github.com/RoaringBitmap/RoaringFormatSpec?tab=readme-ov-file#extension-for-64-bit-implementations>
///
/// However, the `size_in_bytes` field has different semantics:
///
/// **Delta format** (<https://github.com/delta-io/delta/blob/master/PROTOCOL.md#deletion-vector-format>):
/// - `size_in_bytes` represents only the size of the serialized Roaring bitmap data
/// - The binary layout is: `[4-byte size prefix][bitmap data][4-byte CRC checksum]`
/// - Delta's `size_in_bytes` excludes the 4-byte size prefix and 4-byte CRC
///
/// **Iceberg format** (<https://iceberg.apache.org/puffin-spec/#deletion-vector-v1-blob-type>):
/// - `size_in_bytes` represents the total blob size including all framing
/// - This includes the size prefix + bitmap data + CRC checksum
///
/// Therefore, when converting from Delta to Iceberg's [`DeletionVectorInfo`], we add 8 bytes
/// (4 for size prefix + 4 for CRC) to Delta's `size_in_bytes`.
///
/// # Returns
/// A [`DeletionVectorInfo`] containing the DV location and size information.
pub(crate) fn extract_deletion_vector_content(
    dv: &DeletionVectorDescriptor,
) -> DeltaResult<DeletionVectorInfo> {
    let location = match dv.storage_type {
        DeletionVectorStorageType::PersistedAbsolute => {
            // Use absolute path as-is
            dv.path_or_inline_dv.clone()
        }
        DeletionVectorStorageType::PersistedRelative => {
            // Decode to relative path
            dv.relative_path()?
        }
        DeletionVectorStorageType::Inline => {
            return Err(Error::DeletionVector(
                "Inline deletion vectors are not supported. They must be persisted first."
                    .to_string(),
            ));
        }
    };
    // Add 8 bytes to convert from Delta's size (bitmap only) to Iceberg's size (full blob):
    // - 4 bytes: size prefix
    // - 4 bytes: CRC checksum
    Ok(DeletionVectorInfo {
        location,
        offset: dv.offset.map(|v| v as i64).unwrap_or(0),
        size_in_bytes: dv.size_in_bytes as i64 + 8,
        cardinality: dv.cardinality,
    })
}

/// Cache for DV bitmaps with lazy deserialization
struct DvCache {
    /// Original serialized manifest_dv bytes (from previous commits)
    /// Kept as reference (Bytes is Rc-based, cheap to clone)
    serialized_manifest_dv: Option<Bytes>,

    /// Lazily deserialized manifest_dv (only populated when modified)
    manifest_dv: Option<roaring::RoaringTreemap>,

    /// Positions deleted in the current commit (always starts empty).
    deleted_positions: roaring::RoaringTreemap,

    /// Positions replaced (DV changed) in the current commit (always starts empty).
    replaced_positions: roaring::RoaringTreemap,

    /// Track if this entry was modified (deserialized)
    dirty: bool,

    /// Total number of entries in the manifest (for bounds checking)
    /// Cached from manifest_info to avoid O(n) scans
    total_entry_count: i64,
}

impl DvCache {
    fn new(serialized_manifest_dv: Option<Bytes>, total_entry_count: i64) -> Self {
        Self {
            serialized_manifest_dv,
            manifest_dv: None,
            deleted_positions: roaring::RoaringTreemap::new(),
            replaced_positions: roaring::RoaringTreemap::new(),
            dirty: false,
            total_entry_count,
        }
    }

    /// Deserialize manifest_dv on first access
    fn ensure_manifest_dv_loaded(&mut self) -> DeltaResult<()> {
        if self.manifest_dv.is_some() {
            return Ok(());
        }

        let dv = if let Some(ref bytes) = self.serialized_manifest_dv {
            deserialize_roaring_treemap(bytes)?
        } else {
            roaring::RoaringTreemap::new()
        };

        self.manifest_dv = Some(dv);
        self.dirty = true;
        Ok(())
    }

    /// Unions `indices` into the cumulative `manifest_dv` and, per `update_kind`, into the
    /// matching per-commit bitmap.
    fn apply_position_update(
        &mut self,
        indices: &roaring::RoaringTreemap,
        update_kind: LeafPositionUpdate,
    ) -> DeltaResult<()> {
        self.ensure_manifest_dv_loaded()?;
        self.dirty = true;
        let manifest_dv = self.manifest_dv.as_mut().ok_or_else(|| {
            Error::generic("Internal bug: manifest_dv not loaded after ensure_manifest_dv_loaded")
        })?;
        *manifest_dv |= indices;
        match update_kind {
            LeafPositionUpdate::Delete => self.deleted_positions |= indices,
            LeafPositionUpdate::Replace => self.replaced_positions |= indices,
            LeafPositionUpdate::Carryover => {}
        }
        Ok(())
    }
}

/// How leaf manifest positions changed in the current commit. All variants mask the cumulative
/// `manifest_info.dv`; they differ only in which per-commit bitmap records the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeafPositionUpdate {
    /// Files removed this commit, no replacement. Recorded in `tracking.deleted_positions`.
    Delete,
    /// Files superseded by a re-add this commit (DV change, stats backfill). Recorded in
    /// `tracking.replaced_positions`.
    Replace,
    /// The entry carries over unchanged -- its live version is represented elsewhere (rolled up
    /// into the root, or moved to another leaf by reorganization). No per-commit bitmap is set.
    Carryover,
}

/// Builder for creating [`ContentTreeNode`] instances based on V4 ContentTreeNode
pub(crate) struct ContentTreeNodeBuilder {
    table_root: Url,
    pending_entries: Vec<ContentTreeNodeEntry>,
    version: Version,
    /// Table schema for converting stats JSON to content_stats format.
    /// The builder will populate content_stats from the Delta JSON stats blob.
    /// This schema must match the schema used to write the files and must include
    /// PARQUET:field_id metadata on fields for proper stats mapping.
    table_schema: Schema,
    /// Partition type for the partition tuple field in the AMT. Built from the table's partition
    /// columns and their data types. `None` for unpartitioned tables. Field names are logical
    /// column names (matching the keys in Delta's `partitionValues` map). Fields carry
    /// `PARQUET:field_id` metadata from the source column for Iceberg compatibility.
    partition_type: Option<StructType>,
    /// Set of seen file paths to prevent duplicate entries.
    /// Only populated when processing existing actions, not new actions.
    values_seen: HashSet<String>,
    /// Cached schema with content_stats. Computed lazily on first use.
    cached_schema: OnceLock<SchemaRef>,
    /// Combined cache for DV bitmaps (manifest_dv + deleted_positions)
    /// Keyed by manifest location. Provides O(1) access and lazy deserialization.
    dv_cache: HashMap<String, DvCache>,
    /// Pre-transformed EngineData batches already in ContentTreeNodeEntry schema.
    /// These bypass the row-by-row visitor path and are produced by the expression
    /// evaluator in `add_from_engine_data_write`.
    pre_built_data: Vec<Box<dyn EngineData>>,
    /// Aggregate stats for each pre-built data batch, computed at add time.
    pre_built_aggregates: Vec<BatchAggregates>,
}

/// Lightweight aggregate stats computed when adding pre-built columnar batches.
struct BatchAggregates {
    added_file_count: i32,
    existing_file_count: i32,
    total_record_count: i64,
}

/// Converts a `usize` length to an `i32` file count, returning an error on overflow.
fn file_count_from_len(len: usize) -> DeltaResult<i32> {
    len.try_into()
        .map_err(|_| Error::generic(format!("file count {len} exceeds i32::MAX")))
}

impl std::fmt::Debug for ContentTreeNodeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentTreeNodeBuilder")
            .field("table_root", &self.table_root)
            .field("pending_entries", &self.pending_entries.len())
            .field("pre_built_data", &self.pre_built_data.len())
            .field("dv_cache_count", &self.dv_cache.len())
            .field(
                "dv_cache_dirty_count",
                &self.dv_cache.values().filter(|c| c.dirty).count(),
            )
            .finish()
    }
}

/// Builder that can be created from an empty state, or from existing metadata
impl ContentTreeNodeBuilder {
    /// Creates a new ContentTreeNodeBuilder for the given table root and version.
    ///
    /// # Arguments
    /// * `table_root` - The root URL of the table
    /// * `version` - The version of the metadata being built
    /// * `table_schema` - The table schema with PARQUET:field_id metadata for stats conversion.
    ///   This parameter is essential for converting Delta JSON stats (minValues, maxValues,
    ///   nullCount) to the content_stats StructData format when adding entries via `add()`. The
    ///   schema must match the schema used to write the files and must include PARQUET:field_id
    ///   metadata on fields for proper stats field mapping
    pub(crate) fn new_for(table_root: Url, version: Version, table_schema: Schema) -> Self {
        Self {
            table_root,
            pending_entries: Vec::new(),
            version,
            table_schema,
            partition_type: None,
            values_seen: HashSet::new(),
            cached_schema: OnceLock::new(),
            dv_cache: HashMap::new(),
            pre_built_data: Vec::new(),
            pre_built_aggregates: Vec::new(),
        }
    }

    /// Sets the partition struct type for the AMT partition field. Built from the table's
    /// partition columns and their data types. Field names must be logical column names
    /// (matching the keys in Delta's `partitionValues` map). Fields should carry
    /// `PARQUET:field_id` metadata matching the source column's field ID.
    pub(crate) fn with_partition_type(mut self, partition_type: Option<StructType>) -> Self {
        self.partition_type = partition_type;
        self
    }

    /// Creates a [`ContentTreeNodeBuilder`] by reading an existing content root parquet file.
    ///
    /// This reads and validates the content root via
    /// [`ContentTreeNode::from_batches_with_version`], then populates a builder from the
    /// validated entries.
    ///
    /// # Arguments
    /// * `engine` - The engine to use for reading the parquet file
    /// * `content_root` - The content root action referencing the manifest file
    /// * `table_root` - The root URL of the table
    /// * `table_schema` - The table schema with PARQUET:field_id metadata for stats conversion
    /// * `new_version` - The version number for the new metadata being built
    /// * `partition_type` - Optional partition struct type for including partition columns in the
    ///   schema
    #[instrument(
        name = "content_tree.read_root_for_txn",
        skip_all,
        fields(path = %content_root.path),
        err
    )]
    pub(crate) fn from_content_root(
        engine: &dyn Engine,
        content_root: &crate::actions::ContentRoot,
        table_root: Url,
        table_schema: Schema,
        new_version: Version,
        partition_type: Option<StructType>,
    ) -> DeltaResult<Self> {
        let content_root_url = table_root
            .join(&content_root.path)
            .map_err(|e| Error::generic(format!("Failed to parse content root URL: {}", e)))?;

        let (read_result_iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &content_root_url,
            content_root.path.clone(),
            None,
            None,
            None,
        )?;

        let data: Vec<Box<dyn EngineData>> = read_result_iter.collect::<DeltaResult<Vec<_>>>()?;

        let node = ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_root.clone(),
        )?;

        let entries = node.entries()?;
        let mut builder = Self::new_for(table_root, new_version, table_schema)
            .with_partition_type(partition_type);
        for entry in entries {
            // Preserve Added only for entries whose sequence_number matches new_version (no-op
            // rebuild); everything else predates this commit and becomes Existing.
            let entry = if entry.tracking.status == TrackingStatus::Added
                && entry.tracking.sequence_number != Some(new_version as i64)
            {
                entry.with_status(TrackingStatus::Existing)
            } else {
                entry
            };
            builder.add_entry(entry);
        }
        Ok(builder)
    }

    /// Ensures a cache entry exists for the given manifest location.
    /// Cache should be populated in add_entry(), so this is mainly a safety check.
    fn ensure_dv_cache_exists(&mut self, manifest_location: &str) -> DeltaResult<()> {
        // Check if cache entry already exists
        if self.dv_cache.contains_key(manifest_location) {
            return Ok(());
        }

        // This shouldn't happen - cache should be populated in add_entry
        Err(Error::generic(format!(
            "Manifest cache not found at location: {}. This is a bug.",
            manifest_location
        )))
    }

    /// Serializes dirty DVs back into the pending entries.
    /// Only serializes entries that were modified (dirty flag set).
    /// Should be called before building to ensure DVs are properly persisted.
    /// Also updates tracking with snapshot_id and sequence numbers based on status.
    fn serialize_dvs_to_entries(&mut self, snapshot_id: i64) -> DeltaResult<()> {
        // Iterate over entries and look up in cache
        for entry in &mut self.pending_entries {
            // Only process manifest entries
            if !matches!(
                entry.content_type,
                DataContentType::DataManifest | DataContentType::DeleteManifest
            ) {
                continue;
            }

            // Look up in cache by location
            let Some(ref location) = entry.location else {
                continue;
            };

            let Some(cache) = self.dv_cache.get(location) else {
                continue;
            };

            // Only serialize if dirty
            if !cache.dirty {
                continue;
            }

            // Serialize manifest DV into manifest_info.dv
            if let Some(ref manifest_dv) = cache.manifest_dv {
                let dv_bytes = serialize_roaring_treemap(manifest_dv)?;
                let cardinality: i64 = manifest_dv.len().try_into().map_err(|_| {
                    crate::Error::generic(format!(
                        "manifest DV cardinality {} exceeds i64::MAX",
                        manifest_dv.len()
                    ))
                })?;

                let manifest_info = entry.manifest_info.as_mut().ok_or_else(|| {
                    crate::Error::generic(
                        "manifest entry has a dirty DV cache but no manifest_info",
                    )
                })?;
                manifest_info.dv = Some(dv_bytes);
                manifest_info.dv_cardinality = Some(cardinality);

                // If all active entries are deleted, mark manifest as Deleted
                if cardinality == manifest_info.active_entry_count() {
                    entry.tracking.status = TrackingStatus::Deleted;
                }
            }

            // Serialize deleted_positions if non-empty
            if !cache.deleted_positions.is_empty() {
                entry.tracking.deleted_positions =
                    Some(serialize_roaring_treemap(&cache.deleted_positions)?);
            }

            // Serialize replaced_positions if non-empty
            if !cache.replaced_positions.is_empty() {
                entry.tracking.replaced_positions =
                    Some(serialize_roaring_treemap(&cache.replaced_positions)?);
            }

            // Update tracking based on status
            // Only update snapshot_id when status is DELETED
            if entry.tracking.status == TrackingStatus::Deleted {
                entry.tracking.snapshot_id = Some(snapshot_id);
            }
        }

        Ok(())
    }

    /// Gets or creates the cached schema with content_stats.
    /// This is computed once and cached for the lifetime of the builder.
    fn get_schema(&self) -> DeltaResult<SchemaRef> {
        // Check if already cached
        if let Some(schema) = self.cached_schema.get() {
            return Ok(schema.clone());
        }

        let delta_stats = build_delta_stats_schema(&self.table_schema);
        let schema = ContentTreeNodeEntry::to_schema_with_content_stats(
            &self.table_schema,
            &delta_stats,
            self.partition_type.as_ref(),
        )?;
        let schema_ref = Arc::new(schema);

        // Try to cache it (ignore if another thread beat us to it)
        let _ = self.cached_schema.set(schema_ref.clone());

        Ok(schema_ref)
    }

    /// Converts a relative path to a data file from the root of the table
    /// Or, when an absolute path it should keep it untouched.
    /// The path is a URI as specified by [RFC 2396 URI Generic Syntax].
    ///
    /// [RFC 2396 URI Generic Syntax]: https://www.ietf.org/rfc/rfc2396.txt
    #[cfg(test)]
    fn path_to_absolute(&self, path: &str) -> Result<String, crate::Error> {
        // Try to parse the path as an absolute URL
        if let Ok(url) = Url::parse(path) {
            // If it parses successfully, it's an absolute URL
            return Ok(url.to_string());
        }

        // Otherwise, it's a relative path - join it with the table root
        let base_url = try_parse_uri(&self.table_root)?;
        let absolute_url = base_url.join(path).map_err(|e| {
            crate::Error::generic(format!(
                "Failed to join path '{}' with table root '{}': {}",
                path, self.table_root, e
            ))
        })?;

        Ok(absolute_url.to_string())
    }

    /// Adds an [`Add`] action as a [`ContentTreeNodeEntry`] with an explicit tracking status,
    /// skipping duplicate paths. `version` is stored as the entry's `sequence_number`.
    #[cfg(test)]
    pub(crate) fn add_with_status(
        &mut self,
        add: Add,
        version: Version,
        snapshot_id: i64,
        status: TrackingStatus,
    ) -> DeltaResult<()> {
        if !self.values_seen.insert(add.path.clone()) {
            return Ok(());
        }

        let dv_content = add
            .deletion_vector
            .as_ref()
            .map(extract_deletion_vector_content)
            .transpose()?;

        let record_count = match add.stats.as_deref() {
            Some(json) => serde_json::from_str::<serde_json::Value>(json)
                .map_err(|e| Error::generic(format!("failed to parse stats JSON: {e}")))?
                .get("numRecords")
                .and_then(serde_json::Value::as_i64)
                .ok_or_else(|| Error::missing_data("numRecords"))?,
            None => 0, /* TODO: Stats must exist containing at least recordCount when
                        * icebergV4MetadataTree is enabled */
        };
        let content_stats = delta_json_stats_to_content_stats(
            add.stats.as_deref(),
            &self.table_schema,
            add.deletion_vector.is_some().then_some(false),
        )?;

        let partition = self.build_partition_data(&add.partition_values)?;

        let mut builder = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location(add.path)
            .with_tracking(status, version, snapshot_id)
            .deletion_vector_opt(dv_content)
            .record_count(record_count)
            .file_size_in_bytes(add.size)
            .content_stats_opt(content_stats)
            .tags_opt(add.tags);
        if let Some(partition) = partition {
            builder = builder.partition(partition);
        }
        self.pending_entries.push(builder.build());
        Ok(())
    }

    /// Adds an [`Add`] action as a new file (status = [`TrackingStatus::Added`]).
    /// `version` is stored as the entry's `sequence_number`.
    #[cfg(test)]
    pub(crate) fn add(&mut self, add: Add, version: Version, snapshot_id: i64) -> DeltaResult<()> {
        self.add_with_status(add, version, snapshot_id, TrackingStatus::Added)
    }

    /// Builds a [`StructData`] from the `partitionValues` map of an [`Add`] action, using the
    /// builder's `partition_type` to determine field names and types. Returns `None` if the
    /// table is unpartitioned.
    #[cfg(test)]
    fn build_partition_data(
        &self,
        partition_values: &HashMap<String, String>,
    ) -> DeltaResult<Option<crate::expressions::StructData>> {
        let partition_type = match &self.partition_type {
            Some(pt) if pt.fields().len() > 0 => pt,
            _ => return Ok(None),
        };
        let mut fields = Vec::new();
        let mut values = Vec::new();
        for field in partition_type.fields() {
            fields.push(field.clone());
            let scalar = match partition_values.get(field.name()) {
                Some(raw_val) => match field.data_type() {
                    DataType::Primitive(p) => p.parse_scalar(raw_val)?,
                    other => {
                        return Err(Error::generic(format!(
                            "partition field '{}' has non-primitive type {other:?}",
                            field.name()
                        )))
                    }
                },
                None => Scalar::Null(field.data_type().clone()),
            };
            values.push(scalar);
        }
        Ok(Some(crate::expressions::StructData::try_new(
            fields, values,
        )?))
    }

    /// Adds write metadata from `EngineData` to the metadata using columnar transformation.
    ///
    /// This method transforms the input write metadata (path, partitionValues, size,
    /// modificationTime, stats) directly into ContentTreeNodeEntry schema using the engine's
    /// expression evaluator, avoiding the row-by-row visitor pattern.
    ///
    /// When stats are in AMT format (after successful `try_pre_convert_stats_column`),
    /// the full stats are passed through and record counts are extracted. When stats
    /// are not in AMT format (e.g., empty or unconverted), content_stats is set to null
    /// and record_count defaults to 0.
    ///
    /// The write metadata input schema does not carry `tags`, so entries produced by this method
    /// always have `tags = None`. See the inline TODO in `evaluate_write_transform` for how to
    /// extend this once `add_files` exposes tags.
    ///
    /// # Arguments
    /// * `engine` - The engine to use for expression evaluation
    /// * `engine_data` - The engine data containing write metadata records to extract and add
    /// * `version` - The version at which these files are being added
    /// * `snapshot_id` - Optional snapshot ID to use for tracking info
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if there was an error evaluating the expression
    pub(crate) fn add_from_engine_data_write(
        &mut self,
        engine: &dyn crate::Engine,
        engine_data: &dyn EngineData,
        version: Version,
        snapshot_id: i64,
    ) -> DeltaResult<()> {
        if engine_data.is_empty() {
            return Ok(());
        }

        let output_schema = self.get_schema()?;
        let stats_struct = stats::stats_schema(&self.table_schema)?;

        // Try fast path: full AMT stats schema (works when stats were pre-converted)
        let result = self.evaluate_write_transform(
            engine,
            engine_data,
            version,
            snapshot_id,
            &output_schema,
            Some(&stats_struct),
        );

        match result {
            Ok((transformed, agg)) => {
                self.pre_built_data.push(transformed);
                self.pre_built_aggregates.push(agg);
                Ok(())
            }
            Err(_) => {
                // Fall back: empty stats schema (stats not in AMT format)
                let (transformed, agg) = self.evaluate_write_transform(
                    engine,
                    engine_data,
                    version,
                    snapshot_id,
                    &output_schema,
                    None,
                )?;
                self.pre_built_data.push(transformed);
                self.pre_built_aggregates.push(agg);
                Ok(())
            }
        }
    }

    /// Build and evaluate a write metadata transformation expression.
    ///
    /// When `stats_struct` is `Some`, the input schema includes the AMT stats struct
    /// and content_stats/recordCount are derived from it. When `None`, stats are treated
    /// as an empty struct and content_stats is null with recordCount = 0.
    fn evaluate_write_transform(
        &self,
        engine: &dyn crate::Engine,
        engine_data: &dyn EngineData,
        version: Version,
        snapshot_id: i64,
        output_schema: &SchemaRef,
        stats_struct: Option<&StructType>,
    ) -> DeltaResult<(Box<dyn EngineData>, BatchAggregates)> {
        let stats_type = match stats_struct {
            Some(ss) => DataType::Struct(Box::new(ss.clone())),
            None => DataType::Struct(Box::new(StructType::new_unchecked(vec![]))),
        };

        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::not_null("path", DataType::STRING),
            StructField::not_null(
                "partitionValues",
                MapType::new(DataType::STRING, DataType::STRING, true),
            ),
            StructField::not_null("size", DataType::LONG),
            StructField::not_null("modificationTime", DataType::LONG),
            StructField::nullable("stats", stats_type),
        ]));

        let (record_count, content_stats) = match stats_struct {
            Some(ss) => {
                let rc = match ss.fields().next() {
                    Some(first_col) => Expression::coalesce([
                        Expression::column([
                            "stats",
                            first_col.name().as_str(),
                            crate::content_tree::VALUE_COUNT,
                        ]),
                        Expression::literal(Scalar::Long(0)),
                    ]),
                    None => Expression::literal(Scalar::Long(0)),
                };
                (rc, Some(Expression::column(["stats"])))
            }
            None => (Expression::literal(Scalar::Long(0)), None),
        };

        let partition = self
            .partition_type
            .as_ref()
            .map(|_| Expression::map_to_struct(Expression::column(["partitionValues"])));

        let projections = ContentTreeEntryProjections {
            status: TrackingStatus::Added,
            snapshot_id,
            location: Expression::column(["path"]),
            file_size_in_bytes: Expression::column(["size"]),
            sequence_number: Expression::literal(Scalar::Long(version as i64)),
            dv_info: None,
            record_count,
            content_stats,
            partition,
            // TODO: Thread tags through the blind-append write path. The write metadata input
            // schema (path, partitionValues, size, modificationTime, stats) does not carry tags,
            // so tag values written via `add_from_engine_data_write` are silently dropped. If
            // connector-supplied tags need to be preserved in AMT entries on blind-append commits,
            // the write metadata schema and this projection would need to be extended.
            tags_expr: None,
        };

        let evaluator = engine.evaluation_handler().new_expression_evaluator(
            input_schema,
            Arc::new(build_content_tree_entry_expression(
                output_schema,
                &projections,
            )),
            DataType::Struct(Box::new(output_schema.as_ref().clone())),
        )?;
        let transformed = evaluator.evaluate(engine_data)?;

        // Compute lightweight aggregates from the transformed output (flat i64 columns)
        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(transformed.as_ref())?;

        let aggregates = BatchAggregates {
            added_file_count: file_count_from_len(engine_data.len())?,
            existing_file_count: 0,
            total_record_count: agg_visitor.total_record_count,
        };

        Ok((transformed, aggregates))
    }

    /// Adds a raw ContentTreeNodeEntry to the builder.
    ///
    /// This is useful when copying entries from existing metadata.
    pub(crate) fn add_entry(&mut self, mut entry: ContentTreeNodeEntry) {
        // Create DvCache for manifest entries
        if matches!(
            entry.content_type,
            DataContentType::DataManifest | DataContentType::DeleteManifest
        ) {
            if let Some(ref location) = entry.location {
                // Get total entry count from manifest_info for bounds checking
                let total_entry_count = entry
                    .manifest_info
                    .as_ref()
                    .map_or(0, |mi| mi.total_entry_count());

                // Read DV bytes from manifest_info.dv, clone into cache
                // Bytes is Rc-based, so clone is cheap (just increments refcount)
                let dv_bytes = entry.manifest_info.as_ref().and_then(|mi| mi.dv.clone());
                let cache = DvCache::new(dv_bytes, total_entry_count);
                self.dv_cache.insert(location.clone(), cache);

                // Always clear per-commit position tracking (starts empty for new commit)
                entry.tracking.deleted_positions = None;
                entry.tracking.replaced_positions = None;
            }
        }

        // Add entry to Vec (manifest_dv serialized bytes kept intact)
        self.pending_entries.push(entry);
    }

    /// Returns true if this builder has any pending entries.
    pub(crate) fn has_entries(&self) -> bool {
        !self.pending_entries.is_empty() || !self.pre_built_data.is_empty()
    }

    /// Returns `true` if the builder has a leaf manifest entry registered at `path`.
    ///
    /// Used to distinguish leaf removes that target the current content root (and must be applied
    /// via [`update_leaf_positions`](Self::update_leaf_positions)) from removes that
    /// reference an older content root (which are already handled by file-key deduplication).
    pub(crate) fn has_leaf_manifest(&self, path: &str) -> bool {
        self.dv_cache.contains_key(path)
    }

    /// Remove data file entries by path. Only used when moving values in the root
    /// to the leaves (otherwise mark_deleted should be used).
    ///
    /// This removes entries where the location matches and there is no referenced_file
    /// (i.e., data file entries, not DV entries).
    ///
    /// # Arguments
    /// * `file_path` - The file path to match against entry locations
    pub(crate) fn remove_data_file(&mut self, file_path: &str) -> DeltaResult<()> {
        self.pending_entries.retain(|entry| {
            // Only match data files (location matches, content type is Data)
            let is_data_file = entry.location.as_deref() == Some(file_path)
                && entry.content_type == DataContentType::Data;
            !is_data_file
        });

        // Remove from cache by location
        self.dv_cache.remove(file_path);
        self.values_seen.remove(file_path);
        Ok(())
    }

    /// Remove DV entries by DV location or referenced file. Only used when moving values in the
    /// root to the leaves (otherwise mark deleted it should be used.
    ///
    /// This removes entries where the location OR referenced_file matches the given path.
    /// This handles both standalone DV entries and DV entries that reference data files.
    ///
    /// # Arguments
    /// * `dv_identifier` - The DV path to match (can be location or referenced file)
    pub(crate) fn remove_dv(&mut self, dv_identifier: &str) -> DeltaResult<()> {
        self.pending_entries.retain(|entry| {
            // Match DVs by location (DV info is now inline on data entries)
            let is_dv = entry.location.as_deref() == Some(dv_identifier);
            !is_dv
        });

        // Remove from cache by location
        self.dv_cache.remove(dv_identifier);
        self.values_seen.remove(dv_identifier);
        Ok(())
    }

    /// Clears all data file and DV entries from the root manifest.
    ///
    /// This removes all entries where content_type is Data, PositionDeletes, or EqualityDeletes.
    /// Leaf manifest references (DataManifest, DeleteManifest, ManifestDV) are preserved.
    ///
    /// This is used when the client takes control of root/leaf separation via
    /// Transaction::release_root_and_delta_actions(). The client will re-add files
    /// to the appropriate leaves, so we clear the root to avoid duplicates.
    ///
    /// Note: When metadata is loaded from a content root, it only contains entries for:
    /// - Data files in the root manifest
    /// - DVs in the root manifest
    /// - Leaf manifest references (DataManifest, DeleteManifest, ManifestDV)
    ///
    /// Data files inside leaf manifests are not loaded into pending_entries - they're stored
    /// in separate parquet files referenced by the manifest entries.
    pub(crate) fn clear_root_data_and_dv_entries(&mut self) {
        // Collect locations of entries being removed
        let removed_locations: Vec<String> = self
            .pending_entries
            .iter()
            .filter(|entry| {
                !matches!(
                    entry.content_type,
                    DataContentType::DataManifest | DataContentType::DeleteManifest
                )
            })
            .filter_map(|entry| entry.location.clone())
            .collect();

        self.pending_entries.retain(|entry| {
            // Keep only manifest reference entries (these point to leaf manifests)
            // Remove actual data/DV entries from root
            matches!(
                entry.content_type,
                DataContentType::DataManifest | DataContentType::DeleteManifest
            )
        });

        // Remove from cache
        for location in removed_locations {
            self.dv_cache.remove(&location);
        }

        // Clear values_seen since we removed root entries
        // Note: We keep the HashSet structure but clear it because we want to track
        // deduplication for entries added after this point
        self.values_seen.clear();

        // Clear pre-built data (these are data file entries, not manifest references)
        self.pre_built_data.clear();
        self.pre_built_aggregates.clear();
    }

    /// Mutates the `deletion_vector` field of a non-deleted Data entry in `pending_entries`
    /// whose `location` matches `file_path`. All other fields -- including `tracking.status`,
    /// `tracking.sequence_number`, and `tracking.snapshot_id` -- are left untouched.
    ///
    /// Used by the DV-update flow during manifest commits. A DV update is metadata-only, so
    /// the original `sequence_number` is preserved rather than reset to the current
    /// `commit_version` that a delete + re-add would imply.
    ///
    /// # Returns
    /// * `true` if a matching entry was found and updated in place.
    /// * `false` if no matching entry exists (caller should fall back to the delete + re-add flow,
    ///   e.g. for leaf-resident files whose entries aren't in `pending_entries`).
    pub(crate) fn update_dv(
        &mut self,
        file_path: &str,
        new_dv: Option<DeletionVectorInfo>,
    ) -> bool {
        for entry in &mut self.pending_entries {
            if entry.location.as_deref() == Some(file_path)
                && entry.content_type == DataContentType::Data
                && entry.tracking.status != TrackingStatus::Deleted
            {
                entry.deletion_vector = new_dv;
                return true;
            }
        }
        false
    }

    /// Marks existing entries as DELETED based on a matching file path or deletion vector.
    ///
    /// This method searches through pending entries and updates their tracking status to DELETED
    /// if they match the provided criteria. It's used when processing Remove actions that reference
    /// files in the root manifest.
    ///
    /// # Arguments
    /// * `file_path` - Optional file path to match against entry locations
    /// * `dv_path` - Optional deletion vector path to match
    /// * `snapshot_id` - Optional snapshot ID for the deletion tracking info
    pub(crate) fn mark_deleted(
        &mut self,
        file_path: Option<&str>,
        dv_path: Option<&str>,
        snapshot_id: i64,
    ) -> DeltaResult<()> {
        // TODO: we should make pending entries a HashMap<String, ContentTreeNodeEntry> to make this
        // faster
        for entry in &mut self.pending_entries {
            // Check if this entry matches the file path or deletion vector path
            let matches = if let Some(path) = file_path {
                entry.location.as_deref() == Some(path)
            } else if let Some(dv) = dv_path {
                entry.location.as_deref() == Some(dv)
            } else {
                false
            };

            if matches {
                // Update the tracking info to mark as deleted
                entry.tracking.status = TrackingStatus::Deleted;
                entry.tracking.snapshot_id = Some(snapshot_id);
            }
        }

        Ok(())
    }

    /// Invalidates positions in a leaf manifest by masking them in the leaf's ManifestDV.
    ///
    /// Used by the transaction layer when processing manifest DVs from leaf writers.
    ///
    /// # Arguments
    /// * `leaf_file_path` - Path to the leaf manifest file
    /// * `indices` - Roaring bitmap containing the positions to update
    /// * `update_kind` - How to classify `indices` for this commit; see [`LeafPositionUpdate`]
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if the leaf manifest is not found, missing manifest_info, any index is out of
    ///   bounds, or serialization fails
    pub(crate) fn update_leaf_positions(
        &mut self,
        leaf_file_path: &str,
        indices: &roaring::RoaringTreemap,
        update_kind: LeafPositionUpdate,
    ) -> DeltaResult<()> {
        // leaf_file_path is already relative
        // O(1) cache lookup to get/modify bitmaps
        self.ensure_dv_cache_exists(leaf_file_path)?;
        let cache = self.dv_cache.get_mut(leaf_file_path).ok_or_else(|| {
            Error::generic(format!(
                "Internal bug: DV cache not found for manifest after ensure_dv_cache_exists: {}",
                leaf_file_path
            ))
        })?;

        // Validate indices using cached entry count (O(1))
        if let Some(max_index) = indices.max() {
            if max_index >= cache.total_entry_count as u64 {
                return Err(Error::generic(format!(
                    "Index {} out of bounds (total entries: {})",
                    max_index, cache.total_entry_count
                )));
            }
        }

        // tracking will be updated during write_leaf/build when we're already iterating
        cache.apply_position_update(indices, update_kind)
    }

    /// Writes the pending entries as a leaf manifest and returns a ContentTreeNodeEntry referencing
    /// it.
    ///
    /// https://docs.google.com/document/d/1k4x8utgh41Sn1tr98eynDKCWq035SV_f75rtNHcerVw/edit?tab=t.0#heading=h.unn922df0zzw
    ///
    /// This method:
    /// 1. Builds a leaf ContentTreeNode with a unique UUID
    /// 2. Writes it to a parquet file using ContentTreeNodeWriter
    /// 3. Returns a ContentTreeNodeEntry (DataManifest type) that references the written leaf
    ///
    /// The returned ContentTreeNodeEntry can be added to a root manifest to reference this leaf.
    ///
    /// # Arguments
    /// * `engine` - The engine to use for writing the parquet file
    /// * `snapshot_id` - Optional snapshot ID for tracking info
    ///
    /// # Returns
    /// * `Ok(ContentTreeNodeEntry)` - A manifest entry referencing the written leaf file
    /// * `Err` if there was an error building or writing the metadata
    #[instrument(name = "content_tree.write_leaf", skip_all, err)]
    /// Builds and writes a leaf manifest, returning the DataManifest entry for the root.
    ///
    /// Assigns sequential `first_row_id` values to data entries in the leaf using the given
    /// `allocator`. The returned DataManifest entry will have its `first_row_id` set to
    /// the allocator's cursor position at the time of this call. The allocator is advanced past
    /// all assigned row IDs.
    pub(crate) fn write_leaf(
        &mut self,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
        allocator: &mut CursorRowIdAllocator,
    ) -> DeltaResult<ContentTreeNodeEntry> {
        // Capture the starting row ID before build_leaf advances the allocator
        let starting_first_row_id = allocator.current();
        // Build the manifest payload, then write it as a leaf (the writer generates the UUID
        // that disambiguate this manifest's filename from other leaves at the same version).
        let leaf_metadata = self.build(engine, snapshot_id, allocator)?;

        let write_result = ContentTreeNodeWriter::try_new_leaf(leaf_metadata)?.write(engine)?;
        let manifest_path =
            super::relativize_manifest_path(&write_result.location, &self.table_root);
        // Use the actual manifest Parquet file size so bulk_processor can pass it to
        // ParquetObjectReader::with_file_size when reading the leaf manifest back.
        let manifest_file_size = write_result.size_in_bytes as i64;

        // Calculate aggregate stats from pending entries
        let mut record_count: i64 = self.pending_entries.iter().map(|e| e.record_count).sum();

        // Calculate manifest stats (entry counts by status)
        let mut added_files_count = 0i32;
        let mut existing_files_count = 0i32;
        let mut deleted_files_count = 0i32;
        let mut replaced_files_count = 0i32;
        let mut added_rows_count = 0i64;
        let mut existing_rows_count = 0i64;
        let mut deleted_rows_count = 0i64;
        let mut replaced_rows_count = 0i64;
        let mut min_sequence_number = i64::MAX;

        for entry in &self.pending_entries {
            if let Some(seq) = entry.tracking.sequence_number {
                min_sequence_number = min_sequence_number.min(seq);
            }

            match entry.tracking.status {
                TrackingStatus::Added => {
                    added_files_count += 1;
                    added_rows_count += entry.record_count;
                }
                TrackingStatus::Existing => {
                    existing_files_count += 1;
                    existing_rows_count += entry.record_count;
                }
                TrackingStatus::Deleted => {
                    deleted_files_count += 1;
                    deleted_rows_count += entry.record_count;
                }
                // Currently always 0: mark_deleted() uses Deleted for all removals.
                // Per the v4 spec, a Remove+Add for the same file with a new DV should
                // mark the old entry as Replaced, but the transaction layer does not yet
                // correlate removes with adds to distinguish deletes from replacements.
                // TODO: Update mark_deleted() to set Replaced when the same file is re-added
                // with a new DV in the same commit.
                TrackingStatus::Replaced => {
                    replaced_files_count += 1;
                    replaced_rows_count += entry.record_count;
                }
                // A Modified entry is a live file whose deletion vector changed; it contributes
                // its rows, so it is tallied with Existing entries. Not produced until the
                // DV-change flow lands in a later change.
                TrackingStatus::Modified => {
                    existing_files_count += 1;
                    existing_rows_count += entry.record_count;
                }
            }
        }

        // Include pre-built batch aggregates
        for agg in &self.pre_built_aggregates {
            record_count += agg.total_record_count;
            added_files_count += agg.added_file_count;
            existing_files_count += agg.existing_file_count;
            added_rows_count += agg.total_record_count;
            min_sequence_number = min_sequence_number.min(self.version as i64);
        }

        // If no entries, set min_sequence_number to 0
        if min_sequence_number == i64::MAX {
            min_sequence_number = 0;
        }

        let manifest_info = Some(crate::content_tree::ManifestInfo {
            added_files_count,
            existing_files_count,
            deleted_files_count,
            replaced_files_count,
            added_rows_count,
            existing_rows_count,
            deleted_rows_count,
            replaced_rows_count,
            min_sequence_number,
            ..Default::default()
        });

        // Aggregate content_stats from all pending entries
        let content_stats = aggregate_content_stats(
            self.pending_entries
                .iter()
                .map(|e| e.content_stats.as_ref()),
        );

        Ok(
            ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
                .location(manifest_path)
                .tracking(TrackingInfo {
                    status: TrackingStatus::Added,
                    snapshot_id: Some(snapshot_id),
                    // For manifest entries, sequence_number and file_sequence_number are both set
                    // to self.version (the sequence number of the snapshot that adds the manifest)
                    // and must be equal. There is no data-vs-file sequence distinction for
                    // manifests: a manifest is added at a single, known sequence number. Both are
                    // required (non-null) in the root so that leaf data entries can inherit the
                    // value when their own sequence numbers are null and status == Added.
                    sequence_number: Some(self.version as i64),
                    file_sequence_number: Some(self.version as i64),
                    // Set to the starting row ID used for data entries in this leaf
                    first_row_id: Some(starting_first_row_id),
                    dv_snapshot_id: None,
                    deleted_positions: None,
                    replaced_positions: None,
                })
                .record_count(record_count)
                .file_size_in_bytes(manifest_file_size)
                .content_stats_opt(content_stats)
                .manifest_info_opt(manifest_info)
                .build(),
        )
    }

    /// Builds a ContentTreeNode from the builder's accumulated state. The caller decides
    /// whether the resulting metadata is written as a root or leaf manifest by selecting
    /// the appropriate [`ContentTreeNodeWriter`] constructor.
    ///
    /// Assigns sequential `first_row_id` values to entries that don't already have one,
    /// using the given `allocator`. The allocator is advanced past all assigned row IDs.
    pub(crate) fn build(
        &mut self,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
        allocator: &mut CursorRowIdAllocator,
    ) -> DeltaResult<ContentTreeNode> {
        // Serialize all in-memory DVs back to entries
        self.serialize_dvs_to_entries(snapshot_id)?;

        // Assign first_row_id values
        self.assign_first_row_ids_to_pending(allocator);
        self.assign_first_row_ids_pre_built(engine, allocator)?;

        // Use cached schema with content_stats based on table schema
        let schema = self.get_schema()?;

        // Handle empty case early
        if self.pending_entries.is_empty() && self.pre_built_data.is_empty() {
            return Ok(ContentTreeNode {
                table_root: self.table_root.clone(),
                data: vec![],
                version: self.version,
                path_in_log: String::new(),
            });
        }

        let mut data: Vec<Box<dyn EngineData>> = Vec::new();

        // Add scalar-built batch from pending_entries (existing path)
        if !self.pending_entries.is_empty() {
            let fields_per_row = schema.fields().len();
            let mut all_scalars = Vec::with_capacity(self.pending_entries.len() * fields_per_row);
            for entry in &self.pending_entries {
                let scalars = metadata_entry_to_scalars(entry, &schema)?;
                all_scalars.extend(scalars);
            }
            let scalar_row_refs: Vec<&[Scalar]> = all_scalars.chunks(fields_per_row).collect();
            let evaluation_handler = engine.evaluation_handler();
            let engine_data = evaluation_handler.create_many(schema.clone(), &scalar_row_refs)?;
            data.push(engine_data);
        }

        // Add pre-transformed columnar batches
        data.append(&mut self.pre_built_data);

        Ok(ContentTreeNode {
            table_root: self.table_root.clone(),
            data,
            version: self.version,
            path_in_log: String::new(), // Will be set when written
        })
    }

    /// Transforms scan rows into ContentTreeNodeEntry schema, using `TrackingStatus::Existing`
    /// for all rows. `scan_row_input_schema` must include a `stats_parsed` field (Delta JSON
    /// format: `{numRecords, minValues, maxValues, nullCount, tightBounds}`), which is
    /// converted to AMT format for `content_stats` using
    /// [`build_content_stats_from_delta_stats_parsed`].
    ///
    /// If `scan_row_input_schema` has `_dv_location` (flat decoded DV columns appended by
    /// `add_from_existing_scan_rows`), `deletionVector` is projected from those columns with a
    /// nullability predicate so non-DV rows produce a null struct. Otherwise `deletionVector` is
    /// null.
    fn evaluate_scan_row_transform(
        &self,
        engine: &dyn Engine,
        engine_data: &dyn EngineData,
        scan_row_input_schema: &SchemaRef,
        version: Version,
        snapshot_id: i64,
    ) -> DeltaResult<(Box<dyn EngineData>, BatchAggregates)> {
        let output_schema = self.get_schema()?;
        let stats_struct = stats::stats_schema(&self.table_schema)?;

        // Detect whether flat decoded DV columns are present in the input schema.
        let has_decoded_dv = scan_row_input_schema.field(DV_LOCATION).is_some();

        // TODO: Thread partitionValues through the scan-row projection pipeline so partition
        // data can be populated from existing scan rows. Currently the step2_input_schema
        // only projects {path, size, stats, stats_parsed} and partitionValues is dropped.
        let tags_expr = scan_row_input_schema
            .field(TAGS)
            .is_some()
            .then(|| Expression::column([TAGS]));
        let projections = ContentTreeEntryProjections {
            status: TrackingStatus::Existing,
            snapshot_id,
            location: Expression::column(["path"]),
            file_size_in_bytes: Expression::column(["size"]),
            sequence_number: Expression::literal(Scalar::Long(version as i64)),
            dv_info: has_decoded_dv.then(flat_dv_columns_to_dv_info_expr),
            // stats_parsed is always present (Delta format); record_count reads numRecords
            // directly, content_stats converts Delta -> AMT via expressions.
            record_count: Expression::coalesce([
                Expression::column([STATS_PARSED_NAME, DELTA_STATS_NUM_RECORDS]),
                Expression::literal(Scalar::Long(0)),
            ]),
            content_stats: Some(build_content_stats_from_delta_stats_parsed(
                &self.table_schema,
                &stats_struct,
            )?),
            partition: None,
            tags_expr,
        };

        let evaluator = engine.evaluation_handler().new_expression_evaluator(
            scan_row_input_schema.clone(),
            Arc::new(build_content_tree_entry_expression(
                &output_schema,
                &projections,
            )),
            DataType::Struct(Box::new(output_schema.as_ref().clone())),
        )?;
        let transformed = evaluator.evaluate(engine_data)?;

        // Compute aggregates from the transformed output
        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(transformed.as_ref())?;

        let aggregates = BatchAggregates {
            added_file_count: 0,
            existing_file_count: file_count_from_len(engine_data.len())?,
            total_record_count: agg_visitor.total_record_count,
        };

        Ok((transformed, aggregates))
    }

    /// Assigns `first_row_id` to entries that need it, following the spec rules:
    /// - Preserve existing `first_row_id` for entries that already have one
    /// - Assign sequential IDs for Data and DataManifest entries without `first_row_id`
    ///
    /// Uses the given `allocator` to reserve row ID ranges. For entries with existing IDs,
    /// the allocator cursor is advanced past their range without allocating new IDs.
    fn assign_first_row_ids_to_pending(&mut self, allocator: &mut CursorRowIdAllocator) {
        for entry in &mut self.pending_entries {
            let ti = &mut entry.tracking;

            // Deleted entries preserve their existing first_row_id but don't consume row IDs
            if ti.status == TrackingStatus::Deleted {
                continue;
            }

            match entry.content_type {
                DataContentType::Data if ti.first_row_id.is_none() => {
                    ti.first_row_id = Some(allocator.reserve_row_ids(entry.record_count));
                }
                DataContentType::DataManifest if ti.first_row_id.is_none() => {
                    let row_increment = entry
                        .manifest_info
                        .as_ref()
                        .map(|mi| mi.added_rows_count + mi.existing_rows_count)
                        .unwrap_or(0);
                    ti.first_row_id = Some(allocator.reserve_row_ids(row_increment));
                }
                // PositionDeletes, EqualityDeletes, or already assigned: no-op
                _ => {}
            }
        }
    }

    /// Assigns `first_row_id` to pre-built EngineData batches that contain opaque columnar data.
    ///
    /// Uses a visitor to read `recordCount` per row, computes sequential `first_row_id` values
    /// via the given `allocator`, then uses `append_columns` + expression evaluator to replace
    /// `tracking.firstRowId`.
    fn assign_first_row_ids_pre_built(
        &mut self,
        engine: &dyn crate::Engine,
        allocator: &mut CursorRowIdAllocator,
    ) -> DeltaResult<()> {
        if self.pre_built_data.is_empty() {
            return Ok(());
        }

        let output_schema = self.get_schema()?;
        let mut new_pre_built = Vec::with_capacity(self.pre_built_data.len());

        for batch in self.pre_built_data.drain(..) {
            // Step 1: Visit to get record counts per row
            let mut record_counts_visitor = RecordCountVisitor::with_capacity(batch.len());
            record_counts_visitor.visit_rows_of(batch.as_ref())?;

            // Step 2: Compute first_row_id for each row, preserving existing non-null values.
            // This mirrors `assign_first_row_ids_to_pending` which checks `is_none()` before
            // allocating.
            let mut first_row_ids = Vec::with_capacity(record_counts_visitor.record_counts.len());
            for (rc, existing_id) in record_counts_visitor
                .record_counts
                .iter()
                .zip(record_counts_visitor.first_row_ids.iter())
            {
                if let Some(id) = existing_id {
                    first_row_ids.push(*id);
                } else {
                    first_row_ids.push(allocator.reserve_row_ids(*rc));
                }
            }

            // Step 3: Append _first_row_id column to the batch
            let append_schema = Arc::new(StructType::new_unchecked(vec![StructField::nullable(
                "_first_row_id",
                DataType::LONG,
            )]));
            let first_row_id_array =
                ArrayData::try_new(ArrayType::new(DataType::LONG, true), first_row_ids)?;
            let augmented = batch.append_columns(append_schema, vec![first_row_id_array])?;

            // Step 4: Use a Transform to replace tracking.firstRowId with _first_row_id,
            // preserving existing non-null values via coalesce, then drop the helper column.
            // Input schema = output_schema + _first_row_id
            let mut input_fields: Vec<StructField> =
                output_schema.fields().cloned().collect::<Vec<_>>();
            input_fields.push(StructField::nullable("_first_row_id", DataType::LONG));
            let input_schema = Arc::new(StructType::new_unchecked(input_fields));

            let transform_expr = Expression::transform(
                Transform::new_top_level()
                    .with_replaced_field(
                        "tracking",
                        Arc::new(Expression::transform(
                            Transform::new_nested(["tracking"]).with_replaced_field(
                                "firstRowId",
                                Arc::new(Expression::coalesce([
                                    Expression::column(["tracking", "firstRowId"]),
                                    Expression::column(["_first_row_id"]),
                                ])),
                            ),
                        )),
                    )
                    .with_dropped_field("_first_row_id"),
            );
            let evaluator = engine.evaluation_handler().new_expression_evaluator(
                input_schema,
                Arc::new(transform_expr),
                DataType::Struct(Box::new(output_schema.as_ref().clone())),
            )?;
            let result = evaluator.evaluate(augmented.as_ref())?;
            new_pre_built.push(result);
        }

        self.pre_built_data = new_pre_built;
        Ok(())
    }

    /// Adds file metadata from existing scan rows to the leaf manifest.
    ///
    /// Unlike `add_from_engine_data_write` (for new files), this method handles rows from
    /// a scan over an existing Delta table, writing them as `TrackingStatus::Existing` entries.
    ///
    /// The input data must include a `stats_parsed` column (added by `include_stats_columns()` in
    /// the scan). All rows are processed via a single expression-evaluator path:
    /// 1. DV columns are decoded (base85 UUID → relative path, sizes widened to LONG, +8 bytes for
    ///    Iceberg framing) by `DecodedDvVisitor`.
    /// 2. `stats_parsed` is resolved via `coalesce(stats_parsed, parse_json(stats, schema))`:
    ///    prefers the already-parsed struct; falls back to parsing the raw JSON stats string.
    /// 3. The final transform maps the augmented schema → ContentTreeNodeEntry output.
    pub(crate) fn add_from_existing_scan_rows(
        &mut self,
        engine: &dyn Engine,
        engine_data: &dyn EngineData,
        selection_vector: &[bool],
        version: Version,
        snapshot_id: i64,
    ) -> DeltaResult<()> {
        if engine_data.is_empty() {
            return Ok(());
        }

        // Step 1: Detect + decode DV columns in one pass from the original engine_data.
        // (Done first so we can use the original data's nullable DV fields directly.)
        let mut dv_visitor = DecodedDvVisitor::for_scan_rows(engine_data.len());
        dv_visitor.visit_rows_of(engine_data)?;

        // Step 2: Produce {path, size, stats_parsed, tags} via coalesce(stats_parsed,
        // parse_json(stats)). The input is always expected to have a `stats_parsed` column (added
        // by `include_stats_columns()` in the scan). For rows sourced from JSON commits,
        // `stats_parsed` will be null and the coalesce falls back to parsing the `stats` JSON.
        // This mirrors the identical pattern in checkpoint/stats_transform.rs.
        //
        // `tags` is flattened out of `fileConstantValues.tags` into a top-level column so that
        // `evaluate_scan_row_transform` can reference it uniformly.
        let delta_stats_schema = Arc::new(build_delta_stats_schema(&self.table_schema));
        let stats_parsed_type = DataType::Struct(Box::new(delta_stats_schema.as_ref().clone()));
        let tags_map_type = MapType::new(DataType::STRING, DataType::STRING, true);
        let step2_input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable("stats", DataType::STRING),
            StructField::nullable(STATS_PARSED_NAME, stats_parsed_type.clone()),
            StructField::nullable(
                FILE_CONSTANT_VALUES_NAME,
                DataType::Struct(Box::new(StructType::new_unchecked([
                    StructField::nullable(TAGS, DataType::Map(Box::new(tags_map_type.clone()))),
                ]))),
            ),
        ]));
        let stats_augmented_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable(STATS_PARSED_NAME, stats_parsed_type),
            StructField::nullable(TAGS, DataType::Map(Box::new(tags_map_type))),
        ]));
        let parse_stats_expr = Expression::struct_from([
            Expression::column(["path"]),
            Expression::column(["size"]),
            Expression::coalesce([
                Expression::column([STATS_PARSED_NAME]),
                Expression::parse_json(Expression::column(["stats"]), delta_stats_schema),
            ]),
            Expression::column([FILE_CONSTANT_VALUES_NAME, TAGS]),
        ]);
        let stats_evaluator = engine.evaluation_handler().new_expression_evaluator(
            step2_input_schema,
            Arc::new(parse_stats_expr),
            DataType::Struct(Box::new(stats_augmented_schema.as_ref().clone())),
        )?;
        let stats_engine_data = stats_evaluator.evaluate(engine_data)?;

        // Step 3: Append 4 flat _dv_* columns if any DV rows are present; build input schema.
        let (transformed_data, input_schema) = if dv_visitor.has_any_dv() {
            let augmented = dv_visitor.append_decoded_dv_columns(stats_engine_data.as_ref())?;
            let mut fields: Vec<StructField> = stats_augmented_schema.fields().cloned().collect();
            fields.extend(DV_DECODED_FLAT_SCHEMA.fields().cloned());
            let schema = Arc::new(StructType::new_unchecked(fields));
            (augmented, schema)
        } else {
            (stats_engine_data, stats_augmented_schema)
        };

        // Step 4: Final transform → ContentTreeNodeEntry schema.
        let (transformed, _) = self.evaluate_scan_row_transform(
            engine,
            transformed_data.as_ref(),
            &input_schema,
            version,
            snapshot_id,
        )?;

        // Step 5: Apply selection vector, aggregate, and push.
        let filtered = FilteredEngineData::try_new(transformed, selection_vector.to_vec())?
            .apply_selection_vector()?;

        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(filtered.as_ref())?;

        let aggregates = BatchAggregates {
            added_file_count: 0,
            existing_file_count: file_count_from_len(filtered.len())?,
            total_record_count: agg_visitor.total_record_count,
        };
        self.pre_built_data.push(filtered);
        self.pre_built_aggregates.push(aggregates);

        Ok(())
    }

    /// Adds a pre-transformed log batch (already in ContentTreeNodeEntry schema) to this builder.
    ///
    /// Called during AMT rollup to incorporate Add actions replayed from delta log commits.
    /// The input must be in ContentTreeNodeEntry schema — produced by
    /// [`ContentRootRebuildProcessor::process_log_batch`] — with the selection vector already
    /// applied.
    ///
    /// # Arguments
    /// * `data` - ContentTreeNodeEntry-schema engine data with zero or more rows.
    pub(crate) fn add_pre_built_log_batch(&mut self, data: Box<dyn EngineData>) -> DeltaResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(data.as_ref())?;
        let aggregates = BatchAggregates {
            added_file_count: 0,
            existing_file_count: file_count_from_len(data.len())?,
            total_record_count: agg_visitor.total_record_count,
        };
        self.pre_built_data.push(data);
        self.pre_built_aggregates.push(aggregates);
        Ok(())
    }
}

/// Visitor that reads aggregate record count from the transformed output.
/// This reads the flat `recordCount` column that was already computed by the expression
/// evaluator, avoiding the expensive `get_struct()` + `materialize()` per row.
#[derive(Default)]
struct TransformedAggregateVisitor {
    total_record_count: i64,
}

impl RowVisitor for TransformedAggregateVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            let names = vec![column_name!("recordCount")];
            let types = vec![DataType::LONG];
            (names, types).into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        for i in 0..row_count {
            let record_count: i64 = getters[0].get(i, "recordCount")?;
            self.total_record_count += record_count;
        }
        Ok(())
    }
}

/// Visitor that reads per-row record counts and existing `firstRowId` values from pre-built
/// EngineData batches. Used by `assign_first_row_ids_pre_built` to compute sequential
/// first_row_id values while preserving any already-assigned IDs.
struct RecordCountVisitor {
    record_counts: Vec<i64>,
    first_row_ids: Vec<Option<i64>>,
}

impl RecordCountVisitor {
    fn with_capacity(cap: usize) -> Self {
        Self {
            record_counts: Vec::with_capacity(cap),
            first_row_ids: Vec::with_capacity(cap),
        }
    }
}

impl RowVisitor for RecordCountVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        use crate::schema::column_name;
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            let names = vec![
                column_name!("recordCount"),
                column_name!("tracking.firstRowId"),
            ];
            let types = vec![DataType::LONG, DataType::LONG];
            (names, types).into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        for i in 0..row_count {
            let record_count: i64 = getters[0].get(i, "recordCount")?;
            self.record_counts.push(record_count);
            let first_row_id: Option<i64> = getters[1].get_opt(i, "tracking.firstRowId")?;
            self.first_row_ids.push(first_row_id);
        }
        Ok(())
    }
}

/// Returns the minimal schema for reading log commit files during AMT rollup.
///
/// Projects only the fields used by [`LogBatchDedupVisitor`], [`DecodedDvVisitor`], and
/// the `action_evaluator` in [`ContentRootRebuildProcessor`]:
///
/// - `add`: `path`, `size`, `defaultRowCommitVersion`, `stats`, `partitionValues`, `tags`,
///   `deletionVector` (all 5 DV sub-fields for z85 decode)
/// - `remove`: `path`, `deletionVector.{storageType, pathOrInlineDv}` (for key dedup),
///   `backReference` (for leaf-remove accumulation)
pub(crate) fn log_replay_schema() -> SchemaRef {
    let add_dv = DataType::Struct(Box::new(StructType::new_unchecked([
        StructField::nullable("storageType", DataType::STRING),
        StructField::nullable("pathOrInlineDv", DataType::STRING),
        StructField::nullable("offset", DataType::INTEGER),
        StructField::nullable("sizeInBytes", DataType::INTEGER),
        StructField::nullable("cardinality", DataType::LONG),
    ])));
    let remove_dv = DataType::Struct(Box::new(StructType::new_unchecked([
        StructField::nullable("storageType", DataType::STRING),
        StructField::nullable("pathOrInlineDv", DataType::STRING),
    ])));
    Arc::new(StructType::new_unchecked([
        StructField::nullable(
            "add",
            DataType::Struct(Box::new(StructType::new_unchecked([
                StructField::nullable("path", DataType::STRING),
                StructField::nullable("size", DataType::LONG),
                StructField::nullable(DEFAULT_ROW_COMMIT_VERSION_NAME, DataType::LONG),
                StructField::nullable("stats", DataType::STRING),
                StructField::nullable(
                    "partitionValues",
                    MapType::new(DataType::STRING, DataType::STRING, true),
                ),
                StructField::nullable(TAGS, MapType::new(DataType::STRING, DataType::STRING, true)),
                StructField::nullable("deletionVector", add_dv),
            ]))),
        ),
        StructField::nullable(
            "remove",
            DataType::Struct(Box::new(StructType::new_unchecked([
                StructField::nullable("path", DataType::STRING),
                StructField::nullable("deletionVector", remove_dv),
                StructField::nullable("backReference", BackReference::nullable_schema()),
            ]))),
        ),
    ]))
}

/// Builds the Delta JSON stats schema for a given table schema.
///
/// The Delta JSON stats format is:
/// ```json
/// { "numRecords": 100, "minValues": {"col": val, ...}, "maxValues": {"col": val, ...},
///   "nullCount": {"col": 0, ...}, "tightBounds": true }
/// ```
///
/// This schema is used with `Expression::parse_json` to parse the `stats` JSON string
/// from scan rows into a typed struct. Field names match the table schema's field names
/// (physical names when column mapping is enabled).
pub(crate) fn build_delta_stats_schema(
    table_schema: &crate::schema::StructType,
) -> crate::schema::StructType {
    let value_fields: Vec<StructField> = table_schema
        .fields()
        .map(|f| StructField::nullable(f.name(), f.data_type().clone()))
        .collect();
    let null_count_fields: Vec<StructField> = table_schema
        .fields()
        .map(|f| StructField::nullable(f.name(), DataType::LONG))
        .collect();
    // Field order must match `expected_stats_schema` in scan/data_skipping/stats_schema/mod.rs,
    // which is also the order written to parquet checkpoints/sidecars.
    StructType::new_unchecked(vec![
        StructField::nullable(DELTA_STATS_NUM_RECORDS, DataType::LONG),
        StructField::nullable(
            DELTA_STATS_NULL_COUNT,
            DataType::Struct(Box::new(StructType::new_unchecked(null_count_fields))),
        ),
        StructField::nullable(
            DELTA_STATS_MIN_VALUES,
            DataType::Struct(Box::new(StructType::new_unchecked(value_fields.clone()))),
        ),
        StructField::nullable(
            DELTA_STATS_MAX_VALUES,
            DataType::Struct(Box::new(StructType::new_unchecked(value_fields))),
        ),
        StructField::nullable(DELTA_STATS_TIGHT_BOUNDS, DataType::BOOLEAN),
    ])
}

/// Builds a content_stats expression (AMT format) from a `stats_parsed` column in Delta format.
///
/// Converts `stats_parsed: {numRecords, minValues, maxValues, nullCount, tightBounds}` (Delta JSON
/// format) to `content_stats: {col: {lower_bound, upper_bound, tight_bounds, value_count,
/// [null_value_count], ...}}` (AMT format) using struct expressions.
///
/// Column names used for field access come from `amt_schema` field names, which are the
/// `table_schema` field names (physical names when column mapping is enabled, which must match the
/// JSON keys in the Delta stats). `amt_schema` drives the iteration and columns are matched by
/// name: it omits table columns that carry no leaf statistics (array/map, and structs made up
/// solely of them), so pairing the two schemas positionally would shift every column after the
/// first omission onto the wrong statistics.
fn build_content_stats_from_delta_stats_parsed(
    table_schema: &crate::schema::StructType,
    amt_schema: &crate::schema::StructType,
) -> DeltaResult<crate::expressions::Expression> {
    let col_exprs: Vec<Arc<Expression>> = amt_schema
        .fields()
        .map(|amt_field| {
            let col_name = amt_field.name();
            if table_schema.field(col_name).is_none() {
                return Err(crate::Error::generic(format!(
                    "AMT stats field '{col_name}' has no matching column in the table schema"
                )));
            }
            let col_stats_type = match amt_field.data_type() {
                DataType::Struct(s) => s.as_ref().clone(),
                _ => {
                    return Err(crate::Error::generic(format!(
                        "Expected AMT stats field '{}' to be a struct, got {:?}",
                        col_name,
                        amt_field.data_type()
                    )))
                }
            };
            let field_exprs: Vec<Arc<Expression>> = col_stats_type
                .fields()
                .map(|f| {
                    Arc::new(match f.name().as_str() {
                        crate::content_tree::LOWER_BOUND => Expression::column([
                            STATS_PARSED_NAME,
                            DELTA_STATS_MIN_VALUES,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::UPPER_BOUND => Expression::column([
                            STATS_PARSED_NAME,
                            DELTA_STATS_MAX_VALUES,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::TIGHT_BOUNDS => Expression::coalesce([
                            Expression::column([STATS_PARSED_NAME, DELTA_STATS_TIGHT_BOUNDS]),
                            Expression::literal(Scalar::Boolean(true)),
                        ]),
                        crate::content_tree::VALUE_COUNT => {
                            Expression::column([STATS_PARSED_NAME, DELTA_STATS_NUM_RECORDS])
                        }
                        crate::content_tree::NULL_VALUE_COUNT => Expression::column([
                            STATS_PARSED_NAME,
                            DELTA_STATS_NULL_COUNT,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::NAN_VALUE_COUNT => {
                            Expression::null_literal(DataType::LONG)
                        }
                        crate::content_tree::AVG_VALUE_SIZE_IN_BYTES => {
                            Expression::null_literal(DataType::INTEGER)
                        }
                        _ => Expression::null_literal(f.data_type().clone()),
                    })
                })
                .collect();
            Ok(Arc::new(Expression::struct_from(field_exprs)))
        })
        .collect::<DeltaResult<Vec<_>>>()?;
    Ok(Expression::struct_from(col_exprs))
}

/// Intermediate flat decoded-DV columns: path decoded from base85, sizes widened to LONG,
/// `+8` bytes for Iceberg framing.
const DV_LOCATION: &str = "_dv_location";
const DV_OFFSET: &str = "_dv_offset";
const DV_SIZE_IN_BYTES: &str = "_dv_size_in_bytes";
const DV_CARDINALITY: &str = "_dv_cardinality";

/// Schema of [`DV_LOCATION`] etc., for [`EngineData::append_columns`].
static DV_DECODED_FLAT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable(DV_LOCATION, DataType::STRING),
        StructField::nullable(DV_OFFSET, DataType::LONG),
        StructField::nullable(DV_SIZE_IN_BYTES, DataType::LONG),
        StructField::nullable(DV_CARDINALITY, DataType::LONG),
    ]))
});

/// Input schema for [`build_log_batch_evaluator`].
static LOG_BATCH_EVALUATOR_INPUT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(StructType::new_unchecked([StructField::nullable(
        ADD_NAME,
        DataType::Struct(Box::new(StructType::new_unchecked([
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable(DEFAULT_ROW_COMMIT_VERSION_NAME, DataType::LONG),
            StructField::nullable("stats", DataType::STRING),
            StructField::nullable(
                "partitionValues",
                MapType::new(DataType::STRING, DataType::STRING, true),
            ),
            StructField::nullable(TAGS, MapType::new(DataType::STRING, DataType::STRING, true)),
        ]))),
    )]))
});

/// `add.{path, size, defaultRowCommitVersion, partitionValues, tags}`, threaded from the
/// log-batch evaluator into the action-to-entry evaluator.
fn log_add_projection_field() -> StructField {
    StructField::nullable(
        ADD_NAME,
        DataType::Struct(Box::new(StructType::new_unchecked([
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable(DEFAULT_ROW_COMMIT_VERSION_NAME, DataType::LONG),
            StructField::nullable(
                "partitionValues",
                MapType::new(DataType::STRING, DataType::STRING, true),
            ),
            StructField::nullable(TAGS, MapType::new(DataType::STRING, DataType::STRING, true)),
        ]))),
    )
}

/// `dvInfo` from the flat decoded-DV columns; null when `_dv_location` is null (no DV).
fn flat_dv_columns_to_dv_info_expr() -> Expression {
    Expression::struct_with_nullability_from(
        [
            Expression::column([DV_LOCATION]),
            Expression::column([DV_OFFSET]),
            Expression::column([DV_SIZE_IN_BYTES]),
            Expression::column([DV_CARDINALITY]),
        ],
        Expression::from_pred(Predicate::is_not_null(Expression::column([DV_LOCATION]))),
    )
}

/// Per-call-site inputs for [`build_content_tree_entry_expression`].
struct ContentTreeEntryProjections {
    status: TrackingStatus,
    snapshot_id: i64,
    location: Expression,
    file_size_in_bytes: Expression,
    /// Used for both `dataSequenceNumber` and `fileSequenceNumber`.
    sequence_number: Expression,
    /// `None` emits a null of the field's type.
    dv_info: Option<Expression>,
    record_count: Expression,
    /// `None` emits a null of the field's type.
    content_stats: Option<Expression>,
    /// Expression to extract partition values into a typed struct. Typically a `MapToStruct`
    /// expression over the `partitionValues` map column. `None` emits a null of the field's type.
    partition: Option<Expression>,
    /// Expression extracting the tags map column. `None` emits a null of the field's type.
    tags_expr: Option<Expression>,
}

/// Builds the row-to-`ContentTreeNodeEntry` projection expression shared by the blind-append
/// write, scan-row ingest, and AMT log-replay transforms; constant fields come from here,
/// everything that varies between call sites comes from [`ContentTreeEntryProjections`].
fn build_content_tree_entry_expression(
    output_schema: &Schema,
    projections: &ContentTreeEntryProjections,
) -> Expression {
    let tracking = Expression::struct_from([
        Expression::literal(Scalar::Integer(projections.status as i32)),
        Expression::literal(Scalar::Long(projections.snapshot_id)),
        Expression::null_literal(DataType::LONG), // dvSnapshotId
        projections.sequence_number.clone(),      // dataSequenceNumber
        projections.sequence_number.clone(),      // fileSequenceNumber
        Expression::null_literal(DataType::LONG), // firstRowId
        Expression::null_literal(DataType::BINARY), // deletedPositions
        Expression::null_literal(DataType::BINARY), // replacedPositions
    ]);

    let field_exprs: Vec<Arc<Expression>> = output_schema
        .fields()
        .map(|field| {
            let expr = match field.name().as_str() {
                CONTENT_TYPE => Expression::literal(Scalar::Integer(DataContentType::Data as i32)),
                LOCATION => projections.location.clone(),
                FILE_FORMAT => Expression::literal(Scalar::String("parquet".into())),
                TRACKING => tracking.clone(),
                DV_INFO => match &projections.dv_info {
                    Some(expr) => expr.clone(),
                    None => Expression::null_literal(field.data_type().clone()),
                },
                PARTITION_SPEC_ID => Expression::literal(Scalar::Integer(0)),
                PARTITION => match &projections.partition {
                    Some(expr) => expr.clone(),
                    None => Expression::null_literal(field.data_type().clone()),
                },
                SORT_ORDER_ID => Expression::null_literal(DataType::INTEGER),
                RECORD_COUNT => projections.record_count.clone(),
                CONTENT_STATS_FIELD_NAME => match &projections.content_stats {
                    Some(expr) => expr.clone(),
                    None => Expression::null_literal(field.data_type().clone()),
                },
                FILE_SIZE_IN_BYTES => projections.file_size_in_bytes.clone(),
                TAGS => match &projections.tags_expr {
                    Some(expr) => expr.clone(),
                    None => Expression::null_literal(field.data_type().clone()),
                },
                _ => Expression::null_literal(field.data_type().clone()),
            };
            Arc::new(expr)
        })
        .collect();

    Expression::struct_from(field_exprs)
}

/// Projects `add.{path, size, defaultRowCommitVersion, partitionValues, tags}` and parses
/// `add.stats` into `stats_parsed`, shaped to `delta_stats_schema`.
fn build_log_batch_evaluator(
    engine: &dyn Engine,
    delta_stats_schema: &SchemaRef,
) -> DeltaResult<Arc<dyn ExpressionEvaluator>> {
    let output_schema = DataType::Struct(Box::new(StructType::new_unchecked([
        log_add_projection_field(),
        StructField::nullable(
            STATS_PARSED_NAME,
            DataType::Struct(Box::new(delta_stats_schema.as_ref().clone())),
        ),
    ])));

    let expression = Arc::new(Expression::struct_from([
        Expression::struct_from([
            Expression::column([ADD_NAME, "path"]),
            Expression::column([ADD_NAME, "size"]),
            Expression::column([ADD_NAME, DEFAULT_ROW_COMMIT_VERSION_NAME]),
            Expression::column([ADD_NAME, "partitionValues"]),
            Expression::column([ADD_NAME, TAGS]),
        ]),
        Expression::parse_json(
            Expression::column([ADD_NAME, "stats"]),
            delta_stats_schema.clone(),
        ),
    ]));

    engine.evaluation_handler().new_expression_evaluator(
        LOG_BATCH_EVALUATOR_INPUT_SCHEMA.clone(),
        expression,
        output_schema,
    )
}

/// Builds an evaluator that assembles `ContentTreeNodeEntry` rows from the prepared log
/// columns (`add.{path, size, defaultRowCommitVersion}` + parsed `stats_parsed`) and the
/// flat decoded-DV columns, shaped to `output_schema`.
fn build_action_to_content_tree_entry_evaluator(
    engine: &dyn Engine,
    snapshot_id: i64,
    commit_version: i64,
    table_schema: &Schema,
    delta_stats_schema: &SchemaRef,
    output_schema: &SchemaRef,
) -> DeltaResult<Arc<dyn ExpressionEvaluator>> {
    // Output schema's content_stats field is always a struct by construction.
    let amt_content_stats_schema = match output_schema
        .field(CONTENT_STATS_FIELD_NAME)
        .map(|f| f.data_type())
    {
        Some(DataType::Struct(s)) => s.as_ref().clone(),
        _ => StructType::new_unchecked([]),
    };

    let input_schema = Arc::new(StructType::new_unchecked(
        [log_add_projection_field()]
            .into_iter()
            .chain(DV_DECODED_FLAT_SCHEMA.fields().cloned())
            .chain([StructField::nullable(
                STATS_PARSED_NAME,
                DataType::Struct(Box::new(delta_stats_schema.as_ref().clone())),
            )])
            .collect::<Vec<_>>(),
    ));

    let partition = output_schema
        .field(PARTITION)
        .map(|_| Expression::map_to_struct(Expression::column([ADD_NAME, "partitionValues"])));

    let projections = ContentTreeEntryProjections {
        status: TrackingStatus::Existing,
        snapshot_id,
        location: Expression::column([ADD_NAME, "path"]),
        file_size_in_bytes: Expression::column([ADD_NAME, "size"]),
        // `defaultRowCommitVersion` is optional on Add actions (only populated when row
        // tracking is enabled). Coalesce against `commit_version` so entries always carry
        // a meaningful sequence number even when the Add action lacks it.
        sequence_number: Expression::coalesce([
            Expression::column([ADD_NAME, DEFAULT_ROW_COMMIT_VERSION_NAME]),
            Expression::literal(Scalar::Long(commit_version)),
        ]),
        dv_info: Some(flat_dv_columns_to_dv_info_expr()),
        record_count: Expression::coalesce([
            Expression::column([STATS_PARSED_NAME, DELTA_STATS_NUM_RECORDS]),
            // Remove actions have no stats; default the record count to 0.
            Expression::literal(Scalar::Long(0)),
        ]),
        content_stats: Some(build_content_stats_from_delta_stats_parsed(
            table_schema,
            &amt_content_stats_schema,
        )?),
        partition,
        tags_expr: Some(Expression::column([ADD_NAME, TAGS])),
    };

    engine.evaluation_handler().new_expression_evaluator(
        input_schema,
        Arc::new(build_content_tree_entry_expression(
            output_schema,
            &projections,
        )),
        DataType::Struct(Box::new(output_schema.as_ref().clone())),
    )
}

/// Visits rows in one pass, accumulating decoded DV columns.
///
/// For rows with a DV: decodes path (base85 UUID → relative path), widens offset/sizeInBytes
/// to LONG, adds 8 to sizeInBytes (Delta → Iceberg framing), stores cardinality.
/// For rows without a DV: pushes Null scalars for all 4 columns.
struct DecodedDvVisitor {
    decoded_paths: Vec<crate::expressions::Scalar>,
    decoded_offsets: Vec<crate::expressions::Scalar>,
    decoded_sizes: Vec<crate::expressions::Scalar>,
    decoded_cardinalities: Vec<crate::expressions::Scalar>,
    is_log_batch: bool,
}

impl DecodedDvVisitor {
    fn for_scan_rows(n: usize) -> Self {
        Self {
            decoded_paths: Vec::with_capacity(n),
            decoded_offsets: Vec::with_capacity(n),
            decoded_sizes: Vec::with_capacity(n),
            decoded_cardinalities: Vec::with_capacity(n),
            is_log_batch: false,
        }
    }

    fn for_log_batch(n: usize) -> Self {
        Self {
            decoded_paths: Vec::with_capacity(n),
            decoded_offsets: Vec::with_capacity(n),
            decoded_sizes: Vec::with_capacity(n),
            decoded_cardinalities: Vec::with_capacity(n),
            is_log_batch: true,
        }
    }

    fn has_any_dv(&self) -> bool {
        self.decoded_paths.iter().any(|s| !s.is_null())
    }

    fn append_decoded_dv_columns(self, data: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        data.append_columns(
            DV_DECODED_FLAT_SCHEMA.clone(),
            vec![
                ArrayData::try_new(ArrayType::new(DataType::STRING, true), self.decoded_paths)?,
                ArrayData::try_new(ArrayType::new(DataType::LONG, true), self.decoded_offsets)?,
                ArrayData::try_new(ArrayType::new(DataType::LONG, true), self.decoded_sizes)?,
                ArrayData::try_new(
                    ArrayType::new(DataType::LONG, true),
                    self.decoded_cardinalities,
                )?,
            ],
        )
    }
}

impl RowVisitor for DecodedDvVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        if self.is_log_batch {
            static LOG_BATCH: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
                let names = vec![
                    column_name!("add.deletionVector.storageType"),
                    column_name!("add.deletionVector.pathOrInlineDv"),
                    column_name!("add.deletionVector.offset"),
                    column_name!("add.deletionVector.sizeInBytes"),
                    column_name!("add.deletionVector.cardinality"),
                ];
                let types = vec![
                    DataType::STRING,
                    DataType::STRING,
                    DataType::INTEGER,
                    DataType::INTEGER,
                    DataType::LONG,
                ];
                (names, types).into()
            });
            LOG_BATCH.as_ref()
        } else {
            static SCAN_ROW: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
                let names = vec![
                    column_name!("deletionVector.storageType"),
                    column_name!("deletionVector.pathOrInlineDv"),
                    column_name!("deletionVector.offset"),
                    column_name!("deletionVector.sizeInBytes"),
                    column_name!("deletionVector.cardinality"),
                ];
                let types = vec![
                    DataType::STRING,
                    DataType::STRING,
                    DataType::INTEGER,
                    DataType::INTEGER,
                    DataType::LONG,
                ];
                (names, types).into()
            });
            SCAN_ROW.as_ref()
        }
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        for i in 0..row_count {
            let storage_type_opt: Option<String> =
                getters[0].get_opt(i, "deletionVector.storageType")?;
            if let Some(storage_type_str) = storage_type_opt {
                let storage_type: DeletionVectorStorageType = storage_type_str.parse()?;
                let path_or_inline_dv: String =
                    getters[1].get(i, "deletionVector.pathOrInlineDv")?;
                let offset: Option<i32> = getters[2].get_opt(i, "deletionVector.offset")?;
                let size_in_bytes: i32 = getters[3].get(i, "deletionVector.sizeInBytes")?;
                let cardinality: i64 = getters[4].get(i, "deletionVector.cardinality")?;

                let dv = DeletionVectorDescriptor {
                    storage_type,
                    path_or_inline_dv,
                    offset,
                    size_in_bytes,
                    cardinality,
                };
                let deletion_vector = extract_deletion_vector_content(&dv)?;
                self.decoded_paths
                    .push(Scalar::String(deletion_vector.location));
                self.decoded_offsets
                    .push(Scalar::Long(deletion_vector.offset));
                self.decoded_sizes
                    .push(Scalar::Long(deletion_vector.size_in_bytes));
                self.decoded_cardinalities
                    .push(Scalar::Long(deletion_vector.cardinality));
            } else {
                self.decoded_paths.push(Scalar::Null(DataType::STRING));
                self.decoded_offsets.push(Scalar::Null(DataType::LONG));
                self.decoded_sizes.push(Scalar::Null(DataType::LONG));
                self.decoded_cardinalities
                    .push(Scalar::Null(DataType::LONG));
            }
        }
        Ok(())
    }
}

/// Identifies a specific row within a leaf manifest by its path and position.
struct LeafManifestIndex {
    /// Path to the leaf manifest file.
    path: String,
    /// Row position within the leaf manifest.
    position: u64,
}

// ===========================================================================================
// AMT log replay helpers
// ===========================================================================================

/// Single-pass visitor that deduplicates a log batch for AMT replay.
/// Sets `selection_vector[i] = true` for first-seen Add rows; updates `log_action_keys`
/// and `leaf_removes` for Remove rows.
struct LogBatchDedupVisitor<'a> {
    log_action_keys: &'a mut HashSet<FileActionKey>,
    leaf_removes: &'a mut Vec<LeafManifestIndex>,
    /// `true` for surviving Add rows; `false` for all other rows.
    selection_vector: Vec<bool>,
}

impl LogBatchDedupVisitor<'_> {
    const ADD_PATH: usize = 0;
    const ADD_DV_ST: usize = 1;
    const ADD_DV_PATH: usize = 2;
    const REM_PATH: usize = 3;
    const REM_DV_ST: usize = 4;
    const REM_DV_PATH: usize = 5;
    const REM_MANIFEST_PATH: usize = 6;
    const REM_MANIFEST_POS: usize = 7;

    /// Returns the DV location string from raw storage-type and path column getters, or `None`
    /// when no DV is present (storageType is null).
    ///
    /// Uses the same decode logic as [`extract_deletion_vector_content`] so that
    /// [`FileActionKey`] values built here match those built in [`process_content_root_batch`].
    fn dv_location<'a>(
        i: usize,
        getters: &[&'a dyn GetData<'a>],
        st_idx: usize,
        path_idx: usize,
    ) -> DeltaResult<Option<String>> {
        let Some(storage_type): Option<String> =
            getters[st_idx].get_opt(i, "deletionVector.storageType")?
        else {
            return Ok(None);
        };
        let path_or_inline: String = getters[path_idx].get(i, "deletionVector.pathOrInlineDv")?;
        // Build a minimal descriptor to reuse existing location-decode logic.
        let dv = DeletionVectorDescriptor {
            storage_type: storage_type.parse()?,
            path_or_inline_dv: path_or_inline,
            offset: None,
            size_in_bytes: 0,
            cardinality: 0,
        };
        Ok(Some(extract_deletion_vector_content(&dv)?.location))
    }
}

impl RowVisitor for LogBatchDedupVisitor<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            const STRING: DataType = DataType::STRING;
            const LONG: DataType = DataType::LONG;
            let types_and_names = vec![
                (STRING, column_name!("add.path")),
                (STRING, column_name!("add.deletionVector.storageType")),
                (STRING, column_name!("add.deletionVector.pathOrInlineDv")),
                (STRING, column_name!("remove.path")),
                (STRING, column_name!("remove.deletionVector.storageType")),
                (STRING, column_name!("remove.deletionVector.pathOrInlineDv")),
                (STRING, column_name!("remove.backReference.manifest")),
                (LONG, column_name!("remove.backReference.pos")),
            ];
            let (types, names) = types_and_names.into_iter().unzip();
            (names, types).into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        require!(
            getters.len() == 8,
            Error::InternalError(format!(
                "Wrong number of LogBatchDedupVisitor getters: {}",
                getters.len()
            ))
        );
        for i in 0..row_count {
            let add_path: Option<String> = getters[Self::ADD_PATH].get_opt(i, "add.path")?;
            if let Some(path) = add_path {
                // Add row: check dedup and update selection_vector.
                let dv_loc: Option<String> =
                    Self::dv_location(i, getters, Self::ADD_DV_ST, Self::ADD_DV_PATH)?;
                let key = FileActionKey::new(path, dv_loc);
                if self.log_action_keys.contains(&key) {
                    // Duplicate: superseded by a newer commit already processed.
                    self.selection_vector[i] = false;
                } else {
                    self.log_action_keys.insert(key);
                    self.selection_vector[i] = true;
                }
            } else {
                // Remove row: never emitted as a content root entry.
                self.selection_vector[i] = false;
                let rem_path: Option<String> = getters[Self::REM_PATH].get_opt(i, "remove.path")?;
                let Some(path) = rem_path else { continue };
                let dv_loc = Self::dv_location(i, getters, Self::REM_DV_ST, Self::REM_DV_PATH)?;
                self.log_action_keys
                    .insert(FileActionKey::new(path, dv_loc));
                // Collect leaf removes for post-replay DV bitmap updates.
                let leaf_path: Option<String> =
                    getters[Self::REM_MANIFEST_PATH].get_opt(i, "remove.backReference.manifest")?;
                let position: Option<i64> =
                    getters[Self::REM_MANIFEST_POS].get_opt(i, "remove.backReference.pos")?;
                if let (Some(leaf_path), Some(pos)) = (leaf_path, position) {
                    let pos = u64::try_from(pos).map_err(|_| {
                        Error::generic(format!("negative manifest position: {pos}"))
                    })?;
                    self.leaf_removes.push(LeafManifestIndex {
                        path: leaf_path,
                        position: pos,
                    });
                }
            }
        }
        Ok(())
    }
}

// ===========================================================================================
// ContentRootRebuildProcessor
// ===========================================================================================

/// Stateful processor for replaying delta log commits during AMT root manifest rebuild.
///
/// Processes delta log commits in **descending** order (newest first) followed by the existing
/// content root (as a "checkpoint" batch) to build the complete set of entries for a new content
/// root. Uses `(path, dv_location)` deduplication as per spec: first-seen wins, so the newest
/// action for each logical file is authoritative.
///
/// - Log batches (`is_log_batch = true`): [`LogBatchDedupVisitor`] is the single visitor pass; it
///   builds a selection vector marking surviving Add rows. The surviving rows are then converted to
///   ContentTreeNodeEntry schema via a pre-built expression evaluator and returned as a
///   [`FilteredEngineData`].
/// - Content root batches (`is_log_batch = false`): entries whose `(path, dv_location)` key was NOT
///   seen in a prior log batch are emitted unchanged; seen entries are suppressed.
///
/// Leaf manifest removes (Remove actions with `back_reference`) are
/// accumulated for a post-replay pass via [`deleted_leaf_positions_by_location`].
///
/// [`deleted_leaf_positions_by_location`]: ContentRootRebuildProcessor::deleted_leaf_positions_by_location
pub(crate) struct ContentRootRebuildProcessor {
    log_action_keys: HashSet<FileActionKey>,
    leaf_removes: Vec<LeafManifestIndex>,
    /// Builds ContentTreeNodeEntry rows from the log and DV columns.
    action_to_content_tree_entry_evaluator: Arc<dyn ExpressionEvaluator>,
    /// Projects the `add` columns and parses `add.stats` JSON.
    log_batch_evaluator: Arc<dyn ExpressionEvaluator>,
}

impl ContentRootRebuildProcessor {
    /// Creates a new processor and pre-builds the expression evaluators.
    ///
    /// # Parameters
    /// - `engine`: Engine for constructing expression evaluators.
    /// - `snapshot_id`: Stamped into tracking info for each emitted log-batch entry.
    /// - `table_schema`: Physical table schema. Used to derive the content_stats output schema.
    ///   Only read during construction; not retained.
    /// - `partition_type`: The partition struct type, or `None` for unpartitioned tables.
    pub(crate) fn new(
        engine: &dyn Engine,
        snapshot_id: i64,
        commit_version: i64,
        table_schema: &Schema,
        partition_type: Option<&StructType>,
    ) -> DeltaResult<Self> {
        let delta_stats_schema = Arc::new(build_delta_stats_schema(table_schema));
        let output_schema = Arc::new(ContentTreeNodeEntry::to_schema_with_content_stats(
            table_schema,
            &delta_stats_schema,
            partition_type,
        )?);

        let log_batch_evaluator = build_log_batch_evaluator(engine, &delta_stats_schema)?;
        let action_to_content_tree_entry_evaluator = build_action_to_content_tree_entry_evaluator(
            engine,
            snapshot_id,
            commit_version,
            table_schema,
            &delta_stats_schema,
            &output_schema,
        )?;

        Ok(Self {
            log_action_keys: HashSet::new(),
            leaf_removes: Vec::new(),
            action_to_content_tree_entry_evaluator,
            log_batch_evaluator,
        })
    }

    /// Processes a content root batch (`is_log_batch = false`).
    ///
    /// Emits entries whose `(path, dv_location)` key was not seen in a prior log batch.
    /// Any entry still marked `Added` is normalized to `Existing` — entries from the previous
    /// root all predate the current commit by definition.
    pub(crate) fn process_root_batch(
        &mut self,
        batch: FilteredEngineData,
    ) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
        let mut visitor = ContentTreeNodeEntryVisitor::default();
        FilteredRowVisitor::visit_rows_of(&mut visitor, &batch)?;

        let mut entries = Vec::new();
        for entry in visitor.entries {
            let Some(path) = entry.location.as_deref() else {
                continue;
            };
            let dv_loc = entry.deletion_vector.as_ref().map(|d| d.location.clone());
            let key = FileActionKey::new(path, dv_loc);

            if self.log_action_keys.contains(&key) {
                // Superseded by a log action — skip.
                continue;
            }

            // Mark previously "added" entries as "existing"
            // TODO: for DV replacements, "replaced" status?
            let entry = if entry.tracking.status == TrackingStatus::Added {
                entry.with_status(TrackingStatus::Existing)
            } else {
                entry
            };

            entries.push(entry);
        }

        Ok(entries)
    }

    /// Drains accumulated leaf manifest deletions, grouped by leaf path.
    pub(crate) fn deleted_leaf_positions_by_location(
        &mut self,
    ) -> HashMap<String, roaring::RoaringTreemap> {
        let mut result: HashMap<String, roaring::RoaringTreemap> =
            HashMap::with_capacity(self.leaf_removes.len());
        for lr in self.leaf_removes.drain(..) {
            result.entry(lr.path).or_default().insert(lr.position);
        }
        result
    }

    /// Single-pass dedup over a log batch. Updates internal `log_action_keys` and
    /// `leaf_removes`, and returns a selection vector of length `actions.len()` where
    /// `selection_vector[i] == true` indicates row `i` is the latest `Add` action for that
    /// (data file, DV) pair.
    fn dedup_log_batch(&mut self, actions: &dyn EngineData) -> DeltaResult<Vec<bool>> {
        let row_count = actions.len();
        let mut dedup = LogBatchDedupVisitor {
            log_action_keys: &mut self.log_action_keys,
            leaf_removes: &mut self.leaf_removes,
            selection_vector: vec![false; row_count],
        };

        dedup.visit_rows_of(actions)?;
        Ok(dedup.selection_vector)
    }
}

impl LogReplayProcessor for ContentRootRebuildProcessor {
    type Output = FilteredEngineData;

    /// Dedups the batch, then transforms the surviving `Add` rows into `ContentTreeNodeEntry`
    /// rows (decode DV columns, run the log-batch and action-to-entry evaluators) and returns
    /// them paired with the selection vector.
    fn process_actions_batch(&mut self, batch: ActionsBatch) -> DeltaResult<Self::Output> {
        let selection_vector = self.dedup_log_batch(batch.actions.as_ref())?;
        require!(
            selection_vector.len() == batch.actions.len(),
            Error::InternalError(format!(
                "dedup selection vector length {} does not match batch row count {}",
                selection_vector.len(),
                batch.actions.len()
            ))
        );

        // Dedup eliminated every row (e.g. an all-`Remove` commit, or `Add`s all superseded by
        // newer commits): skip the DV decode and evaluator passes and return the batch with the
        // all-`false` vector so the caller drops it.
        if !selection_vector.iter().any(|&b| b) {
            return FilteredEngineData::try_new(batch.actions, selection_vector);
        }

        let actions = batch.actions;

        // Decode DV columns: z85 path decoding cannot be expressed as a kernel expression.
        let mut dv_decoder = DecodedDvVisitor::for_log_batch(actions.len());
        dv_decoder.visit_rows_of(actions.as_ref())?;
        let intermediate = self.log_batch_evaluator.evaluate(actions.as_ref())?;

        // Append the decoded `_dv_*` columns for the action-to-entry evaluator.
        let augmented_with_dv = dv_decoder.append_decoded_dv_columns(intermediate.as_ref())?;

        let result = self
            .action_to_content_tree_entry_evaluator
            .evaluate(augmented_with_dv.as_ref())?;
        FilteredEngineData::try_new(result, selection_vector)
    }

    fn data_skipping_filter(&self) -> Option<&DataSkippingFilter> {
        None
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::actions::deletion_vector::DeletionVectorStorageType;
    use crate::content_tree::{absolute_to_relative_path, parse_or_join_url, ContentTreeNode};

    /// Helper: builds a root manifest, writes it to disk, and reads it back.
    fn build_and_read_root(
        builder: &mut ContentTreeNodeBuilder,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
    ) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
        let root_metadata =
            builder.build(engine, snapshot_id, &mut CursorRowIdAllocator::new(0))?;
        let table_root = root_metadata.table_root.clone();
        let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
            .write(engine)?
            .location;
        let root_path = absolute_to_relative_path(&root_url, &table_root);
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &root_url,
            root_path,
            None,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let root =
            ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
        root.entries()
    }

    // TODO: Add tests for all tracking columns (status, snapshot_id, sequence_number,
    // file_sequence_number, first_row_id, deleted_positions, replaced_positions) to verify they
    // are correctly set during build operations for ADDED, DELETED, and EXISTED manifests.

    #[test]
    fn test_snapshot_builder() -> Result<(), Box<dyn std::error::Error>> {
        let _add_file_action = [json!({
            "add": {
                "path": "part-00000-test.parquet",
                "partitionValues": {},
                "size": 1024,
                "modificationTime": 1587968586000i64,
                "dataChange": true,
                "stats": null,
                "tags": null
            }
        })];
        Ok(())
    }

    /// Helper function to create a minimal table schema for tests.
    /// This schema has the required PARQUET:field_id metadata for content_stats generation.
    fn test_table_schema() -> Schema {
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

        Schema::new_unchecked([
            StructField::new("id", DataType::INTEGER, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-id".to_string()),
                ),
            ]),
        ])
    }

    #[test]
    fn test_path_to_absolute_with_relative_path() -> Result<(), Box<dyn std::error::Error>> {
        // Test with s3:// URL as table root
        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let relative_path = "part-00000-123.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert_eq!(result, "s3://my-bucket/my-table/part-00000-123.parquet");

        // Test with nested relative path
        let relative_path = "year=2023/month=10/part-00001-456.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert_eq!(
            result,
            "s3://my-bucket/my-table/year=2023/month=10/part-00001-456.parquet"
        );

        Ok(())
    }

    #[test]
    fn test_path_to_absolute_with_absolute_s3_path() -> Result<(), Box<dyn std::error::Error>> {
        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let absolute_path = "s3://another-bucket/external/data.parquet";
        let result = builder.path_to_absolute(absolute_path)?;
        assert_eq!(result, "s3://another-bucket/external/data.parquet");
        Ok(())
    }

    #[test]
    fn test_path_to_absolute_with_absolute_https_path() -> Result<(), Box<dyn std::error::Error>> {
        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let absolute_path = "https://example.com/data/file.parquet";
        let result = builder.path_to_absolute(absolute_path)?;
        assert_eq!(result, "https://example.com/data/file.parquet");
        Ok(())
    }

    #[test]
    fn test_path_to_absolute_with_gs_url() -> Result<(), Box<dyn std::error::Error>> {
        // Test with Google Cloud Storage URL
        let table_root = Url::parse("gs://my-gcs-bucket/delta-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let relative_path = "data/part-00000.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert_eq!(
            result,
            "gs://my-gcs-bucket/delta-table/data/part-00000.parquet"
        );

        // Test with absolute GCS path
        let absolute_path = "gs://other-bucket/external.parquet";
        let result = builder.path_to_absolute(absolute_path)?;
        assert_eq!(result, "gs://other-bucket/external.parquet");
        Ok(())
    }

    #[test]
    fn test_path_to_absolute_with_azure_url() -> Result<(), Box<dyn std::error::Error>> {
        // Test with Azure Blob Storage URL
        let table_root = Url::parse("abfss://container@account.dfs.core.windows.net/delta-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let relative_path = "part-00000.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert_eq!(
            result,
            "abfss://container@account.dfs.core.windows.net/delta-table/part-00000.parquet"
        );
        Ok(())
    }

    #[test]
    fn test_path_to_absolute_with_file_url() -> Result<(), Box<dyn std::error::Error>> {
        // Test with file:// URL - use a temp directory that exists
        let temp_dir = std::env::temp_dir();
        let table_root = Url::parse(&format!("file://{}/", temp_dir.to_str().unwrap()))?;
        let builder = ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        let relative_path = "part-00000.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert!(result.starts_with("file://"));
        assert!(result.ends_with("/part-00000.parquet"));

        // Test with absolute file:// path
        let absolute_path = "file:///other/location/data.parquet";
        let result = builder.path_to_absolute(absolute_path)?;
        assert_eq!(result, "file:///other/location/data.parquet");
        Ok(())
    }

    #[test]
    fn test_path_to_absolute_preserves_special_characters() -> Result<(), Box<dyn std::error::Error>>
    {
        // Test that special characters in paths are preserved
        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        let relative_path = "partition=value%20with%20spaces/file.parquet";
        let result = builder.path_to_absolute(relative_path)?;
        assert_eq!(
            result,
            "s3://my-bucket/my-table/partition=value%20with%20spaces/file.parquet"
        );
        Ok(())
    }

    #[test]
    fn test_record_count_from_stats() -> Result<(), Box<dyn std::error::Error>> {
        // Create builder and add entries with specific record counts
        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        for (path, record_count) in [
            ("part-00000.parquet", 100i64),
            ("part-00001.parquet", 250i64),
            ("part-00002.parquet", 0i64),
        ] {
            builder.add_entry(
                ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                    .location(path)
                    .with_tracking(TrackingStatus::Added, 1, 1)
                    .record_count(record_count)
                    .file_size_in_bytes(1024)
                    .build(),
            );
        }

        // Build metadata and verify record counts are preserved through roundtrip
        let engine = crate::engine::sync::SyncEngine::new();
        let metadata = builder.build(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let entries = metadata.entries()?;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].record_count, 100);
        assert_eq!(entries[1].record_count, 250);
        assert_eq!(entries[2].record_count, 0);

        Ok(())
    }

    #[test]
    fn test_content_stats_from_json_stats() -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::Add;
        use crate::content_tree::stats::delta_json_stats_to_content_stats;
        use crate::expressions::Scalar;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

        // Create a table schema with field IDs and column mapping annotations
        // (column mapping is required when metadata tree feature is enabled)
        let table_schema = crate::schema::StructType::new_unchecked([
            StructField::new("id", DataType::LONG, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-id".to_string()),
                ),
            ]),
            StructField::new("name", DataType::STRING, true).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-name".to_string()),
                ),
            ]),
        ]);

        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, table_schema.clone());

        // Add an entry with JSON stats
        let stats_json = r#"{"numRecords":100,"minValues":{"id":1,"name":"alice"},"maxValues":{"id":100,"name":"zoe"},"nullCount":{"id":0,"name":5}}"#;
        let add = Add {
            path: "part-00000.parquet".to_string(),
            partition_values: HashMap::new(),
            size: 1024,
            modification_time: 1587968586000,
            data_change: true,
            stats: Some(stats_json.to_string()),
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            back_reference: None,
        };

        builder.add(add, 1, 1)?;

        // Verify content_stats is populated by directly checking the conversion function
        // (The builder uses this function internally)
        let content_stats =
            delta_json_stats_to_content_stats(Some(stats_json), &table_schema, None)?
                .expect("content_stats should be populated");

        // Helper function to get a field's value from a StructData by field name
        fn get_struct_field<'a>(
            data: &'a crate::expressions::StructData,
            name: &str,
        ) -> Option<&'a Scalar> {
            data.fields()
                .iter()
                .position(|f| f.name() == name)
                .map(|idx| &data.values()[idx])
        }

        // Helper function to get a column's stats field value in AMT format
        fn get_column_stat<'a>(
            stats: &'a crate::expressions::StructData,
            column: &str,
            stat_field: &str,
        ) -> Option<&'a Scalar> {
            if let Some(Scalar::Struct(col_stats)) = get_struct_field(stats, column) {
                get_struct_field(col_stats, stat_field)
            } else {
                None
            }
        }

        // AMT format has one field per column: {id: {...}, name: {...}}
        assert_eq!(content_stats.fields().len(), 2);

        // Check id stats
        assert_eq!(
            get_column_stat(&content_stats, "id", crate::content_tree::VALUE_COUNT),
            Some(&Scalar::Long(100))
        );
        assert_eq!(
            get_column_stat(&content_stats, "id", crate::content_tree::LOWER_BOUND),
            Some(&Scalar::Long(1))
        );
        assert_eq!(
            get_column_stat(&content_stats, "id", crate::content_tree::UPPER_BOUND),
            Some(&Scalar::Long(100))
        );

        // Check name stats
        assert_eq!(
            get_column_stat(
                &content_stats,
                "name",
                crate::content_tree::NULL_VALUE_COUNT
            ),
            Some(&Scalar::Long(5))
        );
        assert_eq!(
            get_column_stat(&content_stats, "name", crate::content_tree::LOWER_BOUND),
            Some(&Scalar::String("alice".to_string()))
        );

        // Verify the builder has the entry with content_stats populated
        // Note: When serialized to EngineData and read back, content_stats is not preserved
        // because it requires the table schema to read. This is expected behavior.
        // The content_stats is used during write operations where the schema is known.
        assert_eq!(builder.pending_entries.len(), 1);
        assert!(
            builder.pending_entries[0].content_stats.is_some(),
            "pending entry should have content_stats"
        );

        Ok(())
    }

    #[test]
    fn test_add_stores_partition_values_in_partition_tuple(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::Add;
        use crate::expressions::Scalar;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

        let table_schema = crate::schema::StructType::new_unchecked([
            StructField::new("id", DataType::LONG, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-id".to_string()),
                ),
            ]),
            StructField::new("category", DataType::STRING, true).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-category".to_string()),
                ),
            ]),
            StructField::new("year", DataType::INTEGER, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(3),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(3),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-year".to_string()),
                ),
            ]),
        ]);

        let partition_columns = vec!["category".to_string(), "year".to_string()];
        let partition_type = build_partition_type(&partition_columns, &table_schema);

        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, table_schema.clone());
        builder = builder.with_partition_type(partition_type);

        let add = Add {
            path: "category=A/year=2024/part-00000.parquet".to_string(),
            partition_values: HashMap::from([
                ("category".to_string(), "A".to_string()),
                ("year".to_string(), "2024".to_string()),
            ]),
            size: 1024,
            modification_time: 1587968586000,
            data_change: true,
            stats: Some(
                r#"{"numRecords":100,"minValues":{"id":1},"maxValues":{"id":50},"nullCount":{"id":0}}"#
                    .to_string(),
            ),
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            back_reference: None,
        };

        builder.add(add, 1, 1)?;

        assert_eq!(builder.pending_entries.len(), 1);
        let entry = &builder.pending_entries[0];

        // content_stats covers all table columns (data + partition) from the stats schema,
        // but partition column entries have null stats -- actual partition values live in the
        // dedicated partition tuple.
        let content_stats = entry
            .content_stats
            .as_ref()
            .expect("should have content_stats");
        assert_eq!(
            content_stats.fields().len(),
            3,
            "stats schema covers all table columns"
        );
        // Partition columns should NOT have real partition values merged in -- their
        // lower_bound/upper_bound should be null since the stats JSON has no min/max for
        // partition columns. (Actual partition values live in the partition tuple.)
        for col in ["category", "year"] {
            let idx = content_stats
                .fields()
                .iter()
                .position(|f| f.name() == col)
                .unwrap_or_else(|| panic!("content_stats should have {col} field"));
            let Scalar::Struct(ref inner) = content_stats.values()[idx] else {
                panic!("{col} stats should be a struct");
            };
            let lb = inner
                .fields()
                .iter()
                .position(|f| f.name() == "lower_bound")
                .map(|i| &inner.values()[i]);
            let ub = inner
                .fields()
                .iter()
                .position(|f| f.name() == "upper_bound")
                .map(|i| &inner.values()[i]);
            assert!(
                lb.is_none_or(|v| v.is_null()),
                "{col} lower_bound should be null"
            );
            assert!(
                ub.is_none_or(|v| v.is_null()),
                "{col} upper_bound should be null"
            );
        }

        // Partition values should be in the dedicated partition tuple
        let partition = entry
            .partition
            .as_ref()
            .expect("should have partition tuple");
        assert_eq!(partition.fields().len(), 2, "should have category and year");

        let cat_idx = partition
            .fields()
            .iter()
            .position(|f| f.name() == "category")
            .expect("partition should have category");
        assert_eq!(partition.values()[cat_idx], Scalar::String("A".to_string()));

        let year_idx = partition
            .fields()
            .iter()
            .position(|f| f.name() == "year")
            .expect("partition should have year");
        assert_eq!(partition.values()[year_idx], Scalar::Integer(2024));

        // Partition fields should carry PARQUET:field_id metadata
        assert_eq!(
            partition.fields()[cat_idx]
                .metadata
                .get(ColumnMetadataKey::ParquetFieldId.as_ref()),
            Some(&MetadataValue::Number(2)),
        );
        assert_eq!(
            partition.fields()[year_idx]
                .metadata
                .get(ColumnMetadataKey::ParquetFieldId.as_ref()),
            Some(&MetadataValue::Number(3)),
        );

        Ok(())
    }

    /// Round-trips Delta `partitionValues` through the AMT partition tuple: parse string values
    /// into typed scalars via `build_partition_data`, then serialize back via
    /// `serialize_partition_value` and assert the original map is recovered.
    #[test]
    fn test_partition_values_round_trip_through_partition_tuple(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::partition::serialization::serialize_partition_value;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

        let table_schema = crate::schema::StructType::new_unchecked([
            StructField::new("id", DataType::LONG, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(1),
            )]),
            StructField::new("region", DataType::STRING, true).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(2),
            )]),
            StructField::new("year", DataType::INTEGER, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(3),
            )]),
            StructField::new("score", DataType::DOUBLE, true).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(4),
            )]),
        ]);

        let partition_columns = vec![
            "region".to_string(),
            "year".to_string(),
            "score".to_string(),
        ];
        let partition_type = build_partition_type(&partition_columns, &table_schema);

        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, table_schema.clone());
        builder = builder.with_partition_type(partition_type);

        let original_partition_values = HashMap::from([
            ("region".to_string(), "us-west-2".to_string()),
            ("year".to_string(), "2024".to_string()),
            ("score".to_string(), "3.14".to_string()),
        ]);

        let partition = builder
            .build_partition_data(&original_partition_values)?
            .expect("partitioned table should produce a partition tuple");

        // Reconstruct partitionValues from the typed partition tuple
        let mut reconstructed: HashMap<String, String> = HashMap::new();
        for (field, value) in partition.fields().iter().zip(partition.values()) {
            if let Some(s) = serialize_partition_value(value)? {
                reconstructed.insert(field.name().to_string(), s);
            }
        }

        assert_eq!(reconstructed, original_partition_values);

        Ok(())
    }

    /// Round-trip with a null partition value: null partition columns produce `Scalar::Null`
    /// in the partition tuple, and `serialize_partition_value` returns `None` for nulls (matching
    /// Delta's convention of omitting null partition values from the map).
    #[test]
    fn test_partition_values_round_trip_with_null_value() -> Result<(), Box<dyn std::error::Error>>
    {
        use crate::partition::serialization::serialize_partition_value;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField};

        let table_schema = crate::schema::StructType::new_unchecked([
            StructField::new("id", DataType::LONG, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(1),
            )]),
            StructField::new("region", DataType::STRING, true).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(2),
            )]),
        ]);

        let partition_columns = vec!["region".to_string()];
        let partition_type = build_partition_type(&partition_columns, &table_schema);

        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, table_schema.clone());
        builder = builder.with_partition_type(partition_type);

        // Empty map -> partition column is null (key absent from partitionValues)
        let original_partition_values = HashMap::new();
        let partition = builder
            .build_partition_data(&original_partition_values)?
            .expect("partitioned table should produce a partition tuple");

        assert_eq!(partition.fields().len(), 1);
        assert!(partition.values()[0].is_null());

        // Serialize back: null should produce None, so the reconstructed map is empty
        let mut reconstructed: HashMap<String, String> = HashMap::new();
        for (field, value) in partition.fields().iter().zip(partition.values()) {
            if let Some(s) = serialize_partition_value(value)? {
                reconstructed.insert(field.name().to_string(), s);
            }
        }

        assert_eq!(reconstructed, original_partition_values);

        Ok(())
    }

    #[test]
    fn test_content_stats_with_test_table_schema() -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::Add;

        let table_root = Url::parse("s3://my-bucket/my-table/")?;
        // Builder with test table schema (has one "id" column)
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        let add = Add {
            path: "part-00000.parquet".to_string(),
            partition_values: HashMap::new(),
            size: 1024,
            modification_time: 1587968586000,
            data_change: true,
            stats: Some(r#"{"numRecords":100}"#.to_string()),
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            back_reference: None,
        };

        builder.add(add, 1, 1)?;

        // Verify the builder has the entry
        assert_eq!(builder.pending_entries.len(), 1);

        // With test_table_schema (one "id" column), content_stats should have AMT format
        // with fields: {id: {value_count, ...}}
        let content_stats = &builder.pending_entries[0].content_stats;
        if let Some(stats) = content_stats {
            // AMT format has one field per column
            assert!(
                !stats.fields().is_empty(),
                "AMT stats should have at least one column stats field, got {} fields",
                stats.fields().len()
            );
            // Check that id column stats are present
            let has_id = stats.fields().iter().any(|f| f.name() == "id");
            assert!(has_id, "should have id column stats field");
        }
        // content_stats can also be None if stats JSON parsing returned None

        Ok(())
    }

    #[test]
    fn test_write_leaf_aggregates_content_stats() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        use crate::content_tree::stats::delta_json_stats_to_content_stats;
        use crate::engine::sync::SyncEngine;
        use crate::expressions::Scalar;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField, StructType};

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a table schema with field IDs (required for stats schema generation)
        let table_schema = StructType::new_unchecked([
            StructField::new("id", DataType::LONG, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(1),
            )]),
            StructField::new("name", DataType::STRING, true).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(2),
            )]),
        ]);

        // Create a builder with the table schema
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, table_schema.clone());

        // Create content_stats for file 1: id=[1, 50], name=["alice", "mike"]
        let stats1_json = r#"{"numRecords":100,"minValues":{"id":1,"name":"alice"},"maxValues":{"id":50,"name":"mike"},"nullCount":{"id":0,"name":5}}"#;
        let content_stats_1 =
            delta_json_stats_to_content_stats(Some(stats1_json), &table_schema, None)?;

        let entry1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data/part-00000.parquet")
            .with_tracking(TrackingStatus::Added, 1, 1)
            .record_count(100)
            .file_size_in_bytes(1024)
            .content_stats_opt(content_stats_1)
            .build();

        // Create content_stats for file 2: id=[40, 100], name=["bob", "zoe"]
        let stats2_json = r#"{"numRecords":150,"minValues":{"id":40,"name":"bob"},"maxValues":{"id":100,"name":"zoe"},"nullCount":{"id":0,"name":10}}"#;
        let content_stats_2 =
            delta_json_stats_to_content_stats(Some(stats2_json), &table_schema, None)?;

        let entry2 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data/part-00001.parquet")
            .with_tracking(TrackingStatus::Added, 1, 1)
            .record_count(150)
            .file_size_in_bytes(2048)
            .content_stats_opt(content_stats_2)
            .build();

        builder.add_entry(entry1);
        builder.add_entry(entry2);

        // Write the leaf manifest
        let leaf_manifest_entry =
            builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;

        // Verify content_stats is populated on the leaf manifest entry
        assert!(
            leaf_manifest_entry.content_stats.is_some(),
            "write_leaf should aggregate content_stats from entries"
        );

        let aggregated_stats = leaf_manifest_entry.content_stats.as_ref().unwrap();

        // Helper function to get a field's value from a StructData by field name
        fn get_struct_field<'a>(
            data: &'a crate::expressions::StructData,
            name: &str,
        ) -> Option<&'a Scalar> {
            data.fields()
                .iter()
                .position(|f| f.name() == name)
                .map(|idx| &data.values()[idx])
        }

        // Helper function to get a column's stats field value in AMT format
        fn get_column_stat<'a>(
            stats: &'a crate::expressions::StructData,
            column: &str,
            stat_field: &str,
        ) -> Option<&'a Scalar> {
            if let Some(Scalar::Struct(col_stats)) = get_struct_field(stats, column) {
                get_struct_field(col_stats, stat_field)
            } else {
                None
            }
        }

        // Verify the aggregated stats are in AMT format: {id: {...}, name: {...}}
        assert_eq!(aggregated_stats.fields().len(), 2);

        // Check id stats: value_count=250, lower_bound=1, upper_bound=100
        assert_eq!(
            get_column_stat(aggregated_stats, "id", crate::content_tree::VALUE_COUNT),
            Some(&Scalar::Long(250))
        );
        assert_eq!(
            get_column_stat(aggregated_stats, "id", crate::content_tree::LOWER_BOUND),
            Some(&Scalar::Long(1))
        );
        assert_eq!(
            get_column_stat(aggregated_stats, "id", crate::content_tree::UPPER_BOUND),
            Some(&Scalar::Long(100))
        );

        // Check name stats: null_value_count=15, lower_bound="alice", upper_bound="zoe"
        assert_eq!(
            get_column_stat(
                aggregated_stats,
                "name",
                crate::content_tree::NULL_VALUE_COUNT
            ),
            Some(&Scalar::Long(15)) // 5 + 10
        );
        assert_eq!(
            get_column_stat(aggregated_stats, "name", crate::content_tree::LOWER_BOUND),
            Some(&Scalar::String("alice".to_string()))
        );
        assert_eq!(
            get_column_stat(aggregated_stats, "name", crate::content_tree::UPPER_BOUND),
            Some(&Scalar::String("zoe".to_string()))
        );

        Ok(())
    }

    #[test]
    fn test_write_leaf_no_content_stats_when_entries_have_none(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a builder with empty schema
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Create entries without content_stats
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data/part-00000.parquet")
            .with_tracking(TrackingStatus::Added, 1, 1)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        builder.add_entry(entry);

        // Write the leaf manifest
        let leaf_manifest_entry =
            builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;

        // When all entries have None content_stats, the aggregate should also be None
        assert!(
            leaf_manifest_entry.content_stats.is_none(),
            "write_leaf should return None content_stats when all entries have None"
        );

        Ok(())
    }

    #[test]
    fn test_extract_deletion_vector_persisted_relative() -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::deletion_vector::DeletionVectorDescriptor;

        // Test case from the existing deletion_vector tests
        // path_or_inline_dv: "ab^-aqEH.-t@S}K{vb[*k^"
        // prefix: "ab" (2 chars before the 20 char uuid)
        // encoded uuid (20 chars): "^-aqEH.-t@S}K{vb[*k^"
        // which decodes to UUID: d2c639aa-8816-431a-aaf6-d3fe2512ff61
        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedRelative,
            path_or_inline_dv: "ab^-aqEH.-t@S}K{vb[*k^".to_string(),
            offset: Some(4),
            size_in_bytes: 40,
            cardinality: 6,
        };

        let deletion_vector = extract_deletion_vector_content(&dv)?;

        // Should have location set to the relative path
        assert_eq!(
            deletion_vector.location,
            "ab/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
        );

        // Should have offset and size (+8 for size field and CRC)
        assert_eq!(deletion_vector.offset, 4);
        assert_eq!(deletion_vector.size_in_bytes, 48); // 40 + 8
        assert_eq!(deletion_vector.cardinality, 6);

        Ok(())
    }

    #[test]
    fn test_extract_deletion_vector_persisted_relative_no_prefix(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::deletion_vector::DeletionVectorDescriptor;

        // Test case with no prefix (uuid only, 20 chars)
        // This is the test case from dv_example() in deletion_vector.rs
        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedRelative,
            path_or_inline_dv: "vBn[lx{q8@P<9BNH/isA".to_string(),
            offset: Some(1),
            size_in_bytes: 36,
            cardinality: 2,
        };

        let deletion_vector = extract_deletion_vector_content(&dv)?;

        // Should have location set to the relative path (no prefix directory)
        assert_eq!(
            deletion_vector.location,
            "deletion_vector_61d16c75-6994-46b7-a15b-8b538852e50e.bin"
        );

        // Should have offset and size (+8 for size field and CRC)
        assert_eq!(deletion_vector.offset, 1);
        assert_eq!(deletion_vector.size_in_bytes, 44); // 36 + 8
        assert_eq!(deletion_vector.cardinality, 2);

        Ok(())
    }

    #[test]
    fn test_extract_deletion_vector_persisted_absolute() -> Result<(), Box<dyn std::error::Error>> {
        use crate::actions::deletion_vector::DeletionVectorDescriptor;

        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedAbsolute,
            path_or_inline_dv:
                "s3://another-bucket/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
                    .to_string(),
            offset: Some(4),
            size_in_bytes: 40,
            cardinality: 6,
        };

        let deletion_vector = extract_deletion_vector_content(&dv)?;

        // Should preserve the absolute path as-is
        assert_eq!(
            deletion_vector.location,
            "s3://another-bucket/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
        );

        // Should have offset and size (+8)
        assert_eq!(deletion_vector.offset, 4);
        assert_eq!(deletion_vector.size_in_bytes, 48); // 40 + 8
        assert_eq!(deletion_vector.cardinality, 6);

        Ok(())
    }

    #[test]
    fn test_extract_deletion_vector_inline_not_supported() {
        use crate::actions::deletion_vector::DeletionVectorDescriptor;

        // This is the inline DV from dv_inline() in deletion_vector.rs
        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::Inline,
            path_or_inline_dv: "^Bg9^0rr910000000000iXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L"
                .to_string(),
            offset: None,
            size_in_bytes: 44,
            cardinality: 6,
        };

        let result = extract_deletion_vector_content(&dv);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Inline deletion vectors are not supported"));
    }

    #[test]
    fn test_extract_deletion_vector_invalid_relative_path() {
        use crate::actions::deletion_vector::DeletionVectorDescriptor;

        // path_or_inline_dv is too short (less than 20 chars)
        let dv = DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedRelative,
            path_or_inline_dv: "short".to_string(),
            offset: Some(1),
            size_in_bytes: 36,
            cardinality: 2,
        };

        let result = extract_deletion_vector_content(&dv);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid length"));
    }

    /// Test helper: Applies a manifest deletion vector to filter entries from a manifest.
    ///
    /// Manifest deletion vectors (ManifestDV, content_type = 5) can filter out entries
    /// from a manifest by ordinal position without rewriting the manifest file.
    fn apply_manifest_dv(
        entries: Vec<ContentTreeNodeEntry>,
        dv_bytes: &Bytes,
    ) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
        let deleted_positions = crate::content_tree::parse_manifest_dv(dv_bytes)?;

        // Filter entries: keep only those whose ordinal position is NOT in the deletion vector
        let filtered_entries: Vec<ContentTreeNodeEntry> =
            if let Some(deleted_positions) = deleted_positions {
                entries
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, entry)| {
                        // If this position is NOT deleted, keep the entry
                        if !deleted_positions.contains(index as u64) {
                            Some(entry)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                entries
            };

        Ok(filtered_entries)
    }

    #[test]
    fn test_delete_from_leaf_single_entry() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        // Write the leaf
        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create a root with the leaf, then delete entry at index 5
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.insert(5u64);
        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Delete)?;

        // Step 3: Build, write, and read back the root to verify manifest DV is stored inline
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        // Should have: 1 DataManifest entry
        assert_eq!(root_entries.len(), 1);

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        assert_eq!(data_manifest.location.as_ref(), Some(&leaf_path));

        // Verify the manifest_dv field contains the deleted index
        let manifest_dv_bytes = data_manifest
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        assert!(
            manifest_dv_bytes.len() >= 4,
            "Should have magic number prefix"
        );
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(5));
        assert_eq!(treemap.len(), 1);

        // Step 4: Read the leaf and apply manifest DV to verify filtering
        let leaf_url = parse_or_join_url(&leaf_path, &table_root)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &leaf_url,
            leaf_path.clone(),
            None,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let leaf_metadata = ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_root.clone(),
        )?;
        let leaf_entries = leaf_metadata.entries()?;
        assert_eq!(leaf_entries.len(), 10); // Original 10 entries

        // Apply the manifest DV
        let filtered_entries = apply_manifest_dv(leaf_entries, manifest_dv_bytes)?;
        assert_eq!(filtered_entries.len(), 9); // 1 deleted, 9 remaining

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_multiple_entries() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Create root and delete multiple entries
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.extend([2u64, 5, 7]);
        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Delete)?;

        // Build, write, and read back the root to verify
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;
        assert_eq!(root_entries.len(), 1); // DataManifest entry

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .unwrap();

        // Verify all deleted indices in manifest_dv field
        let manifest_dv_bytes = data_manifest.manifest_dv_bytes().unwrap();
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(2));
        assert!(treemap.contains(5));
        assert!(treemap.contains(7));
        assert_eq!(treemap.len(), 3);

        // Apply manifest DV and verify filtering
        let leaf_url = parse_or_join_url(&leaf_path, &table_root)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &leaf_url,
            leaf_path.clone(),
            None,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let leaf_metadata =
            ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
        let leaf_entries = leaf_metadata.entries()?;
        let filtered_entries = apply_manifest_dv(leaf_entries, manifest_dv_bytes)?;
        assert_eq!(filtered_entries.len(), 7); // 3 deleted, 7 remaining

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_all_entries_marks_deleted() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 3 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..3 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Create root and delete all 3 entries
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        // Deleting all entries should automatically mark the manifest as deleted
        let mut indices = RoaringTreemap::new();
        indices.extend([0u64, 1, 2]);
        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Delete)?;

        // Build, write, and read back the root to verify the manifest is marked as deleted
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let leaf_manifest = root_entries
            .iter()
            .find(|e| {
                e.content_type == DataContentType::DataManifest
                    && e.location.as_ref() == Some(&leaf_path)
            })
            .expect("Leaf manifest should exist");

        assert_eq!(leaf_manifest.tracking.status, TrackingStatus::Deleted);

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_index_out_of_bounds() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 10 entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Try to delete index 10 (out of bounds, valid indices are 0-9 for 10 entries)
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.insert(10u64);
        let result =
            root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Delete);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_nonexistent_manifest() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Try to delete from a non-existent leaf
        let mut indices = RoaringTreemap::new();
        indices.insert(5u64);
        let result = root_builder.update_leaf_positions(
            "nonexistent.parquet",
            &indices,
            LeafPositionUpdate::Delete,
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Manifest cache not found"));

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_with_relative_path() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        // Canonicalize the path to match what try_parse_uri does in real usage
        // This ensures paths are consistent (e.g., /private/var instead of /var on macOS)
        let canonical_path = std::fs::canonicalize(temp_dir.path())?;
        let table_root = Url::from_directory_path(canonical_path).unwrap();

        // Create a leaf
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..5 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();
        // leaf_path is now already relative
        let relative_path = &leaf_path;

        // Create root and delete using relative path
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);
        let mut indices = RoaringTreemap::new();
        indices.insert(3u64);
        root_builder.update_leaf_positions(relative_path, &indices, LeafPositionUpdate::Delete)?;

        // Build, write, and read back the root to verify manifest DV is stored inline
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .unwrap();

        assert_eq!(data_manifest.location.as_ref(), Some(&leaf_path));

        // Verify the deletion was recorded in manifest_dv
        let manifest_dv_bytes = data_manifest.manifest_dv_bytes().unwrap();
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(3));

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_with_existing_deleted_entries(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf manifest that already has some deleted entries
        // This simulates a manifest that has been updated over time
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Create a manifest entry with manifest_info showing:
        // - 2 added files (indices 0, 1)
        // - 1 existing file (index 2)
        // - 2 deleted files (indices 3, 4)
        // Total: 5 entries, but only 3 are active (non-deleted)
        let manifest_entry = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("leaf-manifest.parquet")
            .with_tracking(TrackingStatus::Added, 1, 1)
            .record_count(5) // Total entries in the leaf
            .file_size_in_bytes(2048)
            .manifest_info(ManifestInfo {
                added_files_count: 2,
                existing_files_count: 1,
                deleted_files_count: 2, // 2 entries are already deleted
                added_rows_count: 200,
                existing_rows_count: 100,
                deleted_rows_count: 200,
                min_sequence_number: 1,
                ..Default::default()
            })
            .build();

        let leaf_path = manifest_entry.location.as_ref().unwrap().clone();
        root_builder.add_entry(manifest_entry);

        // Delete all 3 active entries (indices 0, 1, 2)
        // With the OLD logic: cardinality (3) != total_entry_count (5), so manifest would NOT be
        // marked deleted With the NEW logic: cardinality (3) == active_entry_count (3), so
        // manifest IS marked deleted
        let mut indices = RoaringTreemap::new();
        indices.extend([0u64, 1, 2]);
        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Delete)?;

        // Build, write, and read back the root to verify the manifest is marked as deleted
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let leaf_manifest = root_entries
            .iter()
            .find(|e| {
                e.content_type == DataContentType::DataManifest
                    && e.location.as_ref() == Some(&leaf_path)
            })
            .expect("Leaf manifest should exist");

        // The critical assertion: manifest should be marked as deleted
        // because all ACTIVE entries (3) have been deleted, even though
        // the total entry count (5) includes 2 already-deleted entries
        assert_eq!(
            leaf_manifest.tracking.status,
            TrackingStatus::Deleted,
            "Manifest should be marked as deleted when all active entries are deleted, \
             even if some entries were already deleted"
        );

        // Verify manifest_dv has cardinality 3 (not 5)
        let manifest_dv_bytes = leaf_manifest
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");

        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert_eq!(
            treemap.len(),
            3,
            "manifest_dv should only track the 3 newly deleted entries"
        );

        Ok(())
    }

    #[test]
    fn test_tracking_deleted_positions_clearing() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("{}data/part-{:05}.parquet", table_root, i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create root and delete entries 2 and 5 (first commit)
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry.clone());
        let mut indices_v1 = RoaringTreemap::new();
        indices_v1.extend([2u64, 5]);
        root_builder.update_leaf_positions(&leaf_path, &indices_v1, LeafPositionUpdate::Delete)?;

        // Step 3: Build, write, and read back the root to verify deleted_positions from first
        // commit
        let entries_v1 = build_and_read_root(&mut root_builder, &engine, 1)?;
        let manifest_v1 = entries_v1
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        // Verify manifest_dv contains both deletions (2 and 5)
        let manifest_dv_v1 = manifest_v1
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative_v1 = RoaringTreemap::deserialize_from(&manifest_dv_v1[4..])?;
        assert!(cumulative_v1.contains(2));
        assert!(cumulative_v1.contains(5));
        assert_eq!(cumulative_v1.len(), 2);

        // Verify deleted_positions contains both deletions from this commit (2 and 5)
        let deleted_positions_v1 = manifest_v1
            .tracking
            .deleted_positions
            .as_ref()
            .expect("deleted_positions should exist");
        let delta_v1 = RoaringTreemap::deserialize_from(&deleted_positions_v1[4..])?;
        assert!(delta_v1.contains(2));
        assert!(delta_v1.contains(5));
        assert_eq!(delta_v1.len(), 2);

        // Step 4: Start a new commit (v2) by loading v1 entries
        // Note: deleted_positions is automatically cleared when entries are added
        let mut root_builder_v2 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 2, test_table_schema());
        for entry in entries_v1 {
            root_builder_v2.add_entry(entry);
        }

        // Step 5: Add new deletions (entries 3 and 7) in the second commit
        let mut indices_v2 = RoaringTreemap::new();
        indices_v2.extend([3u64, 7]);
        root_builder_v2.update_leaf_positions(
            &leaf_path,
            &indices_v2,
            LeafPositionUpdate::Delete,
        )?;

        // Build, write, and read back the root to verify deleted_positions only contains NEW
        // deletions
        let entries_v2 = build_and_read_root(&mut root_builder_v2, &engine, 2)?;
        let manifest_v2 = entries_v2
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        // Verify manifest_dv contains ALL deletions (2, 3, 5, 7)
        let manifest_dv_v2 = manifest_v2
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative_v2 = RoaringTreemap::deserialize_from(&manifest_dv_v2[4..])?;
        assert!(cumulative_v2.contains(2));
        assert!(cumulative_v2.contains(3));
        assert!(cumulative_v2.contains(5));
        assert!(cumulative_v2.contains(7));
        assert_eq!(cumulative_v2.len(), 4);

        // Verify deleted_positions ONLY contains NEW deletions from v2 (3 and 7)
        let deleted_positions_v2 = manifest_v2
            .tracking
            .deleted_positions
            .as_ref()
            .expect("deleted_positions should exist");
        let delta_v2 = RoaringTreemap::deserialize_from(&deleted_positions_v2[4..])?;
        assert!(
            !delta_v2.contains(2),
            "Old deletion (2) should NOT be in delta"
        );
        assert!(delta_v2.contains(3), "New deletion (3) should be in delta");
        assert!(
            !delta_v2.contains(5),
            "Old deletion (5) should NOT be in delta"
        );
        assert!(delta_v2.contains(7), "New deletion (7) should be in delta");
        assert_eq!(
            delta_v2.len(),
            2,
            "Delta should only contain 2 new deletions"
        );

        // Step 8: Start a new commit (v3) by loading v2 entries
        // Note: deleted_positions is automatically cleared when entries are added
        let mut root_builder_v3 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 3, test_table_schema());
        for entry in entries_v2 {
            root_builder_v3.add_entry(entry);
        }

        // Step 9: Delete one additional record (entry 8) in the third commit
        let mut indices_v3 = RoaringTreemap::new();
        indices_v3.insert(8u64);
        root_builder_v3.update_leaf_positions(
            &leaf_path,
            &indices_v3,
            LeafPositionUpdate::Delete,
        )?;

        // Build, write, and read back the root to verify deleted_positions only contains NEW
        // deletion
        let entries_v3 = build_and_read_root(&mut root_builder_v3, &engine, 3)?;
        let manifest_v3 = entries_v3
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        // Verify manifest_dv contains ALL deletions (2, 3, 5, 7, 8)
        let manifest_dv_v3 = manifest_v3
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative_v3 = RoaringTreemap::deserialize_from(&manifest_dv_v3[4..])?;
        assert!(cumulative_v3.contains(2));
        assert!(cumulative_v3.contains(3));
        assert!(cumulative_v3.contains(5));
        assert!(cumulative_v3.contains(7));
        assert!(cumulative_v3.contains(8));
        assert_eq!(cumulative_v3.len(), 5);

        // Verify deleted_positions ONLY contains NEW deletion from v3 (8)
        let deleted_positions_v3 = manifest_v3
            .tracking
            .deleted_positions
            .as_ref()
            .expect("deleted_positions should exist");
        let delta_v3 = RoaringTreemap::deserialize_from(&deleted_positions_v3[4..])?;
        assert!(
            !delta_v3.contains(2),
            "Old deletion (2) should NOT be in delta"
        );
        assert!(
            !delta_v3.contains(3),
            "Old deletion (3) should NOT be in delta"
        );
        assert!(
            !delta_v3.contains(5),
            "Old deletion (5) should NOT be in delta"
        );
        assert!(
            !delta_v3.contains(7),
            "Old deletion (7) should NOT be in delta"
        );
        assert!(delta_v3.contains(8), "New deletion (8) should be in delta");
        assert_eq!(
            delta_v3.len(),
            1,
            "Delta should only contain 1 new deletion"
        );

        // Step 11: Start a new commit (v4) by loading v3 entries
        // Note: deleted_positions is automatically cleared when entries are added
        let mut root_builder_v4 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 4, test_table_schema());
        for entry in entries_v3 {
            root_builder_v4.add_entry(entry);
        }

        // Step 12: Make an unrelated change - add a new data entry (no deletions)
        let new_data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location(format!("{}data/part-{:05}.parquet", table_root, 100))
            .with_tracking(TrackingStatus::Added, 4, 4)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        root_builder_v4.add_entry(new_data_entry);

        // Build, write, and read back the root to verify deleted_positions is None (no deletions)
        let entries_v4 = build_and_read_root(&mut root_builder_v4, &engine, 4)?;
        let manifest_v4 = entries_v4
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        // Verify manifest_dv still contains all previous deletions (2, 3, 5, 7, 8)
        let manifest_dv_v4 = manifest_v4
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative_v4 = RoaringTreemap::deserialize_from(&manifest_dv_v4[4..])?;
        assert_eq!(
            cumulative_v4.len(),
            5,
            "manifest_dv should still have 5 deletions"
        );

        // Verify deleted_positions is None since no deletions were made in v4
        assert!(
            manifest_v4.tracking.deleted_positions.is_none(),
            "deleted_positions should be None when no deletions are made"
        );

        Ok(())
    }

    #[test]
    fn test_tracking_replaced_positions() -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("{}data/part-{:05}.parquet", table_root, i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);
        let mut indices = RoaringTreemap::new();
        indices.extend([1u64, 4]);
        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Replace)?;

        let entries = build_and_read_root(&mut root_builder, &engine, 1)?;
        let manifest = entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        let manifest_dv = manifest
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative = RoaringTreemap::deserialize_from(&manifest_dv[4..])?;
        assert!(cumulative.contains(1));
        assert!(cumulative.contains(4));
        assert_eq!(cumulative.len(), 2);

        assert!(
            manifest.tracking.deleted_positions.is_none(),
            "deleted_positions should not be set for replacements"
        );

        let replaced_positions = manifest
            .tracking
            .replaced_positions
            .as_ref()
            .expect("replaced_positions should exist");
        let delta = RoaringTreemap::deserialize_from(&replaced_positions[4..])?;
        assert!(delta.contains(1));
        assert!(delta.contains(4));
        assert_eq!(delta.len(), 2);

        Ok(())
    }

    #[test]
    fn test_leaf_reorganization_does_not_set_deleted_positions(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        use crate::engine::sync::SyncEngine;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 5 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..5 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("{}data/part-{:05}.parquet", table_root, i))
                .with_tracking(TrackingStatus::Added, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let leaf_manifest_entry =
            leaf_builder.write_leaf(&engine, 1, &mut CursorRowIdAllocator::new(0))?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create root and simulate leaf reorganization (entries moved to a different
        // leaf), which carries entries over rather than deleting them.
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry.clone());

        let mut indices = RoaringTreemap::new();
        indices.insert(2);
        indices.insert(3);

        root_builder.update_leaf_positions(&leaf_path, &indices, LeafPositionUpdate::Carryover)?;

        // Step 3: Build, write, and read back the root to verify deleted_positions is NOT set for
        // leaf reorganization
        let entries = build_and_read_root(&mut root_builder, &engine, 1)?;
        let manifest = entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::DataManifest))
            .expect("DataManifest should exist");

        // Verify manifest_dv contains the deletions (for internal tracking)
        let manifest_dv = manifest
            .manifest_dv_bytes()
            .expect("manifest_dv should exist");
        let cumulative = RoaringTreemap::deserialize_from(&manifest_dv[4..])?;
        assert!(cumulative.contains(2));
        assert!(cumulative.contains(3));
        assert_eq!(cumulative.len(), 2);

        // Verify deleted_positions is NOT set since this was leaf reorganization, not actual
        // deletion
        assert!(
            manifest.tracking.deleted_positions.is_none(),
            "deleted_positions should NOT be set for leaf reorganization"
        );
        assert!(
            manifest.tracking.replaced_positions.is_none(),
            "replaced_positions should NOT be set for leaf reorganization"
        );

        Ok(())
    }

    #[test]
    fn test_remove_entries_by_file_path() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        let temp_dir = tempdir()?;
        // Canonicalize the path to match what try_parse_uri does in real usage
        // This ensures paths are consistent (e.g., /private/var instead of /var on macOS)
        let canonical_path = std::fs::canonicalize(temp_dir.path())?;
        let table_root = Url::from_directory_path(canonical_path).unwrap();

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Add three entries with different file paths
        let entry1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("file1.parquet")
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let entry2 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("file2.parquet")
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let entry3 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("file3.parquet")
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        builder.add_entry(entry1);
        builder.add_entry(entry2);
        builder.add_entry(entry3);

        assert_eq!(builder.pending_entries.len(), 3);

        // Remove file1.parquet
        builder.remove_data_file("file1.parquet")?;

        // Should have 2 entries remaining
        assert_eq!(builder.pending_entries.len(), 2);
        assert!(builder.pending_entries.iter().any(|e| e
            .location
            .as_ref()
            .unwrap()
            .ends_with("file2.parquet")));
        assert!(builder.pending_entries.iter().any(|e| e
            .location
            .as_ref()
            .unwrap()
            .ends_with("file3.parquet")));
        assert!(!builder.pending_entries.iter().any(|e| e
            .location
            .as_ref()
            .unwrap()
            .ends_with("file1.parquet")));

        Ok(())
    }

    #[test]
    fn test_remove_entries_by_dv_path() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        let temp_dir = tempdir()?;
        // Canonicalize the path to match what try_parse_uri does in real usage
        // This ensures paths are consistent (e.g., /private/var instead of /var on macOS)
        let canonical_path = std::fs::canonicalize(temp_dir.path())?;
        let table_root = Url::from_directory_path(canonical_path).unwrap();

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Add a Data entry with inline DV info
        let data_entry_with_dv = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("dv1.bin")
            .record_count(10)
            .file_size_in_bytes(128)
            .build();
        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data1.parquet")
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        builder.add_entry(data_entry_with_dv);
        builder.add_entry(data_entry);

        assert_eq!(builder.pending_entries.len(), 2);

        // Remove by location path
        builder.remove_dv("dv1.bin")?;

        // Should have 1 entry remaining (the data entry)
        assert_eq!(builder.pending_entries.len(), 1);
        assert_eq!(
            builder.pending_entries[0].content_type,
            DataContentType::Data
        );

        Ok(())
    }

    #[test]
    fn test_remove_data_file_keeps_other_entries() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        let temp_dir = tempdir()?;
        // Canonicalize the path to match what try_parse_uri does in real usage
        // This ensures paths are consistent (e.g., /private/var instead of /var on macOS)
        let canonical_path = std::fs::canonicalize(temp_dir.path())?;
        let table_root = Url::from_directory_path(canonical_path).unwrap();

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Add two data entries; data1 has inline DV info, data2 doesn't
        let data_entry1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data1.parquet")
            .deletion_vector(DeletionVectorInfo {
                location: "dv1.bin".to_string(),
                offset: 0,
                size_in_bytes: 48,
                cardinality: 6,
            })
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let data_entry2 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data2.parquet")
            .record_count(50)
            .file_size_in_bytes(512)
            .build();

        builder.add_entry(data_entry1);
        builder.add_entry(data_entry2);

        assert_eq!(builder.pending_entries.len(), 2);

        // Remove only the first data file (also removes its inline DV)
        builder.remove_data_file("data1.parquet")?;

        // Should have 1 entry remaining (data2 without DV)
        assert_eq!(builder.pending_entries.len(), 1);
        assert_eq!(
            builder.pending_entries[0].location.as_deref(),
            Some("data2.parquet")
        );

        Ok(())
    }

    #[test]
    fn test_remove_entries_no_match() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;

        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Add entry
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("file1.parquet")
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        builder.add_entry(entry);

        assert_eq!(builder.pending_entries.len(), 1);

        // Try to remove non-existent file
        builder.remove_data_file("nonexistent.parquet")?;

        // Should still have 1 entry
        assert_eq!(builder.pending_entries.len(), 1);

        Ok(())
    }

    /// Helper that creates a minimal Add action for unit tests.
    fn make_test_add(path: &str) -> Add {
        Add {
            path: path.to_string(),
            partition_values: HashMap::new(),
            size: 1024,
            modification_time: 1000000,
            data_change: true,
            stats: None,
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            back_reference: None,
        }
    }

    #[test]
    fn test_add_stamps_entry_with_commit_version() -> Result<(), Box<dyn std::error::Error>> {
        use tempfile::tempdir;
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Builder version is 5 (new root being built).
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 5, test_table_schema());

        builder.add_with_status(
            make_test_add("file1.parquet"),
            2,
            100,
            TrackingStatus::Existing,
        )?;

        assert_eq!(builder.pending_entries.len(), 1);
        let entry = &builder.pending_entries[0];
        let ti = &entry.tracking;
        assert_eq!(
            ti.sequence_number,
            Some(2),
            "sequence_number must be the commit version, not the root version"
        );
        assert_eq!(
            ti.status,
            TrackingStatus::Existing,
            "file from an earlier version must have Existing status"
        );

        Ok(())
    }

    // Note: Deletion vector extraction from scan rows is tested through integration tests
    // since creating mock scan row data with the complex nested schema structure is difficult.
    // The extraction logic is verified through:
    // - metadata tests (test_dv_with_later_sequence_number_included, etc.)
    // - Full table scans with the backfill tool

    // Disabled complex unit test - see note above

    // --- Tests for assign_first_row_ids_to_pending ---

    fn make_data_entry(
        record_count: i64,
        status: TrackingStatus,
        first_row_id: Option<i64>,
    ) -> ContentTreeNodeEntry {
        ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location(format!("file-{}.parquet", record_count))
            .tracking(TrackingInfo {
                status,
                snapshot_id: Some(1),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                first_row_id,
                dv_snapshot_id: None,
                deleted_positions: None,
                replaced_positions: None,
            })
            .record_count(record_count)
            .file_size_in_bytes(1024)
            .build()
    }

    fn make_manifest_entry(
        added_rows: i64,
        existing_rows: i64,
        status: TrackingStatus,
        first_row_id: Option<i64>,
    ) -> ContentTreeNodeEntry {
        ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location(format!("manifest-{}-{}.parquet", added_rows, existing_rows))
            .tracking(TrackingInfo {
                status,
                snapshot_id: Some(1),
                sequence_number: None,
                file_sequence_number: None,
                first_row_id,
                dv_snapshot_id: None,
                deleted_positions: None,
                replaced_positions: None,
            })
            .record_count(added_rows + existing_rows)
            .file_size_in_bytes(2048)
            .manifest_info_opt(Some(ManifestInfo {
                added_files_count: 1,
                existing_files_count: 1,
                deleted_files_count: 0,
                replaced_files_count: 0,
                added_rows_count: added_rows,
                existing_rows_count: existing_rows,
                deleted_rows_count: 0,
                replaced_rows_count: 0,
                min_sequence_number: 1,
                dv: None,
                dv_cardinality: None,
            }))
            .build()
    }

    #[test]
    fn test_assign_first_row_ids_data_entries_only() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_data_entry(200, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_data_entry(50, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(allocator.current(), 350);
        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(100));
        assert_eq!(builder.pending_entries[2].tracking.first_row_id, Some(300));
    }

    #[test]
    fn test_assign_first_row_ids_combined_manifest_entries() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder
            .pending_entries
            .push(make_manifest_entry(100, 200, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_manifest_entry(50, 50, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(allocator.current(), 400);
        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(300));
    }

    #[test]
    fn test_assign_first_row_ids_preserves_existing() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder.pending_entries.push(make_manifest_entry(
            100,
            200,
            TrackingStatus::Existing,
            Some(0),
        ));
        builder
            .pending_entries
            .push(make_manifest_entry(50, 50, TrackingStatus::Added, None));

        // Allocator starts at HWM+1 = 300 (existed entry covers [0, 300))
        let mut allocator = CursorRowIdAllocator::new(300);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(300));
        assert_eq!(allocator.current(), 400);
    }

    #[test]
    fn test_assign_first_row_ids_deleted_entries_skipped() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_data_entry(200, TrackingStatus::Deleted, Some(999)));
        builder
            .pending_entries
            .push(make_data_entry(50, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(999));
        assert_eq!(builder.pending_entries[2].tracking.first_row_id, Some(100));
        assert_eq!(allocator.current(), 150);
    }

    #[test]
    fn test_assign_first_row_ids_mixed_data_and_manifests() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_manifest_entry(50, 150, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_data_entry(75, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(100));
        assert_eq!(builder.pending_entries[2].tracking.first_row_id, Some(300));
        assert_eq!(allocator.current(), 375);
    }

    #[test]
    fn test_assign_first_row_ids_nonzero_starting_value() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Added, None));
        builder
            .pending_entries
            .push(make_data_entry(200, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(501);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(501));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(601));
        assert_eq!(allocator.current(), 801);
    }

    // --- Tests for Iceberg row lineage compatibility ---

    /// Verifies that the eager first_row_id assignment for data files within a leaf
    /// produces values equivalent to Iceberg's lazy inheritance model: each data file's
    /// first_row_id == manifest's first_row_id + sum of preceding files' record_counts.
    #[test]
    fn test_assign_first_row_ids_iceberg_inheritance_equivalence() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        // Simulate a leaf manifest containing 3 data files: 100, 50, 200 records
        let record_counts = [100i64, 50, 200];
        for &rc in &record_counts {
            builder
                .pending_entries
                .push(make_data_entry(rc, TrackingStatus::Added, None));
        }

        let manifest_first_row_id = 42i64;
        let mut allocator = CursorRowIdAllocator::new(manifest_first_row_id);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        // Verify Iceberg inheritance equivalence:
        // file[i].first_row_id == manifest_first_row_id + sum(record_counts[0..i])
        let mut cumulative = 0i64;
        for (i, &rc) in record_counts.iter().enumerate() {
            let expected = manifest_first_row_id + cumulative;
            assert_eq!(
                builder.pending_entries[i]
                    .tracking
                    .first_row_id,
                Some(expected),
                "file {i}: expected first_row_id={expected} (manifest={manifest_first_row_id} + cumulative={cumulative})"
            );
            cumulative += rc;
        }

        // allocator cursor == manifest_first_row_id + total_records
        assert_eq!(allocator.current(), manifest_first_row_id + cumulative);
    }

    /// Existed data entries with null first_row_id (from scan-row rebuild) get
    /// correctly assigned new IDs, matching Iceberg's rule that all unassigned
    /// first_row_id values require inheritance assignment.
    #[test]
    fn test_assign_first_row_ids_existed_null_get_assigned() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        // Existed entry with null first_row_id (e.g., from table upgrade or scan rebuild)
        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Existing, None));
        // Added entry after it
        builder
            .pending_entries
            .push(make_data_entry(50, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        // The Existed entry should be assigned first_row_id=0
        assert_eq!(
            builder.pending_entries[0].tracking.first_row_id,
            Some(0),
            "Existed entry with null first_row_id should be assigned"
        );
        // The Added entry should follow sequentially
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(100));
        assert_eq!(allocator.current(), 150);
    }

    /// PositionDeletes and EqualityDeletes content types never get first_row_id
    /// assigned, matching Iceberg's rule that delete files always have null first_row_id.
    #[test]
    fn test_assign_first_row_ids_delete_content_types_always_null() {
        let table_root = Url::parse("memory:///test/").unwrap();
        let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

        // Data entry first
        builder
            .pending_entries
            .push(make_data_entry(100, TrackingStatus::Added, None));

        // PositionDeletes entry
        builder.pending_entries.push(
            ContentTreeNodeEntryBuilder::new(DataContentType::PositionDeletes)
                .location("pos-deletes.parquet")
                .tracking(TrackingInfo {
                    status: TrackingStatus::Added,
                    snapshot_id: Some(1),
                    sequence_number: Some(1),
                    file_sequence_number: Some(1),
                    first_row_id: None,
                    dv_snapshot_id: None,
                    deleted_positions: None,
                    replaced_positions: None,
                })
                .record_count(50)
                .file_size_in_bytes(512)
                .build(),
        );

        // EqualityDeletes entry
        builder.pending_entries.push(
            ContentTreeNodeEntryBuilder::new(DataContentType::EqualityDeletes)
                .location("eq-deletes.parquet")
                .tracking(TrackingInfo {
                    status: TrackingStatus::Added,
                    snapshot_id: Some(1),
                    sequence_number: Some(1),
                    file_sequence_number: Some(1),
                    first_row_id: None,
                    dv_snapshot_id: None,
                    deleted_positions: None,
                    replaced_positions: None,
                })
                .record_count(25)
                .file_size_in_bytes(256)
                .build(),
        );

        // Another data entry after the deletes
        builder
            .pending_entries
            .push(make_data_entry(75, TrackingStatus::Added, None));

        let mut allocator = CursorRowIdAllocator::new(0);
        builder.assign_first_row_ids_to_pending(&mut allocator);

        // Data entry gets assigned
        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        // PositionDeletes: no first_row_id assignment
        assert_eq!(
            builder.pending_entries[1].tracking.first_row_id, None,
            "PositionDeletes should not get first_row_id"
        );
        // EqualityDeletes: no first_row_id assignment
        assert_eq!(
            builder.pending_entries[2].tracking.first_row_id, None,
            "EqualityDeletes should not get first_row_id"
        );
        // Next data entry picks up where the first left off (deletes don't consume IDs)
        assert_eq!(builder.pending_entries[3].tracking.first_row_id, Some(100));
        assert_eq!(allocator.current(), 175);
    }

    #[test]
    fn test_content_stats_from_delta_stats_parsed_skips_array_without_shifting_later_columns() {
        fn field_with_id(name: &str, data_type: DataType, field_id: i64) -> StructField {
            StructField::nullable(name, data_type).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(field_id),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(field_id),
                ),
            ])
        }

        // The array column sits between two primitives: it is absent from the AMT stats schema,
        // so 'b' must still resolve to its own Delta stats rather than the array's.
        let table_schema = StructType::new_unchecked([
            field_with_id("a", DataType::INTEGER, 1),
            field_with_id(
                "l",
                DataType::Array(Box::new(ArrayType::new(DataType::INTEGER, true))),
                2,
            ),
            field_with_id("b", DataType::STRING, 3),
        ]);
        let amt_schema = stats::stats_schema(&table_schema).unwrap();
        assert_eq!(
            amt_schema
                .fields()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );

        let expr = build_content_stats_from_delta_stats_parsed(&table_schema, &amt_schema).unwrap();
        let Expression::Struct(col_exprs, _) = expr else {
            panic!("expected a struct expression");
        };
        assert_eq!(col_exprs.len(), 2);

        // Every column reference in a column's stats sub-expression must name that same column.
        for (col_expr, expected_col) in col_exprs.iter().zip(["a", "b"]) {
            let Expression::Struct(field_exprs, _) = col_expr.as_ref() else {
                panic!("expected a struct expression per column");
            };
            for field_expr in field_exprs {
                let Expression::Column(name) = field_expr.as_ref() else {
                    continue;
                };
                // Only per-column stats are nested under minValues/maxValues/nullCount;
                // numRecords and tightBounds are table-wide and carry no column component.
                let parts: Vec<&str> = name.iter().map(String::as_str).collect();
                if parts.len() == 3 && parts[0] == STATS_PARSED_NAME {
                    assert_eq!(parts[2], expected_col);
                }
            }
        }
    }
}
