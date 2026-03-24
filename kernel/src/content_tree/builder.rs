use crate::actions::deletion_vector::DeletionVectorDescriptor;
use crate::actions::Add;
use crate::content_tree::stats::{aggregate_content_stats, delta_json_stats_to_content_stats};
use crate::content_tree::writer::ContentTreeNodeWriter;
use crate::content_tree::{
    absolute_to_relative_path, ContentTreeNode, ContentTreeNodeEntry, ContentTreeNodeEntryBuilder,
    DataContentType, DvInfo, TrackingInfo, TrackingStatus, DELTA_STATS_MAX_VALUES,
    DELTA_STATS_MIN_VALUES, DELTA_STATS_NULL_COUNT, DELTA_STATS_NUM_RECORDS,
    DELTA_STATS_TIGHT_BOUNDS,
};
use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};

#[cfg(test)]
use crate::content_tree::ManifestStats;
use crate::expressions::StructData;
use crate::schema::{ColumnName, ColumnNamesAndTypes, DataType, Schema, SchemaRef};
#[cfg(test)]
use crate::utils::try_parse_uri;
use crate::{DeltaResult, Engine, EngineData, Error, FilteredEngineData, Version};
use bytes::Bytes;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, OnceLock};
use tracing::instrument;
use url::Url;

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

/// Extracts deletion vector content from a DeletionVectorDescriptor.
///
/// This function decodes the `path_or_inline_dv` field based on the storage type:
///
/// - `PersistedRelative`: The format is `<random prefix - optional><base85 encoded uuid>`.
///   The UUID is 20 characters (base85 encoded), and any characters before that are the
///   optional random prefix. The function reconstructs the absolute path to the DV file.
///
/// - `PersistedAbsolute`: The `path_or_inline_dv` contains the absolute path to the DV file.
///
/// - `Inline`: Currently not supported - returns an error. Inline DVs would need to be
///   persisted first before being added to metadata.
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
/// Therefore, when converting from Delta to Iceberg's [`DvInfo`], we add 8 bytes
/// (4 for size prefix + 4 for CRC) to Delta's `size_in_bytes`.
///
/// # Returns
/// A [`DvInfo`] containing the DV location and size information.
pub(crate) fn extract_deletion_vector_content(
    dv: &DeletionVectorDescriptor,
) -> DeltaResult<DvInfo> {
    use crate::actions::deletion_vector::DeletionVectorStorageType;
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
    Ok(DvInfo {
        location,
        offset: dv.offset.map(|v| v as i64).unwrap_or(0),
        size_in_bytes: dv.size_in_bytes as i64 + 8,
        cardinality: dv.cardinality,
    })
}

/// Extracts record_count from content_stats by finding the first column's value_count.
///
/// In the content_stats format, each column has a stats struct containing value_count,
/// which represents the number of records in the file. This value is the same for all
/// columns in a properly formed stats struct.
///
/// # Arguments
/// * `content_stats` - Optional reference to the content_stats StructData
///
/// # Returns
/// The record count (value_count from the first column's stats), or 0 if not available.
fn extract_record_count_from_stats(content_stats: Option<&StructData>) -> i64 {
    use crate::expressions::Scalar;

    let Some(stats) = content_stats else {
        return 0;
    };

    // Iterate through the columns to find the first one with value_count
    for value in stats.values() {
        if let Scalar::Struct(column_stats) = value {
            // Look for value_count field in the column's stats struct
            for (field, field_value) in column_stats.fields().iter().zip(column_stats.values()) {
                if field.name() == crate::content_tree::VALUE_COUNT {
                    if let Scalar::Long(count) = field_value {
                        return *count;
                    }
                }
            }
        }
    }

    0
}

/// Cache for DV bitmaps with lazy deserialization
struct DvCache {
    /// Original serialized manifest_dv bytes (from previous commits)
    /// Kept as reference (Bytes is Rc-based, cheap to clone)
    serialized_manifest_dv: Option<Bytes>,

    /// Lazily deserialized manifest_dv (only populated when modified)
    manifest_dv: Option<roaring::RoaringTreemap>,

    /// Changes DV for current commit (always starts empty)
    changes_dv: roaring::RoaringTreemap,

    /// Track if this entry was modified (deserialized)
    dirty: bool,

    /// Total number of entries in the manifest (for bounds checking)
    /// Cached from manifest_stats to avoid O(n) scans
    total_entry_count: i64,
}

impl DvCache {
    fn new(serialized_manifest_dv: Option<Bytes>, total_entry_count: i64) -> Self {
        Self {
            serialized_manifest_dv,
            manifest_dv: None,
            changes_dv: roaring::RoaringTreemap::new(),
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

    /// Get mutable references to both bitmaps at once (avoids borrow checker issues)
    /// Ensures manifest_dv is loaded first
    fn get_both_dvs_mut(
        &mut self,
    ) -> DeltaResult<(&mut roaring::RoaringTreemap, &mut roaring::RoaringTreemap)> {
        self.ensure_manifest_dv_loaded()?;
        self.dirty = true;
        let manifest_dv = self.manifest_dv.as_mut().ok_or_else(|| {
            Error::generic("Internal bug: manifest_dv not loaded after ensure_manifest_dv_loaded")
        })?;
        Ok((manifest_dv, &mut self.changes_dv))
    }
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
    /// Set of seen file paths to prevent duplicate entries.
    /// Only populated when processing existing actions, not new actions.
    values_seen: HashSet<String>,
    /// Cached schema with content_stats. Computed lazily on first use.
    cached_schema: OnceLock<SchemaRef>,
    /// Combined cache for DV bitmaps (manifest_dv + changes_dv)
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
    added_file_count: i64,
    existing_file_count: i64,
    total_record_count: i64,
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
    ///   This parameter is essential for converting Delta JSON stats (minValues, maxValues, nullCount)
    ///   to the content_stats StructData format when adding entries via `add()`. The schema must
    ///   match the schema used to write the files and must include PARQUET:field_id metadata on
    ///   fields for proper stats field mapping
    pub(crate) fn new_for(table_root: Url, version: Version, table_schema: Schema) -> Self {
        Self {
            table_root,
            pending_entries: Vec::new(),
            version,
            table_schema,
            values_seen: HashSet::new(),
            cached_schema: OnceLock::new(),
            dv_cache: HashMap::new(),
            pre_built_data: Vec::new(),
            pre_built_aggregates: Vec::new(),
        }
    }

    /// Creates a [`ContentTreeNodeBuilder`] by reading an existing content root parquet file.
    ///
    /// This reads and validates the content root via [`ContentTreeNode::from_batches_with_version`],
    /// then populates a builder from the validated entries.
    ///
    /// # Arguments
    /// * `engine` - The engine to use for reading the parquet file
    /// * `content_root` - The content root action referencing the manifest file
    /// * `table_root` - The root URL of the table
    /// * `table_schema` - The table schema with PARQUET:field_id metadata for stats conversion
    /// * `new_version` - The version number for the new metadata being built
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
        )?;

        let data: Vec<Box<dyn EngineData>> = read_result_iter.collect::<DeltaResult<Vec<_>>>()?;

        let node = ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_root.clone(),
        )?;

        let entries = node.entries()?;
        let mut builder = Self::new_for(table_root, new_version, table_schema);
        for entry in entries {
            let entry = if entry.tracking.status == TrackingStatus::Added
                && entry.tracking.sequence_number != Some(new_version as i64)
            {
                entry.with_status(TrackingStatus::Existed)
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
                DataContentType::DataManifest
                    | DataContentType::DeleteManifest
                    | DataContentType::CombinedManifest
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

            // Serialize manifest_dv if it was deserialized
            if let Some(ref manifest_dv) = cache.manifest_dv {
                entry.manifest_dv = Some(serialize_roaring_treemap(manifest_dv)?);

                // Update tracking status based on DV cardinality
                // If all active entries are deleted, mark manifest as Deleted
                if let Some(ref manifest_stats) = entry.manifest_stats {
                    let active_entry_count =
                        manifest_stats.added_files_count + manifest_stats.existing_files_count;
                    let cardinality = manifest_dv.len() as i64;

                    if cardinality == active_entry_count {
                        entry.tracking.status = TrackingStatus::Deleted;
                    }
                }
            }

            // Serialize changes_dv if non-empty
            if !cache.changes_dv.is_empty() {
                entry.tracking.changes_dv = Some(serialize_roaring_treemap(&cache.changes_dv)?);
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

        // Compute the schema (include all columns for writing)
        let delta_stats = build_delta_stats_schema(&self.table_schema);
        let schema =
            ContentTreeNodeEntry::to_schema_with_content_stats(&self.table_schema, &delta_stats)?;
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
        use url::Url;

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
                path, &self.table_root, e
            ))
        })?;

        Ok(absolute_url.to_string())
    }

    /// Add a data file entry, deduplicating by file path.
    ///
    /// Accepts pre-computed content_stats directly as a [`StructData`], avoiding the need
    /// to serialize/deserialize JSON stats.
    ///
    /// # Arguments
    /// * `path` - The file path (relative to table root)
    /// * `size` - The file size in bytes
    /// * `content_stats` - Optional content_stats as StructData
    /// * `version` - The version to use for tracking info
    /// * `snapshot_id` - The snapshot ID for tracking info
    /// * `dv_info` - Optional deletion vector info
    pub(crate) fn add_file(
        &mut self,
        path: String,
        size: i64,
        content_stats: Option<StructData>,
        version: Version,
        snapshot_id: i64,
        dv_info: Option<DvInfo>,
    ) -> DeltaResult<()> {
        // Check for duplicates and skip if already seen
        if !self.values_seen.insert(path.clone()) {
            // Already seen this file path - skip it
            return Ok(());
        }

        // Extract record_count from the content_stats (from any column's value_count)
        let record_count = extract_record_count_from_stats(content_stats.as_ref());

        let data_file_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location(path)
            .with_tracking(version, self.version, snapshot_id)
            .dv_info_opt(dv_info)
            .record_count(record_count)
            .file_size_in_bytes(size)
            .content_stats_opt(content_stats)
            .build();

        self.pending_entries.push(data_file_entry);
        Ok(())
    }

    /// Add an entry from an [`Add`] action, deduplicating by file path.
    ///
    /// Extracts deletion vector content, parses and converts stats from the `Add` action,
    /// then delegates to [`add_file`](Self::add_file).
    ///
    /// # Arguments
    /// * `add` - The Add action to convert to a ContentTreeNodeEntry
    /// * `version` - The version to use for tracking info
    /// * `snapshot_id` - The snapshot ID for tracking info
    pub(crate) fn add(&mut self, add: Add, version: Version, snapshot_id: i64) -> DeltaResult<()> {
        // Extract deletion vector content if present
        let dv_content = add
            .deletion_vector
            .as_ref()
            .map(extract_deletion_vector_content)
            .transpose()?;

        // TODO: Check if parsed_stats is set and prefer that over the JSON blob
        // Convert Delta JSON stats to content_stats. When the file has a deletion vector and
        // tightBounds is null/absent, treat bounds as not tight (safe for data skipping).
        let content_stats = delta_json_stats_to_content_stats(
            add.stats.as_deref(),
            &self.table_schema,
            add.deletion_vector.is_some().then_some(false),
        )?;

        self.add_file(
            add.path,
            add.size,
            content_stats,
            version,
            snapshot_id,
            dv_content,
        )
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
        use crate::content_tree::stats;

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
        stats_struct: Option<&crate::schema::StructType>,
    ) -> DeltaResult<(Box<dyn EngineData>, BatchAggregates)> {
        use crate::content_tree::{DataContentType, TrackingStatus, CONTENT_STATS_FIELD_NAME};
        use crate::expressions::{Expression, Scalar};
        use crate::schema::{MapType, StructField, StructType};

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

        let version_i64 = version as i64;

        // Build recordCount and content_stats expressions based on stats availability
        let (record_count_expr, content_stats_expr) = match stats_struct {
            Some(ss) => {
                let rc = if let Some(first_col) = ss.fields().next().map(|f| f.name().clone()) {
                    Expression::coalesce([
                        Expression::column([
                            "stats",
                            first_col.as_str(),
                            crate::content_tree::VALUE_COUNT,
                        ]),
                        Expression::literal(Scalar::Long(0)),
                    ])
                } else {
                    Expression::literal(Scalar::Long(0))
                };
                (rc, Expression::column(["stats"]))
            }
            None => {
                let stats_null_type = output_schema
                    .field(CONTENT_STATS_FIELD_NAME)
                    .map(|f| f.data_type().clone())
                    .unwrap_or(DataType::STRING);
                (
                    Expression::literal(Scalar::Long(0)),
                    Expression::null_literal(stats_null_type),
                )
            }
        };

        // Build field expressions mapping input → output for each ContentTreeNodeEntry field
        let mut field_exprs: Vec<Arc<Expression>> = Vec::new();
        for field in output_schema.fields() {
            let expr: Expression = match field.name().as_str() {
                "contentType" => Expression::literal(Scalar::Integer(DataContentType::Data as i32)),
                "location" => Expression::column(["path"]),
                "fileFormat" => Expression::literal(Scalar::String("parquet".into())),
                "tracking" => {
                    if !matches!(field.data_type(), DataType::Struct(_)) {
                        return Err(crate::Error::generic(
                            "tracking field should be a struct type",
                        ));
                    }
                    let snapshot_id_expr = Expression::literal(Scalar::Long(snapshot_id));
                    Expression::struct_from([
                        Expression::literal(Scalar::Integer(TrackingStatus::Added as i32)),
                        snapshot_id_expr,
                        Expression::literal(Scalar::Long(version_i64)),
                        Expression::literal(Scalar::Long(version_i64)),
                        Expression::null_literal(DataType::LONG), // firstRowId
                        Expression::null_literal(DataType::BINARY), // changesDv
                    ])
                }
                "dvInfo" => Expression::null_literal(field.data_type().clone()),
                "partitionSpecId" => Expression::literal(Scalar::Long(0)),
                "sortOrderId" => Expression::null_literal(DataType::LONG),
                "recordCount" => record_count_expr.clone(),
                "fileSizeInBytes" => Expression::column(["size"]),
                CONTENT_STATS_FIELD_NAME => content_stats_expr.clone(),
                "manifestStats" => Expression::null_literal(field.data_type().clone()),
                "referencedFile" => Expression::null_literal(DataType::STRING),
                "manifestDv" => Expression::null_literal(DataType::BINARY),
                _ => Expression::null_literal(field.data_type().clone()),
            };
            field_exprs.push(Arc::new(expr));
        }

        // Create the struct transform expression
        let transform_expr = Expression::struct_from(field_exprs);

        // Create evaluator and evaluate
        let evaluator = engine.evaluation_handler().new_expression_evaluator(
            input_schema,
            Arc::new(transform_expr),
            DataType::Struct(Box::new(output_schema.as_ref().clone())),
        )?;
        let transformed = evaluator.evaluate(engine_data)?;

        // Compute lightweight aggregates from the transformed output (flat i64 columns)
        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(transformed.as_ref())?;

        let aggregates = BatchAggregates {
            added_file_count: engine_data.len() as i64,
            existing_file_count: 0,
            total_record_count: agg_visitor.total_record_count,
        };

        Ok((transformed, aggregates))
    }

    /// Adds file metadata from scan row format `EngineData` to the metadata.
    ///
    /// This method is designed for scenarios where the data comes from a scan operation
    /// and has the scan row schema format (path, size, modificationTime, stats at top level,
    /// with fileConstantValues.partitionValues nested).
    ///
    /// # Arguments
    /// * `engine_data` - The engine data containing scan row records to extract and add
    /// * `version` - The version at which these files are being added
    /// * `snapshot_id` - Optional snapshot ID to use for tracking info
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if there was an error visiting the engine data
    pub(crate) fn add_from_scan_row_data(
        &mut self,
        engine_data: &dyn EngineData,
        version: Version,
        snapshot_id: i64,
    ) -> Result<(), crate::Error> {
        let mut visitor = ScanRowToAddVisitor {
            adds: vec![],
            selection_vector: vec![],
            row_offset: 0,
        };
        visitor.visit_rows_of(engine_data)?;

        for add in visitor.adds {
            self.add(add, version, snapshot_id)?;
        }

        Ok(())
    }

    /// Adds a raw ContentTreeNodeEntry to the builder.
    ///
    /// This is useful when copying entries from existing metadata.
    pub(crate) fn add_entry(&mut self, mut entry: ContentTreeNodeEntry) {
        // Create DvCache for manifest entries
        if matches!(
            entry.content_type,
            DataContentType::DataManifest
                | DataContentType::DeleteManifest
                | DataContentType::CombinedManifest
        ) {
            if let Some(ref location) = entry.location {
                // Get total entry count from manifest_stats for bounds checking
                let total_entry_count = if let Some(ref manifest_stats) = entry.manifest_stats {
                    manifest_stats.added_files_count
                        + manifest_stats.existing_files_count
                        + manifest_stats.deletes_files_count
                } else {
                    0
                };

                // Keep serialized manifest_dv bytes in entry, clone into cache
                // Bytes is Rc-based, so clone is cheap (just increments refcount)
                let cache = DvCache::new(entry.manifest_dv.clone(), total_entry_count);
                self.dv_cache.insert(location.clone(), cache);

                // Always clear changes_dv from entries (starts empty for new commit)
                entry.tracking.changes_dv = None;
            }
        }

        // Add entry to Vec (manifest_dv serialized bytes kept intact)
        self.pending_entries.push(entry);
    }

    /// Returns true if this builder has any pending entries.
    pub(crate) fn has_entries(&self) -> bool {
        !self.pending_entries.is_empty() || !self.pre_built_data.is_empty()
    }

    /// Remove data file entries by path. Only used when moving values in the root
    /// to the leaves (otherwise mark deleted it should be used.
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

    /// Remove DV entries by DV location or referenced file. Only used when moving values in the root
    /// to the leaves (otherwise mark deleted it should be used.
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
        use crate::content_tree::DataContentType;

        // Collect locations of entries being removed
        let removed_locations: Vec<String> = self
            .pending_entries
            .iter()
            .filter(|entry| {
                !matches!(
                    entry.content_type,
                    DataContentType::DataManifest
                        | DataContentType::DeleteManifest
                        | DataContentType::CombinedManifest
                )
            })
            .filter_map(|entry| entry.location.clone())
            .collect();

        self.pending_entries.retain(|entry| {
            // Keep only manifest reference entries (these point to leaf manifests)
            // Remove actual data/DV entries from root
            matches!(
                entry.content_type,
                DataContentType::DataManifest
                    | DataContentType::DeleteManifest
                    | DataContentType::CombinedManifest
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
    ///
    pub(crate) fn mark_deleted(
        &mut self,
        file_path: Option<&str>,
        dv_path: Option<&str>,
        snapshot_id: i64,
    ) -> DeltaResult<()> {
        // TODO: we should make pending entries a HashMap<String, ContentTreeNodeEntry> to make this faster
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

    /// Delete multiple entries from a leaf manifest by marking them as deleted via ManifestDV.
    ///
    /// Used by the transaction layer when processing manifest DVs from leaf writers.
    ///
    /// # Arguments
    /// * `leaf_file_path` - Path to the leaf manifest file
    /// * `indices` - Roaring bitmap containing indices to mark as deleted
    /// * `set_changes_dv` - If true, sets tracking.changes_dv (for actual deletions).
    ///   If false, only updates manifest_dv (for leaf reorganization).
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if the leaf manifest is not found, missing manifest_stats, any index is out of bounds, or serialization fails
    pub(crate) fn delete_multiple_from_leaf(
        &mut self,
        leaf_file_path: &str,
        indices: &roaring::RoaringTreemap,
        set_changes_dv: bool,
    ) -> DeltaResult<()> {
        self.delete_indices_from_leaf(leaf_file_path, indices, set_changes_dv)
    }

    /// Core implementation for marking entries in a leaf manifest as deleted.
    ///
    /// Updates the manifest entry's DV fields to mark entries as deleted.
    ///
    /// # Arguments
    /// * `leaf_file_path` - Path to the leaf manifest file
    /// * `indices` - Roaring bitmap containing indices to mark as deleted
    /// * `set_changes_dv` - If true, sets tracking.changes_dv to track this as an actual deletion.
    ///   If false (e.g., when moving entries between leaves), only updates manifest_dv.
    fn delete_indices_from_leaf(
        &mut self,
        leaf_file_path: &str,
        indices: &roaring::RoaringTreemap,
        set_changes_dv: bool,
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

        let (combined_bitmap, delta_bitmap) = cache.get_both_dvs_mut()?;

        // Update bitmaps
        *combined_bitmap |= indices;
        if set_changes_dv {
            *delta_bitmap |= indices;
        }

        // tracking will be updated during write_leaf/build when we're already iterating

        Ok(())
    }

    /// Writes the pending entries as a leaf manifest and returns a ContentTreeNodeEntry referencing it.
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
    /// Builds and writes a leaf manifest, returning the CombinedManifest entry for the root.
    ///
    /// Assigns sequential `first_row_id` values to data entries in the leaf starting from
    /// `starting_first_row_id`. The returned CombinedManifest entry will have its
    /// `first_row_id` set to the starting value used for the leaf's data entries.
    /// Returns the manifest entry and the next available row ID after all assignments.
    pub(crate) fn write_leaf(
        &mut self,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
        starting_first_row_id: i64,
    ) -> DeltaResult<(ContentTreeNodeEntry, i64)> {
        // Build the leaf metadata with a UUID
        let (leaf_metadata, next_row_id) =
            self.build_leaf(engine, snapshot_id, starting_first_row_id)?;

        let write_result = ContentTreeNodeWriter::try_new(leaf_metadata)?.write(engine)?;
        let manifest_path = absolute_to_relative_path(&write_result.location, &self.table_root)?;
        // Use the actual manifest Parquet file size so bulk_processor can pass it to
        // ParquetObjectReader::with_file_size when reading the leaf manifest back.
        let manifest_file_size = write_result.size_in_bytes as i64;

        // Calculate aggregate stats from pending entries
        let mut record_count: i64 = self.pending_entries.iter().map(|e| e.record_count).sum();

        // Calculate manifest stats (entry counts by status)
        let mut added_files_count = 0i64;
        let mut existing_files_count = 0i64;
        let mut deletes_files_count = 0i64;
        let mut added_rows_count = 0i64;
        let mut existing_rows_count = 0i64;
        let mut delete_rows_count = 0i64;
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
                TrackingStatus::Existed => {
                    existing_files_count += 1;
                    existing_rows_count += entry.record_count;
                }
                TrackingStatus::Deleted => {
                    deletes_files_count += 1;
                    delete_rows_count += entry.record_count;
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

        let manifest_stats = Some(crate::content_tree::ManifestStats {
            added_files_count,
            existing_files_count,
            deletes_files_count,
            added_rows_count,
            existing_rows_count,
            delete_rows_count,
            min_sequence_number,
        });

        // Aggregate content_stats from all pending entries
        let content_stats = aggregate_content_stats(
            self.pending_entries
                .iter()
                .map(|e| e.content_stats.as_ref()),
        );

        let manifest_entry = ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location(manifest_path)
            .tracking(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(snapshot_id),
                // TODO: Manifest entries in root should have sequence_number and file_sequence_number
                // set to self.version so that leaf entries can inherit them when null.
                sequence_number: None,
                file_sequence_number: None,
                // Set to the starting row ID used for data entries in this leaf
                first_row_id: Some(starting_first_row_id),
                changes_dv: None,
            })
            .record_count(record_count)
            .file_size_in_bytes(manifest_file_size)
            .content_stats_opt(content_stats)
            .manifest_stats_opt(manifest_stats)
            .build();

        Ok((manifest_entry, next_row_id))
    }

    /// Builds a root ContentTreeNode instance (leaf is `None`).
    ///
    /// Assigns sequential `first_row_id` values to entries that don't already have one,
    /// starting from `starting_first_row_id`. Returns the built node and the next available
    /// row ID (for updating the high water mark).
    pub(crate) fn build(
        &mut self,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
        starting_first_row_id: i64,
    ) -> DeltaResult<(ContentTreeNode, i64)> {
        use crate::content_tree::metadata_entry_to_scalars;
        use crate::expressions::Scalar;

        // Serialize all in-memory DVs back to entries
        self.serialize_dvs_to_entries(snapshot_id)?;

        // Assign first_row_id values
        let next_row_id = {
            let cursor = self.assign_first_row_ids(starting_first_row_id);
            self.assign_first_row_ids_pre_built(engine, cursor)?
        };

        // Use cached schema with content_stats based on table schema
        let schema = self.get_schema()?;

        // Handle empty case early
        if self.pending_entries.is_empty() && self.pre_built_data.is_empty() {
            return Ok((
                ContentTreeNode {
                    table_root: self.table_root.clone(),
                    data: vec![],
                    version: self.version,
                    path_in_log: String::new(),
                    leaf: None,
                },
                next_row_id,
            ));
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

        Ok((
            ContentTreeNode {
                table_root: self.table_root.clone(),
                data,
                version: self.version,
                path_in_log: String::new(), // Will be set when written
                leaf: None,
            },
            next_row_id,
        ))
    }

    /// Builds a leaf ContentTreeNode instance with a generated UUID.
    ///
    /// Assigns sequential `first_row_id` values to data entries starting from
    /// `starting_first_row_id`. Returns the built node and the next available row ID.
    pub(crate) fn build_leaf(
        &mut self,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
        starting_first_row_id: i64,
    ) -> DeltaResult<(ContentTreeNode, i64)> {
        use crate::content_tree::metadata_entry_to_scalars;
        use crate::expressions::Scalar;

        // Serialize all in-memory DVs back to entries
        self.serialize_dvs_to_entries(snapshot_id)?;

        // Assign first_row_id values
        let next_row_id = {
            let cursor = self.assign_first_row_ids(starting_first_row_id);
            self.assign_first_row_ids_pre_built(engine, cursor)?
        };

        // Use cached schema with content_stats based on table schema
        let schema = self.get_schema()?;

        // Handle empty case early
        if self.pending_entries.is_empty() && self.pre_built_data.is_empty() {
            return Ok((
                ContentTreeNode {
                    table_root: self.table_root.clone(),
                    data: vec![],
                    version: self.version,
                    path_in_log: String::new(),
                    leaf: Some(uuid::Uuid::new_v4()),
                },
                next_row_id,
            ));
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

        Ok((
            ContentTreeNode {
                table_root: self.table_root.clone(),
                data,
                version: self.version,
                path_in_log: String::new(), // Will be set when written
                leaf: Some(uuid::Uuid::new_v4()),
            },
            next_row_id,
        ))
    }

    /// Build and evaluate a scan-row transformation expression.
    ///
    /// Transforms scan rows into ContentTreeNodeEntry schema, using `TrackingStatus::Existed`
    /// for all rows. `scan_row_input_schema` must include a `stats_parsed` field (Delta JSON format:
    /// `{numRecords, minValues, maxValues, nullCount, tightBounds}`), which is converted to AMT
    /// format for `content_stats` using [`build_content_stats_from_delta_stats_parsed`].
    ///
    /// If `scan_row_input_schema` has `_dv_location` (flat decoded DV columns appended by
    /// `add_from_existing_scan_rows`), `dvInfo` is projected from those columns with a
    /// nullability predicate so non-DV rows produce a null struct. Otherwise `dvInfo` is null.
    fn evaluate_scan_row_transform(
        &self,
        engine: &dyn Engine,
        engine_data: &dyn EngineData,
        scan_row_input_schema: &SchemaRef,
        version: Version,
        snapshot_id: i64,
    ) -> DeltaResult<(Box<dyn EngineData>, BatchAggregates)> {
        use crate::content_tree::stats;
        use crate::content_tree::{DataContentType, TrackingStatus, CONTENT_STATS_FIELD_NAME};
        use crate::expressions::{Expression, Predicate, Scalar};
        use crate::schema::DataType;

        let output_schema = self.get_schema()?;
        let stats_struct = stats::stats_schema(&self.table_schema)?;
        let version_i64 = version as i64;

        // Detect whether flat decoded DV columns are present in the input schema.
        let has_decoded_dv = scan_row_input_schema.field("_dv_location").is_some();

        // stats_parsed is always present (added by step 2 in add_from_existing_scan_rows, Delta format).
        // record_count reads numRecords directly; content_stats converts Delta→AMT via expressions.
        let record_count_expr = Expression::coalesce([
            Expression::column(["stats_parsed", DELTA_STATS_NUM_RECORDS]),
            Expression::literal(Scalar::Long(0)),
        ]);
        let content_stats_expr =
            build_content_stats_from_delta_stats_parsed(&self.table_schema, &stats_struct)?;

        // Build field expressions mapping scan_row input → ContentTreeNodeEntry output
        let mut field_exprs: Vec<Arc<Expression>> = Vec::new();
        for field in output_schema.fields() {
            let expr: Expression = match field.name().as_str() {
                "contentType" => Expression::literal(Scalar::Integer(DataContentType::Data as i32)),
                "location" => Expression::column(["path"]),
                "fileFormat" => Expression::literal(Scalar::String("parquet".into())),
                "tracking" => {
                    if !matches!(field.data_type(), DataType::Struct(_)) {
                        return Err(crate::Error::generic(
                            "tracking field should be a struct type",
                        ));
                    }
                    let snapshot_id_expr = Expression::literal(Scalar::Long(snapshot_id));
                    Expression::struct_from([
                        Expression::literal(Scalar::Integer(TrackingStatus::Existed as i32)),
                        snapshot_id_expr,
                        Expression::literal(Scalar::Long(version_i64)),
                        Expression::literal(Scalar::Long(version_i64)),
                        Expression::null_literal(DataType::LONG), // firstRowId
                        Expression::null_literal(DataType::BINARY), // changesDv
                    ])
                }
                "dvInfo" => {
                    if has_decoded_dv {
                        // Flat decoded DV columns: project into dvInfo struct.
                        // Nullability predicate: null struct for non-DV rows (_dv_location is null).
                        if !matches!(field.data_type(), DataType::Struct(_)) {
                            return Err(crate::Error::generic(
                                "dvInfo field should be a struct type",
                            ));
                        }
                        let nullability =
                            Expression::from_pred(Predicate::is_not_null(Expression::column([
                                "_dv_location",
                            ])));
                        Expression::struct_with_nullability_from(
                            [
                                Expression::column(["_dv_location"]),
                                Expression::column(["_dv_offset"]),
                                Expression::column(["_dv_size_in_bytes"]),
                                Expression::column(["_dv_cardinality"]),
                            ],
                            nullability,
                        )
                    } else {
                        Expression::null_literal(field.data_type().clone())
                    }
                }
                "partitionSpecId" => Expression::literal(Scalar::Long(0)),
                "sortOrderId" => Expression::null_literal(DataType::LONG),
                "recordCount" => record_count_expr.clone(),
                "fileSizeInBytes" => Expression::column(["size"]),
                CONTENT_STATS_FIELD_NAME => content_stats_expr.clone(),
                "manifestStats" => Expression::null_literal(field.data_type().clone()),
                "referencedFile" => Expression::null_literal(DataType::STRING),
                "manifestDv" => Expression::null_literal(DataType::BINARY),
                _ => Expression::null_literal(field.data_type().clone()),
            };
            field_exprs.push(Arc::new(expr));
        }

        let transform_expr = Expression::struct_from(field_exprs);

        let evaluator = engine.evaluation_handler().new_expression_evaluator(
            scan_row_input_schema.clone(),
            Arc::new(transform_expr),
            DataType::Struct(Box::new(output_schema.as_ref().clone())),
        )?;
        let transformed = evaluator.evaluate(engine_data)?;

        // Compute aggregates from the transformed output
        let mut agg_visitor = TransformedAggregateVisitor::default();
        agg_visitor.visit_rows_of(transformed.as_ref())?;

        let aggregates = BatchAggregates {
            added_file_count: 0,
            existing_file_count: engine_data.len() as i64,
            total_record_count: agg_visitor.total_record_count,
        };

        Ok((transformed, aggregates))
    }

    /// Assigns `first_row_id` to entries that need it, following the spec rules:
    /// - Preserve existing `first_row_id` for entries that already have one
    /// - Assign sequential IDs for Data and CombinedManifest entries without `first_row_id`
    ///
    /// `starting_row_id` is the first assignable row ID (high_water_mark + 1).
    /// Returns the next available row ID after all assignments (new high watermark + 1).
    fn assign_first_row_ids(&mut self, starting_row_id: i64) -> i64 {
        let mut cursor = starting_row_id;

        for entry in &mut self.pending_entries {
            let ti = &mut entry.tracking;

            // Deleted entries preserve their existing first_row_id but don't consume row IDs
            if ti.status == TrackingStatus::Deleted {
                continue;
            }

            match entry.content_type {
                DataContentType::Data => {
                    if let Some(existing) = ti.first_row_id {
                        // Preserve existing, advance cursor past its range
                        cursor = existing + entry.record_count;
                    } else {
                        ti.first_row_id = Some(cursor);
                        cursor += entry.record_count;
                    }
                }
                DataContentType::CombinedManifest => {
                    let row_increment = entry
                        .manifest_stats
                        .as_ref()
                        .map(|ms| ms.added_rows_count + ms.existing_rows_count)
                        .unwrap_or(0);
                    if let Some(existing) = ti.first_row_id {
                        cursor = existing + row_increment;
                    } else {
                        ti.first_row_id = Some(cursor);
                        cursor += row_increment;
                    }
                }
                // PositionDeletes, EqualityDeletes: no first_row_id assignment
                _ => {}
            }
        }
        cursor
    }

    /// Assigns `first_row_id` to pre-built EngineData batches that contain opaque columnar data.
    ///
    /// Uses a visitor to read `recordCount` per row, computes sequential `first_row_id` values,
    /// then uses `append_columns` + expression evaluator to replace `trackingInfo.firstRowId`.
    ///
    /// Returns the cursor (next available row ID) after all assignments.
    fn assign_first_row_ids_pre_built(
        &mut self,
        engine: &dyn crate::Engine,
        starting_row_id: i64,
    ) -> DeltaResult<i64> {
        use crate::expressions::{ArrayData, Expression};
        use crate::schema::{ArrayType, StructField, StructType};

        if self.pre_built_data.is_empty() {
            return Ok(starting_row_id);
        }

        let output_schema = self.get_schema()?;
        let mut cursor = starting_row_id;
        let mut new_pre_built = Vec::with_capacity(self.pre_built_data.len());

        for batch in self.pre_built_data.drain(..) {
            // Step 1: Visit to get record counts per row
            let mut record_counts_visitor = RecordCountVisitor::with_capacity(batch.len());
            record_counts_visitor.visit_rows_of(batch.as_ref())?;

            // Step 2: Compute first_row_id for each row
            let mut first_row_ids = Vec::with_capacity(record_counts_visitor.record_counts.len());
            for rc in &record_counts_visitor.record_counts {
                first_row_ids.push(cursor);
                cursor += rc;
            }

            // Step 3: Append _first_row_id column to the batch
            let append_schema = Arc::new(StructType::new_unchecked(vec![StructField::nullable(
                "_first_row_id",
                DataType::LONG,
            )]));
            let first_row_id_array =
                ArrayData::try_new(ArrayType::new(DataType::LONG, true), first_row_ids)?;
            let augmented = batch.append_columns(append_schema, vec![first_row_id_array])?;

            // Step 4: Build expression that replaces trackingInfo.firstRowId with _first_row_id
            // Input schema = output_schema + _first_row_id
            let mut input_fields: Vec<StructField> =
                output_schema.fields().cloned().collect::<Vec<_>>();
            input_fields.push(StructField::nullable("_first_row_id", DataType::LONG));
            let input_schema = Arc::new(StructType::new_unchecked(input_fields));

            let mut field_exprs: Vec<Arc<Expression>> = Vec::new();
            for field in output_schema.fields() {
                let expr = if field.name() == "trackingInfo" {
                    // Rebuild trackingInfo struct, replacing firstRowId with the appended column
                    Expression::struct_from([
                        Expression::column(["trackingInfo", "status"]),
                        Expression::column(["trackingInfo", "snapshotId"]),
                        Expression::column(["trackingInfo", "sequenceNumber"]),
                        Expression::column(["trackingInfo", "fileSequenceNumber"]),
                        Expression::column(["_first_row_id"]),
                        Expression::column(["trackingInfo", "changesDv"]),
                    ])
                } else {
                    Expression::column([field.name().as_str()])
                };
                field_exprs.push(Arc::new(expr));
            }

            let transform_expr = Expression::struct_from(field_exprs);
            let evaluator = engine.evaluation_handler().new_expression_evaluator(
                input_schema,
                Arc::new(transform_expr),
                DataType::Struct(Box::new(output_schema.as_ref().clone())),
            )?;
            let result = evaluator.evaluate(augmented.as_ref())?;
            new_pre_built.push(result);
        }

        self.pre_built_data = new_pre_built;
        Ok(cursor)
    }

    /// Adds file metadata from existing scan rows to the leaf manifest.
    ///
    /// Unlike `add_from_engine_data_write` (for new files), this method handles rows from
    /// a scan over an existing Delta table, writing them as `TrackingStatus::Existed` entries.
    ///
    /// The input data must include a `stats_parsed` column (added by `include_stats_columns()` in
    /// the scan). All rows are processed via a single expression-evaluator path:
    /// 1. DV columns are decoded (base85 UUID → relative path, sizes widened to LONG, +8 bytes
    ///    for Iceberg framing) by `DecodedDvVisitor`.
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
        use crate::expressions::{ArrayData, Expression};
        use crate::schema::{ArrayType, StructField, StructType};

        if engine_data.is_empty() {
            return Ok(());
        }

        // Step 1: Detect + decode DV columns in one pass from the original engine_data.
        // (Done first so we can use the original data's nullable DV fields directly.)
        let mut dv_visitor = DecodedDvVisitor::with_capacity(engine_data.len());
        dv_visitor.visit_rows_of(engine_data)?;

        // Step 2: Produce {path, size, stats_parsed} via coalesce(stats_parsed, parse_json(stats)).
        // The input is always expected to have a `stats_parsed` column (added by
        // `include_stats_columns()` in the scan). For rows sourced from JSON commits,
        // `stats_parsed` will be null and the coalesce falls back to parsing the `stats` JSON.
        // This mirrors the identical pattern in checkpoint/stats_transform.rs.
        let delta_stats_schema = Arc::new(build_delta_stats_schema(&self.table_schema));
        let stats_parsed_type = DataType::Struct(Box::new(delta_stats_schema.as_ref().clone()));
        let step2_input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable("stats", DataType::STRING),
            StructField::nullable("stats_parsed", stats_parsed_type.clone()),
        ]));
        let stats_augmented_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable("stats_parsed", stats_parsed_type),
        ]));
        let parse_stats_expr = Expression::struct_from([
            Expression::column(["path"]),
            Expression::column(["size"]),
            Expression::coalesce([
                Expression::column(["stats_parsed"]),
                Expression::parse_json(Expression::column(["stats"]), delta_stats_schema),
            ]),
        ]);
        let stats_evaluator = engine.evaluation_handler().new_expression_evaluator(
            step2_input_schema,
            Arc::new(parse_stats_expr),
            DataType::Struct(Box::new(stats_augmented_schema.as_ref().clone())),
        )?;
        let stats_engine_data = stats_evaluator.evaluate(engine_data)?;

        // Step 3: Append 4 flat _dv_* columns if any DV rows are present; build input schema.
        let (transformed_data, input_schema) = if dv_visitor.has_any_dv() {
            let augmented = stats_engine_data.append_columns(
                DV_DECODED_FLAT_SCHEMA.clone(),
                vec![
                    ArrayData::try_new(
                        ArrayType::new(DataType::STRING, true),
                        dv_visitor.decoded_paths,
                    )?,
                    ArrayData::try_new(
                        ArrayType::new(DataType::LONG, true),
                        dv_visitor.decoded_offsets,
                    )?,
                    ArrayData::try_new(
                        ArrayType::new(DataType::LONG, true),
                        dv_visitor.decoded_sizes,
                    )?,
                    ArrayData::try_new(
                        ArrayType::new(DataType::LONG, true),
                        dv_visitor.decoded_cardinalities,
                    )?,
                ],
            )?;
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
            existing_file_count: filtered.len() as i64,
            total_record_count: agg_visitor.total_record_count,
        };
        self.pre_built_data.push(filtered);
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
        use crate::schema::column_name;
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

/// Visitor that reads per-row record counts from pre-built EngineData batches.
/// Used by `assign_first_row_ids_pre_built` to compute sequential first_row_id values.
struct RecordCountVisitor {
    record_counts: Vec<i64>,
}

impl RecordCountVisitor {
    fn with_capacity(cap: usize) -> Self {
        Self {
            record_counts: Vec::with_capacity(cap),
        }
    }
}

impl RowVisitor for RecordCountVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        use crate::schema::column_name;
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
            self.record_counts.push(record_count);
        }
        Ok(())
    }
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
    use crate::schema::{StructField, StructType};
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
/// format) to `content_stats: {col: {value_count, [null_value_count], lower_bound, upper_bound,
/// exact_bounds}}` (AMT format) using struct expressions.
///
/// Column names used for field access come from `table_schema` field names (physical names when
/// column mapping is enabled, which must match the JSON keys in the Delta stats).
fn build_content_stats_from_delta_stats_parsed(
    table_schema: &crate::schema::StructType,
    amt_schema: &crate::schema::StructType,
) -> DeltaResult<crate::expressions::Expression> {
    use crate::expressions::{Expression, Scalar};
    let col_exprs: Vec<Arc<Expression>> = table_schema
        .fields()
        .zip(amt_schema.fields())
        .map(|(table_field, amt_field)| {
            let col_name = table_field.name();
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
                        crate::content_tree::VALUE_COUNT => {
                            Expression::column(["stats_parsed", DELTA_STATS_NUM_RECORDS])
                        }
                        crate::content_tree::NULL_VALUE_COUNT => Expression::column([
                            "stats_parsed",
                            DELTA_STATS_NULL_COUNT,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::NAN_VALUE_COUNT => {
                            Expression::null_literal(DataType::LONG)
                        }
                        crate::content_tree::AVG_VALUE_SIZE => {
                            Expression::null_literal(DataType::INTEGER)
                        }
                        crate::content_tree::MAX_VALUE_SIZE => {
                            Expression::null_literal(DataType::INTEGER)
                        }
                        crate::content_tree::LOWER_BOUND => Expression::column([
                            "stats_parsed",
                            DELTA_STATS_MIN_VALUES,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::UPPER_BOUND => Expression::column([
                            "stats_parsed",
                            DELTA_STATS_MAX_VALUES,
                            col_name.as_str(),
                        ]),
                        crate::content_tree::EXACT_BOUNDS => Expression::coalesce([
                            Expression::column(["stats_parsed", DELTA_STATS_TIGHT_BOUNDS]),
                            Expression::literal(Scalar::Boolean(true)),
                        ]),
                        _ => Expression::null_literal(f.data_type().clone()),
                    })
                })
                .collect();
            Ok(Arc::new(Expression::struct_from(field_exprs)))
        })
        .collect::<DeltaResult<Vec<_>>>()?;
    Ok(Expression::struct_from(col_exprs))
}

/// Schema for the 4 flat DV columns appended to scan-row data before the final transform.
/// These columns carry decoded DV info (path decoded, sizes widened to LONG, +8 for Iceberg framing).
static DV_DECODED_FLAT_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    use crate::schema::{StructField, StructType};
    Arc::new(StructType::new_unchecked(vec![
        StructField::nullable("_dv_location", DataType::STRING),
        StructField::nullable("_dv_offset", DataType::LONG),
        StructField::nullable("_dv_size_in_bytes", DataType::LONG),
        StructField::nullable("_dv_cardinality", DataType::LONG),
    ]))
});

/// Visits all scan-row rows in one pass, accumulating decoded DV columns.
///
/// For rows with a DV: decodes path (base85 UUID → relative path), widens offset/sizeInBytes
/// to LONG, adds 8 to sizeInBytes (Delta → Iceberg framing), stores cardinality.
/// For rows without a DV: pushes Null scalars for all 4 columns.
struct DecodedDvVisitor {
    decoded_paths: Vec<crate::expressions::Scalar>,
    decoded_offsets: Vec<crate::expressions::Scalar>,
    decoded_sizes: Vec<crate::expressions::Scalar>,
    decoded_cardinalities: Vec<crate::expressions::Scalar>,
}

impl DecodedDvVisitor {
    fn with_capacity(n: usize) -> Self {
        Self {
            decoded_paths: Vec::with_capacity(n),
            decoded_offsets: Vec::with_capacity(n),
            decoded_sizes: Vec::with_capacity(n),
            decoded_cardinalities: Vec::with_capacity(n),
        }
    }

    fn has_any_dv(&self) -> bool {
        self.decoded_paths.iter().any(|s| !s.is_null())
    }
}

impl RowVisitor for DecodedDvVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        use crate::schema::column_name;
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
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
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        use crate::actions::deletion_vector::{
            DeletionVectorDescriptor, DeletionVectorStorageType,
        };
        use crate::expressions::Scalar;

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
                let dv_info = extract_deletion_vector_content(&dv)?;
                self.decoded_paths.push(Scalar::String(dv_info.location));
                self.decoded_offsets.push(Scalar::Long(dv_info.offset));
                self.decoded_sizes.push(Scalar::Long(dv_info.size_in_bytes));
                self.decoded_cardinalities
                    .push(Scalar::Long(dv_info.cardinality));
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

/// Visitor that extracts Add-like data from scan row schema.
///
/// The scan row schema has a different structure than the log Add action schema:
/// - path (direct, not nested under "add")
/// - size (direct)
/// - modificationTime (direct)
/// - stats (direct)
/// - fileConstantValues.partitionValues (nested)
/// - deletionVector (nested)
///
/// This visitor extracts these fields and constructs Add structs.
#[derive(Default)]
struct ScanRowToAddVisitor {
    pub adds: Vec<Add>,
    /// Selection vector controlling which rows to process. Empty means all rows selected.
    selection_vector: Vec<bool>,
    /// Running row offset across multiple `visit()` calls (for multi-batch inputs).
    row_offset: usize,
}

impl RowVisitor for ScanRowToAddVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        use crate::schema::{column_name, MapType};
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            let names = vec![
                column_name!("path"),
                column_name!("size"),
                column_name!("modificationTime"),
                column_name!("stats"),
                column_name!("deletionVector.storageType"),
                column_name!("deletionVector.pathOrInlineDv"),
                column_name!("deletionVector.offset"),
                column_name!("deletionVector.sizeInBytes"),
                column_name!("deletionVector.cardinality"),
                column_name!("fileConstantValues.partitionValues"),
                column_name!("fileConstantValues.dataManifestPath"),
                column_name!("fileConstantValues.dataManifestPosition"),
            ];
            let types = vec![
                DataType::STRING,
                DataType::LONG,
                DataType::LONG,
                DataType::STRING,
                DataType::STRING,  // deletionVector.storageType
                DataType::STRING,  // deletionVector.pathOrInlineDv
                DataType::INTEGER, // deletionVector.offset
                DataType::INTEGER, // deletionVector.sizeInBytes
                DataType::LONG,    // deletionVector.cardinality
                DataType::Map(Box::new(MapType::new(
                    DataType::STRING,
                    DataType::STRING,
                    true,
                ))),
                DataType::STRING,
                DataType::LONG,
            ];
            (names, types).into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        use crate::actions::deletion_vector::{
            DeletionVectorDescriptor, DeletionVectorStorageType,
        };

        for i in 0..row_count {
            let global_i = self.row_offset + i;
            if global_i < self.selection_vector.len() && !self.selection_vector[global_i] {
                continue;
            }

            if let Some(path) = getters[0].get_opt(i, "scanRow.path")? {
                let size: i64 = getters[1].get(i, "scanRow.size")?;
                let modification_time: i64 = getters[2].get(i, "scanRow.modificationTime")?;
                let stats: Option<String> = getters[3].get_opt(i, "scanRow.stats")?;

                let storage_type_str_opt: Option<String> =
                    getters[4].get_opt(i, "scanRow.deletionVector.storageType")?;
                let deletion_vector = if let Some(storage_type_str) = storage_type_str_opt {
                    let storage_type: DeletionVectorStorageType = storage_type_str.parse()?;
                    let path_or_inline_dv: String =
                        getters[5].get(i, "scanRow.deletionVector.pathOrInlineDv")?;
                    let offset: Option<i32> =
                        getters[6].get_opt(i, "scanRow.deletionVector.offset")?;
                    let size_in_bytes: i32 =
                        getters[7].get(i, "scanRow.deletionVector.sizeInBytes")?;
                    let cardinality: i64 =
                        getters[8].get(i, "scanRow.deletionVector.cardinality")?;

                    Some(DeletionVectorDescriptor {
                        storage_type,
                        path_or_inline_dv,
                        offset,
                        size_in_bytes,
                        cardinality,
                    })
                } else {
                    None
                };

                let partition_values: HashMap<String, String> = getters[9]
                    .get_opt(i, "scanRow.fileConstantValues.partitionValues")?
                    .unwrap_or_default();

                let data_manifest_path: Option<String> =
                    getters[10].get_opt(i, "scanRow.fileConstantValues.dataManifestPath")?;
                let data_manifest_position: Option<i64> =
                    getters[11].get_opt(i, "scanRow.fileConstantValues.dataManifestPosition")?;

                let add = Add {
                    path,
                    partition_values,
                    size,
                    modification_time,
                    data_change: true,
                    stats,
                    tags: None,
                    deletion_vector,
                    base_row_id: None,
                    default_row_commit_version: None,
                    clustering_provider: None,
                    data_manifest_path,
                    data_manifest_position,
                };
                self.adds.push(add);
            }
        }
        self.row_offset += row_count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::deletion_vector::DeletionVectorStorageType;
    use crate::content_tree::ContentTreeNode;
    use serde_json::json;

    /// Helper: builds a root manifest, writes it to disk, and reads it back.
    fn build_and_read_root(
        builder: &mut ContentTreeNodeBuilder,
        engine: &dyn crate::Engine,
        snapshot_id: i64,
    ) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
        let (root_metadata, _) = builder.build(engine, snapshot_id, 0)?;
        let table_root = root_metadata.table_root.clone();
        let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
            .write(engine)?
            .location;
        let root_path = crate::content_tree::absolute_to_relative_path(&root_url, &table_root)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &root_url,
            root_path,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let root =
            ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
        root.entries()
    }

    // TODO: Add tests for all tracking columns (status, snapshot_id, sequence_number,
    // file_sequence_number, first_row_id, changes_dv) to verify they are correctly set during
    // build operations for ADDED, DELETED, and EXISTED manifests.

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
                    .with_tracking(1, 1, 1)
                    .record_count(record_count)
                    .file_size_in_bytes(1024)
                    .build(),
            );
        }

        // Build metadata and verify record counts are preserved through roundtrip
        let engine = crate::engine::sync::SyncEngine::new();
        let (metadata, _) = builder.build(&engine, 1, 0)?;
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
            data_manifest_path: None,
            data_manifest_position: None,
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
            data_manifest_path: None,
            data_manifest_position: None,
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
        use crate::content_tree::stats::delta_json_stats_to_content_stats;
        use crate::engine::sync::SyncEngine;
        use crate::expressions::Scalar;
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructField, StructType};
        use tempfile::tempdir;

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
            .with_tracking(1, 1, 1)
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
            .with_tracking(1, 1, 1)
            .record_count(150)
            .file_size_in_bytes(2048)
            .content_stats_opt(content_stats_2)
            .build();

        builder.add_entry(entry1);
        builder.add_entry(entry2);

        // Write the leaf manifest
        let (leaf_manifest_entry, _) = builder.write_leaf(&engine, 1, 0)?;

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
        use crate::engine::sync::SyncEngine;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a builder with empty schema
        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Create entries without content_stats
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("data/part-00000.parquet")
            .with_tracking(1, 1, 1)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        builder.add_entry(entry);

        // Write the leaf manifest
        let (leaf_manifest_entry, _) = builder.write_leaf(&engine, 1, 0)?;

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

        let dv_info = extract_deletion_vector_content(&dv)?;

        // Should have location set to the relative path
        assert_eq!(
            dv_info.location,
            "ab/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
        );

        // Should have offset and size (+8 for size field and CRC)
        assert_eq!(dv_info.offset, 4);
        assert_eq!(dv_info.size_in_bytes, 48); // 40 + 8
        assert_eq!(dv_info.cardinality, 6);

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

        let dv_info = extract_deletion_vector_content(&dv)?;

        // Should have location set to the relative path (no prefix directory)
        assert_eq!(
            dv_info.location,
            "deletion_vector_61d16c75-6994-46b7-a15b-8b538852e50e.bin"
        );

        // Should have offset and size (+8 for size field and CRC)
        assert_eq!(dv_info.offset, 1);
        assert_eq!(dv_info.size_in_bytes, 44); // 36 + 8
        assert_eq!(dv_info.cardinality, 2);

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

        let dv_info = extract_deletion_vector_content(&dv)?;

        // Should preserve the absolute path as-is
        assert_eq!(
            dv_info.location,
            "s3://another-bucket/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin"
        );

        // Should have offset and size (+8)
        assert_eq!(dv_info.offset, 4);
        assert_eq!(dv_info.size_in_bytes, 48); // 40 + 8
        assert_eq!(dv_info.cardinality, 6);

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
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        // Write the leaf
        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create a root with the leaf, then delete entry at index 5
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.insert(5u64);
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices, true)?;

        // Step 3: Build, write, and read back the root to verify manifest DV is stored inline
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        // Should have: 1 CombinedManifest (DV is now inline on this entry)
        assert_eq!(root_entries.len(), 1);

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        assert_eq!(data_manifest.location.as_ref(), Some(&leaf_path));

        // Verify the manifest_dv field contains the deleted index
        let manifest_dv_bytes = data_manifest
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        assert!(
            manifest_dv_bytes.len() >= 4,
            "Should have magic number prefix"
        );
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(5));
        assert_eq!(treemap.len(), 1);

        // Step 4: Read the leaf and apply manifest DV to verify filtering
        let leaf_url = table_root.join(&leaf_path)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &leaf_url,
            leaf_path.clone(),
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
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Create root and delete multiple entries
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.extend([2u64, 5, 7]);
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices, true)?;

        // Build, write, and read back the root to verify
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;
        assert_eq!(root_entries.len(), 1); // CombinedManifest (DV is inline)

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .unwrap();

        // Verify all deleted indices in manifest_dv field
        let manifest_dv_bytes = data_manifest.manifest_dv.as_ref().unwrap();
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(2));
        assert!(treemap.contains(5));
        assert!(treemap.contains(7));
        assert_eq!(treemap.len(), 3);

        // Apply manifest DV and verify filtering
        let leaf_url = table_root.join(&leaf_path)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &leaf_url,
            leaf_path.clone(),
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
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 3 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..3 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Create root and delete all 3 entries
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        // Deleting all entries should automatically mark the manifest as deleted
        let mut indices = RoaringTreemap::new();
        indices.extend([0u64, 1, 2]);
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices, true)?;

        // Build, write, and read back the root to verify the manifest is marked as deleted
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let leaf_manifest = root_entries
            .iter()
            .find(|e| {
                e.content_type == DataContentType::CombinedManifest
                    && e.location.as_ref() == Some(&leaf_path)
            })
            .expect("Leaf manifest should exist");

        assert_eq!(leaf_manifest.tracking.status, TrackingStatus::Deleted);

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_index_out_of_bounds() -> Result<(), Box<dyn std::error::Error>> {
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf with 10 entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("data/part-{:05}.parquet", i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Try to delete index 10 (out of bounds, valid indices are 0-9 for 10 entries)
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);

        let mut indices = RoaringTreemap::new();
        indices.insert(10u64);
        let result = root_builder.delete_multiple_from_leaf(&leaf_path, &indices, true);
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
        let result = root_builder.delete_multiple_from_leaf("nonexistent.parquet", &indices, true);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Manifest cache not found"));

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_with_relative_path() -> Result<(), Box<dyn std::error::Error>> {
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

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
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();
        // leaf_path is now already relative
        let relative_path = &leaf_path;

        // Create root and delete using relative path
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry);
        let mut indices = RoaringTreemap::new();
        indices.insert(3u64);
        root_builder.delete_multiple_from_leaf(relative_path, &indices, true)?;

        // Build, write, and read back the root to verify manifest DV is stored inline
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let data_manifest = root_entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .unwrap();

        assert_eq!(data_manifest.location.as_ref(), Some(&leaf_path));

        // Verify the deletion was recorded in manifest_dv
        let manifest_dv_bytes = data_manifest.manifest_dv.as_ref().unwrap();
        let treemap = RoaringTreemap::deserialize_from(&manifest_dv_bytes[4..])?;
        assert!(treemap.contains(3));

        Ok(())
    }

    #[test]
    fn test_delete_from_leaf_with_existing_deleted_entries(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a leaf manifest that already has some deleted entries
        // This simulates a manifest that has been updated over time
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());

        // Create a manifest entry with manifest_stats showing:
        // - 2 added files (indices 0, 1)
        // - 1 existing file (index 2)
        // - 2 deleted files (indices 3, 4)
        // Total: 5 entries, but only 3 are active (non-deleted)
        let manifest_entry = ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location("leaf-manifest.parquet")
            .with_tracking(1, 1, 1)
            .record_count(5) // Total entries in the leaf
            .file_size_in_bytes(2048)
            .manifest_stats(ManifestStats {
                added_files_count: 2,
                existing_files_count: 1,
                deletes_files_count: 2, // 2 entries are already deleted
                added_rows_count: 200,
                existing_rows_count: 100,
                delete_rows_count: 200,
                min_sequence_number: 1,
            })
            .build();

        let leaf_path = manifest_entry.location.as_ref().unwrap().clone();
        root_builder.add_entry(manifest_entry);

        // Delete all 3 active entries (indices 0, 1, 2)
        // With the OLD logic: cardinality (3) != total_entry_count (5), so manifest would NOT be marked deleted
        // With the NEW logic: cardinality (3) == active_entry_count (3), so manifest IS marked deleted
        let mut indices = RoaringTreemap::new();
        indices.extend([0u64, 1, 2]);
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices, true)?;

        // Build, write, and read back the root to verify the manifest is marked as deleted
        let root_entries = build_and_read_root(&mut root_builder, &engine, 1)?;

        let leaf_manifest = root_entries
            .iter()
            .find(|e| {
                e.content_type == DataContentType::CombinedManifest
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
            .manifest_dv
            .as_ref()
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
    fn test_tracking_changes_dv_clearing() -> Result<(), Box<dyn std::error::Error>> {
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 10 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..10 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("{}data/part-{:05}.parquet", table_root, i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create root and delete entries 2 and 5 (first commit)
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry.clone());
        let mut indices_v1 = RoaringTreemap::new();
        indices_v1.extend([2u64, 5]);
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices_v1, true)?;

        // Step 3: Build, write, and read back the root to verify changes_dv from first commit
        let entries_v1 = build_and_read_root(&mut root_builder, &engine, 1)?;
        let manifest_v1 = entries_v1
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        // Verify manifest_dv contains both deletions (2 and 5)
        let manifest_dv_v1 = manifest_v1
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        let cumulative_v1 = RoaringTreemap::deserialize_from(&manifest_dv_v1[4..])?;
        assert!(cumulative_v1.contains(2));
        assert!(cumulative_v1.contains(5));
        assert_eq!(cumulative_v1.len(), 2);

        // Verify changes_dv contains both deletions from this commit (2 and 5)
        let changes_dv_v1 = manifest_v1
            .tracking
            .changes_dv
            .as_ref()
            .expect("changes_dv should exist");
        let delta_v1 = RoaringTreemap::deserialize_from(&changes_dv_v1[4..])?;
        assert!(delta_v1.contains(2));
        assert!(delta_v1.contains(5));
        assert_eq!(delta_v1.len(), 2);

        // Step 4: Start a new commit (v2) by loading v1 entries
        // Note: changes_dv is automatically cleared when entries are added
        let mut root_builder_v2 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 2, test_table_schema());
        for entry in entries_v1 {
            root_builder_v2.add_entry(entry);
        }

        // Step 5: Add new deletions (entries 3 and 7) in the second commit
        let mut indices_v2 = RoaringTreemap::new();
        indices_v2.extend([3u64, 7]);
        root_builder_v2.delete_multiple_from_leaf(&leaf_path, &indices_v2, true)?;

        // Build, write, and read back the root to verify changes_dv only contains NEW deletions
        let entries_v2 = build_and_read_root(&mut root_builder_v2, &engine, 2)?;
        let manifest_v2 = entries_v2
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        // Verify manifest_dv contains ALL deletions (2, 3, 5, 7)
        let manifest_dv_v2 = manifest_v2
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        let cumulative_v2 = RoaringTreemap::deserialize_from(&manifest_dv_v2[4..])?;
        assert!(cumulative_v2.contains(2));
        assert!(cumulative_v2.contains(3));
        assert!(cumulative_v2.contains(5));
        assert!(cumulative_v2.contains(7));
        assert_eq!(cumulative_v2.len(), 4);

        // Verify changes_dv ONLY contains NEW deletions from v2 (3 and 7)
        let changes_dv_v2 = manifest_v2
            .tracking
            .changes_dv
            .as_ref()
            .expect("changes_dv should exist");
        let delta_v2 = RoaringTreemap::deserialize_from(&changes_dv_v2[4..])?;
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
        // Note: changes_dv is automatically cleared when entries are added
        let mut root_builder_v3 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 3, test_table_schema());
        for entry in entries_v2 {
            root_builder_v3.add_entry(entry);
        }

        // Step 9: Delete one additional record (entry 8) in the third commit
        let mut indices_v3 = RoaringTreemap::new();
        indices_v3.insert(8u64);
        root_builder_v3.delete_multiple_from_leaf(&leaf_path, &indices_v3, true)?;

        // Build, write, and read back the root to verify changes_dv only contains NEW deletion
        let entries_v3 = build_and_read_root(&mut root_builder_v3, &engine, 3)?;
        let manifest_v3 = entries_v3
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        // Verify manifest_dv contains ALL deletions (2, 3, 5, 7, 8)
        let manifest_dv_v3 = manifest_v3
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        let cumulative_v3 = RoaringTreemap::deserialize_from(&manifest_dv_v3[4..])?;
        assert!(cumulative_v3.contains(2));
        assert!(cumulative_v3.contains(3));
        assert!(cumulative_v3.contains(5));
        assert!(cumulative_v3.contains(7));
        assert!(cumulative_v3.contains(8));
        assert_eq!(cumulative_v3.len(), 5);

        // Verify changes_dv ONLY contains NEW deletion from v3 (8)
        let changes_dv_v3 = manifest_v3
            .tracking
            .changes_dv
            .as_ref()
            .expect("changes_dv should exist");
        let delta_v3 = RoaringTreemap::deserialize_from(&changes_dv_v3[4..])?;
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
        // Note: changes_dv is automatically cleared when entries are added
        let mut root_builder_v4 =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 4, test_table_schema());
        for entry in entries_v3 {
            root_builder_v4.add_entry(entry);
        }

        // Step 12: Make an unrelated change - add a new data entry (no deletions)
        let new_data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location(format!("{}data/part-{:05}.parquet", table_root, 100))
            .with_tracking(4, 4, 4)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        root_builder_v4.add_entry(new_data_entry);

        // Build, write, and read back the root to verify changes_dv is None (no deletions)
        let entries_v4 = build_and_read_root(&mut root_builder_v4, &engine, 4)?;
        let manifest_v4 = entries_v4
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        // Verify manifest_dv still contains all previous deletions (2, 3, 5, 7, 8)
        let manifest_dv_v4 = manifest_v4
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        let cumulative_v4 = RoaringTreemap::deserialize_from(&manifest_dv_v4[4..])?;
        assert_eq!(
            cumulative_v4.len(),
            5,
            "manifest_dv should still have 5 deletions"
        );

        // Verify changes_dv is None since no deletions were made in v4
        assert!(
            manifest_v4.tracking.changes_dv.is_none(),
            "changes_dv should be None when no deletions are made"
        );

        Ok(())
    }

    #[test]
    fn test_leaf_reorganization_does_not_set_changes_dv() -> Result<(), Box<dyn std::error::Error>>
    {
        use crate::engine::sync::SyncEngine;
        use roaring::RoaringTreemap;
        use tempfile::tempdir;

        let engine = SyncEngine::new();
        let temp_dir = tempdir()?;
        let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

        // Step 1: Create a leaf with 5 data entries
        let mut leaf_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        for i in 0..5 {
            let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                .location(format!("{}data/part-{:05}.parquet", table_root, i))
                .with_tracking(1, 1, 1)
                .record_count(100)
                .file_size_in_bytes(1024)
                .build();
            leaf_builder.add_entry(data_entry);
        }

        let (leaf_manifest_entry, _) = leaf_builder.write_leaf(&engine, 1, 0)?;
        let leaf_path = leaf_manifest_entry.location.as_ref().unwrap().clone();

        // Step 2: Create root and simulate leaf reorganization by calling delete_multiple_from_leaf
        // with set_changes_dv=false (simulating moving entries to a different leaf)
        let mut root_builder =
            ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
        root_builder.add_entry(leaf_manifest_entry.clone());

        let mut indices = RoaringTreemap::new();
        indices.insert(2);
        indices.insert(3);

        // Call delete_multiple_from_leaf with set_changes_dv=false to simulate leaf reorganization
        root_builder.delete_multiple_from_leaf(&leaf_path, &indices, false)?;

        // Step 3: Build, write, and read back the root to verify changes_dv is NOT set for leaf reorganization
        let entries = build_and_read_root(&mut root_builder, &engine, 1)?;
        let manifest = entries
            .iter()
            .find(|e| matches!(e.content_type, DataContentType::CombinedManifest))
            .expect("CombinedManifest should exist");

        // Verify manifest_dv contains the deletions (for internal tracking)
        let manifest_dv = manifest
            .manifest_dv
            .as_ref()
            .expect("manifest_dv should exist");
        let cumulative = RoaringTreemap::deserialize_from(&manifest_dv[4..])?;
        assert!(cumulative.contains(2));
        assert!(cumulative.contains(3));
        assert_eq!(cumulative.len(), 2);

        // Verify changes_dv is NOT set since this was leaf reorganization, not actual deletion
        assert!(
            manifest.tracking.changes_dv.is_none(),
            "changes_dv should NOT be set for leaf reorganization (set_changes_dv=false)"
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
            .dv_info(DvInfo {
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

    // Note: Deletion vector extraction from scan rows is tested through integration tests
    // since creating mock scan row data with the complex nested schema structure is difficult.
    // The extraction logic is verified through:
    // - metadata tests (test_dv_with_later_sequence_number_included, etc.)
    // - Full table scans with the backfill tool

    // Disabled complex unit test - see note above

    // --- Tests for assign_first_row_ids ---

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
                changes_dv: None,
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
        ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location(format!("manifest-{}-{}.parquet", added_rows, existing_rows))
            .tracking(TrackingInfo {
                status,
                snapshot_id: Some(1),
                sequence_number: None,
                file_sequence_number: None,
                first_row_id,
                changes_dv: None,
            })
            .record_count(added_rows + existing_rows)
            .file_size_in_bytes(2048)
            .manifest_stats_opt(Some(ManifestStats {
                added_files_count: 1,
                existing_files_count: 1,
                deletes_files_count: 0,
                added_rows_count: added_rows,
                existing_rows_count: existing_rows,
                delete_rows_count: 0,
                min_sequence_number: 1,
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

        let next = builder.assign_first_row_ids(0);

        assert_eq!(next, 350);
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

        let next = builder.assign_first_row_ids(0);

        assert_eq!(next, 400);
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
            TrackingStatus::Existed,
            Some(0),
        ));
        builder
            .pending_entries
            .push(make_manifest_entry(50, 50, TrackingStatus::Added, None));

        let next = builder.assign_first_row_ids(0);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(300));
        assert_eq!(next, 400);
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

        let next = builder.assign_first_row_ids(0);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(999));
        assert_eq!(builder.pending_entries[2].tracking.first_row_id, Some(100));
        assert_eq!(next, 150);
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

        let next = builder.assign_first_row_ids(0);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(0));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(100));
        assert_eq!(builder.pending_entries[2].tracking.first_row_id, Some(300));
        assert_eq!(next, 375);
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

        let next = builder.assign_first_row_ids(501);

        assert_eq!(builder.pending_entries[0].tracking.first_row_id, Some(501));
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(601));
        assert_eq!(next, 801);
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
        let next = builder.assign_first_row_ids(manifest_first_row_id);

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

        // next_row_id == manifest_first_row_id + total_records
        assert_eq!(next, manifest_first_row_id + cumulative);
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
            .push(make_data_entry(100, TrackingStatus::Existed, None));
        // Added entry after it
        builder
            .pending_entries
            .push(make_data_entry(50, TrackingStatus::Added, None));

        let next = builder.assign_first_row_ids(0);

        // The Existed entry should be assigned first_row_id=0
        assert_eq!(
            builder.pending_entries[0].tracking.first_row_id,
            Some(0),
            "Existed entry with null first_row_id should be assigned"
        );
        // The Added entry should follow sequentially
        assert_eq!(builder.pending_entries[1].tracking.first_row_id, Some(100));
        assert_eq!(next, 150);
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
                    changes_dv: None,
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
                    changes_dv: None,
                })
                .record_count(25)
                .file_size_in_bytes(256)
                .build(),
        );

        // Another data entry after the deletes
        builder
            .pending_entries
            .push(make_data_entry(75, TrackingStatus::Added, None));

        let next = builder.assign_first_row_ids(0);

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
        assert_eq!(next, 175);
    }
}
