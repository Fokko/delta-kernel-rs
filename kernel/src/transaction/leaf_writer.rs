use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use roaring::RoaringTreemap;
use url::Url;

use crate::content_tree::builder::ContentTreeNodeBuilder;
use crate::content_tree::ContentTreeNodeEntry;
use crate::engine_data::{GetData, TypedGetData};
use crate::expressions::ColumnName;
use crate::row_tracking::CursorRowIdAllocator;
use crate::schema::{DataType, StructType};
use crate::{DeltaResult, Engine, EngineData, FilteredEngineData, RowVisitor, SchemaRef, Version};

/// Output from finishing a leaf writer.
/// Contains metadata needed to incorporate the leaf into a transaction.
///
/// This is an opaque type - use it by passing to
/// [`crate::transaction::ManifestCommitState::add_leaf`].
#[derive(Debug)]
pub struct LeafNodeWriterResult {
    /// Map of manifest paths (relative to table root) to roaring bitmaps indicating which entries
    /// are deleted. These are manifest deletion vectors that need to be applied to existing
    /// manifests.
    pub(crate) manifest_dvs: HashMap<String, RoaringTreemap>,

    /// Written data file manifest (if any data files were added).
    pub(crate) data_file_manifest_written: Option<ContentTreeNodeEntry>,

    /// Root entries to remove (file paths that should be removed from root)
    pub(crate) root_entries_to_remove: HashSet<String>,

    /// Root DV entries to remove (DV paths that should be removed from root DV manifest)
    pub(crate) root_dv_entries_to_remove: HashSet<String>,

    /// Final row ID cursor position after all assignments in this leaf.
    pub(crate) final_cursor: i64,
}

/// Builder for creating leaf manifests.
///
/// This wraps a ContentTreeNodeBuilder and adds tracking for:
/// - Manifest deletion vectors (for marking entries in other manifests as deleted)
/// - Root entries to remove
/// - Deletion vectors to write
pub struct LeafNodeWriter {
    /// Wrapped ContentTreeNodeBuilder for file management
    data_builder: ContentTreeNodeBuilder,

    /// Version of the snapshot being written
    /// TODO: This field should not be needed for leaf writer. It's currently required
    /// as a workaround to force action tracking status to Existing (rather than Added).
    /// We need a better API - see usage at add_existing_actions() for details.
    version: Version,

    /// Table schema
    table_schema: SchemaRef,

    /// Manifest deletion vectors: which entries to mark deleted in existing manifests
    manifest_dvs: HashMap<String, RoaringTreemap>,

    root_entries_to_remove: HashSet<String>,

    /// Root DV entries to remove (DV paths that should be removed from root DV manifest)
    root_dv_entries_to_remove: HashSet<String>,

    /// Snapshot ID for tracking info
    snapshot_id: i64,

    /// Optional root manifest relative path for validation
    /// Used to prevent updating DVs for files still in the root manifest
    root_manifest_path: Option<String>,

    /// Whether to track root entries for removal.
    /// Set to false when Transaction has released root to client control.
    track_root_removals: bool,

    /// Row ID allocator for assigning `first_row_id` values in the leaf.
    allocator: CursorRowIdAllocator,
}

/// Context for tracking manifest entry deletions
struct ManifestRemovalContext<'a> {
    root_manifest_path: Option<String>,
    manifest_dvs: &'a mut HashMap<String, RoaringTreemap>,
    root_entries_to_remove: &'a mut HashSet<String>,
    track_root_removals: bool,
}

/// Helper to track manifest entries for deletion
/// Updates either data manifest DVs or root removal sets based on manifest location
fn track_manifest_entry_for_removal(
    manifest_path: Option<String>,
    manifest_position: Option<i64>,
    path: String,
    ctx: &mut ManifestRemovalContext<'_>,
) -> DeltaResult<()> {
    if let (Some(manifest_path_str), Some(position)) = (manifest_path, manifest_position) {
        // Check if this is the root manifest by comparing relative paths
        let is_from_root = ctx.root_manifest_path.as_deref() == Some(manifest_path_str.as_str());

        if is_from_root {
            if ctx.track_root_removals {
                ctx.root_entries_to_remove.insert(path);
            }
        } else {
            // Store the relative path directly
            let entry = ctx.manifest_dvs.entry(manifest_path_str).or_default();
            entry.insert(position as u64);
        }
    } else {
        // Files without manifest info are in root
        if ctx.track_root_removals {
            ctx.root_entries_to_remove.insert(path);
        }
    }
    Ok(())
}

/// Helper visitor to extract Add actions from scan rows and add them to a leaf
struct ScanRowVisitor<'a> {
    leaf_writer: &'a mut LeafNodeWriter,
    selection_vector: Vec<bool>,
}

type ColumnNamesAndTypes = (Vec<ColumnName>, Vec<DataType>);

/// Scan columns (all primitive leaf values — no struct columns needed).
static BASE_SCAN_COLUMNS: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
    use crate::schema::{column_name, MapType};
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
        column_name!("fileConstantValues.baseRowId"),
        column_name!("fileConstantValues.defaultRowCommitVersion"),
        column_name!("fileConstantValues.dataManifestPath"),
        column_name!("fileConstantValues.dataManifestPosition"),
    ];
    let types = vec![
        DataType::STRING,
        DataType::LONG,
        DataType::LONG,
        DataType::STRING,
        DataType::STRING,
        DataType::STRING,
        DataType::INTEGER,
        DataType::INTEGER,
        DataType::LONG,
        DataType::Map(Box::new(MapType::new(
            DataType::STRING,
            DataType::STRING,
            true,
        ))),
        DataType::LONG,
        DataType::LONG,
        DataType::STRING,
        DataType::LONG,
    ];
    (names, types)
});

impl<'a> RowVisitor for ScanRowVisitor<'a> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        (&BASE_SCAN_COLUMNS.0, &BASE_SCAN_COLUMNS.1)
    }

    fn visit<'b>(&mut self, row_count: usize, getters: &[&'b dyn GetData<'b>]) -> DeltaResult<()> {
        // Fixed getter indices for all columns (all primitive leaf values):
        // Layout: path, size, modificationTime, stats, + 5 DV fields, partitionValues,
        //         baseRowId, defaultRowCommitVersion, dataManifestPath, dataManifestPosition
        // Note: stats_parsed is pre-extracted into self.stats_per_row before visit() is called.
        // Note: tags is intentionally skipped (not extracted) as it has nullable values
        //       which are not yet supported in the scan API
        const PATH_IDX: usize = 0;
        const DATA_MANIFEST_PATH_IDX: usize = 12;
        const DATA_MANIFEST_POSITION_IDX: usize = 13;

        let root_manifest_path = self.leaf_writer.root_manifest_path.clone();

        for i in 0..row_count {
            // Skip rows that are not selected according to the selection vector
            // If selection vector is shorter than row_count, remaining rows are assumed selected
            if i < self.selection_vector.len() && !self.selection_vector[i] {
                continue;
            }

            let path_opt: Option<String> = getters[PATH_IDX].get_opt(i, "path")?;
            let Some(path) = path_opt else {
                continue;
            };

            // Track data file manifest entry for removal
            let data_manifest_path: Option<String> = getters[DATA_MANIFEST_PATH_IDX]
                .get_opt(i, "fileConstantValues.dataManifestPath")?;
            let data_manifest_position: Option<i64> = getters[DATA_MANIFEST_POSITION_IDX]
                .get_opt(i, "fileConstantValues.dataManifestPosition")?;
            let mut ctx = ManifestRemovalContext {
                root_manifest_path: root_manifest_path.clone(),
                manifest_dvs: &mut self.leaf_writer.manifest_dvs,
                root_entries_to_remove: &mut self.leaf_writer.root_entries_to_remove,
                track_root_removals: self.leaf_writer.track_root_removals,
            };
            track_manifest_entry_for_removal(
                data_manifest_path,
                data_manifest_position,
                path.clone(),
                &mut ctx,
            )?;
        }
        Ok(())
    }
}

impl LeafNodeWriter {
    /// Creates a new LeafNodeWriter.
    ///
    /// # Arguments
    /// * `table_root` - The root URL of the Delta table
    /// * `version` - The version this leaf is being written for
    /// * `snapshot_id` - The snapshot ID for tracking info
    /// * `table_schema` - The table's data schema with PARQUET:field_id metadata
    /// * `partition_type` - The partition struct type, or `None` for unpartitioned tables
    /// * `track_root_removals` - Whether to track root entries for removal
    /// * `root_manifest_url` - Optional URL of the root manifest for validation
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        table_root: Url,
        version: Version,
        snapshot_id: i64,
        table_schema: SchemaRef,
        partition_type: Option<StructType>,
        track_root_removals: bool,
        root_manifest_path: Option<String>,
        starting_first_row_id: i64,
    ) -> Self {
        Self {
            data_builder: ContentTreeNodeBuilder::new_for(
                table_root.clone(),
                version,
                table_schema.as_ref().clone(),
            )
            .with_partition_type(partition_type),
            version,
            table_schema: table_schema.clone(),
            manifest_dvs: HashMap::new(),
            root_entries_to_remove: HashSet::new(),
            root_dv_entries_to_remove: HashSet::new(),
            snapshot_id,
            root_manifest_path,
            track_root_removals,
            allocator: CursorRowIdAllocator::new(starting_first_row_id),
        }
    }

    /// Buffer net new files for writing to data manifest.
    ///
    /// This method is designed for manifest commit scenarios where the data contains simple
    /// write metadata (path, partitionValues, size, modificationTime, stats) rather than
    /// full Add actions.
    ///
    /// The stats field is expected to be a StructData in content_stats format (matching the
    /// stats schema generated from the table schema).
    ///
    /// # Arguments
    /// * `add_metadata` - EngineData with write metadata format (path, partitionValues, size,
    ///   modificationTime, stats as StructData). Stats can be in either AMT format (per-column
    ///   stats) or Delta JSON format (numRecords, minValues, etc.) - Delta JSON format is
    ///   automatically converted to AMT format.
    pub fn add_files(
        &mut self,
        engine: &dyn Engine,
        add_metadata: Box<dyn EngineData>,
    ) -> DeltaResult<()> {
        let converted = crate::content_tree::stats::try_pre_convert_stats_column(
            engine,
            add_metadata.as_ref(),
            "stats",
            &self.table_schema,
            &crate::transaction::BASE_ADD_FILES_SCHEMA,
        )?;
        let data: &dyn EngineData = match &converted {
            Some(c) => c.as_ref(),
            None => add_metadata.as_ref(),
        };
        self.data_builder.add_from_engine_data_write(
            engine,
            data,
            self.version,
            self.snapshot_id,
        )?;

        Ok(())
    }

    /// Move existing files from tree/log into this leaf.
    ///
    /// For tree entries: marks source manifest entries as deleted via manifest DVs.
    ///
    /// # Arguments
    /// * `scan_metadata` - FilteredEngineData with scan row schema including metadata columns
    pub fn add_existing_actions(
        &mut self,
        engine: &dyn Engine,
        scan_metadata: FilteredEngineData,
    ) -> DeltaResult<()> {
        let selection_vector = scan_metadata.selection_vector().to_vec();
        let data = scan_metadata.data();

        // Write files to new leaf using expression-based transform (scan rows → manifest entries).
        // Stats handling (stats_parsed vs parse_json fallback) is managed inside this method.
        self.data_builder.add_from_existing_scan_rows(
            engine,
            data,
            &selection_vector,
            self.version,
            self.snapshot_id,
        )?;

        // Track manifest entry removals (mark source manifest entries as deleted via DVs).
        let mut visitor = ScanRowVisitor {
            leaf_writer: self,
            selection_vector,
        };
        visitor.visit_rows_of(data)?;

        Ok(())
    }

    /// Write parquet files for data/DV manifests and return metadata.
    ///
    /// # Arguments
    /// * `engine` - The engine to use for writing
    ///
    /// # Returns
    /// LeafNodeWriterResult with written manifests and tracking info
    pub fn finish(mut self, engine: &dyn Engine) -> DeltaResult<LeafNodeWriterResult> {
        // Write data manifest using ContentTreeNodeBuilder's write_leaf()
        // DV info is inline on data entries, so no separate DV manifest is needed.
        let data_manifest_entry = if self.data_builder.has_entries() {
            Some(
                self.data_builder
                    .write_leaf(engine, self.snapshot_id, &mut self.allocator)?,
            )
        } else {
            None
        };

        Ok(LeafNodeWriterResult {
            manifest_dvs: self.manifest_dvs,
            root_entries_to_remove: self.root_entries_to_remove,
            root_dv_entries_to_remove: self.root_dv_entries_to_remove,
            data_file_manifest_written: data_manifest_entry,
            final_cursor: self.allocator.current(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::content_tree::parse_or_join_url;
    use crate::schema::{
        ColumnMetadataKey, DataType, MapType, MetadataValue, StructField, StructType,
    };

    /// Helper to create a test engine, table root URL, and schema
    fn test_setup() -> (Arc<dyn Engine>, Url, SchemaRef) {
        use crate::engine::default::DefaultEngineBuilder;
        use crate::object_store::local::LocalFileSystem;

        let temp_path = tempfile::tempdir().unwrap().keep();
        // Use LocalFileSystem::new() (no prefix) so that absolute URLs passed to write_parquet_file
        // resolve correctly: Path::from_url_path strips the leading '/', then LocalFileSystem
        // with no prefix uses the filesystem root, matching the absolute path.
        let store = Arc::new(LocalFileSystem::new());
        let engine: Arc<dyn Engine> = Arc::new(DefaultEngineBuilder::new(store).build());
        let table_root = Url::from_directory_path(&temp_path).unwrap();

        // Create a simple test schema with parquet field IDs
        let schema = Arc::new(
            StructType::try_new(vec![
                StructField::not_null("id", DataType::INTEGER).with_metadata([(
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                )]),
                StructField::nullable("value", DataType::STRING).with_metadata([(
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(2),
                )]),
            ])
            .unwrap(),
        );

        (engine, table_root, schema)
    }

    /// Helper to create add files metadata with Delta JSON format stats.
    ///
    /// This mimics what the engine produces when writing parquet files: stats in Delta JSON
    /// format with numRecords, minValues, maxValues, nullCount, and tightBounds.
    /// The stats will be automatically converted to AMT format by the builder.
    ///
    /// Parameters for each file: (path, size, mod_time, num_records, id_min, id_max, id_null_count,
    /// value_min, value_max, value_null_count)
    #[allow(clippy::type_complexity)]
    fn create_test_add_metadata_with_delta_json_stats(
        files: Vec<(
            &str,
            i64,
            i64,
            i64,
            i32,
            i32,
            i64,
            Option<&str>,
            Option<&str>,
            i64,
        )>,
    ) -> DeltaResult<Box<dyn EngineData>> {
        use crate::arrow::array::{
            Array, ArrayRef, BooleanArray, Int32Array, Int64Array, MapArray, StringArray,
            StructArray,
        };
        use crate::arrow::buffer::OffsetBuffer;
        use crate::arrow::datatypes::{DataType as ArrowDataType, Field, Fields};
        use crate::arrow::record_batch::RecordBatch;
        use crate::engine::arrow_data::ArrowEngineData;

        let num_files = files.len();

        // Build arrays for each file
        let path_array = StringArray::from(
            files
                .iter()
                .map(|(p, _, _, _, _, _, _, _, _, _)| *p)
                .collect::<Vec<_>>(),
        );
        let size_array = Int64Array::from(
            files
                .iter()
                .map(|(_, s, _, _, _, _, _, _, _, _)| *s)
                .collect::<Vec<_>>(),
        );
        let mod_time_array = Int64Array::from(
            files
                .iter()
                .map(|(_, _, m, _, _, _, _, _, _, _)| *m)
                .collect::<Vec<_>>(),
        );

        // Create empty map for partitionValues
        let entries_field = Arc::new(Field::new(
            "key_value",
            ArrowDataType::Struct(Fields::from(vec![
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
            ])),
            false,
        ));
        let empty_keys = StringArray::from(Vec::<&str>::new());
        let empty_values = StringArray::from(Vec::<Option<&str>>::new());
        let empty_entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(empty_keys) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                Arc::new(empty_values) as ArrayRef,
            ),
        ]);
        let offsets = OffsetBuffer::from_lengths(vec![0; num_files]);
        let partition_values_array = Arc::new(MapArray::new(
            entries_field,
            offsets,
            empty_entries,
            None,
            false,
        ));

        // Build stats struct in Delta JSON format (like engine produces):
        // { numRecords, nullCount: {id, value}, minValues: {id, value}, maxValues: {id, value},
        // tightBounds }
        let num_records = Int64Array::from(
            files
                .iter()
                .map(|(_, _, _, n, _, _, _, _, _, _)| *n)
                .collect::<Vec<_>>(),
        );

        // nullCount struct: { id: i64, value: i64 }
        let null_count_id = Int64Array::from(
            files
                .iter()
                .map(|(_, _, _, _, _, _, nc, _, _, _)| *nc)
                .collect::<Vec<_>>(),
        );
        let null_count_value = Int64Array::from(
            files
                .iter()
                .map(|(_, _, _, _, _, _, _, _, _, nc)| *nc)
                .collect::<Vec<_>>(),
        );
        let null_count_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("id", ArrowDataType::Int64, true)),
                Arc::new(null_count_id) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Int64, true)),
                Arc::new(null_count_value) as ArrayRef,
            ),
        ]);

        // minValues struct: { id: i32, value: string }
        let min_id = Int32Array::from(
            files
                .iter()
                .map(|(_, _, _, _, min, _, _, _, _, _)| *min)
                .collect::<Vec<_>>(),
        );
        let min_value = StringArray::from(
            files
                .iter()
                .map(|(_, _, _, _, _, _, _, min, _, _)| *min)
                .collect::<Vec<_>>(),
        );
        let min_values_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("id", ArrowDataType::Int32, true)),
                Arc::new(min_id) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                Arc::new(min_value) as ArrayRef,
            ),
        ]);

        // maxValues struct: { id: i32, value: string }
        let max_id = Int32Array::from(
            files
                .iter()
                .map(|(_, _, _, _, _, max, _, _, _, _)| *max)
                .collect::<Vec<_>>(),
        );
        let max_value = StringArray::from(
            files
                .iter()
                .map(|(_, _, _, _, _, _, _, _, max, _)| *max)
                .collect::<Vec<_>>(),
        );
        let max_values_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("id", ArrowDataType::Int32, true)),
                Arc::new(max_id) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                Arc::new(max_value) as ArrayRef,
            ),
        ]);

        // tightBounds: boolean (all true)
        let tight_bounds = BooleanArray::from(vec![true; num_files]);

        // Combine into top-level stats struct (Delta JSON format)
        let stats_struct = StructArray::from(vec![
            (
                Arc::new(Field::new("numRecords", ArrowDataType::Int64, true)),
                Arc::new(num_records) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "nullCount",
                    null_count_struct.data_type().clone(),
                    true,
                )),
                Arc::new(null_count_struct) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "minValues",
                    min_values_struct.data_type().clone(),
                    true,
                )),
                Arc::new(min_values_struct) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "maxValues",
                    max_values_struct.data_type().clone(),
                    true,
                )),
                Arc::new(max_values_struct) as ArrayRef,
            ),
            (
                Arc::new(Field::new("tightBounds", ArrowDataType::Boolean, true)),
                Arc::new(tight_bounds) as ArrayRef,
            ),
        ]);

        // Build the Arrow schema
        let arrow_schema = Arc::new(crate::arrow::datatypes::Schema::new(vec![
            Field::new("path", ArrowDataType::Utf8, false),
            Field::new(
                "partitionValues",
                ArrowDataType::Map(
                    Arc::new(Field::new(
                        "key_value",
                        ArrowDataType::Struct(Fields::from(vec![
                            Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                            Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                        ])),
                        false,
                    )),
                    false,
                ),
                false,
            ),
            Field::new("size", ArrowDataType::Int64, false),
            Field::new("modificationTime", ArrowDataType::Int64, false),
            Field::new("stats", stats_struct.data_type().clone(), true),
        ]));

        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(path_array) as ArrayRef,
                partition_values_array as ArrayRef,
                Arc::new(size_array) as ArrayRef,
                Arc::new(mod_time_array) as ArrayRef,
                Arc::new(stats_struct) as ArrayRef,
            ],
        )?;

        Ok(Box::new(ArrowEngineData::new(batch)))
    }

    /// Helper to create add files metadata for testing without stats (null stats).
    fn create_test_add_metadata(files: Vec<(&str, i64, i64)>) -> DeltaResult<Box<dyn EngineData>> {
        use crate::arrow::array::{ArrayRef, Int64Array, MapArray, StringArray, StructArray};
        use crate::arrow::buffer::{NullBuffer, OffsetBuffer};
        use crate::arrow::datatypes::{DataType as ArrowDataType, Field, Fields};
        use crate::arrow::record_batch::RecordBatch;
        use crate::engine::arrow_conversion::TryFromKernel;
        use crate::engine::arrow_data::ArrowEngineData;

        let num_files = files.len();

        // Create schema for add files (path, partitionValues, size, modificationTime, stats)
        // Note: stats is nullable struct with empty fields - we use null struct values
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::not_null("path", DataType::STRING),
            StructField::not_null(
                "partitionValues",
                DataType::Map(Box::new(MapType::new(
                    DataType::STRING,
                    DataType::STRING,
                    true,
                ))),
            ),
            StructField::not_null("size", DataType::LONG),
            StructField::not_null("modificationTime", DataType::LONG),
            // Empty struct for stats - will be all nulls
            StructField::nullable("stats", DataType::struct_type_unchecked(vec![])),
        ]));

        // Build arrays for each file
        let path_array = StringArray::from(files.iter().map(|(p, _, _)| *p).collect::<Vec<_>>());
        let size_array = Int64Array::from(files.iter().map(|(_, s, _)| *s).collect::<Vec<_>>());
        let mod_time_array = Int64Array::from(files.iter().map(|(_, _, m)| *m).collect::<Vec<_>>());

        // Create empty map for partitionValues
        let entries_field = Arc::new(Field::new(
            "key_value",
            ArrowDataType::Struct(Fields::from(vec![
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
            ])),
            false,
        ));
        let empty_keys = StringArray::from(Vec::<&str>::new());
        let empty_values = StringArray::from(Vec::<Option<&str>>::new());
        let empty_entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(empty_keys) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                Arc::new(empty_values) as ArrayRef,
            ),
        ]);
        let offsets = OffsetBuffer::from_lengths(vec![0; num_files]);
        let partition_values_array = Arc::new(MapArray::new(
            entries_field,
            offsets,
            empty_entries,
            None,
            false,
        ));

        // Create all-null struct array for stats (empty struct with no fields)
        // Use new_empty_fields for struct arrays with no child fields
        let stats_array = StructArray::new_empty_fields(
            num_files,
            Some(NullBuffer::from(vec![false; num_files])),
        );

        let batch = RecordBatch::try_new(
            Arc::new(TryFromKernel::try_from_kernel(schema.as_ref())?),
            vec![
                Arc::new(path_array) as ArrayRef,
                partition_values_array as ArrayRef,
                Arc::new(size_array) as ArrayRef,
                Arc::new(mod_time_array) as ArrayRef,
                Arc::new(stats_array) as ArrayRef,
            ],
        )?;

        Ok(Box::new(ArrowEngineData::new(batch)))
    }

    /// Returns the kernel `DataType` for the `stats_parsed` column (Delta JSON format)
    /// for the test table schema {id: INTEGER, value: STRING}.
    fn stats_parsed_kernel_type() -> DataType {
        use crate::content_tree::builder::build_delta_stats_schema;
        let test_schema = StructType::try_new(vec![
            StructField::not_null("id", DataType::INTEGER),
            StructField::nullable("value", DataType::STRING),
        ])
        .expect("valid test schema");
        DataType::Struct(Box::new(build_delta_stats_schema(&test_schema)))
    }

    /// Returns the kernel scan row schema (including `stats_parsed`) for the test table.
    fn scan_row_schema_for_tests() -> Arc<StructType> {
        Arc::new(StructType::new_unchecked(vec![
            StructField::not_null("path", DataType::STRING),
            StructField::not_null("size", DataType::LONG),
            StructField::not_null("modificationTime", DataType::LONG),
            StructField::nullable("stats", DataType::STRING),
            StructField::nullable("stats_parsed", stats_parsed_kernel_type()),
            StructField::nullable(
                "deletionVector",
                DataType::struct_type_unchecked(vec![
                    StructField::nullable("storageType", DataType::STRING),
                    StructField::nullable("pathOrInlineDv", DataType::STRING),
                    StructField::nullable("offset", DataType::INTEGER),
                    StructField::nullable("sizeInBytes", DataType::INTEGER),
                    StructField::nullable("cardinality", DataType::LONG),
                ]),
            ),
            StructField::not_null(
                "fileConstantValues",
                DataType::struct_type_unchecked(vec![
                    StructField::not_null(
                        "partitionValues",
                        DataType::Map(Box::new(MapType::new(
                            DataType::STRING,
                            DataType::STRING,
                            true,
                        ))),
                    ),
                    StructField::nullable("baseRowId", DataType::LONG),
                    StructField::nullable("defaultRowCommitVersion", DataType::LONG),
                    StructField::nullable("dataManifestPath", DataType::STRING),
                    StructField::nullable("dataManifestPosition", DataType::LONG),
                ]),
            ),
            StructField::nullable("numRecords", DataType::LONG),
        ]))
    }

    /// Creates scan row `EngineData` by parsing JSON strings with the engine's JSON handler.
    ///
    /// Each JSON row is parsed against [`scan_row_schema_for_tests`]. Absent nullable fields
    /// default to null. Include `"stats_parsed": {...}` for the preferred-path test, or
    /// `"stats": "{...escaped...}"` for the JSON-fallback-path test.
    fn create_scan_row_engine_data(
        engine: &Arc<dyn Engine>,
        json_rows: &[&str],
    ) -> DeltaResult<Box<dyn EngineData>> {
        use crate::arrow::array::StringArray;
        use crate::utils::test_utils::string_array_to_engine_data;
        let strings: StringArray = json_rows.to_vec().into();
        engine.json_handler().parse_json(
            string_array_to_engine_data(strings),
            scan_row_schema_for_tests(),
        )
    }

    /// Verifies content_stats in a written manifest file.
    #[allow(clippy::too_many_arguments)]
    fn verify_manifest_content_stats(
        engine: &Arc<dyn Engine>,
        manifest_entry: &ContentTreeNodeEntry,
        table_root: &Url,
        schema: &SchemaRef,
        expected_record_count: i64,
        expected_id_min: i32,
        expected_id_max: i32,
        expected_value_min: &str,
        expected_value_max: &str,
        expected_value_nc: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::arrow::array::{Array, Int32Array, Int64Array, StringArray, StructArray};
        use crate::engine::arrow_data::ArrowEngineData as ArrowEngineData2;
        use crate::FileMeta;

        let manifest_location = manifest_entry
            .location
            .as_ref()
            .expect("Manifest should have location");
        let manifest_url = parse_or_join_url(manifest_location, table_root)?;
        let manifest_path = manifest_url.to_file_path().unwrap();
        let manifest_file_size = std::fs::metadata(&manifest_path)?.len();
        let file_meta = FileMeta {
            location: manifest_url,
            last_modified: 0,
            size: manifest_file_size,
        };
        let delta_stats = crate::content_tree::builder::build_delta_stats_schema(schema);
        let read_schema = Arc::new(
            crate::content_tree::ContentTreeNodeEntry::to_schema_with_content_stats(
                schema,
                &delta_stats,
                None,
            )?,
        );
        let mut found = false;
        for batch_result in
            engine
                .parquet_handler()
                .read_parquet_files(&[file_meta], read_schema, None)?
        {
            let batch = batch_result?;
            let record_batch = batch
                .any_ref()
                .downcast_ref::<ArrowEngineData2>()
                .expect("Expected ArrowEngineData")
                .record_batch();

            let rc = record_batch
                .column_by_name("recordCount")
                .expect("recordCount");
            assert_eq!(
                rc.as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64")
                    .value(0),
                expected_record_count,
                "recordCount mismatch"
            );

            let cs = record_batch
                .column_by_name(crate::content_tree::CONTENT_STATS_FIELD_NAME)
                .expect("content_stats");
            assert!(cs.is_valid(0), "content_stats should not be null");
            let stats = cs
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("StructArray");

            let id = stats.column_by_name("id").expect("id");
            let id_struct = id
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("StructArray");
            assert_eq!(
                id_struct
                    .column_by_name(crate::content_tree::LOWER_BOUND)
                    .expect("lower_bound")
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32")
                    .value(0),
                expected_id_min,
                "id.lower_bound mismatch"
            );
            assert_eq!(
                id_struct
                    .column_by_name(crate::content_tree::UPPER_BOUND)
                    .expect("upper_bound")
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32")
                    .value(0),
                expected_id_max,
                "id.upper_bound mismatch"
            );

            let val = stats.column_by_name("value").expect("value");
            let val_struct = val
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("StructArray");
            assert_eq!(
                val_struct
                    .column_by_name(crate::content_tree::LOWER_BOUND)
                    .expect("lower_bound")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("StringArray")
                    .value(0),
                expected_value_min,
                "value.lower_bound mismatch"
            );
            assert_eq!(
                val_struct
                    .column_by_name(crate::content_tree::UPPER_BOUND)
                    .expect("upper_bound")
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("StringArray")
                    .value(0),
                expected_value_max,
                "value.upper_bound mismatch"
            );
            assert_eq!(
                val_struct
                    .column_by_name(crate::content_tree::NULL_VALUE_COUNT)
                    .expect("null_count")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64")
                    .value(0),
                expected_value_nc,
                "value.null_value_count mismatch"
            );
            found = true;
        }
        assert!(found, "Should have read the manifest");
        Ok(())
    }

    #[test]
    fn test_basic_leaf_writing() -> Result<(), Box<dyn std::error::Error>> {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let mut writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema.clone(),
            None,
            true,
            None,
            0,
        );

        // Add files with Delta JSON format stats (like the engine produces when writing parquet).
        // The stats will be automatically converted to AMT format by the builder.
        // Parameters: (path, size, mod_time, num_records, id_min, id_max, id_null_count, value_min,
        // value_max, value_null_count)
        let metadata = create_test_add_metadata_with_delta_json_stats(vec![(
            "file1.parquet",
            1024,          // size
            1000000,       // modification_time
            100,           // num_records
            1,             // id min
            1000,          // id max
            0,             // id null count
            Some("alice"), // value min
            Some("zoe"),   // value max
            5,             // value null count
        )])?;
        writer.add_files(engine.as_ref(), metadata)?;

        // Finish and verify result
        let result = writer.finish(engine.as_ref())?;

        // VERIFY: Result structure
        assert!(
            result.data_file_manifest_written.is_some(),
            "Data manifest should be written"
        );
        assert!(
            result.manifest_dvs.is_empty(),
            "Manifest DVs should be empty"
        );
        assert!(
            result.root_entries_to_remove.is_empty(),
            "Root entries to remove should be empty"
        );

        // VERIFY: Manifest entry has a location
        let manifest_entry = result.data_file_manifest_written.unwrap();
        assert!(
            manifest_entry.location.is_some(),
            "Manifest entry should have a location"
        );

        // Read back the manifest parquet file directly to verify content_stats columns
        // The ContentTreeNodeEntryVisitor doesn't read content_stats (it's table-schema-dependent),
        // so we read the parquet file directly and check the columns are present.
        let manifest_location = manifest_entry.location.as_ref().unwrap();
        let manifest_url = parse_or_join_url(manifest_location, &table_root)?;

        // Use the engine's parquet handler to read the file with a schema that includes
        // content_stats
        use crate::arrow::array::Array;
        use crate::engine::arrow_data::ArrowEngineData;
        use crate::FileMeta;

        // Get actual manifest file size from the filesystem so parquet reader can locate the
        // footer.
        let manifest_path = manifest_url.to_file_path().unwrap();
        let manifest_file_size = std::fs::metadata(&manifest_path)?.len();
        let file_meta = FileMeta {
            location: manifest_url.clone(),
            last_modified: 0,
            size: manifest_file_size,
        };

        let parquet_handler = engine.parquet_handler();

        // Use the production schema to ensure test matches actual behavior
        // This generates the full ContentTreeNodeEntry schema with content_stats based on table
        // schema
        let delta_stats = crate::content_tree::builder::build_delta_stats_schema(&schema);
        let read_schema = Arc::new(
            crate::content_tree::ContentTreeNodeEntry::to_schema_with_content_stats(
                &schema,
                &delta_stats,
                None,
            )?,
        );

        let read_result_iter =
            parquet_handler.read_parquet_files(&[file_meta], read_schema, None)?;

        let mut found_stats = false;
        for batch_result in read_result_iter {
            let batch = batch_result?;
            let arrow_data = batch
                .any_ref()
                .downcast_ref::<ArrowEngineData>()
                .expect("Expected ArrowEngineData");
            let record_batch = arrow_data.record_batch();

            // Check that content_stats column exists and has data
            let content_stats_col =
                record_batch.column_by_name(crate::content_tree::CONTENT_STATS_FIELD_NAME);
            assert!(
                content_stats_col.is_some(),
                "content_stats column should exist in manifest"
            );

            let stats_array = content_stats_col.unwrap();
            assert_eq!(stats_array.len(), 1, "Should have 1 row");

            // Verify the stats are not null
            assert!(
                stats_array.is_valid(0),
                "content_stats should not be null for the entry"
            );

            // Access the struct array to verify nested values
            let stats_struct = stats_array
                .as_any()
                .downcast_ref::<crate::arrow::array::StructArray>()
                .expect("content_stats should be a struct array");

            // Verify 'id' column stats exist
            let id_stats = stats_struct.column_by_name("id");
            assert!(id_stats.is_some(), "id stats should exist in content_stats");
            let id_struct = id_stats
                .unwrap()
                .as_any()
                .downcast_ref::<crate::arrow::array::StructArray>()
                .expect("id stats should be a struct");

            // Verify id.value_count
            let id_value_count = id_struct
                .column_by_name(crate::content_tree::VALUE_COUNT)
                .expect("id.value_count should exist");
            let id_vc_array = id_value_count
                .as_any()
                .downcast_ref::<crate::arrow::array::Int64Array>()
                .expect("value_count should be Int64");
            assert_eq!(id_vc_array.value(0), 100, "id.value_count should be 100");

            // Verify id.lower_bound
            let id_lower = id_struct
                .column_by_name(crate::content_tree::LOWER_BOUND)
                .expect("id.lower_bound should exist");
            let id_lb_array = id_lower
                .as_any()
                .downcast_ref::<crate::arrow::array::Int32Array>()
                .expect("lower_bound should be Int32");
            assert_eq!(id_lb_array.value(0), 1, "id.lower_bound should be 1");

            // Verify id.upper_bound
            let id_upper = id_struct
                .column_by_name(crate::content_tree::UPPER_BOUND)
                .expect("id.upper_bound should exist");
            let id_ub_array = id_upper
                .as_any()
                .downcast_ref::<crate::arrow::array::Int32Array>()
                .expect("upper_bound should be Int32");
            assert_eq!(id_ub_array.value(0), 1000, "id.upper_bound should be 1000");

            // Verify id.tight_bounds
            let id_tight = id_struct
                .column_by_name(crate::content_tree::TIGHT_BOUNDS)
                .expect("id.tight_bounds should exist");
            let id_tb_array = id_tight
                .as_any()
                .downcast_ref::<crate::arrow::array::BooleanArray>()
                .expect("tight_bounds should be Boolean");
            assert!(id_tb_array.value(0), "id.tight_bounds should be true");

            // Verify 'value' column stats exist
            let value_stats = stats_struct.column_by_name("value");
            assert!(
                value_stats.is_some(),
                "value stats should exist in content_stats"
            );
            let value_struct = value_stats
                .unwrap()
                .as_any()
                .downcast_ref::<crate::arrow::array::StructArray>()
                .expect("value stats should be a struct");

            // Verify value.value_count
            let value_value_count = value_struct
                .column_by_name(crate::content_tree::VALUE_COUNT)
                .expect("value.value_count should exist");
            let value_vc_array = value_value_count
                .as_any()
                .downcast_ref::<crate::arrow::array::Int64Array>()
                .expect("value_count should be Int64");
            assert_eq!(
                value_vc_array.value(0),
                100,
                "value.value_count should be 100"
            );

            // Verify value.null_count
            let value_null_count = value_struct
                .column_by_name(crate::content_tree::NULL_VALUE_COUNT)
                .expect("value.null_count should exist");
            let value_nc_array = value_null_count
                .as_any()
                .downcast_ref::<crate::arrow::array::Int64Array>()
                .expect("null_count should be Int64");
            assert_eq!(value_nc_array.value(0), 5, "value.null_count should be 5");

            // Verify value.lower_bound
            let value_lower = value_struct
                .column_by_name(crate::content_tree::LOWER_BOUND)
                .expect("value.lower_bound should exist");
            let value_lb_array = value_lower
                .as_any()
                .downcast_ref::<crate::arrow::array::StringArray>()
                .expect("lower_bound should be String");
            assert_eq!(
                value_lb_array.value(0),
                "alice",
                "value.lower_bound should be 'alice'"
            );

            // Verify value.upper_bound
            let value_upper = value_struct
                .column_by_name(crate::content_tree::UPPER_BOUND)
                .expect("value.upper_bound should exist");
            let value_ub_array = value_upper
                .as_any()
                .downcast_ref::<crate::arrow::array::StringArray>()
                .expect("upper_bound should be String");
            assert_eq!(
                value_ub_array.value(0),
                "zoe",
                "value.upper_bound should be 'zoe'"
            );

            // Verify value.tight_bounds
            let value_tight = value_struct
                .column_by_name(crate::content_tree::TIGHT_BOUNDS)
                .expect("value.tight_bounds should exist");
            let value_tb_array = value_tight
                .as_any()
                .downcast_ref::<crate::arrow::array::BooleanArray>()
                .expect("tight_bounds should be Boolean");
            assert!(value_tb_array.value(0), "value.tight_bounds should be true");

            found_stats = true;
        }

        assert!(found_stats, "Should have read stats from manifest");

        Ok(())
    }

    #[test]
    fn test_multiple_files_in_leaf() -> Result<(), Box<dyn std::error::Error>> {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let mut writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema,
            None,
            true,
            None,
            0,
        );

        // Add 10 files (path, size, modification_time)
        let files: Vec<_> = (0..10)
            .map(|i| {
                (
                    format!("file{}.parquet", i).leak() as &str,
                    1024 + i * 100,
                    1000000 + i,
                )
            })
            .collect();

        let metadata = create_test_add_metadata(files)?;
        writer.add_files(engine.as_ref(), metadata)?;

        // Finish and verify result
        let result = writer.finish(engine.as_ref())?;

        // VERIFY: Result shows 1 data manifest
        assert!(
            result.data_file_manifest_written.is_some(),
            "Data manifest should be written"
        );
        Ok(())
    }

    #[test]
    fn test_empty_leaf() -> Result<(), Box<dyn std::error::Error>> {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema,
            None,
            true,
            None,
            0,
        );

        // Don't add any files, just call finish
        let result = writer.finish(engine.as_ref())?;

        // VERIFY: No manifests written
        assert!(
            result.data_file_manifest_written.is_none(),
            "Data manifest should not be written"
        );
        assert!(
            result.manifest_dvs.is_empty(),
            "Manifest DVs should be empty"
        );
        assert!(
            result.root_entries_to_remove.is_empty(),
            "Root entries to remove should be empty"
        );

        Ok(())
    }

    #[test]
    fn test_selection_vector_filtering() -> Result<(), Box<dyn std::error::Error>> {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let mut writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema,
            None,
            true,
            None,
            0,
        );

        // 4 files with no stats — selection vector keeps rows 0 and 2
        let engine_data = create_scan_row_engine_data(
            &engine,
            &[
                r#"{"path":"file0.parquet","size":1000,"modificationTime":1000000,"fileConstantValues":{"partitionValues":{}}}"#,
                r#"{"path":"file1.parquet","size":2000,"modificationTime":1000001,"fileConstantValues":{"partitionValues":{}}}"#,
                r#"{"path":"file2.parquet","size":3000,"modificationTime":1000002,"fileConstantValues":{"partitionValues":{}}}"#,
                r#"{"path":"file3.parquet","size":4000,"modificationTime":1000003,"fileConstantValues":{"partitionValues":{}}}"#,
            ],
        )?;

        // Create a selection vector that marks rows 0 and 2 as selected, rows 1 and 3 as deleted
        let selection_vector = vec![true, false, true, false];
        let filtered_data = FilteredEngineData::try_new(engine_data, selection_vector)?;

        // Add the existing actions with the filtered data
        writer.add_existing_actions(engine.as_ref(), filtered_data)?;

        // Finish and check the result
        let result = writer.finish(engine.as_ref())?;

        assert!(
            result.data_file_manifest_written.is_some(),
            "Data manifest should be written"
        );

        // Verify the manifest contains exactly 2 files (the selected ones: file0.parquet and
        // file2.parquet)
        let manifest_entry = result.data_file_manifest_written.unwrap();
        let manifest_location = manifest_entry
            .location
            .as_ref()
            .expect("Manifest should have location");

        // Read back the manifest file to verify contents
        use crate::content_tree::ContentTreeNode;

        // manifest_location is now a relative path, join with table_root
        let manifest_url = parse_or_join_url(manifest_location, &table_root)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &manifest_url,
            manifest_location.to_string(),
            None,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let manifest_metadata = ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_root.clone(),
        )?;
        let entries = manifest_metadata.entries()?;

        // The manifest should contain exactly 2 entries
        assert_eq!(
            entries.len(),
            2,
            "Manifest should contain exactly 2 files (selected rows 0 and 2), but found {}",
            entries.len()
        );

        // Extract the location (file path) from the entries
        // These are now relative paths, so we can use them directly
        let paths: Vec<String> = entries
            .iter()
            .filter_map(|entry| {
                entry.location.as_ref().map(|loc| {
                    // Extract just the filename from the path (handles both relative paths and
                    // URLs)
                    loc.rsplit('/').next().unwrap_or(loc).to_string()
                })
            })
            .collect();

        // Should have exactly 2 paths
        assert_eq!(paths.len(), 2, "Should have collected 2 file paths");

        // The paths should be file0.parquet and file2.parquet (rows 0 and 2 from the selection
        // vector)
        assert!(
            paths.contains(&"file0.parquet".to_string()),
            "Manifest should contain file0.parquet, but got: {:?}",
            paths
        );
        assert!(
            paths.contains(&"file2.parquet".to_string()),
            "Manifest should contain file2.parquet, but got: {:?}",
            paths
        );
        assert!(
            !paths.contains(&"file1.parquet".to_string()),
            "Manifest should NOT contain file1.parquet (was filtered out), but got: {:?}",
            paths
        );
        assert!(
            !paths.contains(&"file3.parquet".to_string()),
            "Manifest should NOT contain file3.parquet (was filtered out), but got: {:?}",
            paths
        );

        Ok(())
    }

    // Note: LW-3 (deletion vectors) and LW-4 (manifest DVs tracking) tests require
    // more complex setup with scan data and deletion vector descriptors.
    // These will be better tested in the Level 2 (transaction) tests where we can
    // use the full table infrastructure.

    #[test]
    /// Tests that `stats` JSON string (fallback path of the coalesce) is parsed into
    /// `content_stats` correctly. This covers the case where scan rows come from JSON
    /// commit log files (stats_parsed is null, stats is a JSON string).
    fn test_existing_actions_stats_preserved() -> Result<(), Box<dyn std::error::Error>> {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let mut writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema.clone(),
            None,
            true,
            None,
            0,
        );

        // Scan row with non-null `stats` JSON and null `stats_parsed` — fallback path of coalesce.
        let engine_data = create_scan_row_engine_data(
            &engine,
            &[
                r#"{"path":"file0.parquet","size":1024,"modificationTime":1000000,"stats":"{\"numRecords\":100,\"minValues\":{\"id\":1,\"value\":\"alice\"},\"maxValues\":{\"id\":100,\"value\":\"zoe\"},\"nullCount\":{\"id\":0,\"value\":5},\"tightBounds\":true}","fileConstantValues":{"partitionValues":{}}}"#,
            ],
        )?;

        let filtered_data = FilteredEngineData::try_new(engine_data, vec![true])?;
        writer.add_existing_actions(engine.as_ref(), filtered_data)?;
        let result = writer.finish(engine.as_ref())?;

        assert!(
            result.data_file_manifest_written.is_some(),
            "Data manifest should be written"
        );
        verify_manifest_content_stats(
            &engine,
            result.data_file_manifest_written.as_ref().unwrap(),
            &table_root,
            &schema,
            100,
            1,
            100,
            "alice",
            "zoe",
            5,
        )
    }

    #[test]
    /// Tests that a pre-populated `stats_parsed` column (preferred path of the coalesce) is used
    /// directly even when `stats` JSON is null. This covers scan rows from checkpoint scans where
    /// `include_stats_columns()` has already populated `stats_parsed` and `stats` is absent.
    fn test_existing_actions_prefers_stats_parsed_column() -> Result<(), Box<dyn std::error::Error>>
    {
        let (engine, table_root, schema) = test_setup();
        let version = 1;
        let snapshot_id = 12345;

        let mut writer = LeafNodeWriter::new(
            table_root.clone(),
            version,
            snapshot_id,
            schema.clone(),
            None,
            true,
            None,
            0,
        );

        // Scan row with null `stats` JSON and non-null `stats_parsed` — preferred path of coalesce.
        // The stats_parsed struct is parsed directly from the JSON (simulating what a checkpoint
        // scan produces when include_stats_columns() adds the pre-parsed column).
        let engine_data = create_scan_row_engine_data(
            &engine,
            &[
                r#"{"path":"file0.parquet","size":1024,"modificationTime":1000000,"stats_parsed":{"numRecords":100,"minValues":{"id":1,"value":"alice"},"maxValues":{"id":100,"value":"zoe"},"nullCount":{"id":0,"value":5},"tightBounds":true},"fileConstantValues":{"partitionValues":{}}}"#,
            ],
        )?;

        let filtered_data = FilteredEngineData::try_new(engine_data, vec![true])?;
        writer.add_existing_actions(engine.as_ref(), filtered_data)?;
        let result = writer.finish(engine.as_ref())?;

        assert!(
            result.data_file_manifest_written.is_some(),
            "Data manifest should be written"
        );
        verify_manifest_content_stats(
            &engine,
            result.data_file_manifest_written.as_ref().unwrap(),
            &table_root,
            &schema,
            100,
            1,
            100,
            "alice",
            "zoe",
            5,
        )
    }
}
