//! End-to-end integration tests for manifest-level data skipping in Adaptive Metadata Trees (AMT).
//!
//! These tests verify that:
//! 1. Stats are properly written to root manifest entries (content_stats field)
//! 2. Manifest-level filtering works via Scan API
//! 3. Manifests are correctly pruned based on predicates
//!
//! TODO: Simplify test data creation - currently uses verbose Arrow array construction.
//! Consider extending test_utils::create_add_files_metadata() to accept min/max stat values,
//! or using JSON commits if we can resolve timing/caching issues with in-memory store.

use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::arrow::array::{
    ArrayRef, BooleanArray, Int64Array, MapArray, StringArray, StructArray,
};
use delta_kernel::arrow::buffer::OffsetBuffer;
use delta_kernel::arrow::datatypes::{DataType as ArrowDataType, Field};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryFromKernel;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine_data::{RowVisitor, TypedGetData};
use delta_kernel::expressions::{column_expr, Expression as Expr, Predicate as Pred};
use delta_kernel::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};
use delta_kernel::{DeltaResult, EngineData, Snapshot};

use test_utils::{create_table, engine_store_setup};

mod common;

/// Describes a test file with its metadata and stats for `create_add_files_with_stats`.
struct TestFileStats<'a> {
    path: &'a str,
    size: i64,
    num_records: i64,
    min_id: i64,
    max_id: i64,
    min_row_id: i64,
    max_row_id: i64,
}

/// Helper to create a field with column mapping metadata
fn field_with_metadata(name: &str, data_type: DataType, field_id: i64) -> StructField {
    StructField::nullable(name, data_type).with_metadata([
        (
            ColumnMetadataKey::ColumnMappingId.as_ref(),
            MetadataValue::Number(field_id),
        ),
        (
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            MetadataValue::String(name.to_string()),
        ),
    ])
}

/// Creates a test schema with field IDs required for content root stats generation
fn create_test_schema_with_field_ids() -> Arc<StructType> {
    Arc::new(
        StructType::try_new(vec![
            field_with_metadata("id", DataType::LONG, 1),
            field_with_metadata("name", DataType::STRING, 2),
            field_with_metadata("_row_id", DataType::LONG, 2147483540),
        ])
        .unwrap(),
    )
}

/// Helper to build a stats struct field (minValues, maxValues, or nullCount)
fn build_stats_struct_field(
    field: &Field,
    id_array: &Int64Array,
    num_files: usize,
) -> DeltaResult<ArrayRef> {
    let ArrowDataType::Struct(schema) = field.data_type() else {
        return Err(delta_kernel::Error::generic(format!(
            "{} should be a struct",
            field.name()
        )));
    };

    let fields: Vec<_> = schema
        .iter()
        .map(|f| {
            let array = if f.name() == "id" {
                Arc::new(id_array.clone()) as ArrayRef
            } else {
                Arc::new(delta_kernel::arrow::array::new_null_array(
                    f.data_type(),
                    num_files,
                ))
            };
            (f.clone(), array)
        })
        .collect();

    Ok(Arc::new(StructArray::from(fields)))
}

fn build_min_max_values_field(
    field: &Field,
    id_array: &Int64Array,
    row_id_array: &Int64Array,
    num_files: usize,
) -> DeltaResult<ArrayRef> {
    let ArrowDataType::Struct(schema) = field.data_type() else {
        return Err(delta_kernel::Error::generic(format!(
            "{} should be a struct",
            field.name()
        )));
    };

    let fields: Vec<_> = schema
        .iter()
        .map(|f| {
            let array = if f.name() == "id" {
                Arc::new(id_array.clone()) as ArrayRef
            } else if f.name() == "_row_id" {
                Arc::new(row_id_array.clone()) as ArrayRef
            } else {
                Arc::new(delta_kernel::arrow::array::new_null_array(
                    f.data_type(),
                    num_files,
                ))
            };
            (f.clone(), array)
        })
        .collect();

    Ok(Arc::new(StructArray::from(fields)))
}

/// Creates add file metadata with stats for testing
fn create_add_files_with_stats(
    add_files_schema: &Arc<StructType>,
    files: Vec<TestFileStats<'_>>,
) -> DeltaResult<Box<dyn EngineData>> {
    let num_files = files.len();

    // Build basic arrays
    let path_array = StringArray::from(files.iter().map(|f| f.path).collect::<Vec<_>>());
    let size_array = Int64Array::from(files.iter().map(|f| f.size).collect::<Vec<_>>());
    let mod_time_array = Int64Array::from(vec![0i64; num_files]);
    let num_records_array =
        Int64Array::from(files.iter().map(|f| f.num_records).collect::<Vec<_>>());
    let min_id_array = Int64Array::from(files.iter().map(|f| f.min_id).collect::<Vec<_>>());
    let max_id_array = Int64Array::from(files.iter().map(|f| f.max_id).collect::<Vec<_>>());
    let min_row_id_array = Int64Array::from(files.iter().map(|f| f.min_row_id).collect::<Vec<_>>());
    let max_row_id_array = Int64Array::from(files.iter().map(|f| f.max_row_id).collect::<Vec<_>>());

    // Create empty partition values map
    let partition_values_array = Arc::new(MapArray::new(
        Arc::new(Field::new(
            "key_value",
            ArrowDataType::Struct(
                vec![
                    Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                    Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                ]
                .into(),
            ),
            false,
        )),
        OffsetBuffer::from_lengths(vec![0; num_files]),
        StructArray::from(vec![
            (
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as ArrayRef,
            ),
        ]),
        None,
        false,
    ));

    // Build stats struct from schema
    let arrow_schema: delta_kernel::arrow::datatypes::Schema =
        TryFromKernel::try_from_kernel(add_files_schema.as_ref())?;
    let stats_field = arrow_schema
        .field_with_name("stats")
        .map_err(|_| delta_kernel::Error::generic("stats field should exist"))?;
    let ArrowDataType::Struct(stats_schema) = stats_field.data_type() else {
        return Err(delta_kernel::Error::generic(
            "stats field should be a struct",
        ));
    };

    let stats_fields: DeltaResult<Vec<_>> = stats_schema
        .iter()
        .map(|field| {
            let array: ArrayRef = match field.name().as_str() {
                "numRecords" => Arc::new(num_records_array.clone()),
                "tightBounds" => Arc::new(BooleanArray::from(vec![Some(true); num_files])),
                "minValues" => {
                    build_min_max_values_field(field, &min_id_array, &min_row_id_array, num_files)?
                }
                "maxValues" => {
                    build_min_max_values_field(field, &max_id_array, &max_row_id_array, num_files)?
                }
                "nullCount" => build_stats_struct_field(
                    field,
                    &Int64Array::from(vec![0i64; num_files]),
                    num_files,
                )?,
                _ => Arc::new(delta_kernel::arrow::array::new_null_array(
                    field.data_type(),
                    num_files,
                )),
            };
            Ok((field.clone(), array))
        })
        .collect();

    let batch = delta_kernel::arrow::array::RecordBatch::try_new(
        Arc::new(arrow_schema),
        vec![
            Arc::new(path_array) as ArrayRef,
            partition_values_array as ArrayRef,
            Arc::new(size_array) as ArrayRef,
            Arc::new(mod_time_array) as ArrayRef,
            Arc::new(StructArray::from(stats_fields?)) as ArrayRef,
        ],
    )
    .map_err(|e| delta_kernel::Error::generic(format!("Failed to create batch: {}", e)))?;

    Ok(Box::new(ArrowEngineData::new(batch)))
}

/// Helper to verify stats in add file data
fn verify_stats(
    data: &dyn EngineData,
    expected: &HashMap<String, (i64, i64, i64, i64, i64)>, // path -> (num_records, min_id, max_id, min_row_id, max_row_id)
) -> DeltaResult<()> {
    use delta_kernel::expressions::ColumnName;

    struct StatsVisitor<'a> {
        expected: &'a HashMap<String, (i64, i64, i64, i64, i64)>,
        verified: usize,
    }

    impl<'a> RowVisitor for StatsVisitor<'a> {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            use std::sync::LazyLock;
            static NAMES: LazyLock<Vec<ColumnName>> = LazyLock::new(|| {
                vec![
                    ColumnName::new(["path"]),
                    ColumnName::new(["stats", "numRecords"]),
                    ColumnName::new(["stats", "minValues", "id"]),
                    ColumnName::new(["stats", "maxValues", "id"]),
                    ColumnName::new(["stats", "minValues", "_row_id"]),
                    ColumnName::new(["stats", "maxValues", "_row_id"]),
                ]
            });
            static TYPES: LazyLock<Vec<DataType>> = LazyLock::new(|| {
                vec![
                    DataType::STRING,
                    DataType::LONG,
                    DataType::LONG,
                    DataType::LONG,
                    DataType::LONG,
                    DataType::LONG,
                ]
            });
            (NAMES.as_slice(), TYPES.as_slice())
        }

        fn visit<'b>(
            &mut self,
            row_count: usize,
            getters: &[&'b dyn delta_kernel::engine_data::GetData<'b>],
        ) -> DeltaResult<()> {
            for i in 0..row_count {
                let path: String = getters[0].get(i, "path")?;
                let (
                    expected_records,
                    expected_min,
                    expected_max,
                    expected_min_row_id,
                    expected_max_row_id,
                ) = self.expected.get(&path).ok_or_else(|| {
                    delta_kernel::Error::generic(format!("Unexpected file: {}", path))
                })?;

                let num_records: i64 = getters[1].get(i, "stats.numRecords")?;
                let min_id: i64 = getters[2].get(i, "stats.minValues.id")?;
                let max_id: i64 = getters[3].get(i, "stats.maxValues.id")?;
                let min_row_id: i64 = getters[4].get(i, "stats.minValues._row_id")?;
                let max_row_id: i64 = getters[5].get(i, "stats.maxValues._row_id")?;
                assert_eq!(num_records, *expected_records);
                assert_eq!(min_id, *expected_min);
                assert_eq!(max_id, *expected_max);
                assert_eq!(min_row_id, *expected_min_row_id);
                assert_eq!(max_row_id, *expected_max_row_id);
                self.verified += 1;
            }
            Ok(())
        }
    }

    let mut visitor = StatsVisitor {
        expected,
        verified: 0,
    };
    visitor.visit_rows_of(data)?;
    assert_eq!(
        visitor.verified,
        expected.len(),
        "Should verify all expected files"
    );
    Ok(())
}

/// Helper to create and add a leaf node with files
fn add_leaf_with_files(
    mc: &mut delta_kernel::transaction::ManifestCommitState,
    engine: &dyn delta_kernel::Engine,
    add_files_schema: &Arc<StructType>,
    files: Vec<TestFileStats<'_>>,
) -> DeltaResult<()> {
    let mut leaf = mc.new_leaf_node_writer(engine)?;
    let data = create_add_files_with_stats(add_files_schema, files)?;
    leaf.add_files(engine, data)?;
    mc.add_leaf(leaf.finish(engine)?)?;
    Ok(())
}

/// Helper to count scan metadata batches and files
fn count_scan_metadata_and_files(
    scan: delta_kernel::scan::Scan,
    engine: &dyn delta_kernel::Engine,
) -> DeltaResult<(usize, usize)> {
    let mut batches = 0;
    let mut files = 0;
    for metadata in scan.scan_metadata(engine)? {
        batches += 1;
        files = metadata?.visit_scan_files(files, |count: &mut usize, _| *count += 1)?;
    }
    Ok((batches, files))
}

#[tokio::test]
async fn test_manifest_level_data_skipping_e2e() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    // Setup table with AMT features
    let (store, engine, table_url) = engine_store_setup("manifest_skipping_e2e", None);
    let engine = Arc::new(engine);
    let schema = create_test_schema_with_field_ids();

    create_table(
        store,
        table_url.clone(),
        schema.clone(),
        &[],
        true,
        vec!["columnMapping", "metadataTree-experimental"],
        vec!["columnMapping", "metadataTree-experimental"],
    )
    .await?;

    // Create transaction with manifest commit
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;

    // Build add files schema with full stats
    let stats_schema = txn.stats_schema()?;
    let add_files_schema = {
        use delta_kernel::schema::StructTypeBuilder;
        let mut builder = StructTypeBuilder::new();
        for field in txn.add_files_schema().fields() {
            if field.name() == "stats" {
                builder = builder.add_field(StructField::nullable(
                    "stats",
                    DataType::Struct(Box::new((*stats_schema).clone())),
                ));
            } else {
                builder = builder.add_field(field.clone());
            }
        }
        builder.build_arc_unchecked()
    };

    // Define test files with distinct ID ranges
    let (file1, file2, file3, file4) = (
        "part-00001.parquet",
        "part-00002.parquet",
        "part-00003.parquet",
        "part-00004.parquet",
    );

    // Add files to leaves: leaf1 (files 1-2), leaf2 (file3)
    {
        let mc = txn.with_manifest_commit();
        add_leaf_with_files(
            mc,
            engine.as_ref(),
            &add_files_schema,
            vec![
                TestFileStats {
                    path: file1,
                    size: 100,
                    num_records: 100,
                    min_id: 1,
                    max_id: 100,
                    min_row_id: 5,
                    max_row_id: 20,
                },
                TestFileStats {
                    path: file2,
                    size: 100,
                    num_records: 200,
                    min_id: 101,
                    max_id: 200,
                    min_row_id: 105,
                    max_row_id: 120,
                },
            ],
        )?;

        add_leaf_with_files(
            mc,
            engine.as_ref(),
            &add_files_schema,
            vec![TestFileStats {
                path: file3,
                size: 100,
                num_records: 100,
                min_id: 201,
                max_id: 300,
                min_row_id: 205,
                max_row_id: 220,
            }],
        )?;
    }

    // Verify stats in leaf data
    let leaf1_data = create_add_files_with_stats(
        &add_files_schema,
        vec![
            TestFileStats {
                path: file1,
                size: 100,
                num_records: 100,
                min_id: 1,
                max_id: 100,
                min_row_id: 5,
                max_row_id: 20,
            },
            TestFileStats {
                path: file2,
                size: 100,
                num_records: 200,
                min_id: 101,
                max_id: 200,
                min_row_id: 105,
                max_row_id: 120,
            },
        ],
    )?;
    verify_stats(
        leaf1_data.as_ref(),
        &HashMap::from([
            (file1.to_string(), (100, 1, 100, 5, 20)),
            (file2.to_string(), (200, 101, 200, 105, 120)),
        ]),
    )?;

    // Add file directly to root
    let root_file_data = create_add_files_with_stats(
        &add_files_schema,
        vec![TestFileStats {
            path: file4,
            size: 100,
            num_records: 400,
            min_id: 301,
            max_id: 400,
            min_row_id: 305,
            max_row_id: 320,
        }],
    )?;
    verify_stats(
        root_file_data.as_ref(),
        &HashMap::from([(file4.to_string(), (400, 301, 400, 305, 320))]),
    )?;
    txn.add_files(root_file_data);

    // Commit and verify content root
    assert!(matches!(
        txn.commit(engine.as_ref())?,
        delta_kernel::transaction::CommitResult::CommittedTransaction(_)
    ));
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert!(
        snapshot.checkpoint_action().is_some(),
        "Content root should exist"
    );

    // Test scan with filtering (id < 50)
    let predicate = Arc::new(Pred::lt(column_expr!("id"), Expr::literal(50i64)));
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_predicate(predicate.clone())
        .build()?;

    // Collect scanned files
    let mut scanned_files = std::collections::HashSet::new();
    for metadata in scan.scan_metadata(engine.as_ref())? {
        scanned_files = metadata?.visit_scan_files(
            scanned_files,
            |files: &mut std::collections::HashSet<_>, file| {
                files.insert(file.path.to_string());
            },
        )?;
    }

    // Verify filtering with id < 50:
    // - file1 (IDs 1-100): INCLUDED (overlaps with <50)
    // - file2 (IDs 101-200): FILTERED OUT by file-level skipping (min=101 > 50)
    // - file3 (IDs 201-300): FILTERED OUT by manifest-level skipping (leaf2 filtered)
    // - file4 (IDs 301-400): FILTERED OUT by file-level skipping (min=301 > 50)
    let expected_files: std::collections::HashSet<_> = [file1.to_string()].into_iter().collect();
    assert_eq!(scanned_files, expected_files);

    // Verify using multi-phase scan planning counts
    // TODO: Replace metadata batch count with ManifestReferences count once exposed in multi-phase planning API
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let unfiltered_scan = snapshot.scan_builder().build()?;
    let (unfiltered_batches, unfiltered_files) =
        count_scan_metadata_and_files(unfiltered_scan, engine.as_ref())?;
    assert_eq!(
        (unfiltered_batches, unfiltered_files),
        (3, 4),
        "Without filter: 3 batches (2 leaves + root), 4 files"
    );

    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let filtered_scan = snapshot.scan_builder().with_predicate(predicate).build()?;
    let (filtered_batches, filtered_files) =
        count_scan_metadata_and_files(filtered_scan, engine.as_ref())?;

    // With predicate id < 50:
    // - 1 batch: leaf1 (only file1 after filtering)
    // - leaf2 filtered at manifest level (min=201 > 50)
    // - file2 filtered at file level (min=101 > 50)
    // - file4 filtered at file level (min=301 > 50)
    assert_eq!(
        (filtered_batches, filtered_files),
        (1, 1),
        "All data skipping works: only file1 (IDs 1-100) overlaps with id < 50"
    );

    // Test scan with filtering on metadata column (_row_id < 10)
    let metadata_predicate = Arc::new(Pred::lt(column_expr!("_row_id"), Expr::literal(10i64)));
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let metadata_scan = snapshot
        .clone()
        .scan_builder()
        .with_predicate(metadata_predicate.clone())
        .build()?;

    // Collect scanned files
    let mut scanned_files = std::collections::HashSet::new();
    for metadata in metadata_scan.scan_metadata(engine.as_ref())? {
        scanned_files = metadata?.visit_scan_files(
            scanned_files,
            |files: &mut std::collections::HashSet<_>, file| {
                files.insert(file.path.to_string());
            },
        )?;
    }

    // Verify filtering with _row_id < 10:
    // - file1 (row IDs 5-20): INCLUDED (overlaps with <10)
    // - file2 (row IDs 105-120): FILTERED OUT by file-level skipping (min=105 > 10)
    // - file3 (row IDs 205-220): FILTERED OUT by manifest-level skipping (leaf2 filtered)
    // - file4 (row IDs 305-320): FILTERED OUT by file-level skipping (min=305 > 50)
    let expected_files: std::collections::HashSet<_> = [file1.to_string()].into_iter().collect();
    assert_eq!(scanned_files, expected_files);

    // Verify using multi-phase scan planning counts
    // TODO: Replace metadata batch count with ManifestReferences count once exposed in multi-phase planning API
    let unfiltered_metadata_scan = snapshot.clone().scan_builder().build()?;
    let (unfiltered_metadata_batches, unfiltered_files) =
        count_scan_metadata_and_files(unfiltered_metadata_scan, engine.as_ref())?;
    assert_eq!(
        (unfiltered_metadata_batches, unfiltered_files),
        (3, 4),
        "Without filter: 3 batches (2 leaves + root), 4 files"
    );

    let snapshot = Snapshot::builder_for(table_url).build(engine.as_ref())?;
    let filtered_scan = snapshot
        .scan_builder()
        .with_predicate(metadata_predicate)
        .build()?;
    let (filtered_batches, filtered_files) =
        count_scan_metadata_and_files(filtered_scan, engine.as_ref())?;

    // With predicate _row_id < 10:
    // - 1 batch: leaf1 (only file1 after filtering)
    // - leaf2 filtered at manifest level (min=205 > 10)
    // - file2 filtered at file level (min=105 > 10)
    // - file4 filtered at file level (min=305 > 10)
    assert_eq!(
        (filtered_batches, filtered_files),
        (1, 1),
        "All data skipping works: only file1 (row IDs 5-20) overlaps with _row_id < 10"
    );

    Ok(())
}
