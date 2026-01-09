use std::path::PathBuf;
use std::sync::Arc;

use crate::arrow::array::BooleanArray;
use crate::arrow::compute::filter_record_batch;
use crate::arrow::record_batch::RecordBatch;
use crate::engine::arrow_data::ArrowEngineData;
use crate::engine::sync::SyncEngine;
use crate::expressions::{column_expr, column_pred, Expression as Expr, Predicate as Pred};
use crate::scan::state::ScanFile;
use crate::schema::{ColumnMetadataKey, DataType, StructField, StructType};
use crate::{EngineData, Snapshot};

use super::*;

#[test]
fn test_static_skipping() {
    const NULL: Pred = Pred::null_literal();
    let test_cases = [
        (false, column_pred!("a")),
        (true, Pred::literal(false)),
        (false, Pred::literal(true)),
        (true, NULL),
        (true, Pred::and(column_pred!("a"), Pred::literal(false))),
        (false, Pred::or(column_pred!("a"), Pred::literal(true))),
        (false, Pred::or(column_pred!("a"), Pred::literal(false))),
        (false, Pred::lt(column_expr!("a"), Expr::literal(10))),
        (false, Pred::lt(Expr::literal(10), Expr::literal(100))),
        (true, Pred::gt(Expr::literal(10), Expr::literal(100))),
        (true, Pred::and(NULL, column_pred!("a"))),
    ];
    for (should_skip, predicate) in test_cases {
        assert_eq!(
            can_statically_skip_all_files(&predicate),
            should_skip,
            "Failed for predicate: {predicate:#?}"
        );
    }
}

#[test]
fn test_physical_predicate() {
    let logical_schema = StructType::new_unchecked(vec![
        StructField::nullable("a", DataType::LONG),
        StructField::nullable("b", DataType::LONG).with_metadata([(
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            "phys_b",
        )]),
        StructField::nullable("phys_b", DataType::LONG).with_metadata([(
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            "phys_c",
        )]),
        StructField::nullable(
            "nested",
            StructType::new_unchecked(vec![
                StructField::nullable("x", DataType::LONG),
                StructField::nullable("y", DataType::LONG).with_metadata([(
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    "phys_y",
                )]),
            ]),
        ),
        StructField::nullable(
            "mapped",
            StructType::new_unchecked(vec![StructField::nullable("n", DataType::LONG)
                .with_metadata([(
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    "phys_n",
                )])]),
        )
        .with_metadata([(
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            "phys_mapped",
        )]),
    ]);

    // NOTE: We break several column mapping rules here because they don't matter for this
    // test. For example, we do not provide field ids, and not all columns have physical names.
    let test_cases = [
        (Pred::literal(true), Some(PhysicalPredicate::None)),
        (Pred::literal(false), Some(PhysicalPredicate::StaticSkipAll)),
        (column_pred!("x"), None), // no such column
        (
            column_pred!("a"),
            Some(PhysicalPredicate::Some(
                column_pred!("a").into(),
                StructType::new_unchecked(vec![StructField::nullable("a", DataType::LONG)]).into(),
            )),
        ),
        (
            column_pred!("b"),
            Some(PhysicalPredicate::Some(
                column_pred!("phys_b").into(),
                StructType::new_unchecked(vec![StructField::nullable("phys_b", DataType::LONG)
                    .with_metadata([(
                        ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                        "phys_b",
                    )])])
                .into(),
            )),
        ),
        (
            column_pred!("nested.x"),
            Some(PhysicalPredicate::Some(
                column_pred!("nested.x").into(),
                StructType::new_unchecked(vec![StructField::nullable(
                    "nested",
                    StructType::new_unchecked(vec![StructField::nullable("x", DataType::LONG)]),
                )])
                .into(),
            )),
        ),
        (
            column_pred!("nested.y"),
            Some(PhysicalPredicate::Some(
                column_pred!("nested.phys_y").into(),
                StructType::new_unchecked(vec![StructField::nullable(
                    "nested",
                    StructType::new_unchecked(vec![StructField::nullable(
                        "phys_y",
                        DataType::LONG,
                    )
                    .with_metadata([(
                        ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                        "phys_y",
                    )])]),
                )])
                .into(),
            )),
        ),
        (
            column_pred!("mapped.n"),
            Some(PhysicalPredicate::Some(
                column_pred!("phys_mapped.phys_n").into(),
                StructType::new_unchecked(vec![StructField::nullable(
                    "phys_mapped",
                    StructType::new_unchecked(vec![StructField::nullable(
                        "phys_n",
                        DataType::LONG,
                    )
                    .with_metadata([(
                        ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                        "phys_n",
                    )])]),
                )
                .with_metadata([(
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    "phys_mapped",
                )])])
                .into(),
            )),
        ),
        (
            Pred::and(column_pred!("mapped.n"), Pred::literal(true)),
            Some(PhysicalPredicate::Some(
                Pred::and(column_pred!("phys_mapped.phys_n"), Pred::literal(true)).into(),
                StructType::new_unchecked(vec![StructField::nullable(
                    "phys_mapped",
                    StructType::new_unchecked(vec![StructField::nullable(
                        "phys_n",
                        DataType::LONG,
                    )
                    .with_metadata([(
                        ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                        "phys_n",
                    )])]),
                )
                .with_metadata([(
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    "phys_mapped",
                )])])
                .into(),
            )),
        ),
        (
            Pred::and(column_pred!("mapped.n"), Pred::literal(false)),
            Some(PhysicalPredicate::StaticSkipAll),
        ),
    ];

    for (predicate, expected) in test_cases {
        let result =
            PhysicalPredicate::try_new(&predicate, &logical_schema, ColumnMappingMode::Name).ok();
        assert_eq!(
            result, expected,
            "Failed for predicate: {predicate:#?}, expected {expected:#?}, got {result:#?}"
        );
    }
}

pub(crate) fn get_files_for_scan(scan: Scan, engine: &dyn Engine) -> DeltaResult<Vec<String>> {
    let scan_metadata_iter = scan.scan_metadata(engine)?;
    fn scan_metadata_callback(paths: &mut Vec<String>, scan_file: ScanFile) {
        paths.push(scan_file.path.to_string());
        assert!(scan_file.dv_info.deletion_vector.is_none());
    }
    let mut files = vec![];
    for res in scan_metadata_iter {
        let scan_metadata = res?;
        files = scan_metadata.visit_scan_files(files, scan_metadata_callback)?;
    }
    Ok(files)
}

#[test]
fn test_scan_metadata_paths() {
    let path =
        std::fs::canonicalize(PathBuf::from("./tests/data/table-without-dv-small/")).unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = SyncEngine::new();

    let snapshot = Snapshot::builder_for(url).build(&engine).unwrap();
    let scan = snapshot.scan_builder().build().unwrap();
    let files = get_files_for_scan(scan, &engine).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(
        files[0],
        "part-00000-517f5d32-9c95-48e8-82b4-0229cc194867-c000.snappy.parquet"
    );
}

#[test_log::test]
fn test_scan_metadata() {
    let path =
        std::fs::canonicalize(PathBuf::from("./tests/data/table-without-dv-small/")).unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = Arc::new(SyncEngine::new());

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref()).unwrap();
    let scan = snapshot.scan_builder().build().unwrap();
    let files: Vec<Box<dyn EngineData>> = scan.execute(engine).unwrap().try_collect().unwrap();

    assert_eq!(files.len(), 1);
    let num_rows = files[0].as_ref().len();
    assert_eq!(num_rows, 10)
}

#[test_log::test]
fn test_scan_metadata_from_same_version() {
    let path =
        std::fs::canonicalize(PathBuf::from("./tests/data/table-without-dv-small/")).unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = Arc::new(SyncEngine::new());

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref()).unwrap();
    let version = snapshot.version();
    let scan = snapshot.scan_builder().build().unwrap();
    let files: Vec<_> = scan
        .scan_metadata(engine.as_ref())
        .unwrap()
        .map_ok(|ScanMetadata { scan_files, .. }| {
            let (underlying_data, selection_vector) = scan_files.into_parts();
            let batch: RecordBatch = ArrowEngineData::try_from_engine_data(underlying_data)
                .unwrap()
                .into();
            let filtered_batch =
                filter_record_batch(&batch, &BooleanArray::from(selection_vector)).unwrap();
            Box::new(ArrowEngineData::from(filtered_batch)) as Box<dyn EngineData>
        })
        .try_collect()
        .unwrap();
    let new_files: Vec<_> = scan
        .scan_metadata_from(engine.as_ref(), version, files, None)
        .unwrap()
        .try_collect()
        .unwrap();

    assert_eq!(new_files.len(), 1);
}

// reading v0 with 3 files.
// updating to v1 with 3 more files added.
#[test_log::test]
fn test_scan_metadata_from_with_update() {
    let path = std::fs::canonicalize(PathBuf::from("./tests/data/basic_partitioned/")).unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let engine = Arc::new(SyncEngine::new());

    let snapshot = Snapshot::builder_for(url.clone())
        .at_version(0)
        .build(engine.as_ref())
        .unwrap();
    let scan = snapshot.scan_builder().build().unwrap();
    let files: Vec<_> = scan
        .scan_metadata(engine.as_ref())
        .unwrap()
        .map_ok(|ScanMetadata { scan_files, .. }| {
            let (underlying_data, selection_vector) = scan_files.into_parts();
            let batch: RecordBatch = ArrowEngineData::try_from_engine_data(underlying_data)
                .unwrap()
                .into();
            filter_record_batch(&batch, &BooleanArray::from(selection_vector)).unwrap()
        })
        .try_collect()
        .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].num_rows(), 3);

    let files: Vec<_> = files
        .into_iter()
        .map(|b| Box::new(ArrowEngineData::from(b)) as Box<dyn EngineData>)
        .collect();
    let snapshot = Snapshot::builder_for(url)
        .at_version(1)
        .build(engine.as_ref())
        .unwrap();
    let scan = snapshot.scan_builder().build().unwrap();
    let new_files: Vec<_> = scan
        .scan_metadata_from(engine.as_ref(), 0, files, None)
        .unwrap()
        .map_ok(|ScanMetadata { scan_files, .. }| {
            let (underlying_data, selection_vector) = scan_files.into_parts();
            let batch: RecordBatch = ArrowEngineData::try_from_engine_data(underlying_data)
                .unwrap()
                .into();
            filter_record_batch(&batch, &BooleanArray::from(selection_vector)).unwrap()
        })
        .try_collect()
        .unwrap();
    assert_eq!(new_files.len(), 2);
    assert_eq!(new_files[0].num_rows(), 3);
    assert_eq!(new_files[1].num_rows(), 3);
}

#[test]
fn test_get_partition_value() {
    let cases = [
        (
            "string",
            PrimitiveType::String,
            Scalar::String("string".to_string()),
        ),
        ("123", PrimitiveType::Integer, Scalar::Integer(123)),
        ("1234", PrimitiveType::Long, Scalar::Long(1234)),
        ("12", PrimitiveType::Short, Scalar::Short(12)),
        ("1", PrimitiveType::Byte, Scalar::Byte(1)),
        ("1.1", PrimitiveType::Float, Scalar::Float(1.1)),
        ("10.10", PrimitiveType::Double, Scalar::Double(10.1)),
        ("true", PrimitiveType::Boolean, Scalar::Boolean(true)),
        ("2024-01-01", PrimitiveType::Date, Scalar::Date(19723)),
        ("1970-01-01", PrimitiveType::Date, Scalar::Date(0)),
        (
            "1970-01-01 00:00:00",
            PrimitiveType::Timestamp,
            Scalar::Timestamp(0),
        ),
        (
            "1970-01-01 00:00:00.123456",
            PrimitiveType::Timestamp,
            Scalar::Timestamp(123456),
        ),
        (
            "1970-01-01 00:00:00.123456789",
            PrimitiveType::Timestamp,
            Scalar::Timestamp(123456),
        ),
    ];

    for (raw, data_type, expected) in &cases {
        let value = crate::transforms::parse_partition_value_raw(
            Some(&raw.to_string()),
            &DataType::Primitive(data_type.clone()),
        )
        .unwrap();
        assert_eq!(value, *expected);
    }
}

#[test]
fn test_replay_for_scan_metadata() {
    let path = std::fs::canonicalize(PathBuf::from("./tests/data/parquet_row_group_skipping/"));
    let url = url::Url::from_directory_path(path.unwrap()).unwrap();
    let engine = SyncEngine::new();

    let snapshot = Snapshot::builder_for(url).build(&engine).unwrap();
    let scan = snapshot.scan_builder().build().unwrap();
    let data: Vec<_> = scan
        .replay_for_scan_metadata(&engine)
        .unwrap()
        .try_collect()
        .unwrap();
    // No predicate pushdown attempted, because at most one part of a multi-part checkpoint
    // could be skipped when looking for adds/removes.
    //
    // NOTE: Each checkpoint part is a single-row file -- guaranteed to produce one row group.
    assert_eq!(data.len(), 5);
}

#[test]
fn test_data_row_group_skipping() {
    let path = std::fs::canonicalize(PathBuf::from("./tests/data/parquet_row_group_skipping/"));
    let url = url::Url::from_directory_path(path.unwrap()).unwrap();
    let engine = Arc::new(SyncEngine::new());

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref()).unwrap();

    // No predicate pushdown attempted, so the one data file should be returned.
    //
    // NOTE: The data file contains only five rows -- near guaranteed to produce one row group.
    let scan = snapshot.clone().scan_builder().build().unwrap();
    let data: Vec<_> = scan.execute(engine.clone()).unwrap().try_collect().unwrap();
    assert_eq!(data.len(), 1);

    // Ineffective predicate pushdown attempted, so the one data file should be returned.
    let int_col = column_expr!("numeric.ints.int32");
    let value = Expr::literal(1000i32);
    let predicate = Arc::new(int_col.clone().gt(value.clone()));
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_predicate(predicate)
        .build()
        .unwrap();
    let data: Vec<_> = scan.execute(engine.clone()).unwrap().try_collect().unwrap();
    assert_eq!(data.len(), 1);

    // TODO(#860): we disable predicate pushdown until we support row indexes. Update this test
    // accordingly after support is reintroduced.
    //
    // Effective predicate pushdown, so no data files should be returned. BUT since we disabled
    // predicate pushdown, the one data file is still returned.
    let predicate = Arc::new(int_col.lt(value));
    let scan = snapshot
        .scan_builder()
        .with_predicate(predicate)
        .build()
        .unwrap();
    let data: Vec<_> = scan.execute(engine).unwrap().try_collect().unwrap();
    assert_eq!(data.len(), 1);
}

#[test]
fn test_missing_column_row_group_skipping() {
    let path = std::fs::canonicalize(PathBuf::from("./tests/data/parquet_row_group_skipping/"));
    let url = url::Url::from_directory_path(path.unwrap()).unwrap();
    let engine = Arc::new(SyncEngine::new());

    let snapshot = Snapshot::builder_for(url).build(engine.as_ref()).unwrap();

    // Predicate over a logically valid but physically missing column. No data files should be
    // returned because the column is inferred to be all-null.
    //
    // WARNING: https://github.com/delta-io/delta-kernel-rs/issues/434 - This
    // optimization is currently disabled, so the one data file is still returned.
    let predicate = Arc::new(column_expr!("missing").lt(Expr::literal(1000i64)));
    let scan = snapshot
        .clone()
        .scan_builder()
        .with_predicate(predicate)
        .build()
        .unwrap();
    let data: Vec<_> = scan.execute(engine.clone()).unwrap().try_collect().unwrap();
    assert_eq!(data.len(), 1);

    // Predicate over a logically missing column fails the scan
    let predicate = Arc::new(column_expr!("numeric.ints.invalid").lt(Expr::literal(1000)));
    snapshot
        .scan_builder()
        .with_predicate(predicate)
        .build()
        .expect_err("unknown column");
}

#[test_log::test]
fn test_scan_with_checkpoint() -> DeltaResult<()> {
    let path = std::fs::canonicalize(PathBuf::from(
        "./tests/data/with_checkpoint_no_last_checkpoint/",
    ))?;

    let url = url::Url::from_directory_path(path).unwrap();
    let engine = SyncEngine::new();

    let snapshot = Snapshot::builder_for(url).build(&engine).unwrap();
    let scan = snapshot.scan_builder().build()?;
    let files = get_files_for_scan(scan, &engine)?;
    // test case:
    //
    // commit0:     P and M, no add/remove
    // commit1:     add file-ad1
    // commit2:     remove file-ad1, add file-a19
    // checkpoint2: remove file-ad1, add file-a19
    // commit3:     remove file-a19, add file-70b
    //
    // thus replay should produce only file-70b
    assert_eq!(
        files,
        vec!["part-00000-70b1dcdf-0236-4f63-a072-124cdbafd8a0-c000.snappy.parquet"]
    );
    Ok(())
}

#[test]
fn test_replay_for_scan_metadata_with_content_root_contiguous() -> DeltaResult<()> {
    use crate::actions::visitors::AddVisitor;
    use crate::engine::default::DefaultEngine;
    use crate::path::{LogPathFileType, ParsedLogPath};
    use crate::RowVisitor;
    use futures::executor::block_on;
    use object_store::{memory::InMemory, path::Path, ObjectStore};

    // Setup: Create an in-memory store
    let store = Arc::new(InMemory::new());
    let table_root = Url::parse("memory:///").unwrap();
    let log_root = table_root.join("_delta_log/").unwrap();

    // Create initial commit with protocol and metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":1}}
{"metaData":{"id":"testId","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"value\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}
{"add":{"path":"part-v00000.parquet","partitionValues":{},"size":1024,"modificationTime":1677811178336,"dataChange":true}}"#;
    let path0 = Path::from("_delta_log/00000000000000000000.json");
    block_on(async { store.put(&path0, commit0_content.into()).await }).unwrap();

    // Create engine first so we can use it for MetadataWriter
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Create metadata for content_root using MetadataBuilder
    let content_root_url = {
        use crate::actions::Add;
        use crate::metadata::builder::MetadataBuilder;
        use crate::metadata::writer::MetadataWriter;

        let mut builder = MetadataBuilder::new_for(table_root.clone(), 3);

        // Add the action that should be in content_root
        let add = Add {
            path: "part-content-root.parquet".to_string(),
            partition_values: Default::default(),
            size: 2048,
            modification_time: 1677811178336,
            data_change: true,
            stats: None,
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            data_manifest_path: None,
            data_manifest_position: None,
            delete_manifest_path: None,
            delete_manifest_position: None,
        };
        builder.add(add, 3, None)?;

        let metadata = builder.build(engine.as_ref()).unwrap();
        let writer = MetadataWriter::try_new(metadata).unwrap();
        writer.write(engine.as_ref()).unwrap()
    };

    // Create commit files: versions 1, 2, 3, 4, 5
    // Content root is at version 3, so we should only read commits 4 and 5
    for version in 1..=5 {
        let commit_content = if version == 3 {
            // Version 3 has the contentRoot action pointing to the content root file
            format!(
                r#"{{"add":{{"path":"part-v{:05}.parquet","partitionValues":{{}},"size":1024,"modificationTime":1677811178336,"dataChange":true}}}}
{{"contentRoot":{{"path":"{}","sizeInBytes":1024}}}}"#,
                version, content_root_url
            )
        } else {
            format!(
                r#"{{"add":{{"path":"part-v{:05}.parquet","partitionValues":{{}},"size":1024,"modificationTime":1677811178336,"dataChange":true}}}}"#,
                version
            )
        };
        let path = Path::from(format!("_delta_log/{:020}.json", version).as_str());
        block_on(async { store.put(&path, commit_content.into()).await }).unwrap();
    }

    // Create ParsedLogPath objects for commits
    let mut commit_files = vec![];
    for version in 0..=5 {
        let location = log_root.join(&format!("{:020}.json", version)).unwrap();
        commit_files.push(ParsedLogPath {
            location: FileMeta {
                location,
                last_modified: 0,
                size: 100,
            },
            filename: format!("{:020}.json", version),
            extension: "json".to_string(),
            version,
            file_type: LogPathFileType::Commit,
        });
    }

    let latest_commit_file = commit_files.last().cloned();
    let log_segment = crate::log_segment::LogSegment {
        end_version: 5,
        checkpoint_version: None,
        log_root: log_root.clone(),
        ascending_commit_files: commit_files,
        ascending_compaction_files: vec![],
        checkpoint_parts: vec![],
        latest_crc_file: None,
        latest_commit_file,
        checkpoint_schema: None,
    };

    // Create a Snapshot from the log_segment
    let snapshot = Arc::new(crate::snapshot::Snapshot::try_new_from_log_segment(
        table_root.clone(),
        log_segment,
        engine.as_ref(),
        None,
    )?);

    let scan = snapshot.scan_builder().build()?;

    // Call replay_for_scan_metadata and collect all actions
    let action_batches: Vec<_> = scan
        .replay_for_scan_metadata(engine.as_ref())?
        .try_collect()?;

    // Extract all add action paths and track which came from log batches vs content root
    let mut add_paths = vec![];
    let mut log_batch_paths = vec![];
    let mut content_root_paths = vec![];

    for batch in action_batches {
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        for add in visitor.adds {
            add_paths.push(add.path.clone());
            if batch.is_log_batch {
                log_batch_paths.push(add.path);
            } else {
                content_root_paths.push(add.path);
            }
        }
    }

    // Verify we got:
    // 1. Actions from commits 4 and 5 (after content root) with is_log_batch=true
    // 2. Action from content root itself with is_log_batch=false
    // 3. NO actions from commits 0, 1, 2, 3 (at or before content root version)
    assert!(
        add_paths.contains(&"part-v00004.parquet".to_string()),
        "Should have action from commit 4"
    );
    assert!(
        add_paths.contains(&"part-v00005.parquet".to_string()),
        "Should have action from commit 5"
    );
    assert!(
        add_paths.contains(&"part-content-root.parquet".to_string()),
        "Should have action from content root"
    );

    // Verify old commits are NOT included
    assert!(
        !add_paths.contains(&"part-v00000.parquet".to_string()),
        "Should NOT have action from commit 0"
    );
    assert!(
        !add_paths.contains(&"part-v00001.parquet".to_string()),
        "Should NOT have action from commit 1"
    );
    assert!(
        !add_paths.contains(&"part-v00002.parquet".to_string()),
        "Should NOT have action from commit 2"
    );
    assert!(
        !add_paths.contains(&"part-v00003.parquet".to_string()),
        "Should NOT have action from commit 3"
    );

    // Verify is_log_batch flags are correct
    assert!(
        log_batch_paths.contains(&"part-v00004.parquet".to_string()),
        "Commit 4 should have is_log_batch=true"
    );
    assert!(
        log_batch_paths.contains(&"part-v00005.parquet".to_string()),
        "Commit 5 should have is_log_batch=true"
    );
    assert!(
        content_root_paths.contains(&"part-content-root.parquet".to_string()),
        "Content root should have is_log_batch=false"
    );
    assert_eq!(
        log_batch_paths.len(),
        2,
        "Should have exactly 2 actions from log batches"
    );
    assert_eq!(
        content_root_paths.len(),
        1,
        "Should have exactly 1 action from content root"
    );

    Ok(())
}

#[test]
fn test_replay_for_scan_metadata_with_content_root_gaps() -> DeltaResult<()> {
    use crate::actions::visitors::AddVisitor;
    use crate::engine::default::DefaultEngine;
    use crate::path::{LogPathFileType, ParsedLogPath};
    use crate::RowVisitor;
    use futures::executor::block_on;
    use object_store::{memory::InMemory, path::Path, ObjectStore};

    // Setup: Create an in-memory store
    let store = Arc::new(InMemory::new());
    let table_root = Url::parse("memory:///").unwrap();
    let log_root = table_root.join("_delta_log/").unwrap();

    // Create initial commit with protocol and metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":1,"minWriterVersion":1}}
{"metaData":{"id":"testId","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[{\"name\":\"value\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}
{"add":{"path":"part-v00000.parquet","partitionValues":{},"size":1024,"modificationTime":1677811178336,"dataChange":true}}"#;
    let path0 = Path::from("_delta_log/00000000000000000000.json");
    block_on(async { store.put(&path0, commit0_content.into()).await }).unwrap();

    // Create engine
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Create metadata for content_root using MetadataBuilder
    let content_root_url = {
        use crate::actions::Add;
        use crate::metadata::builder::MetadataBuilder;
        use crate::metadata::writer::MetadataWriter;

        let mut builder = MetadataBuilder::new_for(table_root.clone(), 10);

        // Add the action that should be in content_root
        let add = Add {
            path: "part-gap-content-root.parquet".to_string(),
            partition_values: Default::default(),
            size: 2048,
            modification_time: 1677811178336,
            data_change: true,
            stats: None,
            tags: None,
            deletion_vector: None,
            base_row_id: None,
            default_row_commit_version: None,
            clustering_provider: None,
            data_manifest_path: None,
            data_manifest_position: None,
            delete_manifest_path: None,
            delete_manifest_position: None,
        };
        builder.add(add, 10, None)?;

        let metadata = builder.build(engine.as_ref()).unwrap();
        let writer = MetadataWriter::try_new(metadata).unwrap();
        writer.write(engine.as_ref()).unwrap()
    };

    // Create commit files: versions 1, 2, 5, 10, 15, 20
    // Content root is at version 10
    // Commits before version 10 should be ignored (0, 1, 2, 5, 10)
    // Only commits 15 and 20 should be included
    let versions = vec![1, 2, 5, 10, 15, 20];
    for version in &versions {
        let commit_content = if *version == 10 {
            // Version 10 has the contentRoot action pointing to the content root file
            format!(
                r#"{{"add":{{"path":"part-v{:05}.parquet","partitionValues":{{}},"size":1024,"modificationTime":1677811178336,"dataChange":true}}}}
{{"contentRoot":{{"path":"{}","sizeInBytes":1024}}}}"#,
                version, content_root_url
            )
        } else {
            format!(
                r#"{{"add":{{"path":"part-v{:05}.parquet","partitionValues":{{}},"size":1024,"modificationTime":1677811178336,"dataChange":true}}}}"#,
                version
            )
        };
        let path = Path::from(format!("_delta_log/{:020}.json", version).as_str());
        block_on(async { store.put(&path, commit_content.into()).await }).unwrap();
    }

    // Create ParsedLogPath objects for commits (including version 0)
    let all_versions = vec![0, 1, 2, 5, 10, 15, 20];
    let mut commit_files = vec![];
    for version in &all_versions {
        let location = log_root.join(&format!("{:020}.json", version)).unwrap();
        commit_files.push(ParsedLogPath {
            location: FileMeta {
                location,
                last_modified: 0,
                size: 100,
            },
            filename: format!("{:020}.json", version),
            extension: "json".to_string(),
            version: *version,
            file_type: LogPathFileType::Commit,
        });
    }

    let log_segment = crate::log_segment::LogSegment {
        end_version: 20,
        checkpoint_version: None,
        log_root: log_root.clone(),
        ascending_commit_files: commit_files,
        ascending_compaction_files: vec![],
        checkpoint_parts: vec![],
        latest_crc_file: None,
        latest_commit_file: None,
        checkpoint_schema: None,
    };

    // Create a Snapshot from the log_segment
    let snapshot = Arc::new(crate::snapshot::Snapshot::try_new_from_log_segment(
        table_root.clone(),
        log_segment,
        engine.as_ref(),
        None,
    )?);

    let scan = snapshot.scan_builder().build()?;

    // Call replay_for_scan_metadata and collect all actions
    let action_batches: Vec<_> = scan
        .replay_for_scan_metadata(engine.as_ref())?
        .try_collect()?;

    // Extract all add action paths and track which came from log batches vs content root
    let mut add_paths = vec![];
    let mut log_batch_paths = vec![];
    let mut content_root_paths = vec![];

    for batch in action_batches {
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        for add in visitor.adds {
            add_paths.push(add.path.clone());
            if batch.is_log_batch {
                log_batch_paths.push(add.path);
            } else {
                content_root_paths.push(add.path);
            }
        }
    }

    // Verify we got:
    // 1. Actions from commits 15 and 20 (after content root, with gaps) with is_log_batch=true
    // 2. Action from content root itself with is_log_batch=false
    // 3. NO actions from commits 0, 1, 2, 5, 10 (at or before content root version)
    assert!(
        add_paths.contains(&"part-v00015.parquet".to_string()),
        "Should have action from commit 15"
    );
    assert!(
        add_paths.contains(&"part-v00020.parquet".to_string()),
        "Should have action from commit 20"
    );
    assert!(
        add_paths.contains(&"part-gap-content-root.parquet".to_string()),
        "Should have action from content root"
    );

    // Verify old commits are NOT included, even though there are version gaps
    assert!(
        !add_paths.contains(&"part-v00000.parquet".to_string()),
        "Should NOT have action from commit 0"
    );
    assert!(
        !add_paths.contains(&"part-v00001.parquet".to_string()),
        "Should NOT have action from commit 1"
    );
    assert!(
        !add_paths.contains(&"part-v00002.parquet".to_string()),
        "Should NOT have action from commit 2"
    );
    assert!(
        !add_paths.contains(&"part-v00005.parquet".to_string()),
        "Should NOT have action from commit 5"
    );
    assert!(
        !add_paths.contains(&"part-v00010.parquet".to_string()),
        "Should NOT have action from commit 10 (the content root version itself)"
    );

    // Verify we got exactly 3 actions (2 from later commits + 1 from content root)
    assert_eq!(add_paths.len(), 3, "Should have exactly 3 add actions");

    // Verify is_log_batch flags are correct
    assert!(
        log_batch_paths.contains(&"part-v00015.parquet".to_string()),
        "Commit 15 should have is_log_batch=true"
    );
    assert!(
        log_batch_paths.contains(&"part-v00020.parquet".to_string()),
        "Commit 20 should have is_log_batch=true"
    );
    assert!(
        content_root_paths.contains(&"part-gap-content-root.parquet".to_string()),
        "Content root should have is_log_batch=false"
    );
    assert_eq!(
        log_batch_paths.len(),
        2,
        "Should have exactly 2 actions from log batches"
    );
    assert_eq!(
        content_root_paths.len(),
        1,
        "Should have exactly 1 action from content root"
    );

    Ok(())
}
