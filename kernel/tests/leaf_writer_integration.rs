//! Integration tests for LeafNodeWriter transaction flow.
//!
//! These tests verify that the LeafNodeWriter correctly writes data files to leaf manifests
//! and that recursive manifest loading works properly in the scan path.
//!
//! ## Test Coverage
//!
//! - TXN-1: Basic transaction write with leaf writer
//! - TXN-2: Multiple leaves in one transaction
//! - TXN-5: Sequential commits with leaf writers

use std::collections::HashSet;
use std::sync::Arc;

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::engine_data::{GetData, TypedGetData};
use delta_kernel::object_store::ObjectStore;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::transaction::CommitResult;
use delta_kernel::{DeltaResult, RowVisitor, Snapshot};
use test_utils::create_add_files_metadata;
use url::Url;

/// Helper to setup test tables with column mapping enabled (id mode)
async fn setup_test_tables_with_column_mapping(
    schema: SchemaRef,
    partition_columns: &[&str],
    table_base_name: &str,
) -> Result<
    Vec<(
        Url,
        DefaultEngine<TokioBackgroundExecutor>,
        Arc<dyn ObjectStore>,
        &'static str,
    )>,
    Box<dyn std::error::Error>,
> {
    use test_utils::{create_table, engine_store_setup};

    let table_name_37 = format!("{table_base_name}_37");
    let (store_37, engine_37, table_location_37) = engine_store_setup(table_name_37.as_str(), None);

    Ok(vec![(
        create_table(
            store_37.clone(),
            table_location_37,
            schema.clone(),
            partition_columns,
            true,
            vec!["columnMapping", "metadataTree-experimental"],
            vec!["columnMapping", "metadataTree-experimental"],
        )
        .await?,
        engine_37,
        store_37,
        "test_table_37_with_column_mapping",
    )])
}

/// Create the standard test schema (id INTEGER, value STRING) with column mapping for id mode
/// TODO: Both mapping IDs should not be needed here, look at flow more closely.
fn create_test_schema() -> Result<Arc<StructType>, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER).with_metadata([
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
                MetadataValue::String("col-1".to_string()),
            ),
        ]),
        StructField::nullable("value", DataType::STRING).with_metadata([
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
                MetadataValue::String("col-2".to_string()),
            ),
        ]),
    ])?))
}

/// Information collected from scanning a snapshot
#[derive(Debug)]
struct ScannedFiles {
    /// Unique file paths found in scan
    file_paths: HashSet<String>,
    /// File paths with deletion vectors
    files_with_dvs: HashSet<String>,
}

/// Helper to collect all files from a snapshot via scan
fn collect_scanned_files(
    snapshot: Arc<Snapshot>,
    engine: &dyn delta_kernel::Engine,
) -> DeltaResult<ScannedFiles> {
    use delta_kernel::expressions::ColumnName;

    struct FileCollector<'a> {
        file_paths: HashSet<String>,
        files_with_dvs: HashSet<String>,
        selection_vector: &'a [bool],
    }

    impl<'a> RowVisitor for FileCollector<'a> {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            use std::sync::LazyLock;

            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["path"]),
                            ColumnName::new(["deletionVector", "storageType"]),
                        ],
                        vec![DataType::STRING, DataType::STRING],
                    )
                });
            (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
        }

        fn visit<'b>(
            &mut self,
            row_count: usize,
            getters: &[&'b dyn GetData<'b>],
        ) -> DeltaResult<()> {
            for i in 0..row_count {
                // Skip rows not selected by the selection vector
                if i < self.selection_vector.len() && !self.selection_vector[i] {
                    continue;
                }

                let path: String = getters[0].get(i, "path")?;

                // Check if this is a duplicate - insert returns false if already present
                if !self.file_paths.insert(path.clone()) {
                    return Err(delta_kernel::Error::generic(format!(
                        "Duplicate file path '{}' found in scan. Each file should appear exactly once.",
                        path
                    )));
                }

                // Check if this file has a deletion vector
                let dv_storage_type: Option<String> =
                    getters[1].get_opt(i, "deletionVector.storageType")?;
                if dv_storage_type.is_some() {
                    self.files_with_dvs.insert(path);
                }
            }
            Ok(())
        }
    }

    let scan = snapshot.scan_builder().build()?;
    let mut all_file_paths = HashSet::new();
    let mut all_files_with_dvs = HashSet::new();

    for scan_metadata_result in scan.scan_metadata(engine)? {
        let scan_metadata = scan_metadata_result?;
        let selection_vector = scan_metadata.scan_files.selection_vector();
        let mut collector = FileCollector {
            file_paths: HashSet::new(),
            files_with_dvs: HashSet::new(),
            selection_vector,
        };
        collector.visit_rows_of(scan_metadata.scan_files.data())?;

        // Merge results
        for path in collector.file_paths {
            if !all_file_paths.insert(path.clone()) {
                return Err(delta_kernel::Error::generic(format!(
                    "Duplicate file path '{}' found across scan batches.",
                    path
                )));
            }
        }
        all_files_with_dvs.extend(collector.files_with_dvs);
    }

    Ok(ScannedFiles {
        file_paths: all_file_paths,
        files_with_dvs: all_files_with_dvs,
    })
}

/// Helper to verify expected files are present with no duplicates
fn verify_scanned_files(
    scanned: &ScannedFiles,
    expected_files: &[&str],
    expected_files_with_dvs: &[&str],
) {
    // Convert expected to HashSet for comparison
    let expected_set: HashSet<String> = expected_files.iter().map(|s| s.to_string()).collect();

    // Verify all expected files are present
    for expected_file in expected_files {
        assert!(
            scanned.file_paths.contains(*expected_file),
            "Expected file '{}' not found in scan. Found files: {:?}",
            expected_file,
            scanned.file_paths
        );
    }

    // Verify no unexpected files (duplicates or extras)
    for scanned_file in &scanned.file_paths {
        assert!(
            expected_set.contains(scanned_file),
            "Unexpected file '{}' found in scan. Expected only: {:?}",
            scanned_file,
            expected_files
        );
    }

    // Verify file count matches (no duplicates)
    assert_eq!(
        scanned.file_paths.len(),
        expected_files.len(),
        "File count mismatch. Expected {}, found {}. This could indicate duplicates.",
        expected_files.len(),
        scanned.file_paths.len()
    );

    // Verify DVs
    for expected_dv_file in expected_files_with_dvs {
        assert!(
            scanned.files_with_dvs.contains(*expected_dv_file),
            "Expected file '{}' to have a deletion vector but it doesn't",
            expected_dv_file
        );
    }

    // Verify DV count
    assert_eq!(
        scanned.files_with_dvs.len(),
        expected_files_with_dvs.len(),
        "DV count mismatch. Expected {}, found {}",
        expected_files_with_dvs.len(),
        scanned.files_with_dvs.len()
    );
}

#[tokio::test]
async fn test_transaction_basic_leaf_write() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store, _name) in
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_basic").await?
    {
        // Step 1: Create transaction with manifest commit enabled
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_operation("WRITE".to_string());
        let add_files_schema = txn.add_files_schema();

        // Step 2-3: Create leaf, add files, and add to manifest commit
        {
            let mc = txn.with_manifest_commit();
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("part-001.parquet", 2048, 1000000, 50),
                    ("part-002.parquet", 3072, 1000001, 75),
                ],
            )?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }

        // Step 4: Commit
        let committed = match txn.commit(&engine)? {
            CommitResult::CommittedTransaction(c) => c,
            other => panic!("Expected success, got {:?}", other),
        };

        let commit_version = committed.commit_version();

        // Step 5: Verify table state via scan
        let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_eq!(new_snapshot.version(), commit_version);

        // Verify files are present with no duplicates
        let scanned = collect_scanned_files(new_snapshot, &engine)?;
        verify_scanned_files(
            &scanned,
            &["part-001.parquet", "part-002.parquet"],
            &[], // No DVs expected
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_transaction_multiple_leaves() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store, _name) in
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_multi_leaves").await?
    {
        // Create transaction with manifest commit enabled
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_operation("WRITE".to_string());
        let add_files_schema = txn.add_files_schema();

        // Create and add 3 leaves with different files
        {
            let mc = txn.with_manifest_commit();
            for i in 0..3 {
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let files = vec![
                    (
                        format!("leaf{}_file1.parquet", i).leak() as &str,
                        1024 + i * 100,
                        1000000 + i,
                        10 + i,
                    ),
                    (
                        format!("leaf{}_file2.parquet", i).leak() as &str,
                        2048 + i * 100,
                        1000010 + i,
                        20 + i,
                    ),
                ];
                let metadata = create_add_files_metadata(add_files_schema, files)?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }
        }

        // Commit
        let _committed = match txn.commit(&engine)? {
            CommitResult::CommittedTransaction(c) => c,
            other => panic!("Expected success, got {:?}", other),
        };

        // Verify via scan - should have 6 unique files (3 leaves * 2 files each)
        let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let scanned = collect_scanned_files(new_snapshot, &engine)?;

        let expected_files = &[
            "leaf0_file1.parquet",
            "leaf0_file2.parquet",
            "leaf1_file1.parquet",
            "leaf1_file2.parquet",
            "leaf2_file1.parquet",
            "leaf2_file2.parquet",
        ];
        verify_scanned_files(&scanned, expected_files, &[]);
    }
    Ok(())
}

#[tokio::test]
async fn test_transaction_sequential_commits() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store, _name) in
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_sequential").await?
    {
        // Commit transaction 1 with files A, B
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?
                .with_operation("WRITE".to_string());
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("fileA.parquet", 1024, 1000000, 10),
                        ("fileB.parquet", 2048, 1000001, 20),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }
            let _committed = match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => c,
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Commit transaction 2 with files C, D
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?
                .with_operation("WRITE".to_string());
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("fileC.parquet", 3072, 1000002, 30),
                        ("fileD.parquet", 4096, 1000003, 40),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }
            let _committed = match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => c,
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Commit transaction 3 with files E, F
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?
                .with_operation("WRITE".to_string());
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("fileE.parquet", 5120, 1000004, 50),
                        ("fileF.parquet", 6144, 1000005, 60),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }
            let _committed = match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => c,
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify final state via scan - all 6 unique files present
        let final_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_eq!(
            final_snapshot.version(),
            3,
            "Should be at version 3 after 3 commits"
        );

        let scanned = collect_scanned_files(final_snapshot, &engine)?;
        verify_scanned_files(
            &scanned,
            &[
                "fileA.parquet",
                "fileB.parquet",
                "fileC.parquet",
                "fileD.parquet",
                "fileE.parquet",
                "fileF.parquet",
            ],
            &[], // No DVs
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_move_files_from_leaf_to_leaf() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store, _name) in
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_move_leaf_to_leaf").await?
    {
        // Commit 0: Create files in a leaf (leaf A)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?
                .with_operation("WRITE".to_string());
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("fileA.parquet", 2048, 1000000, 50),
                        ("fileB.parquet", 3072, 1000001, 75),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(_) => {}
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify files are in leaf A via scan
        let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_eq!(snapshot_v1.version(), 1);
        let scanned = collect_scanned_files(snapshot_v1.clone(), &engine)?;
        verify_scanned_files(&scanned, &["fileA.parquet", "fileB.parquet"], &[]);

        // Commit 1: Move files from leaf A to leaf B
        {
            // Scan to get existing files (stats_parsed column is required by add_existing_actions)
            let scan = snapshot_v1
                .clone()
                .scan_builder()
                .include_all_stats_columns()
                .build()?;

            let mut txn = snapshot_v1
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?
                .with_operation("OPTIMIZE".to_string());

            let mut scan_metadata_iter = scan.scan_metadata(&engine)?;

            // Get the first (and only) scan metadata batch
            let scan_metadata = scan_metadata_iter
                .next()
                .expect("Should have scan metadata")?;

            {
                let mc = txn.with_manifest_commit();
                // Create new leaf (leaf B) and move files from leaf A
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                leaf.add_existing_actions(&engine, scan_metadata.scan_files)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            // Commit
            let _committed = match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => c,
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify via scan - files should be in new leaf with no duplicates
        let final_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_eq!(final_snapshot.version(), 2);
        let scanned = collect_scanned_files(final_snapshot, &engine)?;

        // This test will fail if manifest DVs are not properly applied or if files are duplicated
        verify_scanned_files(
            &scanned,
            &["fileA.parquet", "fileB.parquet"],
            &[], // No DVs
        );
    }
    Ok(())
}
