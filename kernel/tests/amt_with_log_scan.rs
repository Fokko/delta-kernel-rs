//! Tests for AMT (Adaptive Metadata Tree) root manifest + delta log interplay
//!
//! These tests verify that when a root manifest exists at version N, subsequent log
//! commits at N+1, N+2,... correctly interact with the root manifest during table scans.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::engine_data::TypedGetData;
use delta_kernel::object_store::ObjectStore;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::transaction::CommitResult;
use delta_kernel::{DeltaResult, Engine, Snapshot};
use test_utils::{
    collect_file_paths, create_add_files_metadata, create_table, engine_store_setup,
    remove_scan_files_with_selection,
};
use url::Url;

/// Test Scenario: Files Added in log commits after an initial manifest commit
/// are subsequently rolled up in next manifest commit
#[tokio::test]
async fn test_files_added_after_root() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "files_after_root").await?
    {
        // v1: Manifest commit adds file1, file2
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("file1.parquet", 2048, 1000000, 100),
                        ("file2.parquet", 1024, 1000001, 50),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 1);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // v2: Manifest commit creates root manifest
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 2);
                    let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(
                        new_snapshot.checkpoint_action().is_some(),
                        "Root manifest should exist"
                    );
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v2: Root manifest contains file1, file2
        // Tests: Root manifest correctly stores files from manifest commit
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = ["file1.parquet", "file2.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v2: Root manifest should contain exactly file1, file2",
            );
        }

        // v3: Regular commit adds file3 to log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file3.parquet", 512, 1000002, 25)],
            )?;
            txn.add_files(metadata);

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 3);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // v4: Regular commit adds file4 to log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file4.parquet", 768, 1000003, 30)],
            )?;
            txn.add_files(metadata);

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 4);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v4: Scan should show all 4 files (2 from root + 2 from log)
        // Tests: Log replay correctly merges files from root manifest (v2) + delta log commits (v3,
        // v4)
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file2.parquet",
                "file3.parquet",
                "file4.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v4: Should contain 2 files from root + 2 files from log",
            );
        }

        // v5: Manifest commit creates NEW root (rolling up log) + adds file5
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let add_files_schema = txn.add_files_schema();
            {
                // Add file5 as part of the new root creation
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![("file5.parquet", 2048, 1000004, 100)],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 5);
                    let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(new_snapshot.checkpoint_action().is_some());
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v5: New root should contain all 5 files (4 rolled up + 1 new)
        // Tests: New root manifest correctly rolls up previous root + delta log changes + new files
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file2.parquet",
                "file3.parquet",
                "file4.parquet",
                "file5.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v5: New root should contain 4 rolled-up files + 1 newly added file",
            );
        }
    }
    Ok(())
}

/// Test Scenario: File Removal of Root Entry in Log
#[tokio::test]
#[ignore = "BUG: New root manifest does not roll up Remove actions from delta log in commit 3 - file2 reappears"]
async fn test_file_removal_of_root_entry_in_log() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "file_removal_flat").await?
    {
        // v1: Manifest commit with files DIRECTLY in root (no leaf)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("file1.parquet", 2048, 1000000, 100),
                    ("file2.parquet", 1024, 1000001, 50),
                    ("file3.parquet", 3072, 1000002, 150),
                    ("file4.parquet", 1536, 1000003, 75),
                ],
            )?;
            txn.add_files(metadata);

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 1);
                    let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(
                        snapshot_v1.checkpoint_action().is_some(),
                        "v1 should create root manifest"
                    );
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v1: Root contains all 4 files
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file2.parquet",
                "file3.parquet",
                "file4.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v1: Root manifest should contain all 4 files",
            );
        }

        // v2: Log commit removes file2
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            let scan = snapshot.clone().scan_builder().build()?;

            let mut files_seen = 0;
            let removed_count = remove_scan_files_with_selection(
                &mut txn,
                scan,
                &engine,
                |_batch_idx, selection_vector| {
                    for selected in selection_vector.iter_mut() {
                        if *selected {
                            files_seen += 1;
                            *selected = files_seen == 2; // Remove 2nd file
                        }
                    }
                    selection_vector.iter().any(|&x| x)
                },
            )?;

            assert_eq!(removed_count, 1, "Should remove exactly 1 file");

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 2);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v2: file2 removed
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = ["file1.parquet", "file3.parquet", "file4.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v2: Should show 3 files (file2 removed by delta log)",
            );
        }

        // v3: Manifest commit adds file5 and creates new root
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file5.parquet", 1024, 1000004, 50)],
            )?;
            txn.add_files(metadata);

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 3);
                    let new_snapshot: Arc<Snapshot> =
                        Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(new_snapshot.checkpoint_action().is_some());
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Final verification: v3 should show 4 files (3 rolled up + 1 new)
        // Tests: New root correctly rolls up Remove action from flat structure
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths: HashSet<String> = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file3.parquet",
                "file4.parquet",
                "file5.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v3: New root manifest should contain 3 files (file2 removed) + 1 newly added file",
            );
        }
    }
    Ok(())
}

/// Test File Removal of Leaf Entry in Log rolls up in subsequent manifest commit
#[tokio::test]
#[ignore = "BUG: New root manifest does not roll up Remove actions from delta log in commit 3 - file2 reappears"]
async fn test_file_removal_of_leaf_entry_in_log() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "file_removal_leaf").await?
    {
        // v1: Manifest commit with 4 files via leaf writer (creates root manifest)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let add_files_schema = txn.add_files_schema();
            {
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("file1.parquet", 2048, 1000000, 100),
                        ("file2.parquet", 1024, 1000001, 50),
                        ("file3.parquet", 3072, 1000002, 150),
                        ("file4.parquet", 1536, 1000003, 75),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 1);
                    let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(
                        snapshot_v1.checkpoint_action().is_some(),
                        "v1 should create root manifest"
                    );
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v1: Root contains all 4 files
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file2.parquet",
                "file3.parquet",
                "file4.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v1: Root manifest should contain all 4 files",
            );
        }

        // v2: Log commit removes file2
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            let scan = snapshot.clone().scan_builder().build()?;

            // Remove only the 2nd file (file2.parquet) by position
            let mut files_seen = 0;
            let removed_count: usize = remove_scan_files_with_selection(
                &mut txn,
                scan,
                &engine,
                |_batch_idx, selection_vector| {
                    for selected in selection_vector.iter_mut() {
                        if *selected {
                            files_seen += 1;
                            *selected = files_seen == 2; // Remove 2nd file
                        }
                    }
                    selection_vector.iter().any(|&x| x)
                },
            )?;

            assert_eq!(removed_count, 1, "Should remove exactly 1 file");

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 2);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v2: file2 removed
        // Tests: Delta log Remove action correctly filters out file from root manifest
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = ["file1.parquet", "file3.parquet", "file4.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v2: Should show 3 files (file2 removed by delta log)",
            );
        }

        // v3: Manifest commit creates NEW root + adds file5
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let add_files_schema = txn.add_files_schema();
            {
                // Add file5 via leaf writer as part of new root creation
                let mc = txn.with_manifest_commit();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![("file5.parquet", 1024, 1000004, 50)],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 3);
                    let new_snapshot: Arc<Snapshot> =
                        Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(new_snapshot.checkpoint_action().is_some());
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Final verification: v3 should show 4 files (3 rolled up + 1 new)
        // Tests: New root correctly rolls up Remove action from delta log (file2 stays removed) +
        // adds new file
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths: HashSet<String> = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> = [
                "file1.parquet",
                "file3.parquet",
                "file4.parquet",
                "file5.parquet",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            assert_sets_equal(
                &expected,
                &paths,
                "v3: New root manifest should contain 3 files (file2 removed) + 1 newly added file",
            );
        }
    }
    Ok(())
}

/// Test Scenario: DV Replacement in log rolled up in subsequent manifest commit
///
/// Setup:
/// - v1: Manifest commit adds file to root (no DV)
/// - v2: Manifest commit adds DV to that file
/// - v3: Regular commit replaces DV via delta log
/// - v4: Manifest commit creates new root
///
/// Expected: v4 should have file with DV from v3 (replacement), not v2 (original)
/// Actual: v4 has file with NO DV - BUG: manifest commit does not roll up DV replacements from
/// delta log
#[tokio::test]
#[ignore = "BUG: Manifest commit at v4 does not roll up DV replacement from delta log - file has no DV"]
async fn test_dv_replacement() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "dv_replacement").await?
    {
        // v1: Manifest commit - add file1 to root (no DV)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file1.parquet", 2048, 1000000, 100)],
            )?;
            txn.add_files(metadata);

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 1);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v1: file1 present in root, no DV
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot, &engine)?;
            let expected: HashSet<String> =
                ["file1.parquet"].iter().map(|s| s.to_string()).collect();
            assert_sets_equal(&expected, &paths, "v1: Root should contain file1");
        }

        // v2: Manifest commit - add DV to file1 in root
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            // Scan to get file1
            let scan = snapshot.clone().scan_builder().build()?;
            let all_scan_metadata: Vec<_> = scan
                .scan_metadata(&engine)?
                .collect::<Result<Vec<_>, _>>()?;
            let scan_files: Vec<_> = all_scan_metadata
                .into_iter()
                .map(|sm| sm.scan_files)
                .collect();

            // Create DV descriptor for file1
            let mut dv_map = std::collections::HashMap::new();
            let dv_v2 = DeletionVectorDescriptor {
                storage_type: DeletionVectorStorageType::PersistedRelative,
                path_or_inline_dv: "12345678-1234-1234-1234-123456789abc".to_string(),
                offset: Some(0),
                size_in_bytes: 10,
                cardinality: 5,
            };
            dv_map.insert("file1.parquet".to_string(), dv_v2);

            // Add DV to file1
            txn.update_deletion_vectors(dv_map, scan_files.into_iter().map(Ok))?;

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 2);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v2: file1 present with DV from v2
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files_with_dvs = collect_files_with_dvs(snapshot, &engine)?;

            assert_eq!(files_with_dvs.len(), 1, "v2: Should have exactly 1 file");
            let file1_dv = files_with_dvs
                .get("file1.parquet")
                .expect("file1.parquet should be present")
                .as_ref()
                .expect("file1.parquet should have a DV");

            assert_eq!(
                file1_dv.path_or_inline_dv, "12345678-1234-1234-1234-123456789abc",
                "v2: DV should be from v2"
            );
            assert_eq!(file1_dv.cardinality, 5, "v2: DV cardinality should be 5");
            assert_eq!(
                file1_dv.storage_type, "u",
                "v2: DV storage type should be 'u' (PersistedRelative)"
            );
        }

        // v3: Regular commit - replace DV via delta log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            // Scan to get file1 with current DV
            let scan = snapshot.clone().scan_builder().build()?;
            let all_scan_metadata: Vec<_> = scan
                .scan_metadata(&engine)?
                .collect::<Result<Vec<_>, _>>()?;
            let scan_files: Vec<_> = all_scan_metadata
                .into_iter()
                .map(|sm| sm.scan_files)
                .collect();

            // Create NEW DV descriptor for file1 (replacement)
            let mut dv_map = std::collections::HashMap::new();
            let dv_v3 = DeletionVectorDescriptor {
                storage_type: DeletionVectorStorageType::PersistedRelative,
                path_or_inline_dv: "87654321-4321-4321-4321-cba987654321".to_string(),
                offset: Some(0),
                size_in_bytes: 15,
                cardinality: 8,
            };
            dv_map.insert("file1.parquet".to_string(), dv_v3);

            // Replace DV via delta log
            txn.update_deletion_vectors(dv_map, scan_files.into_iter().map(Ok))?;

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 3);
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v3: file1 present with REPLACED DV from v3 (not v2!)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files_with_dvs = collect_files_with_dvs(snapshot, &engine)?;

            assert_eq!(files_with_dvs.len(), 1, "v3: Should have exactly 1 file");
            let file1_dv = files_with_dvs
                .get("file1.parquet")
                .expect("file1.parquet should be present")
                .as_ref()
                .expect("file1.parquet should have a DV");

            assert_eq!(
                file1_dv.path_or_inline_dv, "87654321-4321-4321-4321-cba987654321",
                "v3: DV should be REPLACED with v3 DV (not v2!)"
            );
            assert_eq!(file1_dv.cardinality, 8, "v3: DV cardinality should be 8");
            assert_eq!(
                file1_dv.storage_type, "u",
                "v3: DV storage type should be 'u' (PersistedRelative)"
            );
        }

        // v4: Manifest commit - create new root and verify DV rollup
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit();

            match txn.commit(&engine)? {
                CommitResult::CommittedTransaction(c) => {
                    assert_eq!(c.commit_version(), 4);
                    let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
                    assert!(
                        new_snapshot.checkpoint_action().is_some(),
                        "v4 should create new root manifest"
                    );
                }
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Verify v4: file1 present with DV from v3 (NOT v2) rolled up into new root
        // manifest commit should roll up the REPLACED DV from delta log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files_with_dvs = collect_files_with_dvs(snapshot, &engine)?;

            assert_eq!(files_with_dvs.len(), 1, "v4: Should have exactly 1 file");
            let file1_dv = files_with_dvs
                .get("file1.parquet")
                .expect("file1.parquet should be present")
                .as_ref()
                .expect("file1.parquet should have DV from v3 rolled up");

            // Must be the v3 DV (replacement), not the v2 DV (original)
            assert_eq!(
                file1_dv.path_or_inline_dv, "87654321-4321-4321-4321-cba987654321",
                "v4: New root MUST have the REPLACED DV from v3, not the original from v2!"
            );
            assert_eq!(
                file1_dv.cardinality, 8,
                "v4: DV cardinality should be 8 (from v3)"
            );
            assert_eq!(
                file1_dv.storage_type, "p",
                "v4: DV storage type should be 'p' (PersistedRelative)"
            );
        }
    }
    Ok(())
}

/// Simplified DV details for verification
#[derive(Debug, Clone, PartialEq, Eq)]
struct DvDetails {
    storage_type: String,
    path_or_inline_dv: String,
    cardinality: i64,
}

/// Collects files with their DV details from a snapshot
fn collect_files_with_dvs(
    snapshot: Arc<Snapshot>,
    engine: &dyn Engine,
) -> DeltaResult<HashMap<String, Option<DvDetails>>> {
    use delta_kernel::engine_data::{GetData, RowVisitor};
    use delta_kernel::expressions::ColumnName;

    struct DvCollector<'a> {
        files: HashMap<String, Option<DvDetails>>,
        selection_vector: &'a [bool],
    }

    impl<'a> RowVisitor for DvCollector<'a> {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            use std::sync::LazyLock;

            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["path"]),
                            ColumnName::new(["deletionVector", "storageType"]),
                            ColumnName::new(["deletionVector", "pathOrInlineDv"]),
                            ColumnName::new(["deletionVector", "cardinality"]),
                        ],
                        vec![
                            DataType::STRING,
                            DataType::STRING,
                            DataType::STRING,
                            DataType::LONG,
                        ],
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

                // Collect DV details if present
                let dv_details = if let Some(storage_type) =
                    getters[1].get_opt(i, "deletionVector.storageType")?
                {
                    let path_or_inline_dv: String =
                        getters[2].get(i, "deletionVector.pathOrInlineDv")?;
                    let cardinality: i64 = getters[3].get(i, "deletionVector.cardinality")?;

                    Some(DvDetails {
                        storage_type,
                        path_or_inline_dv,
                        cardinality,
                    })
                } else {
                    None
                };

                self.files.insert(path, dv_details);
            }
            Ok(())
        }
    }

    let scan = snapshot.scan_builder().build()?;
    let mut all_files = HashMap::new();

    for scan_metadata_result in scan.scan_metadata(engine)? {
        let scan_metadata = scan_metadata_result?;
        let selection_vector = scan_metadata.scan_files.selection_vector();
        let mut collector = DvCollector {
            files: HashMap::new(),
            selection_vector,
        };
        collector.visit_rows_of(scan_metadata.scan_files.data())?;
        all_files.extend(collector.files);
    }

    Ok(all_files)
}

async fn setup_amt_test_tables(
    schema: SchemaRef,
    table_base_name: &str,
) -> Result<
    Vec<(
        Url,
        DefaultEngine<TokioBackgroundExecutor>,
        Arc<dyn ObjectStore>,
    )>,
    Box<dyn std::error::Error>,
> {
    let table_name = format!("{table_base_name}_37");
    let (store, engine, table_location) = engine_store_setup(table_name.as_str(), None);

    Ok(vec![(
        create_table(
            store.clone(),
            table_location,
            schema.clone(),
            &[],
            true,
            vec![
                "columnMapping",
                "metadataTree-experimental",
                "deletionVectors",
            ],
            vec![
                "columnMapping",
                "metadataTree-experimental",
                "deletionVectors",
            ],
        )
        .await?,
        engine,
        store,
    )])
}

fn create_test_schema() -> Result<Arc<StructType>, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )
    .with_metadata([
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
    ])])?))
}

/// Assert that two sets contain exactly the same elements (order-independent).
/// Shows clear error messages with missing and unexpected elements.
fn assert_sets_equal<T: std::fmt::Debug + std::hash::Hash + Eq + Ord>(
    expected: &HashSet<T>,
    actual: &HashSet<T>,
    context: &str,
) {
    if expected == actual {
        return;
    }

    let missing = {
        let mut v: Vec<_> = expected.difference(actual).collect();
        v.sort();
        v
    };
    let unexpected = {
        let mut v: Vec<_> = actual.difference(expected).collect();
        v.sort();
        v
    };

    let expected_sorted = {
        let mut v: Vec<_> = expected.iter().collect();
        v.sort();
        v
    };
    let actual_sorted = {
        let mut v: Vec<_> = actual.iter().collect();
        v.sort();
        v
    };

    panic!(
        "{}\nMissing files (expected but not found): {:?}\nUnexpected files (found but not expected): {:?}\nExpected: {:?}\nActual: {:?}",
        context, missing, unexpected, expected_sorted, actual_sorted
    );
}
