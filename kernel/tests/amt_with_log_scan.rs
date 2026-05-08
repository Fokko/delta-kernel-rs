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
use delta_kernel::{DeltaResult, Engine, Snapshot, TrackingStatus};
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
                let mc = txn.with_manifest_commit().unwrap();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("file1.parquet", 2048, 1000000, Some(100)),
                        ("file2.parquet", 1024, 1000001, Some(50)),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 1);
        }

        // v2: Manifest commit creates root manifest
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit().unwrap();

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 2);
            let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(
                new_snapshot.checkpoint_action().is_some(),
                "Root manifest should exist"
            );
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
                vec![("file3.parquet", 512, 1000002, Some(25))],
            )?;
            txn.add_files(metadata);

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 3);
        }

        // v4: Regular commit adds file4 to log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file4.parquet", 768, 1000003, Some(30))],
            )?;
            txn.add_files(metadata);

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 4);
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
                let mc = txn.with_manifest_commit().unwrap();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![("file5.parquet", 2048, 1000004, Some(100))],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 5);
            let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(new_snapshot.checkpoint_action().is_some());
        }

        // Verify v5: New root should contain all 5 files (4 rolled up + 1 new)
        // Tests: New root manifest correctly rolls up previous root + delta log changes + new files
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths = collect_file_paths(snapshot.clone(), &engine)?;
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

            // file3 and file4 were added by delta log commits at v3 and v4 respectively,
            // and are stored as Data entries directly in the root.
            // Sequence numbers must reflect the actual commit version, not the root version (5).
            // Both should have Existed status since they were rolled up from prior versions.
            let tracking = collect_root_manifest_tracking_info(snapshot, &engine)?;
            let file3 = tracking.get("file3.parquet").expect("file3 in root");
            assert_eq!(file3.seq_num, Some(3), "file3 seq_num");
            assert_eq!(file3.status, TrackingStatus::Existed as i32, "file3 status");
            let file4 = tracking.get("file4.parquet").expect("file4 in root");
            assert_eq!(file4.seq_num, Some(4), "file4 seq_num");
            assert_eq!(file4.status, TrackingStatus::Existed as i32, "file4 status");
        }
    }
    Ok(())
}

/// Test Scenario: File Removal of Root Entry in Log
#[tokio::test]
async fn test_file_removal_of_root_entry_in_log() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "file_removal_flat").await?
    {
        // v1: Manifest commit with files DIRECTLY in root (no leaf)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit().unwrap();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("file1.parquet", 2048, 1000000, Some(100)),
                    ("file2.parquet", 1024, 1000001, Some(50)),
                    ("file3.parquet", 3072, 1000002, Some(150)),
                    ("file4.parquet", 1536, 1000003, Some(75)),
                ],
            )?;
            txn.add_files(metadata);

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 1);
            let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(
                snapshot_v1.checkpoint_action().is_some(),
                "v1 should create root manifest"
            );
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

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 2);
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
            txn.with_manifest_commit().unwrap();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file5.parquet", 1024, 1000004, Some(50))],
            )?;
            txn.add_files(metadata);

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 3);
            let new_snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(new_snapshot.checkpoint_action().is_some());
        }

        // Final verification: v3 should show 4 files (3 rolled up + 1 new)
        // Tests: New root correctly rolls up Remove action from flat structure
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let paths: HashSet<String> = collect_file_paths(snapshot.clone(), &engine)?;
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

            // file1/file3/file4 were rolled up from v1 (Existed); file5 was added at v3 (Added).
            let tracking = collect_root_manifest_tracking_info(snapshot, &engine)?;
            for name in ["file1.parquet", "file3.parquet", "file4.parquet"] {
                let e = tracking
                    .get(name)
                    .unwrap_or_else(|| panic!("{name} in root"));
                assert_eq!(e.seq_num, Some(1), "{name} seq_num");
                assert_eq!(e.status, TrackingStatus::Existed as i32, "{name} status");
            }
            let file5 = tracking.get("file5.parquet").expect("file5 in root");
            assert_eq!(file5.seq_num, Some(3), "file5 seq_num");
            assert_eq!(file5.status, TrackingStatus::Added as i32, "file5 status");
        }
    }
    Ok(())
}

/// Test File Removal of Leaf Entry in Log rolls up in subsequent manifest commit
#[tokio::test]
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
                let mc = txn.with_manifest_commit().unwrap();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![
                        ("file1.parquet", 2048, 1000000, Some(100)),
                        ("file2.parquet", 1024, 1000001, Some(50)),
                        ("file3.parquet", 3072, 1000002, Some(150)),
                        ("file4.parquet", 1536, 1000003, Some(75)),
                    ],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 1);
            let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(
                snapshot_v1.checkpoint_action().is_some(),
                "v1 should create root manifest"
            );
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

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 2);
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
                let mc = txn.with_manifest_commit().unwrap();
                let mut leaf = mc.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![("file5.parquet", 1024, 1000004, Some(50))],
                )?;
                leaf.add_files(&engine, metadata)?;
                mc.add_leaf(leaf.finish(&engine)?)?;
            }

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 3);
            let new_snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(new_snapshot.checkpoint_action().is_some());
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
async fn test_dv_replacement() -> Result<(), Box<dyn std::error::Error>> {
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "dv_replacement").await?
    {
        // v1: Manifest commit - add file1 to root (no DV)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit().unwrap();

            let add_files_schema = txn.add_files_schema();
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![("file1.parquet", 2048, 1000000, Some(100))],
            )?;
            txn.add_files(metadata);

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 1);
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
            txn.with_manifest_commit().unwrap();

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

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 2);
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

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 3);
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
            txn.with_manifest_commit().unwrap();

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 4);
            let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(
                new_snapshot.checkpoint_action().is_some(),
                "v4 should create new root manifest"
            );
        }

        // Verify v4: file1 present with DV from v3 (NOT v2) rolled up into new root
        // manifest commit should roll up the REPLACED DV from delta log
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files_with_dvs = collect_files_with_dvs(snapshot.clone(), &engine)?;

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
                file1_dv.storage_type, "u",
                "v4: DV storage type should be 'u' (PersistedRelative)"
            );

            let tracking = collect_root_manifest_tracking_info(snapshot, &engine)?;
            let file1_tracking = tracking.get("file1.parquet").expect("file1 in root");
            assert_eq!(file1_tracking.seq_num, Some(1), "file1 seq_num");
            assert_eq!(
                file1_tracking.status,
                TrackingStatus::Existed as i32,
                "file1 status must be Existed (rolled up from v3 into v4 root)"
            );
        }
    }
    Ok(())
}

/// Test Scenario: DV Addition and Replacement for a File in a Leaf Manifest
///
/// Setup:
/// - v1: Batch commit adds file to a leaf manifest (no DV)
/// - v2: Regular commit adds DV to that file via delta log
/// - v3: Batch commit creates new root (first rollup)
/// - v4: Regular commit replaces DV via delta log
/// - v5: Batch commit creates new root (second rollup)
///
/// Expected:
/// - v3 root should have file with DV from v2
/// - v5 root should have file with DV from v4 (replacement)
#[tokio::test]
async fn test_dv_addition_and_replacement_leaf_manifest() -> Result<(), Box<dyn std::error::Error>>
{
    let schema = create_test_schema()?;

    for (table_url, engine, _store) in
        setup_amt_test_tables(schema.clone(), "dv_leaf_rollup").await?
    {
        // v1: Batch commit adds file1 to a leaf manifest (no DV)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let add_files_schema = txn.add_files_schema();
            {
                let batch = txn.with_manifest_commit().unwrap();
                let mut leaf = batch.new_leaf_node_writer(&engine)?;
                let metadata = create_add_files_metadata(
                    add_files_schema,
                    vec![("file1.parquet", 2048, 1000000, Some(100))],
                )?;
                leaf.add_files(&engine, metadata)?;
                batch.add_leaf(leaf.finish(&engine)?)?;
            }

            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 1);
            let s = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(s.checkpoint_action().is_some(), "v1 should have root");
        }

        // Verify v1: file1 present, no DV
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files = collect_files_with_dvs(snapshot, &engine)?;
            assert_eq!(files.len(), 1, "v1: should have 1 file");
            assert!(
                files["file1.parquet"].is_none(),
                "v1: file1 should have no DV"
            );
        }

        // v2: Regular commit adds DV to file1 via delta log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let scan = snapshot.clone().scan_builder().build()?;
            let scan_files: Vec<_> = scan
                .scan_metadata(&engine)?
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|sm| sm.scan_files)
                .collect();
            let mut dv_map = std::collections::HashMap::new();
            dv_map.insert(
                "file1.parquet".to_string(),
                DeletionVectorDescriptor {
                    storage_type: DeletionVectorStorageType::PersistedRelative,
                    path_or_inline_dv: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa".to_string(),
                    offset: Some(0),
                    size_in_bytes: 10,
                    cardinality: 5,
                },
            );
            txn.update_deletion_vectors(dv_map, scan_files.into_iter().map(Ok))?;
            assert_eq!(txn.commit(&engine)?.unwrap_committed().commit_version(), 2);
        }

        // Verify v2: file1 has DV from v2
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files = collect_files_with_dvs(snapshot, &engine)?;
            assert_eq!(files.len(), 1, "v2: should have 1 file");
            let dv = files["file1.parquet"]
                .as_ref()
                .expect("v2: file1 should have DV");
            assert_eq!(dv.path_or_inline_dv, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
            assert_eq!(dv.cardinality, 5);
        }

        // v3: Batch commit creates new root, rolling up v2 DV
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit().unwrap();
            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 3);
            let s = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(s.checkpoint_action().is_some(), "v3 should have root");
        }

        // Verify v3: file1 appears exactly once with DV from v2; rolled up as Existed
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files = collect_files_with_dvs(snapshot.clone(), &engine)?;
            assert_eq!(
                files.len(),
                1,
                "v3: should have exactly 1 file (not a duplicate from leaf + data entry)"
            );
            let dv = files["file1.parquet"]
                .as_ref()
                .expect("v3: file1 should have DV from v2");
            assert_eq!(
                dv.path_or_inline_dv, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "v3: DV should be from v2"
            );
            assert_eq!(dv.cardinality, 5);

            let tracking = collect_root_manifest_tracking_info(snapshot, &engine)?;
            let file1 = tracking.get("file1.parquet").expect("file1 in root at v3");
            assert_eq!(
                file1.status,
                TrackingStatus::Existed as i32,
                "v3: file1 rolled up from v2 log commit must be Existed"
            );
            assert_eq!(file1.seq_num, Some(1), "v3: file1 seq_num");
        }

        // v4: Regular commit replaces DV via delta log
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot
                .clone()
                .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            let scan = snapshot.clone().scan_builder().build()?;
            let scan_files: Vec<_> = scan
                .scan_metadata(&engine)?
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|sm| sm.scan_files)
                .collect();
            let mut dv_map = std::collections::HashMap::new();
            dv_map.insert(
                "file1.parquet".to_string(),
                DeletionVectorDescriptor {
                    storage_type: DeletionVectorStorageType::PersistedRelative,
                    path_or_inline_dv: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb".to_string(),
                    offset: Some(0),
                    size_in_bytes: 15,
                    cardinality: 8,
                },
            );
            txn.update_deletion_vectors(dv_map, scan_files.into_iter().map(Ok))?;
            assert_eq!(txn.commit(&engine)?.unwrap_committed().commit_version(), 4);
        }

        // Verify v4: file1 has DV from v4
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files = collect_files_with_dvs(snapshot, &engine)?;
            assert_eq!(files.len(), 1, "v4: should have 1 file");
            let dv = files["file1.parquet"]
                .as_ref()
                .expect("v4: file1 should have DV");
            assert_eq!(dv.path_or_inline_dv, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb");
            assert_eq!(dv.cardinality, 8);
        }

        // v5: Batch commit creates second new root, rolling up v4 DV replacement
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
            txn.with_manifest_commit().unwrap();
            let c = txn.commit(&engine)?.unwrap_committed();
            assert_eq!(c.commit_version(), 5);
            let s = Snapshot::builder_for(table_url.clone()).build(&engine)?;
            assert!(s.checkpoint_action().is_some(), "v5 should have root");
        }

        // Verify v5: file1 appears exactly once with DV from v4 (replacement); rolled up as Existed
        {
            let snapshot: Arc<Snapshot> =
                Snapshot::builder_for(table_url.clone()).build(&engine)?;
            let files = collect_files_with_dvs(snapshot.clone(), &engine)?;
            assert_eq!(
                files.len(),
                1,
                "v5: should have exactly 1 file (not a duplicate from leaf + data entry)"
            );
            let dv = files["file1.parquet"]
                .as_ref()
                .expect("v5: file1 should have DV from v4");
            assert_eq!(
                dv.path_or_inline_dv, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "v5: DV should be from v4 (replacement), not v2"
            );
            assert_eq!(dv.cardinality, 8);

            let tracking = collect_root_manifest_tracking_info(snapshot, &engine)?;
            let file1 = tracking.get("file1.parquet").expect("file1 in root at v5");
            assert_eq!(
                file1.status,
                TrackingStatus::Existed as i32,
                "v5: file1 rolled up from v4 log commit must be Existed"
            );
            assert_eq!(file1.seq_num, Some(1), "v5: file1 seq_num");
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

/// Per-entry data collected from the root manifest parquet in a single read.
struct TrackingEntry {
    /// Raw tracking status: `TrackingStatus::Existed as i32 == 0`, `Added == 1`, `Deleted == 2`.
    status: i32,
    seq_num: Option<i64>,
}

/// Reads the root manifest parquet for `snapshot` and returns a map from file path to
/// [`TrackingEntry`]. Entries without a location are skipped. Returns an empty map when the
/// snapshot has no root manifest.
fn collect_root_manifest_tracking_info(
    snapshot: Arc<Snapshot>,
    engine: &dyn Engine,
) -> DeltaResult<HashMap<String, TrackingEntry>> {
    use std::sync::LazyLock;

    use delta_kernel::engine_data::{GetData, RowVisitor};
    use delta_kernel::expressions::ColumnName;
    use delta_kernel::schema::{DataType, StructField, StructType};
    use delta_kernel::FileMeta;

    let Some(checkpoint_action) = snapshot.checkpoint_action() else {
        return Ok(HashMap::new());
    };

    let root_url = snapshot
        .table_root()
        .join(checkpoint_action.path())
        .map_err(|e| delta_kernel::Error::generic(format!("bad content root URL: {e}")))?;

    let schema = Arc::new(
        StructType::try_new([
            StructField::nullable("location", DataType::STRING),
            StructField::nullable(
                "tracking",
                DataType::Struct(Box::new(
                    StructType::try_new([
                        StructField::nullable("status", DataType::INTEGER),
                        StructField::nullable("sequenceNumber", DataType::LONG),
                    ])
                    .unwrap(),
                )),
            ),
        ])
        .unwrap(),
    );

    let file_meta = FileMeta {
        location: root_url,
        last_modified: 0,
        size: 0,
    };
    let batches: Vec<_> = engine
        .parquet_handler()
        .read_parquet_files(&[file_meta], schema, None)?
        .collect::<DeltaResult<Vec<_>>>()?;

    struct TrackingCollector {
        entries: HashMap<String, TrackingEntry>,
    }

    impl RowVisitor for TrackingCollector {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["location"]),
                            ColumnName::new(["tracking", "status"]),
                            ColumnName::new(["tracking", "sequenceNumber"]),
                        ],
                        vec![DataType::STRING, DataType::INTEGER, DataType::LONG],
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
                if let Some(path) = getters[0].get_opt(i, "location")? {
                    let status: i32 = getters[1]
                        .get_opt(i, "tracking.status")?
                        .unwrap_or(TrackingStatus::Existed as i32);
                    let seq_num: Option<i64> = getters[2].get_opt(i, "tracking.sequenceNumber")?;
                    self.entries.insert(path, TrackingEntry { status, seq_num });
                }
            }
            Ok(())
        }
    }

    let mut collector = TrackingCollector {
        entries: HashMap::new(),
    };
    for batch in &batches {
        collector.visit_rows_of(batch.as_ref())?;
    }
    Ok(collector.entries)
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
                "domainMetadata",
                "rowTracking",
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
