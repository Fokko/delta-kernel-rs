//! Integration tests for how a manifest commit rolls up prior plain delta log commits sitting
//! on top of an existing content root.

#[path = "support/amt_test_utils.rs"]
mod amt_test_utils;

use std::collections::HashMap;
use std::sync::Arc;

use amt_test_utils::{
    add_files, add_leaf, assert_entries, assert_leaf_entries, assert_root_entries, assert_scan_dv,
    collect_root_entries, dv_descriptor, leaf_path, remove_files_by_path, setup_amt_test_tables,
    single_id_column_schema, update_dvs_by_path, DataFile, Entry, ExpectedDv,
};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::object_store::ObjectStoreExt as _;
use delta_kernel::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};
use delta_kernel::{DataContentType, Snapshot, TrackingStatus};
use test_utils::{create_table, engine_store_setup};
use uuid::Uuid;

/// Test Scenario: Files Added in log commits after an initial manifest commit are
/// subsequently written into the root by the next manifest commit
#[tokio::test]
async fn test_files_added_after_root() -> Result<(), Box<dyn std::error::Error>> {
    let schema = single_id_column_schema()?;

    let (table_url, engine) = setup_amt_test_tables(schema.clone(), "files_after_root").await?;

    // v1: Manifest commit adds file1, file2 to a leaf
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_leaf(
            &mut txn,
            &engine,
            add_files_schema,
            &[
                DataFile {
                    location: "file1.parquet",
                    size: 2048,
                    mod_time: 1000000,
                    num_records: Some(100),
                },
                DataFile {
                    location: "file2.parquet",
                    size: 1024,
                    mod_time: 1000001,
                    num_records: Some(50),
                },
            ],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 1);
    }

    // Verify v1: root has exactly one leaf reference, freshly Added
    let leaf1_path;
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            snapshot.checkpoint_action().is_some(),
            "v1 should create root manifest"
        );
        leaf1_path = leaf_path(&snapshot, &engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Added).sequence_number(1)],
        )?;
    }

    // v2: Manifest commit creates root manifest (no adds/removes/DV updates of its own)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 2);
    }

    // Verify v2: a no-op manifest commit reuses v1's exact root file, so the entry is
    // still Added.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            snapshot.checkpoint_action().is_some(),
            "Root manifest should exist"
        );
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Added).sequence_number(1)],
        )?;
    }

    // v3: Regular commit adds file3 to log
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_files(
            &mut txn,
            add_files_schema,
            &[DataFile {
                location: "file3.parquet",
                size: 512,
                mod_time: 1000002,
                num_records: Some(25),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 3);
    }

    // v4: Regular commit adds file4 to log
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_files(
            &mut txn,
            add_files_schema,
            &[DataFile {
                location: "file4.parquet",
                size: 768,
                mod_time: 1000003,
                num_records: Some(30),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 4);
    }

    // v5: Manifest commit creates NEW root (rolling up log) + adds file5 via a leaf
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_leaf(
            &mut txn,
            &engine,
            add_files_schema,
            &[DataFile {
                location: "file5.parquet",
                size: 2048,
                mod_time: 1000004,
                num_records: Some(100),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 5);
        let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(new_snapshot.checkpoint_action().is_some());
    }

    // Verify v5: file3/file4 written as root-direct entries with their own commit's
    // sequence_number; file5 referenced via a new second leaf.
    {
        let snapshot: Arc<Snapshot> = Snapshot::builder_for(table_url.clone()).build(&engine)?;

        // file3/file4: paths are ours, known a priori -- exhaustive check, no discovery.
        let root_entries = collect_root_entries(&snapshot, &engine)?;
        let data_entries: Vec<Entry> = root_entries
            .iter()
            .filter(|e| e.content_type == DataContentType::Data)
            .cloned()
            .collect();
        assert_entries(
            &data_entries,
            &[
                Entry::new("file3.parquet", TrackingStatus::Existing).sequence_number(3),
                Entry::new("file4.parquet", TrackingStatus::Existing).sequence_number(4),
            ],
            "root Data entries",
        );

        // Leaf references: leaf1's path is known; leaf2 (file5's) is a fresh UUID we
        // can't predict, so check it by exclusion instead of by path.
        let leaf_refs: Vec<&Entry> = root_entries
            .iter()
            .filter(|e| e.content_type == DataContentType::DataManifest)
            .collect();
        assert_eq!(leaf_refs.len(), 2, "expected 2 leaf references");
        let leaf1 = leaf_refs
            .iter()
            .find(|e: &&&Entry| e.path == leaf1_path)
            .expect("leaf1 still referenced");
        assert_eq!(
            **leaf1,
            Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Existing).sequence_number(1)
        );
        let leaf2 = leaf_refs
            .iter()
            .find(|e| e.path != leaf1_path)
            .expect("a new leaf reference for file5");
        assert_eq!(
            **leaf2,
            Entry::leaf_ref(leaf2.path.clone(), TrackingStatus::Added).sequence_number(5)
        );
    }
    Ok(())
}

/// Test Scenario: File Removal of Root Entry in Log
#[tokio::test]
async fn test_file_removal_of_root_entry_in_log() -> Result<(), Box<dyn std::error::Error>> {
    let schema = single_id_column_schema()?;

    let (table_url, engine) = setup_amt_test_tables(schema.clone(), "file_removal_flat").await?;

    // v1: Manifest commit with files DIRECTLY in root (no leaf)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let add_files_schema = txn.add_files_schema();
        add_files(
            &mut txn,
            add_files_schema,
            &[
                DataFile {
                    location: "file1.parquet",
                    size: 2048,
                    mod_time: 1000000,
                    num_records: Some(100),
                },
                DataFile {
                    location: "file2.parquet",
                    size: 1024,
                    mod_time: 1000001,
                    num_records: Some(50),
                },
                DataFile {
                    location: "file3.parquet",
                    size: 3072,
                    mod_time: 1000002,
                    num_records: Some(150),
                },
                DataFile {
                    location: "file4.parquet",
                    size: 1536,
                    mod_time: 1000003,
                    num_records: Some(75),
                },
            ],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 1);
    }

    // Verify v1: Root contains all 4 files, all freshly Added
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            snapshot.checkpoint_action().is_some(),
            "v1 should create root manifest"
        );
        assert_root_entries(
            &snapshot,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file2.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file3.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file4.parquet", TrackingStatus::Added).sequence_number(1),
            ],
        )?;
    }

    // v2: Log commit removes file2
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let scan = snapshot.clone().scan_builder().build()?;
        let removed_count = remove_files_by_path(&mut txn, scan, &engine, &["file2.parquet"])?;
        assert_eq!(removed_count, 1, "Should remove exactly 1 file");

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 2);
    }

    // v3: Manifest commit adds file5 and creates new root
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let add_files_schema = txn.add_files_schema();
        add_files(
            &mut txn,
            add_files_schema,
            &[DataFile {
                location: "file5.parquet",
                size: 1024,
                mod_time: 1000004,
                num_records: Some(50),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 3);
    }

    // Final verification: file2's removal is a plain log Remove, not an AMT
    // Replaced/Deleted entry -- it's gone from the rolled-up root entirely.
    {
        let snapshot: Arc<Snapshot> = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(snapshot.checkpoint_action().is_some());
        assert_root_entries(
            &snapshot,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Existing).sequence_number(1),
                Entry::new("file3.parquet", TrackingStatus::Existing).sequence_number(1),
                Entry::new("file4.parquet", TrackingStatus::Existing).sequence_number(1),
                Entry::new("file5.parquet", TrackingStatus::Added).sequence_number(3),
            ],
        )?;
    }
    Ok(())
}

/// Test File Removal of Leaf Entry in Log Written by a Subsequent Manifest Commit
#[tokio::test]
async fn test_file_removal_of_leaf_entry_in_log() -> Result<(), Box<dyn std::error::Error>> {
    let schema = single_id_column_schema()?;

    let (table_url, engine) = setup_amt_test_tables(schema.clone(), "file_removal_leaf").await?;

    // v1: Manifest commit with 4 files via leaf writer (creates root manifest)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_leaf(
            &mut txn,
            &engine,
            add_files_schema,
            &[
                DataFile {
                    location: "file1.parquet",
                    size: 2048,
                    mod_time: 1000000,
                    num_records: Some(100),
                },
                DataFile {
                    location: "file2.parquet",
                    size: 1024,
                    mod_time: 1000001,
                    num_records: Some(50),
                },
                DataFile {
                    location: "file3.parquet",
                    size: 3072,
                    mod_time: 1000002,
                    num_records: Some(150),
                },
                DataFile {
                    location: "file4.parquet",
                    size: 1536,
                    mod_time: 1000003,
                    num_records: Some(75),
                },
            ],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 1);
    }

    // Verify v1: all 4 files in one leaf, both the reference and the leaf's own
    // content freshly Added.
    let leaf1_path;
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            snapshot.checkpoint_action().is_some(),
            "v1 should create root manifest"
        );
        leaf1_path = leaf_path(&snapshot, &engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Added).sequence_number(1)],
        )?;
        assert_leaf_entries(
            &table_url,
            &leaf1_path,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file2.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file3.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file4.parquet", TrackingStatus::Added).sequence_number(1),
            ],
        )?;
    }

    // v2: Log commit removes file2 (leaf-resident)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let scan = snapshot.clone().scan_builder().build()?;
        let removed_count = remove_files_by_path(&mut txn, scan, &engine, &["file2.parquet"])?;
        assert_eq!(removed_count, 1, "Should remove exactly 1 file");

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 2);
    }

    // v3: Manifest commit creates NEW root + adds file5 via a second leaf
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_leaf(
            &mut txn,
            &engine,
            add_files_schema,
            &[DataFile {
                location: "file5.parquet",
                size: 1024,
                mod_time: 1000004,
                num_records: Some(50),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 3);
    }

    // Final verification: file2's removal is recorded as a dead position in leaf1's
    // reference row (manifest_dv) -- the leaf file itself is never rewritten.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(snapshot.checkpoint_action().is_some());

        // No root-direct Data entries in this test -- everything lives in leaf1 or leaf2.
        let root_entries = collect_root_entries(&snapshot, &engine)?;
        let data_entries: Vec<Entry> = root_entries
            .iter()
            .filter(|e| e.content_type == DataContentType::Data)
            .cloned()
            .collect();
        assert_entries(&data_entries, &[], "root Data entries");

        // leaf1's path is known; leaf2 (file5's) is a fresh UUID, checked by exclusion.
        let leaf_refs: Vec<&Entry> = root_entries
            .iter()
            .filter(|e| e.content_type == DataContentType::DataManifest)
            .collect();
        assert_eq!(leaf_refs.len(), 2, "expected 2 leaf references");
        let leaf1 = leaf_refs
            .iter()
            .find(|e| e.path == leaf1_path)
            .expect("leaf1 still referenced");
        assert_eq!(
            **leaf1,
            Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Existing)
                .sequence_number(1)
                .manifest_dv_cardinality(1)
        );
        let leaf2 = leaf_refs
            .iter()
            .find(|e| e.path != leaf1_path)
            .expect("a new leaf reference for file5");
        assert_eq!(
            **leaf2,
            Entry::leaf_ref(leaf2.path.clone(), TrackingStatus::Added).sequence_number(3)
        );

        assert_leaf_entries(
            &table_url,
            &leaf1_path,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file2.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file3.parquet", TrackingStatus::Added).sequence_number(1),
                Entry::new("file4.parquet", TrackingStatus::Added).sequence_number(1),
            ],
        )?;
    }
    Ok(())
}

/// Test Scenario: DV Addition and Replacement for a File in a Leaf Manifest
///
/// Setup:
/// - v1: Batch commit adds file to a leaf manifest (no DV)
/// - v2: Regular commit adds DV to that file via delta log
/// - v3: Batch commit creates a new root, writing v2's DV into it
/// - v4: Regular commit replaces DV via delta log
/// - v5: Batch commit creates a new root, writing v4's DV into it
///
/// Expected:
/// - v3 root should have file with DV from v2
/// - v5 root should have file with DV from v4 (replacement)
#[tokio::test]
async fn test_dv_addition_and_replacement_leaf_manifest() -> Result<(), Box<dyn std::error::Error>>
{
    let schema = single_id_column_schema()?;

    let (table_url, engine) =
        setup_amt_test_tables(schema.clone(), "dv_leaf_add_and_replace").await?;

    // v1: Batch commit adds file1 to a leaf manifest (no DV)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let add_files_schema = txn.add_files_schema();
        add_leaf(
            &mut txn,
            &engine,
            add_files_schema,
            &[DataFile {
                location: "file1.parquet",
                size: 2048,
                mod_time: 1000000,
                num_records: Some(100),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 1);
    }

    // Verify v1: file1 in a leaf, no DV, both entries freshly Added.
    let leaf1_path;
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            snapshot.checkpoint_action().is_some(),
            "v1 should have root"
        );
        let scan = snapshot.clone().scan_builder().build()?;
        assert_scan_dv(scan, &engine, "file1.parquet", None)?;

        leaf1_path = leaf_path(&snapshot, &engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Added).sequence_number(1)],
        )?;
        assert_leaf_entries(
            &table_url,
            &leaf1_path,
            &engine,
            &[Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1)],
        )?;
    }

    // v2: Regular commit adds DV to file1 via delta log
    let (dv_v2, dv_v2_location) = dv_descriptor(
        Uuid::parse_str("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")?,
        0,
        10,
        5,
    );
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let scan = snapshot.clone().scan_builder().build()?;
        update_dvs_by_path(
            &mut txn,
            scan,
            &engine,
            HashMap::from([("file1.parquet".to_string(), dv_v2.clone())]),
        )?;
        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 2);
    }

    // Verify v2: DV added (log commit, nothing persisted yet).
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let scan = snapshot.scan_builder().build()?;
        assert_scan_dv(
            scan,
            &engine,
            "file1.parquet",
            Some(&ExpectedDv {
                storage_type: "u".to_string(),
                path_or_inline_dv: dv_v2.path_or_inline_dv.clone(),
                cardinality: 5,
            }),
        )?;
    }

    // v3: Batch commit creates new root, rolling up v2 DV
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 3);
        let s = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(s.checkpoint_action().is_some(), "v3 should have root");
    }

    // Verify v3: a leaf-resident DV change moves the file to a root-direct entry;
    // leaf1's reference had only file1, now fully superseded, so it's Deleted. The
    // leaf file itself is untouched.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Existing)
                    .sequence_number(1)
                    .deletion_vector(dv_v2_location.clone(), 5),
                Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Deleted)
                    .sequence_number(1)
                    .manifest_dv_cardinality(1),
            ],
        )?;
        assert_leaf_entries(
            &table_url,
            &leaf1_path,
            &engine,
            &[Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1)],
        )?;
    }

    // v4: Regular commit replaces DV via delta log
    let (dv_v4, dv_v4_location) = dv_descriptor(
        Uuid::parse_str("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")?,
        0,
        15,
        8,
    );
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let scan = snapshot.clone().scan_builder().build()?;
        update_dvs_by_path(
            &mut txn,
            scan,
            &engine,
            HashMap::from([("file1.parquet".to_string(), dv_v4.clone())]),
        )?;
        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 4);
    }

    // Verify v4: DV replaced (log commit, nothing persisted yet).
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let scan = snapshot.scan_builder().build()?;
        assert_scan_dv(
            scan,
            &engine,
            "file1.parquet",
            Some(&ExpectedDv {
                storage_type: "u".to_string(),
                path_or_inline_dv: dv_v4.path_or_inline_dv.clone(),
                cardinality: 8,
            }),
        )?;
    }

    // v5: Batch commit creates second new root, rolling up v4 DV replacement
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 5);
        let s = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(s.checkpoint_action().is_some(), "v5 should have root");
    }

    // Verify v5: v4's DV is now written into the root; leaf1's reference stays Deleted.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[
                Entry::new("file1.parquet", TrackingStatus::Existing)
                    .sequence_number(1)
                    .deletion_vector(dv_v4_location.clone(), 8),
                Entry::leaf_ref(leaf1_path.clone(), TrackingStatus::Deleted)
                    .sequence_number(1)
                    .manifest_dv_cardinality(1),
            ],
        )?;
    }
    Ok(())
}

/// Test Scenario: DV Replacement in the Log, Written by a Subsequent Manifest Commit
///
/// Setup:
/// - v1: Manifest commit adds file to root (no DV)
/// - v2: Manifest commit adds DV to that file
/// - v3: Regular commit replaces DV via delta log
/// - v4: Manifest commit creates new root
///
/// Expected: v4 should have file with DV from v3 (replacement), not v2 (original)
#[tokio::test]
async fn test_dv_replacement() -> Result<(), Box<dyn std::error::Error>> {
    let schema = single_id_column_schema()?;

    let (table_url, engine) = setup_amt_test_tables(schema.clone(), "dv_replacement").await?;

    // v1: Manifest commit - add file1 to root (no DV)
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let add_files_schema = txn.add_files_schema();
        add_files(
            &mut txn,
            add_files_schema,
            &[DataFile {
                location: "file1.parquet",
                size: 2048,
                mod_time: 1000000,
                num_records: Some(100),
            }],
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 1);
    }

    // Verify v1: file1 present in root, freshly Added, no DV
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::new("file1.parquet", TrackingStatus::Added).sequence_number(1)],
        )?;
    }

    // v2: Manifest commit - add DV to file1 in root
    let (dv_v2, dv_v2_location) = dv_descriptor(
        Uuid::parse_str("12345678-1234-1234-1234-123456789abc")?,
        0,
        10,
        5,
    );
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;
        let scan = snapshot.clone().scan_builder().build()?;
        update_dvs_by_path(
            &mut txn,
            scan,
            &engine,
            HashMap::from([("file1.parquet".to_string(), dv_v2.clone())]),
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 2);
    }

    // Verify v2: file1's DV is set in place.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::new("file1.parquet", TrackingStatus::Existing)
                .sequence_number(1)
                .deletion_vector(dv_v2_location.clone(), 5)],
        )?;
    }

    // v3: Regular commit - replace DV via delta log
    let (dv_v3, dv_v3_location) = dv_descriptor(
        Uuid::parse_str("87654321-4321-4321-4321-cba987654321")?,
        0,
        15,
        8,
    );
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        let scan = snapshot.clone().scan_builder().build()?;
        update_dvs_by_path(
            &mut txn,
            scan,
            &engine,
            HashMap::from([("file1.parquet".to_string(), dv_v3.clone())]),
        )?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 3);
    }

    // Verify v3: DV replaced (v3, not v2); log commit, nothing persisted yet.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let scan = snapshot.scan_builder().build()?;
        assert_scan_dv(
            scan,
            &engine,
            "file1.parquet",
            Some(&ExpectedDv {
                storage_type: "u".to_string(),
                path_or_inline_dv: dv_v3.path_or_inline_dv.clone(),
                cardinality: 8,
            }),
        )?;
    }

    // v4: Manifest commit - create new root and write v3's DV into it
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), &engine)?;
        txn.with_manifest_commit()?;

        let c = txn.commit(&engine)?.unwrap_committed();
        assert_eq!(c.commit_version(), 4);
        let new_snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert!(
            new_snapshot.checkpoint_action().is_some(),
            "v4 should create new root manifest"
        );
    }

    // Verify v4: v3's DV is now written into the root, rolled up from the log.
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        assert_root_entries(
            &snapshot,
            &engine,
            &[Entry::new("file1.parquet", TrackingStatus::Existing)
                .sequence_number(1)
                .deletion_vector(dv_v3_location.clone(), 8)],
        )?;
    }
    Ok(())
}

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

fn create_test_schema_with_field_ids() -> Arc<StructType> {
    Arc::new(StructType::try_new(vec![field_with_metadata("id", DataType::LONG, 1)]).unwrap())
}

/// Test root manifest filtering by writing a delta commit file with stats, then doing a
/// manifest commit to convert those stats to content_stats.
#[tokio::test]
async fn test_manifest_commit_no_op_when_up_to_date() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let (store, engine, table_url) = engine_store_setup("no_op_manifest_commit", None);
    let engine = Arc::new(engine);
    let schema = create_test_schema_with_field_ids();

    // Create table
    create_table(
        store.clone(),
        table_url.clone(),
        schema.clone(),
        &[],
        true,
        vec!["columnMapping", "metadataTree-experimental"],
        vec![
            "columnMapping",
            "domainMetadata",
            "metadataTree-experimental",
            "rowTracking",
        ],
    )
    .await?;

    // Write version 1 with an Add action (similar to first test)
    let commit_json = r#"{"add":{"path":"part-00001.parquet","partitionValues":{},"size":100,"modificationTime":1,"dataChange":true,"defaultRowCommitVersion":1,"stats":"{\"numRecords\":100,\"minValues\":{\"id\":1},\"maxValues\":{\"id\":100},\"nullCount\":{\"id\":0}}"}}
"#;
    let commit_path = delta_kernel::object_store::path::Path::from(format!(
        "no_op_manifest_commit/_delta_log/{:020}.json",
        1
    ));
    store
        .put(&commit_path, commit_json.as_bytes().to_vec().into())
        .await?;

    // Manifest commit to create content root
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    println!(
        "Snapshot version before first manifest commit: {}",
        snapshot.version()
    );

    let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;
    txn.with_manifest_commit().unwrap();
    let _first_commit_result = txn.commit(engine.as_ref())?;

    // Verify content root was created
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert!(
        snapshot.checkpoint_action().is_some(),
        "Content root should exist after first manifest commit"
    );
    let content_root_version = snapshot.checkpoint_action().unwrap().version();
    println!(
        "After first manifest commit - snapshot version: {}, content root version: {}",
        snapshot.version(),
        content_root_version
    );

    // Now call manifest commit again with no new data
    // This should be a no-op since content_root.version == snapshot.version
    let mut txn = snapshot.transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?;
    txn.with_manifest_commit().unwrap();
    let result = txn.commit(engine.as_ref())?;

    let new_commit_version =
        if let delta_kernel::transaction::CommitResult::CommittedTransaction(committed) = result {
            committed.commit_version()
        } else {
            panic!("Expected committed transaction");
        };
    println!(
        "Second manifest commit created version: {}",
        new_commit_version
    );

    // Check the new snapshot - content root version should not have changed
    let new_snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let new_content_root_version = new_snapshot.checkpoint_action().map(|cr| cr.version());

    println!(
        "After second manifest commit - snapshot version: {}, content root version: {:?}",
        new_snapshot.version(),
        new_content_root_version
    );

    // The content root should still point to the same version (no rebuild occurred)
    assert_eq!(
        new_content_root_version,
        Some(content_root_version),
        "Content root version should not change when content root is already up-to-date"
    );

    Ok(())
}
