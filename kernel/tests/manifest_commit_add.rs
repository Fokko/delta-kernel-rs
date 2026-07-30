//! Integration tests for adds via the manifest-commit write path.

#[path = "support/amt_test_utils.rs"]
mod amt_test_utils;

use amt_test_utils::{
    collect_scanned_files, id_and_value_schema, setup_test_tables_with_column_mapping,
    verify_scanned_files,
};
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::Snapshot;
use test_utils::create_add_files_metadata;

#[tokio::test]
async fn test_transaction_basic_leaf_write() -> Result<(), Box<dyn std::error::Error>> {
    let schema = id_and_value_schema()?;

    let (table_url, engine) =
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_basic").await?;

    // Step 1: Create transaction with manifest commit enabled
    let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)?
        .with_operation("WRITE".to_string());
    let add_files_schema = txn.add_files_schema();

    // Step 2-3: Create leaf, add files, and add to manifest commit
    {
        let mc = txn.with_manifest_commit().unwrap();
        let mut leaf = mc.new_leaf_node_writer(&engine)?;
        let metadata = create_add_files_metadata(
            add_files_schema,
            vec![
                ("part-001.parquet", 2048, 1000000, Some(50)),
                ("part-002.parquet", 3072, 1000001, Some(75)),
            ],
        )?;
        leaf.add_files(&engine, metadata)?;
        mc.add_leaf(leaf.finish(&engine)?)?;
    }

    // Step 4: Commit
    let committed = txn.commit(&engine)?.unwrap_committed();

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
    Ok(())
}

#[tokio::test]
async fn test_transaction_multiple_leaves() -> Result<(), Box<dyn std::error::Error>> {
    let schema = id_and_value_schema()?;

    let (table_url, engine) =
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_multi_leaves").await?;

    // Create transaction with manifest commit enabled
    let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)?
        .with_operation("WRITE".to_string());
    let add_files_schema = txn.add_files_schema();

    // Create and add 3 leaves with different files
    {
        let mc = txn.with_manifest_commit().unwrap();
        for i in 0..3 {
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            let files = vec![
                (
                    format!("leaf{}_file1.parquet", i).leak() as &str,
                    1024 + i * 100,
                    1000000 + i,
                    Some(10 + i),
                ),
                (
                    format!("leaf{}_file2.parquet", i).leak() as &str,
                    2048 + i * 100,
                    1000010 + i,
                    Some(20 + i),
                ),
            ];
            let metadata = create_add_files_metadata(add_files_schema, files)?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }
    }

    // Commit
    txn.commit(&engine)?.unwrap_committed();

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
    Ok(())
}

#[tokio::test]
async fn test_transaction_sequential_commits() -> Result<(), Box<dyn std::error::Error>> {
    let schema = id_and_value_schema()?;

    let (table_url, engine) =
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_sequential").await?;

    // Commit transaction 1 with files A, B
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_operation("WRITE".to_string());
        let add_files_schema = txn.add_files_schema();
        {
            let mc = txn.with_manifest_commit().unwrap();
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("fileA.parquet", 1024, 1000000, Some(10)),
                    ("fileB.parquet", 2048, 1000001, Some(20)),
                ],
            )?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }
        txn.commit(&engine)?.unwrap_committed();
    }

    // Commit transaction 2 with files C, D
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_operation("WRITE".to_string());
        let add_files_schema = txn.add_files_schema();
        {
            let mc = txn.with_manifest_commit().unwrap();
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("fileC.parquet", 3072, 1000002, Some(30)),
                    ("fileD.parquet", 4096, 1000003, Some(40)),
                ],
            )?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }
        txn.commit(&engine)?.unwrap_committed();
    }

    // Commit transaction 3 with files E, F
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_operation("WRITE".to_string());
        let add_files_schema = txn.add_files_schema();
        {
            let mc = txn.with_manifest_commit().unwrap();
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            let metadata = create_add_files_metadata(
                add_files_schema,
                vec![
                    ("fileE.parquet", 5120, 1000004, Some(50)),
                    ("fileF.parquet", 6144, 1000005, Some(60)),
                ],
            )?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }
        txn.commit(&engine)?.unwrap_committed();
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
    Ok(())
}
