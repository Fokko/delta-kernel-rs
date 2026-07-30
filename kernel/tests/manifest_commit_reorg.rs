//! Integration tests for reorganizing content between leaves via the manifest-commit write path.

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
async fn test_move_files_from_leaf_to_leaf() -> Result<(), Box<dyn std::error::Error>> {
    let schema = id_and_value_schema()?;

    let (table_url, engine) =
        setup_test_tables_with_column_mapping(schema.clone(), &[], "txn_move_leaf_to_leaf").await?;

    // Commit 0: Create files in a leaf (leaf A)
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
                    ("fileA.parquet", 2048, 1000000, Some(50)),
                    ("fileB.parquet", 3072, 1000001, Some(75)),
                ],
            )?;
            leaf.add_files(&engine, metadata)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }

        txn.commit(&engine)?.unwrap_committed();
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
            let mc = txn.with_manifest_commit().unwrap();
            // Create new leaf (leaf B) and move files from leaf A
            let mut leaf = mc.new_leaf_node_writer(&engine)?;
            leaf.add_existing_actions(&engine, scan_metadata.scan_files)?;
            mc.add_leaf(leaf.finish(&engine)?)?;
        }

        // Commit
        txn.commit(&engine)?.unwrap_committed();
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
    Ok(())
}
