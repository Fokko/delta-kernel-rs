//! Integration tests for manifest commits with content trees during table creation.
//!
//! These tests verify that a `create_table` transaction can be combined with
//! `with_manifest_commit` and leaf writers so that the initial commit (version 0)
//! creates the table and builds a content tree with leaf manifests in one step.

use std::collections::HashSet;
use std::sync::Arc;

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::CommitResult;
use test_utils::{collect_file_paths, create_add_files_metadata, test_table_setup};

/// A `create_table` transaction with `metadataTree-experimental` and column mapping 'id'
/// combined with two leaf writers commits at version 0, produces a content root, and
/// makes all written files visible via a subsequent scan with no duplicates.
#[tokio::test]
async fn test_create_table_manifest_commit_with_leaves() -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup()?;
    let schema = Arc::new(StructType::try_new(vec![
        StructField::new("id", DataType::INTEGER, false),
        StructField::new("value", DataType::STRING, true),
    ])?);

    let mut txn = create_table(&table_path, schema, "TestEngine/1.0")
        .with_table_properties([
            ("delta.columnMapping.mode", "id"),
            ("delta.feature.metadataTree-experimental", "supported"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

    let add_files_schema = txn.add_files_schema();

    {
        let mc = txn.with_manifest_commit();

        let mut leaf1 = mc.new_leaf_node_writer(engine.as_ref())?;
        leaf1.add_files(
            engine.as_ref(),
            create_add_files_metadata(
                add_files_schema,
                vec![
                    ("leaf1-part1.parquet", 1024, 1_000_000, 10),
                    ("leaf1-part2.parquet", 2048, 1_000_001, 20),
                ],
            )?,
        )?;
        mc.add_leaf(leaf1.finish(engine.as_ref())?)?;

        let mut leaf2 = mc.new_leaf_node_writer(engine.as_ref())?;
        leaf2.add_files(
            engine.as_ref(),
            create_add_files_metadata(
                add_files_schema,
                vec![
                    ("leaf2-part1.parquet", 3072, 1_000_002, 30),
                    ("leaf2-part2.parquet", 4096, 1_000_003, 40),
                ],
            )?,
        )?;
        mc.add_leaf(leaf2.finish(engine.as_ref())?)?;
    }

    let committed = match txn.commit(engine.as_ref())? {
        CommitResult::CommittedTransaction(c) => c,
        other => panic!("Expected committed transaction, got {other:?}"),
    };
    assert_eq!(committed.commit_version(), 0);

    let snapshot =
        Snapshot::builder_for(delta_kernel::try_parse_uri(&table_path)?).build(engine.as_ref())?;
    assert_eq!(snapshot.version(), 0);
    assert!(
        snapshot.checkpoint_action().is_some(),
        "Manifest commit should produce a content root"
    );

    let paths = collect_file_paths(snapshot, engine.as_ref())?;
    let expected: HashSet<String> = [
        "leaf1-part1.parquet",
        "leaf1-part2.parquet",
        "leaf2-part1.parquet",
        "leaf2-part2.parquet",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        paths, expected,
        "Scan must return exactly the four written files with no duplicates"
    );

    Ok(())
}
