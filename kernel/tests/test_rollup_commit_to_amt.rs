//! Test root manifest filtering by writing a delta commit file with stats,
//! then doing a manifest commit to convert those stats to content_stats.

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::object_store::ObjectStoreExt as _;
use delta_kernel::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};
use delta_kernel::Snapshot;
use std::sync::Arc;
use test_utils::{create_table, engine_store_setup};

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
        vec!["columnMapping", "metadataTree-experimental"],
    )
    .await?;

    // Write version 1 with an Add action (similar to first test)
    let commit_json = r#"{"add":{"path":"part-00001.parquet","partitionValues":{},"size":100,"modificationTime":1,"dataChange":true,"stats":"{\"numRecords\":100,\"minValues\":{\"id\":1},\"maxValues\":{\"id\":100},\"nullCount\":{\"id\":0}}"}}
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
    txn.with_manifest_commit();
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
    txn.with_manifest_commit();
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
