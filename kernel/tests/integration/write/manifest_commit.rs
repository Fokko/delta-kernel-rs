//! Integration tests for the manifest-commit write path (metadata tree /
//! `metadataTree-experimental`).

use std::sync::Arc;

use delta_kernel::arrow::array::Int32Array;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::object_store::path::Path;
use delta_kernel::object_store::ObjectStoreExt as _;
use delta_kernel::schema::DataType;
use delta_kernel::transaction::CommitResult;
use delta_kernel::{DeltaResult, Snapshot};
use itertools::Itertools;
use serde_json::Deserializer;
use url::Url;

#[path = "../../support/manifest_commit_setup.rs"]
mod manifest_commit_setup;
use manifest_commit_setup::{
    add_files_to_transaction, create_column_mapping_schema, setup_manifest_commit_test_tables,
    write_data_to_table,
};

use crate::common::write_utils::{
    batch_write_data_and_check_result_and_stats, remove_all_scan_files,
    write_data_and_check_result_and_stats,
};

#[tokio::test]
async fn test_manifest_commit_no_add_actions() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    // Column mapping is required for manifest_commit mode.
    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;

    for (table_url, engine, store, table_name) in
        setup_manifest_commit_test_tables(schema.clone(), &[], "test_table").await?
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_engine_info("manifest commit test");
        let _ = txn.with_manifest_commit();

        assert!(txn.commit(&engine)?.is_committed());

        let commit1 = store
            .get(&Path::from(format!(
                "/{table_name}/_delta_log/00000000000000000001.json"
            )))
            .await?;

        let parsed_actions: Vec<_> = Deserializer::from_slice(&commit1.bytes().await?)
            .into_iter::<serde_json::Value>()
            .try_collect()?;

        // Manifest commit with no add files should only contain commitInfo.
        assert_eq!(parsed_actions.len(), 1, "Expected only commit info action");
        assert!(parsed_actions[0].get("commitInfo").is_some());
    }
    Ok(())
}

#[tokio::test]
async fn test_manifest_commit_with_add_files() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;

    for (table_url, engine, store, table_name) in
        setup_manifest_commit_test_tables(schema.clone(), &[], "test_table").await?
    {
        let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
        let mut txn = snapshot
            .transaction(Box::new(FileSystemCommitter::new()), &engine)?
            .with_engine_info("manifest commit test")
            .with_data_change(true);
        let _ = txn.with_manifest_commit();

        // Create two batches to append.
        let append_data = [[1, 2, 3], [4, 5, 6]].map(|data| -> DeltaResult<_> {
            let data = RecordBatch::try_new(
                Arc::new(schema.as_ref().try_into_arrow()?),
                vec![Arc::new(Int32Array::from(data.to_vec()))],
            )?;
            Ok(Box::new(ArrowEngineData::new(data)))
        });

        let engine = Arc::new(engine);
        let write_context = Arc::new(txn.unpartitioned_write_context()?);
        let tasks = append_data.into_iter().map(|data| {
            let engine = engine.clone();
            let write_context = write_context.clone();
            tokio::task::spawn(async move {
                engine
                    .write_parquet(data.as_ref().unwrap(), write_context.as_ref())
                    .await
            })
        });

        let add_files_metadata = futures::future::join_all(tasks).await.into_iter().flatten();
        for meta in add_files_metadata {
            txn.add_files(meta?);
        }

        let result = txn.commit(engine.as_ref())?;
        assert!(result.is_committed(), "Manifest commit should succeed");

        let commit1 = store
            .get(&Path::from(format!(
                "/{table_name}/_delta_log/00000000000000000001.json"
            )))
            .await?;

        let parsed_actions: Vec<_> = Deserializer::from_slice(&commit1.bytes().await?)
            .into_iter::<serde_json::Value>()
            .try_collect()?;

        // JSON log must contain commitInfo and a checkpoint action; no add actions.
        assert!(
            parsed_actions.iter().any(|a| a.get("commitInfo").is_some()),
            "Expected commitInfo action in commit. Actions: {:?}",
            parsed_actions
        );
        assert!(
            parsed_actions.iter().any(|a| a.get("checkpoint").is_some()),
            "Expected checkpoint action in commit. Actions: {:?}",
            parsed_actions
        );
        assert!(
            !parsed_actions.iter().any(|a| a.get("add").is_some()),
            "Manifest commit should not write add actions to JSON log"
        );
    }
    Ok(())
}

/// Verifies that a manifest commit creates a `contentRoot` action detected during log replay.
///
/// Steps:
/// 1. Creates a table with initial data (commit 1).
/// 2. Performs a manifest commit (commit 2) which writes a `contentRoot` action.
/// 3. Asserts `contentRoot` is present in the commit JSON.
/// 4. Builds a fresh `Snapshot` and confirms the log segment is correct.
#[tokio::test]
async fn test_manifest_commit_content_root_detected_in_scan(
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;

    for (table_url, engine, store, _table_name) in
        setup_manifest_commit_test_tables(schema.clone(), &[], "manifest_commit_test").await?
    {
        let engine = Arc::new(engine);

        // Commit 1: append initial data via a normal (non-manifest) write.
        write_data_to_table(&table_url, &engine, schema.clone(), vec![1, 2, 3]).await?;
        let snapshot1 = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        assert_eq!(snapshot1.version(), 1);

        // Commit 2: manifest commit with additional data.
        let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        let mut manifest_txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
            .with_engine_info("manifest commit test")
            .with_operation("BATCH_COMMIT".to_string());
        let _ = manifest_txn.with_manifest_commit();

        add_files_to_transaction(&mut manifest_txn, &engine, schema.clone(), vec![7, 8, 9]).await?;

        let manifest_result = manifest_txn.commit(engine.as_ref())?;
        let manifest_version = match manifest_result {
            CommitResult::CommittedTransaction(committed) => {
                assert_eq!(committed.commit_version(), 2);
                committed.commit_version()
            }
            _ => panic!("Manifest commit should succeed"),
        };

        // Commit 2 must contain a `contentRoot` action.
        let table_path = table_url
            .path()
            .trim_start_matches('/')
            .trim_end_matches('/');
        let commit2 = store
            .get(&Path::from(format!(
                "{table_path}/_delta_log/00000000000000000002.json"
            )))
            .await?;
        let commit_content = String::from_utf8(commit2.bytes().await?.to_vec())?;
        assert!(
            commit_content.contains("contentRoot"),
            "Manifest commit should contain a contentRoot action. Commit content: {}",
            commit_content
        );

        // A fresh Snapshot must see the correct version and log segment.
        let fresh_snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        assert_eq!(fresh_snapshot.version(), 2);
        let log_segment = fresh_snapshot.log_segment();
        assert_eq!(log_segment.end_version, 2);
        assert_eq!(log_segment.listed.ascending_commit_files.len(), 3); // commits 0, 1, 2
        assert_eq!(manifest_version, 2);
    }

    Ok(())
}

/// Stages a batch remove-all-files transaction for each test table and passes the ready
/// transaction, engine, and table URL to `on_commit` for the caller to commit and assert.
async fn batch_remove_all_files_impl(
    with_existing_root: bool,
    mut on_commit: impl FnMut(
        delta_kernel::transaction::Transaction,
        Arc<DefaultEngine<TokioBackgroundExecutor>>,
        Url,
    ) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;

    for (table_url, engine, _store, _table_name) in
        setup_manifest_commit_test_tables(schema.clone(), &[], "test_table").await?
    {
        let engine = Arc::new(engine);

        if with_existing_root {
            // Establish a checkpoint action via a batch (manifest) write.
            batch_write_data_and_check_result_and_stats(
                table_url.clone(),
                schema.clone(),
                engine.clone(),
                1,
            )
            .await?;
        } else {
            // Add files via a non-manifest commit — no checkpoint action established.
            write_data_and_check_result_and_stats(
                table_url.clone(),
                schema.clone(),
                engine.clone(),
                1,
            )
            .await?;
        }

        let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        assert_eq!(snapshot.checkpoint_action().is_some(), with_existing_root);

        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
            .with_engine_info("test engine")
            .with_operation("DELETE".to_string())
            .with_data_change(true);
        let _ = txn.with_manifest_commit();

        let removed =
            remove_all_scan_files(&mut txn, snapshot.scan_builder().build()?, engine.as_ref())?;
        assert!(removed > 0);

        on_commit(txn, engine, table_url)?;
    }
    Ok(())
}

#[tokio::test]
async fn test_remove_files_manifest_commit_mode() -> Result<(), Box<dyn std::error::Error>> {
    // remove_files in manifest commit mode without an existing checkpoint action must fail.
    batch_remove_all_files_impl(false, |txn, engine, _url| {
        assert!(
            txn.commit(engine.as_ref()).is_err(),
            "expected error when removing files in manifest commit mode without a checkpoint action"
        );
        Ok(())
    })
    .await
}

#[tokio::test]
async fn test_remove_files_manifest_commit_mode_with_existing_root(
) -> Result<(), Box<dyn std::error::Error>> {
    // remove_files in manifest commit mode succeeds when a checkpoint action already exists.
    batch_remove_all_files_impl(true, |txn, engine, table_url| {
        match txn.commit(engine.as_ref())? {
            CommitResult::CommittedTransaction(committed) => {
                let new_snapshot = Snapshot::builder_for(table_url)
                    .at_version(committed.commit_version())
                    .build(engine.as_ref())?;
                let mut file_count = 0;
                for metadata in new_snapshot
                    .scan_builder()
                    .build()?
                    .scan_metadata(engine.as_ref())?
                {
                    file_count += metadata?.scan_files.data().len();
                }
                assert_eq!(
                    file_count, 0,
                    "all files should be removed after batch remove commit"
                );
            }
            _ => panic!("expected committed transaction"),
        }
        Ok(())
    })
    .await
}
