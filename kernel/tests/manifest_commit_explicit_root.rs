//! `Transaction::with_explicit_root_manifest` integration tests.

use std::sync::{Arc, Once};

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::transaction::CommitResult;
use delta_kernel::{FileMeta, Snapshot};
use test_utils::assert_result_error_with_message;
use url::Url;

#[path = "support/forbid_parquet_writes_engine.rs"]
mod forbid_parquet_writes_engine;
#[path = "support/manifest_commit_setup.rs"]
mod manifest_commit_setup;

use forbid_parquet_writes_engine::engine_forbid_parquet_writes;
use manifest_commit_setup::{
    add_files_to_transaction, create_column_mapping_schema, generate_and_add_data_file,
    setup_manifest_commit_test_tables, write_data_to_table,
};

static INIT_TRACING: Once = Once::new();

fn init_tracing() {
    INIT_TRACING.call_once(|| {
        let _ = tracing_subscriber::fmt::try_init();
    });
}

/// [`FileMeta`] for a would-be root manifest under `table_url` (no I/O; object may not exist yet).
fn new_root_file_meta(
    table_url: &Url,
    name: &str,
    size: u64,
) -> Result<FileMeta, Box<dyn std::error::Error>> {
    Ok(FileMeta::new(table_url.join(name)?, 0, size))
}

/// v0 create, v1 append, v2 manifest commit with checkpoint.
async fn table_v2_with_checkpoint(
    table_base: &str,
) -> Result<
    (
        Url,
        Arc<DefaultEngine<TokioBackgroundExecutor>>,
        Arc<DynObjectStore>,
        Arc<StructType>,
    ),
    Box<dyn std::error::Error>,
> {
    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;
    let (table_url, engine, store, _) =
        setup_manifest_commit_test_tables(schema.clone(), &[], table_base)
            .await?
            .into_iter()
            .next()
            .expect("one table");
    let engine = Arc::new(engine);
    write_data_to_table(&table_url, &engine, schema.clone(), vec![1, 2, 3]).await?;

    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("manifest commit")
        .with_operation("BATCH_COMMIT".to_string());
    txn.with_manifest_commit();
    add_files_to_transaction(&mut txn, &engine, schema.clone(), vec![7, 8, 9]).await?;
    match txn.commit(engine.as_ref())? {
        CommitResult::CommittedTransaction(c) => assert_eq!(c.commit_version(), 2),
        other => panic!("unexpected commit: {other:?}"),
    }

    let snap = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert_eq!(snap.version(), 2);
    assert!(snap.checkpoint_action().is_some());
    Ok((table_url, engine, store, schema))
}

#[tokio::test]
async fn test_explicit_root_manifest_commit_writes_supplied_content_root(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (table_url, engine, _store, _) = table_v2_with_checkpoint("set_new_root_positive").await?;
    let name = "custom-replacement-root.bin";
    let file_meta = new_root_file_meta(&table_url, name, 1024)?;

    let forbid_engine = engine_forbid_parquet_writes(engine.clone(), name.to_string());
    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), forbid_engine.as_ref())?
        .with_engine_info("set new root")
        .with_operation("SET_EXPLICIT_ROOT".to_string());
    txn.with_explicit_root_manifest(file_meta.clone())?;
    let v = match txn.commit(forbid_engine.as_ref())? {
        CommitResult::CommittedTransaction(c) => c.commit_version(),
        other => panic!("{other:?}"),
    };
    assert_eq!(v, 3);

    // Reload snapshot so `checkpoint_action` reflects log replay, not raw commit bytes.
    let snap = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert_eq!(snap.version(), 3);
    let ca = snap
        .checkpoint_action()
        .expect("v3 manifest commit should include a checkpoint action");
    assert!(ca.path().ends_with(name));
    assert_eq!(ca.content_root_size_in_bytes(), file_meta.size);
    Ok(())
}

#[tokio::test]
async fn test_explicit_root_manifest_requires_existing_checkpoint(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let schema = create_column_mapping_schema("number", DataType::INTEGER)?;
    let (table_url, engine, _store, _) =
        setup_manifest_commit_test_tables(schema.clone(), &[], "set_new_root_no_ckpt")
            .await?
            .into_iter()
            .next()
            .unwrap();
    let engine = Arc::new(engine);
    write_data_to_table(&table_url, &engine, schema, vec![1]).await?;
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert!(snapshot.checkpoint_action().is_none());

    let meta = new_root_file_meta(&table_url, "orphan.bin", 1)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test");
    assert_result_error_with_message(
        txn.with_explicit_root_manifest(meta),
        "requires an existing checkpoint action",
    );
    Ok(())
}

#[tokio::test]
async fn test_explicit_root_manifest_second_call_errors() -> Result<(), Box<dyn std::error::Error>>
{
    init_tracing();
    let (table_url, engine, _store, _) = table_v2_with_checkpoint("set_new_root_twice").await?;
    let a = new_root_file_meta(&table_url, "root-a.bin", 1)?;
    let b = new_root_file_meta(&table_url, "root-b.bin", 2)?;

    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test");
    txn.with_explicit_root_manifest(a)?;
    assert_result_error_with_message(
        txn.with_explicit_root_manifest(b),
        "explicit root manifest may only be set once per transaction",
    );
    Ok(())
}

#[tokio::test]
async fn test_explicit_root_manifest_errors_after_leaf_based_commit(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (table_url, engine, _store, _) =
        table_v2_with_checkpoint("set_new_root_after_leaf").await?;
    let meta = new_root_file_meta(&table_url, "root.bin", 1)?;

    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test");
    txn.with_manifest_commit();
    assert_result_error_with_message(
        txn.with_explicit_root_manifest(meta),
        "explicit root manifest and manifest commit are mutually exclusive",
    );
    Ok(())
}

#[tokio::test]
async fn test_leaf_based_commit_errors_at_commit_after_explicit_root_set(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (table_url, engine, _store, _) =
        table_v2_with_checkpoint("leaf_after_explicit_root").await?;
    let meta = new_root_file_meta(&table_url, "root.bin", 1)?;

    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test");
    txn.with_explicit_root_manifest(meta)?;
    txn.with_manifest_commit();
    assert_result_error_with_message(
        txn.commit(engine.as_ref()),
        "manifest commit and explicit root manifest are mutually exclusive",
    );
    Ok(())
}

#[tokio::test]
async fn test_explicit_root_manifest_commit_errors_with_add_files(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (table_url, engine, _store, schema) =
        table_v2_with_checkpoint("set_new_root_with_adds").await?;
    let meta = new_root_file_meta(&table_url, "root.bin", 1)?;

    let mut txn = Snapshot::builder_for(table_url.clone())
        .build(engine.as_ref())?
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test")
        .with_data_change(true);
    txn.with_explicit_root_manifest(meta)?;
    generate_and_add_data_file(&mut txn, engine.as_ref(), schema, vec![99]).await?;
    assert_result_error_with_message(txn.commit(engine.as_ref()), "cannot include add_files");
    Ok(())
}

#[tokio::test]
async fn test_explicit_root_manifest_rejects_path_outside_table_root(
) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let (table_url, engine, _, _) = table_v2_with_checkpoint("set_new_root_outside").await?;

    let cases: &[(&str, &str)] = &[
        // different path
        (
            "memory:///other_table_root/out.bin",
            "is not under the table root",
        ),
        // same path but different host (bucket)
        ("s3://bucket2/table1/out.bin", "is not under the table root"),
        // same path but different scheme (cloud provider)
        (
            "gs://anybucket/table1/out.bin",
            "is not under the table root",
        ),
    ];

    for (bad_url, expected_msg) in cases {
        let mut txn = Snapshot::builder_for(table_url.clone())
            .build(engine.as_ref())?
            .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
            .with_engine_info("test");
        let bad = FileMeta::new(Url::parse(bad_url)?, 0, 1);
        assert_result_error_with_message(txn.with_explicit_root_manifest(bad), expected_msg);
    }
    Ok(())
}
