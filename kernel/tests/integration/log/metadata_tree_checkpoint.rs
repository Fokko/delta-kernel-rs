//! Checkpoint writing on metadata tree (AMT) tables.
//!
//! A checkpoint is supposed to be a complete statement of a table's state at a version, which
//! is what makes it safe for a reader to start from one and ignore everything below it. On a
//! table whose file inventory lives in a content tree, producing that statement means reading
//! the tree.

use std::sync::Arc;

use delta_kernel::arrow::array::Int32Array;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::{DeltaResult, Engine, Snapshot};
use test_utils::{insert_data, test_table_setup_mt};
use url::Url;

use crate::common::amt_test_utils::collect_scanned_files;
use crate::common::manifest_commit_setup::{commit_at, create_manifest_commit_table, write_leaf};

/// Files folded into the content tree by [`build_metadata_tree_table`].
const FILES_IN_TREE: usize = 2;

/// Creates a metadata tree table whose single commit folds [`FILES_IN_TREE`] files into a leaf.
///
/// The files are never written to disk; nothing here reads past their metadata.
fn build_metadata_tree_table(
    table_path: &str,
    engine: &dyn Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut txn = create_manifest_commit_table(table_path, engine)?;
    let schema = txn.add_files_schema();
    txn.with_manifest_commit()?;
    write_leaf(
        &mut txn,
        engine,
        schema,
        vec![
            ("file1.parquet", 1024, 1_000_000, 10),
            ("file2.parquet", 2048, 1_000_001, 20),
        ],
    )?;
    commit_at(txn, engine, 0)?;
    Ok(())
}

/// The set of files a fresh snapshot of `url` reports, sorted.
fn live_paths(url: &Url, engine: &dyn Engine) -> DeltaResult<Vec<String>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine)?;
    let mut paths: Vec<String> = collect_scanned_files(snapshot, engine)?
        .file_paths
        .into_iter()
        .collect();
    paths.sort();
    Ok(paths)
}

/// Checkpointing a table must not change which files it has.
///
/// It does here, and it loses all of them. `CheckpointWriter` replays the log segment through
/// `LogSegment::read_actions`, which passes no checkpoint action and so never reaches the
/// content tree, even though the writer holds the snapshot that has one. The resulting
/// checkpoint records the tree's files nowhere. The content root is lost with them: the
/// pointer to it lives in a `checkpoint` action in the commit, and the checkpoint schema has
/// no field to carry one, so a reader starting from the checkpoint cannot recover the tree
/// either. An empty table is the result.
///
/// This is the one content tree gap whose bad answer is durable. Everything else returns a
/// wrong result to one caller, who at worst acts on it once; this writes the wrong result into
/// the log, together with a `_last_checkpoint` hint telling every later reader to trust it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "CheckpointWriter replays only the log segment, so checkpointing a metadata tree \
            table writes a checkpoint that omits every file in the tree (#249)"]
async fn checkpointing_a_metadata_tree_table_preserves_its_files(
) -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    build_metadata_tree_table(&table_path, engine.as_ref())?;
    let url = Url::from_directory_path(&table_path).expect("table path is absolute");

    let before = live_paths(&url, engine.as_ref())?;
    assert_eq!(
        before.len(),
        FILES_IN_TREE,
        "the tree should hold the files"
    );

    Snapshot::builder_for(url.clone())
        .build(engine.as_ref())?
        .checkpoint(engine.as_ref(), None)?;

    assert_eq!(
        live_paths(&url, engine.as_ref())?,
        before,
        "checkpointing must not change which files the table has"
    );

    Ok(())
}

/// The control: checkpointing does preserve files when they live in the log.
///
/// Without this the failure above could be read as the harness checkpointing wrongly rather
/// than the writer missing the tree.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn checkpointing_a_log_table_preserves_its_files() -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let schema = Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )])?);
    create_table(&table_path, schema, "TestEngine/1.0")
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_committed();
    let url = Url::from_directory_path(&table_path).expect("table path is absolute");

    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    insert_data(snapshot, &engine, vec![Arc::new(Int32Array::from(vec![1]))])
        .await?
        .unwrap_committed();

    let before = live_paths(&url, engine.as_ref())?;
    assert_eq!(before.len(), 1);

    Snapshot::builder_for(url.clone())
        .build(engine.as_ref())?
        .checkpoint(engine.as_ref(), None)?;

    assert_eq!(live_paths(&url, engine.as_ref())?, before);

    Ok(())
}
