//! Metrics for metadata tree (AMT) tables.
//!
//! These tables keep their file inventory in parquet manifests reached through the content
//! root rather than in `add` actions, which moves where the I/O happens: loading a snapshot
//! only reads the log, and the manifests are read later, during the scan. Every other shape in
//! [`super::snapshot_load`] pays for its file inventory up front, so none of them cover this.

use std::sync::Arc;

use delta_kernel::object_store::local::LocalFileSystem;
use delta_kernel::{DeltaResult, Snapshot};
use test_utils::test_table_setup;
use url::Url;

use super::measuring_engine;
use crate::common::manifest_commit_setup::{commit_at, create_manifest_commit_table, write_leaf};

/// Files written into the content tree by [`build_metadata_tree_table`].
const FILES_IN_TREE: u64 = 2;

/// Creates a metadata tree table whose only commit folds [`FILES_IN_TREE`] files into a leaf.
///
/// The files are never written to disk. Nothing here reads them: the scan stops at metadata.
fn build_metadata_tree_table() -> Result<(tempfile::TempDir, Url), Box<dyn std::error::Error>> {
    let (temp_dir, table_path, engine) = test_table_setup()?;
    let mut txn = create_manifest_commit_table(&table_path, engine.as_ref())?;
    let schema = txn.add_files_schema();
    txn.with_manifest_commit()?;
    write_leaf(
        &mut txn,
        engine.as_ref(),
        schema,
        vec![
            ("file1.parquet", 1024, 1_000_000, 10),
            ("file2.parquet", 2048, 1_000_001, 20),
        ],
    )?;
    commit_at(txn, engine.as_ref(), 0)?;

    Ok((temp_dir, Url::from_directory_path(&table_path).unwrap()))
}

/// Loading a snapshot of a metadata tree table reads the log and nothing else.
///
/// `checkpoint_files` counts classic checkpoint parts, and a content root is not one, so it
/// stays zero. That is not an undercount: the root really is not read here. It is read when
/// something asks for files, which is the next test.
#[test]
fn metadata_tree_snapshot_load_reads_only_the_log() -> DeltaResult<()> {
    let (_temp_dir, table_url) = build_metadata_tree_table().expect("table builds");

    let (engine, reporter, _guard) = measuring_engine(Arc::new(LocalFileSystem::new()));
    let _snap = Snapshot::builder_for(table_url).build(&engine)?;

    assert_eq!(reporter.snapshot_completions.get(), 1);
    assert_eq!(reporter.log_segment_loads.get(), 1);
    assert_eq!(reporter.commit_files.get(), 1);
    assert_eq!(reporter.checkpoint_files.get(), 0);
    assert_eq!(reporter.compaction_files.get(), 0);

    assert_eq!(reporter.json_read_calls.get(), 1);
    assert_eq!(
        reporter.parquet_read_calls.get(),
        0,
        "no manifest is read until something asks for files"
    );

    Ok(())
}

/// Listing a metadata tree table's files reads its manifests, and the metrics say so.
///
/// The root and the leaf are separate reads, so a caller watching parquet I/O sees the tree
/// being walked rather than a scan that appears to cost nothing.
#[test]
fn metadata_tree_scan_metadata_reads_the_manifests() -> DeltaResult<()> {
    let (_temp_dir, table_url) = build_metadata_tree_table().expect("table builds");

    let (engine, reporter, _guard) = measuring_engine(Arc::new(LocalFileSystem::new()));
    let snapshot = Snapshot::builder_for(table_url).build(&engine)?;
    reporter.reset();

    let scan = snapshot.scan_builder().build()?;
    let files: u64 = scan
        .scan_metadata(&engine)?
        .map(|metadata| {
            Ok(metadata?
                .scan_files
                .selection_vector()
                .iter()
                .filter(|s| **s)
                .count() as u64)
        })
        .sum::<DeltaResult<u64>>()?;
    assert_eq!(files, FILES_IN_TREE);

    assert!(
        reporter.parquet_read_calls.get() >= 2,
        "the root and the leaf are read separately, got {} call(s)",
        reporter.parquet_read_calls.get()
    );
    assert!(reporter.parquet_files_read.get() >= 2);

    Ok(())
}

/// The bytes read walking the content tree should include the root manifest.
///
/// The root is read through a `FileMeta` whose size is hardcoded to zero, and `bytes_read` is
/// sourced from those sizes, so the root's bytes never reach the reporter. Only the leaf's do.
/// That understates what a scan of a metadata tree table costs, by exactly the size of the one
/// file every such scan has to read.
#[test]
#[ignore = "the content root is read with FileMeta::size hardcoded to 0, so its bytes are \
            missing from parquet_bytes_read"]
fn metadata_tree_scan_reports_bytes_for_every_manifest_it_reads() -> DeltaResult<()> {
    let (_temp_dir, table_url) = build_metadata_tree_table().expect("table builds");

    let (engine, reporter, _guard) = measuring_engine(Arc::new(LocalFileSystem::new()));
    let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
    reporter.reset();

    let scan = snapshot.scan_builder().build()?;
    for metadata in scan.scan_metadata(&engine)? {
        let _ = metadata?;
    }

    let on_disk: u64 = walkdir::WalkDir::new(table_url.to_file_path().unwrap())
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    assert!(on_disk > 0, "the tree should have manifests on disk");
    assert_eq!(
        reporter.parquet_bytes_read.get(),
        on_disk,
        "every manifest byte read should be reported"
    );

    Ok(())
}
