//! Tests verifying that `TrackingInfo` (status, snapshot_id, sequence_number,
//! file_sequence_number) is correctly computed and preserved across multi-commit
//! scenarios for Added, Existed, and Deleted statuses.
//!
//! Each version is handled as a separate commit: the manifest is written to parquet,
//! then read back into a fresh builder via `from_content_root`, mirroring the
//! production round-trip.

use url::Url;

use crate::actions::{Add, CheckpointAction, ContentRoot, Metadata, Protocol};
use crate::content_tree::builder::ContentTreeNodeBuilder;
use crate::content_tree::writer::ContentTreeNodeWriter;
use crate::content_tree::{
    absolute_to_relative_path, ContentTreeNode, ContentTreeNodeEntry, DataContentType,
    TrackingStatus,
};
use crate::row_tracking::CursorRowIdAllocator;
use crate::schema::{ColumnMetadataKey, DataType, MetadataValue, Schema, StructField};
use crate::{DeltaResult, Version};

/// Minimal table schema with the required PARQUET:field_id metadata.
fn test_table_schema() -> Schema {
    Schema::new_unchecked([
        StructField::new("id", DataType::INTEGER, false).with_metadata([
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
                MetadataValue::String("col-id".to_string()),
            ),
        ]),
    ])
}

/// Creates an `Add` action with minimal fields.
fn make_add(path: &str, size: i64) -> Add {
    Add {
        path: path.to_string(),
        partition_values: Default::default(),
        size,
        modification_time: 0,
        data_change: true,
        stats: None,
        tags: None,
        deletion_vector: None,
        base_row_id: None,
        default_row_commit_version: None,
        clustering_provider: None,
        data_manifest_path: None,
        data_manifest_position: None,
    }
}

/// Creates a [`CheckpointAction`] from a written manifest path and version.
fn make_checkpoint_action(path: String, version: Version) -> CheckpointAction {
    CheckpointAction {
        version,
        content_root: ContentRoot {
            path,
            size_in_bytes: 0,
        },
        protocol: Protocol::try_new(1, 1, None::<Vec<String>>, None::<Vec<String>>).unwrap(),
        meta_data: Metadata::try_new(
            None,
            None,
            std::sync::Arc::new(crate::schema::StructType::new_unchecked([])),
            vec![],
            0,
            std::collections::HashMap::new(),
        )
        .unwrap(),
    }
}

/// Builds and writes a root manifest to parquet, returns the relative path.
fn write_root_manifest(
    builder: &mut ContentTreeNodeBuilder,
    engine: &dyn crate::Engine,
    table_root: &Url,
    snapshot_id: i64,
) -> DeltaResult<String> {
    let root = builder.build(engine, snapshot_id, &mut CursorRowIdAllocator::new(0))?;
    let root_url = ContentTreeNodeWriter::try_new(root)?
        .write(engine)?
        .location;
    absolute_to_relative_path(&root_url, table_root)
}

/// Builds a root manifest, writes it to parquet, reads it back, and returns the entries.
fn build_and_read_root(
    builder: &mut ContentTreeNodeBuilder,
    engine: &dyn crate::Engine,
    snapshot_id: i64,
) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
    let root_metadata = builder.build(engine, snapshot_id, &mut CursorRowIdAllocator::new(0))?;
    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &root_url, root_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    root.entries()
}

/// Builds a leaf manifest, writes it to parquet, reads it back, and returns the entries.
fn build_and_read_leaf(
    builder: &mut ContentTreeNodeBuilder,
    engine: &dyn crate::Engine,
    snapshot_id: i64,
) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
    let leaf_metadata =
        builder.build_leaf(engine, snapshot_id, &mut CursorRowIdAllocator::new(0))?;
    let table_root = leaf_metadata.table_root.clone();
    let leaf_url = ContentTreeNodeWriter::try_new(leaf_metadata)?
        .write(engine)?
        .location;
    let leaf_path = absolute_to_relative_path(&leaf_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &leaf_url, leaf_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let leaf = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    leaf.entries()
}

/// Finds an entry by its location path.
fn find_entry<'a>(entries: &'a [ContentTreeNodeEntry], path: &str) -> &'a ContentTreeNodeEntry {
    entries
        .iter()
        .find(|e| e.location.as_deref() == Some(path))
        .unwrap_or_else(|| panic!("entry with path '{path}' not found"))
}

/// Two sequential commits to the root manifest with full round-trip:
///   Version 1 (snapshot_id=1): Add file_a → write manifest
///   Version 2 (snapshot_id=2): Read V1 manifest → add file_b → read back
///
/// file_a becomes Existed at V2 (was Added at V1), file_b is Added at V2.
/// `from_content_root` flips Added→Existed for entries from prior versions.
#[test]
fn test_two_commits_to_root_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let engine = crate::engine::sync::SyncEngine::new();
    let temp_dir = tempfile::tempdir()?;
    let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

    // V1: add file_a and write the manifest
    let mut builder = ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
    builder.add(make_add("file_a.parquet", 1024), 1, 1)?;
    let v1_path = write_root_manifest(&mut builder, &engine, &table_root, 1)?;

    // V2: read V1 manifest into a fresh builder, add file_b, then read back
    let mut builder = ContentTreeNodeBuilder::from_content_root(
        &engine,
        &make_checkpoint_action(v1_path, 1).content_root,
        table_root,
        test_table_schema(),
        2,
    )?;
    builder.add(make_add("file_b.parquet", 2048), 2, 2)?;

    let entries = build_and_read_root(&mut builder, &engine, 2)?;
    assert_eq!(entries.len(), 2);

    // file_a was Added at V1, but now at V2 it should be Existed
    let a = find_entry(&entries, "file_a.parquet");
    let a_tracking = &a.tracking;
    assert_eq!(a_tracking.status, TrackingStatus::Existed);
    assert_eq!(a_tracking.snapshot_id, Some(1));
    assert_eq!(a_tracking.sequence_number, Some(1));
    assert_eq!(a_tracking.file_sequence_number, Some(1));

    // file_b is newly added at V2
    let b = find_entry(&entries, "file_b.parquet");
    let b_tracking = &b.tracking;
    assert_eq!(b_tracking.status, TrackingStatus::Added);
    assert_eq!(b_tracking.snapshot_id, Some(2));
    assert_eq!(b_tracking.sequence_number, Some(2));
    assert_eq!(b_tracking.file_sequence_number, Some(2));

    // snapshot_ids differ between entries
    assert_ne!(a_tracking.snapshot_id, b_tracking.snapshot_id);
    // sequence_numbers are sequential
    assert_eq!(
        a_tracking.sequence_number.unwrap() + 1,
        b_tracking.sequence_number.unwrap()
    );

    Ok(())
}

/// Two commits then moving entries to a leaf manifest at version 3 with full round-trip:
///   Version 1: Add file_a → write manifest
///   Version 2: Read V1 → add file_b → write manifest
///   Version 3: Read V2 → write leaf + read back
///
/// Both entries become Existed at V3 (were Added in prior versions).
/// The write_leaf produces a CombinedManifest entry.
#[test]
fn test_two_commits_move_to_leaf_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let engine = crate::engine::sync::SyncEngine::new();
    let temp_dir = tempfile::tempdir()?;
    let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

    // V1: add file_a and write the manifest
    let mut builder = ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
    builder.add(make_add("file_a.parquet", 1024), 1, 1)?;
    let v1_path = write_root_manifest(&mut builder, &engine, &table_root, 1)?;

    // V2: read V1 manifest, add file_b, write manifest
    let mut builder = ContentTreeNodeBuilder::from_content_root(
        &engine,
        &make_checkpoint_action(v1_path, 1).content_root,
        table_root.clone(),
        test_table_schema(),
        2,
    )?;
    builder.add(make_add("file_b.parquet", 2048), 2, 2)?;
    let v2_path = write_root_manifest(&mut builder, &engine, &table_root, 2)?;

    // V3: read V2 manifest into a fresh builder
    let mut builder = ContentTreeNodeBuilder::from_content_root(
        &engine,
        &make_checkpoint_action(v2_path, 2).content_root,
        table_root,
        test_table_schema(),
        3,
    )?;

    // Write as a leaf manifest and verify the CombinedManifest entry
    let manifest_entry = builder.write_leaf(&engine, 3, &mut CursorRowIdAllocator::new(0))?;
    assert_eq!(
        manifest_entry.content_type,
        DataContentType::CombinedManifest
    );
    let manifest_info = &manifest_entry.tracking;
    assert_eq!(manifest_info.status, TrackingStatus::Added);
    assert_eq!(manifest_info.snapshot_id, Some(3));

    // Verify min_sequence_number in manifest_stats
    let manifest_stats = manifest_entry
        .manifest_stats
        .as_ref()
        .expect("manifest_stats");
    assert_eq!(manifest_stats.min_sequence_number, 1);

    // Read back the leaf entries (pending_entries are preserved after write_leaf)
    let entries = build_and_read_leaf(&mut builder, &engine, 3)?;
    assert_eq!(entries.len(), 2);

    // Both entries were Added in prior versions, now at V3 they should be Existed
    let a = find_entry(&entries, "file_a.parquet");
    let a_tracking = &a.tracking;
    assert_eq!(a_tracking.status, TrackingStatus::Existed);
    assert_eq!(a_tracking.snapshot_id, Some(1));
    assert_eq!(a_tracking.sequence_number, Some(1));
    assert_eq!(a_tracking.file_sequence_number, Some(1));

    let b = find_entry(&entries, "file_b.parquet");
    let b_tracking = &b.tracking;
    assert_eq!(b_tracking.status, TrackingStatus::Existed);
    assert_eq!(b_tracking.snapshot_id, Some(2));
    assert_eq!(b_tracking.sequence_number, Some(2));
    assert_eq!(b_tracking.file_sequence_number, Some(2));

    // snapshot_ids differ between entries
    assert_ne!(a_tracking.snapshot_id, b_tracking.snapshot_id);
    // sequence_numbers are sequential
    assert_eq!(
        a_tracking.sequence_number.unwrap() + 1,
        b_tracking.sequence_number.unwrap()
    );

    Ok(())
}

/// Two commits then deleting the first file with full round-trip:
///   Version 1: Add file_a → write manifest
///   Version 2: Read V1 → add file_b → write manifest
///   Version 3: Read V2 → delete file_a → read back
///
/// file_a becomes Deleted (snapshot_id and sequence_number updated to V3,
/// file_sequence_number preserved from original add at V1).
/// file_b becomes Existed at V3 (was Added at V2).
#[test]
fn test_two_commits_delete_first_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let engine = crate::engine::sync::SyncEngine::new();
    let temp_dir = tempfile::tempdir()?;
    let table_root = Url::from_directory_path(temp_dir.path()).unwrap();

    // V1: add file_a and write the manifest
    let mut builder = ContentTreeNodeBuilder::new_for(table_root.clone(), 1, test_table_schema());
    builder.add(make_add("file_a.parquet", 1024), 1, 1)?;
    let v1_path = write_root_manifest(&mut builder, &engine, &table_root, 1)?;

    // V2: read V1 manifest, add file_b, write manifest
    let mut builder = ContentTreeNodeBuilder::from_content_root(
        &engine,
        &make_checkpoint_action(v1_path, 1).content_root,
        table_root.clone(),
        test_table_schema(),
        2,
    )?;
    builder.add(make_add("file_b.parquet", 2048), 2, 2)?;
    let v2_path = write_root_manifest(&mut builder, &engine, &table_root, 2)?;

    // V3: read V2 manifest, delete file_a, read back
    let mut builder = ContentTreeNodeBuilder::from_content_root(
        &engine,
        &make_checkpoint_action(v2_path, 2).content_root,
        table_root,
        test_table_schema(),
        3,
    )?;
    builder.mark_deleted(Some("file_a.parquet"), None, 3)?;

    let entries = build_and_read_root(&mut builder, &engine, 3)?;
    assert_eq!(entries.len(), 2);

    let a = find_entry(&entries, "file_a.parquet");
    let a_tracking = &a.tracking;
    assert_eq!(a_tracking.status, TrackingStatus::Deleted);
    // snapshot_id updated to deletion snapshot
    assert_eq!(a_tracking.snapshot_id, Some(3));
    // sequence_number preserved from original add
    assert_eq!(a_tracking.sequence_number, Some(1));
    // file_sequence_number preserved from original add
    assert_eq!(a_tracking.file_sequence_number, Some(1));

    // file_b was Added at V2, but now at V3 it should be Existed
    let b = find_entry(&entries, "file_b.parquet");
    let b_tracking = &b.tracking;
    assert_eq!(b_tracking.status, TrackingStatus::Existed);
    assert_eq!(b_tracking.snapshot_id, Some(2));
    assert_eq!(b_tracking.sequence_number, Some(2));
    assert_eq!(b_tracking.file_sequence_number, Some(2));

    // snapshot_ids differ between entries
    assert_ne!(a_tracking.snapshot_id, b_tracking.snapshot_id);

    Ok(())
}
