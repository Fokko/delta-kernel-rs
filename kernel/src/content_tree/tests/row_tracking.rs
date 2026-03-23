//! Tests for `first_row_id` assignment that exercise the public builder API end-to-end,
//! including parquet write/read round-trips.
//!
//! Unit tests for the internal `assign_first_row_ids` method live in `builder::tests`
//! where they can access private fields directly.

use std::sync::Arc;

use crate::Engine;

use crate::content_tree::builder::ContentTreeNodeBuilder;
use crate::content_tree::writer::ContentTreeNodeWriter;
use crate::content_tree::{
    absolute_to_relative_path, ContentTreeNode, ContentTreeNodeEntryBuilder, DataContentType,
    TrackingInfo, TrackingStatus,
};
use crate::schema::{ColumnMetadataKey, DataType, MetadataValue, Schema, StructField};
use crate::DeltaResult;

fn test_table_schema() -> Schema {
    Schema::new_unchecked([
        StructField::new("id", DataType::INTEGER, false).with_metadata([(
            ColumnMetadataKey::ParquetFieldId.as_ref(),
            MetadataValue::Number(1),
        )]),
        StructField::new("value", DataType::STRING, true).with_metadata([(
            ColumnMetadataKey::ParquetFieldId.as_ref(),
            MetadataValue::Number(2),
        )]),
    ])
}

fn make_data_entry(
    path: &str,
    record_count: i64,
    status: TrackingStatus,
) -> crate::content_tree::ContentTreeNodeEntry {
    ContentTreeNodeEntryBuilder::new(DataContentType::Data)
        .location(path)
        .tracking(TrackingInfo {
            status,
            snapshot_id: Some(1),
            sequence_number: Some(1),
            file_sequence_number: Some(1),
            first_row_id: None,
            changes_dv: None,
        })
        .record_count(record_count)
        .file_size_in_bytes(1024)
        .build()
}

/// Builds a root manifest with row tracking, writes to parquet, reads back,
/// and verifies that first_row_id values survive the round-trip.
#[test]
fn test_first_row_id_roundtrip_through_root_manifest() -> DeltaResult<()> {
    use crate::engine::default::DefaultEngineBuilder;
    use object_store::local::LocalFileSystem;

    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();

    let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

    builder.add_entry(make_data_entry(
        "file-a.parquet",
        100,
        TrackingStatus::Added,
    ));
    builder.add_entry(make_data_entry(
        "file-b.parquet",
        200,
        TrackingStatus::Added,
    ));

    // Build with row tracking starting at 42
    let (root_metadata, next_row_id) = builder.build(&engine, 1, 42)?;
    assert_eq!(next_row_id, 342);

    // Write to parquet and read back
    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(&engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &root_url, root_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    let entries = root.entries()?;

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].tracking.first_row_id, Some(42));
    assert_eq!(entries[1].tracking.first_row_id, Some(142));

    Ok(())
}

/// Verifies that deleted entries receive null first_row_id after a round-trip,
/// and that subsequent entries are assigned correctly.
#[test]
fn test_first_row_id_deleted_entries_null_after_roundtrip() -> DeltaResult<()> {
    use crate::engine::default::DefaultEngineBuilder;
    use object_store::local::LocalFileSystem;

    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();

    let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

    builder.add_entry(make_data_entry(
        "file-a.parquet",
        100,
        TrackingStatus::Added,
    ));
    builder.add_entry(make_data_entry(
        "file-deleted.parquet",
        200,
        TrackingStatus::Deleted,
    ));
    builder.add_entry(make_data_entry("file-b.parquet", 50, TrackingStatus::Added));

    let (root_metadata, next_row_id) = builder.build(&engine, 1, 0)?;
    // Deleted entry does not consume IDs: 0 + 100 + 50 = 150
    assert_eq!(next_row_id, 150);

    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(&engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &root_url, root_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    let entries = root.entries()?;

    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].tracking.first_row_id, Some(0));
    // Deleted entry has null first_row_id
    assert_eq!(entries[1].tracking.first_row_id, None);
    assert_eq!(entries[2].tracking.first_row_id, Some(100));

    Ok(())
}

/// Verifies that a nonzero starting HWM offsets all assigned first_row_id values correctly
/// after a round-trip.
#[test]
fn test_first_row_id_nonzero_hwm_roundtrip() -> DeltaResult<()> {
    use crate::engine::default::DefaultEngineBuilder;
    use object_store::local::LocalFileSystem;

    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();

    let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

    builder.add_entry(make_data_entry(
        "file-a.parquet",
        100,
        TrackingStatus::Added,
    ));
    builder.add_entry(make_data_entry(
        "file-b.parquet",
        200,
        TrackingStatus::Added,
    ));

    // Starting from HWM of 500 (so starting_row_id = 501)
    let (root_metadata, next_row_id) = builder.build(&engine, 1, 501)?;
    assert_eq!(next_row_id, 801);

    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(&engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &root_url, root_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    let entries = root.entries()?;

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].tracking.first_row_id, Some(501));
    assert_eq!(entries[1].tracking.first_row_id, Some(601));

    Ok(())
}

/// Builds a root manifest with CombinedManifest entries, verifies first_row_id
/// assignment uses added_rows_count + existing_rows_count (matching Iceberg's
/// manifest list first_row_id computation), and survives a parquet round-trip.
#[test]
fn test_first_row_id_combined_manifest_entries_roundtrip() -> DeltaResult<()> {
    use crate::content_tree::builder::ContentTreeNodeBuilder;
    use crate::content_tree::writer::ContentTreeNodeWriter;
    use crate::content_tree::{
        absolute_to_relative_path, ContentTreeNode, ContentTreeNodeEntryBuilder, DataContentType,
        ManifestStats, TrackingInfo, TrackingStatus,
    };
    use crate::engine::default::DefaultEngineBuilder;
    use object_store::local::LocalFileSystem;

    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();

    let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

    // Manifest 1: 100 added + 200 existing = 300 row ID slots
    builder.add_entry(
        ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location("manifest-a.parquet")
            .tracking(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: None,
                file_sequence_number: None,
                first_row_id: None,
                changes_dv: None,
            })
            .record_count(300)
            .file_size_in_bytes(4096)
            .manifest_stats(ManifestStats {
                added_files_count: 2,
                existing_files_count: 3,
                deletes_files_count: 0,
                added_rows_count: 100,
                existing_rows_count: 200,
                delete_rows_count: 0,
                min_sequence_number: 1,
            })
            .build(),
    );

    // Manifest 2: 50 added + 50 existing = 100 row ID slots
    builder.add_entry(
        ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location("manifest-b.parquet")
            .tracking(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: None,
                file_sequence_number: None,
                first_row_id: None,
                changes_dv: None,
            })
            .record_count(100)
            .file_size_in_bytes(2048)
            .manifest_stats(ManifestStats {
                added_files_count: 1,
                existing_files_count: 1,
                deletes_files_count: 0,
                added_rows_count: 50,
                existing_rows_count: 50,
                delete_rows_count: 0,
                min_sequence_number: 1,
            })
            .build(),
    );

    let starting_row_id = 1000;
    let (root_metadata, next_row_id) = builder.build(&engine, 1, starting_row_id)?;
    // 1000 + 300 + 100 = 1400
    assert_eq!(next_row_id, 1400);

    // Write and read back
    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(&engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) =
        ContentTreeNode::open_stream(engine.parquet_handler(), &root_url, root_path, None, None)?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    let entries = root.entries()?;

    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].tracking.first_row_id,
        Some(1000),
        "First manifest should start at 1000"
    );
    assert_eq!(
        entries[1].tracking.first_row_id,
        Some(1300),
        "Second manifest should start at 1000 + 300 = 1300"
    );

    Ok(())
}

/// Mix of Existed (with pre-assigned IDs) and Added entries preserves existing IDs
/// and assigns new ones contiguously, matching Iceberg's requirement that existing
/// manifest entries preserve their first_row_id.
#[test]
fn test_first_row_id_mixed_existed_and_added_roundtrip() -> DeltaResult<()> {
    use crate::engine::default::DefaultEngineBuilder;
    use object_store::local::LocalFileSystem;

    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();

    let mut builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());

    // Existed file with pre-assigned first_row_id from a previous commit
    let mut existed_entry = make_data_entry("existed-file.parquet", 100, TrackingStatus::Existed);
    existed_entry.tracking.first_row_id = Some(500);
    builder.add_entry(existed_entry);

    // New Added file (should be assigned IDs starting after the existed file's range)
    builder.add_entry(make_data_entry(
        "new-file.parquet",
        50,
        TrackingStatus::Added,
    ));

    let (root_metadata, next_row_id) = builder.build(&engine, 1, 0)?;
    // Existed file has range [500, 600), cursor jumps to 600
    // Added file gets [600, 650)
    assert_eq!(next_row_id, 650);

    // Write and read back
    let table_root = root_metadata.table_root.clone();
    let root_url = crate::content_tree::writer::ContentTreeNodeWriter::try_new(root_metadata)?
        .write(&engine)?
        .location;
    let root_path = crate::content_tree::absolute_to_relative_path(&root_url, &table_root)?;
    let (iter, version, path_in_log) = crate::content_tree::ContentTreeNode::open_stream(
        engine.parquet_handler(),
        &root_url,
        root_path,
        None,
        None,
    )?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = crate::content_tree::ContentTreeNode::from_batches_with_version(
        data,
        version,
        path_in_log,
        table_root,
    )?;
    let entries = root.entries()?;

    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].tracking.first_row_id,
        Some(500),
        "Existed entry should preserve its original first_row_id"
    );
    assert_eq!(
        entries[1].tracking.first_row_id,
        Some(600),
        "Added entry should start after the existed entry's range"
    );

    Ok(())
}
