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
        .tracking_info(TrackingInfo {
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
    assert_eq!(
        entries[0].tracking_info.as_ref().unwrap().first_row_id,
        Some(42)
    );
    assert_eq!(
        entries[1].tracking_info.as_ref().unwrap().first_row_id,
        Some(142)
    );

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
    assert_eq!(
        entries[0].tracking_info.as_ref().unwrap().first_row_id,
        Some(0)
    );
    // Deleted entry has null first_row_id
    assert_eq!(
        entries[1].tracking_info.as_ref().unwrap().first_row_id,
        None
    );
    assert_eq!(
        entries[2].tracking_info.as_ref().unwrap().first_row_id,
        Some(100)
    );

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
    assert_eq!(
        entries[0].tracking_info.as_ref().unwrap().first_row_id,
        Some(501)
    );
    assert_eq!(
        entries[1].tracking_info.as_ref().unwrap().first_row_id,
        Some(601)
    );

    Ok(())
}
