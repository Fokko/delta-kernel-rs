//! Tests for `first_row_id` assignment that exercise the public builder API end-to-end,
//! including parquet write/read round-trips.
//!
//! Unit tests for the internal `assign_first_row_ids_to_pending` method live in `builder::tests`
//! where they can access private fields directly.

use std::sync::Arc;

use rstest::rstest;

use crate::content_tree::builder::ContentTreeNodeBuilder;
use crate::content_tree::writer::ContentTreeNodeWriter;
use crate::content_tree::{
    absolute_to_relative_path, ContentTreeNode, ContentTreeNodeEntry, ContentTreeNodeEntryBuilder,
    DataContentType, ManifestInfo, TrackingInfo, TrackingStatus,
};
use crate::engine::default::executor::tokio::TokioBackgroundExecutor;
use crate::engine::default::{DefaultEngine, DefaultEngineBuilder};
use crate::object_store::local::LocalFileSystem;
use crate::row_tracking::CursorRowIdAllocator;
use crate::schema::{ColumnMetadataKey, DataType, MetadataValue, Schema, StructField};
use crate::{DeltaResult, Engine};

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

fn make_data_entry(path: &str, record_count: i64, status: TrackingStatus) -> ContentTreeNodeEntry {
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

/// Creates a [`DefaultEngine`] backed by local storage and a [`ContentTreeNodeBuilder`]
/// rooted at a temporary directory.
fn setup_engine_and_builder() -> (
    DefaultEngine<TokioBackgroundExecutor>,
    ContentTreeNodeBuilder,
) {
    let temp_path = tempfile::tempdir().unwrap().keep();
    let store = Arc::new(LocalFileSystem::new());
    let engine = DefaultEngineBuilder::new(store).build();
    let table_root = url::Url::from_directory_path(&temp_path).unwrap();
    let builder = ContentTreeNodeBuilder::new_for(table_root, 1, test_table_schema());
    (engine, builder)
}

/// Builds the manifest, writes it to parquet, reads it back, and returns the
/// round-tripped entries. Shared by every test in this file.
fn build_and_roundtrip(
    engine: &DefaultEngine<TokioBackgroundExecutor>,
    mut builder: ContentTreeNodeBuilder,
    allocator: &mut CursorRowIdAllocator,
) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
    let root_metadata = builder.build(engine, 1, allocator)?;
    let table_root = root_metadata.table_root.clone();
    let root_url = ContentTreeNodeWriter::try_new(root_metadata)?
        .write(engine)?
        .location;
    let root_path = absolute_to_relative_path(&root_url, &table_root);
    let (iter, version, path_in_log) = ContentTreeNode::open_stream(
        engine.parquet_handler(),
        &root_url,
        root_path,
        None,
        None,
        None,
    )?;
    let data = iter.collect::<DeltaResult<Vec<_>>>()?;
    let root = ContentTreeNode::from_batches_with_version(data, version, path_in_log, table_root)?;
    root.entries()
}

/// Builds a root manifest from plain data entries, writes/reads it, and verifies
/// `first_row_id` assignment under different mixes of statuses and starting HWMs.
///
/// Deleted entries do not consume IDs and receive null `first_row_id` after the
/// round-trip; Added entries consume `record_count` IDs each contiguously.
#[rstest]
#[case::two_added_from_42(
    vec![(100, TrackingStatus::Added), (200, TrackingStatus::Added)],
    42,
    342,
    vec![Some(42), Some(142)],
)]
#[case::deleted_entry_yields_null(
    vec![
        (100, TrackingStatus::Added),
        (200, TrackingStatus::Deleted),
        (50, TrackingStatus::Added),
    ],
    0,
    150,
    vec![Some(0), None, Some(100)],
)]
#[case::nonzero_starting_hwm(
    vec![(100, TrackingStatus::Added), (200, TrackingStatus::Added)],
    501,
    801,
    vec![Some(501), Some(601)],
)]
fn test_first_row_id_data_entries_roundtrip(
    #[case] entries: Vec<(i64, TrackingStatus)>,
    #[case] starting_hwm: i64,
    #[case] expected_current: i64,
    #[case] expected_first_row_ids: Vec<Option<i64>>,
) -> DeltaResult<()> {
    let (engine, mut builder) = setup_engine_and_builder();

    for (i, (record_count, status)) in entries.iter().enumerate() {
        builder.add_entry(make_data_entry(
            &format!("file-{i}.parquet"),
            *record_count,
            *status,
        ));
    }

    let mut allocator = CursorRowIdAllocator::new(starting_hwm);
    let result_entries = build_and_roundtrip(&engine, builder, &mut allocator)?;
    assert_eq!(allocator.current(), expected_current);

    assert_eq!(result_entries.len(), expected_first_row_ids.len());
    for (i, expected) in expected_first_row_ids.iter().enumerate() {
        assert_eq!(result_entries[i].tracking.first_row_id, *expected);
    }
    Ok(())
}

/// Builds a root manifest with DataManifest entries, verifies first_row_id
/// assignment uses added_rows_count + existing_rows_count (matching Iceberg's
/// manifest list first_row_id computation), and survives a parquet round-trip.
#[test]
fn test_first_row_id_combined_manifest_entries_roundtrip() -> DeltaResult<()> {
    let (engine, mut builder) = setup_engine_and_builder();

    // Manifest 1: 100 added + 200 existing = 300 row ID slots
    builder.add_entry(
        ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
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
            .manifest_info(ManifestInfo {
                added_files_count: 2,
                existing_files_count: 3,
                deleted_files_count: 0,
                replaced_files_count: 0,
                added_rows_count: 100,
                existing_rows_count: 200,
                deleted_rows_count: 0,
                replaced_rows_count: 0,
                min_sequence_number: 1,
                dv: None,
                dv_cardinality: None,
            })
            .build(),
    );

    // Manifest 2: 50 added + 50 existing = 100 row ID slots
    builder.add_entry(
        ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
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
            .manifest_info(ManifestInfo {
                added_files_count: 1,
                existing_files_count: 1,
                deleted_files_count: 0,
                replaced_files_count: 0,
                added_rows_count: 50,
                existing_rows_count: 50,
                deleted_rows_count: 0,
                replaced_rows_count: 0,
                min_sequence_number: 1,
                dv: None,
                dv_cardinality: None,
            })
            .build(),
    );

    let mut allocator = CursorRowIdAllocator::new(1000);
    let entries = build_and_roundtrip(&engine, builder, &mut allocator)?;
    // 1000 + 300 + 100 = 1400
    assert_eq!(allocator.current(), 1400);

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
    let (engine, mut builder) = setup_engine_and_builder();

    // Existed file with pre-assigned first_row_id from a previous commit
    let mut existed_entry = make_data_entry("existed-file.parquet", 100, TrackingStatus::Existing);
    existed_entry.tracking.first_row_id = Some(500);
    builder.add_entry(existed_entry);

    // New Added file (should be assigned IDs starting after the existed file's range)
    builder.add_entry(make_data_entry(
        "new-file.parquet",
        50,
        TrackingStatus::Added,
    ));

    // Allocator starts at HWM+1 = 600 (existed entry covers [500, 600))
    let mut allocator = CursorRowIdAllocator::new(600);
    let entries = build_and_roundtrip(&engine, builder, &mut allocator)?;
    // Added file gets [600, 650)
    assert_eq!(allocator.current(), 650);

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
