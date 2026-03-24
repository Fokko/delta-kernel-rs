use std::collections::HashMap;
use std::sync::Arc;

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::Snapshot;
use url::Url;

use delta_kernel::arrow::array::Int32Array;
use delta_kernel::arrow::record_batch::RecordBatch;

use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine_data::FilteredEngineData;
use delta_kernel::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};
use delta_kernel::transaction::CommitResult;

use itertools::Itertools;
use object_store::path::Path;
use object_store::ObjectStore;
use rstest::rstest;
use serde_json::Deserializer;
use tempfile::tempdir;

use test_utils::{create_table, engine_store_setup};

/// Test that verifies baseRowId and defaultRowCommitVersion are correctly populated
/// when row tracking is enabled on the table when a remove action is generated for a
/// a file that had row tracking enabled.
///
/// This test creates a table with row tracking enabled, writes data to it, and then
/// removes the data. It then verifies the remove action row ID fields. Propogating the
/// values is required by the delta protocol [1].
///
/// This complements the existing test `test_remove_files_adds_expected_entries` which
/// verifies that baseRowId and defaultRowCommitVersion are absent when row tracking is NOT enabled.
///
/// [1]: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#writer-requirements-for-row-tracking
#[rstest]
#[case::log_commit(false)]
#[case::batch_commit(true)]
#[tokio::test]
async fn test_row_tracking_fields_in_add_and_remove_actions(
    #[case] use_batch_commit: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    // Batch commit mode requires column mapping
    let (schema, reader_features, writer_features) = if use_batch_commit {
        let schema = Arc::new(StructType::try_new(vec![StructField::nullable(
            "number",
            DataType::INTEGER,
        )
        .with_metadata([
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
                MetadataValue::String("col-1".to_string()),
            ),
        ])])?);
        (
            schema,
            vec!["columnMapping", "metadataTree-experimental"],
            vec!["columnMapping", "metadataTree-experimental"],
        )
    } else {
        let schema = Arc::new(StructType::try_new(vec![StructField::nullable(
            "number",
            DataType::INTEGER,
        )])?);
        (schema, vec![], vec!["rowTracking", "domainMetadata"])
    };

    let tmp_dir = tempdir()?;
    let tmp_test_dir_url = Url::from_directory_path(tmp_dir.path()).unwrap();

    let (store, engine, table_location) =
        engine_store_setup("test_row_tracking", Some(&tmp_test_dir_url));

    let table_url = create_table(
        store.clone(),
        table_location,
        schema.clone(),
        &[],
        true,
        reader_features,
        writer_features,
    )
    .await?;

    // ===== FIRST COMMIT: Add files with row tracking =====
    let snapshot = Snapshot::builder_for(table_url.clone()).build(&engine)?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), &engine)?
        .with_engine_info("row tracking test")
        .with_data_change(true);

    if use_batch_commit {
        txn.with_batch_commit();
    }

    let data = RecordBatch::try_new(
        Arc::new(schema.as_ref().try_into_arrow()?),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
    )?;

    let engine_arc = Arc::new(engine);
    let write_context = Arc::new(txn.get_write_context());
    let add_files_metadata = engine_arc
        .write_parquet(
            &ArrowEngineData::new(data),
            write_context.as_ref(),
            HashMap::new(),
            &Default::default(),
        )
        .await?;

    txn.add_files(add_files_metadata);

    let result = txn.commit(engine_arc.as_ref())?;
    match result {
        CommitResult::CommittedTransaction(committed) => {
            assert_eq!(committed.commit_version(), 1);
        }
        _ => panic!("First commit should be committed"),
    }

    // ===== VERIFY ADD: Check row tracking fields =====
    let commit1_url = tmp_test_dir_url
        .join("test_row_tracking/_delta_log/00000000000000000001.json")
        .unwrap();
    let commit1 = store
        .get(&Path::from_url_path(commit1_url.path()).unwrap())
        .await?;

    let parsed_commits: Vec<_> = Deserializer::from_slice(&commit1.bytes().await?)
        .into_iter::<serde_json::Value>()
        .try_collect()?;

    // Verify HWM domain metadata is present for both modes
    let row_tracking_dm: Vec<_> = parsed_commits
        .iter()
        .filter_map(|action| {
            action.get("domainMetadata").filter(|meta| {
                meta.get("domain").and_then(|d| d.as_str()) == Some("delta.rowTracking")
            })
        })
        .collect();
    assert_eq!(
        row_tracking_dm.len(),
        1,
        "Expected exactly one row tracking domain metadata action"
    );
    let hwm_config: serde_json::Value =
        serde_json::from_str(row_tracking_dm[0]["configuration"].as_str().unwrap())?;
    let hwm = hwm_config["rowIdHighWaterMark"]
        .as_i64()
        .expect("rowIdHighWaterMark should be present");
    // 5 rows, starting at 0: HWM should be 4 (last row ID)
    assert_eq!(hwm, 4, "HWM should be num_records - 1");

    if !use_batch_commit {
        // For log commits, also verify JSON add action fields directly
        let add_actions: Vec<_> = parsed_commits
            .iter()
            .filter(|action| action.get("add").is_some())
            .collect();
        assert_eq!(add_actions.len(), 1, "Expected exactly one add action");

        let add = &add_actions[0]["add"];
        let base_row_id = add["baseRowId"]
            .as_i64()
            .expect("baseRowId should be an i64");
        assert_eq!(base_row_id, 0, "First file should have baseRowId 0");

        let default_row_commit_version = add["defaultRowCommitVersion"]
            .as_i64()
            .expect("Missing defaultRowCommitVersion");
        assert_eq!(default_row_commit_version, 1);
    } else {
        // For batch commits, add actions are in the content tree, not JSON
        let has_add_actions = parsed_commits
            .iter()
            .any(|action| action.get("add").is_some());
        assert!(
            !has_add_actions,
            "Batch commit should not have add actions in JSON log"
        );
    }

    // ===== SECOND COMMIT: Remove the file =====
    let snapshot2 = Snapshot::builder_for(table_url.clone()).build(engine_arc.as_ref())?;
    let mut txn2 = snapshot2
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine_arc.as_ref())?
        .with_engine_info("row tracking remove test")
        .with_data_change(true);

    if use_batch_commit {
        txn2.with_batch_commit();
    }

    let scan = snapshot2.scan_builder().build()?;
    let scan_metadata = scan.scan_metadata(engine_arc.as_ref())?.next().unwrap()?;

    let (data, selection_vector) = scan_metadata.scan_files.into_parts();
    let remove_metadata = FilteredEngineData::try_new(data, selection_vector)?;

    txn2.remove_files(remove_metadata);

    let result2 = txn2.commit(engine_arc.as_ref())?;
    match result2 {
        CommitResult::CommittedTransaction(committed) => {
            assert_eq!(committed.commit_version(), 2);
        }
        _ => panic!("Second commit should be committed"),
    }

    // ===== VERIFY REMOVE =====
    // Verify all files are removed via snapshot scan
    let snapshot3 = Snapshot::builder_for(table_url.clone())
        .at_version(2)
        .build(engine_arc.as_ref())?;
    let scan3 = snapshot3.scan_builder().build()?;
    let mut file_count = 0;
    for metadata in scan3.scan_metadata(engine_arc.as_ref())? {
        file_count += metadata?.scan_files.data().len();
    }
    assert_eq!(file_count, 0, "All files should be removed after commit 2");

    if !use_batch_commit {
        // For log commits, verify remove action row tracking fields in JSON
        let commit2_url = tmp_test_dir_url
            .join("test_row_tracking/_delta_log/00000000000000000002.json")
            .unwrap();
        let commit2 = store
            .get(&Path::from_url_path(commit2_url.path()).unwrap())
            .await?;

        let parsed_commits2: Vec<_> = Deserializer::from_slice(&commit2.bytes().await?)
            .into_iter::<serde_json::Value>()
            .try_collect()?;

        let remove_actions: Vec<_> = parsed_commits2
            .iter()
            .filter(|action| action.get("remove").is_some())
            .collect();
        assert_eq!(remove_actions.len(), 1);

        let remove = &remove_actions[0]["remove"];
        let remove_base_row_id = remove["baseRowId"].as_i64().expect("Missing baseRowId");
        assert_eq!(remove_base_row_id, 0);

        let remove_default_row_commit_version = remove["defaultRowCommitVersion"]
            .as_i64()
            .expect("Missing defaultRowCommitVersion");
        assert_eq!(remove_default_row_commit_version, 1);
    }

    Ok(())
}
