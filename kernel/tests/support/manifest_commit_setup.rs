//! Shared helpers for manifest-commit integration tests (metadata tree).

use std::sync::Arc;

use delta_kernel::arrow::array::Int32Array;
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::object_store::DynObjectStore;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::transaction::{CommitResult, Transaction};
use delta_kernel::{Snapshot, Version};
use test_utils::{create_table, engine_store_setup};
use url::Url;

/// Create a simple schema with column mapping enabled (required for manifest_commit mode).
pub fn create_column_mapping_schema(
    field_name: &str,
    data_type: DataType,
) -> Result<Arc<StructType>, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![StructField::nullable(
        field_name, data_type,
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
    ])])?))
}

/// Creates tables with MetadataTreeExperimental feature for manifest commit tests.
pub async fn setup_manifest_commit_test_tables(
    schema: Arc<StructType>,
    partition_columns: &[&str],
    table_base_name: &str,
) -> Result<
    Vec<(
        Url,
        DefaultEngine<TokioBackgroundExecutor>,
        Arc<DynObjectStore>,
        &'static str,
    )>,
    Box<dyn std::error::Error>,
> {
    let table_name_37 = format!("{table_base_name}_37");
    let (store_37, engine_37, table_location_37) = engine_store_setup(table_name_37.as_str(), None);

    Ok(vec![(
        create_table(
            store_37.clone(),
            table_location_37,
            schema.clone(),
            partition_columns,
            true,
            vec!["columnMapping", "metadataTree-experimental"],
            vec!["columnMapping", "metadataTree-experimental"],
        )
        .await?,
        engine_37,
        store_37,
        "test_table_37",
    )])
}

/// Write one append commit of parquet data to the table.
pub async fn write_data_to_table(
    table_url: &Url,
    engine: &Arc<DefaultEngine<TokioBackgroundExecutor>>,
    schema: SchemaRef,
    values: Vec<i32>,
) -> Result<Version, Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("test");

    add_files_to_transaction(&mut txn, engine, schema, values).await?;

    let result = txn.commit(engine.as_ref())?;
    match result {
        CommitResult::CommittedTransaction(committed) => Ok(committed.commit_version()),
        _ => panic!("Transaction should be committed"),
    }
}

/// Add parquet rows to an open transaction.
pub async fn add_files_to_transaction(
    txn: &mut Transaction,
    engine: &Arc<DefaultEngine<TokioBackgroundExecutor>>,
    schema: SchemaRef,
    values: Vec<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = RecordBatch::try_new(
        Arc::new(schema.as_ref().try_into_arrow()?),
        vec![Arc::new(Int32Array::from(values))],
    )?;

    let write_context = Arc::new(txn.unpartitioned_write_context()?);
    let add_files_metadata = engine
        .write_parquet(&ArrowEngineData::new(data), write_context.as_ref())
        .await?;
    txn.add_files(add_files_metadata);
    Ok(())
}

/// Generate one parquet file from `values` and append via `add_files`.
pub async fn generate_and_add_data_file(
    txn: &mut Transaction,
    engine: &DefaultEngine<TokioBackgroundExecutor>,
    schema: SchemaRef,
    values: Vec<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = RecordBatch::try_new(
        Arc::new(schema.as_ref().try_into_arrow()?),
        vec![Arc::new(Int32Array::from(values))],
    )?;

    let write_context = Arc::new(txn.unpartitioned_write_context()?);
    let file_meta = engine
        .write_parquet(&ArrowEngineData::new(data), write_context.as_ref())
        .await?;
    txn.add_files(file_meta);
    Ok(())
}
