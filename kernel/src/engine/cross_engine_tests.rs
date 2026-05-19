//! Cross-engine tests: verifies that both the default (Arrow/Tokio) and sync engines exhibit
//! consistent behavior for [`JsonHandler`] and [`ParquetHandler`].
//!
//! Contract tests (things any [`ParquetHandler`] implementation must satisfy) call into
//! [`super::tests`]. Internal implementation tests (Arrow-specific behavior that both kernel
//! engines share) are defined as local helpers here.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use rstest::rstest;
use tempfile::tempdir;
use url::Url;

use crate::arrow::array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use crate::arrow::datatypes::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
use crate::engine::arrow_conversion::TryIntoKernel as _;
use crate::engine::arrow_data::ArrowEngineData;
use crate::engine::default::executor::tokio::TokioBackgroundExecutor;
use crate::engine::default::json::DefaultJsonHandler;
use crate::engine::default::parquet::DefaultParquetHandler;
use crate::engine::sync::json::SyncJsonHandler;
use crate::engine::sync::SyncParquetHandler;
use crate::object_store::local::LocalFileSystem;
use crate::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use crate::parquet::arrow::arrow_writer::ArrowWriter;
use crate::parquet::arrow::{ARROW_SCHEMA_META_KEY, PARQUET_FIELD_ID_META_KEY};
use crate::schema::{ColumnMetadataKey, DataType, MetadataValue, StructField, StructType};
use crate::{EngineData, JsonHandler, ParquetHandler};

fn default_parquet_handler() -> Box<dyn ParquetHandler> {
    Box::new(DefaultParquetHandler::new(
        Arc::new(LocalFileSystem::new()),
        Arc::new(TokioBackgroundExecutor::new()),
    ))
}

fn sync_parquet_handler() -> Box<dyn ParquetHandler> {
    Box::new(SyncParquetHandler)
}

fn default_json_handler() -> Box<dyn JsonHandler> {
    Box::new(DefaultJsonHandler::new(
        Arc::new(LocalFileSystem::new()),
        Arc::new(TokioBackgroundExecutor::new()),
    ))
}

fn sync_json_handler() -> Box<dyn JsonHandler> {
    Box::new(SyncJsonHandler)
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_reads_footer(#[case] handler: Box<dyn ParquetHandler>) {
    super::tests::test_parquet_handler_reads_footer(handler.as_ref());
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_footer_errors_on_missing_file(#[case] handler: Box<dyn ParquetHandler>) {
    super::tests::test_parquet_handler_footer_errors_on_missing_file(handler.as_ref());
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_footer_preserves_field_ids(#[case] handler: Box<dyn ParquetHandler>) {
    super::tests::test_parquet_handler_footer_preserves_field_ids(handler.as_ref());
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_write_always_overwrites(#[case] handler: Box<dyn ParquetHandler>) {
    super::tests::test_parquet_handler_write_always_overwrites(handler.as_ref());
}

// Both kernel engines configure their parquet readers and writers to skip the Arrow IPC schema
// (ARROW:schema) in file metadata. The following tests verify this shared behavior.

fn assert_no_arrow_schema(handler: &dyn ParquetHandler) {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("no_arrow_schema.parquet");
    let url = Url::from_file_path(&file_path).unwrap();

    let data: Box<dyn EngineData> = Box::new(ArrowEngineData::new(
        RecordBatch::try_from_iter(vec![(
            "id",
            Arc::new(Int64Array::from(vec![1, 2])) as Arc<dyn Array>,
        )])
        .unwrap(),
    ));
    handler
        .write_parquet_file(
            url,
            Box::new(std::iter::once(Ok(data))),
            &Default::default(),
        )
        .unwrap();

    let builder =
        ParquetRecordBatchReaderBuilder::try_new(File::open(&file_path).unwrap()).unwrap();
    let kv = builder.metadata().file_metadata().key_value_metadata();
    let has = kv
        .map(|kv| kv.iter().any(|e| e.key == ARROW_SCHEMA_META_KEY))
        .unwrap_or(false);
    assert!(
        !has,
        "Parquet file should not contain embedded Arrow schema metadata"
    );
}

fn assert_reads_file_with_arrow_schema_metadata(handler: &dyn ParquetHandler) {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("with_arrow_schema.parquet");

    let batch = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![10, 20, 30])) as Arc<dyn Array>,
    )])
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(&file_path).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let file_meta = super::tests::file_meta_for(&file_path);
    let schema = Arc::new(batch.schema().as_ref().try_into_kernel().unwrap());
    let batches: Vec<RecordBatch> = handler
        .read_parquet_files(&[file_meta], schema, None)
        .unwrap()
        .map(|r| {
            ArrowEngineData::try_from_engine_data(r.unwrap())
                .unwrap()
                .into()
        })
        .collect();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[10, 20, 30]
    );
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_write_file_omits_arrow_schema(#[case] handler: Box<dyn ParquetHandler>) {
    assert_no_arrow_schema(handler.as_ref());
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_reads_file_with_arrow_schema_metadata(#[case] handler: Box<dyn ParquetHandler>) {
    assert_reads_file_with_arrow_schema_metadata(handler.as_ref());
}

#[rstest]
#[case::default_engine(default_json_handler())]
#[case::sync_engine(sync_json_handler())]
fn test_json_file_path_contract(#[case] handler: Box<dyn JsonHandler>) {
    super::tests::test_json_handler_file_path_contract(handler.as_ref());
}

// Both kernel engines must rename output columns to match the kernel schema when fields are
// matched by parquet field ID (rather than by name).
fn assert_field_id_matching_renames_columns(handler: &dyn ParquetHandler) {
    let make_field = |name: &str, ty: ArrowDataType, nullable: bool, id: &str| {
        Field::new(name, ty, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    };

    // Write parquet with snake_case names and field IDs
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        make_field("content_type", ArrowDataType::Int32, false, "134"),
        make_field("file_location", ArrowDataType::Utf8, true, "100"),
        make_field("record_count", ArrowDataType::Int64, false, "103"),
    ]));

    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("field_id_rename.parquet");
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 3])),
            Arc::new(StringArray::from(vec![
                Some("/data/file.parquet"),
                Some("/metadata/manifest.parquet"),
            ])),
            Arc::new(Int64Array::from(vec![100, 5])),
        ],
    )
    .unwrap();

    let file = File::create(&file_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, arrow_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Kernel schema uses camelCase names with matching field IDs
    let kernel_schema = Arc::new(
        StructType::try_new(vec![
            StructField::new("contentType", DataType::INTEGER, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(134),
            )]),
            StructField::new("location", DataType::STRING, true).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(100),
            )]),
            StructField::new("recordCount", DataType::LONG, false).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(103),
            )]),
        ])
        .unwrap(),
    );

    let file_meta = super::tests::file_meta_for(&file_path);
    let batches: Vec<RecordBatch> = handler
        .read_parquet_files(&[file_meta], kernel_schema, None)
        .unwrap()
        .map(|r| {
            ArrowEngineData::try_from_engine_data(r.unwrap())
                .unwrap()
                .into()
        })
        .collect();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // Output schema must use the kernel schema names, not the parquet file names
    let schema = batch.schema();
    assert_eq!(schema.field(0).name(), "contentType");
    assert_eq!(schema.field(1).name(), "location");
    assert_eq!(schema.field(2).name(), "recordCount");

    // Data should be correctly matched by field ID
    let ct = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ct.values(), &[0, 3]);

    let loc = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(loc.value(0), "/data/file.parquet");

    let rc = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(rc.values(), &[100, 5]);
}

#[rstest]
#[case::default_engine(default_parquet_handler())]
#[case::sync_engine(sync_parquet_handler())]
fn test_field_id_matching_renames_columns(#[case] handler: Box<dyn ParquetHandler>) {
    assert_field_id_matching_renames_columns(handler.as_ref());
}
