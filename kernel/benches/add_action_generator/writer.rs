// Copyright 2025 The Delta Kernel Rust Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Parquet writer with Delta checkpoint schema format.
//!
//! Writes Add actions to parquet with structured stats (stats_parsed) following
//! the Delta checkpoint schema specification.

use std::fs::File;
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::DeletionVectorStorageType;
use delta_kernel::arrow::array::builder::{
    Int32Builder, Int64Builder, MapBuilder, MapFieldNames, StringBuilder,
};
use delta_kernel::arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StructArray};
use delta_kernel::arrow::datatypes::{DataType, Field, Fields, Schema};
use delta_kernel::parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use delta_kernel::parquet::basic::Compression;
use delta_kernel::parquet::file::properties::WriterProperties;
use delta_kernel::schema::ColumnMetadataKey;
use delta_kernel::DeltaResult;

use crate::generator::AddActionMetadata;

/// Write Add actions to a parquet file in checkpoint format
pub fn write_checkpoint_parquet(
    actions: Vec<AddActionMetadata>,
    output_path: &str,
) -> DeltaResult<()> {
    // Build checkpoint schema
    let schema = build_checkpoint_add_schema();

    // Convert actions to record batch
    let batch = build_record_batch(&actions, &schema)?;

    // Write to parquet file
    let file = File::create(output_path).map_err(|e| {
        delta_kernel::Error::generic(format!("Failed to create output file: {}", e))
    })?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let options = ArrowWriterOptions::new()
        .with_skip_arrow_metadata(true)
        .with_properties(props);
    let mut writer =
        ArrowWriter::try_new_with_options(file, schema.clone(), options).map_err(|e| {
            delta_kernel::Error::generic(format!("Failed to create Arrow writer: {}", e))
        })?;

    writer
        .write(&batch)
        .map_err(|e| delta_kernel::Error::generic(format!("Failed to write batch: {}", e)))?;

    writer
        .close()
        .map_err(|e| delta_kernel::Error::generic(format!("Failed to close writer: {}", e)))?;

    Ok(())
}

/// Build the checkpoint Add schema with structured stats_parsed
fn build_checkpoint_add_schema() -> Arc<Schema> {
    // Schema for minValues and maxValues (use column types)
    // Fields are nullable but we always populate values (never null)
    // Use the same fields with field IDs as defined in get_stats_columns_fields()
    let stats_columns_fields = get_stats_columns_fields();

    // For nullCount, all values are Int64 (counts), not the original column types
    let null_count_fields = get_null_count_fields();

    // stats_parsed schema - nullable but always populated
    let stats_parsed_fields = Fields::from(vec![
        Field::new("numRecords", DataType::Int64, false),
        Field::new(
            "minValues",
            DataType::Struct(stats_columns_fields.clone()),
            true,
        ),
        Field::new("maxValues", DataType::Struct(stats_columns_fields), true),
        Field::new("nullCount", DataType::Struct(null_count_fields), true),
    ]);

    // deletionVector schema
    let dv_fields = Fields::from(vec![
        Field::new("storageType", DataType::Utf8, false),
        Field::new("pathOrInlineDv", DataType::Utf8, false),
        Field::new("offset", DataType::Int32, true),
        Field::new("sizeInBytes", DataType::Int32, false),
        Field::new("cardinality", DataType::Int64, false),
    ]);

    // partitionValues schema (Map<String, String>)
    let partition_values_field = Field::new_map(
        "partitionValues",
        "entries",
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, true),
        false, // keys_sorted
        false, // not nullable
    );

    // tags schema (Map<String, String>)
    let tags_field = Field::new_map(
        "tags",
        "entries",
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, true),
        false, // keys_sorted
        true,  // nullable
    );

    // Add action schema
    let add_fields = Fields::from(vec![
        Field::new("path", DataType::Utf8, false),
        partition_values_field,
        Field::new("size", DataType::Int64, false),
        Field::new("modificationTime", DataType::Int64, false),
        Field::new("dataChange", DataType::Boolean, false),
        Field::new("stats_parsed", DataType::Struct(stats_parsed_fields), true),
        Field::new("deletionVector", DataType::Struct(dv_fields), true),
        tags_field,
        Field::new("baseRowId", DataType::Int64, true),
        Field::new("defaultRowCommitVersion", DataType::Int64, true),
    ]);

    // Top-level schema with "add" wrapper
    Arc::new(Schema::new(vec![Field::new(
        "add",
        DataType::Struct(add_fields),
        true,
    )]))
}

/// Build a record batch from Add action metadata
fn build_record_batch(
    actions: &[AddActionMetadata],
    schema: &Arc<Schema>,
) -> DeltaResult<RecordBatch> {
    let n = actions.len();

    // Extract the "add" field schema
    let add_field = schema.field(0);
    let add_fields = match add_field.data_type() {
        DataType::Struct(fields) => fields,
        _ => {
            return Err(delta_kernel::Error::generic(
                "Expected add field to be a struct",
            ))
        }
    };

    // Build individual arrays
    let path_array = build_path_array(actions);
    let partition_values_array = build_partition_values_array(actions, n)?;
    let size_array = build_size_array(actions);
    let mod_time_array = build_mod_time_array(actions);
    let data_change_array = build_data_change_array(n);
    let stats_parsed_array = build_stats_parsed_array(actions)?;
    let dv_array = build_deletion_vector_array(actions)?;
    let tags_array = build_tags_array(n)?;
    let base_row_id_array = build_null_int64_array(n);
    let default_row_commit_version_array = build_null_int64_array(n);

    // Create Add struct array
    let add_struct = StructArray::new(
        add_fields.clone(),
        vec![
            path_array,
            partition_values_array,
            size_array,
            mod_time_array,
            data_change_array,
            stats_parsed_array,
            dv_array,
            tags_array,
            base_row_id_array,
            default_row_commit_version_array,
        ],
        None,
    );

    // Create record batch with top-level "add" column
    RecordBatch::try_new(schema.clone(), vec![Arc::new(add_struct)])
        .map_err(|e| delta_kernel::Error::generic(format!("Failed to create record batch: {}", e)))
}

fn build_path_array(actions: &[AddActionMetadata]) -> ArrayRef {
    let paths: Vec<&str> = actions.iter().map(|a| a.path.as_str()).collect();
    Arc::new(delta_kernel::arrow::array::StringArray::from(paths))
}

fn build_partition_values_array(_actions: &[AddActionMetadata], n: usize) -> DeltaResult<ArrayRef> {
    // Empty partition values for all actions
    let mut builder = MapBuilder::new(
        Some(MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        }),
        StringBuilder::new(),
        StringBuilder::new(),
    );

    for _ in 0..n {
        builder.append(true).map_err(|e| {
            delta_kernel::Error::generic(format!("Failed to append partition values: {}", e))
        })?;
    }

    Ok(Arc::new(builder.finish()))
}

fn build_size_array(actions: &[AddActionMetadata]) -> ArrayRef {
    let sizes: Vec<i64> = actions.iter().map(|a| a.size).collect();
    Arc::new(Int64Array::from(sizes))
}

fn build_mod_time_array(actions: &[AddActionMetadata]) -> ArrayRef {
    let times: Vec<i64> = actions.iter().map(|a| a.modification_time).collect();
    Arc::new(Int64Array::from(times))
}

fn build_data_change_array(n: usize) -> ArrayRef {
    Arc::new(delta_kernel::arrow::array::BooleanArray::from(vec![
        true;
        n
    ]))
}

fn build_stats_parsed_array(actions: &[AddActionMetadata]) -> DeltaResult<ArrayRef> {
    let n = actions.len();

    // Build numRecords array
    let num_records: Vec<i64> = actions.iter().map(|a| a.stats.num_records).collect();
    let num_records_array = Arc::new(Int64Array::from(num_records));

    // Build minValues struct
    let min_values_array = build_stats_columns_struct(
        actions,
        |s| s.phonetic_min.as_str(),
        |s| s.city_min.as_str(),
        |s| s.state_min.as_str(),
        |s| s.num1_min,
        |s| s.num2_min,
        |s| s.num3_min,
        |s| s.num4_min,
        |s| s.num5_min,
        |s| s.num6_min,
        |s| s.num7_min,
        |s| s.num8_min,
        |s| s.num9_min,
        |s| s.num10_min,
        |s| s.num11_min,
        |s| s.num12_min,
        |s| s.num13_min,
        |s| s.num14_min,
        |s| s.num15_min,
        |s| s.num16_min,
        |s| s.id_value,
    )?;

    // Build maxValues struct
    let max_values_array = build_stats_columns_struct(
        actions,
        |s| s.phonetic_max.as_str(),
        |s| s.city_max.as_str(),
        |s| s.state_max.as_str(),
        |s| s.num1_max,
        |s| s.num2_max,
        |s| s.num3_max,
        |s| s.num4_max,
        |s| s.num5_max,
        |s| s.num6_max,
        |s| s.num7_max,
        |s| s.num8_max,
        |s| s.num9_max,
        |s| s.num10_max,
        |s| s.num11_max,
        |s| s.num12_max,
        |s| s.num13_max,
        |s| s.num14_max,
        |s| s.num15_max,
        |s| s.num16_max,
        |s| s.id_value,
    )?;

    // Build nullCount struct (all zeros)
    let null_count_array = build_null_count_struct(n)?;

    // Build stats_parsed struct
    let stats_fields = Fields::from(vec![
        Field::new("numRecords", DataType::Int64, false),
        Field::new(
            "minValues",
            DataType::Struct(get_stats_columns_fields()),
            true,
        ),
        Field::new(
            "maxValues",
            DataType::Struct(get_stats_columns_fields()),
            true,
        ),
        Field::new("nullCount", DataType::Struct(get_null_count_fields()), true),
    ]);

    Ok(Arc::new(StructArray::new(
        stats_fields,
        vec![
            num_records_array,
            min_values_array,
            max_values_array,
            null_count_array,
        ],
        None,
    )))
}

fn get_stats_columns_fields() -> Fields {
    use std::collections::HashMap;

    // Add field IDs to match the table schema's column mapping
    Fields::from(vec![
        Field::new("phonetic", DataType::Utf8, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "1".to_string(),
        )])),
        Field::new("city", DataType::Utf8, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "2".to_string(),
        )])),
        Field::new("state", DataType::Utf8, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "3".to_string(),
        )])),
        Field::new("num1", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "4".to_string(),
        )])),
        Field::new("num2", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "5".to_string(),
        )])),
        Field::new("num3", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "6".to_string(),
        )])),
        Field::new("num4", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "7".to_string(),
        )])),
        Field::new("num5", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "8".to_string(),
        )])),
        Field::new("num6", DataType::Float64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "9".to_string(),
        )])),
        Field::new("num7", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "10".to_string(),
        )])),
        Field::new("num8", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "11".to_string(),
        )])),
        Field::new("num9", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "12".to_string(),
        )])),
        Field::new("num10", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "13".to_string(),
        )])),
        Field::new("num11", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "14".to_string(),
        )])),
        Field::new("num12", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "15".to_string(),
        )])),
        Field::new("num13", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "16".to_string(),
        )])),
        Field::new("num14", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "17".to_string(),
        )])),
        Field::new("num15", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "18".to_string(),
        )])),
        Field::new("num16", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "19".to_string(),
        )])),
        Field::new("id", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "20".to_string(),
        )])),
    ])
}

/// Get the fields for nullCount struct - all fields are Int64 (counts)
/// regardless of the original column types
fn get_null_count_fields() -> Fields {
    use std::collections::HashMap;

    // nullCount values are always Long (Int64) since they represent counts
    Fields::from(vec![
        Field::new("phonetic", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "1".to_string(),
        )])),
        Field::new("city", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "2".to_string(),
        )])),
        Field::new("state", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "3".to_string(),
        )])),
        Field::new("num1", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "4".to_string(),
        )])),
        Field::new("num2", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "5".to_string(),
        )])),
        Field::new("num3", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "6".to_string(),
        )])),
        Field::new("num4", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "7".to_string(),
        )])),
        Field::new("num5", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "8".to_string(),
        )])),
        Field::new("num6", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "9".to_string(),
        )])),
        Field::new("num7", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "10".to_string(),
        )])),
        Field::new("num8", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "11".to_string(),
        )])),
        Field::new("num9", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "12".to_string(),
        )])),
        Field::new("num10", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "13".to_string(),
        )])),
        Field::new("num11", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "14".to_string(),
        )])),
        Field::new("num12", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "15".to_string(),
        )])),
        Field::new("num13", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "16".to_string(),
        )])),
        Field::new("num14", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "17".to_string(),
        )])),
        Field::new("num15", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "18".to_string(),
        )])),
        Field::new("num16", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "19".to_string(),
        )])),
        Field::new("id", DataType::Int64, true).with_metadata(HashMap::from([(
            ColumnMetadataKey::ParquetFieldId.as_ref().to_string(),
            "20".to_string(),
        )])),
    ])
}

#[allow(clippy::too_many_arguments)]
fn build_stats_columns_struct<
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
>(
    actions: &[AddActionMetadata],
    phonetic_fn: F1,
    city_fn: F2,
    state_fn: F3,
    num1_fn: F4,
    num2_fn: F5,
    num3_fn: F6,
    num4_fn: F7,
    num5_fn: F8,
    num6_fn: F9,
    num7_fn: F10,
    num8_fn: F11,
    num9_fn: F12,
    num10_fn: F13,
    num11_fn: F14,
    num12_fn: F15,
    num13_fn: F16,
    num14_fn: F17,
    num15_fn: F18,
    num16_fn: F19,
    id_fn: F20,
) -> DeltaResult<ArrayRef>
where
    F1: Fn(&crate::stats::GeneratedStats) -> &str,
    F2: Fn(&crate::stats::GeneratedStats) -> &str,
    F3: Fn(&crate::stats::GeneratedStats) -> &str,
    F4: Fn(&crate::stats::GeneratedStats) -> i64,
    F5: Fn(&crate::stats::GeneratedStats) -> i64,
    F6: Fn(&crate::stats::GeneratedStats) -> i64,
    F7: Fn(&crate::stats::GeneratedStats) -> i64,
    F8: Fn(&crate::stats::GeneratedStats) -> i64,
    F9: Fn(&crate::stats::GeneratedStats) -> f64,
    F10: Fn(&crate::stats::GeneratedStats) -> i64,
    F11: Fn(&crate::stats::GeneratedStats) -> i64,
    F12: Fn(&crate::stats::GeneratedStats) -> i64,
    F13: Fn(&crate::stats::GeneratedStats) -> i64,
    F14: Fn(&crate::stats::GeneratedStats) -> i64,
    F15: Fn(&crate::stats::GeneratedStats) -> i64,
    F16: Fn(&crate::stats::GeneratedStats) -> i64,
    F17: Fn(&crate::stats::GeneratedStats) -> i64,
    F18: Fn(&crate::stats::GeneratedStats) -> i64,
    F19: Fn(&crate::stats::GeneratedStats) -> i64,
    F20: Fn(&crate::stats::GeneratedStats) -> i64,
{
    let phonetic: Vec<&str> = actions.iter().map(|a| phonetic_fn(&a.stats)).collect();
    let city: Vec<&str> = actions.iter().map(|a| city_fn(&a.stats)).collect();
    let state: Vec<&str> = actions.iter().map(|a| state_fn(&a.stats)).collect();
    let num1: Vec<i64> = actions.iter().map(|a| num1_fn(&a.stats)).collect();
    let num2: Vec<i64> = actions.iter().map(|a| num2_fn(&a.stats)).collect();
    let num3: Vec<i64> = actions.iter().map(|a| num3_fn(&a.stats)).collect();
    let num4: Vec<i64> = actions.iter().map(|a| num4_fn(&a.stats)).collect();
    let num5: Vec<i64> = actions.iter().map(|a| num5_fn(&a.stats)).collect();
    let num6: Vec<f64> = actions.iter().map(|a| num6_fn(&a.stats)).collect();
    let num7: Vec<i64> = actions.iter().map(|a| num7_fn(&a.stats)).collect();
    let num8: Vec<i64> = actions.iter().map(|a| num8_fn(&a.stats)).collect();
    let num9: Vec<i64> = actions.iter().map(|a| num9_fn(&a.stats)).collect();
    let num10: Vec<i64> = actions.iter().map(|a| num10_fn(&a.stats)).collect();
    let num11: Vec<i64> = actions.iter().map(|a| num11_fn(&a.stats)).collect();
    let num12: Vec<i64> = actions.iter().map(|a| num12_fn(&a.stats)).collect();
    let num13: Vec<i64> = actions.iter().map(|a| num13_fn(&a.stats)).collect();
    let num14: Vec<i64> = actions.iter().map(|a| num14_fn(&a.stats)).collect();
    let num15: Vec<i64> = actions.iter().map(|a| num15_fn(&a.stats)).collect();
    let num16: Vec<i64> = actions.iter().map(|a| num16_fn(&a.stats)).collect();
    let id: Vec<i64> = actions.iter().map(|a| id_fn(&a.stats)).collect();

    Ok(Arc::new(StructArray::new(
        get_stats_columns_fields(),
        vec![
            Arc::new(delta_kernel::arrow::array::StringArray::from(phonetic)),
            Arc::new(delta_kernel::arrow::array::StringArray::from(city)),
            Arc::new(delta_kernel::arrow::array::StringArray::from(state)),
            Arc::new(Int64Array::from(num1)),
            Arc::new(Int64Array::from(num2)),
            Arc::new(Int64Array::from(num3)),
            Arc::new(Int64Array::from(num4)),
            Arc::new(Int64Array::from(num5)),
            Arc::new(Float64Array::from(num6)),
            Arc::new(Int64Array::from(num7)),
            Arc::new(Int64Array::from(num8)),
            Arc::new(Int64Array::from(num9)),
            Arc::new(Int64Array::from(num10)),
            Arc::new(Int64Array::from(num11)),
            Arc::new(Int64Array::from(num12)),
            Arc::new(Int64Array::from(num13)),
            Arc::new(Int64Array::from(num14)),
            Arc::new(Int64Array::from(num15)),
            Arc::new(Int64Array::from(num16)),
            Arc::new(Int64Array::from(id)),
        ],
        None,
    )))
}

fn build_null_count_struct(n: usize) -> DeltaResult<ArrayRef> {
    // All null counts are zero - all values are Int64 (counts)
    Ok(Arc::new(StructArray::new(
        get_null_count_fields(),
        vec![
            Arc::new(Int64Array::from(vec![0i64; n])), // phonetic
            Arc::new(Int64Array::from(vec![0i64; n])), // city
            Arc::new(Int64Array::from(vec![0i64; n])), // state
            Arc::new(Int64Array::from(vec![0i64; n])), // num1
            Arc::new(Int64Array::from(vec![0i64; n])), // num2
            Arc::new(Int64Array::from(vec![0i64; n])), // num3
            Arc::new(Int64Array::from(vec![0i64; n])), // num4
            Arc::new(Int64Array::from(vec![0i64; n])), // num5
            Arc::new(Int64Array::from(vec![0i64; n])), // num6 (count, not float)
            Arc::new(Int64Array::from(vec![0i64; n])), // num7
            Arc::new(Int64Array::from(vec![0i64; n])), // num8
            Arc::new(Int64Array::from(vec![0i64; n])), // num9
            Arc::new(Int64Array::from(vec![0i64; n])), // num10
            Arc::new(Int64Array::from(vec![0i64; n])), // num11
            Arc::new(Int64Array::from(vec![0i64; n])), // num12
            Arc::new(Int64Array::from(vec![0i64; n])), // num13
            Arc::new(Int64Array::from(vec![0i64; n])), // num14
            Arc::new(Int64Array::from(vec![0i64; n])), // num15
            Arc::new(Int64Array::from(vec![0i64; n])), // num16
            Arc::new(Int64Array::from(vec![0i64; n])),
        ],
        None,
    )))
}

fn build_deletion_vector_array(actions: &[AddActionMetadata]) -> DeltaResult<ArrayRef> {
    let mut storage_type_builder = StringBuilder::new();
    let mut path_or_inline_dv_builder = StringBuilder::new();
    let mut offset_builder = Int32Builder::new();
    let mut size_in_bytes_builder = Int32Builder::new();
    let mut cardinality_builder = Int64Builder::new();

    let mut nulls = Vec::new();

    for action in actions {
        if let Some(dv) = &action.deletion_vector {
            storage_type_builder.append_value(match dv.storage_type {
                DeletionVectorStorageType::Inline => "i",
                DeletionVectorStorageType::PersistedRelative => "u",
                DeletionVectorStorageType::PersistedAbsolute => "p",
            });
            path_or_inline_dv_builder.append_value(&dv.path_or_inline_dv);
            if let Some(offset) = dv.offset {
                offset_builder.append_value(offset);
            } else {
                offset_builder.append_null();
            }
            size_in_bytes_builder.append_value(dv.size_in_bytes);
            cardinality_builder.append_value(dv.cardinality);
            nulls.push(true); // Mark as valid/present
        } else {
            // For null DVs, append placeholder values but mark struct as null
            storage_type_builder.append_value("");
            path_or_inline_dv_builder.append_value("");
            offset_builder.append_null();
            size_in_bytes_builder.append_value(0);
            cardinality_builder.append_value(0);
            nulls.push(false); // Mark as null
        }
    }

    let dv_fields = Fields::from(vec![
        Field::new("storageType", DataType::Utf8, false),
        Field::new("pathOrInlineDv", DataType::Utf8, false),
        Field::new("offset", DataType::Int32, true),
        Field::new("sizeInBytes", DataType::Int32, false),
        Field::new("cardinality", DataType::Int64, false),
    ]);

    let null_buffer = delta_kernel::arrow::buffer::NullBuffer::from(nulls);

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(storage_type_builder.finish()) as ArrayRef,
        Arc::new(path_or_inline_dv_builder.finish()) as ArrayRef,
        Arc::new(offset_builder.finish()) as ArrayRef,
        Arc::new(size_in_bytes_builder.finish()) as ArrayRef,
        Arc::new(cardinality_builder.finish()) as ArrayRef,
    ];

    Ok(Arc::new(StructArray::new(
        dv_fields,
        arrays,
        Some(null_buffer),
    )))
}

fn build_tags_array(n: usize) -> DeltaResult<ArrayRef> {
    // All tags are null
    let mut builder = MapBuilder::new(
        Some(MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        }),
        StringBuilder::new(),
        StringBuilder::new(),
    );

    for _ in 0..n {
        builder
            .append(false)
            .map_err(|e| delta_kernel::Error::generic(format!("Failed to append tags: {}", e)))?;
    }

    Ok(Arc::new(builder.finish()))
}

fn build_null_int64_array(n: usize) -> ArrayRef {
    Arc::new(Int64Array::from(vec![None::<i64>; n]))
}
