//! Lookup join implementation for efficient key-based joins with cached data.

use std::collections::HashMap;
use std::sync::Arc;

use crate::arrow::array::cast::AsArray;
use crate::arrow::array::{Array, RecordBatch};
use crate::arrow::compute::interleave_record_batch;
use crate::arrow::datatypes::Schema as ArrowSchema;
use crate::engine::arrow_data::{extract_record_batch, ArrowEngineData};
use crate::engine::arrow_expression::evaluate_expression::extract_column;
use crate::engine_data::FilteredEngineData;
use crate::schema::{ColumnName, DataType, SchemaRef, StructField, StructType};
use crate::{DeltaResult, EngineData, Error, EvaluationHandler, LookupJoiner};

/// Helper function to look up a field in a schema by path.
/// For nested paths, navigates through struct types.
fn lookup_field_by_path<'a>(
    schema: &'a StructType,
    path: &[String],
) -> DeltaResult<&'a StructField> {
    if path.is_empty() {
        return Err(Error::missing_column("Empty column path"));
    }

    let mut current_field = schema
        .field(&path[0])
        .ok_or_else(|| Error::missing_column(path[0].clone()))?;

    for field_name in &path[1..] {
        match current_field.data_type() {
            DataType::Struct(inner_struct) => {
                current_field = inner_struct
                    .field(field_name)
                    .ok_or_else(|| Error::missing_column(field_name.clone()))?;
            }
            _ => {
                return Err(Error::generic(format!(
                    "Cannot navigate to field '{}' in non-struct type",
                    field_name
                )));
            }
        }
    }

    Ok(current_field)
}

/// Location of a value within the batches vector, along with its version.
#[derive(Debug, Clone, Copy)]
struct ValueLocation {
    batch_index: usize,
    row_index: usize,
    /// Version of this lookup entry (only meaningful if version column was specified)
    version: i64,
}

/// Arrow-based implementation of [`LookupJoiner`] that maintains an in-memory hash map
/// from String keys to data locations for O(1) lookup performance.
///
/// # Data Structure
///
/// - `batches[0]`: Single null row (used for keys not found or null keys)
/// - `batches[1..]`: Lookup data batches containing only value columns
/// - `key_to_location`: HashMap from String keys to (batch_index, row_index)
///
/// # Key Type
///
/// Only String keys are currently supported. The key column must be of String type.
///
/// # Null Handling
///
/// - Null keys in lookup data: Not added to hash map
/// - Null keys in input data: Mapped to (0, 0) - the null row
/// - Keys not found: Mapped to (0, 0) - the null row
///
/// # Version Filtering
///
/// Version columns are required and must not contain null values:
/// - Lookup data stores version with each key
/// - During join, if input version > lookup version, returns null row (stale data)
/// - Null versions in either lookup or input data will cause an error
pub(crate) struct ArrowLookupJoiner {
    /// Paths to the value columns in the original lookup schema
    value_column_paths: Vec<ColumnName>,
    /// Path to the key column in the original lookup schema
    key_column_path: ColumnName,
    /// Path to the version column in the lookup schema
    lookup_version_column: ColumnName,
    /// Batches of lookup data: [0] = null row, [1+] = lookup data (value columns only)
    batches: Vec<RecordBatch>,
    /// Hash map from String key to location in batches (including version)
    key_to_location: HashMap<String, ValueLocation>,
}

impl ArrowLookupJoiner {
    /// Create a new ArrowLookupJoiner with the given schema, column configuration, and initial
    /// data.
    ///
    /// This constructor validates the schema, creates a null row, and populates the joiner
    /// with the provided initial lookup data.
    pub(crate) fn new(
        handler: &dyn EvaluationHandler,
        lookup_schema: SchemaRef,
        key_column: &ColumnName,
        value_columns: &[ColumnName],
        lookup_version_column: &ColumnName,
        initial_data: &[&FilteredEngineData],
    ) -> DeltaResult<Self> {
        // Validate that at least one value column is provided
        if value_columns.is_empty() {
            return Err(Error::generic(
                "Lookup join requires at least one value column",
            ));
        }

        // Validate that key column exists and is String type
        let key_field = lookup_field_by_path(&lookup_schema, key_column.path())?;

        if key_field.data_type() != &DataType::STRING {
            return Err(Error::unexpected_column_type(format!(
                "Lookup join requires String key column, got: {:?}",
                key_field.data_type()
            )));
        }

        // Validate that all value columns exist
        for value_col in value_columns {
            lookup_field_by_path(&lookup_schema, value_col.path())?;
        }

        // Validate version column
        let version_field = lookup_field_by_path(&lookup_schema, lookup_version_column.path())?;
        if version_field.data_type() != &DataType::LONG {
            return Err(Error::unexpected_column_type(format!(
                "Lookup join version column must be Int64 type, got: {:?}",
                version_field.data_type()
            )));
        }

        // Build value schema from value columns
        let value_schema = extract_value_schema(&lookup_schema, value_columns)?;

        // Create null row (batch[0])
        let null_row_engine_data = handler.null_row(value_schema.clone())?;
        let null_row_batch = extract_record_batch(null_row_engine_data.as_ref())?.clone();

        let mut joiner = ArrowLookupJoiner {
            value_column_paths: value_columns.to_vec(),
            key_column_path: key_column.clone(),
            lookup_version_column: lookup_version_column.clone(),
            batches: vec![null_row_batch],
            key_to_location: HashMap::new(),
        };

        // Add initial lookup data if provided
        if !initial_data.is_empty() {
            joiner.extend_from_raw(initial_data)?;
        }

        Ok(joiner)
    }

    /// Extract keys from a batch and populate the HashMap, respecting the selection vector.
    ///
    /// Only selected rows with non-null keys are added to the HashMap.
    /// Uses latest-version-wins semantics: for duplicate keys, keeps the entry with the highest
    /// version.
    fn populate_key_map(
        batch: &RecordBatch,
        key_column: &ColumnName,
        version_column: &ColumnName,
        selection_vector: &[bool],
        batch_index: usize,
        key_to_location: &mut HashMap<String, ValueLocation>,
    ) -> DeltaResult<()> {
        let key_array = extract_column(batch, key_column.path())?;

        // Validate that key column is String type
        let string_array = key_array.as_string_opt::<i32>().ok_or_else(|| {
            Error::unexpected_column_type(format!(
                "Key column must be String type, got: {:?}",
                key_array.data_type()
            ))
        })?;

        // Extract version array
        let version_arr = extract_column(batch, version_column.path())?;
        let version_array = version_arr
            .as_primitive_opt::<crate::arrow::datatypes::Int64Type>()
            .ok_or_else(|| {
                Error::unexpected_column_type(format!(
                    "Version column must be Int64 type, got: {:?}",
                    version_arr.data_type()
                ))
            })?;

        let num_rows = batch.num_rows();

        for row_idx in 0..num_rows {
            // Check selection vector
            let selected = if row_idx < selection_vector.len() {
                selection_vector[row_idx]
            } else {
                true // Rows beyond selection vector are assumed selected
            };

            // Only process selected rows with non-null keys
            if selected && string_array.is_valid(row_idx) {
                let key = string_array.value(row_idx).to_string();

                // Extract version - must not be null
                if !version_array.is_valid(row_idx) {
                    return Err(Error::generic(format!(
                        "Null version found in lookup data at row {}. Version column must not contain null values.",
                        row_idx
                    )));
                }
                let version = version_array.value(row_idx);

                // Latest-version-wins semantics: keep entry with highest version
                key_to_location
                    .entry(key)
                    .and_modify(|existing| {
                        // Update if new version is higher
                        if version > existing.version {
                            *existing = ValueLocation {
                                batch_index,
                                row_index: row_idx,
                                version,
                            };
                        }
                    })
                    .or_insert(ValueLocation {
                        batch_index,
                        row_index: row_idx,
                        version,
                    });
            }
        }

        Ok(())
    }

    /// Project a batch to only include value columns.
    fn project_to_value_columns(
        batch: &RecordBatch,
        value_column_paths: &[ColumnName],
    ) -> DeltaResult<RecordBatch> {
        let mut columns = Vec::with_capacity(value_column_paths.len());
        for value_col in value_column_paths {
            let col_array = extract_column(batch, value_col.path())?;
            columns.push(col_array);
        }

        // Build schema from value columns
        let arrow_schema = Arc::new(ArrowSchema::new(
            value_column_paths
                .iter()
                .zip(&columns)
                .map(|(col_name, array)| {
                    crate::arrow::datatypes::Field::new(
                        col_name.to_string(),
                        array.data_type().clone(),
                        true, // nullable
                    )
                })
                .collect::<Vec<_>>(),
        ));

        Ok(RecordBatch::try_new(arrow_schema, columns)?)
    }

    /// Append two record batches horizontally (by columns).
    fn append_batches(base: &RecordBatch, to_append: &RecordBatch) -> DeltaResult<RecordBatch> {
        if base.num_rows() != to_append.num_rows() {
            return Err(Error::generic(format!(
                "Cannot append batches with different row counts: {} vs {}",
                base.num_rows(),
                to_append.num_rows()
            )));
        }

        // Combine schemas
        let mut fields: Vec<_> = base.schema().fields().iter().cloned().collect();
        fields.extend(to_append.schema().fields().iter().cloned());
        let combined_schema = Arc::new(ArrowSchema::new(fields));

        // Combine columns
        let mut columns = base.columns().to_vec();
        columns.extend_from_slice(to_append.columns());

        Ok(RecordBatch::try_new(combined_schema, columns)?)
    }

    /// Extends the joiner with additional lookup data.
    ///
    /// Helper method to add more rows to the lookup cache. For duplicate keys, uses
    /// latest-version-wins semantics: the entry with the highest version is kept.
    fn extend_from_raw(&mut self, data: &[&FilteredEngineData]) -> DeltaResult<()> {
        for filtered_data in data {
            let batch = extract_record_batch(filtered_data.data())?;
            let selection_vector = filtered_data.selection_vector();

            // Project to value columns only
            let projected_batch = Self::project_to_value_columns(batch, &self.value_column_paths)?;

            // Add to batches vector and get the index
            let batch_index = self.batches.len();
            self.batches.push(projected_batch);

            // Extract keys and populate the hash map (latest-version-wins semantics)
            Self::populate_key_map(
                batch,
                &self.key_column_path,
                &self.lookup_version_column,
                selection_vector,
                batch_index,
                &mut self.key_to_location,
            )?;
        }

        Ok(())
    }
}

impl LookupJoiner for ArrowLookupJoiner {
    fn join_raw(
        &self,
        input_data: &dyn EngineData,
        input_selection: &[bool],
        input_key_column: &ColumnName,
        input_version_column: &ColumnName,
    ) -> DeltaResult<Box<dyn EngineData>> {
        let input_batch = extract_record_batch(input_data)?;

        // Extract keys from input data
        let key_array = extract_column(input_batch, input_key_column.path())?;

        // Validate that key column is String type
        let string_array = key_array.as_string_opt::<i32>().ok_or_else(|| {
            Error::unexpected_column_type(format!(
                "Input key column must be String type, got: {:?}",
                key_array.data_type()
            ))
        })?;

        // Extract input version array
        let input_version_arr = extract_column(input_batch, input_version_column.path())?;
        let input_version_array = input_version_arr
            .as_primitive_opt::<crate::arrow::datatypes::Int64Type>()
            .ok_or_else(|| {
                Error::unexpected_column_type(format!(
                    "Input version column must be Int64 type, got: {:?}",
                    input_version_arr.data_type()
                ))
            })?;

        // Build indices vector for interleave
        let num_rows = input_batch.num_rows();
        let mut indices = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let location = if string_array.is_valid(row_idx) {
                let key = string_array.value(row_idx);

                // Extract input version - must not be null
                if !input_version_array.is_valid(row_idx) {
                    return Err(Error::generic(format!(
                        "Null version found in input data at row {}. Version column must not contain null values.",
                        row_idx
                    )));
                }
                let input_version = input_version_array.value(row_idx);

                if let Some(loc) = self.key_to_location.get(key) {
                    // If input version is greater than lookup version, use null row (stale lookup
                    // data) DV applies if DV sequence number >= data sequence
                    // number
                    if input_version > loc.version {
                        (0, 0) // Stale lookup data -> null row
                    } else {
                        (loc.batch_index, loc.row_index)
                    }
                } else {
                    (0, 0) // Not found -> null row
                }
            } else {
                (0, 0) // Null key -> null row
            };
            indices.push(location);
        }

        // Use interleave_record_batch to gather all value columns at once
        let batch_refs: Vec<&RecordBatch> = self.batches.iter().collect();
        let value_batch = interleave_record_batch(&batch_refs, &indices)?;

        // Append value columns to input batch
        let result_batch = Self::append_batches(input_batch, &value_batch)?;

        // Apply selection vector and return as EngineData
        let filtered = FilteredEngineData::try_new(
            Box::new(ArrowEngineData::new(result_batch)),
            input_selection.to_vec(),
        )?;
        filtered.apply_selection_vector()
    }
}

/// Extract a schema containing only the specified value columns from the full schema.
///
/// Lookup joins (similar to SQL LEFT JOINs) can introduce nulls when keys are not found
/// in the lookup table. Therefore, all value columns must already be nullable in the source
/// schema. This function validates that all value columns are nullable and returns an error
/// if any non-nullable columns are found.
fn extract_value_schema(
    full_schema: &SchemaRef,
    value_columns: &[ColumnName],
) -> DeltaResult<SchemaRef> {
    let mut fields = Vec::with_capacity(value_columns.len());
    for value_col in value_columns {
        let field = lookup_field_by_path(full_schema, value_col.path())?;

        // Validate that the field is nullable, since lookup joins can produce nulls
        // for unmatched keys (similar to SQL LEFT JOINs)
        if !field.is_nullable() {
            return Err(Error::generic(format!(
                "Lookup join value column '{}' must be nullable because joins can introduce nulls for unmatched keys. \
                 Please update the schema to mark this column as nullable.",
                value_col
            )));
        }

        // Create a new field with the column name as the field name
        let new_field = StructField::new(
            value_col.to_string(),
            field.data_type().clone(),
            field.is_nullable(), // Preserve nullability (already validated to be true)
        );
        fields.push(new_field);
    }

    Ok(Arc::new(crate::schema::StructType::new_unchecked(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::arrow_expression::ArrowEvaluationHandler;
    use crate::expressions::{column_name, Scalar};
    use crate::schema::{DataType, StructField, StructType};

    fn create_test_handler() -> ArrowEvaluationHandler {
        ArrowEvaluationHandler
    }

    fn create_test_schema() -> SchemaRef {
        Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
            StructField::new("version", DataType::LONG, false),
        ]))
    }

    fn create_filtered_data(
        handler: &ArrowEvaluationHandler,
        schema: SchemaRef,
        rows: &[&[Scalar]],
        selection: Vec<bool>,
    ) -> DeltaResult<FilteredEngineData> {
        let data = handler.create_many(schema, rows)?;
        FilteredEngineData::try_new(data, selection)
    }

    /// Helper to compare two RecordBatches for equality
    fn assert_batches_eq(expected: &RecordBatch, actual: &RecordBatch) {
        assert_eq!(expected.num_rows(), actual.num_rows(), "Row count mismatch");
        assert_eq!(
            expected.num_columns(),
            actual.num_columns(),
            "Column count mismatch"
        );

        // Compare schemas
        assert_eq!(expected.schema(), actual.schema(), "Schema mismatch");

        // Compare each column
        for (col_idx, (expected_col, actual_col)) in expected
            .columns()
            .iter()
            .zip(actual.columns().iter())
            .enumerate()
        {
            assert_eq!(
                expected_col,
                actual_col,
                "Column {} mismatch",
                expected.schema().field(col_idx).name()
            );
        }
    }

    #[test]
    fn test_basic_join() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data: keys "a", "b", "c" with versions 100, 200, 300
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
            Scalar::Long(200),
        ];
        let row3 = [
            Scalar::String("c".to_string()),
            Scalar::String("value_c".to_string()),
            Scalar::Integer(3),
            Scalar::Long(300),
        ];
        let lookup_rows = vec![&row1[..], &row2[..], &row3[..]];
        let lookup_data = create_filtered_data(
            &handler,
            schema.clone(),
            &lookup_rows,
            vec![true, true, true],
        )?;

        // Create joiner
        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input data with keys "a", "b", "d" (d not in lookup)
        // Input versions all <= lookup versions, so all should match
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(10),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(20),
            Scalar::String("b".to_string()),
            Scalar::Long(100),
        ];
        let in_row3 = [
            Scalar::Integer(30),
            Scalar::String("d".to_string()),
            Scalar::Long(100),
        ];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        // Perform join
        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row1 = [
            Scalar::Integer(10),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
        ];
        let exp_row2 = [
            Scalar::Integer(20),
            Scalar::String("b".to_string()),
            Scalar::Long(100),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
        ];
        let exp_row3 = [
            Scalar::Integer(30),
            Scalar::String("d".to_string()),
            Scalar::Long(100),
            Scalar::Null(DataType::STRING),
            Scalar::Null(DataType::INTEGER),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_version_filtering() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with versions
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100), // version = 100
        ];
        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
            Scalar::Long(200), // version = 200
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input data where:
        // - key "a" with version 50 (50 <= 100) -> should match
        // - key "a" with version 150 (150 > 100) -> should return null (stale)
        // - key "b" with version 200 (200 <= 200) -> should match
        // - key "b" with version 201 (201 > 200) -> should return null (stale)
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(1),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(2),
            Scalar::String("a".to_string()),
            Scalar::Long(150),
        ];
        let in_row3 = [
            Scalar::Integer(3),
            Scalar::String("b".to_string()),
            Scalar::Long(200),
        ];
        let in_row4 = [
            Scalar::Integer(4),
            Scalar::String("b".to_string()),
            Scalar::Long(201),
        ];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..], &in_row4[..]];
        let input_data = create_filtered_data(
            &handler,
            input_schema,
            &input_rows,
            vec![true, true, true, true],
        )?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Expected results
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row1 = [
            Scalar::Integer(1),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("value_a".to_string()), // Matched
            Scalar::Integer(1),
        ];
        let exp_row2 = [
            Scalar::Integer(2),
            Scalar::String("a".to_string()),
            Scalar::Long(150),
            Scalar::Null(DataType::STRING), // Stale (150 > 100)
            Scalar::Null(DataType::INTEGER),
        ];
        let exp_row3 = [
            Scalar::Integer(3),
            Scalar::String("b".to_string()),
            Scalar::Long(200),
            Scalar::String("value_b".to_string()), // Matched
            Scalar::Integer(2),
        ];
        let exp_row4 = [
            Scalar::Integer(4),
            Scalar::String("b".to_string()),
            Scalar::Long(201),
            Scalar::Null(DataType::STRING), // Stale (201 > 200)
            Scalar::Null(DataType::INTEGER),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..], &exp_row4[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_null_key_in_lookup() -> DeltaResult<()> {
        let handler = create_test_handler();
        // Use nullable schema for this test since we're testing null keys
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, true), // nullable
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data with a null key
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::Null(DataType::STRING),
            Scalar::String("value_null".to_string()),
            Scalar::Integer(99),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Try to join with null key - should get null values
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, true),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row = [Scalar::Null(DataType::STRING), Scalar::Long(50)];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result with null values
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, true),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row = [
            Scalar::Null(DataType::STRING),
            Scalar::Long(50),
            Scalar::Null(DataType::STRING),
            Scalar::Null(DataType::INTEGER),
        ];
        let expected_rows = vec![&exp_row[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_duplicate_keys() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with duplicate key "a" - first should win
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("first".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("a".to_string()),
            Scalar::String("second".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Join with key "a"
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row = [Scalar::String("a".to_string()), Scalar::Long(50)];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result - should get "first" (not "second")
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
        ]));
        let exp_row = [
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("first".to_string()),
        ];
        let expected_rows = vec![&exp_row[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_extend_multiple_batches() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // First batch
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let lookup_rows1 = vec![&row1[..]];
        let lookup_data1 =
            create_filtered_data(&handler, schema.clone(), &lookup_rows1, vec![true])?;

        // Second batch
        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100),
        ];
        let lookup_rows2 = vec![&row2[..]];
        let lookup_data2 =
            create_filtered_data(&handler, schema.clone(), &lookup_rows2, vec![true])?;

        // Create joiner with both datasets
        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data1, &lookup_data2],
        )?;

        // Join with both keys
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [Scalar::String("a".to_string()), Scalar::Long(50)];
        let in_row2 = [Scalar::String("b".to_string()), Scalar::Long(50)];
        let input_rows = vec![&in_row1[..], &in_row2[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
        ]));
        let exp_row1 = [
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("value_a".to_string()),
        ];
        let exp_row2 = [
            Scalar::String("b".to_string()),
            Scalar::Long(50),
            Scalar::String("value_b".to_string()),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_selection_vector() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with selection vector [true, false, true]
        // Only first and third rows should be indexed
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100),
        ];
        let row3 = [
            Scalar::String("c".to_string()),
            Scalar::String("value_c".to_string()),
            Scalar::Integer(3),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..], &row3[..]];
        let lookup_data = create_filtered_data(
            &handler,
            schema.clone(),
            &lookup_rows,
            vec![true, false, true],
        )?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Try to join with key "b" - should not find it (unselected)
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [Scalar::String("a".to_string()), Scalar::Long(50)];
        let in_row2 = [Scalar::String("b".to_string()), Scalar::Long(50)];
        let in_row3 = [Scalar::String("c".to_string()), Scalar::Long(50)];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result: "a" found, "b" not found (unselected), "c" found
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
        ]));
        let exp_row1 = [
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("value_a".to_string()),
        ];
        let exp_row2 = [
            Scalar::String("b".to_string()),
            Scalar::Long(50),
            Scalar::Null(DataType::STRING), // unselected -> null
        ];
        let exp_row3 = [
            Scalar::String("c".to_string()),
            Scalar::Long(50),
            Scalar::String("value_c".to_string()),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_empty_lookup() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create joiner with no data
        let joiner = ArrowLookupJoiner::new(
            &handler,
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[],
        )?;

        // Join should return all nulls
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row = [Scalar::String("a".to_string()), Scalar::Long(50)];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result - empty lookup means all nulls
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
        ]));
        let exp_row = [
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::Null(DataType::STRING),
        ];
        let expected_rows = vec![&exp_row[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_join_with_struct_values() -> DeltaResult<()> {
        use crate::expressions::StructData;

        let handler = create_test_handler();

        // Create schema with nested Struct value column
        let address_type = StructType::new_unchecked(vec![
            StructField::new("street", DataType::STRING, true),
            StructField::new("city", DataType::STRING, true),
            StructField::new("zipcode", DataType::INTEGER, true),
        ]);
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new(
                "address",
                DataType::Struct(Box::new(address_type.clone())),
                true,
            ),
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data with struct values
        let addr1 = StructData::try_new(
            address_type.fields().cloned().collect(),
            vec![
                Scalar::String("123 Main St".to_string()),
                Scalar::String("NYC".to_string()),
                Scalar::Integer(10001),
            ],
        )?;
        let addr2 = StructData::try_new(
            address_type.fields().cloned().collect(),
            vec![
                Scalar::String("456 Oak Ave".to_string()),
                Scalar::String("LA".to_string()),
                Scalar::Integer(90001),
            ],
        )?;

        let row1 = [
            Scalar::String("user1".to_string()),
            Scalar::Struct(addr1.clone()),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("user2".to_string()),
            Scalar::Struct(addr2.clone()),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        // Create joiner
        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("address")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input data
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(1),
            Scalar::String("user1".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(2),
            Scalar::String("user2".to_string()),
            Scalar::Long(50),
        ];
        let in_row3 = [
            Scalar::Integer(3),
            Scalar::String("user999".to_string()),
            Scalar::Long(50),
        ]; // Not found
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        // Perform join
        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result with structs
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new(
                "address",
                DataType::Struct(Box::new(address_type.clone())),
                true,
            ),
        ]));
        let exp_row1 = [
            Scalar::Integer(1),
            Scalar::String("user1".to_string()),
            Scalar::Long(50),
            Scalar::Struct(addr1.clone()),
        ];
        let exp_row2 = [
            Scalar::Integer(2),
            Scalar::String("user2".to_string()),
            Scalar::Long(50),
            Scalar::Struct(addr2.clone()),
        ];
        let exp_row3 = [
            Scalar::Integer(3),
            Scalar::String("user999".to_string()),
            Scalar::Long(50),
            Scalar::Null(DataType::Struct(Box::new(address_type.clone()))),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_input_selection_vector_preserved() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..]];
        let lookup_data = create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true])?;

        let joiner = handler.new_lookup_join_handler(
            schema,
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input with non-trivial selection vector: [true, false, true]
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(1),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(2),
            Scalar::String("b".to_string()),
            Scalar::Long(50),
        ];
        let in_row3 = [
            Scalar::Integer(3),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
        ];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];

        // Important: selection vector with middle row unselected
        let input_selection = vec![true, false, true];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, input_selection.clone())?;

        // Perform join
        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Verify the result batch has the right shape (only selected rows: 2 out of 3)
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_eq!(result_batch.num_rows(), 2); // Only rows where selection_vector was true
        assert_eq!(result_batch.num_columns(), 4); // id, key, version, value1

        // Verify the values are from the selected rows (row 1 and row 3)
        let id_array = result_batch
            .column(0)
            .as_primitive::<crate::arrow::datatypes::Int32Type>();
        assert_eq!(id_array.value(0), 1); // First selected row
        assert_eq!(id_array.value(1), 3); // Third selected row

        Ok(())
    }

    #[test]
    fn test_type_validation() {
        let handler = create_test_handler();

        // Try to create joiner with non-String key column
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::INTEGER, false), // Wrong type
            StructField::new("value1", DataType::STRING, true),
            StructField::new("version", DataType::LONG, false),
        ]));

        let result = ArrowLookupJoiner::new(
            &handler,
            schema,
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_different_ordering() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with keys in order: [a, b, c]
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100),
        ];
        let row3 = [
            Scalar::String("c".to_string()),
            Scalar::String("value_c".to_string()),
            Scalar::Integer(3),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..], &row3[..]];
        let lookup_data = create_filtered_data(
            &handler,
            schema.clone(),
            &lookup_rows,
            vec![true, true, true],
        )?;

        let joiner = handler.new_lookup_join_handler(
            schema,
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input with keys in DIFFERENT order: [c, a, b]
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(10),
            Scalar::String("c".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(20),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
        ];
        let in_row3 = [
            Scalar::Integer(30),
            Scalar::String("b".to_string()),
            Scalar::Long(50),
        ];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        // Perform join
        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result - values should match keys, not input order
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row1 = [
            Scalar::Integer(10),
            Scalar::String("c".to_string()),
            Scalar::Long(50),
            Scalar::String("value_c".to_string()), // c's value
            Scalar::Integer(3),
        ];
        let exp_row2 = [
            Scalar::Integer(20),
            Scalar::String("a".to_string()),
            Scalar::Long(50),
            Scalar::String("value_a".to_string()), // a's value
            Scalar::Integer(1),
        ];
        let exp_row3 = [
            Scalar::Integer(30),
            Scalar::String("b".to_string()),
            Scalar::Long(50),
            Scalar::String("value_b".to_string()), // b's value
            Scalar::Integer(2),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_non_nullable_value_column_errors() {
        let handler = create_test_handler();

        // Create lookup schema with NON-NULLABLE value column
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, false), // Non-nullable!
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..]];
        let lookup_data = create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true])
            .expect("Failed to create lookup data");

        // Attempt to create joiner with non-nullable value column - should error
        let result = handler.new_lookup_join_handler(
            schema,
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data],
        );

        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("must be nullable"),
                "Error should mention that the column must be nullable. Got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_mixed_nullable_non_nullable_value_columns() {
        let handler = create_test_handler();

        // Create lookup schema with mixed nullable and non-nullable value columns
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, true), // Nullable - OK
            StructField::new("value2", DataType::INTEGER, false), // Non-nullable - should error
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..]];
        let lookup_data = create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true])
            .expect("Failed to create lookup data");

        // Attempt to create joiner with mixed nullable/non-nullable - should error
        let result = handler.new_lookup_join_handler(
            schema,
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        );

        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("must be nullable"),
                "Error should mention that the column must be nullable. Got: {}",
                error_msg
            );
            assert!(
                error_msg.contains("value2"),
                "Error should mention the non-nullable column name. Got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_all_nullable_value_columns_succeeds() -> DeltaResult<()> {
        let handler = create_test_handler();

        // Create lookup schema with all NULLABLE value columns
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, true), // Nullable
            StructField::new("value2", DataType::INTEGER, true), // Nullable
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Integer(1),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..]];
        let lookup_data = create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true])?;

        // Create joiner with nullable value columns - should succeed
        let result = handler.new_lookup_join_handler(
            schema,
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        );

        assert!(
            result.is_ok(),
            "Joiner creation should succeed with nullable value columns"
        );

        Ok(())
    }

    #[test]
    fn test_empty_value_columns_validation() {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Try to create joiner with no value columns - should error
        let result = ArrowLookupJoiner::new(
            &handler,
            schema,
            &column_name!("key"),
            &[], // Empty value columns
            &column_name!("version"),
            &[],
        );

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("at least one value column"));
        }
    }

    #[test]
    fn test_join_with_map_values() -> DeltaResult<()> {
        use crate::expressions::MapData;
        use crate::schema::MapType;

        let handler = create_test_handler();

        // Create schema with Map<String, Integer> value column
        let map_type = MapType::new(DataType::STRING, DataType::INTEGER, true);
        let schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new(
                "properties",
                DataType::Map(Box::new(map_type.clone())),
                true,
            ),
            StructField::new("version", DataType::LONG, false),
        ]));

        // Create lookup data with map values
        let map1 = MapData::try_new(
            map_type.clone(),
            vec![("name", Scalar::Integer(100)), ("age", Scalar::Integer(25))],
        )?;
        let map2 = MapData::try_new(
            map_type.clone(),
            vec![
                ("name", Scalar::Integer(200)),
                ("status", Scalar::Integer(1)),
            ],
        )?;

        let row1 = [
            Scalar::String("user1".to_string()),
            Scalar::Map(map1.clone()),
            Scalar::Long(100),
        ];
        let row2 = [
            Scalar::String("user2".to_string()),
            Scalar::Map(map2.clone()),
            Scalar::Long(100),
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        // Create joiner
        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("properties")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Create input data
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [
            Scalar::Integer(1),
            Scalar::String("user1".to_string()),
            Scalar::Long(50),
        ];
        let in_row2 = [
            Scalar::Integer(2),
            Scalar::String("user2".to_string()),
            Scalar::Long(50),
        ];
        let in_row3 = [
            Scalar::Integer(3),
            Scalar::String("user999".to_string()),
            Scalar::Long(50),
        ]; // Not found
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        // Perform join
        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Create expected result with maps
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new(
                "properties",
                DataType::Map(Box::new(map_type.clone())),
                true,
            ),
        ]));
        let exp_row1 = [
            Scalar::Integer(1),
            Scalar::String("user1".to_string()),
            Scalar::Long(50),
            Scalar::Map(map1.clone()),
        ];
        let exp_row2 = [
            Scalar::Integer(2),
            Scalar::String("user2".to_string()),
            Scalar::Long(50),
            Scalar::Map(map2.clone()),
        ];
        let exp_row3 = [
            Scalar::Integer(3),
            Scalar::String("user999".to_string()),
            Scalar::Long(50),
            Scalar::Null(DataType::Map(Box::new(map_type.clone()))),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        // Compare
        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_null_version_errors() -> DeltaResult<()> {
        let handler = create_test_handler();

        // Test 1: Null version in lookup data should error
        let lookup_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("version", DataType::LONG, true), // Nullable version
        ]));

        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("value_a".to_string()),
            Scalar::Null(DataType::LONG), // Null version
        ];
        let lookup_rows = vec![&row1[..]];
        let lookup_data =
            create_filtered_data(&handler, lookup_schema.clone(), &lookup_rows, vec![true])?;

        // This should error due to null version in lookup data
        let result = handler.new_lookup_join_handler(
            lookup_schema.clone(),
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data],
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Null version found in lookup data"));
        }

        // Test 2: Null version in input data should error
        let lookup_schema2 = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("version", DataType::LONG, false), // Non-nullable
        ]));

        let row2 = [
            Scalar::String("b".to_string()),
            Scalar::String("value_b".to_string()),
            Scalar::Long(100),
        ];
        let lookup_rows2 = vec![&row2[..]];
        let lookup_data2 =
            create_filtered_data(&handler, lookup_schema2.clone(), &lookup_rows2, vec![true])?;

        let joiner2 = handler.new_lookup_join_handler(
            lookup_schema2,
            &column_name!("key"),
            &[column_name!("value1")],
            &column_name!("version"),
            &[&lookup_data2],
        )?;

        // Create input with null version
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, true), // Nullable version
        ]));
        let in_row = [
            Scalar::String("b".to_string()),
            Scalar::Null(DataType::LONG),
        ];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        // This should error due to null version in input data
        let result = joiner2.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Null version found in input data"));
        }

        Ok(())
    }

    #[test]
    fn test_latest_version_wins_same_batch() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with duplicate key "a" with different versions
        // Version 100 should win over version 50
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("older".to_string()),
            Scalar::Integer(1),
            Scalar::Long(50), // Lower version
        ];
        let row2 = [
            Scalar::String("a".to_string()),
            Scalar::String("newer".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100), // Higher version - should win
        ];
        let lookup_rows = vec![&row1[..], &row2[..]];
        let lookup_data =
            create_filtered_data(&handler, schema.clone(), &lookup_rows, vec![true, true])?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Join with key "a"
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row = [Scalar::String("a".to_string()), Scalar::Long(75)];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Should get "newer" with value2=2 (from version 100, not version 50)
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row = [
            Scalar::String("a".to_string()),
            Scalar::Long(75),
            Scalar::String("newer".to_string()),
            Scalar::Integer(2),
        ];
        let expected_rows = vec![&exp_row[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_latest_version_wins_multiple_batches() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // First batch: key "a" with version 50
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("old_value".to_string()),
            Scalar::Integer(10),
            Scalar::Long(50),
        ];
        let batch1_rows = vec![&row1[..]];
        let batch1_data = create_filtered_data(&handler, schema.clone(), &batch1_rows, vec![true])?;

        // Second batch: key "a" with version 150 (higher - should win)
        let row2 = [
            Scalar::String("a".to_string()),
            Scalar::String("new_value".to_string()),
            Scalar::Integer(20),
            Scalar::Long(150),
        ];
        let batch2_rows = vec![&row2[..]];
        let batch2_data = create_filtered_data(&handler, schema.clone(), &batch2_rows, vec![true])?;

        // Third batch: key "a" with version 100 (middle - should not win)
        let row3 = [
            Scalar::String("a".to_string()),
            Scalar::String("middle_value".to_string()),
            Scalar::Integer(15),
            Scalar::Long(100),
        ];
        let batch3_rows = vec![&row3[..]];
        let batch3_data = create_filtered_data(&handler, schema.clone(), &batch3_rows, vec![true])?;

        // Create joiner with all three batches (order shouldn't matter)
        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&batch1_data, &batch2_data, &batch3_data],
        )?;

        // Join with key "a"
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row = [Scalar::String("a".to_string()), Scalar::Long(125)];
        let input_rows = vec![&in_row[..]];
        let input_data = create_filtered_data(&handler, input_schema, &input_rows, vec![true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Should get "new_value" with value2=20 (from version 150, the highest)
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row = [
            Scalar::String("a".to_string()),
            Scalar::Long(125),
            Scalar::String("new_value".to_string()),
            Scalar::Integer(20),
        ];
        let expected_rows = vec![&exp_row[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }

    #[test]
    fn test_latest_version_wins_mixed_keys() -> DeltaResult<()> {
        let handler = create_test_handler();
        let schema = create_test_schema();

        // Create lookup data with multiple keys, each with different versions
        // Key "a": version 50, then 100 (100 should win)
        let row1 = [
            Scalar::String("a".to_string()),
            Scalar::String("a_old".to_string()),
            Scalar::Integer(1),
            Scalar::Long(50),
        ];
        let row2 = [
            Scalar::String("a".to_string()),
            Scalar::String("a_new".to_string()),
            Scalar::Integer(2),
            Scalar::Long(100),
        ];
        // Key "b": version 200, then 150 (200 should win)
        let row3 = [
            Scalar::String("b".to_string()),
            Scalar::String("b_new".to_string()),
            Scalar::Integer(3),
            Scalar::Long(200),
        ];
        let row4 = [
            Scalar::String("b".to_string()),
            Scalar::String("b_old".to_string()),
            Scalar::Integer(4),
            Scalar::Long(150),
        ];
        // Key "c": only one version
        let row5 = [
            Scalar::String("c".to_string()),
            Scalar::String("c_value".to_string()),
            Scalar::Integer(5),
            Scalar::Long(75),
        ];
        let lookup_rows = vec![&row1[..], &row2[..], &row3[..], &row4[..], &row5[..]];
        let lookup_data = create_filtered_data(
            &handler,
            schema.clone(),
            &lookup_rows,
            vec![true, true, true, true, true],
        )?;

        let joiner = handler.new_lookup_join_handler(
            schema.clone(),
            &column_name!("key"),
            &[column_name!("value1"), column_name!("value2")],
            &column_name!("version"),
            &[&lookup_data],
        )?;

        // Join with all three keys
        let input_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
        ]));
        let in_row1 = [Scalar::String("a".to_string()), Scalar::Long(80)];
        let in_row2 = [Scalar::String("b".to_string()), Scalar::Long(180)];
        let in_row3 = [Scalar::String("c".to_string()), Scalar::Long(60)];
        let input_rows = vec![&in_row1[..], &in_row2[..], &in_row3[..]];
        let input_data =
            create_filtered_data(&handler, input_schema, &input_rows, vec![true, true, true])?;

        let result = joiner.join_raw(
            input_data.data(),
            input_data.selection_vector(),
            &column_name!("key"),
            &column_name!("version"),
        )?;

        // Expected: a_new (v100), b_new (v200), c_value (v75)
        let expected_schema = Arc::new(StructType::new_unchecked(vec![
            StructField::new("key", DataType::STRING, false),
            StructField::new("version", DataType::LONG, false),
            StructField::new("value1", DataType::STRING, true),
            StructField::new("value2", DataType::INTEGER, true),
        ]));
        let exp_row1 = [
            Scalar::String("a".to_string()),
            Scalar::Long(80),
            Scalar::String("a_new".to_string()),
            Scalar::Integer(2),
        ];
        let exp_row2 = [
            Scalar::String("b".to_string()),
            Scalar::Long(180),
            Scalar::String("b_new".to_string()),
            Scalar::Integer(3),
        ];
        let exp_row3 = [
            Scalar::String("c".to_string()),
            Scalar::Long(60),
            Scalar::String("c_value".to_string()),
            Scalar::Integer(5),
        ];
        let expected_rows = vec![&exp_row1[..], &exp_row2[..], &exp_row3[..]];
        let expected_data = handler.create_many(expected_schema, &expected_rows)?;
        let expected_batch = extract_record_batch(expected_data.as_ref())?;

        let result_batch = extract_record_batch(result.as_ref())?;
        assert_batches_eq(expected_batch, result_batch);

        Ok(())
    }
}
