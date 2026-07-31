//! Adaptive metadata tree (AMT) log-commit helpers.
//!
//! When `metadataTree-experimental` is enabled and the transaction commits removes (or DV
//! updates) to the Delta log, remove actions must follow the Iceberg V4 adaptive metadata RFC:
//! null `deletionTimestamp`, extended file metadata, and statistics.

use std::sync::{Arc, LazyLock};

use crate::actions::visitors::visit_back_reference_at;
use crate::engine_data::{GetData, TypedGetData as _};
use crate::error::Error;
use crate::expressions::{column_name, ColumnName, Transform};
use crate::scan::data_skipping::stats_schema::STATS_NUM_RECORDS;
use crate::scan::log_replay::STATS_PARSED_NAME;
use crate::scan::scan_row_schema;
use crate::schema::{ColumnNamesAndTypes, DataType, SchemaRef, StructField, StructType};
use crate::transaction::update;
use crate::utils::require;
use crate::{
    DeltaResult, Engine, EngineData, Expression, FilteredEngineData, FilteredRowVisitor,
    RowIndexIterator,
};

/// Builds the evaluator that [`prepare_remove_scan_batch_for_amt`] uses to synthesize stats, or
/// `None` when `uses_amt_log_commit` is false and no stats synthesis is needed.
///
/// `columns_to_drop` identifies which batch shape this remove generation pass will see: each pass
/// is homogeneous, so DV-update removes (carrying the temporary `newDeletionVector` column) need an
/// evaluator built for the wider intermediate schema, while plain removes use scan rows.
pub(crate) fn build_remove_stats_prep_evaluator(
    engine: &dyn Engine,
    uses_amt_log_commit: bool,
    columns_to_drop: &[&str],
) -> DeltaResult<Option<Arc<dyn crate::ExpressionEvaluator>>> {
    uses_amt_log_commit
        .then(|| {
            let input_schema = if update::is_deletion_vector_update(columns_to_drop) {
                update::intermediate_dv_schema().clone()
            } else {
                scan_row_schema()
            };
            build_stats_prep_evaluator_for_input(engine, input_schema)
        })
        .transpose()
}

/// Builds an evaluator that appends a minimal `stats_parsed` column from `numRecords` for the
/// given input schema.
fn build_stats_prep_evaluator_for_input(
    engine: &dyn Engine,
    input_schema: SchemaRef,
) -> DeltaResult<Arc<dyn crate::ExpressionEvaluator>> {
    // Insert after the last input field so the synthesized column lands at the end, matching
    // `output_schema` below.
    let insert_after = input_schema
        .fields()
        .next_back()
        .map(|field| field.name().clone());

    let mut fields: Vec<StructField> = input_schema.fields().cloned().collect();
    fields.push(StructField::nullable(
        STATS_PARSED_NAME,
        StructType::new_unchecked([StructField::nullable(STATS_NUM_RECORDS, DataType::LONG)]),
    ));
    let output_schema = Arc::new(StructType::new_unchecked(fields));
    let transform = Expression::transform(Transform::new_top_level().with_inserted_field(
        insert_after,
        Expression::struct_from([Expression::column([STATS_NUM_RECORDS])]).into(),
    ));
    engine.evaluation_handler().new_expression_evaluator(
        input_schema,
        Arc::new(transform),
        output_schema.into(),
    )
}

/// When scan metadata lacks `stats_parsed` but has `numRecords` (common for content-tree
/// entries), synthesize a minimal `stats_parsed` column so remove generation can coalesce stats.
///
/// Returns `None` (meaning "use `batch` as-is") when `prep_evaluator` is `None` (non-AMT commit),
/// when the batch already has a `stats_parsed` column, or when it has no `numRecords` to synthesize
/// from.
pub(crate) fn prepare_remove_scan_batch_for_amt(
    prep_evaluator: Option<&dyn crate::ExpressionEvaluator>,
    batch: &dyn EngineData,
) -> DeltaResult<Option<Box<dyn EngineData>>> {
    let Some(prep_evaluator) = prep_evaluator else {
        return Ok(None);
    };
    // DV-update batches carry a temporary `newDeletionVector` column; stats prep still applies.
    if batch.has_field(&ColumnName::new([STATS_PARSED_NAME])) {
        return Ok(None);
    }
    if !batch.has_field(&ColumnName::new([STATS_NUM_RECORDS])) {
        return Ok(None);
    }

    Ok(Some(prep_evaluator.evaluate(batch)?))
}

/// Validates scan metadata batches that will be transformed into AMT log-commit remove actions
/// (including DV-update remove halves, which use the same scan rows).
///
/// Each selected row must carry a non-null `size` and statistics. Statistics may be sourced from
/// the `stats` JSON string, a `numRecords` column, or a `stats_parsed` column; the `stats` JSON
/// string is only required when the batch exposes neither of the latter two (i.e. it is the sole
/// possible stats source). A `backReference`, when present, must be structurally well-formed (both
/// `manifest` and `pos` set), but it is not required: normal Delta removes of files that never
/// entered the AMT legitimately lack one.
pub(crate) fn validate_remove_metadata_for_amt_log_commit<'a>(
    batches: impl Iterator<Item = &'a FilteredEngineData>,
) -> DeltaResult<()> {
    let mut first_failure: Option<String> = None;
    for batch in batches {
        let data = batch.data();
        let has_stats_parsed = data.has_field(&ColumnName::new([STATS_PARSED_NAME]));
        let has_num_records = data.has_field(&ColumnName::new([STATS_NUM_RECORDS]));
        let mut visitor = AmtRemoveMetadataValidator {
            first_failure: &mut first_failure,
            require_stats_string: !has_stats_parsed && !has_num_records,
        };
        visitor.visit_rows_of(batch)?;
        if first_failure.is_some() {
            break;
        }
    }
    if let Some(message) = first_failure {
        return Err(Error::invalid_transaction_state(message));
    }
    Ok(())
}

struct AmtRemoveMetadataValidator<'a> {
    first_failure: &'a mut Option<String>,
    /// True when the batch exposes neither a `stats_parsed` nor a `numRecords` column, so the
    /// `stats` JSON string is the only possible stats source and must therefore be present. When a
    /// `numRecords`/`stats_parsed` column exists, stats are supplied through the remove-generation
    /// pipeline (see `build_remove_stats_prep_evaluator`) rather than the JSON string.
    require_stats_string: bool,
}

impl FilteredRowVisitor for AmtRemoveMetadataValidator<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            (
                vec![
                    column_name!("path"),
                    column_name!("size"),
                    column_name!("stats"),
                    column_name!("fileConstantValues.backReference.manifest"),
                    column_name!("fileConstantValues.backReference.pos"),
                ],
                vec![
                    DataType::STRING,
                    DataType::LONG,
                    DataType::STRING,
                    DataType::STRING,
                    DataType::LONG,
                ],
            )
                .into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit_filtered<'b>(
        &mut self,
        getters: &[&'b dyn GetData<'b>],
        rows: RowIndexIterator<'_>,
    ) -> DeltaResult<()> {
        require!(
            getters.len() == 5,
            Error::internal_error(format!(
                "Expected 5 getters for AMT remove validation, got {}",
                getters.len()
            ))
        );

        // `rows` yields only the selected row indices; deselected rows are already filtered out.
        for i in rows {
            // A remove row should always carry a path. Skip defensively if one doesn't, since a
            // pathless row cannot identify a file to validate (or report in an error message).
            let Some(path): Option<String> = getters[0].get_opt(i, "path")? else {
                continue;
            };

            let size: Option<i64> = getters[1].get_opt(i, "size")?;
            let stats: Option<String> = getters[2].get_opt(i, "stats")?;

            // Structural check only: a `backReference` is not required (normal Delta removes of
            // files that never entered the AMT legitimately lack one), but a partial one (only
            // `manifest` or only `pos` set) is malformed and rejected by `visit_back_reference_at`.
            let _ = visit_back_reference_at(
                i,
                &getters[3..5],
                "fileConstantValues.backReference.manifest",
                "fileConstantValues.backReference.pos",
            )?;

            // Extended file metadata (RFC): every remove must carry a non-null `size`, and the
            // `stats` JSON string when it is the only available stats source (see
            // `require_stats_string`). Otherwise stats flow through the `numRecords`/`stats_parsed`
            // columns during remove generation.
            let missing_stats = self.require_stats_string && stats.is_none();
            if size.is_none() || missing_stats {
                *self.first_failure = Some(format!(
                    "metadataTree-experimental log commits require remove actions to carry stats \
                     and size (extended file metadata), but they are missing for file '{path}'"
                ));
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::arrow::array::StringArray;
    use crate::engine::default::DefaultEngine;
    use crate::object_store::memory::InMemory;
    use crate::scan::scan_row_schema;
    use crate::utils::test_utils::string_array_to_engine_data;
    use crate::Engine;

    fn test_engine() -> Arc<dyn Engine> {
        Arc::new(DefaultEngine::new(Arc::new(InMemory::new())))
    }

    fn scan_row_batch(
        engine: &Arc<dyn Engine>,
        json_rows: &[&str],
        selection: Vec<bool>,
    ) -> FilteredEngineData {
        let strings: StringArray = json_rows.iter().map(|s| Some(*s)).collect();
        let data = engine
            .json_handler()
            .parse_json(string_array_to_engine_data(strings), scan_row_schema())
            .unwrap();
        FilteredEngineData::try_new(data, selection).unwrap()
    }

    fn err_message(result: DeltaResult<()>) -> String {
        result.unwrap_err().to_string()
    }

    const FC_VALUES: &str = r#""fileConstantValues":{"partitionValues":{},"baseRowId":null,"defaultRowCommitVersion":null,"tags":null,"clusteringProvider":null"#;

    #[test]
    fn valid_log_only_remove_passes() {
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"log.parquet","size":100,"modificationTime":1,"stats":"{{\"numRecords\":10}}","deletionVector":null,{FC_VALUES},"backReference":null}},"numRecords":null}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        validate_remove_metadata_for_amt_log_commit(std::iter::once(&batch)).unwrap();
    }

    #[test]
    fn content_tree_row_without_back_reference_passes() {
        // A backReference is not required: removes of files that never entered the AMT lack one.
        // Stats are supplied at the value level via `numRecords`, so validation succeeds.
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"leaf.parquet","size":100,"modificationTime":1,"stats":null,"deletionVector":null,{FC_VALUES},"backReference":null}},"numRecords":10}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        validate_remove_metadata_for_amt_log_commit(std::iter::once(&batch)).unwrap();
    }

    #[test]
    fn tree_row_with_null_stats_and_null_num_records_value_passes() {
        // Tree-resident removes commonly have a null `stats` JSON string and a null top-level
        // `numRecords` value; the `numRecords` column presence signals that stats are supplied
        // through the remove-generation pipeline, so validation must not require the JSON string.
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"leaf.parquet","size":100,"modificationTime":1,"stats":null,"deletionVector":null,{FC_VALUES},"backReference":null}},"numRecords":null}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        validate_remove_metadata_for_amt_log_commit(std::iter::once(&batch)).unwrap();
    }

    #[test]
    fn content_tree_row_with_back_reference_passes() {
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"leaf.parquet","size":100,"modificationTime":1,"stats":null,"deletionVector":null,{FC_VALUES},"backReference":{{"manifest":"m.json","pos":2}}}},"numRecords":10}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        validate_remove_metadata_for_amt_log_commit(std::iter::once(&batch)).unwrap();
    }

    #[test]
    fn missing_size_fails() {
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"f.parquet","size":null,"modificationTime":1,"stats":"{{\"numRecords\":10}}","deletionVector":null,{FC_VALUES},"backReference":null}},"numRecords":null}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        let err = err_message(validate_remove_metadata_for_amt_log_commit(
            std::iter::once(&batch),
        ));
        assert!(err.contains("stats and size"), "unexpected error: {err}");
    }

    #[test]
    fn partial_back_reference_fails() {
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"leaf.parquet","size":100,"modificationTime":1,"stats":null,"deletionVector":null,{FC_VALUES},"backReference":{{"manifest":"m.json","pos":null}}}},"numRecords":10}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![true]);
        let err = err_message(validate_remove_metadata_for_amt_log_commit(
            std::iter::once(&batch),
        ));
        assert!(
            err.contains("backReference.manifest") || err.contains("must both be present"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unselected_rows_are_skipped() {
        let engine = test_engine();
        let row = format!(
            r#"{{"path":"leaf.parquet","size":100,"modificationTime":1,"stats":null,"deletionVector":null,{FC_VALUES},"backReference":null}},"numRecords":10}}"#
        );
        let batch = scan_row_batch(&engine, &[&row], vec![false]);
        validate_remove_metadata_for_amt_log_commit(std::iter::once(&batch)).unwrap();
    }
}
