//! Manifest-level data skipping for Adaptive ContentTreeNode Tree (AMT).
//!
//! This module provides filtering capabilities that apply predicates to manifest entries
//! during parquet reading phase, before ContentTreeNodeEntry materialization.

use std::collections::HashSet;
use std::sync::Arc;

use tracing::{debug, error};

use super::stats::create_content_stats_to_stats_parsed_expr;
use crate::actions::visitors::SelectionVectorVisitor;
use crate::expressions::{Expression, PredicateRef};
use crate::kernel_predicates::KernelPredicateEvaluator;
use crate::scan::data_skipping::DataSkippingPredicateCreator;
use crate::schema::{DataType, SchemaRef, StructField, StructType};
use crate::{
    DeltaResult, EngineData, EvaluationHandler, ExpressionEvaluator, PredicateEvaluator,
    RowVisitor as _,
};

/// Filter that applies data skipping to manifest entries during parquet reading.
///
/// Evaluates predicates against content_stats at the EngineData level,
/// before materializing ContentTreeNodeEntry objects. This allows skipping entire
/// rows/entries that don't match the predicate.
///
/// Uses a three-stage evaluation pipeline:
/// 1. Transform: content_stats → stats_parsed format
/// 2. Skip: evaluate rewritten predicate on stats_parsed
/// 3. Filter: convert predicate result to selection vector
///
/// TODO: Stages 2-4 duplicate logic from scan::data_skipping::DataSkippingFilter.
/// Consider refactoring into a shared helper function that takes stats_parsed as input:
/// `apply_data_skipping_predicate(stats_parsed, skipping_evaluator, filter_evaluator) -> Vec<bool>`
pub(crate) struct ManifestDataSkippingFilter {
    /// Expression evaluator to transform content_stats → stats_parsed
    transform_evaluator: Arc<dyn ExpressionEvaluator>,
    /// Schema for the transformed stats_parsed data
    _stats_schema: Arc<StructType>,
    /// Evaluator for the data skipping predicate
    skipping_evaluator: Arc<dyn PredicateEvaluator>,
    /// Evaluator to convert predicate result to selection vector (DISTINCT(output, false))
    filter_evaluator: Arc<dyn PredicateEvaluator>,
}

impl ManifestDataSkippingFilter {
    /// Creates a new manifest filter.
    ///
    /// Returns `None` if filtering not possible (no predicate, no table schema, predicate not
    /// eligible for data skipping, engine doesn't support required operations, etc.)
    ///
    /// # Arguments
    ///
    /// * `evaluation_handler` - The evaluation handler for expression/predicate evaluation
    /// * `predicate` - The predicate to evaluate for data skipping
    /// * `stats_schema` - The expected stats schema (from table configuration or predicate columns)
    /// * `table_schema` - The table's physical schema with field ID metadata
    /// * `manifest_batch_schema` - The schema of the manifest EngineData batches (contains
    ///   content_stats column)
    ///
    /// # Returns
    ///
    /// `Some(Self)` if filtering can be applied, `None` otherwise.
    pub(crate) fn new(
        evaluation_handler: &Arc<dyn EvaluationHandler>,
        predicate: &PredicateRef,
        stats_schema: &StructType,
        table_schema: &StructType,
        manifest_batch_schema: &SchemaRef,
    ) -> Option<Self> {
        debug!(
            "Creating manifest data skipping filter for {:#?}",
            predicate
        );

        // Check if manifest batch has content_stats column
        manifest_batch_schema
            .as_ref()
            .field(crate::content_tree::CONTENT_STATS_FIELD_NAME)?;

        // Step 1: Create transform expression (content_stats → stats_parsed)
        // Only create transforms for columns present in stats_schema (may be subset of table
        // columns)
        let transform_expr = create_content_stats_to_stats_parsed_expr(table_schema, stats_schema)
            .inspect_err(|e| {
                error!("Failed to create content_stats transform expression: {e}");
            })
            .ok()?;

        // Wrap the flat stats expression and schema in a `stats_parsed` field to match
        // the column references emitted by DataSkippingPredicateCreator (e.g.
        // `stats_parsed.minValues.id`).
        let flat_stats_schema = Arc::new(stats_schema.clone());
        let wrapped_expr = Arc::new(Expression::struct_from([transform_expr]));
        let wrapped_schema = Arc::new(StructType::new_unchecked([StructField::nullable(
            "stats_parsed",
            DataType::Struct(Box::new((*flat_stats_schema).clone())),
        )]));

        // Create evaluator for the transform expression: manifest batch -> { stats_parsed: { ... }
        // }
        let transform_evaluator = evaluation_handler
            .new_expression_evaluator(
                manifest_batch_schema.clone(),
                wrapped_expr,
                DataType::Struct(Box::new((*wrapped_schema).clone())),
            )
            .inspect_err(|e| error!("Failed to create transform evaluator: {e}"))
            .ok()?;

        // Step 2: Rewrite predicate for data skipping
        let data_skipping_pred =
            DataSkippingPredicateCreator::new(&HashSet::new()).eval_sql_where(predicate)?;
        debug!("Data skipping predicate: {:#?}", data_skipping_pred);

        // Create evaluator for the data skipping predicate
        let skipping_evaluator = evaluation_handler
            .new_predicate_evaluator(wrapped_schema.clone(), Arc::new(data_skipping_pred))
            .inspect_err(|e| error!("Failed to create skipping evaluator: {e}"))
            .ok()?;

        // Step 3: Create filter predicate: DISTINCT(output, false)
        // This converts the boolean predicate result to a selection vector
        use crate::expressions::column_expr;
        let filter_pred = Arc::new(column_expr!("output").distinct(Expression::literal(false)));

        let filter_evaluator = evaluation_handler
            .new_predicate_evaluator(wrapped_schema.clone(), filter_pred)
            .inspect_err(|e| error!("Failed to create filter evaluator: {e}"))
            .ok()?;

        Some(Self {
            transform_evaluator,
            _stats_schema: wrapped_schema,
            skipping_evaluator,
            filter_evaluator,
        })
    }

    /// Apply filter to a manifest batch, returning selection vector.
    ///
    /// # Arguments
    ///
    /// * `batch` - The manifest EngineData batch containing content_stats column
    ///
    /// # Returns
    ///
    /// A `Vec<bool>` selection vector where `true` means keep the row, `false` means skip.
    ///
    /// # Errors
    ///
    /// Returns error if evaluation fails at any stage.
    pub(crate) fn apply(&self, batch: &dyn EngineData) -> DeltaResult<Vec<bool>> {
        use crate::utils::require;

        // Stage 1: Transform content_stats to stats_parsed
        let stats_parsed = self.transform_evaluator.evaluate(batch)?;
        require!(
            stats_parsed.len() == batch.len(),
            crate::Error::generic(format!(
                "Transform evaluator produced {} rows but expected {}",
                stats_parsed.len(),
                batch.len()
            ))
        );

        // Stage 2: Evaluate data skipping predicate on stats_parsed
        let skipping_result = self.skipping_evaluator.evaluate(&*stats_parsed)?;
        require!(
            skipping_result.len() == batch.len(),
            crate::Error::generic(format!(
                "Predicate evaluator produced {} rows but expected {}",
                skipping_result.len(),
                batch.len()
            ))
        );

        // Stage 3: Convert predicate result to selection vector
        let selection_vector_data = self.filter_evaluator.evaluate(skipping_result.as_ref())?;
        require!(
            selection_vector_data.len() == batch.len(),
            crate::Error::generic(format!(
                "Filter evaluator produced {} rows but expected {}",
                selection_vector_data.len(),
                batch.len()
            ))
        );

        // Visit the engine's selection vector to produce a Vec<bool>
        let mut visitor = SelectionVectorVisitor::default();
        visitor.visit_rows_of(selection_vector_data.as_ref())?;
        debug!("Final selection vector: {:?}", visitor.selection_vector);
        Ok(visitor.selection_vector)
    }
}
