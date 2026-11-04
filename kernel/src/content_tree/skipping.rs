//! Inclusive data skipping over the `content_stats` of Adaptive Metadata Tree (AMT) entries.
//!
//! [`AmtSkippingFilter`] rewrites a scan predicate into a predicate over the `content_stats` of a
//! manifest entry batch, following the Iceberg spec's inclusive rules: an entry is dropped only
//! when its stats prove that none of its rows match, and kept whenever stats are missing or cannot
//! decide. The same rules apply to data file entries, whose stats cover the file's rows, and to
//! `DataManifest` entries, whose stats cover every data file in the referenced leaf manifest.
//!
//! Bounds are valid whether or not the entry's stats are tight: tightness only promises they are
//! exact. Truncated string bounds remain valid bounds, and v4 timestamp bounds are not truncated,
//! so neither needs an adjustment.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use tracing::warn;

use crate::actions::visitors::SelectionVectorVisitor;
use crate::content_tree::stats::{field_with_id, stats_field_name, stats_schema};
use crate::content_tree::{
    CONTENT_STATS_FIELD_NAME, LOWER_BOUND, NAN_VALUE_COUNT, NULL_VALUE_COUNT, UPPER_BOUND,
    VALUE_COUNT,
};
use crate::expressions::{
    BinaryPredicateOp, ColumnName, Expression as Expr, JunctionPredicateOp, OpaquePredicateOpRef,
    Predicate as Pred, Scalar, lit,
};
use crate::kernel_predicates::{
    DataSkippingPredicateEvaluator, KernelPredicateEvaluator as _, KernelPredicateEvaluatorDefaults,
};
use crate::scan::data_skipping::stats_schema::is_skipping_eligible_datatype;
use crate::scan::data_skipping::{collect_junction_preds, comparison_predicate};
use crate::schema::{DataType, StructField, StructType};
use crate::utils::require;
use crate::{Engine, EngineData, KernelError, KernelResult, PredicateEvaluator, RowVisitor as _};

/// Parquet field ID of the `content_stats` column of a manifest entry.
const CONTENT_STATS_FIELD_ID: i32 = 146;

/// Drops the manifest entries whose `content_stats` prove that no row matches a predicate.
#[derive(Clone)]
pub(crate) struct AmtSkippingFilter {
    /// The `content_stats` column, narrowed to the predicate's leaves, that a batch passed to
    /// [`Self::apply`] must include.
    content_stats: StructField,
    /// Evaluates `DISTINCT(<rewritten predicate>, FALSE)`, which is false only for the entries
    /// proven to have no matching rows.
    evaluator: Arc<dyn PredicateEvaluator>,
}

impl AmtSkippingFilter {
    /// Builds a filter for `predicate`, whose column references are leaves of the physical
    /// `table_schema` (which carries the `parquet.field.id` of every leaf).
    ///
    /// Returns `None`, meaning every entry is kept, when no part of the predicate can be decided
    /// from stats or the filter cannot be built: skipping is an optimization, so it never fails a
    /// read.
    pub(crate) fn try_new(
        engine: &dyn Engine,
        predicate: &Pred,
        table_schema: &StructType,
    ) -> Option<Self> {
        let referenced: HashSet<_> = predicate
            .references()
            .into_iter()
            .map(stats_field_name)
            .collect();
        // Read only the stats of the referenced leaves.
        let stats = stats_schema(table_schema)
            .inspect_err(|e| warn!("Not skipping AMT entries: {e}"))
            .ok()?
            .into_fields()
            .filter(|field| referenced.contains(field.name()))
            .collect::<Vec<_>>();
        if stats.is_empty() {
            return None;
        }
        let stats = StructType::new_unchecked(stats);
        let skipping_predicate =
            AmtSkippingPredicateCreator { stats: &stats }.eval_sql_where(predicate)?;
        let content_stats = field_with_id(
            CONTENT_STATS_FIELD_NAME,
            stats.into(),
            true,
            CONTENT_STATS_FIELD_ID,
        );
        let input_schema = Arc::new(StructType::new_unchecked([content_stats.clone()]));
        let filter = Expr::from_pred(skipping_predicate).distinct(lit(false));
        let evaluator = engine
            .evaluation_handler()
            .new_predicate_evaluator(input_schema, Arc::new(filter))
            .inspect_err(|e| warn!("Not skipping AMT entries: {e}"))
            .ok()?;
        Some(Self {
            content_stats,
            evaluator,
        })
    }

    /// The `content_stats` column a manifest read must include for [`Self::apply`].
    pub(crate) fn content_stats_field(&self) -> &StructField {
        &self.content_stats
    }

    /// Returns, for each entry of `batch`, whether its rows might match the predicate. `batch` may
    /// hold other columns, but must include [`Self::content_stats_field`].
    pub(crate) fn apply(&self, batch: &dyn EngineData) -> KernelResult<Vec<bool>> {
        let keep = self.evaluator.evaluate(batch)?;
        let mut visitor = SelectionVectorVisitor::default();
        visitor.visit_rows_of(keep.as_ref())?;
        require!(
            visitor.selection_vector.len() == batch.len(),
            KernelError::internal_error(format!(
                "AMT skipping evaluated {} rows for a batch of {}",
                visitor.selection_vector.len(),
                batch.len()
            ))
        );
        Ok(visitor.selection_vector)
    }
}

/// Rewrites a predicate over physical leaf columns into an inclusive predicate over the
/// `content_stats.<leaf>.*` columns of an entry batch.
///
/// Counts use `value_count` rather than `record_count`, which counts entries, not rows, on a
/// `DataManifest` entry. Both count rules stay valid when the stats cover a superset of the live
/// rows, such as rows a deletion vector removes.
struct AmtSkippingPredicateCreator<'a> {
    /// The `content_stats` schema the rewritten predicate reads (see [`stats_schema`]).
    stats: &'a StructType,
}

impl AmtSkippingPredicateCreator<'_> {
    /// The `content_stats.<leaf>.<stat>` column of `col` and its type, or `None` if `col` has no
    /// such stat.
    fn stat_with_type(&self, col: &ColumnName, stat: &str) -> Option<(Expr, &DataType)> {
        let name = stats_field_name(col);
        let DataType::Struct(leaf) = self.stats.field(&name)?.data_type() else {
            return None;
        };
        let data_type = leaf.field(stat)?.data_type();
        let column = Expr::column([CONTENT_STATS_FIELD_NAME, name.as_str(), stat]);
        Some((column, data_type))
    }

    /// The `content_stats.<leaf>.<stat>` column of `col`, or `None` if `col` has no such stat.
    fn stat(&self, col: &ColumnName, stat: &str) -> Option<Expr> {
        self.stat_with_type(col, stat).map(|(column, _)| column)
    }

    /// The `stat` bound of `col`, or `None` unless it has type `data_type` and supports
    /// bounds-based skipping. The engine does not coerce types, so a bound of another type than
    /// the literal is never compared.
    fn bound(&self, col: &ColumnName, data_type: &DataType, stat: &str) -> Option<Expr> {
        let (bound, bound_type) = self.stat_with_type(col, stat)?;
        let eligible =
            matches!(data_type, DataType::Primitive(ptype) if is_skipping_eligible_datatype(ptype));
        (bound_type == data_type && eligible).then_some(bound)
    }

    /// Compares the `stat` bound of `col` against `val`, like
    /// [`DataSkippingPredicateEvaluator::partial_cmp_min_stat`].
    fn partial_cmp_bound(
        &self,
        col: &ColumnName,
        val: &Scalar,
        ord: Ordering,
        inverted: bool,
        stat: &str,
    ) -> Option<Pred> {
        let bound = self.bound(col, &val.data_type(), stat)?;
        let cmp = comparison_predicate(ord, bound, val, inverted);
        match val {
            // Total order separates -0.0 from 0.0 while IEEE order equates them, so bounds cannot
            // decide a comparison against zero.
            Scalar::Float(v) if *v == 0.0 => None,
            Scalar::Double(v) if *v == 0.0 => None,
            // Bounds exclude NaN, but the engine orders NaN beyond every number, so a NaN row can
            // match a comparison the bounds rule out. Trust the bounds only without NaNs.
            Scalar::Float(_) | Scalar::Double(_) => {
                let nans = self.stat(col, NAN_VALUE_COUNT)?;
                Some(Pred::or(cmp, Pred::ne(nans, lit(0i64))))
            }
            _ => Some(cmp),
        }
    }
}

impl DataSkippingPredicateEvaluator for AmtSkippingPredicateCreator<'_> {
    type Output = Pred;
    type ColumnStat = Expr;

    fn get_min_stat(&self, col: &ColumnName, data_type: &DataType) -> Option<Expr> {
        self.bound(col, data_type, LOWER_BOUND)
    }

    fn get_max_stat(&self, col: &ColumnName, data_type: &DataType) -> Option<Expr> {
        self.bound(col, data_type, UPPER_BOUND)
    }

    fn get_nullcount_stat(&self, col: &ColumnName) -> Option<Expr> {
        self.stat(col, NULL_VALUE_COUNT)
    }

    /// No entry-level stat counts rows for every entry type; see [`Self::eval_pred_is_null`].
    fn get_rowcount_stat(&self) -> Option<Expr> {
        None
    }

    fn partial_cmp_min_stat(
        &self,
        col: &ColumnName,
        val: &Scalar,
        ord: Ordering,
        inverted: bool,
    ) -> Option<Pred> {
        self.partial_cmp_bound(col, val, ord, inverted, LOWER_BOUND)
    }

    fn partial_cmp_max_stat(
        &self,
        col: &ColumnName,
        val: &Scalar,
        ord: Ordering,
        inverted: bool,
    ) -> Option<Pred> {
        self.partial_cmp_bound(col, val, ord, inverted, UPPER_BOUND)
    }

    fn eval_partial_cmp(
        &self,
        ord: Ordering,
        col: Expr,
        val: &Scalar,
        inverted: bool,
    ) -> Option<Pred> {
        Some(comparison_predicate(ord, col, val, inverted))
    }

    fn eval_pred_scalar(&self, val: &Scalar, inverted: bool) -> Option<Pred> {
        KernelPredicateEvaluatorDefaults::eval_pred_scalar(val, inverted).map(Pred::literal)
    }

    fn eval_pred_scalar_is_null(&self, val: &Scalar, inverted: bool) -> Option<Pred> {
        KernelPredicateEvaluatorDefaults::eval_pred_scalar_is_null(val, inverted).map(Pred::literal)
    }

    /// IS NULL cannot match an entry without nulls, and IS NOT NULL cannot match an entry whose
    /// values (`value_count` includes nulls) are all null.
    fn eval_pred_is_null(&self, col: &ColumnName, inverted: bool) -> Option<Pred> {
        let nulls = self.stat(col, NULL_VALUE_COUNT)?;
        let safe_to_skip = match inverted {
            true => self.stat(col, VALUE_COUNT)?,
            false => lit(0i64),
        };
        Some(Pred::ne(nulls, safe_to_skip))
    }

    fn eval_pred_binary_scalars(
        &self,
        op: BinaryPredicateOp,
        left: &Scalar,
        right: &Scalar,
        inverted: bool,
    ) -> Option<Pred> {
        KernelPredicateEvaluatorDefaults::eval_pred_binary_scalars(op, left, right, inverted)
            .map(Pred::literal)
    }

    fn eval_pred_cast(
        &self,
        _op: BinaryPredicateOp,
        _col: &ColumnName,
        _target: &DataType,
        _val: &Scalar,
        _inverted: bool,
    ) -> Option<Pred> {
        None
    }

    /// Opaque rewrites assume Delta stats semantics, so opaque predicates are never decided.
    fn eval_pred_opaque(
        &self,
        _op: &OpaquePredicateOpRef,
        _exprs: &[Expr],
        _inverted: bool,
    ) -> Option<Pred> {
        None
    }

    fn finish_eval_pred_junction(
        &self,
        op: JunctionPredicateOp,
        preds: &mut dyn Iterator<Item = Option<Pred>>,
        inverted: bool,
    ) -> Option<Pred> {
        Some(collect_junction_preds(op, preds, inverted))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::content_tree::TOTAL_BYTES;
    use crate::engine::sync::SyncEngine;
    use crate::expressions::{StructData, col, column_name};

    /// A physical table schema with a leaf of each type the tests cover, numbered by field ID.
    fn table_schema() -> StructType {
        let nested = StructType::new_unchecked([field_with_id("x", DataType::INTEGER, true, 7)]);
        StructType::new_unchecked([
            field_with_id("i", DataType::INTEGER, true, 1),
            field_with_id("d", DataType::DOUBLE, true, 2),
            field_with_id("f", DataType::FLOAT, true, 3),
            field_with_id("s", DataType::STRING, true, 4),
            field_with_id("ts", DataType::TIMESTAMP, true, 5),
            field_with_id("n", nested.into(), true, 6),
            field_with_id("b", DataType::BOOLEAN, true, 8),
        ])
    }

    /// One leaf's stats in an entry; `None` writes a null stat.
    #[derive(Clone, Default)]
    struct ColStats {
        lower: Option<Scalar>,
        upper: Option<Scalar>,
        values: Option<i64>,
        nulls: Option<i64>,
        nans: Option<i64>,
    }

    impl ColStats {
        fn with_nans(self, nans: Option<i64>) -> Self {
            Self { nans, ..self }
        }
    }

    /// Stats of 10 non-null, non-NaN values within `[lower, upper]`.
    fn bounds(lower: impl Into<Scalar>, upper: impl Into<Scalar>) -> ColStats {
        ColStats {
            lower: Some(lower.into()),
            upper: Some(upper.into()),
            values: Some(10),
            nulls: Some(0),
            nans: Some(0),
        }
    }

    /// Stats with counts but no bounds.
    fn counts(values: Option<i64>, nulls: Option<i64>) -> ColStats {
        ColStats {
            values,
            nulls,
            ..Default::default()
        }
    }

    /// One entry's `content_stats` value: `None` is a null struct, and a leaf absent from `entry`
    /// has a null stats struct.
    type Entry = Option<Vec<(ColumnName, ColStats)>>;

    /// The `content_stats` value of `entry` for the narrowed stats type `data_type`.
    fn stats_row(data_type: &DataType, entry: &Entry) -> Scalar {
        let (DataType::Struct(stats_type), Some(entry)) = (data_type, entry) else {
            return Scalar::Null(data_type.clone());
        };
        let values = stats_type.fields().map(|leaf_field| {
            match entry
                .iter()
                .find(|(c, _)| stats_field_name(c) == *leaf_field.name())
            {
                Some((_, stats)) => leaf_stats_row(leaf_field.data_type(), stats),
                None => Scalar::Null(leaf_field.data_type().clone()),
            }
        });
        let values = values.collect();
        Scalar::Struct(StructData::try_new(stats_type.fields().cloned().collect(), values).unwrap())
    }

    fn leaf_stats_row(data_type: &DataType, stats: &ColStats) -> Scalar {
        let DataType::Struct(leaf_type) = data_type else {
            panic!("leaf stats must be a struct: {data_type:?}");
        };
        let long = |v: Option<i64>| v.map_or(Scalar::Null(DataType::LONG), Scalar::Long);
        let values = leaf_type.fields().map(|f| {
            let null = || Scalar::Null(f.data_type().clone());
            match f.name().as_str() {
                LOWER_BOUND => stats.lower.clone().unwrap_or_else(null),
                UPPER_BOUND => stats.upper.clone().unwrap_or_else(null),
                VALUE_COUNT => long(stats.values),
                NULL_VALUE_COUNT => long(stats.nulls),
                NAN_VALUE_COUNT => long(stats.nans),
                TOTAL_BYTES => long(None),
                other => panic!("unexpected stat {other}"),
            }
        });
        let values = values.collect();
        Scalar::Struct(StructData::try_new(leaf_type.fields().cloned().collect(), values).unwrap())
    }

    /// Runs `predicate`'s filter over a batch with one row per entry through the engine's
    /// evaluator, returning which entries are kept.
    fn keeps(predicate: &Pred, entries: &[Entry]) -> Vec<bool> {
        let engine = SyncEngine::new();
        let schema = table_schema();
        let filter = AmtSkippingFilter::try_new(&engine, predicate, &schema).unwrap();
        let field = filter.content_stats_field().clone();
        let rows = entries
            .iter()
            .map(|entry| vec![stats_row(field.data_type(), entry)])
            .collect();
        let batch_schema = Arc::new(StructType::new_unchecked([field]));
        let batch = engine
            .evaluation_handler()
            .create_many(batch_schema, rows)
            .unwrap();
        filter.apply(batch.as_ref()).unwrap()
    }

    fn i_in(lower: i32, upper: i32) -> Vec<(ColumnName, ColStats)> {
        vec![(column_name!("i"), bounds(lower, upper))]
    }

    fn d_in(lower: f64, upper: f64, nans: Option<i64>) -> Vec<(ColumnName, ColStats)> {
        vec![(column_name!("d"), bounds(lower, upper).with_nans(nans))]
    }

    #[rstest]
    #[case::lt_at_lower(Pred::lt(col!("i"), lit(10)), i_in(10, 20), false)]
    #[case::lt_above_lower(Pred::lt(col!("i"), lit(11)), i_in(10, 20), true)]
    #[case::le_at_lower(Pred::le(col!("i"), lit(10)), i_in(10, 20), true)]
    #[case::le_below_lower(Pred::le(col!("i"), lit(9)), i_in(10, 20), false)]
    #[case::gt_at_upper(Pred::gt(col!("i"), lit(20)), i_in(10, 20), false)]
    #[case::ge_at_upper(Pred::ge(col!("i"), lit(20)), i_in(10, 20), true)]
    #[case::eq_within(Pred::eq(col!("i"), lit(15)), i_in(10, 20), true)]
    #[case::eq_above(Pred::eq(col!("i"), lit(21)), i_in(10, 20), false)]
    #[case::ne_within_range(Pred::ne(col!("i"), lit(15)), i_in(10, 20), true)]
    #[case::ne_single_value(Pred::ne(col!("i"), lit(10)), i_in(10, 10), false)]
    #[case::not_lt(Pred::not(Pred::lt(col!("i"), lit(21))), i_in(10, 20), false)]
    #[case::missing_leaf_stats(Pred::gt(col!("i"), lit(20)), vec![], true)]
    #[case::missing_bounds(
        Pred::gt(col!("i"), lit(20)),
        vec![(column_name!("i"), counts(Some(10), Some(0)))],
        true
    )]
    // The engine does not coerce, so a literal of another type is never compared (or an error).
    #[case::literal_type_mismatch(Pred::gt(col!("i"), lit(30i64)), i_in(10, 20), true)]
    #[case::cast_is_not_bounded(
        Pred::gt(Expr::cast(col!("i"), DataType::LONG), lit(30i64)),
        i_in(10, 20),
        true
    )]
    #[case::is_null_without_nulls(
        Pred::is_null(col!("i")),
        vec![(column_name!("i"), counts(Some(10), Some(0)))],
        false
    )]
    #[case::is_null_with_nulls(
        Pred::is_null(col!("i")),
        vec![(column_name!("i"), counts(Some(10), Some(3)))],
        true
    )]
    #[case::is_null_unknown_nulls(
        Pred::is_null(col!("i")),
        vec![(column_name!("i"), counts(Some(10), None))],
        true
    )]
    #[case::is_not_null_all_null(
        Pred::is_not_null(col!("i")),
        vec![(column_name!("i"), counts(Some(10), Some(10)))],
        false
    )]
    #[case::is_not_null_some_null(
        Pred::is_not_null(col!("i")),
        vec![(column_name!("i"), counts(Some(10), Some(3)))],
        true
    )]
    #[case::is_not_null_unknown_values(
        Pred::is_not_null(col!("i")),
        vec![(column_name!("i"), counts(None, Some(10)))],
        true
    )]
    // SQL WHERE semantics: a comparison never matches a null, so an all-null entry is skipped.
    #[case::comparison_on_all_null(
        Pred::gt(col!("i"), lit(5)),
        vec![(column_name!("i"), counts(Some(10), Some(10)))],
        false
    )]
    #[case::double_without_nans(Pred::gt(col!("d"), lit(3.0)), d_in(1.0, 2.0, Some(0)), false)]
    #[case::double_with_nans(Pred::gt(col!("d"), lit(3.0)), d_in(1.0, 2.0, Some(1)), true)]
    #[case::double_unknown_nans(Pred::gt(col!("d"), lit(3.0)), d_in(1.0, 2.0, None), true)]
    #[case::double_below_lower_with_nans(
        Pred::lt(col!("d"), lit(0.5)),
        d_in(1.0, 2.0, Some(1)),
        true
    )]
    #[case::double_eq_nan_without_nans(
        Pred::eq(col!("d"), lit(f64::NAN)),
        d_in(1.0, 2.0, Some(0)),
        false
    )]
    #[case::double_eq_nan_with_nans(
        Pred::eq(col!("d"), lit(f64::NAN)),
        d_in(1.0, 2.0, Some(2)),
        true
    )]
    #[case::double_gt_positive_zero(Pred::gt(col!("d"), lit(0.0)), d_in(-0.0, -0.0, Some(0)), true)]
    #[case::double_lt_negative_zero(
        Pred::lt(col!("d"), lit(-0.0)),
        d_in(0.0, 1.0, Some(0)),
        true
    )]
    #[case::float_without_nans(
        Pred::gt(col!("f"), lit(3.0f32)),
        vec![(column_name!("f"), bounds(1.0f32, 2.0f32))],
        false
    )]
    // Truncated string bounds are still bounds: "ab" <= every value <= "ac".
    #[case::string_within_truncated_bounds(
        Pred::eq(col!("s"), lit("abz")),
        vec![(column_name!("s"), bounds("ab", "ac"))],
        true
    )]
    #[case::string_below_truncated_lower(
        Pred::lt(col!("s"), lit("aa")),
        vec![(column_name!("s"), bounds("ab", "ac"))],
        false
    )]
    #[case::string_above_truncated_upper(
        Pred::gt(col!("s"), lit("ad")),
        vec![(column_name!("s"), bounds("ab", "ac"))],
        false
    )]
    // Iceberg timestamp bounds are exact microseconds, unlike Delta's millisecond `maxValues`.
    #[case::timestamp_gt_upper(
        Pred::gt(col!("ts"), lit(Scalar::Timestamp(4_000_500))),
        vec![(column_name!("ts"), bounds(Scalar::Timestamp(0), Scalar::Timestamp(4_000_500)))],
        false
    )]
    #[case::timestamp_ge_upper(
        Pred::ge(col!("ts"), lit(Scalar::Timestamp(4_000_500))),
        vec![(column_name!("ts"), bounds(Scalar::Timestamp(0), Scalar::Timestamp(4_000_500)))],
        true
    )]
    #[case::nested_leaf(
        Pred::gt(col!("n.x"), lit(5)),
        vec![(column_name!("n.x"), bounds(1, 5))],
        false
    )]
    // Booleans have no bounds-based skipping, but their null counts still apply.
    #[case::boolean_comparison(
        Pred::eq(col!("b"), lit(true)),
        vec![(column_name!("b"), counts(Some(10), Some(0)))],
        true
    )]
    #[case::boolean_is_null_without_nulls(
        Pred::is_null(col!("b")),
        vec![(column_name!("b"), counts(Some(10), Some(0)))],
        false
    )]
    #[case::and_with_undecidable_arm(
        Pred::and(Pred::gt(col!("i"), lit(20)), Pred::eq(col!("z"), lit(1))),
        i_in(10, 20),
        false
    )]
    #[case::or_with_undecidable_arm(
        Pred::or(Pred::gt(col!("i"), lit(20)), Pred::eq(col!("z"), lit(1))),
        i_in(10, 20),
        true
    )]
    #[case::or_of_skippable_arms(
        Pred::or(Pred::gt(col!("i"), lit(20)), Pred::gt(col!("d"), lit(3.0))),
        [i_in(10, 20), d_in(1.0, 2.0, Some(0))].concat(),
        false
    )]
    fn skips_only_entries_whose_stats_rule_out_a_match(
        #[case] predicate: Pred,
        #[case] entry: Vec<(ColumnName, ColStats)>,
        #[case] keep: bool,
    ) {
        assert_eq!(keeps(&predicate, &[Some(entry)]), vec![keep]);
    }

    #[test]
    fn filter_decides_each_entry_of_a_batch() {
        let predicate = Pred::gt(col!("i"), lit(20));
        let entries = [None, Some(i_in(10, 20)), Some(i_in(15, 25))];
        assert_eq!(keeps(&predicate, &entries), vec![true, false, true]);
    }

    #[rstest]
    #[case::unknown_column(Pred::gt(col!("z"), lit(1)))]
    #[case::struct_column(Pred::is_null(col!("n")))]
    fn filter_is_not_built_when_stats_cannot_decide(#[case] predicate: Pred) {
        let filter = AmtSkippingFilter::try_new(&SyncEngine::new(), &predicate, &table_schema());
        assert!(filter.is_none());
    }

    #[test]
    fn filter_is_not_built_when_a_referenced_leaf_lacks_a_field_id() {
        let schema = StructType::new_unchecked([StructField::nullable("i", DataType::INTEGER)]);
        let predicate = Pred::gt(col!("i"), lit(1));
        assert!(AmtSkippingFilter::try_new(&SyncEngine::new(), &predicate, &schema).is_none());
    }
}
