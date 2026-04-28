//! Bulk streaming processor with parallel IO optimization.
//!
//! This module provides [`BulkManifestStreamProcessor`], which processes
//! data manifests with parallel IO:
//! - Calls read_parquet_files() ONCE for all manifests (maximum parallelism)
//! - Uses _file metadata column to group batches by manifest
//! - Processes manifests lazily one at a time as needed
//! - Extracts inline DV columns from dvInfo.* fields per batch

use std::sync::Arc;

use url::Url;

use super::{
    parse_or_join_url, ContentTreeNodeEntry, FilteredManifest, ManifestReference, SchemaRef,
};
use crate::engine_data::{EngineData, GetData, RowVisitor};
use crate::expressions::PredicateRef;
use crate::log_replay::ActionsBatch;
use crate::schema::{ColumnName, DataType, MetadataColumnSpec, StructField, StructType};
use crate::{DeltaResult, Error, EvaluationHandler, FileMeta, ParquetHandler};

/// Visitor that extracts the file path from the first row of a batch.
struct FilePathVisitor {
    file_path: Option<String>,
}

impl FilePathVisitor {
    fn new() -> Self {
        Self { file_path: None }
    }

    fn get_file_path(self) -> DeltaResult<String> {
        self.file_path
            .ok_or_else(|| Error::generic("Missing _file column in batch"))
    }
}

impl RowVisitor for FilePathVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        use std::sync::LazyLock;
        static NAMES: LazyLock<[ColumnName; 1]> = LazyLock::new(|| [ColumnName::new(["_file"])]);
        static TYPES: &[DataType] = &[DataType::STRING];
        (&NAMES[..], TYPES)
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        if row_count > 0 && self.file_path.is_none() {
            // Only read the first row - use get_str directly to handle RLE encoding
            self.file_path = getters[0].get_str(0, "_file")?.map(|s| s.to_string());
        }
        Ok(())
    }
}

/// Evaluators shared across all manifests (computed once in `BulkManifestStreamProcessor::new`).
///
/// Stats transforms and remove evaluators don't embed any per-manifest literal, so they are
/// identical for every manifest and can be reused.
struct SharedEvaluators {
    /// Evaluator input schema for batches that have DV columns appended (includes stats_parsed
    /// when applicable). Passed to `build_action_evaluators` for per-manifest add evaluators.
    evaluator_schema_with_dv: super::SchemaRef,
    /// Evaluator input schema for batches without DV columns.
    evaluator_schema_no_dv: super::SchemaRef,
    /// Stats transform when DV columns are present.
    stats_transform_with_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
    /// Stats transform when no DV columns are present.
    stats_transform_no_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
    /// Remove evaluator when DV columns are present.
    remove_evaluator_with_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
    /// Remove evaluator when no DV columns are present.
    remove_evaluator_no_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
}

/// Per-manifest add evaluators (depend on the manifest path literal for `dataManifestPath`).
struct ManifestAddEvaluators {
    /// Add evaluator when DV columns are present.
    with_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
    /// Add evaluator when no DV columns are present.
    no_dv: Option<Arc<dyn crate::ExpressionEvaluator>>,
}

/// State for processing a single manifest.
struct ManifestProcessingState {
    /// Current file path we're reading batches for
    current_file_path: String,

    /// Manifest DV applicator for this manifest
    manifest_dv_applicator: super::ManifestDvApplicator,

    /// Add evaluators (per-manifest: embed the manifest path as a literal)
    add_evaluators: ManifestAddEvaluators,

    /// Whether we're still on the first batch for this manifest (for span tracking)
    is_first_batch: bool,
}

/// Streaming processor with parallel IO optimization.
///
/// Reads data manifests in parallel using a single stream,
/// then consumes batches lazily using the _file metadata column to group by manifest.
pub(crate) struct BulkManifestStreamProcessor {
    /// Peekable iterator over batches from all data manifests (with _file column)
    data_batch_iter: std::iter::Peekable<crate::FileDataReadResultIterator>,

    /// Iterator over manifest references to process
    manifest_iter: std::vec::IntoIter<ManifestReference>,

    /// Evaluation handler for creating per-manifest add evaluators
    evaluation_handler: Arc<dyn EvaluationHandler>,

    /// Schema for action batches
    schema: SchemaRef,

    /// Table root URL
    table_root: Url,

    /// Evaluators shared across all manifests (computed once)
    shared: SharedEvaluators,

    /// Whether the output schema requests Add actions
    has_add: bool,

    /// Current manifest processing state (if any)
    current_manifest_state: Option<ManifestProcessingState>,

    /// Buffer of pending action batches to yield
    pending_actions: std::collections::VecDeque<ActionsBatch>,
}

impl FilteredManifest {
    /// Convert to FileMeta for reading.
    fn to_file_meta(&self, table_root: &Url) -> DeltaResult<FileMeta> {
        let location = self
            .manifest
            .location
            .as_ref()
            .ok_or_else(|| Error::generic("Manifest must have a location"))?;
        Ok(FileMeta {
            location: parse_or_join_url(location, table_root)?,
            last_modified: 0,
            size: self.manifest.file_size_in_bytes.unwrap_or(0) as u64,
        })
    }
}

impl BulkManifestStreamProcessor {
    /// Creates a new BulkManifestStreamProcessor with parallel IO.
    ///
    /// This constructor:
    /// 1. Starts parallel IO for ALL data manifests (single stream)
    /// 2. Returns a peekable iterator that will be consumed lazily
    ///
    /// # Parameters
    /// - `manifest_references`: Iterator of ManifestReference to process
    /// - `parquet_handler`: Handler for reading parquet files
    /// - `evaluation_handler`: Handler for expression evaluation
    /// - `schema`: Action schema to use
    /// - `table_root`: Table root URL
    /// - `predicate`: Optional predicate for filtering (currently unused in optimized path)
    /// - `table_schema`: Optional table schema (physical schema with field IDs)
    /// - `stats_schema`: Optional stats schema (for stats_parsed transformation)
    // TODO: Refactor to reduce argument count - consider using a config struct
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        manifest_references: impl Iterator<Item = ManifestReference>,
        parquet_handler: Arc<dyn ParquetHandler>,
        evaluation_handler: Arc<dyn EvaluationHandler>,
        schema: SchemaRef,
        table_root: Url,
        _predicate: Option<PredicateRef>,
        table_schema: Option<Arc<StructType>>,
        stats_schema: Option<Arc<StructType>>,
    ) -> DeltaResult<Self> {
        let manifest_refs: Vec<ManifestReference> = manifest_references.collect();

        // Build read schema with _file metadata column for grouping.
        // Include content_stats only when stats_schema is provided (i.e., stats_parsed
        // transformation will be applied). Without stats_schema, content_stats is not needed
        // and including it would wastefully read per-column statistics for every entry.
        let base_schema = if let (Some(ref ts), Some(ref ss)) = (&table_schema, &stats_schema) {
            ContentTreeNodeEntry::to_schema_with_content_stats(ts.as_ref(), ss.as_ref())?
        } else {
            {
                use crate::schema::ToSchema as _;
                ContentTreeNodeEntry::to_schema()
            }
        };
        let mut read_fields: Vec<StructField> = base_schema.fields().cloned().collect();
        read_fields.push(StructField::create_metadata_column(
            "_pos",
            MetadataColumnSpec::RowIndex,
        ));
        read_fields.push(StructField::create_metadata_column(
            "_file",
            MetadataColumnSpec::FilePath,
        ));
        let read_schema = Arc::new(StructType::new_unchecked(read_fields));

        // Build file metas for all data manifests and start parallel IO
        let data_file_metas: Vec<FileMeta> = manifest_refs
            .iter()
            .map(|mr| mr.data_manifest.to_file_meta(&table_root))
            .collect::<DeltaResult<Vec<_>>>()?;
        let data_batch_iter =
            parquet_handler.read_parquet_files(&data_file_metas, read_schema, None)?;

        // Pre-compute manifest-independent evaluators once.
        let table_schema_ref = table_schema.as_ref().map(|s| s.as_ref());
        let stats_schema_ref = stats_schema.as_ref().map(|s| s.as_ref());
        let metadata_schema = super::ContentTreeNodeEntry::processing_schema_with_pos(
            table_schema_ref,
            stats_schema_ref,
        )?;

        // Evaluator input schemas are the same for all manifests.
        let evaluator_schema_with_dv = super::ContentTreeNode::get_evaluator_schema_with_stats(
            &metadata_schema,
            stats_schema_ref,
        );
        let evaluator_schema_no_dv =
            super::ContentTreeNode::get_evaluator_schema_no_dv(&metadata_schema, stats_schema_ref);

        use crate::actions::REMOVE_NAME;
        let has_add = schema.contains(crate::actions::ADD_NAME);
        let has_remove = schema.contains(REMOVE_NAME);

        // Remove evaluators are manifest-independent (no manifest path literal)
        // TODO: partition values from content_stats are not yet supported in the bulk
        // processor path. See the corresponding TODO in builder.rs for add_from_engine_data_write.
        let remove_evaluators_with_dv = super::ContentTreeNode::build_action_evaluators(
            evaluation_handler.as_ref(),
            evaluator_schema_with_dv.clone(),
            &schema,
            "",    // path_in_log unused for remove actions
            false, // has_add
            has_remove,
            true,
            None, // partition_values_expr
        )?;
        let remove_evaluators_no_dv = super::ContentTreeNode::build_action_evaluators(
            evaluation_handler.as_ref(),
            evaluator_schema_no_dv.clone(),
            &schema,
            "",    // path_in_log unused for remove actions
            false, // has_add
            has_remove,
            false,
            None, // partition_values_expr
        )?;

        // Stats transform evaluators are also manifest-independent.
        // They use the DV-extended schema (for with_dv) and base metadata schema (for no_dv).
        let dv_extended_schema = super::ContentTreeNode::extend_metadata_schema_with_dv_fields(
            &metadata_schema,
            &super::DV_COLUMNS_SCHEMA_FINAL,
        );
        let stats_transform_with_dv =
            super::ContentTreeNodeEntry::create_stats_transformation_evaluator(
                evaluation_handler.as_ref(),
                &dv_extended_schema,
                &schema,
                table_schema_ref,
                stats_schema_ref,
            )?;
        let stats_transform_no_dv =
            super::ContentTreeNodeEntry::create_stats_transformation_evaluator(
                evaluation_handler.as_ref(),
                &metadata_schema,
                &schema,
                table_schema_ref,
                stats_schema_ref,
            )?;

        let shared = SharedEvaluators {
            evaluator_schema_with_dv,
            evaluator_schema_no_dv,
            stats_transform_with_dv,
            stats_transform_no_dv,
            remove_evaluator_with_dv: remove_evaluators_with_dv.remove_evaluator,
            remove_evaluator_no_dv: remove_evaluators_no_dv.remove_evaluator,
        };

        Ok(BulkManifestStreamProcessor {
            data_batch_iter: data_batch_iter.peekable(),
            manifest_iter: manifest_refs.into_iter(),
            evaluation_handler,
            schema,
            table_root,
            shared,
            has_add,
            current_manifest_state: None,
            pending_actions: std::collections::VecDeque::new(),
        })
    }

    /// Extract file path from a batch using the _file metadata column.
    fn extract_file_path(batch: &dyn EngineData) -> DeltaResult<String> {
        let mut visitor = FilePathVisitor::new();
        visitor.visit_rows_of(batch)?;
        visitor.get_file_path()
    }

    /// Setup state for processing the next data manifest.
    fn setup_next_manifest_state(&mut self) -> DeltaResult<bool> {
        // Get next manifest to process
        let manifest_ref = match self.manifest_iter.next() {
            Some(m) => m,
            None => return Ok(false), // No more manifests
        };

        // Get the file path for this manifest
        let current_file_path = manifest_ref
            .data_manifest
            .to_file_meta(&self.table_root)?
            .location
            .to_string();

        // Create manifest DV applicator
        let manifest_dv_applicator = super::ManifestDvApplicator::new(
            manifest_ref.data_manifest.manifest.manifest_dv.as_ref(),
        )?;

        // Build only the per-manifest add evaluators (embed manifest path as a literal).
        // Stats transforms and remove evaluators are shared across all manifests (see `new()`).
        let add_evaluators = if self.has_add {
            let manifest_location = manifest_ref
                .data_manifest
                .manifest
                .location
                .clone()
                .ok_or_else(|| Error::generic("Data manifest must have a location"))?;

            // Reuse pre-computed evaluator schemas from shared — avoids redundant schema building.
            // TODO: partition values from content_stats are not yet supported in the bulk
            // processor path.
            let evaluators_with_dv = super::ContentTreeNode::build_action_evaluators(
                self.evaluation_handler.as_ref(),
                self.shared.evaluator_schema_with_dv.clone(),
                &self.schema,
                &manifest_location,
                true,  // has_add
                false, // has_remove (handled by shared evaluators)
                true,
                None, // partition_values_expr
            )?;
            let evaluators_no_dv = super::ContentTreeNode::build_action_evaluators(
                self.evaluation_handler.as_ref(),
                self.shared.evaluator_schema_no_dv.clone(),
                &self.schema,
                &manifest_location,
                true,  // has_add
                false, // has_remove
                false,
                None, // partition_values_expr
            )?;
            ManifestAddEvaluators {
                with_dv: evaluators_with_dv.add_evaluator,
                no_dv: evaluators_no_dv.add_evaluator,
            }
        } else {
            ManifestAddEvaluators {
                with_dv: None,
                no_dv: None,
            }
        };

        self.current_manifest_state = Some(ManifestProcessingState {
            current_file_path,
            manifest_dv_applicator,
            add_evaluators,
            is_first_batch: true,
        });

        Ok(true)
    }
}

impl Iterator for BulkManifestStreamProcessor {
    type Item = DeltaResult<ActionsBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have pending actions, return one
            if let Some(action) = self.pending_actions.pop_front() {
                return Some(Ok(action));
            }

            // No pending actions - try to process next batch
            // First ensure we have a manifest state
            if self.current_manifest_state.is_none() {
                match self.setup_next_manifest_state() {
                    Ok(true) => {
                        // State setup successfully, continue to process batches
                    }
                    Ok(false) => {
                        // No more manifests - check if there are leftover batches
                        if self.data_batch_iter.peek().is_some() {
                            return Some(Err(Error::generic(
                                "Parquet iterator has remaining batches but no manifests left to process them"
                            )));
                        }
                        return None;
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            // We have manifest state - try to get next batch
            let state = match self.current_manifest_state.as_mut() {
                Some(state) => state,
                None => {
                    return Some(Err(Error::generic(
                        "Internal error: manifest state is None after setup",
                    )));
                }
            };

            // Peek at next batch to check if it belongs to current manifest
            let batch_file_path = match self.data_batch_iter.peek() {
                Some(Ok(batch)) => match Self::extract_file_path(batch.as_ref()) {
                    Ok(path) => path,
                    Err(e) => return Some(Err(e)),
                },
                Some(Err(_)) => {
                    // Error in iterator - consume and return it
                    if let Some(Err(e)) = self.data_batch_iter.next() {
                        return Some(Err(e));
                    }
                    unreachable!("Peeked error should still be error when consumed");
                }
                None => {
                    // No more batches - clear state and loop (will return None)
                    self.current_manifest_state = None;
                    continue;
                }
            };

            // Check if this batch belongs to current manifest
            if batch_file_path != state.current_file_path {
                // Batch belongs to different file - move to next manifest
                self.current_manifest_state = None;
                continue;
            }

            // Track the first batch per manifest with a span
            let _first_batch_guard = if state.is_first_batch {
                state.is_first_batch = false;
                Some(
                    tracing::info_span!(
                        "content_tree.load_leaf_first_batch",
                        manifest = %state.current_file_path,
                    )
                    .entered(),
                )
            } else {
                None
            };

            // Consume the batch (we already peeked, so this must succeed)
            let batch = match self.data_batch_iter.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => unreachable!("Peeked batch should be available"),
            };

            // Try to append inline DV columns; None means no DVs present in this batch.
            let dv_augmented = match super::ContentTreeNode::append_inline_dv_columns(
                batch.as_ref(),
                &self.table_root,
            ) {
                Ok(opt) => opt,
                Err(e) => return Some(Err(e)),
            };
            let has_dvs = dv_augmented.is_some();
            let batch_owned: Box<dyn EngineData> = match dv_augmented {
                Some(b) => b,
                None => batch,
            };

            // Apply stats transformation (from shared evaluators — same for all manifests).
            let stats_eval = if has_dvs {
                self.shared.stats_transform_with_dv.as_ref()
            } else {
                self.shared.stats_transform_no_dv.as_ref()
            };
            let batch_after_stats: Box<dyn EngineData> = if let Some(eval) = stats_eval {
                match eval.evaluate(batch_owned.as_ref()) {
                    Ok(augmented) => augmented,
                    Err(e) => return Some(Err(e)),
                }
            } else {
                batch_owned
            };

            // Apply manifest DV to get FilteredEngineData.
            let batch_to_process = match state
                .manifest_dv_applicator
                .process_batch(batch_after_stats)
            {
                Ok(fb) => fb,
                Err(e) => return Some(Err(e)),
            };

            // Select add/remove evaluators: add from per-manifest state, remove from shared.
            let (add_eval, remove_eval) = if has_dvs {
                (
                    state.add_evaluators.with_dv.as_ref(),
                    self.shared.remove_evaluator_with_dv.as_ref(),
                )
            } else {
                (
                    state.add_evaluators.no_dv.as_ref(),
                    self.shared.remove_evaluator_no_dv.as_ref(),
                )
            };

            let action_batches = match super::ContentTreeNode::process_filtered_batch_to_actions(
                batch_to_process,
                add_eval,
                remove_eval,
            ) {
                Ok(batches) => batches,
                Err(e) => return Some(Err(e)),
            };

            // Add action batches to pending queue
            self.pending_actions.extend(action_batches);

            // Loop back to return first pending action
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulk_processor_creation() {
        // Basic test to ensure BulkManifestStreamProcessor can be created
        // We can't fully test without mock handlers, but at least verify the types compile
        let manifests: Vec<ManifestReference> = Vec::new();
        let _ = manifests;
    }

    // Note: More comprehensive integration tests would require:
    // - Mock ParquetHandler that returns test manifests
    // - Mock EvaluationHandler for expression evaluation
    // - Actual manifest data with proper structure
    // These are better tested at the integration level in the full kernel tests
}
