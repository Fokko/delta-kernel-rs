//! Lazy iterator for reading content root (Adaptive ContentTreeNode Tree) files.
//!
//! This module provides a lazy evaluation approach to reading content root metadata:
//! - Opens the parquet I/O stream immediately
//! - Defers metadata construction until the iterator is consumed
//! - Defers leaf manifest reading until root manifests are exhausted

use std::sync::Arc;

use tracing::instrument;
use url::Url;

use crate::log_replay::ActionsBatch;
use crate::schema::{SchemaRef, StructType};
use crate::{DeltaResult, EngineData, EvaluationHandler, ParquetHandler, PredicateRef, Version};

/// Lazy iterator that defers content root metadata construction until data is requested.
///
/// Opens the parquet I/O stream immediately but defers:
/// 1. Collecting batches and constructing ContentTreeNode object
/// 2. Getting root action batches
/// 3. Getting leaf action batches (if not skipping)
pub(crate) struct LazyContentRootIterator {
    state: LazyContentRootState,
    leaf_state: Option<LazyContentRootState>,
}

/// Shared context used across multiple states in the lazy content root iterator
struct ContentRootContext {
    parquet_handler: Arc<dyn ParquetHandler>,
    evaluation_handler: Arc<dyn EvaluationHandler>,
    checkpoint_read_schema: SchemaRef,
    table_root: Url,
    data_predicate: Option<PredicateRef>,
    skip_leaf_manifests: bool,
    /// Stats schema (from table configuration or predicate columns)
    stats_schema: Option<StructType>,
    /// Table schema (physical schema with field IDs for AMT)
    table_schema: Option<StructType>,
}

enum LazyContentRootState {
    /// Initial state - parquet stream is open but ContentTreeNode not yet constructed
    NotStarted {
        parquet_batches: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
        version: Version,
        path_in_log: String,
        context: ContentRootContext,
    },
    /// Currently reading from root manifest batches
    ReadingRoot {
        root_iter: Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>,
        metadata: Box<crate::content_tree::ContentTreeNode>,
        context: ContentRootContext,
    },
    /// Currently reading from leaf manifest batches
    ReadingLeaves {
        leaf_iter: Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>,
    },
    /// All batches have been read
    Done,
}

impl LazyContentRootIterator {
    /// Factory method to create a lazy iterator from a content root.
    ///
    /// Opens the parquet stream immediately but defers metadata construction
    /// and batch processing until the iterator is consumed.
    ///
    /// # Parameters
    /// - `parquet_handler`: Handler for reading parquet files
    /// - `evaluation_handler`: Handler for evaluating expressions
    /// - `content_root_url`: URL to the content root metadata file
    /// - `path_in_log`: Path as it appears in the Delta log
    /// - `table_root`: Table root URL
    /// - `checkpoint_read_schema`: Schema to use for reading actions
    /// - `data_predicate`: Optional predicate for manifest-level data skipping
    /// - `skip_leaf_manifests`: When true, only read root manifest
    /// - `stats_schema`: Optional stats schema (from table configuration or predicate columns)
    /// - `table_schema`: Optional table physical schema (with field IDs) for AMT content_stats
    ///   reading
    #[instrument(
        name = "content_tree.open_root",
        skip_all,
        fields(path = %content_root_url),
        err
    )]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_content_root(
        parquet_handler: Arc<dyn ParquetHandler>,
        evaluation_handler: Arc<dyn EvaluationHandler>,
        content_root_url: &Url,
        path_in_log: String,
        table_root: Url,
        checkpoint_read_schema: SchemaRef,
        data_predicate: Option<PredicateRef>,
        skip_leaf_manifests: bool,
        stats_schema: Option<&StructType>,
        table_schema: Option<&StructType>,
    ) -> DeltaResult<Self> {
        // Convert schemas to owned for storage in context
        let stats_schema = stats_schema.cloned();
        let table_schema = table_schema.cloned();

        // Open the parquet stream using the metadata helper
        // Pass both schemas so content_stats is filtered to only requested columns
        let (parquet_batches, version, path_in_log) =
            crate::content_tree::ContentTreeNode::open_stream(
                parquet_handler.clone(),
                content_root_url,
                path_in_log,
                table_schema.as_ref(),
                stats_schema.as_ref(),
            )?;

        let context = ContentRootContext {
            parquet_handler,
            evaluation_handler,
            checkpoint_read_schema,
            table_root,
            data_predicate,
            skip_leaf_manifests,
            stats_schema,
            table_schema,
        };

        Ok(Self {
            state: LazyContentRootState::NotStarted {
                parquet_batches,
                version,
                path_in_log,
                context,
            },
            leaf_state: None,
        })
    }
}

impl Iterator for LazyContentRootIterator {
    type Item = DeltaResult<ActionsBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match std::mem::replace(&mut self.state, LazyContentRootState::Done) {
                LazyContentRootState::NotStarted {
                    parquet_batches,
                    version,
                    path_in_log,
                    context,
                } => {
                    // Lazily construct the ContentTreeNode object on first access
                    let data: Vec<Box<dyn EngineData>> = {
                        let _span =
                            tracing::info_span!("content_tree.collect_root_batches").entered();
                        match parquet_batches.collect::<DeltaResult<Vec<_>>>() {
                            Ok(d) => d,
                            Err(e) => return Some(Err(e)),
                        }
                    };

                    // Construct metadata from collected batches with the parsed version.
                    // This also validates that the root only contains supported entry types.
                    let metadata = Box::new(
                        match crate::content_tree::ContentTreeNode::from_batches_with_version(
                            data,
                            version,
                            path_in_log,
                            context.table_root.clone(),
                        ) {
                            Ok(m) => m,
                            Err(e) => return Some(Err(e)),
                        },
                    );
                    // Root batches exhausted. If skipping leaves, we're done
                    self.leaf_state = if context.skip_leaf_manifests {
                        None
                    } else {
                        let _span = tracing::info_span!("content_tree.get_leaves").entered();
                        // Lazily read leaf manifests now that root is exhausted
                        // Construct manifest batch schema with filtered content_stats for data
                        // skipping
                        let manifest_batch_schema = context
                            .table_schema
                            .as_ref()
                            .zip(context.stats_schema.as_ref())
                            .and_then(|(ts, ss)| {
                                crate::content_tree::ContentTreeNodeEntry::to_schema_with_content_stats(ts, ss)
                                    .ok()
                                    .map(Arc::new)
                            });

                        let leaf_refs = match metadata.manifest_references(
                            context.data_predicate.as_ref(),
                            Some(&context.evaluation_handler),
                            context.stats_schema.as_ref(),
                            context.table_schema.as_ref(),
                            manifest_batch_schema.as_ref(),
                        ) {
                            Ok(refs) => refs,
                            Err(e) => return Some(Err(e)),
                        };

                        let leaf_iter =
                            match crate::content_tree::ContentTreeNode::non_root_action_batches(
                                leaf_refs,
                                context.parquet_handler.clone(),
                                context.evaluation_handler.clone(),
                                &context.checkpoint_read_schema,
                                &context.table_root,
                                context.data_predicate.as_ref(),
                                context.table_schema.as_ref(),
                                context.stats_schema.as_ref(),
                            ) {
                                Ok(iter) => iter,
                                Err(e) => return Some(Err(e)),
                            };
                        Some(LazyContentRootState::ReadingLeaves { leaf_iter })
                    };
                    // Get root batches using the handler-based method
                    let root_iter = match metadata.root_action_batches_with_handler(
                        context.evaluation_handler.as_ref(),
                        &context.checkpoint_read_schema,
                        &[],
                        context.data_predicate.as_ref(),
                        context.table_schema.as_ref(),
                        context.stats_schema.as_ref(),
                    ) {
                        Ok(iter) => iter,
                        Err(e) => return Some(Err(e)),
                    };

                    // Transition to ReadingRoot state
                    self.state = LazyContentRootState::ReadingRoot {
                        root_iter,
                        metadata,
                        context,
                    };
                    continue;
                }
                LazyContentRootState::ReadingRoot {
                    mut root_iter,
                    metadata,
                    context,
                } => {
                    // Try to get next batch from root
                    if let Some(batch) = root_iter.next() {
                        self.state = LazyContentRootState::ReadingRoot {
                            root_iter,
                            metadata,
                            context,
                        };
                        return Some(batch);
                    }

                    self.state = self.leaf_state.take().unwrap_or(LazyContentRootState::Done);
                    continue;
                }
                LazyContentRootState::ReadingLeaves { mut leaf_iter } => {
                    // Try to get next batch from leaves
                    if let Some(batch) = leaf_iter.next() {
                        self.state = LazyContentRootState::ReadingLeaves { leaf_iter };
                        return Some(batch);
                    }

                    // All batches exhausted
                    self.state = LazyContentRootState::Done;
                    return None;
                }
                LazyContentRootState::Done => return None,
            }
        }
    }
}
