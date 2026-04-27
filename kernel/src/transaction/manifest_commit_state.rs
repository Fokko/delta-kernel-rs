use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use url::Url;

use super::leaf_writer::{LeafNodeWriter, LeafNodeWriterResult};
use crate::content_tree::builder::{
    log_replay_schema, ContentRootRebuildProcessor, ContentTreeNodeBuilder,
};
use crate::content_tree::{ContentTreeNode, ContentTreeNodeEntry};
use crate::error::Error;
use crate::log_reader::commit::CommitReader;
use crate::log_replay::ActionsBatch;
use crate::snapshot::SnapshotRef;
use crate::utils::require;
use crate::{DeltaResult, Engine, FileMeta, Version};

/// Commit mode that uses a caller-supplied root manifest instead of having kernel build one.
///
/// Constructed via [`crate::transaction::Transaction::with_explicit_root_manifest`]. On commit,
/// the checkpoint action references the supplied file as the content root; kernel writes no new
/// root manifest parquet file. This mode is mutually exclusive with [`ManifestCommitState`].
pub struct ExplicitRootManifestCommit {
    /// The caller-supplied root manifest file.
    pub(super) file: FileMeta,
}

impl ExplicitRootManifestCommit {
    /// Validates snapshot preconditions and file location, then constructs an
    /// [`ExplicitRootManifestCommit`].
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot has no existing checkpoint action, if the checkpoint
    /// does not cover the snapshot version, or if `file.location` is not under the table root
    /// (same scheme, host, and path prefix).
    pub(super) fn new(file: FileMeta, read_snapshot: &SnapshotRef) -> DeltaResult<Self> {
        let Some(checkpoint_action) = read_snapshot.checkpoint_action() else {
            return Err(Error::invalid_transaction_state(
                "explicit root manifest commit requires an existing checkpoint action on the table",
            ));
        };

        require!(
            checkpoint_action.version >= read_snapshot.version(),
            Error::invalid_transaction_state(format!(
                "explicit root manifest commit requires no delta log commits after the latest \
                 checkpoint that are pending metadata-tree replay; checkpoint covers version {} \
                 but snapshot is at {}",
                checkpoint_action.version,
                read_snapshot.version()
            ))
        );

        let table_root = read_snapshot.table_root();
        require!(
            file.location.scheme() == table_root.scheme()
                && file.location.host_str() == table_root.host_str()
                && file.location.path().starts_with(table_root.path()),
            Error::generic(format!(
                "manifest location {:?} is not under the table root {:?}",
                file.location, table_root
            ))
        );

        Ok(ExplicitRootManifestCommit { file })
    }
}

/// Replay delta log commit files at or after `from_version` through `processor`.
///
/// Returns pre-transformed [`EngineData`] batches in ContentTreeNodeEntry schema — one per
/// non-empty surviving commit batch. The caller pushes these to a [`ContentTreeNodeBuilder`]
/// via [`add_pre_built_log_batch`].
///
/// [`EngineData`]: crate::EngineData
/// [`add_pre_built_log_batch`]: ContentTreeNodeBuilder::add_pre_built_log_batch
fn replay_log_commits(
    processor: &mut ContentRootRebuildProcessor,
    engine: &dyn Engine,
    log_segment: &crate::log_segment::LogSegment,
    from_version: Version,
) -> DeltaResult<Vec<Box<dyn crate::EngineData>>> {
    let content_root_version = from_version.checked_sub(1);
    let reader = CommitReader::try_new(
        engine,
        log_segment,
        log_replay_schema(),
        content_root_version,
    )?;
    let mut batches = Vec::new();
    for batch in reader {
        if let Some(fed) = processor.process_log_batch(batch?)? {
            batches.push(fed.apply_selection_vector()?);
        }
    }
    Ok(batches)
}

/// Applies `processor` over the existing content root and returns the live entries.
fn replay_content_root(
    processor: &mut ContentRootRebuildProcessor,
    engine: &dyn Engine,
    root_path_str: &str,
    table_root: &Url,
) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
    let content_root_url = table_root
        .join(root_path_str)
        .map_err(|e| Error::generic(format!("Failed to parse content root URL: {e}")))?;
    let (content_root_iter, _, _) = ContentTreeNode::open_stream(
        engine.parquet_handler(),
        &content_root_url,
        root_path_str.to_owned(),
        None,
        None,
    )?;
    let mut entries = Vec::new();
    for batch in content_root_iter {
        entries.extend(processor.process_root_batch(ActionsBatch::new(batch?, false))?);
    }
    Ok(entries)
}

/// State for a manifest commit (content-tree update).
///
/// Obtained by calling [`crate::transaction::Transaction::with_manifest_commit`]. Holds all
/// tree-manipulation state and exposes tree-focused methods for partition-aware compaction
/// workflows. `Transaction` retains access to all normal builder and commit methods while
/// this struct exists.
///
/// # Lifetime
///
/// `ManifestCommitState` holds a mutable reference into the owning `Transaction` (via
/// `Option<ManifestCommitState>` stored inside it). Drop `ManifestCommitState` before calling
/// [`crate::transaction::Transaction::commit`] or any other `&mut Transaction` method.
pub struct ManifestCommitState {
    // Snapshot info copied from Transaction at construction (SnapshotRef is Arc, clone is cheap).
    pub(super) version_to_write: Version,
    pub(super) snapshot_id: i64,
    pub(super) read_snapshot: SnapshotRef,

    // Manifest-commit-specific state, moved out of Transaction.
    pub(super) aggregated_manifest_dvs: HashMap<String, roaring::RoaringTreemap>,
    pub(super) aggregated_unreconciled: HashSet<String>,
    pub(super) aggregated_root_dv_actions: HashSet<String>,
    pub(super) leaf_manifests: Vec<ContentTreeNodeEntry>,
    pub(super) root_released: bool,
    pub(super) cached_root_manifest_url: OnceCell<Option<Url>>,
}

impl ManifestCommitState {
    /// Create a new `ManifestCommitState` from copied snapshot fields.
    pub(super) fn new(
        version_to_write: Version,
        snapshot_id: i64,
        read_snapshot: SnapshotRef,
    ) -> Self {
        ManifestCommitState {
            version_to_write,
            snapshot_id,
            read_snapshot,
            aggregated_manifest_dvs: HashMap::new(),
            aggregated_unreconciled: HashSet::new(),
            aggregated_root_dv_actions: HashSet::new(),
            leaf_manifests: Vec::new(),
            root_released: false,
            cached_root_manifest_url: OnceCell::new(),
        }
    }

    /// Returns a [`Scan`] that replays actions from both the root manifest (if present) and the
    /// delta log.
    ///
    /// After calling this method, the transaction records that the root has been "released" to the
    /// client. Any subsequent [`LeafNodeWriter`] instances created via [`new_leaf_node_writer`]
    /// will NOT track root entries for removal, since the client is responsible for managing which
    /// actions move from root to leaves.
    ///
    /// This is useful for partition-aware compaction workflows where the client wants to:
    /// 1. Read all actions from root + delta log.
    /// 2. Process and partition them according to custom logic.
    /// 3. Write partitioned actions to leaf manifests.
    /// 4. Commit the transaction with only the leaf manifests (root stays unchanged).
    ///
    /// # Returns
    ///
    /// A [`Scan`] that will return all Add actions from:
    /// - The root manifest (if present in the checkpoint) -- entries where `dataManifestPath` is
    ///   NULL.
    /// - All delta log files since the checkpoint -- entries where `dataManifestPath` is NULL.
    ///
    /// The scan explicitly excludes actions from leaf manifests (where `dataManifestPath` is
    /// non-NULL) using an internal skip mechanism.
    ///
    /// # Errors
    ///
    /// Returns an error if called more than once per transaction.
    ///
    /// [`Scan`]: crate::scan::Scan
    /// [`new_leaf_node_writer`]: ManifestCommitState::new_leaf_node_writer
    pub fn release_root_and_delta_actions(&mut self) -> DeltaResult<crate::scan::Scan> {
        if self.root_released {
            return Err(Error::generic(
                "release_root_and_delta_actions() can only be called once per transaction",
            ));
        }
        self.root_released = true;

        // TODO: we need custom replay here to:
        // 1. Add any currently added/removed actions to the log replay.
        // 2. Do leaf book-keeping for incrementally add/removed files (primarily DV updates).
        //
        // Create a scan that ONLY reads root + delta log (excluding leaf manifests).
        // Include stats columns so that parsed stats are available for AMT leaf population.
        let scan = crate::scan::ScanBuilder::new(self.read_snapshot.clone())
            .skip_leaf_manifests(true)
            .include_all_stats_columns()
            .build()?;

        Ok(scan)
    }

    /// Create a new [`LeafNodeWriter`] for this transaction.
    ///
    /// The writer can be used to add files to a leaf manifest, which will be written and
    /// incorporated into the root manifest when the transaction commits.
    ///
    /// # Arguments
    ///
    /// * `engine` - The engine to use for fetching the root manifest URL (only on first call;
    ///   subsequent calls use the cached value).
    ///
    /// # Returns
    ///
    /// A new [`LeafNodeWriter`] initialized with the transaction's table root, version, snapshot
    /// ID, and root manifest URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the root manifest URL cannot be constructed.
    pub fn new_leaf_node_writer(&self, engine: &dyn Engine) -> DeltaResult<LeafNodeWriter> {
        let root_manifest_url = if let Some(url) = self.cached_root_manifest_url.get() {
            url.clone()
        } else {
            let url = self.root_manifest_url(engine)?;
            let _ = self.cached_root_manifest_url.set(url.clone());
            url
        };

        let track_root_removals = !self.root_released;

        let root_manifest_path = root_manifest_url.as_ref().map(|url| {
            crate::content_tree::absolute_to_relative_path(url, self.read_snapshot.table_root())
        });

        let column_mapping_mode = self
            .read_snapshot
            .table_configuration()
            .column_mapping_mode();
        let physical_schema = Arc::new(
            self.read_snapshot
                .schema()
                .as_ref()
                .make_physical(column_mapping_mode)?,
        );

        let writer = LeafNodeWriter::new(
            self.read_snapshot.table_root().clone(),
            self.version_to_write,
            self.snapshot_id,
            physical_schema,
            track_root_removals,
            root_manifest_path,
        );

        Ok(writer)
    }

    /// Returns the URL of the root manifest from the latest content root, if one exists.
    ///
    /// # Arguments
    ///
    /// * `engine` - Unused; reserved for future I/O if needed.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Url))` - The URL of the root manifest.
    /// * `Ok(None)` - No checkpoint action exists yet.
    /// * `Err` - Error constructing the URL.
    pub fn root_manifest_url(&self, _engine: &dyn Engine) -> DeltaResult<Option<Url>> {
        let checkpoint_action = self.read_snapshot.checkpoint_action();
        let table_root = self.read_snapshot.table_root();
        Ok(checkpoint_action.and_then(|ca| table_root.join(&ca.content_root.path).ok()))
    }

    /// Incorporate leaf writer results into this manifest commit.
    ///
    /// - Detects duplicate unreconciled files across leaves (returns an error if found).
    /// - Unions manifest deletion vectors (roaring bitmaps) across leaves.
    /// - Collects leaf manifest entries to include in the root when the transaction commits.
    ///
    /// # Arguments
    ///
    /// * `leaf_result` - The result from calling `finish()` on a [`LeafNodeWriter`].
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn add_leaf(&mut self, leaf_result: LeafNodeWriterResult) -> DeltaResult<()> {
        self.aggregated_unreconciled
            .extend(leaf_result.root_entries_to_remove);

        self.aggregated_root_dv_actions
            .extend(leaf_result.root_dv_entries_to_remove);

        for (manifest_url, row_indices) in leaf_result.manifest_dvs {
            let entry = self
                .aggregated_manifest_dvs
                .entry(manifest_url)
                .or_default();
            *entry |= row_indices;
        }

        if let Some(data_manifest) = leaf_result.data_file_manifest_written {
            self.leaf_manifests.push(data_manifest);
        }
        Ok(())
    }

    /// Creates and populates a [`ContentTreeNodeBuilder`] from the current table state.
    ///
    /// This is the setup phase of a manifest commit. It determines the appropriate baseline for
    /// the new content tree by inspecting the existing checkpoint action:
    ///
    /// - If [`release_root_and_delta_actions`](Self::release_root_and_delta_actions) was called,
    ///   the root is cleared — the client will repopulate it via leaf manifests.
    /// - If delta log commits exist since the last checkpoint, replays them through a
    ///   [`ContentRootRebuildProcessor`] to produce a correct merged view of the content root.
    /// - If the content root is already current (no log commits since checkpoint), loads it
    ///   directly without replay.
    /// - If no checkpoint exists, returns an empty builder.
    ///
    /// The returned builder is ready to accept new file additions and leaf manifest updates via
    /// [`apply_to_builder`](Self::apply_to_builder).
    ///
    /// # Arguments
    ///
    /// * `engine` - Engine for reading log commit files and the content root parquet file.
    pub(super) fn initialize_content_root_builder(
        &self,
        engine: &dyn Engine,
    ) -> DeltaResult<ContentTreeNodeBuilder> {
        let column_mapping_mode = self
            .read_snapshot
            .table_configuration()
            .column_mapping_mode();
        let physical_schema = self
            .read_snapshot
            .schema()
            .as_ref()
            .make_physical(column_mapping_mode)?;
        let table_root = self.read_snapshot.table_root().clone();
        let current_version = self.read_snapshot.version();

        // If a content root exists and is current, load it directly and return — no replay needed.
        // Otherwise fall through: either no checkpoint (replay from v0) or log commits exist
        // after the checkpoint version (replay from checkpoint.version + 1).
        let (log_start_version, root_path) =
            if let Some(checkpoint_action) = self.read_snapshot.checkpoint_action() {
                let log_start_version = checkpoint_action.version + 1;
                if log_start_version > current_version {
                    let mut builder = ContentTreeNodeBuilder::from_content_root(
                        engine,
                        &checkpoint_action.content_root,
                        table_root,
                        physical_schema,
                        self.version_to_write,
                    )?;
                    if self.root_released {
                        builder.clear_root_data_and_dv_entries();
                    }
                    return Ok(builder);
                }
                (
                    log_start_version,
                    Some(checkpoint_action.content_root.path.clone()),
                )
            } else {
                (0, None)
            };

        let mut builder = ContentTreeNodeBuilder::new_for(
            table_root.clone(),
            self.version_to_write,
            physical_schema.clone(),
        );

        if self.root_released {
            builder.clear_root_data_and_dv_entries();
            // TODO: Process incremental removes from delta log and mark them as DELETED in the
            // appropriate leaf manifests. This can be done by calling `replay_log_commits`,
            // discarding the returned batches, and then applying
            // `processor.deleted_leaf_positions_by_location()` to `builder`.
            return Ok(builder);
        }

        let mut processor =
            ContentRootRebuildProcessor::new(engine, self.snapshot_id, physical_schema)?;
        let log_segment = self.read_snapshot.log_segment();
        for data in replay_log_commits(&mut processor, engine, log_segment, log_start_version)? {
            builder.add_pre_built_log_batch(data)?;
        }
        if let Some(root_path) = root_path.as_deref() {
            for entry in replay_content_root(&mut processor, engine, root_path, &table_root)? {
                builder.add_entry(entry);
            }
            for (leaf_path, bitmap) in processor.deleted_leaf_positions_by_location() {
                if builder.has_leaf_manifest(&leaf_path) {
                    builder.delete_multiple_from_leaf(&leaf_path, &bitmap, true)?;
                }
            }
        }

        Ok(builder)
    }

    /// Applies all accumulated manifest commit state to a [`ContentTreeNodeBuilder`].
    ///
    /// Called during commit to incorporate leaf manifests and deletions into the content tree
    /// after [`create_builder`](Self::create_builder) has populated the baseline state.
    pub(super) fn apply_to_builder(&self, builder: &mut ContentTreeNodeBuilder) -> DeltaResult<()> {
        for entry in &self.leaf_manifests {
            builder.add_entry(entry.clone());
        }
        for file_path in &self.aggregated_unreconciled {
            builder.remove_data_file(file_path.as_str())?;
        }
        for dv_path in &self.aggregated_root_dv_actions {
            builder.remove_dv(dv_path.as_str())?;
        }
        // set_changes_dv=false because this is leaf reorganization, not actual user-facing deletion
        for (manifest_path, entry_indices) in &self.aggregated_manifest_dvs {
            builder.delete_multiple_from_leaf(manifest_path, entry_indices, false)?;
        }
        Ok(())
    }
}
