use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use url::Url;

use crate::content_tree::ContentTreeNodeEntry;
use crate::error::Error;
use crate::snapshot::SnapshotRef;
use crate::{DeltaResult, Engine, Version};

use super::leaf_writer::{LeafNodeWriter, LeafNodeWriterResult};

/// State for a batch (content-tree) commit.
///
/// Obtained by calling [`crate::transaction::Transaction::with_batch_commit`]. Holds all
/// tree-manipulation state and exposes tree-focused methods for partition-aware compaction
/// workflows. `Transaction` retains access to all normal builder and commit methods while
/// this struct exists.
///
/// # Lifetime
///
/// `BatchState` holds a mutable reference into the owning `Transaction` (via
/// `Option<BatchState>` stored inside it). Drop `BatchState` before calling
/// [`crate::transaction::Transaction::commit`] or any other `&mut Transaction` method.
pub struct BatchState {
    // Snapshot info copied from Transaction at construction (SnapshotRef is Arc, clone is cheap).
    pub(super) version_to_write: Version,
    pub(super) snapshot_id: i64,
    pub(super) read_snapshot: SnapshotRef,

    // Batch-specific state, moved out of Transaction.
    pub(super) aggregated_manifest_dvs: HashMap<String, roaring::RoaringTreemap>,
    pub(super) aggregated_unreconciled: HashSet<String>,
    pub(super) aggregated_root_dv_actions: HashSet<String>,
    pub(super) leaf_manifests: Vec<ContentTreeNodeEntry>,
    pub(super) root_released: bool,
    pub(super) cached_root_manifest_url: OnceCell<Option<Url>>,

    /// Running cursor for row tracking first_row_id assignment across leaves.
    /// `Some(next_available_row_id)` when row tracking is enabled, `None` otherwise.
    pub(super) row_id_cursor: Option<i64>,
}

impl BatchState {
    /// Create a new `BatchState` from copied snapshot fields.
    pub(super) fn new(
        version_to_write: Version,
        snapshot_id: i64,
        read_snapshot: SnapshotRef,
    ) -> Self {
        BatchState {
            version_to_write,
            snapshot_id,
            read_snapshot,
            aggregated_manifest_dvs: HashMap::new(),
            aggregated_unreconciled: HashSet::new(),
            aggregated_root_dv_actions: HashSet::new(),
            leaf_manifests: Vec::new(),
            root_released: false,
            cached_root_manifest_url: OnceCell::new(),
            row_id_cursor: None,
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
    /// - The root manifest (if present in the checkpoint) -- entries where `dataManifestPath` is NULL.
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
    /// [`new_leaf_node_writer`]: BatchState::new_leaf_node_writer
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
    pub fn new_leaf_node_writer(&mut self, engine: &dyn Engine) -> DeltaResult<LeafNodeWriter> {
        let root_manifest_url = if let Some(url) = self.cached_root_manifest_url.get() {
            url.clone()
        } else {
            let url = self.root_manifest_url(engine)?;
            let _ = self.cached_root_manifest_url.set(url.clone());
            url
        };

        let track_root_removals = !self.root_released;

        let root_manifest_path = root_manifest_url
            .as_ref()
            .map(|url| {
                crate::content_tree::absolute_to_relative_path(url, self.read_snapshot.table_root())
            })
            .transpose()?;

        let column_mapping_mode = self
            .read_snapshot
            .table_configuration()
            .column_mapping_mode();
        let physical_schema = Arc::new(
            self.read_snapshot
                .schema()
                .as_ref()
                .make_physical(column_mapping_mode),
        );

        // Lazily initialize the row_id_cursor from the snapshot HWM on first use
        let needs_row_tracking = self
            .read_snapshot
            .table_configuration()
            .should_write_row_tracking();
        if needs_row_tracking && self.row_id_cursor.is_none() {
            use crate::row_tracking::RowTrackingDomainMetadata;
            let hwm = RowTrackingDomainMetadata::get_high_water_mark(&self.read_snapshot, engine)?;
            self.row_id_cursor = Some(hwm.unwrap_or(-1) + 1);
        }

        let writer = LeafNodeWriter::new(
            self.read_snapshot.table_root().clone(),
            self.version_to_write,
            self.snapshot_id,
            physical_schema,
            track_root_removals,
            root_manifest_path,
            self.row_id_cursor,
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
    /// * `Ok(None)` - No content root exists yet.
    /// * `Err` - Error constructing the URL.
    pub fn root_manifest_url(&self, _engine: &dyn Engine) -> DeltaResult<Option<Url>> {
        let content_root = self.read_snapshot.content_root();
        let table_root = self.read_snapshot.table_root();
        Ok(content_root.and_then(|cr| table_root.join(&cr.path).ok()))
    }

    /// Incorporate leaf writer results into this batch.
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

        // Advance the row ID cursor with the leaf's next available row ID
        if let Some(next) = leaf_result.next_row_id {
            self.row_id_cursor = Some(next);
        }

        Ok(())
    }

    /// Applies all accumulated batch state to a [`crate::content_tree::builder::ContentTreeNodeBuilder`].
    ///
    /// Called during commit to incorporate leaf manifests and deletions into the content tree
    /// before it is written out.
    pub(super) fn apply_to_builder(
        &self,
        builder: &mut crate::content_tree::builder::ContentTreeNodeBuilder,
    ) -> DeltaResult<()> {
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
