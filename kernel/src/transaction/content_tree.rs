//! Utilities for updating the content tree during manifest commits.

use std::collections::HashMap;
use std::sync::LazyLock;

use roaring::RoaringTreemap;

use crate::engine_data::{EngineData, GetData, TypedGetData as _};
use crate::expressions::{column_name, ColumnName};
use crate::schema::DataType;
use crate::{DeltaResult, Error, RowVisitor};

// Columns needed to process scan metadata for remove actions (manifest commit path only).
// Indices: path=0, dv_path_or_inline=1, data_manifest_path=2, data_manifest_position=3
pub(super) static REMOVE_SCAN_COLUMNS: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
    LazyLock::new(|| {
        (
            vec![
                column_name!("path"),
                column_name!("deletionVector.pathOrInlineDv"),
                column_name!("fileConstantValues.dataManifestPath"),
                column_name!("fileConstantValues.dataManifestPosition"),
            ],
            vec![
                DataType::STRING,
                DataType::STRING,
                DataType::STRING,
                DataType::LONG,
            ],
        )
    });

/// Visits scan row batches and routes each selected row to either a root deletion
/// (via `on_root_deletion`) or accumulates it into `leaf_deletions` for batch processing.
pub(super) struct ScanMetadataRemoveVisitor<'a, F: FnMut(&str, Option<&str>) -> DeltaResult<()>> {
    pub(super) selection_vector: &'a [bool],
    root_manifest_path: Option<&'a str>,
    /// Called for each root entry: (file_path, dv_path_or_inline)
    on_root_deletion: F,
    /// Leaf manifest path → row indices to delete (batched for delete_multiple_from_leaf)
    pub(super) leaf_deletions: HashMap<String, RoaringTreemap>,
}

impl<'a, F: FnMut(&str, Option<&str>) -> DeltaResult<()>> ScanMetadataRemoveVisitor<'a, F> {
    pub(super) fn new(root_manifest_path: Option<&'a str>, on_root_deletion: F) -> Self {
        Self {
            selection_vector: &[],
            root_manifest_path,
            on_root_deletion,
            leaf_deletions: HashMap::new(),
        }
    }
}

impl<'a, F: FnMut(&str, Option<&str>) -> DeltaResult<()>> RowVisitor
    for ScanMetadataRemoveVisitor<'a, F>
{
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        (&REMOVE_SCAN_COLUMNS.0, &REMOVE_SCAN_COLUMNS.1)
    }

    fn visit<'b>(&mut self, row_count: usize, getters: &[&'b dyn GetData<'b>]) -> DeltaResult<()> {
        for i in 0..row_count {
            let is_selected = i >= self.selection_vector.len() || self.selection_vector[i];
            if !is_selected {
                continue;
            }
            let Some(path): Option<String> = getters[0].get_opt(i, "path")? else {
                continue;
            };
            let dv_path: Option<String> = getters[1].get_opt(i, "deletionVector.pathOrInlineDv")?;
            let data_manifest_path: Option<String> =
                getters[2].get_opt(i, "fileConstantValues.dataManifestPath")?;
            let data_manifest_position: Option<i64> =
                getters[3].get_opt(i, "fileConstantValues.dataManifestPosition")?;

            // Invariant: path and position must be present together or absent together.
            if data_manifest_path.is_some() != data_manifest_position.is_some() {
                return Err(Error::missing_data(format!(
                    "data_manifest_path and data_manifest_position must both be present or \
                     absent for entry: {path}"
                )));
            }

            // Determine file location from data_manifest_path:
            //   - data_manifest_path differs from root -> file lives in a leaf manifest
            //   - data_manifest_path equals root -> file is in the root manifest
            //   - data_manifest_path is None -> file predates the content tree (no tracking)
            match (data_manifest_path.as_deref(), data_manifest_position) {
                (Some(mp), Some(pos)) if self.root_manifest_path != Some(mp) => {
                    self.leaf_deletions
                        .entry(mp.to_owned())
                        .or_default()
                        .insert(pos as u64);
                }
                (Some(_), _) => {
                    (self.on_root_deletion)(&path, dv_path.as_deref())?;
                }
                _ => {
                    // data_manifest_path is absent: either this is the first manifest commit
                    // and the file was added then removed in the same transaction, or the
                    // remove cancels a file not yet written to any leaf. Nothing to do.
                }
            }
        }
        Ok(())
    }
}

// === Classify DV-matched rows as root-resident vs leaf-resident ===

static DV_UPDATE_DECISION_COLUMNS: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
    LazyLock::new(|| {
        (
            vec![
                column_name!("path"),
                column_name!("fileConstantValues.dataManifestPath"),
                column_name!("fileConstantValues.dataManifestPosition"),
            ],
            vec![DataType::STRING, DataType::STRING, DataType::LONG],
        )
    });

/// Per-batch classification of DV-matched scan rows into root-resident and leaf-resident.
///
/// The DV-update flow handles root-resident files by mutating their entry's `deletion_vector`
/// in place (preserving `sequence_number`); leaf-resident files are removed from their leaf
/// manifest's DV bitmap and re-added to the root by the existing delete + re-add machinery.
pub(super) struct DvUpdateDecisions {
    /// File paths whose root-resident entry should have its DV mutated in place.
    pub(super) root_paths: Vec<String>,
    /// Leaf manifest path -> row indices to clear from that leaf's `manifest_dv` bitmap.
    pub(super) leaf_deletions: HashMap<String, RoaringTreemap>,
    /// Selection vector for the rows that should be re-added to the root in Phase 2. Same
    /// length as the source batch; rows handled in place have their bit cleared.
    pub(super) leaf_only_selection: Vec<bool>,
}

/// Walks the path + data_manifest_path/position columns of a DV-matched scan batch and
/// returns the per-batch classification used by the DV-update flow.
pub(super) fn collect_dv_update_decisions(
    data: &dyn EngineData,
    selection_vector: &[bool],
    root_manifest_path: Option<&str>,
) -> DeltaResult<DvUpdateDecisions> {
    let mut visitor = DvUpdateDecisionVisitor {
        selection_vector,
        root_manifest_path,
        decisions: DvUpdateDecisions {
            root_paths: Vec::new(),
            leaf_deletions: HashMap::new(),
            leaf_only_selection: Vec::new(),
        },
    };
    visitor.visit_rows_of(data)?;
    Ok(visitor.decisions)
}

struct DvUpdateDecisionVisitor<'a> {
    selection_vector: &'a [bool],
    root_manifest_path: Option<&'a str>,
    decisions: DvUpdateDecisions,
}

impl RowVisitor for DvUpdateDecisionVisitor<'_> {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        (&DV_UPDATE_DECISION_COLUMNS.0, &DV_UPDATE_DECISION_COLUMNS.1)
    }

    fn visit<'b>(&mut self, row_count: usize, getters: &[&'b dyn GetData<'b>]) -> DeltaResult<()> {
        // Start from the source selection vector; clear bits for rows handled in place.
        self.decisions.leaf_only_selection = (0..row_count)
            .map(|i| i < self.selection_vector.len() && self.selection_vector[i])
            .collect();

        for i in 0..row_count {
            if !self.decisions.leaf_only_selection[i] {
                continue;
            }
            let Some(path): Option<String> = getters[0].get_opt(i, "path")? else {
                self.decisions.leaf_only_selection[i] = false;
                continue;
            };
            let data_manifest_path: Option<String> =
                getters[1].get_opt(i, "fileConstantValues.dataManifestPath")?;
            let data_manifest_position: Option<i64> =
                getters[2].get_opt(i, "fileConstantValues.dataManifestPosition")?;

            // Invariant: path and position must be present together or absent together.
            if data_manifest_path.is_some() != data_manifest_position.is_some() {
                return Err(Error::missing_data(format!(
                    "data_manifest_path and data_manifest_position must both be present or \
                     absent for entry: {path}"
                )));
            }

            match (data_manifest_path.as_deref(), data_manifest_position) {
                (Some(mp), Some(pos)) if self.root_manifest_path != Some(mp) => {
                    // Leaf-resident: keep bit set; caller deletes from leaf + re-adds to root.
                    self.decisions
                        .leaf_deletions
                        .entry(mp.to_owned())
                        .or_default()
                        .insert(pos as u64);
                }
                (Some(_), _) => {
                    // Root-resident: caller mutates DV in place. Drop from leaf selection.
                    self.decisions.root_paths.push(path);
                    self.decisions.leaf_only_selection[i] = false;
                }
                _ => {
                    // Predates the content tree -- leave bit set; caller handles via re-add.
                }
            }
        }
        Ok(())
    }
}
