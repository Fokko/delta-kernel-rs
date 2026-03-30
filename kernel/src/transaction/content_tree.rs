//! Utilities for updating the content tree during manifest commits.

use std::collections::HashMap;
use std::sync::LazyLock;

use roaring::RoaringTreemap;

use crate::engine_data::{GetData, TypedGetData as _};
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
