use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
use crate::schema::{ColumnName, ColumnNamesAndTypes, DataType};
use crate::{DeltaResult, Error};
use bytes::Bytes;
use std::str::FromStr;
use std::sync::LazyLock;

use super::{
    ContentTreeNodeEntry, DataContentType, DataFileFormat, DvInfo, ManifestStats, TrackingInfo,
    TrackingStatus,
};

/// Visitor that extracts ContentTreeNodeEntry structs from EngineData
#[derive(Default)]
pub(super) struct ContentTreeNodeEntryVisitor {
    pub(super) entries: Vec<ContentTreeNodeEntry>,
}

impl RowVisitor for ContentTreeNodeEntryVisitor {
    fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
        static NAMES_AND_TYPES: LazyLock<ColumnNamesAndTypes> = LazyLock::new(|| {
            use crate::schema::ToSchema as _;
            let all_leaves = ContentTreeNodeEntry::to_schema().leaves(None::<&str>);
            let (names, types) = all_leaves.as_ref();
            // Filter out array types (splitOffsets, equalityIds) that GetData doesn't support
            let (filtered_names, filtered_types): (Vec<_>, Vec<_>) = names
                .iter()
                .zip(types.iter())
                .filter(|(_, dt)| !matches!(dt, DataType::Array(_)))
                .map(|(n, t)| (n.clone(), t.clone()))
                .unzip();
            (filtered_names, filtered_types).into()
        });
        NAMES_AND_TYPES.as_ref()
    }

    fn visit<'a>(&mut self, row_count: usize, getters: &[&'a dyn GetData<'a>]) -> DeltaResult<()> {
        // The number of getters should match the number of leaf fields in ContentTreeNodeEntry schema
        // We'll validate this implicitly by accessing each field

        for i in 0..row_count {
            let entry = visit_metadata_entry_at(i, getters)?;
            self.entries.push(entry);
        }
        Ok(())
    }
}

fn visit_metadata_entry_at<'a>(
    row_index: usize,
    getters: &[&'a dyn GetData<'a>],
) -> DeltaResult<ContentTreeNodeEntry> {
    // The getters are in order of flattened leaf fields (26 total, excluding array types):
    // 0: content_type
    // 1: location
    // 2: file_format
    // 3-8: tracking fields (status, snapshot_id, sequence_number, file_sequence_number, first_row_id, changes_dv)
    // 9-12: dv_info fields (location, offset, size_in_bytes, cardinality)
    // 13: partition_spec_id
    // 14: sort_order_id
    // 15: record_count
    // 16: file_size_in_bytes
    // (content_stats excluded from schema)
    // 17-23: manifest_stats fields (7 fields)
    // 24: key_metadata
    // (split_offsets excluded - array type not supported by GetData)
    // (equality_ids excluded - array type not supported by GetData)
    // 25: manifest_dv

    // Extract content_type
    let content_type_int: i32 = getters[0].get(row_index, "content_type")?;
    let content_type = match content_type_int {
        0 => DataContentType::Data,
        1 => DataContentType::PositionDeletes,
        2 => DataContentType::EqualityDeletes,
        3 => DataContentType::DataManifest,
        4 => DataContentType::DeleteManifest,
        5 => DataContentType::CombinedManifest,
        _ => {
            return Err(Error::generic(format!(
                "Invalid content_type value: {}",
                content_type_int
            )))
        }
    };

    // Extract location
    let location: Option<String> = getters[1].get_opt(row_index, "location")?;

    // Extract file_format
    let file_format_str: String = getters[2].get(row_index, "file_format")?;
    let file_format = DataFileFormat::from_str(&file_format_str)?;

    // Extract tracking fields
    let tracking_status_int: i32 = getters[3].get(row_index, "tracking.status")?;
    let tracking_status = match tracking_status_int {
        0 => TrackingStatus::Existed,
        1 => TrackingStatus::Added,
        2 => TrackingStatus::Deleted,
        _ => {
            return Err(Error::generic(format!(
                "Invalid tracking status value: {}",
                tracking_status_int
            )))
        }
    };

    let tracking_snapshot_id: Option<i64> =
        getters[4].get_opt(row_index, "tracking.snapshot_id")?;
    let tracking_sequence_number: Option<i64> =
        getters[5].get_opt(row_index, "tracking.sequence_number")?;
    let tracking_file_sequence_number: Option<i64> =
        getters[6].get_opt(row_index, "tracking.file_sequence_number")?;
    let tracking_first_row_id: Option<i64> =
        getters[7].get_opt(row_index, "tracking.first_row_id")?;
    let tracking_changes_dv: Option<&[u8]> =
        getters[8].get_opt(row_index, "tracking.changes_dv")?;
    let tracking_changes_dv_bytes = tracking_changes_dv.map(Bytes::copy_from_slice);

    let tracking = TrackingInfo {
        status: tracking_status,
        snapshot_id: tracking_snapshot_id,
        sequence_number: tracking_sequence_number,
        file_sequence_number: tracking_file_sequence_number,
        first_row_id: tracking_first_row_id,
        changes_dv: tracking_changes_dv_bytes,
    };

    // Extract dv_info fields (location, offset, size_in_bytes, cardinality)
    let dv_location: Option<String> = getters[9].get_opt(row_index, "dv_info.location")?;
    let dv_info = dv_location
        .map(|location| -> DeltaResult<DvInfo> {
            let offset: i64 = getters[10].get(row_index, "dv_info.offset")?;
            let size_in_bytes: i64 = getters[11].get(row_index, "dv_info.size_in_bytes")?;
            let cardinality: i64 = getters[12].get(row_index, "dv_info.cardinality")?;
            Ok(DvInfo {
                location,
                offset,
                size_in_bytes,
                cardinality,
            })
        })
        .transpose()?;

    // Extract scalar fields
    let partition_spec_id: i64 = getters[13].get(row_index, "partition_spec_id")?;
    let sort_order_id: Option<i64> = getters[14].get_opt(row_index, "sort_order_id")?;
    let record_count: i64 = getters[15].get(row_index, "record_count")?;
    let file_size_in_bytes: Option<i64> = getters[16].get_opt(row_index, "file_size_in_bytes")?;

    // content_stats has no fields, so no getters

    // Extract manifest_stats fields
    let ms_added_files_count: Option<i64> =
        getters[17].get_opt(row_index, "manifest_stats.added_files_count")?;
    let ms_existing_files_count: Option<i64> =
        getters[18].get_opt(row_index, "manifest_stats.existing_files_count")?;
    let ms_deletes_files_count: Option<i64> =
        getters[19].get_opt(row_index, "manifest_stats.deletes_files_count")?;
    let ms_added_rows_count: Option<i64> =
        getters[20].get_opt(row_index, "manifest_stats.added_rows_count")?;
    let ms_existing_rows_count: Option<i64> =
        getters[21].get_opt(row_index, "manifest_stats.existing_rows_count")?;
    let ms_delete_rows_count: Option<i64> =
        getters[22].get_opt(row_index, "manifest_stats.delete_rows_count")?;
    let ms_min_sequence_number: Option<i64> =
        getters[23].get_opt(row_index, "manifest_stats.min_sequence_number")?;

    let manifest_stats = ms_added_files_count.map(|added_files_count| ManifestStats {
        added_files_count,
        existing_files_count: ms_existing_files_count.unwrap_or(0),
        deletes_files_count: ms_deletes_files_count.unwrap_or(0),
        added_rows_count: ms_added_rows_count.unwrap_or(0),
        existing_rows_count: ms_existing_rows_count.unwrap_or(0),
        delete_rows_count: ms_delete_rows_count.unwrap_or(0),
        min_sequence_number: ms_min_sequence_number.unwrap_or(0),
    });

    // Extract key_metadata
    let key_metadata: Option<&[u8]> = getters[24].get_opt(row_index, "key_metadata")?;
    let key_metadata_bytes = key_metadata.map(Bytes::copy_from_slice);

    // Note: split_offsets and equality_ids are array types not supported by GetData

    // Extract manifest_dv
    let manifest_dv: Option<&[u8]> = getters[25].get_opt(row_index, "manifest_dv")?;
    let manifest_dv_bytes = manifest_dv.map(Bytes::copy_from_slice);

    Ok(ContentTreeNodeEntry {
        content_type,
        location,
        file_format,
        tracking,
        dv_info,
        partition_spec_id,
        sort_order_id,
        record_count,
        file_size_in_bytes,
        content_stats: None, // Requires table schema to read - not included in base schema
        manifest_stats,
        key_metadata: key_metadata_bytes,
        split_offsets: None, // Array type not supported by GetData
        equality_ids: None,  // Array type not supported by GetData
        manifest_dv: manifest_dv_bytes,
    })
}
