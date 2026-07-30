//! Shared helpers for AMT (content tree) write-path tests.
//!
//! Included via `#[path]` into several test binaries, each using only a subset of these helpers.
//! The `#![allow(dead_code)]` on such shared test-helper modules is the standard Rust convention
//! for silencing the dead-code warnings the unused subset triggers per binary.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::engine_data::{
    FilteredEngineData, FilteredRowVisitor, GetData, RowIndexIterator, RowVisitor,
    TypedGetData as _,
};
use delta_kernel::expressions::ColumnName;
use delta_kernel::scan::Scan;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::transaction::Transaction;
use delta_kernel::{
    DataContentType, DeltaResult, Engine, EngineData, FileMeta, Snapshot, TrackingStatus,
};
use roaring::RoaringTreemap;
use test_utils::{create_add_files_metadata, create_table, engine_store_setup};
use url::Url;
use uuid::Uuid;

/// A file to add, with the minimal fields tests need.
#[derive(Clone, Copy)]
pub struct DataFile<'a> {
    pub location: &'a str,
    pub size: i64,
    pub mod_time: i64,
    pub num_records: Option<i64>,
    // ToDo: Stats
}

impl DataFile<'_> {
    /// Converts `files` into the `EngineData` shape `Transaction::add_files` expects.
    pub fn to_engine_data(
        files: &[DataFile],
        schema: &SchemaRef,
    ) -> Result<Box<dyn EngineData>, Box<dyn std::error::Error>> {
        let tuples: Vec<_> = files
            .iter()
            .map(|f| (f.location, f.size, f.mod_time, f.num_records))
            .collect();
        create_add_files_metadata(schema, tuples)
    }
}

/// Adds `files` to the root.
pub fn add_files(
    txn: &mut Transaction,
    schema: &SchemaRef,
    files: &[DataFile],
) -> Result<(), Box<dyn std::error::Error>> {
    let metadata: Box<dyn EngineData> = DataFile::to_engine_data(files, schema)?;
    txn.add_files(metadata);
    Ok(())
}

/// Adds `files` to a new leaf manifest via a manifest commit.
pub fn add_leaf(
    txn: &mut Transaction,
    engine: &dyn Engine,
    schema: &SchemaRef,
    files: &[DataFile],
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_commit_state = txn.with_manifest_commit()?;
    let mut leaf: delta_kernel::transaction::LeafNodeWriter =
        manifest_commit_state.new_leaf_node_writer(engine)?;
    let metadata = DataFile::to_engine_data(files, schema)?;
    leaf.add_files(engine, metadata)?;
    manifest_commit_state.add_leaf(leaf.finish(engine)?)?;
    Ok(())
}

/// Applies `new_dvs` (keyed by path) to the matching rows in `scan`.
pub fn update_dvs_by_path(
    txn: &mut Transaction,
    scan: Scan,
    engine: &dyn Engine,
    new_dvs: HashMap<String, DeletionVectorDescriptor>,
) -> Result<(), Box<dyn std::error::Error>> {
    let current_files: Vec<_> = scan
        .scan_metadata(engine)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|sm| sm.scan_files)
        .collect();
    txn.update_deletion_vectors(new_dvs, current_files.into_iter().map(Ok))?;
    Ok(())
}

/// Removes every row in `scan` whose path is in `paths`, returning the count removed.
pub fn remove_files_by_path(
    txn: &mut Transaction,
    scan: Scan,
    engine: &dyn Engine,
    paths: &[&str],
) -> Result<usize, Box<dyn std::error::Error>> {
    let targets: HashSet<String> = paths.iter().map(|s| s.to_string()).collect();
    let mut total_removed = 0;
    for entry in scan.scan_metadata(engine)? {
        let scan_metadata = entry?;
        let selection_vector = select_matching_paths(&scan_metadata.scan_files, &targets)?;
        if selection_vector.iter().any(|&selected| selected) {
            total_removed += selection_vector
                .iter()
                .filter(|&&selected| selected)
                .count();
            let (data, _) = scan_metadata.scan_files.into_parts();
            txn.remove_files(FilteredEngineData::try_new(data, selection_vector)?);
        }
    }
    Ok(total_removed)
}

/// Returns a selection vector, no larger than `scan_files`'s own, that's true only for rows
/// that were already selected *and* whose `path` column is in `targets`.
fn select_matching_paths(
    scan_files: &FilteredEngineData,
    targets: &HashSet<String>,
) -> DeltaResult<Vec<bool>> {
    struct PathMatchVisitor<'a> {
        targets: &'a HashSet<String>,
        selection_vector: Vec<bool>,
    }

    impl FilteredRowVisitor for PathMatchVisitor<'_> {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| (vec![ColumnName::new(["path"])], vec![DataType::STRING]));
            (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
        }

        fn visit_filtered<'a>(
            &mut self,
            getters: &[&'a dyn GetData<'a>],
            rows: RowIndexIterator<'_>,
        ) -> DeltaResult<()> {
            self.selection_vector = vec![false; rows.num_rows()];
            for row_index in rows {
                let path: String = getters[0].get(row_index, "path")?;
                self.selection_vector[row_index] = self.targets.contains(&path);
            }
            Ok(())
        }
    }

    let mut visitor = PathMatchVisitor {
        targets,
        selection_vector: Vec::new(),
    };
    visitor.visit_rows_of(scan_files)?;
    Ok(visitor.selection_vector)
}

/// Returns the `path` of the first `DataManifest` leaf-reference row in `snapshot`'s
/// root manifest.
pub fn leaf_path(
    snapshot: &Snapshot,
    engine: &dyn Engine,
) -> Result<String, Box<dyn std::error::Error>> {
    let root_entries = collect_root_entries(snapshot, engine)?;
    Ok(root_entries
        .iter()
        .find(|e| e.content_type == DataContentType::DataManifest)
        .expect("a leaf-reference entry")
        .path
        .clone())
}

/// Creates a table with column mapping, AMT, and deletion vectors enabled.
pub async fn setup_amt_test_tables(
    schema: SchemaRef,
    table_base_name: &str,
) -> Result<(Url, DefaultEngine<TokioBackgroundExecutor>), Box<dyn std::error::Error>> {
    let (store, engine, table_location) = engine_store_setup(table_base_name, None);

    Ok((
        create_table(
            store,
            table_location,
            schema.clone(),
            &[],
            true,
            vec![
                "columnMapping",
                "metadataTree-experimental",
                "deletionVectors",
            ],
            vec![
                "columnMapping",
                "metadataTree-experimental",
                "deletionVectors",
                "domainMetadata",
                "rowTracking",
            ],
        )
        .await?,
        engine,
    ))
}

/// Single integer column `id`.
pub fn single_id_column_schema() -> Result<Arc<StructType>, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )
    .with_metadata([
        (
            ColumnMetadataKey::ColumnMappingId.as_ref(),
            MetadataValue::Number(1),
        ),
        (
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            MetadataValue::String("col-1".to_string()),
        ),
    ])])?))
}

/// Integer `id` column and string `value` column.
pub fn id_and_value_schema() -> Result<Arc<StructType>, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER).with_metadata([
            (
                ColumnMetadataKey::ColumnMappingId.as_ref(),
                MetadataValue::Number(1),
            ),
            (
                ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                MetadataValue::String("col-1".to_string()),
            ),
        ]),
        StructField::nullable("value", DataType::STRING).with_metadata([
            (
                ColumnMetadataKey::ColumnMappingId.as_ref(),
                MetadataValue::Number(2),
            ),
            (
                ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                MetadataValue::String("col-2".to_string()),
            ),
        ]),
    ])?))
}

/// Creates a table with column mapping id mode.
pub async fn setup_test_tables_with_column_mapping(
    schema: SchemaRef,
    partition_columns: &[&str],
    table_base_name: &str,
) -> Result<(Url, DefaultEngine<TokioBackgroundExecutor>), Box<dyn std::error::Error>> {
    let (store, engine, table_location) = engine_store_setup(table_base_name, None);

    Ok((
        create_table(
            store,
            table_location,
            schema.clone(),
            partition_columns,
            true,
            vec!["columnMapping", "metadataTree-experimental"],
            vec!["columnMapping", "metadataTree-experimental"],
        )
        .await?,
        engine,
    ))
}

/// Information collected from scanning a snapshot.
#[derive(Debug)]
pub struct ScannedFiles {
    /// Unique file paths found in scan
    pub file_paths: HashSet<String>,
    /// File paths with deletion vectors
    pub files_with_dvs: HashSet<String>,
}

/// Helper to collect all files from a snapshot via scan.
pub fn collect_scanned_files(
    snapshot: Arc<Snapshot>,
    engine: &dyn Engine,
) -> DeltaResult<ScannedFiles> {
    struct FileCollector {
        file_paths: HashSet<String>,
        files_with_dvs: HashSet<String>,
    }

    impl FilteredRowVisitor for FileCollector {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["path"]),
                            ColumnName::new(["deletionVector", "storageType"]),
                        ],
                        vec![DataType::STRING, DataType::STRING],
                    )
                });
            (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
        }

        fn visit_filtered<'a>(
            &mut self,
            getters: &[&'a dyn GetData<'a>],
            rows: RowIndexIterator<'_>,
        ) -> DeltaResult<()> {
            for row_index in rows {
                let path: String = getters[0].get(row_index, "path")?;

                // Check if this is a duplicate - insert returns false if already present
                if !self.file_paths.insert(path.clone()) {
                    return Err(delta_kernel::Error::generic(format!(
                        "Duplicate file path '{}' found in scan. Each file should appear exactly once.",
                        path
                    )));
                }

                // Check if this file has a deletion vector
                let dv_storage_type: Option<String> =
                    getters[1].get_opt(row_index, "deletionVector.storageType")?;
                if dv_storage_type.is_some() {
                    self.files_with_dvs.insert(path);
                }
            }
            Ok(())
        }
    }

    let scan = snapshot.scan_builder().build()?;
    let mut all_file_paths = HashSet::new();
    let mut all_files_with_dvs = HashSet::new();

    for scan_metadata_result in scan.scan_metadata(engine)? {
        let scan_metadata = scan_metadata_result?;
        let mut collector = FileCollector {
            file_paths: HashSet::new(),
            files_with_dvs: HashSet::new(),
        };
        collector.visit_rows_of(&scan_metadata.scan_files)?;

        // Merge results
        for path in collector.file_paths {
            if !all_file_paths.insert(path.clone()) {
                return Err(delta_kernel::Error::generic(format!(
                    "Duplicate file path '{}' found across scan batches.",
                    path
                )));
            }
        }
        all_files_with_dvs.extend(collector.files_with_dvs);
    }

    Ok(ScannedFiles {
        file_paths: all_file_paths,
        files_with_dvs: all_files_with_dvs,
    })
}

/// Helper to verify expected files are present with no duplicates.
pub fn verify_scanned_files(
    scanned: &ScannedFiles,
    expected_files: &[&str],
    expected_files_with_dvs: &[&str],
) {
    let expected: HashSet<String> = expected_files.iter().map(|s| s.to_string()).collect();
    let missing: Vec<_> = expected.difference(&scanned.file_paths).collect();
    let unexpected: Vec<_> = scanned.file_paths.difference(&expected).collect();
    assert!(
        missing.is_empty() && unexpected.is_empty(),
        "Scanned files mismatch. Missing: {missing:?}, unexpected: {unexpected:?}"
    );

    let expected_dvs: HashSet<String> = expected_files_with_dvs
        .iter()
        .map(|s| s.to_string())
        .collect();
    let missing_dvs: Vec<_> = expected_dvs.difference(&scanned.files_with_dvs).collect();
    let unexpected_dvs: Vec<_> = scanned.files_with_dvs.difference(&expected_dvs).collect();
    assert!(
        missing_dvs.is_empty() && unexpected_dvs.is_empty(),
        "DVs mismatch. Missing: {missing_dvs:?}, unexpected: {unexpected_dvs:?}"
    );
}

/// Builds a `PersistedRelative` deletion vector descriptor for `uuid`, plus its persisted
/// location `deletion_vector_{uuid}.bin`.
pub fn dv_descriptor(
    uuid: Uuid,
    offset: i32,
    size_in_bytes: i32,
    cardinality: i64,
) -> (DeletionVectorDescriptor, String) {
    (
        DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedRelative,
            path_or_inline_dv: z85::encode(uuid.as_bytes()),
            offset: Some(offset),
            size_in_bytes,
            cardinality,
        },
        format!("deletion_vector_{uuid}.bin"),
    )
}

/// A data file's deletion vector, as persisted on its manifest entry.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EntryDv {
    pub location: String,
    pub cardinality: i64,
}

/// Persisted state of a single manifest entry: a root-resident data file, a leaf-reference
/// row, or a file living inside a leaf. `ContentTreeNodeEntry` isn't exposed outside the
/// crate; this is a test-only structure, explicit about which fields we compare against.
#[derive(Debug, Clone)]
pub struct Entry {
    pub path: String,
    pub content_type: DataContentType,
    pub status: TrackingStatus,
    pub sequence_number: Option<i64>,
    pub dv_snapshot_id: Option<i64>,
    pub deletion_vector: Option<EntryDv>,
    /// Cardinality of positions marked dead within a referenced leaf (`manifestInfo.dv`).
    pub manifest_dv_cardinality: Option<i64>,
    deleted_positions: Option<BTreeSet<u64>>,
    replaced_positions: Option<BTreeSet<u64>>,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.content_type == other.content_type
            && self.status == other.status
            && self.sequence_number == other.sequence_number
            && self.dv_snapshot_id == other.dv_snapshot_id
            && self.deletion_vector == other.deletion_vector
            && self.manifest_dv_cardinality == other.manifest_dv_cardinality
    }
}
impl Eq for Entry {}

impl Entry {
    /// A plain data file entry.
    pub fn new(path: impl Into<String>, status: TrackingStatus) -> Self {
        Self {
            path: path.into(),
            content_type: DataContentType::Data,
            status,
            sequence_number: None,
            dv_snapshot_id: None,
            deletion_vector: None,
            manifest_dv_cardinality: None,
            deleted_positions: None,
            replaced_positions: None,
        }
    }

    /// A leaf-reference (`DataManifest`) entry -- `path` is the leaf's own location.
    pub fn leaf_ref(path: impl Into<String>, status: TrackingStatus) -> Self {
        Self {
            content_type: DataContentType::DataManifest,
            ..Self::new(path, status)
        }
    }

    /// Sets the expected `sequence_number`.
    pub fn sequence_number(mut self, n: i64) -> Self {
        self.sequence_number = Some(n);
        self
    }

    /// Sets the expected `dv_snapshot_id`.
    pub fn dv_snapshot_id(mut self, id: i64) -> Self {
        self.dv_snapshot_id = Some(id);
        self
    }

    /// Sets the expected deletion vector location and cardinality.
    pub fn deletion_vector(mut self, location: impl Into<String>, cardinality: i64) -> Self {
        self.deletion_vector = Some(EntryDv {
            location: location.into(),
            cardinality,
        });
        self
    }

    /// Sets the expected `manifest_dv_cardinality`.
    pub fn manifest_dv_cardinality(mut self, n: i64) -> Self {
        self.manifest_dv_cardinality = Some(n);
        self
    }

    /// Whether `deletedPositions` exactly equals `positions` (`None` counts as empty).
    pub fn has_deleted_positions(&self, positions: impl IntoIterator<Item = u64>) -> bool {
        self.deleted_positions.clone().unwrap_or_default() == positions.into_iter().collect()
    }

    /// Whether `replacedPositions` exactly equals `positions` (`None` counts as empty).
    pub fn has_replaced_positions(&self, positions: impl IntoIterator<Item = u64>) -> bool {
        self.replaced_positions.clone().unwrap_or_default() == positions.into_iter().collect()
    }
}

/// Reads every entry currently persisted in `snapshot`'s root manifest -- root-resident
/// data files AND leaf-reference rows, live and dead. Use this to inspect root state
/// directly (e.g. to pull a leaf's `path` off its `DataManifest` entry before calling
/// [`assert_leaf_entries`]); use [`assert_root_entries`] if you just want to compare
/// against expected values.
pub fn collect_root_entries(snapshot: &Snapshot, engine: &dyn Engine) -> DeltaResult<Vec<Entry>> {
    let Some(checkpoint_action) = snapshot.checkpoint_action() else {
        return Ok(Vec::new());
    };
    let root_url = snapshot
        .table_root()
        .join(checkpoint_action.path())
        .map_err(|e| delta_kernel::Error::generic(format!("bad content root URL: {e}")))?;
    collect_manifest_entries(&root_url, engine)
}

/// Reads every entry currently persisted in the leaf manifest at `location` (relative to
/// `table_root`), live and dead.
pub fn collect_leaf_entries(
    table_root: &Url,
    location: &str,
    engine: &dyn Engine,
) -> DeltaResult<Vec<Entry>> {
    let leaf_url = table_root
        .join(location)
        .map_err(|e| delta_kernel::Error::generic(format!("bad leaf URL: {e}")))?;
    collect_manifest_entries(&leaf_url, engine)
}

/// Asserts `snapshot`'s root manifest contains exactly `expected` -- every entry actually
/// persisted (root-resident files AND leaf-reference rows, live and dead) must match one
/// of `expected` and vice versa. Order-insensitive, but exhaustive: nothing persisted is
/// allowed to go unverified.
pub fn assert_root_entries(
    snapshot: &Snapshot,
    engine: &dyn Engine,
    expected: &[Entry],
) -> DeltaResult<()> {
    assert_entries(&collect_root_entries(snapshot, engine)?, expected, "root");
    Ok(())
}

/// Asserts the leaf manifest at `location` (relative to `table_root`) contains exactly
/// `expected`, live and dead. Same exhaustive semantics as [`assert_root_entries`].
pub fn assert_leaf_entries(
    table_root: &Url,
    location: &str,
    engine: &dyn Engine,
    expected: &[Entry],
) -> DeltaResult<()> {
    assert_entries(
        &collect_leaf_entries(table_root, location, engine)?,
        expected,
        location,
    );
    Ok(())
}

/// Asserts `actual` contains exactly `expected` -- every entry in one must match one in the
/// other. Order-insensitive, but exhaustive. Use this directly (instead of
/// [`assert_root_entries`]/[`assert_leaf_entries`]) against a caller-filtered subset of
/// [`collect_root_entries`]/[`collect_leaf_entries`] when some entries (e.g. a freshly
/// created leaf reference) don't have a predictable `path` to put in `expected` -- filter
/// `actual` down to what you *can* fully describe, and check the rest separately.
pub fn assert_entries(actual: &[Entry], expected: &[Entry], context: &str) {
    let mut remaining_expected: Vec<&Entry> = expected.iter().collect();
    let mut unexpected = Vec::new();
    for entry in actual {
        if let Some(pos) = remaining_expected.iter().position(|e| *e == entry) {
            remaining_expected.remove(pos);
        } else {
            unexpected.push(entry);
        }
    }
    if !remaining_expected.is_empty() || !unexpected.is_empty() {
        panic!(
            "{context}: entries mismatch\nMissing (expected but not found): {:#?}\n\
             Unexpected (found but not expected): {:#?}",
            remaining_expected, unexpected
        );
    }
}

/// The on-disk integer representation of `TrackingStatus`, mirrored here since the
/// crate's own conversion is private. Keep in sync with `kernel/src/content_tree/mod.rs`.
fn tracking_status_from_repr(value: i32) -> TrackingStatus {
    match value {
        0 => TrackingStatus::Existing,
        1 => TrackingStatus::Added,
        2 => TrackingStatus::Deleted,
        3 => TrackingStatus::Replaced,
        4 => TrackingStatus::Modified,
        other => panic!("unknown tracking status: {other}"),
    }
}

/// The on-disk integer representation of `DataContentType`, mirrored here since the
/// crate's own conversion is private. Keep in sync with `kernel/src/content_tree/mod.rs`.
fn data_content_type_from_repr(value: i32) -> DataContentType {
    match value {
        0 => DataContentType::Data,
        1 => DataContentType::PositionDeletes,
        2 => DataContentType::EqualityDeletes,
        3 => DataContentType::DataManifest,
        4 => DataContentType::DeleteManifest,
        other => panic!("unknown content type: {other}"),
    }
}

/// Decodes a position bitmap (4-byte magic number + `RoaringTreemap`), mirrored here since
/// the crate's own decoder is private. Keep in sync with `kernel/src/content_tree/builder.rs`.
fn decode_position_bitmap(bytes: &[u8]) -> BTreeSet<u64> {
    RoaringTreemap::deserialize_from(&bytes[4..])
        .expect("valid roaring bitmap")
        .iter()
        .collect()
}

/// Reads every row of a manifest parquet file (root or leaf), including dead
/// (`Deleted`/`Replaced`) rows. Rows without a location are skipped.
fn collect_manifest_entries(location: &Url, engine: &dyn Engine) -> DeltaResult<Vec<Entry>> {
    let schema = Arc::new(
        StructType::try_new([
            StructField::nullable("location", DataType::STRING),
            StructField::nullable("contentType", DataType::INTEGER),
            StructField::nullable(
                "tracking",
                DataType::Struct(Box::new(
                    StructType::try_new([
                        StructField::nullable("status", DataType::INTEGER),
                        StructField::nullable("sequenceNumber", DataType::LONG),
                        StructField::nullable("dvSnapshotId", DataType::LONG),
                        StructField::nullable("deletedPositions", DataType::BINARY),
                        StructField::nullable("replacedPositions", DataType::BINARY),
                    ])
                    .unwrap(),
                )),
            ),
            StructField::nullable(
                "deletionVector",
                DataType::Struct(Box::new(
                    StructType::try_new([
                        StructField::nullable("location", DataType::STRING),
                        StructField::nullable("cardinality", DataType::LONG),
                    ])
                    .unwrap(),
                )),
            ),
            StructField::nullable(
                "manifestInfo",
                DataType::Struct(Box::new(
                    StructType::try_new([StructField::nullable("dvCardinality", DataType::LONG)])
                        .unwrap(),
                )),
            ),
        ])
        .unwrap(),
    );

    let file_meta = FileMeta {
        location: location.clone(),
        last_modified: 0,
        size: 0,
    };
    let batches: Vec<_> = engine
        .parquet_handler()
        .read_parquet_files(&[file_meta], schema, None)?
        .collect::<DeltaResult<Vec<_>>>()?;

    struct EntryCollector {
        entries: Vec<Entry>,
    }

    impl RowVisitor for EntryCollector {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["location"]),
                            ColumnName::new(["contentType"]),
                            ColumnName::new(["tracking", "status"]),
                            ColumnName::new(["tracking", "sequenceNumber"]),
                            ColumnName::new(["tracking", "dvSnapshotId"]),
                            ColumnName::new(["tracking", "deletedPositions"]),
                            ColumnName::new(["tracking", "replacedPositions"]),
                            ColumnName::new(["deletionVector", "location"]),
                            ColumnName::new(["deletionVector", "cardinality"]),
                            ColumnName::new(["manifestInfo", "dvCardinality"]),
                        ],
                        vec![
                            DataType::STRING,
                            DataType::INTEGER,
                            DataType::INTEGER,
                            DataType::LONG,
                            DataType::LONG,
                            DataType::BINARY,
                            DataType::BINARY,
                            DataType::STRING,
                            DataType::LONG,
                            DataType::LONG,
                        ],
                    )
                });
            (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
        }

        fn visit<'b>(
            &mut self,
            row_count: usize,
            getters: &[&'b dyn GetData<'b>],
        ) -> DeltaResult<()> {
            for i in 0..row_count {
                if let Some(path) = getters[0].get_opt(i, "location")? {
                    let content_type_repr: i32 = getters[1]
                        .get_opt(i, "contentType")?
                        .unwrap_or(DataContentType::Data as i32);
                    let status_repr: i32 = getters[2]
                        .get_opt(i, "tracking.status")?
                        .unwrap_or(TrackingStatus::Existing as i32);
                    let sequence_number: Option<i64> =
                        getters[3].get_opt(i, "tracking.sequenceNumber")?;
                    let dv_snapshot_id: Option<i64> =
                        getters[4].get_opt(i, "tracking.dvSnapshotId")?;
                    let deleted_positions = getters[5]
                        .get_binary(i, "tracking.deletedPositions")?
                        .map(decode_position_bitmap);
                    let replaced_positions = getters[6]
                        .get_binary(i, "tracking.replacedPositions")?
                        .map(decode_position_bitmap);
                    let dv_location: Option<String> =
                        getters[7].get_opt(i, "deletionVector.location")?;
                    let dv_cardinality: Option<i64> =
                        getters[8].get_opt(i, "deletionVector.cardinality")?;
                    let manifest_dv_cardinality: Option<i64> =
                        getters[9].get_opt(i, "manifestInfo.dvCardinality")?;
                    self.entries.push(Entry {
                        path,
                        content_type: data_content_type_from_repr(content_type_repr),
                        status: tracking_status_from_repr(status_repr),
                        sequence_number,
                        dv_snapshot_id,
                        deletion_vector: dv_location.zip(dv_cardinality).map(
                            |(location, cardinality)| EntryDv {
                                location,
                                cardinality,
                            },
                        ),
                        manifest_dv_cardinality,
                        deleted_positions,
                        replaced_positions,
                    });
                }
            }
            Ok(())
        }
    }

    let mut collector = EntryCollector {
        entries: Vec::new(),
    };
    for batch in &batches {
        collector.visit_rows_of(batch.as_ref())?;
    }
    Ok(collector.entries)
}

/// A deletion vector as surfaced by a table scan (protocol shape: storageType/
/// pathOrInlineDv/cardinality, distinct from a manifest entry's own DV encoding).
///
/// Only for checking state that isn't persisted in any manifest yet -- e.g. right after a
/// plain log commit, before the next manifest commit writes it into a manifest. A log commit
/// never writes a root or leaf manifest, so `assert_root_entries`/`assert_leaf_entries` can't
/// see it; the scan (which fuses root + leaves + the log) is the only thing that can. Once a
/// manifest commit has written the state, prefer `assert_root_entries`/`assert_leaf_entries`
/// -- they verify what was actually persisted, not what the read path reconstructs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedDv {
    pub storage_type: String,
    pub path_or_inline_dv: String,
    pub cardinality: i64,
}

/// Asserts `path` currently shows `expected` via a table scan (`None` for no DV).
pub fn assert_scan_dv(
    scan: Scan,
    engine: &dyn Engine,
    path: &str,
    expected: Option<&ExpectedDv>,
) -> DeltaResult<()> {
    let actual = collect_scan_dvs(scan, engine)?;
    let actual_dv = actual
        .get(path)
        .unwrap_or_else(|| panic!("{path}: not found in scan"));
    assert_eq!(actual_dv.as_ref(), expected, "{path}: DV mismatch");
    Ok(())
}

/// Collects each scanned file's DV, keyed by path.
fn collect_scan_dvs(
    scan: Scan,
    engine: &dyn Engine,
) -> DeltaResult<HashMap<String, Option<ExpectedDv>>> {
    struct DvCollector {
        files: HashMap<String, Option<ExpectedDv>>,
    }

    impl FilteredRowVisitor for DvCollector {
        fn selected_column_names_and_types(&self) -> (&'static [ColumnName], &'static [DataType]) {
            static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                LazyLock::new(|| {
                    (
                        vec![
                            ColumnName::new(["path"]),
                            ColumnName::new(["deletionVector", "storageType"]),
                            ColumnName::new(["deletionVector", "pathOrInlineDv"]),
                            ColumnName::new(["deletionVector", "cardinality"]),
                        ],
                        vec![
                            DataType::STRING,
                            DataType::STRING,
                            DataType::STRING,
                            DataType::LONG,
                        ],
                    )
                });
            (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
        }

        fn visit_filtered<'a>(
            &mut self,
            getters: &[&'a dyn GetData<'a>],
            rows: RowIndexIterator<'_>,
        ) -> DeltaResult<()> {
            for row_index in rows {
                let path: String = getters[0].get(row_index, "path")?;
                let dv = if let Some(storage_type) =
                    getters[1].get_opt(row_index, "deletionVector.storageType")?
                {
                    Some(ExpectedDv {
                        storage_type,
                        path_or_inline_dv: getters[2]
                            .get(row_index, "deletionVector.pathOrInlineDv")?,
                        cardinality: getters[3].get(row_index, "deletionVector.cardinality")?,
                    })
                } else {
                    None
                };
                self.files.insert(path, dv);
            }
            Ok(())
        }
    }

    let mut all_files = HashMap::new();
    for scan_metadata_result in scan.scan_metadata(engine)? {
        let scan_metadata = scan_metadata_result?;
        let mut collector = DvCollector {
            files: HashMap::new(),
        };
        collector.visit_rows_of(&scan_metadata.scan_files)?;
        all_files.extend(collector.files);
    }
    Ok(all_files)
}
