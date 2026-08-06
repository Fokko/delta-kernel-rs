//! Differential tests asserting that a table whose file inventory lives in an adaptive
//! metadata tree (AMT) reads identically to the same logical table whose inventory lives in
//! the delta log.
//!
//! No external tool produces `metadataTree-experimental` tables, so AMT reads cannot be
//! validated against the golden tables in `golden_tables.rs`. These tests recover the same
//! signal by replaying one workload under both storage modes and asserting the two tables
//! are indistinguishable through the public `Snapshot`/`Scan` API.
//!
//! Every scenario compares state at *each* version rather than only the latest. AMT's read
//! path diverges from the log path precisely at the content root boundary -- file actions
//! are dropped from commits at or below the root version -- so a bug there is invisible to a
//! latest-version read but shows up immediately under time travel.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use delta_kernel::arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
use delta_kernel::arrow::util::pretty::pretty_format_batches;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::expressions::{column_expr, Scalar};
use delta_kernel::scan::{AfterSequentialScanMetadata, ParallelScanMetadata};
use delta_kernel::schema::{DataType, SchemaRef, StructField, StructType};
use delta_kernel::transaction::create_table::create_table;
use delta_kernel::transaction::data_layout::DataLayout;
use delta_kernel::transaction::CommitResult;
use delta_kernel::{
    Engine, Expression as Expr, Predicate as Pred, PredicateRef, Snapshot, Version,
};
use rstest::rstest;
use test_utils::{engine_store_setup, read_scan};
use url::Url;
use uuid::Uuid;

use crate::common::amt_test_utils::{
    collect_scan_dvs, collect_scanned_files, dv_descriptor, remove_files_by_path,
};

type TestEngine = Arc<DefaultEngine<TokioBackgroundExecutor>>;

/// The partition column used by the partitioned scenarios.
const PARTITION_COL: &str = "category";

// ==============================================================================
// Harness
// ==============================================================================

/// Where a table keeps its file inventory.
///
/// Both modes enable column mapping, since AMT requires it -- that keeps the storage of file
/// inventory the only difference between a scenario's two runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TreeMode {
    /// File inventory lives in root and leaf manifests reached through a content root.
    Amt,
    /// File inventory lives in `add`/`remove` actions in the delta log.
    Log,
}

impl TreeMode {
    /// A suffix making each mode's table name unique within a scenario.
    fn table_suffix(self) -> &'static str {
        match self {
            TreeMode::Amt => "amt",
            TreeMode::Log => "log",
        }
    }
}

/// One step of a workload, replayed identically under both [`TreeMode`]s.
#[derive(Clone, Debug)]
enum Op {
    /// Appends one parquet file holding `values`, in partition `partition` if the table is
    /// partitioned.
    ///
    /// `manifest_commit` folds the new file into the content tree instead of writing an
    /// `add` to the log. It only applies to [`TreeMode::Amt`]; the log run always commits
    /// through the log, which is what makes the two runs comparable.
    Append {
        values: Vec<i32>,
        partition: Option<&'static str>,
        manifest_commit: bool,
    },
    /// Removes every file that the [`Op::Append`] at `origin` produced.
    ///
    /// The removal always goes through a log commit. Under [`TreeMode::Amt`] that is the
    /// RFC-shaped log removal, which handles files the content tree owns and files only the
    /// log knows about alike.
    RemoveFilesFrom { origin: usize },
    /// Attaches a deletion vector of `cardinality` rows to every file the [`Op::Append`] at
    /// `origin` produced.
    ///
    /// The vector's backing file is never written, so a workload containing this op compares
    /// scan metadata only -- see [`Workload::reads_data`].
    AddDvTo { origin: usize, cardinality: i64 },
}

impl Op {
    /// An unpartitioned append that lands in the delta log.
    fn append(values: impl IntoIterator<Item = i32>) -> Self {
        Op::Append {
            values: values.into_iter().collect(),
            partition: None,
            manifest_commit: false,
        }
    }

    /// An unpartitioned append that folds into the content tree under [`TreeMode::Amt`].
    fn manifest_append(values: impl IntoIterator<Item = i32>) -> Self {
        Op::Append {
            values: values.into_iter().collect(),
            partition: None,
            manifest_commit: true,
        }
    }

    /// Sets the partition this append writes into.
    fn in_partition(self, partition: &'static str) -> Self {
        match self {
            Op::Append {
                values,
                manifest_commit,
                ..
            } => Op::Append {
                values,
                partition: Some(partition),
                manifest_commit,
            },
            other => other,
        }
    }
}

/// A scenario: what table to build, what to do to it, and how to look at it.
struct Workload<'a> {
    /// Names the pair of tables this scenario creates.
    scenario: &'a str,
    schema: SchemaRef,
    layout: DataLayout,
    ops: &'a [Op],
}

impl<'a> Workload<'a> {
    fn new(scenario: &'a str, schema: SchemaRef, ops: &'a [Op]) -> Self {
        Self {
            scenario,
            schema,
            layout: DataLayout::None,
            ops,
        }
    }

    fn with_layout(mut self, layout: DataLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Whether scans may read data files. Deletion vectors are recorded as descriptors
    /// pointing at bitmap files the harness never writes, so any workload using them is
    /// limited to comparing scan metadata.
    fn reads_data(&self) -> bool {
        !self.ops.iter().any(|op| matches!(op, Op::AddDvTo { .. }))
    }
}

/// Everything a scan can observe about a table at one version, normalized so that two runs
/// of the same workload are directly comparable.
#[derive(Debug, PartialEq, Eq)]
struct TableState {
    version: Version,
    /// Pretty-printed scan output with data rows sorted, or `None` when the workload cannot
    /// read data files.
    data: Option<Vec<String>>,
    /// Live files, each identified by the op that produced it and by the cardinality of its
    /// deletion vector. Parquet file names contain a fresh UUID per write, so the raw paths
    /// never match across runs.
    files: BTreeSet<(usize, Option<i64>)>,
}

/// Runs `workload` under both tree modes and asserts the two tables look the same at every
/// version.
async fn assert_modes_agree(workload: Workload<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let scenario = workload.scenario;
    let amt = run_workload(&workload, TreeMode::Amt).await?;
    let log = run_workload(&workload, TreeMode::Log).await?;

    assert_eq!(
        amt.len(),
        log.len(),
        "{scenario}: the two modes produced different version counts"
    );
    for (amt_state, log_state) in amt.iter().zip(&log) {
        assert_eq!(
            amt_state, log_state,
            "{scenario}: AMT and log tables diverge at version {}",
            amt_state.version
        );
    }
    Ok(())
}

/// Replays `workload` in `mode`, returning the [`TableState`] observed after each op.
///
/// Also asserts those states are reproducible from cold snapshot loads, so a mode that only
/// reads correctly while a table is growing in place does not pass unnoticed.
async fn run_workload(
    workload: &Workload<'_>,
    mode: TreeMode,
) -> Result<Vec<TableState>, Box<dyn std::error::Error>> {
    let table_name = format!("{}_{}", workload.scenario, mode.table_suffix());
    let (url, engine, file_origin, states) = build_table(workload, mode).await?;

    // Re-read every version from scratch. The states above came from a table that grew in
    // place; these come from cold snapshot loads, which is the path time travel takes.
    let latest = states.last().expect("at least the initial state").version;
    let mut time_traveled = Vec::with_capacity(states.len());
    for version in 0..=latest {
        time_traveled.push(capture_state(
            &url,
            &engine,
            workload,
            Some(version),
            &file_origin,
        )?);
    }
    assert_eq!(
        states, time_traveled,
        "{table_name}: time travel disagrees with the incrementally observed states"
    );

    Ok(states)
}

/// Creates the table in `mode` and applies the workload's ops, capturing a [`TableState`]
/// after each one (plus the initial empty state at version 0).
///
/// Returns the table, its engine, the map from file path to the op that produced it, and the
/// captured states.
#[allow(clippy::type_complexity)]
async fn build_table(
    workload: &Workload<'_>,
    mode: TreeMode,
) -> Result<(Url, TestEngine, HashMap<String, usize>, Vec<TableState>), Box<dyn std::error::Error>>
{
    let table_name = format!("{}_{}", workload.scenario, mode.table_suffix());
    let (_store, engine, url) = engine_store_setup(&table_name, None);
    let engine: TestEngine = Arc::new(engine);
    create_table_for_mode(&url, &engine, workload, mode)?;

    // Records which op produced each file, so states compare independently of the random
    // parquet names every run generates.
    let mut file_origin: HashMap<String, usize> = HashMap::new();
    let mut states = vec![capture_state(&url, &engine, workload, None, &file_origin)?];

    for (op_index, op) in workload.ops.iter().enumerate() {
        match op {
            Op::Append {
                values,
                partition,
                manifest_commit,
            } => {
                append(
                    &url,
                    &engine,
                    &workload.schema,
                    mode,
                    values,
                    *partition,
                    *manifest_commit,
                )
                .await?;
            }
            Op::RemoveFilesFrom { origin } => {
                remove_files_from(&url, &engine, &file_origin, *origin)?;
            }
            Op::AddDvTo {
                origin,
                cardinality,
            } => {
                add_dv_to(&url, &engine, &file_origin, *origin, *cardinality)?;
            }
        }

        // Any path not seen before belongs to the op that just ran.
        for path in live_paths(&url, &engine)? {
            file_origin.entry(path).or_insert(op_index);
        }
        states.push(capture_state(&url, &engine, workload, None, &file_origin)?);
    }

    Ok((url, engine, file_origin, states))
}

/// Creates the table, enabling the metadata tree only for [`TreeMode::Amt`].
fn create_table_for_mode(
    url: &Url,
    engine: &TestEngine,
    workload: &Workload<'_>,
    mode: TreeMode,
) -> Result<(), Box<dyn std::error::Error>> {
    // Column mapping is on in both modes because AMT requires it, and deletion vectors are
    // on so a DV workload does not also change the protocol between modes.
    let mut properties = vec![
        ("delta.columnMapping.mode", "id"),
        ("delta.feature.deletionVectors", "supported"),
    ];
    if mode == TreeMode::Amt {
        properties.push(("delta.feature.metadataTree-experimental", "supported"));
    }

    create_table(url.as_str(), workload.schema.clone(), "amt equivalence")
        .with_data_layout(workload.layout.clone())
        .with_table_properties(properties)
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?
        .commit(engine.as_ref())?
        .unwrap_committed();
    Ok(())
}

/// Appends one parquet file holding `values`.
async fn append(
    url: &Url,
    engine: &TestEngine,
    schema: &SchemaRef,
    mode: TreeMode,
    values: &[i32],
    partition: Option<&str>,
    manifest_commit: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("amt equivalence")
        .with_data_change(true);
    if mode == TreeMode::Amt && manifest_commit {
        txn.with_manifest_commit()?;
    }

    let arrow_schema: Arc<ArrowSchema> = Arc::new(schema.as_ref().try_into_arrow()?);
    let ids: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
    let columns: Vec<ArrayRef> = match partition {
        Some(value) => vec![
            ids,
            Arc::new(StringArray::from(vec![value; values.len()])) as ArrayRef,
        ],
        None => vec![ids],
    };
    let batch = RecordBatch::try_new(arrow_schema, columns)?;

    let write_context = match partition {
        Some(value) => txn.partitioned_write_context(HashMap::from([(
            PARTITION_COL.to_string(),
            Scalar::from(value),
        )]))?,
        None => txn.unpartitioned_write_context()?,
    };
    let add_files_metadata = engine
        .write_parquet(&ArrowEngineData::new(batch), &write_context)
        .await?;
    txn.add_files(add_files_metadata);

    commit(txn, engine)
}

/// Removes every live file produced by the op at `origin`.
fn remove_files_from(
    url: &Url,
    engine: &TestEngine,
    file_origin: &HashMap<String, usize>,
    origin: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let targets = paths_from_op(file_origin, origin);
    let target_refs: Vec<&str> = targets.iter().map(String::as_str).collect();

    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("amt equivalence")
        .with_data_change(true);

    let scan = snapshot.scan_builder().build()?;
    let removed = remove_files_by_path(&mut txn, scan, engine.as_ref(), &target_refs)?;
    assert_eq!(
        removed,
        targets.len(),
        "expected to remove {} files, removed {removed}",
        targets.len()
    );

    commit(txn, engine)
}

/// Attaches a deletion vector to every live file produced by the op at `origin`.
fn add_dv_to(
    url: &Url,
    engine: &TestEngine,
    file_origin: &HashMap<String, usize>,
    origin: usize,
    cardinality: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("amt equivalence")
        .with_operation("UPDATE".to_string())
        .with_data_change(true);

    // The UUID is derived from the target op so both runs produce byte-identical
    // descriptors; a random one would make the two tables trivially unequal.
    let new_dvs: HashMap<String, DeletionVectorDescriptor> = paths_from_op(file_origin, origin)
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            let uuid = Uuid::from_u128((origin * 1000 + index) as u128);
            let (descriptor, _location) = dv_descriptor(uuid, 1, 40, cardinality);
            (path, descriptor)
        })
        .collect();

    let current_files: Vec<_> = snapshot
        .scan_builder()
        .build()?
        .scan_metadata(engine.as_ref())?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|metadata| metadata.scan_files)
        .collect();
    txn.update_deletion_vectors(new_dvs, current_files.into_iter().map(Ok))?;

    commit(txn, engine)
}

/// Commits `txn`, requiring it to succeed.
fn commit(
    txn: delta_kernel::transaction::Transaction,
    engine: &TestEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    match txn.commit(engine.as_ref())? {
        CommitResult::CommittedTransaction(_) => Ok(()),
        other => Err(format!("expected a committed transaction, got {other:?}").into()),
    }
}

/// The paths of every live file produced by the op at `origin`.
fn paths_from_op(file_origin: &HashMap<String, usize>, origin: usize) -> Vec<String> {
    let paths: Vec<String> = file_origin
        .iter()
        .filter(|(_, op_index)| **op_index == origin)
        .map(|(path, _)| path.clone())
        .collect();
    assert!(!paths.is_empty(), "op {origin} produced no files");
    paths
}

/// The paths of every file an unfiltered scan currently sees.
fn live_paths(url: &Url, engine: &TestEngine) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    Ok(collect_scanned_files(snapshot, engine.as_ref())?
        .file_paths
        .into_iter()
        .collect())
}

/// The paths [`Scan::parallel_scan_metadata`] reports, driving both of its phases.
///
/// The second phase only has work when the table has sidecars or a multi-part checkpoint;
/// otherwise the first phase finishes `Done` and this is just the sequential result.
fn parallel_paths(
    url: &Url,
    engine: &TestEngine,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    let engine = engine.clone() as Arc<dyn Engine>;
    let mut sequential = snapshot
        .scan_builder()
        .build()?
        .parallel_scan_metadata(engine.clone())?;

    let mut paths = sequential.try_fold(Vec::new(), |acc, metadata| {
        metadata?.visit_scan_files(acc, |paths: &mut Vec<String>, file| paths.push(file.path))
    })?;

    if let AfterSequentialScanMetadata::Parallel { state, files } = sequential.finish()? {
        let mut parallel = ParallelScanMetadata::try_new(engine, Arc::new(*state), files)?;
        paths = parallel.try_fold(paths, |acc, metadata| {
            metadata?.visit_scan_files(acc, |paths: &mut Vec<String>, file| paths.push(file.path))
        })?;
    }

    paths.sort();
    Ok(paths)
}

/// Asserts the two ways of listing a table's files agree in `mode`.
async fn assert_parallel_matches_scan(mode: TreeMode) -> Result<(), Box<dyn std::error::Error>> {
    // A content root with log commits on either side, so a mode that only reads the log still
    // finds the files from the appends that did not fold into the tree.
    let ops = [
        Op::append([1, 2, 3]),
        Op::manifest_append([4, 5, 6]),
        Op::append([7, 8, 9]),
    ];
    let workload = Workload::new("parallel_scan_metadata", id_schema()?, &ops);
    let (url, engine, _, _) = build_table(&workload, mode).await?;

    let mut expected = live_paths(&url, &engine)?;
    expected.sort();
    assert_eq!(expected.len(), ops.len(), "every append should be live");
    assert_eq!(parallel_paths(&url, &engine)?, expected);

    Ok(())
}

/// The control: on a log table the two agree, so the comparison itself is sound.
#[tokio::test]
async fn test_parallel_scan_metadata_matches_scan_metadata_for_log_tables(
) -> Result<(), Box<dyn std::error::Error>> {
    assert_parallel_matches_scan(TreeMode::Log).await
}

/// `parallel_scan_metadata` should list the same files as `scan_metadata`.
///
/// They are two implementations of one question, and the parallel one is a separate pipeline
/// rather than a wrapper: it reads commits and classic checkpoint parts and never looks at the
/// content root. On a metadata tree table the files living in the tree are therefore invisible
/// to it, and since nothing rejects such a table the caller gets a short list rather than an
/// error -- the worst shape for a distributed read, where the missing rows only show up as a
/// wrong answer much later.
///
/// The appends that stayed in the log are still found, so what this asserts is that the tree
/// residents come back too.
#[tokio::test]
#[ignore = "parallel_scan_metadata replays only commits and classic checkpoints, so files that \
            live in the content tree are missing from its result"]
async fn test_parallel_scan_metadata_matches_scan_metadata_for_metadata_tree_tables(
) -> Result<(), Box<dyn std::error::Error>> {
    assert_parallel_matches_scan(TreeMode::Amt).await
}

/// Captures the table at `version` (latest when `None`).
fn capture_state(
    url: &Url,
    engine: &TestEngine,
    workload: &Workload<'_>,
    version: Option<Version>,
    file_origin: &HashMap<String, usize>,
) -> Result<TableState, Box<dyn std::error::Error>> {
    let mut builder = Snapshot::builder_for(url.clone());
    if let Some(version) = version {
        builder = builder.at_version(version);
    }
    let snapshot = builder.build(engine.as_ref())?;
    let snapshot_version = snapshot.version();

    // Rejects a file surfacing twice, which a content root fused with the log could
    // otherwise do unnoticed.
    collect_scanned_files(snapshot.clone(), engine.as_ref())?;

    let files = collect_scan_dvs(snapshot.clone().scan_builder().build()?, engine.as_ref())?
        .iter()
        .map(|(path, dv)| {
            let origin = *file_origin
                .get(path)
                .unwrap_or_else(|| panic!("file {path} has no recorded origin"));
            (origin, dv.as_ref().map(|dv| dv.cardinality))
        })
        .collect();

    let data = workload
        .reads_data()
        .then(|| {
            let scan = snapshot.scan_builder().build()?;
            read_scan(&scan, engine.clone() as Arc<dyn Engine>)
        })
        .transpose()?
        .map(|batches| sorted_rows(&batches));

    Ok(TableState {
        version: snapshot_version,
        data,
        files,
    })
}

/// Pretty-prints `batches` with the data rows sorted, leaving the table header and footer in
/// place so a mismatch is readable.
fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
    let formatted = pretty_format_batches(batches)
        .expect("batches are printable")
        .to_string();
    let mut lines: Vec<String> = formatted.trim().lines().map(str::to_string).collect();
    if lines.len() > 3 {
        let last = lines.len() - 1;
        lines[2..last].sort_unstable();
    }
    lines
}

// ==============================================================================
// Schemas
// ==============================================================================

/// Single `id` column. Column mapping metadata is assigned by `create_table`.
fn id_schema() -> Result<SchemaRef, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![StructField::nullable(
        "id",
        DataType::INTEGER,
    )])?))
}

/// An `id` column plus a `category` column to partition on.
fn partitioned_schema() -> Result<SchemaRef, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![
        StructField::nullable("id", DataType::INTEGER),
        StructField::nullable(PARTITION_COL, DataType::STRING),
    ])?))
}

// ==============================================================================
// Scenarios
// ==============================================================================

/// Appends only. `manifest_commit_at` picks which append folds into the content tree, which
/// places the content root before, among, and after the log-only commits in turn.
#[rstest]
#[case::root_first(0)]
#[case::root_middle(1)]
#[case::root_last(2)]
#[tokio::test]
async fn test_amt_matches_log_for_appends(
    #[case] manifest_commit_at: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ops: Vec<Op> = [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            if index == manifest_commit_at {
                Op::manifest_append(values)
            } else {
                Op::append(values)
            }
        })
        .collect();

    assert_modes_agree(Workload::new(
        &format!("appends_root_at_{manifest_commit_at}"),
        id_schema()?,
        &ops,
    ))
    .await
}

/// Several manifest commits in a row, so each one rewrites a content root that the previous
/// one produced.
#[tokio::test]
async fn test_amt_matches_log_for_consecutive_manifest_commits(
) -> Result<(), Box<dyn std::error::Error>> {
    let ops = [
        Op::manifest_append([1, 2, 3]),
        Op::manifest_append([4, 5, 6]),
        Op::manifest_append([7, 8, 9]),
    ];

    assert_modes_agree(Workload::new(
        "consecutive_manifest_commits",
        id_schema()?,
        &ops,
    ))
    .await
}

/// Removes a file that the content root owns and a file that only the log knows about, so
/// both the tree-resident and log-resident removal paths are compared against the log-only
/// table.
#[rstest]
#[case::remove_file_in_content_root(0)]
#[case::remove_file_added_after_root(2)]
#[tokio::test]
async fn test_amt_matches_log_for_removes(
    #[case] origin: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ops = [
        Op::append([1, 2, 3]),
        Op::manifest_append([4, 5, 6]),
        Op::append([7, 8, 9]),
        Op::RemoveFilesFrom { origin },
    ];

    assert_modes_agree(Workload::new(
        &format!("removes_origin_{origin}"),
        id_schema()?,
        &ops,
    ))
    .await
}

/// Attaches a deletion vector to a file the content root owns and to a file only the log
/// knows about. Compares scan metadata only, since the vectors' backing files do not exist.
#[rstest]
#[case::dv_on_file_in_content_root(0)]
#[case::dv_on_file_added_after_root(2)]
#[tokio::test]
async fn test_amt_matches_log_for_deletion_vectors(
    #[case] origin: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ops = [
        Op::append([1, 2, 3]),
        Op::manifest_append([4, 5, 6]),
        Op::append([7, 8, 9]),
        Op::AddDvTo {
            origin,
            cardinality: 2,
        },
    ];

    assert_modes_agree(Workload::new(
        &format!("dvs_origin_{origin}"),
        id_schema()?,
        &ops,
    ))
    .await
}

/// Data skipping over a table holding a non-matching file in the content tree, a non-matching
/// file in the log, and two matching files. Both ways a file can reach the tree are covered:
/// folded in from a prior log commit, or written straight there through `add_files`.
///
/// Exact file-set equality would be the wrong bar for a filtered scan, because skipping is
/// best-effort and AMT is meant to prune *more* than the log, not the same. So the comparison
/// asserts the two properties that should hold either way: no matching row is lost, and AMT
/// prunes at least as much as the log.
///
/// Both arrangements pass, and each holds the table to exactly one manifest commit. That is
/// what makes the failure in [`test_amt_skipping_after_a_second_manifest_commit`] specific --
/// bounds arrive in the tree correctly, and are lost later.
#[rstest]
#[case::file_folded_into_tree(
    [
        Op::append([1, 2, 3]),
        Op::manifest_append([40, 50, 60]),
        Op::append([7, 8, 9]),
        Op::append([70, 80, 90]),
    ]
)]
#[case::file_written_into_tree(
    [
        Op::manifest_append([1, 2, 3]),
        Op::append([40, 50, 60]),
        Op::append([7, 8, 9]),
        Op::append([70, 80, 90]),
    ]
)]
#[tokio::test]
async fn test_amt_skips_tree_resident_file_after_one_manifest_commit(
    #[case] ops: [Op; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_skipping_at_least_as_good("skip_one_commit", &ops).await
}

/// The same two arrangements, each given a second manifest commit at op 3.
///
/// Nothing else changes: op 0's file is already in the tree with usable bounds, as
/// [`test_amt_skips_tree_resident_file_after_one_manifest_commit`] shows. The second manifest
/// commit reloads the root the first one wrote, and afterwards op 0 is no longer skippable,
/// while the log-backed run still prunes it. Whichever way op 0 first reached the tree makes
/// no difference, so the loss is in reloading a root rather than in populating it.
#[rstest]
#[case::root_built_from_folded_add(
    [
        Op::append([1, 2, 3]),
        Op::manifest_append([40, 50, 60]),
        Op::append([7, 8, 9]),
        Op::manifest_append([70, 80, 90]),
    ]
)]
#[case::root_built_from_written_add(
    [
        Op::manifest_append([1, 2, 3]),
        Op::manifest_append([40, 50, 60]),
        Op::append([7, 8, 9]),
        Op::append([70, 80, 90]),
    ]
)]
#[tokio::test]
#[ignore = "a manifest commit that reloads an existing content root drops the stats bounds of \
            the entries already in it, so those files stop being skippable"]
async fn test_amt_skipping_after_a_second_manifest_commit(
    #[case] ops: [Op; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_skipping_at_least_as_good("skip_second_commit", &ops).await
}

/// Asserts a filtered scan loses no matching row under either mode, and that AMT prunes at
/// least as many files as the log.
///
/// `ops` must put the non-matching values below `THRESHOLD` and the matching ones above it.
async fn assert_skipping_at_least_as_good(
    scenario: &str,
    ops: &[Op],
) -> Result<(), Box<dyn std::error::Error>> {
    const THRESHOLD: i32 = 30;
    let workload = Workload::new(scenario, id_schema()?, ops);
    let predicate: PredicateRef = Arc::new(Pred::gt(column_expr!("id"), Expr::literal(THRESHOLD)));

    let mut outcomes = Vec::new();
    for mode in [TreeMode::Amt, TreeMode::Log] {
        let (url, engine, file_origin, _) = build_table(&workload, mode).await?;
        let snapshot = Snapshot::builder_for(url).build(engine.as_ref())?;

        let scan = snapshot
            .clone()
            .scan_builder()
            .with_predicate(predicate.clone())
            .build()?;
        // Naming survivors by the op that wrote them keeps the failure readable, since the
        // parquet names are random.
        let mut survivors: Vec<usize> = collect_scan_dvs(scan, engine.as_ref())?
            .keys()
            .map(|path| file_origin[path])
            .collect();
        survivors.sort_unstable();

        let scan = snapshot
            .scan_builder()
            .with_predicate(predicate.clone())
            .build()?;
        let batches = read_scan(&scan, engine.clone() as Arc<dyn Engine>)?;
        outcomes.push((survivors, matching_ids(&batches, THRESHOLD)));
    }
    let (amt_survivors, amt_ids) = &outcomes[0];
    let (log_survivors, log_ids) = &outcomes[1];

    assert_eq!(
        amt_ids, log_ids,
        "data skipping dropped rows matching the predicate"
    );
    assert!(
        amt_survivors.len() <= log_survivors.len(),
        "AMT kept files from ops {amt_survivors:?} where the log kept only {log_survivors:?}; \
         a metadata tree should prune at least as much as the log"
    );
    Ok(())
}

/// The sorted `id` values above `threshold` across `batches`.
///
/// A predicate only tells kernel which files it may skip, so surviving files still carry
/// non-matching rows; filtering here isolates the rows the scan was actually asked for.
fn matching_ids(batches: &[RecordBatch], threshold: i32) -> Vec<i32> {
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name("id")
                .expect("scan output has an id column")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("id is an int column")
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .filter(|id| *id > threshold)
        .collect();
    ids.sort_unstable();
    ids
}

/// Appends to a table clustered on `id`, which puts clustering domain metadata and
/// clustering-driven stats alongside the content tree.
#[tokio::test]
async fn test_amt_matches_log_for_clustered_appends() -> Result<(), Box<dyn std::error::Error>> {
    let ops = [
        Op::append([1, 2, 3]),
        Op::manifest_append([4, 5, 6]),
        Op::append([7, 8, 9]),
    ];

    assert_modes_agree(
        Workload::new("clustered_appends", id_schema()?, &ops)
            .with_layout(DataLayout::clustered(["id"])),
    )
    .await
}

/// Partitioned appends spread over two partitions.
///
/// The two cases differ only in whether a log-committed partitioned add exists before the
/// content root is built, which is what separates writing partition values straight into the
/// tree from folding them in during a root rebuild. Both currently fail, and differently:
/// `all_manifest_commits` reads every partition value back as null, while
/// `root_after_log_append` cannot even commit -- rebuilding the root over a log-committed
/// partitioned add fails evaluating the entry's non-nullable `partition` struct. The log-
/// backed half of each case reads its partition values back correctly.
#[rstest]
#[case::all_manifest_commits(
    [
        Op::manifest_append([1, 2, 3]).in_partition("a"),
        Op::manifest_append([4, 5, 6]).in_partition("b"),
        Op::manifest_append([7, 8, 9]).in_partition("a"),
    ]
)]
#[case::root_after_log_append(
    [
        Op::append([1, 2, 3]).in_partition("a"),
        Op::manifest_append([4, 5, 6]).in_partition("b"),
        Op::append([7, 8, 9]).in_partition("a"),
    ]
)]
#[tokio::test]
#[ignore = "AMT loses partition values: manifest commits read them back as null, and a root \
            rebuild over a log-committed partitioned add errors out"]
async fn test_amt_matches_log_for_partitioned_appends(
    #[case] ops: [Op; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_commits = ops
        .iter()
        .filter(|op| {
            matches!(
                op,
                Op::Append {
                    manifest_commit: true,
                    ..
                }
            )
        })
        .count();

    assert_modes_agree(
        Workload::new(
            &format!("partitioned_appends_{manifest_commits}"),
            partitioned_schema()?,
            &ops,
        )
        .with_layout(DataLayout::partitioned([PARTITION_COL])),
    )
    .await
}
