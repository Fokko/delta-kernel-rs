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

use delta_kernel::arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use delta_kernel::arrow::datatypes::Schema as ArrowSchema;
use delta_kernel::arrow::util::pretty::pretty_format_batches;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryIntoArrow as _;
use delta_kernel::engine::arrow_data::ArrowEngineData;
use delta_kernel::engine::default::executor::tokio::TokioBackgroundExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::expressions::Scalar;
use delta_kernel::schema::{
    ColumnMetadataKey, DataType, MetadataValue, SchemaRef, StructField, StructType,
};
use delta_kernel::transaction::CommitResult;
use delta_kernel::{Engine, Snapshot, Version};
use rstest::rstest;
use test_utils::{create_table, engine_store_setup, read_scan};
use url::Url;

use crate::common::amt_test_utils::{collect_scanned_files, remove_files_by_path};

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
    /// Reader and writer features to create the table with.
    fn table_features(self) -> (Vec<&'static str>, Vec<&'static str>) {
        let mut features = vec!["columnMapping", "deletionVectors"];
        if self == TreeMode::Amt {
            features.push("metadataTree-experimental");
        }
        (features.clone(), features)
    }

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

/// Everything a scan can observe about a table at one version, normalized so that two runs
/// of the same workload are directly comparable.
#[derive(Debug, PartialEq, Eq)]
struct TableState {
    version: Version,
    /// Pretty-printed scan output, with data rows sorted.
    data: Vec<String>,
    /// Live files identified by the op that produced them. Parquet file names contain a
    /// fresh UUID per write, so the raw paths never match across runs.
    files: BTreeSet<usize>,
}

/// Runs `ops` under both tree modes and asserts the two tables look the same at every
/// version.
async fn assert_modes_agree(
    scenario: &str,
    schema: &SchemaRef,
    partition_cols: &[&str],
    ops: &[Op],
) -> Result<(), Box<dyn std::error::Error>> {
    let amt = run_workload(scenario, TreeMode::Amt, schema, partition_cols, ops).await?;
    let log = run_workload(scenario, TreeMode::Log, schema, partition_cols, ops).await?;

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

/// Builds a table in `mode`, applies `ops` in order, and captures a [`TableState`] after
/// each one (plus the initial empty state at version 0).
async fn run_workload(
    scenario: &str,
    mode: TreeMode,
    schema: &SchemaRef,
    partition_cols: &[&str],
    ops: &[Op],
) -> Result<Vec<TableState>, Box<dyn std::error::Error>> {
    let table_name = format!("{scenario}_{}", mode.table_suffix());
    let (store, engine, location) = engine_store_setup(&table_name, None);
    let engine: TestEngine = Arc::new(engine);

    let (reader_features, writer_features) = mode.table_features();
    let url = create_table(
        store,
        location,
        schema.clone(),
        partition_cols,
        true,
        reader_features,
        writer_features,
    )
    .await?;

    // Records which op produced each file, so states compare independently of the random
    // parquet names every run generates.
    let mut file_origin: HashMap<String, usize> = HashMap::new();
    let mut states = vec![capture_state(&url, &engine, None, &file_origin)?];

    for (op_index, op) in ops.iter().enumerate() {
        match op {
            Op::Append {
                values,
                partition,
                manifest_commit,
            } => {
                append(
                    &url,
                    &engine,
                    schema,
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
        }

        // Any path not seen before belongs to the op that just ran.
        for path in live_paths(&url, &engine)? {
            file_origin.entry(path).or_insert(op_index);
        }
        states.push(capture_state(&url, &engine, None, &file_origin)?);
    }

    // Re-read every version from scratch. The states above came from a table that grew in
    // place; these come from cold snapshot loads, which is the path time travel takes.
    let latest = states.last().expect("at least the initial state").version;
    let mut time_traveled = Vec::with_capacity(states.len());
    for version in 0..=latest {
        time_traveled.push(capture_state(&url, &engine, Some(version), &file_origin)?);
    }
    assert_eq!(
        states, time_traveled,
        "{table_name}: time travel disagrees with the incrementally observed states"
    );

    Ok(states)
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

    match txn.commit(engine.as_ref())? {
        CommitResult::CommittedTransaction(_) => Ok(()),
        other => Err(format!("expected a committed transaction, got {other:?}").into()),
    }
}

/// Removes every live file produced by the op at `origin`.
fn remove_files_from(
    url: &Url,
    engine: &TestEngine,
    file_origin: &HashMap<String, usize>,
    origin: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let targets: Vec<&str> = file_origin
        .iter()
        .filter(|(_, op_index)| **op_index == origin)
        .map(|(path, _)| path.as_str())
        .collect();
    assert!(
        !targets.is_empty(),
        "op {origin} produced no files to remove"
    );

    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("amt equivalence")
        .with_data_change(true);

    let scan = snapshot.scan_builder().build()?;
    let removed = remove_files_by_path(&mut txn, scan, engine.as_ref(), &targets)?;
    assert_eq!(
        removed,
        targets.len(),
        "expected to remove {} files, removed {removed}",
        targets.len()
    );

    match txn.commit(engine.as_ref())? {
        CommitResult::CommittedTransaction(_) => Ok(()),
        other => Err(format!("expected a committed transaction, got {other:?}").into()),
    }
}

/// The paths of every file a scan currently sees.
fn live_paths(url: &Url, engine: &TestEngine) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let snapshot = Snapshot::builder_for(url.clone()).build(engine.as_ref())?;
    Ok(collect_scanned_files(snapshot, engine.as_ref())?
        .file_paths
        .into_iter()
        .collect())
}

/// Captures the table at `version` (latest when `None`).
fn capture_state(
    url: &Url,
    engine: &TestEngine,
    version: Option<Version>,
    file_origin: &HashMap<String, usize>,
) -> Result<TableState, Box<dyn std::error::Error>> {
    let mut builder = Snapshot::builder_for(url.clone());
    if let Some(version) = version {
        builder = builder.at_version(version);
    }
    let snapshot = builder.build(engine.as_ref())?;

    let files = collect_scanned_files(snapshot.clone(), engine.as_ref())?
        .file_paths
        .iter()
        .map(|path| {
            *file_origin
                .get(path)
                .unwrap_or_else(|| panic!("file {path} has no recorded origin"))
        })
        .collect();

    let version = snapshot.version();
    let scan = snapshot.scan_builder().build()?;
    let batches = read_scan(&scan, engine.clone() as Arc<dyn Engine>)?;

    Ok(TableState {
        version,
        data: sorted_rows(&batches),
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

/// A field carrying the column-mapping metadata that AMT tables require.
fn mapped_field(name: &str, data_type: DataType, id: i64) -> StructField {
    StructField::nullable(name, data_type).with_metadata([
        (
            ColumnMetadataKey::ColumnMappingId.as_ref(),
            MetadataValue::Number(id),
        ),
        (
            ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
            MetadataValue::String(format!("col-{id}")),
        ),
    ])
}

/// Single `id` column.
fn id_schema() -> Result<SchemaRef, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![mapped_field(
        "id",
        DataType::INTEGER,
        1,
    )])?))
}

/// An `id` column plus a `category` column to partition on.
fn partitioned_schema() -> Result<SchemaRef, Box<dyn std::error::Error>> {
    Ok(Arc::new(StructType::try_new(vec![
        mapped_field("id", DataType::INTEGER, 1),
        mapped_field(PARTITION_COL, DataType::STRING, 2),
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

    assert_modes_agree(
        &format!("appends_root_at_{manifest_commit_at}"),
        &id_schema()?,
        &[],
        &ops,
    )
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

    assert_modes_agree("consecutive_manifest_commits", &id_schema()?, &[], &ops).await
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

    assert_modes_agree(
        &format!("removes_origin_{origin}"),
        &id_schema()?,
        &[],
        &ops,
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
        &format!("partitioned_appends_{manifest_commits}"),
        &partitioned_schema()?,
        &[PARTITION_COL],
        &ops,
    )
    .await
}
