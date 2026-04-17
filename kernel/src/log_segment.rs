//! Represents a segment of a delta log. [`LogSegment`] wraps a set of checkpoint and commit
//! files.
use std::time::Instant;

use crate::actions::visitors::SidecarVisitor;
use crate::actions::{
    get_commit_schema, get_log_add_schema, schema_contains_file_actions, CheckpointAction,
    ContentRoot, Metadata, Protocol, Sidecar, ADD_NAME, CHECKPOINT_ACTION_NAME,
    DOMAIN_METADATA_NAME, METADATA_NAME, PROTOCOL_NAME, REMOVE_NAME, SET_TRANSACTION_NAME,
    SIDECAR_NAME,
};
use crate::committer::CatalogCommit;
use crate::expressions::ColumnName;
use crate::last_checkpoint_hint::{LastCheckpointHint, LastCheckpointHintSummary};
use crate::log_reader::commit::CommitReader;
use crate::log_replay::ActionsBatch;
use crate::metrics::{MetricEvent, MetricId, MetricsReporter};
use crate::path::LogPathFileType::*;
use crate::path::{LogPathFileType, ParsedLogPath};
use crate::schema::{DataType, SchemaRef, StructField, StructType, ToSchema};
use crate::table_features::TableFeature;
use crate::utils::require;
use crate::{
    DeltaResult, Engine, Error, Expression, FileMeta, Predicate, PredicateRef, RowVisitor,
    StorageHandler, Version, PRE_COMMIT_VERSION,
};
use delta_kernel_derive::internal_api;
use std::num::NonZero;
use std::sync::{Arc, LazyLock};

#[cfg(feature = "internal-api")]
pub use crate::log_segment_files::LogSegmentFiles;
#[cfg(not(feature = "internal-api"))]
use crate::log_segment_files::LogSegmentFiles;
use crate::schema::compare::SchemaComparison;

use crate::crc::LazyCrc;
use itertools::Itertools;
use tracing::{debug, info, instrument, warn};
use url::Url;

mod domain_metadata_replay;
mod protocol_metadata_replay;

pub(crate) use domain_metadata_replay::DomainMetadataMap;

#[cfg(test)]
mod crc_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_checkpoint_action_validation;

/// Information about checkpoint reading for data skipping optimization.
///
/// Returned alongside the actions iterator from checkpoint reading functions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[internal_api]
pub(crate) struct CheckpointReadInfo {
    /// Whether the checkpoint has compatible pre-parsed stats for data skipping.
    /// When `true`, checkpoint batches can use stats_parsed directly instead of parsing JSON.
    #[allow(unused)]
    pub has_stats_parsed: bool,
    /// Whether the checkpoint has compatible pre-parsed partition values.
    /// When `true`, checkpoint batches can read typed partition values directly from
    /// `partitionValues_parsed` instead of parsing strings from `partitionValues`.
    #[serde(default)]
    #[allow(unused)]
    pub has_partition_values_parsed: bool,
    /// The schema used to read checkpoint files, potentially including stats_parsed.
    #[allow(unused)]
    pub checkpoint_read_schema: SchemaRef,
}

impl CheckpointReadInfo {
    /// Create a CheckpointReadInfo configured to read checkpoints without using stats_parsed.
    /// This is the standard configuration when stats_parsed optimization is not available.
    #[allow(unused)]
    pub(crate) fn without_stats_parsed() -> Self {
        Self {
            has_stats_parsed: false,
            has_partition_values_parsed: false,
            checkpoint_read_schema: get_log_add_schema().clone(),
        }
    }
}

/// Result of reading actions from a log segment, containing both the actions iterator
/// and checkpoint metadata.
///
/// This struct provides named access to the return values instead of tuple indexing.
#[internal_api]
pub(crate) struct ActionsWithCheckpointInfo<A: Iterator<Item = DeltaResult<ActionsBatch>>> {
    /// Iterator over action batches read from the log segment.
    pub actions: A,
    /// Metadata about checkpoint reading, including the schema used.
    #[allow(unused)]
    pub checkpoint_info: CheckpointReadInfo,
}

/// A [`LogSegment`] represents a contiguous section of the log and is made of checkpoint files
/// and commit files and guarantees the following:
///     1. Commit file versions will not have any gaps between them.
///     2. If checkpoint(s) is/are present in the range, only commits with versions greater than the most
///        recent checkpoint version are retained. There will not be a gap between the checkpoint
///        version and the first commit version.
///     3. All checkpoint_parts must belong to the same checkpoint version, and must form a complete
///        version. Multi-part checkpoints must have all their parts.
///
/// [`LogSegment`] is used in [`Snapshot`] when built with [`LogSegment::for_snapshot`], and
/// in `TableChanges` when built with [`LogSegment::for_table_changes`].
///
/// [`Snapshot`]: crate::snapshot::Snapshot
#[derive(Debug, Clone, PartialEq, Eq)]
#[internal_api]
pub(crate) struct LogSegment {
    pub end_version: Version,
    pub checkpoint_version: Option<Version>,
    pub log_root: Url,
    pub table_root: Url,
    /// Schema of the checkpoint file(s), if known from `_last_checkpoint` hint.
    /// Used to determine if `stats_parsed` is available for data skipping.
    pub checkpoint_schema: Option<SchemaRef>,
    /// The set of log files found during listing.
    pub listed: LogSegmentFiles,
    /// Metadata from the `_last_checkpoint` hint file, stored for use by the
    /// `last_checkpoint_hint_summary()` accessor.
    pub(crate) last_checkpoint_metadata: Option<LastCheckpointHintSummary>,
}

/// A partial commit cover is a set of files that cover is a set of files that is a
/// subset (possibly the complete set) of files needed to cover a commit a range.
/// It is used for chunking together files that should be read together with the
/// same schema and the same predicate. A commit range can have multiple PartialCommitCovers
/// to accomodate special case logic for content metadata trees.
pub(crate) struct PartialCommitCover {
    pub(crate) files: Vec<FileMeta>,
    pub(crate) meta_predicate: Option<PredicateRef>,
    pub(crate) read_schema: SchemaRef,
    /// The maximum published commit version found during listing, if available.
    /// Note that this published commit file maybe not be included in
    /// `LogSegment::ascending_commit_files` if there is a catalog commit present for the same
    /// version that took priority over it.
    #[allow(dead_code)]
    pub max_published_version: Option<Version>,
}

/// Returns the identifying leaf column path for a known action type, used to build IS NOT NULL
/// predicates that enable row group skipping in checkpoint parquet files.
///
/// For `txn`, this is effective because all app ids end up in a single checkpoint part when
/// partitioned by `add.path` as the Delta spec requires. Filtering by a specific app id is not
/// worthwhile since all app ids share one part with a large min/max range (typically UUIDs).
fn action_identifying_column(action_name: &str) -> Option<ColumnName> {
    match action_name {
        METADATA_NAME => Some(ColumnName::new([METADATA_NAME, "id"])),
        PROTOCOL_NAME => Some(ColumnName::new([PROTOCOL_NAME, "minReaderVersion"])),
        SET_TRANSACTION_NAME => Some(ColumnName::new([SET_TRANSACTION_NAME, "appId"])),
        DOMAIN_METADATA_NAME => Some(ColumnName::new([DOMAIN_METADATA_NAME, "domain"])),
        _ => None,
    }
}

/// Builds an IS NOT NULL predicate for row group skipping based on the action types in `schema`.
/// Returns `None` if any top-level field in the schema is not a recognized action type, since
/// an unknown type could have non-null rows in the same row group, making skipping unsafe.
fn schema_to_is_not_null_predicate(schema: &StructType) -> Option<PredicateRef> {
    // Collect identifying columns for every field; short-circuit to None on any unknown field.
    let columns: Vec<ColumnName> = schema
        .fields()
        .map(|f| action_identifying_column(f.name()))
        .collect::<Option<_>>()?;
    let mut predicates = columns
        .into_iter()
        .map(|col| Expression::column(col).is_not_null());
    let first = predicates.next()?;
    Some(Arc::new(predicates.fold(first, Predicate::or)))
}

impl LogSegment {
    /// Creates a synthetic LogSegment for pre-commit transactions (e.g., create-table).
    /// The sentinel version PRE_COMMIT_VERSION indicates no version exists yet on disk.
    /// This is used to construct a pre-commit snapshot that provides table configuration
    /// (protocol, metadata, schema) for operations like CTAS.
    #[allow(dead_code)] // Used by create_table module
    pub(crate) fn for_pre_commit(table_root: Url, log_root: Url) -> Self {
        use crate::PRE_COMMIT_VERSION;
        Self {
            end_version: PRE_COMMIT_VERSION,
            checkpoint_version: None,
            log_root,
            table_root,
            checkpoint_schema: None,
            last_checkpoint_metadata: None,
            listed: LogSegmentFiles::default(),
        }
    }

    #[internal_api]
    pub(crate) fn try_new(
        mut listed_files: LogSegmentFiles,
        log_root: Url,
        end_version: Option<Version>,
        checkpoint_schema: Option<SchemaRef>,
    ) -> DeltaResult<Self> {
        // Strip "_delta_log/" from log_root to get table_root
        let table_root = {
            let log_root_str = log_root.as_str();
            if let Some(stripped) = log_root_str.strip_suffix("_delta_log/") {
                Url::parse(stripped).map_err(|e| {
                    Error::generic(format!("Failed to parse table root from log_root: {}", e))
                })?
            } else {
                return Err(Error::generic(format!(
                    "log_root does not end with '_delta_log/': {}",
                    log_root_str
                )));
            }
        };

        validate_compaction_files(&listed_files.ascending_compaction_files)?;
        validate_checkpoint_parts(&listed_files.checkpoint_parts)?;
        validate_commit_file_types(&listed_files.ascending_commit_files)?;
        validate_commit_files_contiguous(&listed_files.ascending_commit_files)?;

        // Filter commits before/at checkpoint version
        let checkpoint_version =
            if let Some(checkpoint_file) = listed_files.checkpoint_parts.first() {
                let version = checkpoint_file.version;
                listed_files
                    .ascending_commit_files
                    .retain(|log_path| version < log_path.version);
                Some(version)
            } else {
                None
            };

        validate_checkpoint_commit_gap(checkpoint_version, &listed_files.ascending_commit_files)?;
        let effective_version = validate_end_version(
            &listed_files.ascending_commit_files,
            &listed_files.checkpoint_parts,
            end_version,
        )?;

        // Build last_checkpoint_metadata from checkpoint_version and checkpoint_schema
        let last_checkpoint_metadata =
            checkpoint_version.map(|version| LastCheckpointHintSummary {
                version,
                schema: checkpoint_schema.clone(),
            });

        let log_segment = LogSegment {
            end_version: effective_version,
            checkpoint_version,
            log_root,
            table_root,
            checkpoint_schema,
            last_checkpoint_metadata,
            listed: listed_files,
        };

        info!(segment = %log_segment.summary());

        Ok(log_segment)
    }

    /// Returns the checkpoint version from the `_last_checkpoint` hint.
    #[allow(dead_code)] // used in tests
    pub(crate) fn last_checkpoint_version(&self) -> Option<Version> {
        self.checkpoint_version
    }

    /// Returns the checkpoint schema from the `_last_checkpoint` hint when it is safe to use for
    /// this segment's checkpoint parquet.
    ///
    /// Returns `None` if there is no hint, if the hint omitted `checkpointSchema`, if this segment
    /// has no checkpoint on disk, or if the hint's checkpoint version does not match this segment's
    /// checkpoint version.
    pub(crate) fn get_checkpoint_schema(&self) -> Option<SchemaRef> {
        self.checkpoint_schema.clone()
    }

    /// Returns a copy of the stored `_last_checkpoint` hint summary.
    ///
    /// Prefer [`Self::get_checkpoint_schema`] or [`Self::last_checkpoint_version`] when requiring
    /// individual values from the hint.
    #[allow(dead_code)] // used in tests
    pub(crate) fn last_checkpoint_hint_summary(&self) -> Option<LastCheckpointHintSummary> {
        self.last_checkpoint_metadata.clone()
    }

    /// Succinct summary string for logging purposes.
    fn summary(&self) -> String {
        format!(
            "{{v={}, commits={}, checkpoint_v={}, checkpoint_parts={}, compactions={}, crc_v={}, max_pub_v={}}}",
            self.end_version,
            self.listed.ascending_commit_files.len(),
            self.checkpoint_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into()),
            self.listed.checkpoint_parts.len(),
            self.listed.ascending_compaction_files.len(),
            self.listed.latest_crc_file
                .as_ref()
                .map(|f| f.version.to_string())
                .unwrap_or_else(|| "none".into()),
            self.listed.max_published_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into()),
        )
    }

    /// Constructs a [`LogSegment`] to be used for [`Snapshot`]. For a `Snapshot` at version `n`:
    /// Its LogSegment is made of zero or one checkpoint, and all commits between the checkpoint up
    /// to and including the end version `n`. Note that a checkpoint may be made of multiple
    /// parts. All these parts will have the same checkpoint version.
    ///
    /// The options for constructing a LogSegment for Snapshot are as follows:
    /// - `checkpoint_hint`: a `LastCheckpointHint` to start the log segment from (e.g. from reading the `last_checkpoint` file).
    /// - `time_travel_version`: The version of the log that the Snapshot will be at.
    ///
    /// [`Snapshot`]: crate::snapshot::Snapshot
    ///
    /// Reports metrics: `LogSegmentLoaded`.
    #[instrument(name = "log_seg.for_snap", skip_all, err)]
    #[internal_api]
    pub(crate) fn for_snapshot(
        storage: &dyn StorageHandler,
        log_root: Url,
        log_tail: Vec<ParsedLogPath>,
        time_travel_version: impl Into<Option<Version>>,
        reporter: Option<&Arc<dyn MetricsReporter>>,
        operation_id: Option<MetricId>,
    ) -> DeltaResult<Self> {
        let operation_id = operation_id.unwrap_or_default();
        let start = Instant::now();

        let time_travel_version = time_travel_version.into();
        let checkpoint_hint = LastCheckpointHint::try_read(storage, &log_root)?;
        let result = Self::for_snapshot_impl(
            storage,
            log_root,
            log_tail,
            checkpoint_hint,
            time_travel_version,
        );
        let log_segment_loading_duration = start.elapsed();

        match result {
            Ok(log_segment) => {
                reporter.inspect(|r| {
                    r.report(MetricEvent::LogSegmentLoaded {
                        operation_id,
                        duration: log_segment_loading_duration,
                        num_commit_files: log_segment.listed.ascending_commit_files.len() as u64,
                        num_checkpoint_files: log_segment.listed.checkpoint_parts.len() as u64,
                        num_compaction_files: log_segment.listed.ascending_compaction_files.len()
                            as u64,
                    });
                });
                Ok(log_segment)
            }
            Err(e) => Err(e),
        }
    }

    // factored out for testing
    pub(crate) fn for_snapshot_impl(
        storage: &dyn StorageHandler,
        log_root: Url,
        log_tail: Vec<ParsedLogPath>,
        checkpoint_hint: Option<LastCheckpointHint>,
        time_travel_version: Option<Version>,
    ) -> DeltaResult<Self> {
        // Extract checkpoint schema from hint (already an Arc, no clone needed)
        let checkpoint_schema = checkpoint_hint
            .as_ref()
            .and_then(|hint| hint.checkpoint_schema.clone());

        // The end_version is the time_travel_version, if present
        // TODO: When max catalog version is implemented, we would use that as end_version if
        // time_travel_version is not present
        let end_version = time_travel_version;

        // Keep the hint only if it points at or before end_version, or if there is no end_version bound
        let usable_hint = checkpoint_hint.filter(|cp| end_version.is_none_or(|v| cp.version <= v));

        // Cases:
        //
        // 1. usable_hint present, end_version is Some  --> list_with_checkpoint_hint from hint.version TO end_version
        // 2. usable_hint present, end_version is None  --> list_with_checkpoint_hint from hint.version unbounded
        // 3. no usable_hint,      end_version is Some  --> backward-scan for checkpoint before end_version,
        //                                                  list from that checkpoint TO end_version
        //                                                  (falls back to v0 if no checkpoint found)
        // 4. no usable_hint,      end_version is None  --> list from v0 unbounded

        let listed_files = match (usable_hint, end_version) {
            // Cases 1 and 2
            (Some(cp), end_version) => LogSegmentFiles::list_with_checkpoint_hint(
                &cp,
                storage,
                &log_root,
                log_tail,
                end_version,
            )?,
            // Case 3
            (None, Some(end)) => LogSegmentFiles::list_with_backward_checkpoint_scan(
                storage, &log_root, log_tail, end,
            )?,
            // Case 4
            (None, None) => LogSegmentFiles::list(storage, &log_root, log_tail, None, None)?,
        };

        LogSegment::try_new(
            listed_files,
            log_root,
            time_travel_version,
            checkpoint_schema,
        )
    }

    /// Constructs a [`LogSegment`] to be used for `TableChanges`. For a TableChanges between versions
    /// `start_version` and `end_version`: Its LogSegment is made of zero checkpoints and all commits
    /// between versions `start_version` (inclusive) and `end_version` (inclusive). If no `end_version`
    /// is specified it will be the most recent version by default.
    #[internal_api]
    pub(crate) fn for_table_changes(
        storage: &dyn StorageHandler,
        log_root: Url,
        start_version: Version,
        end_version: impl Into<Option<Version>>,
    ) -> DeltaResult<Self> {
        let end_version = end_version.into();
        if let Some(end_version) = end_version {
            if start_version > end_version {
                return Err(Error::generic(
                    "Failed to build LogSegment: start_version cannot be greater than end_version",
                ));
            }
        }

        // TODO: compactions?
        let listed_files =
            LogSegmentFiles::list_commits(storage, &log_root, Some(start_version), end_version)?;
        // - Here check that the start version is correct.
        // - [`LogSegment::try_new`] will verify that the `end_version` is correct if present.
        // - [`LogSegmentFiles::list_commits`] also checks that there are no gaps between commits.
        // If all three are satisfied, this implies that all the desired commits are present.
        require!(
            listed_files
                .ascending_commit_files()
                .first()
                .is_some_and(|first_commit| first_commit.version == start_version),
            Error::generic(format!(
                "Expected the first commit to have version {start_version}, got {:?}",
                listed_files
                    .ascending_commit_files()
                    .first()
                    .map(|c| c.version)
            ))
        );
        LogSegment::try_new(listed_files, log_root, end_version, None)
    }

    #[allow(unused)]
    /// Constructs a [`LogSegment`] to be used for timestamp conversion. This [`LogSegment`] will
    /// consist only of contiguous commit files up to `end_version` (inclusive). If present,
    /// `limit` specifies the maximum length of the returned log segment. The log segment may be
    /// shorter than `limit` if there are missing commits.
    ///
    // This lists all files starting from `end-limit` if `limit` is defined. For large tables,
    // listing with a `limit` can be a significant speedup over listing _all_ the files in the log.
    pub(crate) fn for_timestamp_conversion(
        storage: &dyn StorageHandler,
        log_root: Url,
        end_version: Version,
        limit: Option<NonZero<usize>>,
    ) -> DeltaResult<Self> {
        // Compute the version to start listing from.
        let start_from = limit
            .map(|limit| match NonZero::<Version>::try_from(limit) {
                Ok(limit) => Ok(Version::saturating_sub(end_version, limit.get() - 1)),
                _ => Err(Error::generic(format!(
                    "Invalid limit {limit} when building log segment in timestamp conversion",
                ))),
            })
            .transpose()?;

        // this is a list of commits with possible gaps, we want to take the latest contiguous
        // chunk of commits
        let mut listed_commits =
            LogSegmentFiles::list_commits(storage, &log_root, start_from, Some(end_version))?;

        // remove gaps - return latest contiguous chunk of commits
        let commits = listed_commits.ascending_commit_files_mut();
        if !commits.is_empty() {
            let mut start_idx = commits.len() - 1;
            while start_idx > 0 && commits[start_idx].version == 1 + commits[start_idx - 1].version
            {
                start_idx -= 1;
            }
            commits.drain(..start_idx);
        }

        LogSegment::try_new(listed_commits, log_root, Some(end_version), None)
    }

    /// Creates a new LogSegment with the given commit file added to the end.
    /// TODO: Take in multiple commits when Kernel-RS supports txn retries and conflict rebasing.
    #[allow(unused)]
    pub(crate) fn new_with_commit_appended(
        &self,
        tail_commit_file: ParsedLogPath,
    ) -> DeltaResult<Self> {
        require!(
            tail_commit_file.is_commit(),
            Error::internal_error(format!(
                "Cannot extend and create new LogSegment. Tail log file is not a commit file. \
                Path: {}, Type: {:?}.",
                tail_commit_file.location.location, tail_commit_file.file_type
            ))
        );
        require!(
            tail_commit_file.version == self.end_version.wrapping_add(1),
            Error::internal_error(format!(
                "Cannot extend and create new LogSegment. Tail commit file version ({}) does not \
                equal LogSegment end_version ({}) + 1.",
                tail_commit_file.version, self.end_version
            ))
        );

        let mut new_log_segment = self.clone();

        new_log_segment.end_version = tail_commit_file.version;
        new_log_segment
            .listed
            .ascending_commit_files
            .push(tail_commit_file.clone());
        new_log_segment.listed.latest_commit_file = Some(tail_commit_file.clone());
        new_log_segment.listed.max_published_version = match tail_commit_file.file_type {
            LogPathFileType::Commit => Some(tail_commit_file.version),
            _ => self.listed.max_published_version,
        };

        Ok(new_log_segment)
    }

    /// Creates a new LogSegment reflecting a checkpoint written at this segment's version.
    /// The checkpoint must be at `end_version`. Kernel does not write multi-part checkpoints,
    /// so the checkpoint must be a single file (classic parquet or V2 UUID).
    pub(crate) fn try_new_with_checkpoint(&self, checkpoint: ParsedLogPath) -> DeltaResult<Self> {
        require!(
            matches!(
                checkpoint.file_type,
                LogPathFileType::SinglePartCheckpoint | LogPathFileType::UuidCheckpoint
            ),
            Error::internal_error(format!(
                "Cannot update LogSegment with checkpoint. Path is not a single-file \
                checkpoint. Path: {}, Type: {:?}.",
                checkpoint.location.location, checkpoint.file_type
            ))
        );
        require!(
            checkpoint.version == self.end_version,
            Error::internal_error(format!(
                "Cannot update LogSegment with checkpoint. Checkpoint version ({}) does not \
                equal LogSegment end_version ({}).",
                checkpoint.version, self.end_version
            ))
        );

        let mut new_log_segment = self.clone();
        new_log_segment.checkpoint_version = Some(checkpoint.version);
        new_log_segment.listed.checkpoint_parts = vec![checkpoint];
        // A snapshot at version N only contains commits and compactions at versions <= N,
        // so a checkpoint at N covers everything and we can clear them entirely.
        new_log_segment.listed.ascending_commit_files.clear();
        new_log_segment.listed.ascending_compaction_files.clear();
        // TODO(#839): Once CheckpointWriter exposes the output schema, build a LastCheckpointHintSummary
        // and thread it through here instead of None. Today the schema is computed inside checkpoint_data()
        // but not returned. With None, the next scan will read the checkpoint parquet footer
        // to determine the schema (e.g. whether stats_parsed or sidecar columns exist).
        new_log_segment.last_checkpoint_metadata = None;
        Ok(new_log_segment)
    }

    /// Creates a new LogSegment with the given CRC file recorded as the latest.
    /// The CRC file must be at `end_version`.
    pub(crate) fn try_new_with_crc_file(&self, crc_file: ParsedLogPath<Url>) -> DeltaResult<Self> {
        require!(
            crc_file.file_type == LogPathFileType::Crc,
            Error::internal_error(format!(
                "Cannot update LogSegment with CRC. Path is not a CRC file. \
                Path: {}, Type: {:?}.",
                crc_file.location, crc_file.file_type
            ))
        );
        require!(
            crc_file.version == self.end_version,
            Error::internal_error(format!(
                "Cannot update LogSegment with CRC. CRC version ({}) does not \
                equal LogSegment end_version ({}).",
                crc_file.version, self.end_version
            ))
        );
        // Convert to FileMeta with placeholder metadata (size=0, last_modified=0).
        // Only the URL matters for CRC files: downstream code uses it for version
        // tracking and reading CRC content via `try_read_crc_file`. Neither `size`
        // nor `last_modified` is ever accessed.
        let crc_file = ParsedLogPath {
            location: FileMeta {
                location: crc_file.location,
                last_modified: 0,
                size: 0,
            },
            filename: crc_file.filename,
            extension: crc_file.extension,
            version: crc_file.version,
            file_type: crc_file.file_type,
        };
        let mut new_log_segment = self.clone();
        new_log_segment.listed.latest_crc_file = Some(crc_file);
        Ok(new_log_segment)
    }

    pub(crate) fn new_as_published(&self) -> DeltaResult<Self> {
        // In the future, we can additionally convert the staged commit files to published commit
        // files. That would reqire faking their FileMeta locations.
        let mut new_log_segment = self.clone();
        new_log_segment.listed.max_published_version = Some(self.end_version);
        Ok(new_log_segment)
    }

    pub(crate) fn get_unpublished_catalog_commits(&self) -> DeltaResult<Vec<CatalogCommit>> {
        self.listed
            .ascending_commit_files
            .iter()
            .filter(|file| file.file_type == LogPathFileType::StagedCommit)
            .filter(|file| {
                self.listed
                    .max_published_version
                    .is_none_or(|v| file.version > v)
            })
            .map(|file| CatalogCommit::try_new(&self.log_root, file))
            .collect()
    }

    /// Read a stream of actions from this log segment. This returns an iterator of
    /// [`ActionsBatch`]s which includes EngineData of actions + a boolean flag indicating whether
    /// the data was read from a commit file (true) or a checkpoint file (false).
    ///
    /// The log files will be read from most recent to oldest.
    ///
    /// `commit_read_schema` is the (physical) schema to read the commit files with, and
    /// `checkpoint_read_schema` is the (physical) schema to read checkpoint files with. This can be
    /// used to project the log files to a subset of the columns. Having two different
    /// schemas can be useful as a cheap way of doing additional filtering on the checkpoint files
    /// (e.g. filtering out remove actions).
    ///
    ///  The engine data returned might have extra non-log actions (e.g. sidecar
    ///  actions) that are not part of the schema but this is an implementation
    ///  detail that should not be relied on and will likely change.
    ///
    /// Read a stream of actions from this log segment. This returns an iterator of
    /// [`ActionsBatch`]s which includes EngineData of actions + a boolean flag indicating whether
    /// the data was read from a commit file (true) or a checkpoint file (false).
    ///
    /// Also returns:
    /// - `Option<bool>` indicating if checkpoint has compatible stats_parsed
    /// - The checkpoint read schema (with stats_parsed if compatible)
    ///
    /// # Parameters
    /// - `data_predicate`: Optional predicate for manifest-level data skipping. When reading from
    ///   a content root with hierarchical manifests, this predicate is used to skip child manifests
    ///   whose `content_stats` indicate they cannot contain matching data.
    /// - `skip_leaf_manifests`: When true, skips reading from the content root (leaf manifests).
    ///   Only root manifest + delta log will be read. Used by Transaction::release_root_and_delta_actions().
    ///
    /// Also returns `CheckpointReadInfo` with stats_parsed compatibility and the checkpoint schema.
    ///
    /// `meta_predicate` is an optional expression for row group skipping in checkpoint parquet
    /// files. It is _NOT_ the query's data predicate, but a hint for skipping irrelevant data.
    /// IS NOT NULL predicates are automatically derived from `checkpoint_read_schema` and combined
    /// (AND) with `meta_predicate`, so callers only need to supply query-based skipping predicates.
    #[internal_api]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_actions_with_projected_checkpoint_actions(
        &self,
        engine: &dyn Engine,
        commit_read_schema: SchemaRef,
        checkpoint_read_schema: SchemaRef,
        meta_predicate: Option<PredicateRef>,
        stats_schema: Option<&StructType>,
        partition_schema: Option<&StructType>,
        data_predicate: Option<PredicateRef>,
        checkpoint_action: Option<&CheckpointAction>,
        skip_leaf_manifests: bool,
        table_schema: Option<&StructType>,
    ) -> DeltaResult<
        ActionsWithCheckpointInfo<impl Iterator<Item = DeltaResult<ActionsBatch>> + Send>,
    > {
        // checkpoint_action is now passed in from caller (no I/O needed)
        let checkpoint_action_version = checkpoint_action.map(|ca| ca.version);
        let content_root = checkpoint_action.map(|ca| &ca.content_root);

        // Combine schema-derived IS NOT NULL predicate with any caller-supplied predicate so
        // checkpoint parquet row groups without any relevant action type can be skipped.
        // TODO: The semantics of `meta_predicate` will change in a follow-up PR.
        let is_not_null_pred = schema_to_is_not_null_predicate(&checkpoint_read_schema);
        let effective_predicate = match (is_not_null_pred, meta_predicate) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(Arc::new(Predicate::and((*a).clone(), (*b).clone()))),
        };

        let actions_with_checkpoint_info = self.create_checkpoint_stream(
            engine,
            checkpoint_read_schema,
            effective_predicate,
            stats_schema,
            partition_schema,
            content_root,
            data_predicate,
            skip_leaf_manifests,
            table_schema,
        )?;

        // `replay` expects commit files to be sorted in descending order, so the return value here is correct
        let commit_stream =
            CommitReader::try_new(engine, self, commit_read_schema, checkpoint_action_version)?;

        Ok(ActionsWithCheckpointInfo {
            actions: commit_stream.chain(actions_with_checkpoint_info.actions),
            checkpoint_info: actions_with_checkpoint_info.checkpoint_info,
        })
    }

    fn remove_file_actions_from_schema(schema: SchemaRef) -> DeltaResult<SchemaRef> {
        let file_action_names = [ADD_NAME, REMOVE_NAME, SIDECAR_NAME];
        let non_file_action_names = schema
            .field_names()
            .filter(|name| !file_action_names.contains(&name.as_ref()))
            .collect::<Vec<_>>();
        schema.project(&non_file_action_names)
    }

    /// Same as [`Self::read_actions_with_projected_checkpoint_actions`], but uses the same schema
    /// for reading checkpoints and commits. IS NOT NULL predicates are automatically derived from
    /// the schema, so callers do not need to supply them.
    #[internal_api]
    pub(crate) fn read_actions(
        &self,
        engine: &dyn Engine,
        action_schema: SchemaRef,
    ) -> DeltaResult<impl Iterator<Item = DeltaResult<ActionsBatch>> + Send> {
        let action_with_checkpoint_info = self.read_actions_with_projected_checkpoint_actions(
            engine,
            action_schema.clone(),
            action_schema,
            None,
            None,
            None,
            None,  // No data predicate for manifest-level skipping
            None,  // No checkpoint action available in this context
            false, // Don't skip leaf manifests by default
            None,  // No table schema available in this context
        )?;
        Ok(action_with_checkpoint_info.actions)
    }

    /// find a minimal set to cover the range of commits we want. This is greedy so not always
    /// optimal, but we assume there are rarely overlapping compactions so this is okay. NB: This
    /// returns files is DESCENDING ORDER, as that's what `replay` expects. This function assumes
    /// that all files in `self.ascending_commit_files` and `self.ascending_compaction_files` are in
    /// range for this log segment. This invariant is maintained by our listing code.
    pub(crate) fn find_commit_cover(
        &self,
        commit_read_schema: SchemaRef,
        meta_predicate: Option<PredicateRef>,
        content_root_version: Option<Version>,
    ) -> DeltaResult<Vec<PartialCommitCover>> {
        // Create an iterator sorted in ascending order by (initial version, end version), e.g.
        // [00.json, 00.09.compacted.json, 00.99.compacted.json, 01.json, 02.json, ..., 10.json,
        //  10.19.compacted.json, 11.json, ...]
        let all_files = itertools::Itertools::merge_by(
            self.listed.ascending_commit_files.iter(),
            self.listed.ascending_compaction_files.iter(),
            |path_a, path_b| path_a.version <= path_b.version,
        );

        let mut last_pushed: Option<&ParsedLogPath> = None;

        let mut commit_covers = vec![];
        let mut selected_files = vec![];
        // Only be careful with content root version if we are reading add/remove actions since that is all it contains.
        let mut read_schema = commit_read_schema.clone();

        let content_root_version = if content_root_version.is_some()
            && (commit_read_schema.contains(ADD_NAME) || commit_read_schema.contains(REMOVE_NAME))
        {
            read_schema = Self::remove_file_actions_from_schema(commit_read_schema.clone())?;
            // TODO: Adapt the meta_predicate also if it references file actions. Today this isn't the case.
            content_root_version
        } else {
            None
        };
        // If the read schema has no fields, we can skip all files before the content root version
        // because there are no add/remove actions relevant before it, and no other types requested.
        let skip_before_version =
            if content_root_version.is_some() && read_schema.fields().len() == 0 {
                content_root_version
            } else {
                None
            };
        for next in all_files {
            if skip_before_version.is_some_and(|v| next.version <= v) {
                continue;
            }
            match last_pushed {
                // Resolve version number ties in favor of the later file (it covers a wider range)
                Some(prev) if prev.version == next.version => {
                    let removed = selected_files.pop();
                    debug!("Selecting {next:?} rather than {removed:?}, it covers a wider range");
                }
                // Skip later files whose start overlaps with the previous end
                Some(&ParsedLogPath {
                    file_type: LogPathFileType::CompactedCommit { hi },
                    ..
                }) if next.version <= hi => {
                    debug!("Skipping log file {next:?}, it's already covered.");
                    continue;
                }
                _ => {} // just fall through
            }
            if let Some(root_version) = content_root_version {
                if let LogPathFileType::CompactedCommit { hi } = next.file_type {
                    if root_version >= next.version && root_version <= hi {
                        // For now skip over compactions that include the root to avoid having to deal
                        // with edge cases of mixed add/remove file actions.
                        debug!("Skipping log file {next:?}, it overlaps with the latest content root file.");
                        continue;
                    }
                }
                // Since overlapping compactions are skipped above, the only overlapping file
                // will be a commit.
                if let LogPathFileType::Commit = next.file_type {
                    if root_version == next.version {
                        require!(
                            commit_covers.is_empty(),
                            Error::generic("Expected no commit covers before adding a new one")
                        );
                        selected_files.reverse();
                        let copied_files = selected_files.clone();
                        selected_files = vec![];
                        commit_covers.push(PartialCommitCover {
                            files: copied_files,
                            meta_predicate: meta_predicate.clone(),
                            read_schema: read_schema.clone(),
                            max_published_version: self.listed.max_published_version,
                        });
                    }
                    // Reset back to the full schema for any commits after the root to ensure no
                    // file actions are lost.
                    read_schema = commit_read_schema.clone();
                }
            }
            debug!("Provisionally selecting {next:?}");
            last_pushed = Some(next);
            selected_files.push(next.location.clone());
        }

        selected_files.reverse();
        commit_covers.push(PartialCommitCover {
            files: selected_files,
            meta_predicate: meta_predicate.clone(),
            read_schema: commit_read_schema.clone(),
            max_published_version: self.listed.max_published_version,
        });
        commit_covers.reverse();
        Ok(commit_covers)
    }

    /// Creates an iterator over action batches from a content root (AMT manifest).
    ///
    /// This opens the parquet I/O stream immediately but defers metadata construction
    /// and batch processing until the iterator is consumed.
    ///
    /// # Parameters
    /// - `data_predicate`: Optional predicate for manifest-level data skipping. When provided,
    ///   child manifests whose `content_stats` indicate they cannot contain matching data
    ///   will be skipped (not opened).
    /// - `skip_leaf_manifests`: When true, only read the root manifest, not the leaf manifests.
    #[allow(clippy::too_many_arguments)]
    fn create_content_root_reader(
        engine: &dyn Engine,
        content_root: &ContentRoot,
        checkpoint_read_schema: SchemaRef,
        table_root: &Url,
        data_predicate: Option<PredicateRef>,
        skip_leaf_manifests: bool,
        stats_schema: Option<&StructType>,
        table_schema: Option<&StructType>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>> {
        let content_root_url = table_root
            .join(&content_root.path)
            .map_err(|e| Error::generic(format!("Failed to parse content root URL: {}", e)))?;

        // Create lazy iterator that opens the stream and defers processing
        // Stats schema and table schema are passed for AMT content_stats reading and data skipping
        let lazy_iter =
            crate::content_tree::lazy_reader::LazyContentRootIterator::from_content_root(
                engine.parquet_handler(),
                engine.evaluation_handler(),
                &content_root_url,
                content_root.path.clone(),
                table_root.clone(),
                checkpoint_read_schema,
                data_predicate,
                skip_leaf_manifests,
                stats_schema,
                table_schema,
            )?;

        Ok(Box::new(lazy_iter))
    }

    /// Determines the file actions schema and extracts sidecar file references for checkpoints.
    ///
    /// This function analyzes the checkpoint to determine:
    /// 1. The file actions schema (for stats_parsed / partitionValues_parsed detection)
    /// 2. Sidecar file references if this is a V2 checkpoint
    ///
    /// The logic is:
    /// - No checkpoint parts: return (None, [])
    /// - Multi-part (always V1, no sidecars): return checkpoint schema directly
    /// - UUID-named JSON (always V2): extract sidecars, read first sidecar's schema
    /// - Classic-named or UUID-named parquet (V1 or V2): read checkpoint schema from
    ///   hint or footer, then check for sidecar column to distinguish
    ///   - Has sidecar column (V2): extract sidecars, read first sidecar's schema
    ///   - No sidecar column (V1): use checkpoint schema directly
    fn get_file_actions_schema_and_sidecars(
        &self,
        engine: &dyn Engine,
    ) -> DeltaResult<(Option<SchemaRef>, Vec<FileMeta>)> {
        // Hint schema from `_last_checkpoint` avoids footer reads when available.
        let hint_schema = self.get_checkpoint_schema();

        // All parts of a multi-part checkpoint belong to the same table version and follow
        // the same V1 spec, so reading any one part's schema is sufficient.
        let Some(checkpoint) = self.listed.checkpoint_parts.first() else {
            return Ok((None, vec![]));
        };

        match &checkpoint.file_type {
            MultiPartCheckpoint { .. } => {
                // Multi-part checkpoints are always V1 and never have sidecars.
                let schema =
                    Self::read_checkpoint_schema(engine, checkpoint, hint_schema.as_ref())?;
                Ok((Some(schema), vec![]))
            }
            UuidCheckpoint if checkpoint.extension.as_str() == "json" => {
                // JSON checkpoint is always V2. No checkpoint schema is available since JSON
                // checkpoints don't have a parquet footer to read.
                self.read_sidecar_schema_and_files(engine, checkpoint, None)
            }
            SinglePartCheckpoint | UuidCheckpoint if checkpoint.extension.as_str() == "parquet" => {
                // Parquet checkpoint (classic-named or UUID-named): either can be V1 or V2.
                // Check for sidecar column to distinguish.
                let checkpoint_schema =
                    Self::read_checkpoint_schema(engine, checkpoint, hint_schema.as_ref())?;
                if checkpoint_schema.field(SIDECAR_NAME).is_some() {
                    self.read_sidecar_schema_and_files(engine, checkpoint, Some(&checkpoint_schema))
                } else {
                    Ok((Some(checkpoint_schema), vec![]))
                }
            }
            _ => Ok((None, vec![])),
        }
    }

    /// Returns the checkpoint's parquet schema, using the hint from `_last_checkpoint` if
    /// available or reading the parquet footer otherwise.
    fn read_checkpoint_schema(
        engine: &dyn Engine,
        checkpoint: &ParsedLogPath<FileMeta>,
        hint_schema: Option<&SchemaRef>,
    ) -> DeltaResult<SchemaRef> {
        match hint_schema {
            Some(schema) => Ok(schema.clone()),
            None => Ok(engine
                .parquet_handler()
                .read_parquet_footer(&checkpoint.location)?
                .schema),
        }
    }

    /// Extracts sidecar file references and reads the file actions schema from the first
    /// sidecar's parquet footer. If no sidecars exist, falls back to `checkpoint_schema`
    /// since V2 checkpoints may store add actions directly in the main file.
    fn read_sidecar_schema_and_files(
        &self,
        engine: &dyn Engine,
        checkpoint: &ParsedLogPath<FileMeta>,
        checkpoint_schema: Option<&SchemaRef>,
    ) -> DeltaResult<(Option<SchemaRef>, Vec<FileMeta>)> {
        let sidecar_files = self.extract_sidecar_refs(engine, checkpoint)?;
        let file_actions_schema = match sidecar_files.first() {
            Some(first) => Some(engine.parquet_handler().read_parquet_footer(first)?.schema),
            None => checkpoint_schema.cloned(),
        };
        Ok((file_actions_schema, sidecar_files))
    }

    /// Returns an iterator over checkpoint data, processing sidecar files when necessary.
    ///
    /// For checkpoints that need file actions, this function:
    /// 1. Determines the file actions schema (for stats_parsed / partitionValues_parsed detection)
    /// 2. Extracts sidecar file references if present (V2 checkpoints)
    /// 3. Reads checkpoint and sidecar data using cached sidecar refs
    ///
    /// Returns a tuple of:
    /// - Iterator over action batches from checkpoint and sidecar files
    /// - `Option<bool>` indicating if checkpoint has compatible stats_parsed
    /// - The checkpoint read schema (with stats_parsed if compatible)
    ///
    /// # Parameters
    /// - `data_predicate`: Optional predicate for manifest-level data skipping when reading
    ///   from a content root with hierarchical manifests.
    /// - `skip_leaf_manifests`: When true, don't read from the content root's leaf manifests.
    #[allow(clippy::too_many_arguments)]
    fn create_checkpoint_stream(
        &self,
        engine: &dyn Engine,
        action_schema: SchemaRef,
        meta_predicate: Option<PredicateRef>,
        stats_schema: Option<&StructType>,
        partition_schema: Option<&StructType>,
        content_root: Option<&ContentRoot>,
        data_predicate: Option<PredicateRef>,
        skip_leaf_manifests: bool,
        table_schema: Option<&StructType>,
    ) -> DeltaResult<
        ActionsWithCheckpointInfo<impl Iterator<Item = DeltaResult<ActionsBatch>> + Send>,
    > {
        let need_file_actions = schema_contains_file_actions(&action_schema);

        // Read the content root file if it exists and file actions are necessary.
        // The skip_leaf_manifests flag controls whether we read leaf manifests or just the root.
        // The content root serves the same purpose as a checkpoint file for file actions,
        // so remove file actions from the schema if they are present for actually reading
        // the checkpoint files.
        // If stats_schema is provided, extend action_schema to include stats_parsed for content root
        let action_schema_for_content_root = if let Some(stats_schema) = stats_schema {
            use crate::actions::{Add, ADD_NAME};
            use crate::schema::{DataType, StructField, StructType};

            // Get the base Add schema and add stats_parsed to it
            let add_struct = Add::to_schema();
            let mut add_fields: Vec<StructField> = add_struct.fields().cloned().collect();
            add_fields.push(StructField::nullable(
                "stats_parsed",
                DataType::Struct(Box::new(stats_schema.clone())),
            ));

            Arc::new(StructType::new_unchecked([StructField::nullable(
                ADD_NAME,
                StructType::new_unchecked(add_fields),
            )]))
        } else {
            action_schema.clone()
        };

        let (content_root_stream, read_schema): (
            Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>,
            SchemaRef,
        ) = if let Some(cr) = content_root.filter(|_| need_file_actions) {
            (
                Self::create_content_root_reader(
                    engine,
                    cr,
                    action_schema_for_content_root,
                    &self.table_root,
                    data_predicate,
                    skip_leaf_manifests,
                    stats_schema,
                    table_schema,
                )?,
                Self::remove_file_actions_from_schema(action_schema.clone())?,
            )
        } else {
            (Box::new(std::iter::empty()), action_schema.clone())
        };

        let (file_actions_schema, sidecar_files) = if need_file_actions {
            self.get_file_actions_schema_and_sidecars(engine)?
        } else {
            (None, vec![])
        };

        // Check if checkpoint has compatible stats_parsed and add it to the schema if so
        // Also check if content root is present - it always outputs stats_parsed when stats_schema is Some
        let has_stats_parsed = stats_schema.is_some_and(|stats| {
            // Content root always outputs stats_parsed when stats_schema is provided
            let content_root_has_stats = content_root.is_some() && need_file_actions;

            // Checkpoint/sidecars have stats_parsed if their schema is compatible
            let checkpoint_has_stats = file_actions_schema.as_ref().is_some_and(|file_schema| {
                Self::schema_has_compatible_stats_parsed(file_schema, stats)
            });

            content_root_has_stats || checkpoint_has_stats
        });

        let has_partition_values_parsed = partition_schema
            .zip(file_actions_schema.as_ref())
            .is_some_and(|(ps, fs)| Self::schema_has_compatible_partition_values_parsed(fs, ps));

        // Build final schema with any additional fields needed
        // (stats_parsed, partitionValues_parsed, sidecar)
        let needs_sidecar = need_file_actions && !sidecar_files.is_empty();
        let needs_add_augmentation = has_stats_parsed || has_partition_values_parsed;
        let augmented_checkpoint_read_schema = if needs_add_augmentation || needs_sidecar {
            let mut new_fields: Vec<StructField> = if let (true, Some(add_field)) =
                (needs_add_augmentation, action_schema.field("add"))
            {
                let DataType::Struct(add_struct) = add_field.data_type() else {
                    return Err(Error::internal_error(
                        "add field in action schema must be a struct",
                    ));
                };
                let mut add_fields: Vec<StructField> = add_struct.fields().cloned().collect();
                if let (true, Some(ss)) = (has_stats_parsed, stats_schema) {
                    add_fields.push(StructField::nullable(
                        "stats_parsed",
                        DataType::Struct(Box::new(ss.clone())),
                    ));
                }

                if let (true, Some(ps)) = (has_partition_values_parsed, partition_schema) {
                    add_fields.push(StructField::nullable(
                        "partitionValues_parsed",
                        DataType::Struct(Box::new(ps.clone())),
                    ));
                }

                // Rebuild schema with modified add field
                action_schema
                    .fields()
                    .map(|f| {
                        if f.name() == "add" {
                            StructField::new(
                                add_field.name(),
                                StructType::new_unchecked(add_fields.clone()),
                                add_field.is_nullable(),
                            )
                            .with_metadata(add_field.metadata.clone())
                        } else {
                            f.clone()
                        }
                    })
                    .collect()
            } else {
                action_schema.fields().cloned().collect()
            };

            // Add sidecar column at top-level for V2 checkpoints
            if needs_sidecar {
                new_fields.push(StructField::nullable(SIDECAR_NAME, Sidecar::to_schema()));
            }

            Arc::new(StructType::new_unchecked(new_fields))
        } else {
            // No modifications needed, use schema as-is
            action_schema.clone()
        };

        let checkpoint_info = CheckpointReadInfo {
            has_stats_parsed,
            has_partition_values_parsed,
            checkpoint_read_schema: augmented_checkpoint_read_schema.clone(),
        };

        if read_schema.fields().len() == 0 {
            return Ok(ActionsWithCheckpointInfo {
                actions: content_root_stream,
                checkpoint_info,
            });
        }

        let checkpoint_file_meta: Vec<_> = self
            .listed
            .checkpoint_parts
            .iter()
            .map(|f| f.location.clone())
            .collect();

        let parquet_handler = engine.parquet_handler();

        // Historically, we had a shared file reader trait for JSON and Parquet handlers,
        // but it was removed to avoid unnecessary coupling. This is a concrete case
        // where it *could* have been useful, but for now, we're keeping them separate.
        // If similar patterns start appearing elsewhere, we should reconsider that decision.
        let actions = match self.listed.checkpoint_parts.first() {
            Some(parsed_log_path) if parsed_log_path.extension == "json" => {
                engine.json_handler().read_json_files(
                    &checkpoint_file_meta,
                    augmented_checkpoint_read_schema.clone(),
                    meta_predicate.clone(),
                )?
            }
            Some(parsed_log_path) if parsed_log_path.extension == "parquet" => parquet_handler
                .read_parquet_files(
                    &checkpoint_file_meta,
                    augmented_checkpoint_read_schema.clone(),
                    meta_predicate.clone(),
                )?,
            Some(parsed_log_path) => {
                return Err(Error::generic(format!(
                    "Unsupported checkpoint file type: {}",
                    parsed_log_path.extension,
                )));
            }
            // This is the case when there are no checkpoints in the log segment
            // so we return an empty iterator
            None => Box::new(std::iter::empty()),
        };

        // Read sidecars with the same schema as checkpoint (including stats_parsed if available).
        // The sidecar column will be null in sidecar batches, which is harmless.
        // Both checkpoint and sidecar parquet files share the same `add.stats_parsed.*` column
        // layout, so we reuse the same predicate for row group skipping.
        let sidecar_batches = if !sidecar_files.is_empty() {
            parquet_handler.read_parquet_files(
                &sidecar_files,
                augmented_checkpoint_read_schema.clone(),
                meta_predicate,
            )?
        } else {
            Box::new(std::iter::empty())
        };

        // Chain checkpoint batches with sidecar batches.
        // The boolean flag indicates whether the batch originated from a commit file
        // (true) or a checkpoint file (false).
        let actions_iter = actions
            .map_ok(|batch| ActionsBatch::new(batch, false))
            .chain(sidecar_batches.map_ok(|batch| ActionsBatch::new(batch, false)));

        Ok(ActionsWithCheckpointInfo {
            actions: Box::new(actions_iter),
            checkpoint_info,
        })
    }

    /// Extracts sidecar file references from a checkpoint file.
    fn extract_sidecar_refs(
        &self,
        engine: &dyn Engine,
        checkpoint: &ParsedLogPath,
    ) -> DeltaResult<Vec<FileMeta>> {
        // Read checkpoint with just the sidecar column
        let batches = match checkpoint.extension.as_str() {
            "json" => engine.json_handler().read_json_files(
                std::slice::from_ref(&checkpoint.location),
                Self::sidecar_read_schema(),
                None,
            )?,
            "parquet" => engine.parquet_handler().read_parquet_files(
                std::slice::from_ref(&checkpoint.location),
                Self::sidecar_read_schema(),
                None,
            )?,
            _ => return Ok(vec![]),
        };

        // Extract sidecar file references
        let mut visitor = SidecarVisitor::default();
        for batch_result in batches {
            let batch = batch_result?;
            visitor.visit_rows_of(batch.as_ref())?;
        }

        // Convert to FileMeta
        visitor
            .sidecars
            .iter()
            .map(|sidecar| sidecar.to_filemeta(&self.log_root))
            .try_collect()
    }
}

impl LogSegment {
    /// Validate checkpoint action compatibility with protocol and update root enabled state.
    ///
    /// When a protocol is discovered, this checks that if the protocol lacks the
    /// MetadataTreeExperimental feature, no checkpoint action should have been found.
    /// Updates the root_enabled flag based on the protocol's features.
    fn validate_checkpoint_action_with_protocol(
        protocol: &Protocol,
        checkpoint_action_opt: &Option<CheckpointAction>,
        root_enabled: &mut bool,
    ) -> DeltaResult<()> {
        *root_enabled = protocol.has_reader_feature(&TableFeature::MetadataTreeExperimental);

        // If protocol lacks the feature but we already found a checkpoint action, that's invalid
        if !*root_enabled && checkpoint_action_opt.is_some() {
            return Err(Error::invalid_protocol(
                "Found checkpoint action but protocol does not have MetadataTreeExperimental reader feature enabled"
            ));
        }

        Ok(())
    }

    /// Read protocol, metadata, and checkpoint action from the log segment.
    ///
    /// If an existing_protocol is provided, it will be used to determine the initial
    /// checkpoint action search state. If the protocol doesn't have the MetadataTreeExperimental
    /// feature, checkpoint action search will be skipped entirely. If no existing_protocol is
    /// provided, the search will start optimistically and adjust once protocol is discovered.
    ///
    /// The `lazy_crc` parameter is used for the CRC-optimized P&M path. Callers that want the
    /// CRC to be cached on a shared [`LazyCrc`] instance (e.g. the snapshot's own `lazy_crc`)
    /// should pass that instance here.
    pub(crate) fn protocol_and_metadata_and_checkpoint(
        &self,
        engine: &dyn Engine,
        existing_protocol: Option<&Protocol>,
        lazy_crc: &LazyCrc,
    ) -> DeltaResult<(Option<Metadata>, Option<Protocol>, Option<CheckpointAction>)> {
        // Try CRC-optimized path for P&M
        let (mut metadata_opt, mut protocol_opt) =
            self.read_protocol_metadata_opt(engine, lazy_crc)?;

        // Determine if checkpoint action is enabled based on found or existing protocol
        let effective_protocol = protocol_opt.as_ref().or(existing_protocol);
        let mut root_enabled = match effective_protocol {
            Some(protocol) => protocol.has_reader_feature(&TableFeature::MetadataTreeExperimental),
            None => {
                // No existing protocol - start optimistically
                true
            }
        };

        // If P&M already found and no checkpoint action needed, return early
        if metadata_opt.is_some() && protocol_opt.is_some() && !root_enabled {
            return Ok((metadata_opt, protocol_opt, None));
        }

        // Need to search for checkpoint action (and possibly remaining P&M if not found).
        // Do full replay, skipping P&M search if already populated from CRC.
        let root_enabled_at_start = root_enabled;
        let mut checkpoint_action_opt = None;

        for actions_batch in self.replay_for_pmc(engine)? {
            let actions = actions_batch?.actions;

            // Search for Protocol and Metadata if not already found via CRC
            if metadata_opt.is_none() {
                metadata_opt = Metadata::try_new_from_data(actions.as_ref())?;
            }

            if protocol_opt.is_none() {
                protocol_opt = Protocol::try_new_from_data(actions.as_ref())?;
                if let Some(protocol) = protocol_opt.as_ref() {
                    Self::validate_checkpoint_action_with_protocol(
                        protocol,
                        &checkpoint_action_opt,
                        &mut root_enabled,
                    )?;
                }
            }

            // Only search for checkpoint action if enabled
            if root_enabled && checkpoint_action_opt.is_none() {
                checkpoint_action_opt = CheckpointAction::try_new_from_data(actions.as_ref())?;

                // Extract nested P+M from checkpoint action if top-level ones were not found.
                if let Some(ref ca) = checkpoint_action_opt {
                    let had_protocol = protocol_opt.is_some();
                    ca.fill_missing_pm(&mut protocol_opt, &mut metadata_opt);
                    if !had_protocol {
                        if let Some(protocol) = protocol_opt.as_ref() {
                            Self::validate_checkpoint_action_with_protocol(
                                protocol,
                                &checkpoint_action_opt,
                                &mut root_enabled,
                            )?;
                        }
                    }
                }
            }

            // Early termination: stop when we have everything we need
            if metadata_opt.is_some() && protocol_opt.is_some() {
                // If checkpoint action is disabled, we're done
                if !root_enabled {
                    break;
                }

                // If checkpoint action is enabled:
                // - If we found checkpoint action, we're done
                // - If checkpoint action was just enabled (wasn't enabled at start), we're done
                //   (feature was just turned on, no checkpoint action written yet)
                // - Otherwise keep searching (was enabled at start, should exist)
                if checkpoint_action_opt.is_some() || !root_enabled_at_start {
                    break;
                }
            }
        }

        Ok((metadata_opt, protocol_opt, checkpoint_action_opt))
    }

    /// Try to get P&M via CRC-optimized path.
    ///
    /// Uses CRC files when available to avoid full log replay for Protocol and Metadata.
    fn read_protocol_metadata_opt(
        &self,
        engine: &dyn Engine,
        lazy_crc: &LazyCrc,
    ) -> DeltaResult<(Option<Metadata>, Option<Protocol>)> {
        use crate::crc::CrcLoadResult;

        let crc_version = lazy_crc.crc_version();

        // Case 1: CRC at target version → use directly
        if crc_version == Some(self.end_version) {
            if let CrcLoadResult::Loaded(crc) = lazy_crc.get_or_load(engine) {
                info!("P&M from CRC at target version {}", self.end_version);
                return Ok((Some(crc.metadata.clone()), Some(crc.protocol.clone())));
            }
            warn!(
                "CRC at target version {} failed to load, falling back to log replay",
                self.end_version
            );
        }

        // Case 2: CRC at earlier version → try pruned replay, then CRC fallback
        if let Some(crc_v) = crc_version.filter(|&v| v < self.end_version) {
            info!("Pruning log segment to commits after CRC version {}", crc_v);
            let pruned = self.segment_after_crc(crc_v);
            let (metadata_opt, protocol_opt): (Option<Metadata>, Option<Protocol>) =
                pruned.replay_for_pm(engine, None, None)?;

            if metadata_opt.is_some() && protocol_opt.is_some() {
                return Ok((metadata_opt, protocol_opt));
            }

            // Fall back to CRC for missing P&M
            if let CrcLoadResult::Loaded(crc) = lazy_crc.get_or_load(engine) {
                return Ok((
                    metadata_opt.or_else(|| Some(crc.metadata.clone())),
                    protocol_opt.or_else(|| Some(crc.protocol.clone())),
                ));
            }

            // CRC failed, replay remaining segment
            warn!(
                "CRC at version {} failed to load, replaying remaining segment",
                crc_v
            );
            let remaining = self.segment_through_crc(crc_v);
            return remaining.replay_for_pm(engine, metadata_opt, protocol_opt);
        }

        // Case 3/4: No CRC → return empty, let caller do full replay
        Ok((None, None))
    }

    /// Creates a pruned LogSegment for replay *after* a CRC at `start_v_exclusive`.
    ///
    /// The CRC covers protocol, metadata, and checkpoint state, so this segment drops
    /// checkpoint files, CRC files, and last checkpoint metadata. Only commits and compactions
    /// in `(start_v_exclusive, end_version]` are retained.
    #[allow(dead_code)]
    pub(crate) fn segment_after_crc(&self, start_v_exclusive: Version) -> Self {
        let (commits, compactions) =
            self.filtered_commits_and_compactions(Some(start_v_exclusive), self.end_version);
        LogSegment {
            end_version: self.end_version,
            checkpoint_version: None,
            log_root: self.log_root.clone(),
            table_root: self.table_root.clone(),
            checkpoint_schema: None,
            last_checkpoint_metadata: None,
            listed: LogSegmentFiles {
                ascending_commit_files: commits,
                ascending_compaction_files: compactions,
                checkpoint_parts: vec![],
                latest_crc_file: None,
                latest_commit_file: None,
                max_published_version: None,
            },
        }
    }

    /// Creates a pruned LogSegment for replay *before* a CRC at `end_v_inclusive`.
    ///
    /// Used as fallback when the CRC at `end_v_inclusive` fails to load. Falls back to
    /// checkpoint-based replay, so checkpoint files and metadata are preserved. Only commits
    /// and compactions in `(checkpoint_version, end_v_inclusive]` are retained. Fields not
    /// needed for this replay path (CRC file, latest commit file) are dropped.
    #[allow(dead_code)]
    pub(crate) fn segment_through_crc(&self, end_v_inclusive: Version) -> Self {
        let (commits, compactions) =
            self.filtered_commits_and_compactions(self.checkpoint_version, end_v_inclusive);
        LogSegment {
            end_version: self.end_version,
            checkpoint_version: self.checkpoint_version,
            log_root: self.log_root.clone(),
            table_root: self.table_root.clone(),
            checkpoint_schema: self.checkpoint_schema.clone(),
            last_checkpoint_metadata: self.last_checkpoint_metadata.clone(),
            listed: LogSegmentFiles {
                ascending_commit_files: commits,
                ascending_compaction_files: compactions,
                checkpoint_parts: self.listed.checkpoint_parts.clone(),
                latest_crc_file: None,
                latest_commit_file: None,
                max_published_version: None,
            },
        }
    }

    /// Reads protocol, metadata, and checkpoint actions from the log segment.
    fn replay_for_pmc(
        &self,
        engine: &dyn Engine,
    ) -> DeltaResult<impl Iterator<Item = DeltaResult<ActionsBatch>> + Send> {
        let schema =
            get_commit_schema().project(&[PROTOCOL_NAME, METADATA_NAME, CHECKPOINT_ACTION_NAME])?;
        // IS NOT NULL predicates for protocol/metadata/checkpoint columns are auto-derived
        // from the schema by `read_actions`.
        self.read_actions(engine, schema)
    }

    /// Filters commits and compactions to those within `(lo_exclusive, hi_inclusive]`.
    /// If `lo_exclusive` is `None`, there is no lower bound.
    #[allow(dead_code)]
    fn filtered_commits_and_compactions(
        &self,
        lo_exclusive: Option<Version>,
        hi_inclusive: Version,
    ) -> (Vec<ParsedLogPath>, Vec<ParsedLogPath>) {
        let above_lo = |v: Version| lo_exclusive.is_none_or(|lo| lo < v);
        let commits = self
            .listed
            .ascending_commit_files
            .iter()
            .filter(|c| above_lo(c.version) && c.version <= hi_inclusive)
            .cloned()
            .collect();
        let compactions = self
            .listed
            .ascending_compaction_files
            .iter()
            .filter(|c| {
                matches!(
                    c.file_type,
                    LogPathFileType::CompactedCommit { hi }
                        if above_lo(c.version) && hi <= hi_inclusive
                )
            })
            .cloned()
            .collect();
        (commits, compactions)
    }

    /// How many commits since a checkpoint, according to this log segment.
    /// Returns 0 for pre-commit snapshots (where end_version is PRE_COMMIT_VERSION).
    pub(crate) fn commits_since_checkpoint(&self) -> u64 {
        if self.end_version == PRE_COMMIT_VERSION {
            return 0;
        }
        // we can use 0 as the checkpoint version if there is no checkpoint since `end_version - 0`
        // is the correct number of commits since a checkpoint if there are no checkpoints
        let checkpoint_version = self.checkpoint_version.unwrap_or(0);
        debug_assert!(checkpoint_version <= self.end_version);
        self.end_version - checkpoint_version
    }

    /// How many commits since a log-compaction or checkpoint, according to this log segment.
    /// Returns 0 for pre-commit snapshots (where end_version is PRE_COMMIT_VERSION).
    pub(crate) fn commits_since_log_compaction_or_checkpoint(&self) -> u64 {
        if self.end_version == PRE_COMMIT_VERSION {
            return 0;
        }
        // Annoyingly we have to search all the compaction files to determine this, because we only
        // sort by start version, so technically the max end version could be anywhere in the vec.
        // We can return 0 in the case there is no compaction since end_version - 0 is the correct
        // number of commits since compaction if there are no compactions
        let max_compaction_end = self
            .listed
            .ascending_compaction_files
            .iter()
            .fold(0, |cur, f| {
                if let &ParsedLogPath {
                    file_type: LogPathFileType::CompactedCommit { hi },
                    ..
                } = f
                {
                    Version::max(cur, hi)
                } else {
                    warn!("Found invalid ParsedLogPath in ascending_compaction_files: {f:?}");
                    cur
                }
            });
        // we want to subtract off the max of the max compaction end or the checkpoint version
        let to_sub = Version::max(self.checkpoint_version.unwrap_or(0), max_compaction_end);
        debug_assert!(to_sub <= self.end_version);
        self.end_version - to_sub
    }

    pub(crate) fn validate_published(&self) -> DeltaResult<()> {
        require!(
            self.listed
                .max_published_version
                .is_some_and(|v| v == self.end_version),
            Error::generic("Log segment is not published")
        );
        Ok(())
    }

    /// Schema to read just the sidecar column from a checkpoint file.
    fn sidecar_read_schema() -> SchemaRef {
        static SIDECAR_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
            Arc::new(StructType::new_unchecked([StructField::nullable(
                SIDECAR_NAME,
                Sidecar::to_schema(),
            )]))
        });
        SIDECAR_SCHEMA.clone()
    }

    /// Checks if a checkpoint schema contains a usable `add.stats_parsed` field.
    ///
    /// This validates that:
    /// 1. The `add.stats_parsed` field exists in the checkpoint schema
    /// 2. The types in `stats_parsed` are compatible with the stats schema for data skipping
    ///
    /// The `stats_schema` parameter contains only the columns referenced in the data skipping
    /// predicate. This is built from the predicate and passed in by the caller.
    ///
    /// Both the checkpoint's `stats_parsed` schema and the `stats_schema` for data skipping
    /// use physical column names (not logical names), so direct name comparison is correct.
    ///
    /// Returns `false` if stats_parsed doesn't exist or has incompatible types.
    pub(crate) fn schema_has_compatible_stats_parsed(
        checkpoint_schema: &StructType,
        stats_schema: &StructType,
    ) -> bool {
        // Get add.stats_parsed from the checkpoint schema
        let Some(stats_parsed) = checkpoint_schema
            .field("add")
            .and_then(|f| match f.data_type() {
                DataType::Struct(s) => s.field("stats_parsed"),
                _ => None,
            })
        else {
            debug!("stats_parsed not compatible: checkpoint schema does not contain add.stats_parsed field");
            return false;
        };

        let DataType::Struct(stats_struct) = stats_parsed.data_type() else {
            debug!(
                "stats_parsed not compatible: add.stats_parsed field is not a Struct, got {:?}",
                stats_parsed.data_type()
            );
            return false;
        };

        // Check type compatibility for both minValues and maxValues structs.
        // While these typically have the same schema, the protocol doesn't guarantee it,
        // so we check both to be safe.
        for field_name in ["minValues", "maxValues"] {
            let Some(checkpoint_values_field) = stats_struct.field(field_name) else {
                // stats_parsed exists but no minValues/maxValues - unusual but valid
                continue;
            };

            // minValues/maxValues must be a Struct containing per-column statistics.
            // If it exists but isn't a Struct, the schema is malformed and unusable.
            let DataType::Struct(checkpoint_values) = checkpoint_values_field.data_type() else {
                debug!(
                    "stats_parsed not compatible: stats_parsed.{} is not a Struct, got {:?}",
                    field_name,
                    checkpoint_values_field.data_type()
                );
                return false;
            };

            // Get the corresponding field from stats_schema (e.g., stats_schema.minValues)
            let Some(stats_values_field) = stats_schema.field(field_name) else {
                // stats_schema doesn't have minValues/maxValues, skip this check
                continue;
            };
            let DataType::Struct(stats_values) = stats_values_field.data_type() else {
                // stats_schema.minValues/maxValues isn't a struct - shouldn't happen but skip
                continue;
            };

            // Check type compatibility recursively for nested structs.
            // Only fields that exist in both schemas need compatible types.
            // Extra fields in checkpoint are ignored; missing fields return null.
            if !Self::structs_have_compatible_types(checkpoint_values, stats_values, field_name) {
                return false;
            }
        }

        debug!("Checkpoint schema has compatible stats_parsed for data skipping");
        true
    }

    /// Recursively checks if two struct types have compatible field types.
    ///
    /// Used by both `stats_parsed` and `partitionValues_parsed` compatibility checks.
    /// For each field in `needed`, if it exists in `available` (checkpoint):
    /// - Primitive types: must be compatible via [`PrimitiveType::is_stats_type_compatible_with`]
    ///   (allows type widening and Parquet physical type reinterpretation)
    /// - Nested structs: recursively check inner fields
    /// - Missing fields in checkpoint: OK (will return null when accessed)
    /// - Extra fields in checkpoint: OK (ignored)
    fn structs_have_compatible_types(
        available: &StructType,
        needed: &StructType,
        context: &str,
    ) -> bool {
        for needed_field in needed.fields() {
            let Some(available_field) = available.field(needed_field.name()) else {
                // Field missing in checkpoint - that's OK, it will be null
                continue;
            };

            match (available_field.data_type(), needed_field.data_type()) {
                // Both are structs: recurse
                (DataType::Struct(avail_struct), DataType::Struct(need_struct)) => {
                    let nested_context = format!("{}.{}", context, needed_field.name());
                    if !Self::structs_have_compatible_types(
                        avail_struct,
                        need_struct,
                        &nested_context,
                    ) {
                        return false;
                    }
                }
                // Non-struct types: use stats-specific rules for primitives and standard
                // schema rules otherwise.
                (avail_type, need_type) => {
                    let compatible = match (avail_type, need_type) {
                        (DataType::Primitive(a), DataType::Primitive(b)) => {
                            a.is_stats_type_compatible_with(b)
                        }
                        (a, b) => a.can_read_as(b).is_ok(),
                    };
                    if !compatible {
                        debug!(
                            "stats_parsed not compatible: incompatible type for '{}' in {}: \
                             checkpoint has {:?}, stats schema needs {:?}",
                            needed_field.name(),
                            context,
                            avail_type,
                            need_type
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Checks if a checkpoint schema contains a usable `add.partitionValues_parsed` field.
    ///
    /// Validates that:
    /// 1. The `add.partitionValues_parsed` field exists in the checkpoint schema
    /// 2. The types for partition columns present in both schemas are compatible
    ///
    /// Missing partition columns in the checkpoint are OK (they simply won't contribute
    /// to row group skipping). Returns `false` if `partitionValues_parsed` doesn't exist
    /// or has incompatible types for any shared column.
    pub(crate) fn schema_has_compatible_partition_values_parsed(
        checkpoint_schema: &StructType,
        partition_schema: &StructType,
    ) -> bool {
        let Some(partition_parsed) =
            checkpoint_schema
                .field("add")
                .and_then(|f| match f.data_type() {
                    DataType::Struct(s) => s.field("partitionValues_parsed"),
                    _ => None,
                })
        else {
            debug!("partitionValues_parsed not compatible: checkpoint schema does not contain add.partitionValues_parsed field");
            return false;
        };

        let DataType::Struct(partition_struct) = partition_parsed.data_type() else {
            warn!(
                "partitionValues_parsed not compatible: add.partitionValues_parsed is not a Struct, got {:?}",
                partition_parsed.data_type()
            );
            return false;
        };

        // Flat struct: reuse the recursive type checker (trivial case with no nesting)
        if !Self::structs_have_compatible_types(
            partition_struct,
            partition_schema,
            "partitionValues_parsed",
        ) {
            return false;
        }

        debug!("Checkpoint schema has compatible partitionValues_parsed for partition pruning");
        true
    }
}

fn validate_compaction_files(compactions: &[ParsedLogPath]) -> DeltaResult<()> {
    for (i, f) in compactions.iter().enumerate() {
        let LogPathFileType::CompactedCommit { hi } = f.file_type else {
            return Err(Error::generic(
                "ascending_compaction_files contains non-compaction file",
            ));
        };
        if f.version > hi {
            return Err(Error::generic(format!(
                "compaction file has start version {} > end version {}",
                f.version, hi
            )));
        }
        if let Some(next) = compactions.get(i + 1) {
            // next's type is validated on its own iteration; skip sort check if it isn't a
            // CompactedCommit since the type error will be caught then.
            if let LogPathFileType::CompactedCommit { hi: next_hi } = next.file_type {
                if !(f.version < next.version || (f.version == next.version && hi <= next_hi)) {
                    return Err(Error::generic(format!(
                        "ascending_compaction_files is not sorted: {f:?} -> {next:?}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_checkpoint_parts(parts: &[ParsedLogPath]) -> DeltaResult<()> {
    if parts.is_empty() {
        return Ok(());
    }
    let n = parts.len();
    let first_version = parts[0].version;
    for p in parts {
        if !p.is_checkpoint() {
            return Err(Error::generic(
                "checkpoint_parts contains non-checkpoint file",
            ));
        }
        if p.version != first_version {
            return Err(Error::generic(
                "multi-part checkpoint parts have different versions",
            ));
        }
        match p.file_type {
            LogPathFileType::MultiPartCheckpoint { num_parts, .. } if num_parts as usize == n => {}
            LogPathFileType::MultiPartCheckpoint { num_parts, .. } => {
                return Err(Error::generic(format!(
                    "multi-part checkpoint part count mismatch: slice has {n} parts but num_parts field says {num_parts}"
                )));
            }
            _ if n > 1 => {
                return Err(Error::generic(format!(
                    "multi-part checkpoint part count mismatch: expected {n} multi-part checkpoint files but got a non-multi-part checkpoint"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_commit_file_types(commits: &[ParsedLogPath]) -> DeltaResult<()> {
    for f in commits {
        if !f.is_commit() {
            return Err(Error::generic(
                "ascending_commit_files contains non-commit file",
            ));
        }
    }
    Ok(())
}

fn validate_commit_files_contiguous(commits: &[ParsedLogPath]) -> DeltaResult<()> {
    for pair in commits.windows(2) {
        if pair[0].version + 1 != pair[1].version {
            return Err(Error::generic(format!(
                "Expected contiguous commit files, but found gap: {:?} -> {:?}",
                pair[0], pair[1]
            )));
        }
    }
    Ok(())
}

/// Validates that there is no gap between the checkpoint and the first commit file.
///
/// When a checkpoint exists and commits are also present (after filtering out commits at or before
/// the checkpoint), the first commit must immediately follow the checkpoint (i.e., be at
/// `checkpoint_version + 1`). A gap indicates missing log files.
fn validate_checkpoint_commit_gap(
    checkpoint_version: Option<Version>,
    commits: &[ParsedLogPath],
) -> DeltaResult<()> {
    if let (Some(checkpoint_version), Some(first_commit)) = (checkpoint_version, commits.first()) {
        require!(
            checkpoint_version + 1 == first_commit.version,
            Error::InvalidCheckpoint(format!(
                "Gap between checkpoint version {checkpoint_version} and next commit {}",
                first_commit.version
            ))
        );
    }
    Ok(())
}

/// Validates that the log segment covers exactly `end_version` (when specified) and returns the
/// effective version -- the version of the last commit, or the checkpoint version if no commits
/// are present.
///
/// Returns an error if the segment is empty (no commits and no checkpoint parts), or if the
/// effective version does not match the requested `end_version`.
fn validate_end_version(
    commits: &[ParsedLogPath],
    checkpoint_parts: &[ParsedLogPath],
    end_version: Option<Version>,
) -> DeltaResult<Version> {
    let effective_version = commits
        .last()
        .or(checkpoint_parts.first())
        .ok_or(Error::generic("No files in log segment"))?
        .version;
    if let Some(end_version) = end_version {
        require!(
            effective_version == end_version,
            Error::generic(format!(
                "LogSegment end version {effective_version} not the same as the specified end version {end_version}"
            ))
        );
    }
    Ok(effective_version)
}
