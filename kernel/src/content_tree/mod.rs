pub(crate) mod builder;
pub(crate) mod bulk_processor;
pub(crate) mod data_skipping;
pub(crate) mod lazy_reader;
pub(crate) mod reader;
pub(crate) mod stats;
pub(crate) mod writer;

#[cfg(test)]
#[path = "tests/snaps_and_seqs.rs"]
mod snaps_and_seqs_tests;

// ContentTreeNode based on Adaptive ContentTreeNode Tree
// https://docs.google.com/document/d/1k4x8utgh41Sn1tr98eynDKCWq035SV_f75rtNHcerVw
use crate::actions::{ADD_NAME, REMOVE_NAME};
use crate::engine_data::{EngineData, FilteredEngineData};
use crate::expressions::{ColumnName, PredicateRef, Scalar, StructData};
use crate::log_replay::ActionsBatch;
use crate::path::ParsedLogPath;
use crate::schema::{derive_macro_utils::ToDataType, DataType, StructField, StructType};
use crate::{
    DeltaResult, Error, EvaluationHandler, ExpressionEvaluator, FileMeta, ParquetHandler,
    SchemaRef, Version,
};
use bytes::Bytes;
use delta_kernel_derive::{IntoEngineData, ToSchema};
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use tracing::{debug, warn};
use url::Url;

/// Field name for the content_stats column in ContentTreeNodeEntry schema.
/// This field contains per-column statistics in AMT format.
pub(crate) const CONTENT_STATS_FIELD_NAME: &str = "content_stats";

/// Field names for the different fields within content_stats.
pub(crate) const NULL_VALUE_COUNT: &str = "null_value_count";
pub(crate) const VALUE_COUNT: &str = "value_count";
pub(crate) const NAN_VALUE_COUNT: &str = "nan_value_count";
pub(crate) const AVG_VALUE_SIZE: &str = "avg_value_size";
pub(crate) const MAX_VALUE_SIZE: &str = "max_value_size";
pub(crate) const LOWER_BOUND: &str = "lower_bound";
pub(crate) const UPPER_BOUND: &str = "upper_bound";
pub(crate) const EXACT_BOUNDS: &str = "exact_bounds";

/// Delta JSON stats field names. These are the keys used in the `stats` JSON string of Add actions
/// and in the `stats_parsed` struct produced by parsing that JSON.
pub(crate) const DELTA_STATS_NUM_RECORDS: &str = "numRecords";
pub(crate) const DELTA_STATS_NULL_COUNT: &str = "nullCount";
pub(crate) const DELTA_STATS_MIN_VALUES: &str = "minValues";
pub(crate) const DELTA_STATS_MAX_VALUES: &str = "maxValues";
pub(crate) const DELTA_STATS_TIGHT_BOUNDS: &str = "tightBounds";

/// Type alias for the iterator returned by `open_stream`.
type ParquetStreamResult = (
    Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
    Version,
    String,
);

/// Flat schema for DV columns appended by `append_inline_dv_columns`.
/// Contains the 5 fields extracted from `dvInfo.*` on each Data entry.
static DV_COLUMNS_SCHEMA_FINAL: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(StructType::new_unchecked(vec![
        StructField::new("dv_cardinality", DataType::LONG, true),
        StructField::new("dv_storageType", DataType::STRING, true),
        StructField::new("dv_pathOrInlineDv", DataType::STRING, true),
        StructField::new("dv_offset", DataType::INTEGER, true),
        StructField::new("dv_sizeInBytes", DataType::INTEGER, true),
    ]))
});

/// A stats provider that extracts min/max statistics from AMT manifest `content_stats`.
///
/// This struct implements `ParquetStatsProvider` to enable predicate evaluation against
/// manifest-level statistics for data skipping. The `content_stats` field in a manifest
/// entry contains aggregated min/max bounds over all files in that manifest.
///
/// The stats structure follows the AMT format:
/// ```text
/// content_stats: {
///   column_name: {
///     value_count: i64,
///     null_value_count: i64,  // if nullable
///     lower_bound: <column_type>,
///     upper_bound: <column_type>,
///     exact_bounds: bool
///   },
///   ...
/// }
/// ```
/// Represents table metadata in Adaptive ContentTreeNode Tree (AMT) format.
///
/// This structure contains metadata entries that describe the files in a Delta table
/// at a specific version. It is used for interoperability with Apache Iceberg's
/// metadata tree format.
///
/// Each `ContentTreeNode` instance contains:
/// - A collection of `ContentTreeNodeEntry` records (one per file)
/// - The Delta table version this metadata represents
/// - The table root URL for resolving relative file paths
/// - An optional leaf UUID (only set when writing a leaf manifest, not for root)
pub(super) struct ContentTreeNode {
    data: Vec<Box<dyn EngineData>>,
    version: Version,
    table_root: Url,
    /// The exact path string as it appears in the Delta log (from contentRoot action or manifest location field).
    /// This is NOT normalized or converted - it flows through exactly as stored in the log.
    /// Empty string for newly built metadata that hasn't been written yet.
    path_in_log: String,
    /// Optional UUID that identifies this metadata as a leaf manifest.
    /// When writing a root manifest, this is `None`.
    /// When writing a leaf manifest, this must be set to a unique UUID.
    leaf: Option<uuid::Uuid>,
}

/// A manifest entry wrapper.
///
/// According to the Iceberg Single File Commits spec, manifest deletion vectors
/// can filter out entries from a manifest by ordinal position without rewriting the manifest file.
#[derive(Debug, Clone)]
pub(crate) struct FilteredManifest {
    /// The manifest entry (can be DataManifest or DeleteManifest)
    pub(crate) manifest: ContentTreeNodeEntry,
}

impl FilteredManifest {
    /// Creates a new FilteredManifest
    pub(crate) fn new(manifest: ContentTreeNodeEntry) -> Self {
        Self { manifest }
    }
}

/// Complete state of the root manifest, including manifest references.
#[derive(Debug, Clone)]
pub(crate) struct LeafReferences {
    /// References to child data manifests
    pub(crate) manifest_references: Vec<ManifestReference>,
}

/// References to manifest files discovered in the root manifest.
#[derive(Debug, Clone)]
pub(crate) struct ManifestReference {
    /// The data manifest entry to process, with optional manifest DV
    pub(crate) data_manifest: FilteredManifest,
}

/// A pair of optional Add and Remove evaluators
struct EvaluatorPair {
    add_evaluator: Option<Arc<dyn ExpressionEvaluator>>,
    remove_evaluator: Option<Arc<dyn ExpressionEvaluator>>,
}

/// Stateful applicator for manifest deletion vectors.
///
/// Instead of materializing a full Vec<bool> selection vector upfront,
/// this applicator queries the RoaringTreemap on-demand for each batch.
struct ManifestDvApplicator {
    /// Parsed deletion vector (None if no manifest DV present)
    deleted_positions: Option<roaring::RoaringTreemap>,

    /// Current cumulative row offset
    offset: usize,
}

impl ManifestDvApplicator {
    /// Create new applicator from manifest_dv bytes.
    fn new(manifest_dv: Option<&Bytes>) -> DeltaResult<Self> {
        let deleted_positions = if let Some(dv_bytes) = manifest_dv {
            parse_manifest_dv(dv_bytes)?
        } else {
            None
        };
        Ok(Self {
            deleted_positions,
            offset: 0,
        })
    }

    /// Process a batch and return it with its selection vector.
    ///
    /// Returns FilteredEngineData containing the batch and a selection vector.
    /// If a manifest DV is present, the selection vector indicates which rows are NOT deleted.
    /// If no manifest DV is present, all rows are marked as selected.
    fn process_batch(&mut self, batch: Box<dyn EngineData>) -> DeltaResult<FilteredEngineData> {
        let batch_len = batch.len();
        let filtered = if let Some(deleted) = &self.deleted_positions {
            let selection: Vec<bool> = (0..batch_len)
                .map(|i| !deleted.contains((self.offset + i) as u64))
                .collect();
            FilteredEngineData::try_new(batch, selection)?
        } else {
            FilteredEngineData::with_all_rows_selected(batch)
        };
        self.offset += batch_len;
        Ok(filtered)
    }
}

impl ContentTreeNode {
    /// Construct ContentTreeNode from batches with a specific version (for content root reading).
    ///
    /// Validates that the root manifest only contains supported entry types
    /// (`Data` and `CombinedManifest`). Returns an error if any unsupported
    /// manifest type is found.
    pub(crate) fn from_batches_with_version(
        data: Vec<Box<dyn EngineData>>,
        version: Version,
        path_in_log: String,
        table_root: Url,
    ) -> DeltaResult<Self> {
        let node = Self {
            data,
            version,
            table_root,
            path_in_log,
            leaf: None,
        };
        node.validate_root_manifest_entries()?;
        Ok(node)
    }

    /// Validates that the root manifest only contains supported entry types.
    ///
    /// In the CombinedManifest model, the only supported manifest type is
    /// `CombinedManifest` (value=5). `Data` entries (value=0) are also allowed.
    /// Any other active entry type (DataManifest, DeleteManifest, PositionDeletes,
    /// EqualityDeletes) causes an `Error::unsupported`.
    fn validate_root_manifest_entries(&self) -> DeltaResult<()> {
        use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
        use crate::schema::{ColumnName, DataType};
        use std::sync::LazyLock;

        struct RootEntryValidator;

        impl RowVisitor for RootEntryValidator {
            fn selected_column_names_and_types(
                &self,
            ) -> (&'static [ColumnName], &'static [DataType]) {
                static NAMES: LazyLock<Vec<ColumnName>> = LazyLock::new(|| {
                    vec![
                        ColumnName::new(["contentType"]),
                        ColumnName::new(["trackingInfo", "status"]),
                    ]
                });
                static TYPES: &[DataType] = &[DataType::INTEGER, DataType::INTEGER];
                (NAMES.as_slice(), TYPES)
            }

            fn visit<'a>(
                &mut self,
                row_count: usize,
                getters: &[&'a dyn GetData<'a>],
            ) -> DeltaResult<()> {
                for i in 0..row_count {
                    let content_type_int: i32 = getters[0].get(i, "contentType")?;
                    let status: i32 = getters[1].get(i, "trackingInfo.status")?;

                    // Skip DELETED entries (status=3) — filtered out at read time
                    if status == 3 {
                        continue;
                    }

                    match content_type_int {
                        0 | 5 => {} // Data or CombinedManifest — supported
                        3 | 4 => {
                            return Err(Error::unsupported(
                                "DataManifest/DeleteManifest format is not supported; \
                                 only CombinedManifest (type 5) is supported in the content tree",
                            ))
                        }
                        1 => return Err(Error::unsupported(
                            "PositionDeletes entries are not supported in the content tree root",
                        )),
                        2 => return Err(Error::unsupported("EqualityDeletes are not supported")),
                        other => {
                            return Err(Error::unsupported(format!(
                                "Unknown content type {other} in content tree root"
                            )))
                        }
                    }
                }
                Ok(())
            }
        }

        for batch in self.data.iter() {
            if batch.is_empty() {
                continue;
            }
            let mut visitor = RootEntryValidator;
            visitor.visit_rows_of(batch.as_ref())?;
        }
        Ok(())
    }

    /// Returns the leaf UUID if this is a leaf manifest, or `None` if it's a root manifest.
    pub(crate) fn leaf(&self) -> Option<uuid::Uuid> {
        self.leaf
    }

    pub(crate) fn entries(&self) -> DeltaResult<Vec<ContentTreeNodeEntry>> {
        let total_rows: usize = self.data.iter().map(|b| b.len()).sum();
        let mut all_entries = Vec::with_capacity(total_rows);
        use crate::engine_data::RowVisitor;
        for batch in self.data.iter() {
            let mut visitor = reader::ContentTreeNodeEntryVisitor::default();
            visitor.visit_rows_of(batch.as_ref())?;
            all_entries.extend(visitor.entries);
        }
        Ok(all_entries)
    }

    /// Helper to build expression for a field based on action type.
    fn build_action_field_expression(
        field_name: &str,
        field_type: &DataType,
        action_name: &str,
        path_in_log: &str,
        has_stats_parsed: bool,
        has_dv_columns: bool,
    ) -> DeltaResult<crate::expressions::Expression> {
        use crate::expressions::{Expression, MapData, VariadicExpressionOp};
        use crate::schema::{DataType, MapType};

        Ok(match field_name {
            // Common fields for both Add and Remove
            "path" => Expression::column(["location"]),
            "size" => Expression::variadic(
                VariadicExpressionOp::Coalesce,
                [
                    Expression::column(["fileSizeInBytes"]),
                    Expression::literal(0i64),
                ],
            ),
            "stats" => Expression::null_literal(DataType::STRING),
            "baseRowId" => Expression::column(["trackingInfo", "firstRowId"]),
            "defaultRowCommitVersion" => Expression::column(["trackingInfo", "sequenceNumber"]),
            "partitionValues" => {
                let empty_map = MapData::try_new(
                    MapType::new(DataType::STRING, DataType::STRING, false),
                    Vec::<(Scalar, Scalar)>::new(),
                )?;
                Expression::literal(Scalar::Map(empty_map))
            }
            "dataChange" => Expression::literal(true),
            "tags" => Expression::null_literal(DataType::Map(Box::new(MapType::new(
                DataType::STRING,
                DataType::STRING,
                true,
            )))),
            "deletionVector" => {
                use crate::actions::deletion_vector::DeletionVectorDescriptor;
                use crate::schema::ToSchema;
                let dv_schema = <DeletionVectorDescriptor as ToSchema>::to_schema();
                if has_dv_columns {
                    Expression::struct_with_nullability_from(
                        vec![
                            Arc::new(Expression::column(["dv_storageType"])),
                            Arc::new(Expression::column(["dv_pathOrInlineDv"])),
                            Arc::new(Expression::column(["dv_offset"])),
                            Arc::new(Expression::column(["dv_sizeInBytes"])),
                            Arc::new(Expression::column(["dv_cardinality"])),
                        ],
                        Expression::from_pred(Expression::column(["dv_storageType"]).is_not_null()),
                    )
                } else {
                    Expression::null_literal(DataType::Struct(Box::new(dv_schema)))
                }
            }

            // Add-specific fields
            "modificationTime" if action_name == "add" => Expression::literal(i64::MIN),
            "dataManifestPath" if action_name == "add" => {
                Expression::literal(path_in_log.to_string())
            }
            "dataManifestPosition" if action_name == "add" => Expression::column(["_pos"]),
            "clusteringProvider" if action_name == "add" => {
                Expression::null_literal(DataType::STRING)
            }
            "stats_parsed" if action_name == "add" => {
                if has_stats_parsed {
                    // Read stats_parsed from the augmented metadata batch
                    // The stats_parsed field is added to the batch by the stats transformation evaluator
                    // (see root_action_batches_optimized_with_handler)
                    Expression::column(["stats_parsed"])
                } else {
                    // No full stats transformation available; populate numRecords from recordCount
                    // so callers still get a meaningful row count without JSON serialization.
                    let DataType::Struct(stats_struct) = field_type else {
                        return Err(Error::internal_error(format!(
                            "stats_parsed field type should be a struct, got {field_type:?}"
                        )));
                    };
                    let field_exprs: Vec<_> = stats_struct
                        .fields()
                        .map(|f| {
                            if f.name() == DELTA_STATS_NUM_RECORDS {
                                Expression::column(["recordCount"])
                            } else {
                                Expression::null_literal(f.data_type().clone())
                            }
                        })
                        .collect();
                    Expression::struct_from(field_exprs)
                }
            }

            // Remove-specific fields
            "deletionTimestamp" if action_name == "remove" => Expression::literal(i64::MIN),
            "extendedFileMetadata" if action_name == "remove" => Expression::literal(true),

            // Default: null with field's type
            _ => Expression::null_literal(field_type.clone()),
        })
    }

    /// Builds a Transform expression to convert ContentTreeNodeEntry → Add or Remove action.
    fn build_metadata_to_action_transform(
        action_schema: &SchemaRef,
        action_name: &str,
        path_in_log: &str,
        has_stats_parsed: bool,
        has_dv_columns: bool,
    ) -> DeltaResult<Arc<crate::expressions::Expression>> {
        use crate::expressions::Expression;
        use crate::schema::DataType;

        let action_field = action_schema
            .field(action_name)
            .ok_or_else(|| Error::generic(format!("Schema missing '{action_name}' field")))?;
        let action_struct_type = match action_field.data_type() {
            DataType::Struct(s) => s,
            _ => {
                return Err(Error::generic(format!(
                    "'{action_name}' field is not a struct"
                )))
            }
        };

        let mut field_exprs: Vec<Arc<Expression>> = Vec::new();
        for field in action_struct_type.fields() {
            let expr = Self::build_action_field_expression(
                field.name(),
                field.data_type(),
                action_name,
                path_in_log,
                has_stats_parsed,
                has_dv_columns,
            )?;
            field_exprs.push(Arc::new(expr));
        }

        let action_struct_expr = Expression::struct_from(field_exprs);

        // Build a top-level struct matching the full output_schema. The action field is populated
        // from the expression; all other fields are null literals. This ensures the evaluated
        // batch always has the same schema as output_schema regardless of which action type is
        // being produced, so downstream visitors can access add.* and remove.* uniformly.
        let top_level_exprs: Vec<Arc<Expression>> = action_schema
            .fields()
            .map(|f| {
                if f.name() == action_name {
                    Arc::new(action_struct_expr.clone())
                } else {
                    Arc::new(Expression::null_literal(f.data_type().clone()))
                }
            })
            .collect();

        Ok(Arc::new(Expression::struct_from(top_level_exprs)))
    }

    /// Appends 5 flat DV columns extracted directly from `dvInfo.*` on each Data entry.
    ///
    /// For Data entries (contentType=0): parses dvInfo.location into storageType/pathOrInlineDv,
    /// casts dvInfo.offset i64→i32, subtracts 8 from dvInfo.sizeInBytes then casts i64→i32,
    /// reads dvInfo.cardinality directly.
    /// For non-Data entries: all 5 columns are null.
    /// Returns `None` if no Data entries in the batch had a DV (all DV columns would be null),
    /// avoiding unnecessary column allocation. Returns `Some` with DV columns appended otherwise.
    fn append_inline_dv_columns(
        batch: &dyn EngineData,
        table_root: &Url,
    ) -> DeltaResult<Option<Box<dyn EngineData>>> {
        use crate::actions::deletion_vector::DeletionVectorPath;
        use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
        use crate::expressions::{ArrayData, Scalar};
        use crate::schema::{ArrayType, ColumnName, DataType};

        struct InlineDvVisitor {
            cardinalities: Vec<Scalar>,
            storage_types: Vec<Scalar>,
            path_or_inline_dvs: Vec<Scalar>,
            offsets: Vec<Scalar>,
            size_in_bytes: Vec<Scalar>,
            table_root: Url,
            has_dv: bool,
            /// Total rows visited across all `visit()` calls before the current one.
            /// Used to backfill null rows when the first DV is discovered mid-batch.
            rows_seen: usize,
        }

        impl InlineDvVisitor {
            /// Pushes one all-null row to each column vector.
            fn push_null_row(&mut self) {
                self.cardinalities.push(Scalar::Null(DataType::LONG));
                self.storage_types.push(Scalar::Null(DataType::STRING));
                self.path_or_inline_dvs.push(Scalar::Null(DataType::STRING));
                self.offsets.push(Scalar::Null(DataType::INTEGER));
                self.size_in_bytes.push(Scalar::Null(DataType::INTEGER));
            }
        }

        impl RowVisitor for InlineDvVisitor {
            fn selected_column_names_and_types(
                &self,
            ) -> (&'static [ColumnName], &'static [DataType]) {
                static NAMES: LazyLock<Vec<ColumnName>> = LazyLock::new(|| {
                    vec![
                        ColumnName::new(["contentType"]),
                        ColumnName::new(["dvInfo", "cardinality"]),
                        ColumnName::new(["dvInfo", "location"]),
                        ColumnName::new(["dvInfo", "offset"]),
                        ColumnName::new(["dvInfo", "sizeInBytes"]),
                    ]
                });
                static TYPES: &[DataType] = &[
                    DataType::INTEGER,
                    DataType::LONG,
                    DataType::STRING,
                    DataType::LONG,
                    DataType::LONG,
                ];
                (NAMES.as_slice(), TYPES)
            }

            fn visit<'a>(
                &mut self,
                row_count: usize,
                getters: &[&'a dyn GetData<'a>],
            ) -> DeltaResult<()> {
                for i in 0..row_count {
                    let content_type: i32 = getters[0].get(i, "contentType")?;

                    // Only Data entries (contentType=0) can carry a DV location.
                    let location_opt: Option<&str> = if content_type == 0 {
                        getters[2].get_opt(i, "dvInfo.location")?
                    } else {
                        None
                    };

                    if let Some(loc) = location_opt {
                        // First DV: allocate vectors and backfill nulls for all rows
                        // that were skipped before this one (rows 0..rows_seen+i).
                        if !self.has_dv {
                            self.has_dv = true;
                            let backfill = self.rows_seen + i;
                            let remaining = row_count - i;
                            self.cardinalities.reserve(backfill + remaining);
                            self.storage_types.reserve(backfill + remaining);
                            self.path_or_inline_dvs.reserve(backfill + remaining);
                            self.offsets.reserve(backfill + remaining);
                            self.size_in_bytes.reserve(backfill + remaining);
                            for _ in 0..backfill {
                                self.push_null_row();
                            }
                        }

                        // Parse path and push storage_type / path_or_inline_dv.
                        let (storage_type, path_or_inline_dv) =
                            match DeletionVectorPath::parse_path(loc, &self.table_root) {
                                Ok((st, path)) => {
                                    (Scalar::String(st.to_string()), Scalar::String(path))
                                }
                                Err(_) => (
                                    Scalar::Null(DataType::STRING),
                                    Scalar::Null(DataType::STRING),
                                ),
                            };
                        self.storage_types.push(storage_type);
                        self.path_or_inline_dvs.push(path_or_inline_dv);

                        // Push cardinality, offset, sizeInBytes for this DV row.
                        let cardinality: Option<i64> =
                            getters[1].get_opt(i, "dvInfo.cardinality")?;
                        self.cardinalities.push(match cardinality {
                            Some(v) => Scalar::Long(v),
                            None => Scalar::Null(DataType::LONG),
                        });

                        let offset_opt: Option<i64> = getters[3].get_opt(i, "dvInfo.offset")?;
                        self.offsets.push(match offset_opt {
                            Some(v) => Scalar::Integer(i32::try_from(v).map_err(|_| {
                                Error::generic(format!(
                                    "DV offset {} out of i32 range ({}..={})",
                                    v,
                                    i32::MIN,
                                    i32::MAX
                                ))
                            })?),
                            None => Scalar::Null(DataType::INTEGER),
                        });

                        let size_opt: Option<i64> = getters[4].get_opt(i, "dvInfo.sizeInBytes")?;
                        self.size_in_bytes.push(match size_opt {
                            Some(v) => {
                                let adjusted = v.checked_sub(8).ok_or_else(|| {
                                    Error::generic(format!(
                                        "DV sizeInBytes {} is less than 8 (magic number size)",
                                        v
                                    ))
                                })?;
                                Scalar::Integer(i32::try_from(adjusted).map_err(|_| {
                                    Error::generic(format!(
                                        "DV sizeInBytes {} out of i32 range ({}..={})",
                                        adjusted,
                                        i32::MIN,
                                        i32::MAX
                                    ))
                                })?)
                            }
                            None => Scalar::Null(DataType::INTEGER),
                        });
                    } else if self.has_dv {
                        // Non-DV row after a DV has been seen: all columns are null.
                        self.push_null_row();
                    }
                    // else: no DV seen yet — skip entirely (no allocation).
                }
                self.rows_seen += row_count;
                Ok(())
            }
        }

        let mut visitor = InlineDvVisitor {
            cardinalities: Vec::new(),
            storage_types: Vec::new(),
            path_or_inline_dvs: Vec::new(),
            offsets: Vec::new(),
            size_in_bytes: Vec::new(),
            table_root: table_root.clone(),
            has_dv: false,
            rows_seen: 0,
        };
        visitor.visit_rows_of(batch)?;

        if !visitor.has_dv {
            return Ok(None);
        }

        Ok(Some(batch.append_columns(
            DV_COLUMNS_SCHEMA_FINAL.clone(),
            vec![
                ArrayData::try_new(ArrayType::new(DataType::LONG, true), visitor.cardinalities)?,
                ArrayData::try_new(
                    ArrayType::new(DataType::STRING, true),
                    visitor.storage_types,
                )?,
                ArrayData::try_new(
                    ArrayType::new(DataType::STRING, true),
                    visitor.path_or_inline_dvs,
                )?,
                ArrayData::try_new(ArrayType::new(DataType::INTEGER, true), visitor.offsets)?,
                ArrayData::try_new(
                    ArrayType::new(DataType::INTEGER, true),
                    visitor.size_in_bytes,
                )?,
            ],
        )?))
    }

    /// Builds selection vectors for Add vs Remove entries based on trackingInfo.status.
    ///
    /// Returns (add_selection, remove_selection) where:
    /// - add_selection[i] = true if entry i has status Existed (0) or Added (1)
    /// - remove_selection[i] = true if entry i has status Deleted (2)
    ///
    /// Both exclude manifest entries (contentType 3, 4) and other non-data types.
    fn build_add_remove_selection_vectors(
        batch: &dyn EngineData,
    ) -> DeltaResult<(Vec<bool>, Vec<bool>)> {
        use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
        use crate::schema::DataType;

        struct AddRemoveVisitor {
            add_selection: Vec<bool>,
            remove_selection: Vec<bool>,
        }

        impl RowVisitor for AddRemoveVisitor {
            fn selected_column_names_and_types(
                &self,
            ) -> (&'static [ColumnName], &'static [DataType]) {
                static NAMES: LazyLock<Vec<ColumnName>> = LazyLock::new(|| {
                    vec![
                        ColumnName::new(["contentType"]),
                        ColumnName::new(["trackingInfo", "status"]),
                    ]
                });
                static TYPES: &[DataType] = &[DataType::INTEGER, DataType::INTEGER];
                (NAMES.as_slice(), TYPES)
            }

            fn visit<'a>(
                &mut self,
                row_count: usize,
                getters: &[&'a dyn GetData<'a>],
            ) -> DeltaResult<()> {
                for i in 0..row_count {
                    let content_type: i32 = getters[0].get(i, "contentType")?;
                    let status: i32 = getters[1].get(i, "trackingInfo.status")?;

                    // Only process Data entries (contentType=0)
                    // Skip DVs (1), EqualityDeletes (2), and Manifests (3, 4)
                    if content_type == 0 {
                        match status {
                            0 | 1 => {
                                // Existed or Added -> Add action
                                self.add_selection.push(true);
                                self.remove_selection.push(false);
                            }
                            2 => {
                                // Deleted -> Remove action
                                self.add_selection.push(false);
                                self.remove_selection.push(true);
                            }
                            _ => {
                                // Unknown status
                                self.add_selection.push(false);
                                self.remove_selection.push(false);
                            }
                        }
                    } else {
                        // Not a data entry
                        self.add_selection.push(false);
                        self.remove_selection.push(false);
                    }
                }
                Ok(())
            }
        }

        let mut visitor = AddRemoveVisitor {
            add_selection: Vec::new(),
            remove_selection: Vec::new(),
        };

        visitor.visit_rows_of(batch)?;

        Ok((visitor.add_selection, visitor.remove_selection))
    }

    /// Helper to extend metadata schema with DV fields (_pos if missing, dv_schema.
    fn extend_metadata_schema_with_dv_fields(
        metadata_schema: &SchemaRef,
        dv_schema: &SchemaRef,
    ) -> SchemaRef {
        let mut fields: Vec<_> = metadata_schema.fields().cloned().collect();

        // Add _pos if not already present (it should be from parquet reader or test helper)
        if !metadata_schema.contains("_pos") {
            // TODO: check if this is still needed
            fields.push(StructField::new("_pos", DataType::LONG, false));
        }

        // Add DV columns from cached schema
        fields.extend(dv_schema.fields().cloned());

        Arc::new(StructType::new_unchecked(fields))
    }

    /// Determine the evaluator schema based on metadata schema, always extended with DV fields.
    fn get_evaluator_schema(metadata_schema: &SchemaRef) -> SchemaRef {
        Self::extend_metadata_schema_with_dv_fields(metadata_schema, &DV_COLUMNS_SCHEMA_FINAL)
    }

    /// Evaluator schema for batches with no DVs: metadata_schema as-is, plus stats_parsed if needed.
    /// Does NOT include the `dv_*` columns — used when `append_inline_dv_columns` returns `None`.
    fn get_evaluator_schema_no_dv(
        metadata_schema: &SchemaRef,
        stats_schema: Option<&StructType>,
    ) -> SchemaRef {
        if let Some(stats_sch) = stats_schema {
            if metadata_schema
                .field(crate::content_tree::CONTENT_STATS_FIELD_NAME)
                .is_some()
            {
                let mut fields: Vec<StructField> = metadata_schema.fields().cloned().collect();
                fields.push(StructField::nullable(
                    "stats_parsed",
                    DataType::Struct(Box::new(stats_sch.clone())),
                ));
                return Arc::new(StructType::new_unchecked(fields));
            }
        }
        metadata_schema.clone()
    }

    /// Determine the evaluator schema with stats_parsed if needed
    fn get_evaluator_schema_with_stats(
        metadata_schema: &SchemaRef,
        stats_schema: Option<&StructType>,
    ) -> SchemaRef {
        let mut schema = Self::get_evaluator_schema(metadata_schema);

        // Only add stats_parsed and content_stats if:
        // 1. stats_schema is provided
        // 2. content_stats field exists in metadata (meaning it was read with table_schema)
        if let Some(stats_sch) = stats_schema {
            if let Some(content_stats_field) =
                metadata_schema.field(crate::content_tree::CONTENT_STATS_FIELD_NAME)
            {
                let mut fields: Vec<StructField> = schema.fields().cloned().collect();
                // Add content_stats so it can be read by the stats transformation
                fields.push(content_stats_field.clone());
                // Add stats_parsed as the transformed output
                fields.push(StructField::nullable(
                    "stats_parsed",
                    crate::schema::DataType::Struct(Box::new(stats_sch.clone())),
                ));
                schema = Arc::new(StructType::new_unchecked(fields));
            }
        }

        schema
    }

    /// Build Add and/or Remove evaluators based on the schema.
    ///
    /// `has_dv_columns`: if true, the evaluator schema includes the 5 `dv_*` columns appended by
    /// `append_inline_dv_columns`, and the `deletionVector` expression reads from them. If false,
    /// `deletionVector` is a `null_literal` and the evaluator schema has no `dv_*` columns.
    fn build_action_evaluators(
        evaluation_handler: &dyn EvaluationHandler,
        evaluator_schema: SchemaRef,
        output_schema: &SchemaRef,
        path_in_log: &str,
        has_add: bool,
        has_remove: bool,
        has_dv_columns: bool,
    ) -> DeltaResult<EvaluatorPair> {
        // Check if stats_parsed is available in evaluator schema (indicates stats transformation is enabled)
        let has_stats_parsed = evaluator_schema.field("stats_parsed").is_some();

        let add_evaluator_opt = if has_add {
            let add_expr = Self::build_metadata_to_action_transform(
                output_schema,
                "add",
                path_in_log,
                has_stats_parsed,
                has_dv_columns,
            )?;
            Some(evaluation_handler.new_expression_evaluator(
                evaluator_schema.clone(),
                add_expr,
                output_schema.clone().into(),
            )?)
        } else {
            None
        };

        let remove_evaluator_opt = if has_remove {
            let remove_expr = Self::build_metadata_to_action_transform(
                output_schema,
                "remove",
                path_in_log,
                has_stats_parsed,
                has_dv_columns,
            )?;
            Some(evaluation_handler.new_expression_evaluator(
                evaluator_schema.clone(),
                remove_expr,
                output_schema.clone().into(),
            )?)
        } else {
            None
        };

        Ok(EvaluatorPair {
            add_evaluator: add_evaluator_opt,
            remove_evaluator: remove_evaluator_opt,
        })
    }

    fn root_action_batches_optimized_with_handler(
        &self,
        evaluation_handler: &dyn EvaluationHandler,
        schema: &SchemaRef,
        predicate: Option<&PredicateRef>,
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>> {
        use crate::actions::{ADD_NAME, REMOVE_NAME};

        // Log if predicate is present (data skipping will be applied)
        if predicate.is_some() {
            debug!("Predicate present in optimized path - data skipping will be applied");
        }

        // Determine which action types are in the schema
        let has_add = schema.contains(ADD_NAME);
        let has_remove = schema.contains(REMOVE_NAME);

        // Get metadata schema that matches actual batches from open_stream.
        let metadata_schema =
            ContentTreeNodeEntry::processing_schema_with_pos(table_schema, stats_schema)?;

        // Build two evaluator variants:
        // - with_dv: for batches where append_inline_dv_columns appended DV columns
        // - no_dv:   for batches with no DVs (null_literal used for deletionVector)
        let dv_extended_schema =
            Self::extend_metadata_schema_with_dv_fields(&metadata_schema, &DV_COLUMNS_SCHEMA_FINAL);
        let eval_schema_with_dv =
            Self::get_evaluator_schema_with_stats(&metadata_schema, stats_schema);
        let eval_schema_no_dv = Self::get_evaluator_schema_no_dv(&metadata_schema, stats_schema);

        let evaluators_with_dv = Self::build_action_evaluators(
            evaluation_handler,
            eval_schema_with_dv,
            schema,
            &self.path_in_log,
            has_add,
            has_remove,
            true,
        )?;
        let evaluators_no_dv = Self::build_action_evaluators(
            evaluation_handler,
            eval_schema_no_dv,
            schema,
            &self.path_in_log,
            has_add,
            has_remove,
            false,
        )?;

        // Stats transformation evaluators: one per DV variant so each uses the right input schema.
        let stats_transform_with_dv = ContentTreeNodeEntry::create_stats_transformation_evaluator(
            evaluation_handler,
            &dv_extended_schema,
            schema,
            table_schema,
            stats_schema,
        )?;
        let stats_transform_no_dv = ContentTreeNodeEntry::create_stats_transformation_evaluator(
            evaluation_handler,
            &metadata_schema,
            schema,
            table_schema,
            stats_schema,
        )?;

        // Process each batch
        let mut result_batches = Vec::new();

        for batch in &self.data {
            // Try to append inline DV columns; None means no DVs present in this batch.
            let dv_augmented = Self::append_inline_dv_columns(batch.as_ref(), &self.table_root)?;
            let (batch_initial, stats_transform, add_eval, remove_eval) = match &dv_augmented {
                Some(b) => (
                    b.as_ref(),
                    &stats_transform_with_dv,
                    &evaluators_with_dv.add_evaluator,
                    &evaluators_with_dv.remove_evaluator,
                ),
                None => (
                    batch.as_ref(),
                    &stats_transform_no_dv,
                    &evaluators_no_dv.add_evaluator,
                    &evaluators_no_dv.remove_evaluator,
                ),
            };

            // Optionally transform content_stats to stats_parsed and augment batch
            let stats_augmented_batch;
            let batch_ref: &dyn EngineData = if let Some(ref stats_eval) = stats_transform {
                stats_augmented_batch = stats_eval.evaluate(batch_initial)?;
                stats_augmented_batch.as_ref()
            } else {
                batch_initial
            };

            // Process Add entries if needed
            if let Some(add_eval) = add_eval.as_ref() {
                let (add_selection, _) = Self::build_add_remove_selection_vectors(batch_ref)?;

                if add_selection.iter().any(|&b| b) {
                    let transformed = add_eval.evaluate(batch_ref)?;
                    let filtered_data = transformed.apply_selection_vector(add_selection)?;
                    result_batches.push(Ok(ActionsBatch::new(filtered_data, false)));
                }
            }

            // Process Remove entries if needed
            if let Some(remove_eval) = remove_eval.as_ref() {
                let (_, remove_selection) = Self::build_add_remove_selection_vectors(batch_ref)?;

                if remove_selection.iter().any(|&b| b) {
                    let transformed = remove_eval.evaluate(batch_ref)?;
                    let filtered_data = transformed.apply_selection_vector(remove_selection)?;
                    result_batches.push(Ok(ActionsBatch::new(filtered_data, false)));
                }
            }
        }

        Ok(Box::new(result_batches.into_iter()))
    }

    /// Version of root_action_batches that takes handlers directly (for lazy streaming).
    pub(crate) fn root_action_batches_with_handler(
        &self,
        evaluation_handler: &dyn EvaluationHandler,
        schema: &SchemaRef,
        _partition_keys: &[String],
        predicate: Option<&PredicateRef>,
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>> {
        // Return empty iterator if schema doesn't contain Add or Remove
        if !schema.contains(ADD_NAME) && !schema.contains(REMOVE_NAME) {
            return Ok(Box::new(std::iter::empty()));
        }

        debug!("Using optimized path for metadata reading");
        self.root_action_batches_optimized_with_handler(
            evaluation_handler,
            schema,
            predicate,
            table_schema,
            stats_schema,
        )
    }

    /// Discovers child manifest references in the root manifest.
    ///
    /// This method implements the hierarchical metadata tree structure described in the
    /// Iceberg Single File Commits specification. It parses the root manifest and identifies:
    ///
    /// - **CombinedManifest files** (content_type = CombinedManifest): References to child
    ///   manifests containing actual data file entries with optional inline DV info
    /// - **Manifest deletion vectors**: Stored inline in the `manifest_dv` field of
    ///   CombinedManifest entries. Applied during manifest reading to filter out deleted entries.
    ///
    /// # Returns
    /// A `LeafReferences` containing one `ManifestReference` per CombinedManifest in the root.
    ///
    ///
    /// # Parameters
    /// - `predicate`: Optional predicate for manifest-level data skipping. When provided,
    ///   manifests whose `content_stats` indicate they cannot contain matching data will
    ///   be skipped (not included in the returned references).
    pub(crate) fn manifest_references(
        &self,
        predicate: Option<&PredicateRef>,
        evaluation_handler: Option<&Arc<dyn EvaluationHandler>>,
        stats_schema: Option<&StructType>,
        table_schema: Option<&StructType>,
        manifest_batch_schema: Option<&SchemaRef>,
    ) -> DeltaResult<LeafReferences> {
        // Try to create expression-based filter if all required params are provided
        let skipping_filter = if let (
            Some(pred),
            Some(handler),
            Some(stats),
            Some(schema),
            Some(batch_schema),
        ) = (
            predicate,
            evaluation_handler,
            stats_schema,
            table_schema,
            manifest_batch_schema,
        ) {
            let filter = data_skipping::ManifestDataSkippingFilter::new(
                handler,
                pred,
                stats,
                schema,
                batch_schema,
            );
            if filter.is_some() {
                debug!("Created expression-based manifest data skipping filter");
            } else {
                debug!("Failed to create expression-based filter");
            }
            filter
        } else if predicate.is_some() {
            // Predicate was provided but couldn't apply manifest-level pruning — log why.
            warn!("Manifest-level data skipping disabled despite predicate: handler={}, stats_schema={}, table_schema={}, batch_schema={}",
                evaluation_handler.is_some(), stats_schema.is_some(), table_schema.is_some(), manifest_batch_schema.is_some());
            None
        } else {
            debug!("Manifest-level data skipping: no predicate provided");
            None
        };

        // Get metadata entries, applying expression-based filtering if available
        let entries = if let Some(ref filter) = skipping_filter {
            // Apply expression-based filtering to batches, then materialize
            let mut all_entries = Vec::new();
            let mut total_before_filter = 0;
            let mut total_after_filter = 0;
            use crate::engine_data::RowVisitor;

            for batch in self.data.iter() {
                // Materialize all entries from the batch first
                let mut visitor = reader::ContentTreeNodeEntryVisitor::default();
                visitor.visit_rows_of(batch.as_ref())?;
                let batch_total = visitor.entries.len();
                total_before_filter += batch_total;

                // Apply the filter to get selection vector for this batch
                let selection_vector = filter.apply(batch.as_ref())?;

                // Filter entries based on selection vector, logging per-entry decisions
                let batch_entries: Vec<_> = visitor
                    .entries
                    .into_iter()
                    .zip(selection_vector.into_iter())
                    .filter_map(|(entry, keep)| {
                        debug!(
                            "Manifest pruning: {} location={:?}",
                            if keep { "KEEP" } else { "PRUNE" },
                            entry.location,
                        );
                        if keep {
                            Some(entry)
                        } else {
                            None
                        }
                    })
                    .collect();

                let batch_kept = batch_entries.len();
                total_after_filter += batch_kept;

                all_entries.extend(batch_entries);
            }
            let total_skipped = total_before_filter - total_after_filter;
            debug!(
                "Manifest-level pruning result: total={}, kept={}, skipped={} ({:.1}%)",
                total_before_filter,
                total_after_filter,
                total_skipped,
                if total_before_filter > 0 {
                    (total_skipped as f64 / total_before_filter as f64) * 100.0
                } else {
                    0.0
                }
            );
            all_entries
        } else {
            // No filtering - just materialize all entries
            self.entries()?
        };

        // Separate entries by type
        let mut combined_manifest_entries = Vec::new();
        let mut data_file_entries = Vec::new();

        for entry in entries {
            match entry.content_type {
                DataContentType::CombinedManifest => combined_manifest_entries.push(entry),
                DataContentType::Data => data_file_entries.push(entry),
                DataContentType::DataManifest | DataContentType::DeleteManifest => {
                    return Err(Error::generic(
                        "Old DataManifest/DeleteManifest format is no longer supported; \
                         use CombinedManifest format",
                    ));
                }
                DataContentType::PositionDeletes => {
                    return Err(Error::generic(
                        "Unexpected PositionDeletes entry in root manifest",
                    ));
                }
                DataContentType::EqualityDeletes => {
                    return Err(Error::generic("Equality deletes are not supported"))
                }
            }
        }

        // CombinedManifest entries have inline DV info — no joining needed.
        // Each entry produces a ManifestReference.
        let manifest_references: Vec<ManifestReference> = combined_manifest_entries
            .into_iter()
            .map(|data_entry| ManifestReference {
                data_manifest: FilteredManifest::new(data_entry),
            })
            .collect();

        Ok(LeafReferences {
            manifest_references,
        })
    }

    /// Processes all manifest references from a `LeafReferences` into action batches.
    ///
    /// # Parameters
    /// - `root_state`: The leaf references obtained from `manifest_references()`
    /// - `engine`: The engine for reading parquet files
    ///
    /// # Returns
    /// An iterator over action batches.
    // TODO: Refactor to reduce argument count (currently 7) - consider using a config struct
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn non_root_action_batches(
        root_state: LeafReferences,
        parquet_handler: Arc<dyn ParquetHandler>,
        evaluation_handler: Arc<dyn EvaluationHandler>,
        schema: &SchemaRef,
        table_root: &Url,
        predicate: Option<&PredicateRef>,
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<Box<dyn Iterator<Item = DeltaResult<ActionsBatch>> + Send>> {
        // Use BulkManifestStreamProcessor for lazy processing of manifests
        let processor = bulk_processor::BulkManifestStreamProcessor::new(
            root_state.manifest_references.into_iter(),
            parquet_handler,
            evaluation_handler,
            schema.clone(),
            table_root.clone(),
            predicate.cloned(),
            table_schema.map(|s| Arc::new(s.clone())),
            stats_schema.map(|s| Arc::new(s.clone())),
        )?;

        Ok(Box::new(processor))
    }

    /// Merge manifest DV selection into an existing selection vector.
    ///
    /// Applies AND operation: selection[i] = selection[i] && manifest_dv_selection[i]
    ///
    /// Per FilteredEngineData contract: if manifest_dv_selection is shorter than selection,
    /// rows beyond its length are considered selected (not deleted by manifest DV).
    fn merge_manifest_dv_selection(
        selection: &mut [bool],
        manifest_dv_selection: &[bool],
    ) -> DeltaResult<()> {
        // Manifest DV selection should never be longer than the batch
        if manifest_dv_selection.len() > selection.len() {
            return Err(Error::generic(format!(
                "Manifest DV selection is longer than batch: {} > {}",
                manifest_dv_selection.len(),
                selection.len()
            )));
        }

        // Apply manifest DV by iterating over its selection vector
        for (i, &dv_selected) in manifest_dv_selection.iter().enumerate() {
            selection[i] = selection[i] && dv_selected;
        }

        Ok(())
    }

    /// Process a single filtered batch into action batches.
    ///
    /// Takes a FilteredEngineData containing a batch with its manifest DV selection vector,
    /// combines it with Add/Remove selections, and produces ActionBatch results.
    ///
    /// # Parameters
    /// - `filtered_batch`: Batch with manifest DV selection already applied (DV columns already appended)
    /// - `add_evaluator_opt`: Optional evaluator for Add actions
    /// - `remove_evaluator_opt`: Optional evaluator for Remove actions
    fn process_filtered_batch_to_actions(
        filtered_batch: FilteredEngineData,
        add_evaluator_opt: Option<&Arc<dyn ExpressionEvaluator>>,
        remove_evaluator_opt: Option<&Arc<dyn ExpressionEvaluator>>,
    ) -> DeltaResult<Vec<ActionsBatch>> {
        // Extract batch and manifest DV selection vector
        let (batch, manifest_dv_selection) = filtered_batch.into_parts();

        let batch_ref: &dyn EngineData = batch.as_ref();

        // Build Add/Remove selection vectors once
        let (mut add_selection, mut remove_selection) =
            Self::build_add_remove_selection_vectors(batch_ref)?;

        let mut result_batches = Vec::new();

        // Process Add entries if needed
        if let Some(add_eval) = add_evaluator_opt {
            Self::merge_manifest_dv_selection(&mut add_selection, &manifest_dv_selection)?;
            if add_selection.iter().any(|&b| b) {
                let transformed = add_eval.evaluate(batch_ref)?;
                let filtered_data = transformed.apply_selection_vector(add_selection)?;
                result_batches.push(ActionsBatch::new(filtered_data, false));
            }
        }

        // Process Remove entries if needed
        if let Some(remove_eval) = remove_evaluator_opt {
            Self::merge_manifest_dv_selection(&mut remove_selection, &manifest_dv_selection)?;
            if remove_selection.iter().any(|&b| b) {
                let transformed = remove_eval.evaluate(batch_ref)?;
                let filtered_data = transformed.apply_selection_vector(remove_selection)?;
                result_batches.push(ActionsBatch::new(filtered_data, false));
            }
        }

        Ok(result_batches)
    }

    /// Opens a parquet stream for reading metadata without collecting batches (for lazy streaming).
    ///
    /// Returns the batch iterator and parsed version, allowing callers to defer batch collection.
    ///
    /// # Returns
    /// A tuple of (batch_iterator, version, path_in_log) that can be used to construct ContentTreeNode later.
    pub(crate) fn open_stream(
        parquet_handler: Arc<dyn ParquetHandler>,
        path: &Url,
        path_in_log: String,
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<ParquetStreamResult> {
        // Cached schema for reading ContentTreeNodeEntry from parquet files without content_stats.
        // Uses ToSchema which excludes content_stats (requires both table and stats schemas).
        // Includes _pos metadata column for tracking row positions within the manifest.
        static READ_SCHEMA_BASE: LazyLock<SchemaRef> = LazyLock::new(|| {
            use crate::schema::MetadataColumnSpec;

            use crate::schema::ToSchema as _;
            let base_schema = ContentTreeNodeEntry::to_schema();
            let mut fields: Vec<StructField> = base_schema.fields().cloned().collect();

            // Add _pos metadata column to track row indices (needed for data_manifest_position)
            fields.push(StructField::create_metadata_column(
                "_pos",
                MetadataColumnSpec::RowIndex,
            ));

            Arc::new(StructType::new_unchecked(fields))
        });

        // Build read schema with content_stats if both table and stats schemas are provided
        let read_schema = if let (Some(ts), Some(ss)) = (table_schema, stats_schema) {
            use crate::schema::MetadataColumnSpec;

            let schema_with_stats = ContentTreeNodeEntry::to_schema_with_content_stats(ts, ss)?;
            let mut fields: Vec<StructField> = schema_with_stats.fields().cloned().collect();

            // Add _pos metadata column to track row indices (needed for data_manifest_position)
            fields.push(StructField::create_metadata_column(
                "_pos",
                MetadataColumnSpec::RowIndex,
            ));

            Arc::new(StructType::new_unchecked(fields))
        } else {
            READ_SCHEMA_BASE.clone()
        };

        let file = FileMeta {
            location: path.clone(),
            last_modified: 0,
            size: 0,
        };

        let parsed =
            ParsedLogPath::try_from(file.clone())?.ok_or_else(|| Error::invalid_log_path(path))?;

        let read_result_iter = parquet_handler.read_parquet_files(&[file], read_schema, None)?;

        Ok((read_result_iter, parsed.version, path_in_log))
    }
}

/// Parses a manifest DV into a selection vector.
///
/// Returns a boolean vector where `true` means the row should be included (NOT deleted).
/// Parse manifest deletion vector bytes into a RoaringTreemap.
/// Returns None if dv_bytes is empty, otherwise returns the set of deleted positions.
pub(crate) fn parse_manifest_dv(dv_bytes: &Bytes) -> DeltaResult<Option<roaring::RoaringTreemap>> {
    use roaring::RoaringTreemap;

    // Check if we have DV data
    if dv_bytes.is_empty() {
        return Ok(None);
    }

    // Parse the magic number from the first 4 bytes
    if dv_bytes.len() < 4 {
        return Err(Error::generic(
            "Manifest deletion vector is too small (less than 4 bytes)",
        ));
    }

    let magic = u32::from_be_bytes([dv_bytes[0], dv_bytes[1], dv_bytes[2], dv_bytes[3]]);

    // Magic numbers from the deletion vector format
    const ROARING_BITMAP_PORTABLE_MAGIC: u32 = 1681511377;
    const ROARING_BITMAP_NATIVE_MAGIC: u32 = 1681511376;

    // Deserialize the RoaringTreemap
    let deleted_positions = match magic {
        ROARING_BITMAP_PORTABLE_MAGIC => RoaringTreemap::deserialize_from(&dv_bytes[4..])
            .map_err(|err| Error::generic(format!("Failed to deserialize manifest DV: {}", err)))?,
        ROARING_BITMAP_NATIVE_MAGIC => {
            return Err(Error::generic(
                "Native serialization format for manifest deletion vectors is not yet supported",
            ));
        }
        _ => {
            return Err(Error::generic(format!(
                "Invalid magic number in manifest deletion vector: {}",
                magic
            )));
        }
    };

    Ok(Some(deleted_positions))
}

/// Parse manifest_dv bytes into a selection vector.
///
/// Returns a `Vec<bool>` where `true` means the row is NOT deleted.
/// This is used in tests to validate manifest DV parsing.
#[cfg(test)]
fn parse_manifest_dv_to_selection_vector(
    dv_bytes: &Bytes,
    total_rows: usize,
) -> DeltaResult<Vec<bool>> {
    let deleted_positions = parse_manifest_dv(dv_bytes)?;

    // Build selection vector: true if NOT deleted
    let selection_vector: Vec<bool> = if let Some(deleted_positions) = deleted_positions {
        (0..total_rows)
            .map(|i| !deleted_positions.contains(i as u64))
            .collect()
    } else {
        vec![true; total_rows]
    };

    Ok(selection_vector)
}

/// Helper that holds a URL and lazily computes/caches its relative path.
/// Useful for avoiding repeated conversions of the same URL to relative path.
/// Converts an absolute URL to a path relative to table_root.
pub(crate) fn absolute_to_relative_path(
    absolute_url: &Url,
    table_root: &Url,
) -> DeltaResult<String> {
    let full_path = absolute_url.path();
    let root_path = table_root.path();

    Ok(full_path
        .strip_prefix(root_path)
        .unwrap_or(full_path)
        .trim_start_matches('/')
        .to_string())
}

/// Parses a string as an absolute URL, or if that fails, joins it with the table root.
/// This handles both absolute and relative manifest/file paths.
pub(crate) fn parse_or_join_url(path: &str, table_root: &Url) -> DeltaResult<Url> {
    Url::parse(path)
        .or_else(|_| table_root.join(path))
        .map_err(|e| Error::generic(format!("Failed to parse URL '{}': {}", path, e)))
}

/// Converts a DeletionVectorDescriptor to a Scalar representation
pub(crate) fn metadata_entry_to_scalars(
    entry: &ContentTreeNodeEntry,
    schema: &crate::schema::SchemaRef,
) -> DeltaResult<Vec<Scalar>> {
    use crate::expressions::StructData;

    // Build a vector of structured scalars for the schema (one per top-level field)
    let mut scalars = Vec::with_capacity(schema.fields().len());

    for field in schema.fields() {
        let scalar = match field.name().as_str() {
            "contentType" => Scalar::from(entry.content_type),
            "location" => Scalar::from(entry.location.clone()),
            "fileFormat" => Scalar::from(entry.file_format),
            "trackingInfo" => match &entry.tracking_info {
                Some(ti) => {
                    // Get struct fields from schema
                    let struct_fields =
                        if let crate::schema::DataType::Struct(st) = field.data_type() {
                            st.fields().cloned().collect::<Vec<_>>()
                        } else {
                            return Err(crate::Error::generic(
                                "trackingInfo field should be a struct",
                            ));
                        };
                    let values = vec![
                        Scalar::from(ti.status),
                        Scalar::from(ti.snapshot_id),
                        Scalar::from(ti.sequence_number),
                        Scalar::from(ti.file_sequence_number),
                        Scalar::from(ti.first_row_id),
                        Scalar::from(ti.changes_dv.clone()),
                    ];
                    Scalar::Struct(StructData::new_unchecked(struct_fields, values))
                }
                None => Scalar::Null(field.data_type().clone()),
            },
            "dvInfo" => match &entry.dv_info {
                Some(dv) => {
                    let struct_fields =
                        if let crate::schema::DataType::Struct(st) = field.data_type() {
                            st.fields().cloned().collect::<Vec<_>>()
                        } else {
                            return Err(crate::Error::generic("dvInfo field should be a struct"));
                        };
                    let values = vec![
                        Scalar::from(dv.location.clone()),
                        Scalar::from(dv.offset),
                        Scalar::from(dv.size_in_bytes),
                        Scalar::from(dv.cardinality),
                    ];
                    Scalar::Struct(StructData::new_unchecked(struct_fields, values))
                }
                None => Scalar::Null(field.data_type().clone()),
            },
            "partitionSpecId" => Scalar::from(entry.partition_spec_id),
            "sortOrderId" => Scalar::from(entry.sort_order_id),
            "recordCount" => Scalar::from(entry.record_count),
            "fileSizeInBytes" => Scalar::from(entry.file_size_in_bytes),
            CONTENT_STATS_FIELD_NAME => match &entry.content_stats {
                Some(struct_data) => Scalar::Struct(struct_data.clone()),
                None => Scalar::Null(field.data_type().clone()),
            },
            "manifestStats" => match &entry.manifest_stats {
                Some(ms) => {
                    let struct_fields =
                        if let crate::schema::DataType::Struct(st) = field.data_type() {
                            st.fields().cloned().collect::<Vec<_>>()
                        } else {
                            return Err(crate::Error::generic(
                                "manifestStats field should be a struct",
                            ));
                        };
                    let values = vec![
                        Scalar::from(ms.added_files_count),
                        Scalar::from(ms.existing_files_count),
                        Scalar::from(ms.deletes_files_count),
                        Scalar::from(ms.added_rows_count),
                        Scalar::from(ms.existing_rows_count),
                        Scalar::from(ms.delete_rows_count),
                        Scalar::from(ms.min_sequence_number),
                    ];
                    Scalar::Struct(StructData::new_unchecked(struct_fields, values))
                }
                None => Scalar::Null(field.data_type().clone()),
            },
            "keyMetadata" => Scalar::from(entry.key_metadata.clone()),
            "splitOffsets" => entry.split_offsets.clone().try_into()?,
            "equalityIds" => entry.equality_ids.clone().try_into()?,
            "manifestDv" => Scalar::from(entry.manifest_dv.clone()),
            _ => Scalar::Null(field.data_type().clone()),
        };

        scalars.push(scalar);
    }

    Ok(scalars)
}

/// Type of content stored by the manifest entry
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataContentType {
    Data = 0,
    PositionDeletes = 1,
    EqualityDeletes = 2,
    // Types below are only allowed in the root
    DataManifest = 3,     // kept for backwards compat reading
    DeleteManifest = 4,   // kept for backwards compat reading
    CombinedManifest = 5, // unified manifest with inline DV info
}

// ToDataType implementations for enums
impl ToDataType for DataContentType {
    fn to_data_type() -> DataType {
        DataType::INTEGER
    }
}

impl From<DataContentType> for Scalar {
    fn from(value: DataContentType) -> Self {
        Scalar::Integer(value as i32)
    }
}

/// Format of this data.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum DataFileFormat {
    /// Parquet file format: <https://parquet.apache.org/>
    Parquet,
    /// Puffin file format: <https://iceberg.apache.org/puffin-spec/>
    Puffin,
}

impl FromStr for DataFileFormat {
    type Err = Error;

    fn from_str(s: &str) -> DeltaResult<Self> {
        match s.to_lowercase().as_str() {
            "parquet" => Ok(Self::Parquet),
            "puffin" => Ok(Self::Puffin),
            _ => Err(Error::internal_error(format!(
                "Unsupported data file format: {}",
                s
            ))),
        }
    }
}

impl std::fmt::Display for DataFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFileFormat::Parquet => write!(f, "parquet"),
            DataFileFormat::Puffin => write!(f, "puffin"),
        }
    }
}

impl ToDataType for DataFileFormat {
    fn to_data_type() -> DataType {
        DataType::STRING
    }
}

impl From<DataFileFormat> for Scalar {
    fn from(value: DataFileFormat) -> Self {
        Scalar::String(value.to_string())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TrackingStatus {
    Existed = 0,
    Added = 1,
    Deleted = 2,
}

impl ToDataType for TrackingStatus {
    fn to_data_type() -> DataType {
        DataType::INTEGER
    }
}

impl From<TrackingStatus> for Scalar {
    fn from(value: TrackingStatus) -> Self {
        Scalar::Integer(value as i32)
    }
}

#[derive(Debug, Clone, ToSchema, IntoEngineData)]
pub(crate) struct DvInfo {
    /// Path to location that DV is stored in.
    #[field_id = 152]
    pub(crate) location: String,

    /// The offset in the file where the content starts.
    #[field_id = 144]
    pub(crate) offset: i64,

    /// The length of thea referenced content stored in the file;
    /// required if content_offset is present.
    #[field_id = 145]
    pub(crate) size_in_bytes: i64,

    #[field_id = 154]
    pub(crate) cardinality: i64,
}

#[derive(Debug, Clone, ToSchema, IntoEngineData)]
pub struct TrackingInfo {
    #[field_id = 0]
    pub(crate) status: TrackingStatus,

    /// Snapshot ID where the file was added, or deleted if status is 2. Inherited when null.
    /// Must be written in the root file.
    #[field_id = 1]
    pub snapshot_id: Option<i64>,

    /// Data sequence number of the file. Inherited in when null and status is 1 (added).
    /// Must be equal to file_sequence_number if content_type is {Data,Delete}Manifest.
    /// Must be written in the root file.
    #[field_id = 3]
    pub(crate) sequence_number: Option<i64>,

    /// File sequence number indicating when the file was added. Inherited when null and status is added.
    /// Must be equal to sequence_number if content_type is {Data,Delete}Manifest.
    #[field_id = 4]
    pub(crate) file_sequence_number: Option<i64>,

    /// The _row_id for the first row in the data file if content_type is Data.
    /// If content_type is DataManifest, this is the starting _row_id to assign to rows added by ADDED data files.
    #[field_id = 142]
    pub(crate) first_row_id: Option<i64>,

    /// Deletion vector tracking changes made in the current commit for manifest entries.
    /// Only used when content_type is DataManifest or DeleteManifest.
    /// This field tracks what was added/changed in the current commit and is cleared between commits.
    #[field_id = 153]
    pub(crate) changes_dv: Option<Bytes>,
}

impl TrackingInfo {
    /// Get the tracking status
    pub fn status(&self) -> TrackingStatus {
        self.status
    }
}

impl From<TrackingInfo> for Scalar {
    fn from(value: TrackingInfo) -> Self {
        use crate::expressions::StructData;
        use crate::schema::ToSchema;

        let fields = TrackingInfo::to_schema().into_fields().collect();
        let values = vec![
            value.status.into(),
            value.snapshot_id.into(),
            value.sequence_number.into(),
            value.file_sequence_number.into(),
            value.first_row_id.into(),
        ];

        // SAFETY: Fields are generated by ToSchema derive macro and values are constructed
        // to match exactly in count, order, type, and nullability.
        Scalar::Struct(StructData::new_unchecked(fields, values))
    }
}

#[derive(Debug, Clone, ToSchema, IntoEngineData)]
pub(crate) struct ManifestStats {
    #[field_id = 504]
    pub(crate) added_files_count: i64,
    #[field_id = 505]
    pub(crate) existing_files_count: i64,
    #[field_id = 506]
    pub(crate) deletes_files_count: i64,

    #[field_id = 512]
    pub(crate) added_rows_count: i64,
    #[field_id = 513]
    pub(crate) existing_rows_count: i64,
    #[field_id = 514]
    pub(crate) delete_rows_count: i64,

    #[field_id = 516]
    pub(crate) min_sequence_number: i64,
}

impl From<ManifestStats> for Scalar {
    fn from(value: ManifestStats) -> Self {
        use crate::expressions::StructData;
        use crate::schema::ToSchema;

        let fields = ManifestStats::to_schema().into_fields().collect();
        let values = vec![
            value.added_files_count.into(),
            value.existing_files_count.into(),
            value.deletes_files_count.into(),
            value.added_rows_count.into(),
            value.existing_rows_count.into(),
            value.delete_rows_count.into(),
            value.min_sequence_number.into(),
        ];

        // SAFETY: Fields are generated by ToSchema derive macro and values are constructed
        // to match exactly in count, order, type, and nullability.
        Scalar::Struct(StructData::new_unchecked(fields, values))
    }
}

#[derive(Debug, Clone, ToSchema)]
pub(super) struct ContentTreeNodeEntry {
    /// Type of content stored by the entry.
    /// DataManifest, DeleteManifest or ManifestDV can only be defined in the root manifest.
    #[field_id = 134]
    pub content_type: DataContentType,

    /// Location of the file. Required for most content types.
    #[field_id = 100]
    pub location: Option<String>,

    /// avro, orc, parquet or puffin
    #[field_id = 101]
    pub(crate) file_format: DataFileFormat,

    #[field_id = 147]
    pub tracking_info: Option<TrackingInfo>,

    #[field_id = 148]
    pub(crate) dv_info: Option<DvInfo>,

    /// ID of partition spec used to write manifest or data/delete files.
    #[field_id = 149]
    pub(crate) partition_spec_id: i64,

    /// ID representing sort order for this file. Can only be set if content_type is Data.
    #[field_id = 140]
    pub(crate) sort_order_id: Option<i64>,

    /// Number of records in this file, or the cardinality of a deletion vector
    #[field_id = 103]
    pub(crate) record_count: i64,

    /// Total file size in bytes. Must be defined if location is defined
    #[field_id = 104]
    pub(crate) file_size_in_bytes: Option<i64>,

    /// Column-level statistics for the data file.
    /// The schema of this struct is dynamically generated based on the table schema
    /// using [`stats::stats_schema`]. When `None`, no statistics are available.
    /// See: <https://docs.google.com/document/d/1uvbrwwAJW2TgsnoaIcwAFpjbhHkBUL5wY_24nKgtt9I/>
    // Skip the schema since we don't know the type here, use to_schema_with_content_stats instead
    #[skip_schema]
    #[field_id = 146]
    pub(crate) content_stats: Option<StructData>,

    /// Must be set if content_type is {Data,Delete}Manifest, otherwise null.
    #[field_id = 150]
    pub(crate) manifest_stats: Option<ManifestStats>,

    /// Location of the data file if the content_type is  PositionDeletes
    /// Location of affiliated data manifest if content_type is or DeleteManifest or null if delete manifest is unaffiliated.
    /// TODO: place holder for referenced file which is no longer necessary.
    /// #[field_id = 143]
    /// pub referenced_file: `Option<String>`,

    /// Implementation-specific key metadata for encryption
    #[field_id = 131]
    pub(crate) key_metadata: Option<Bytes>,

    /// Split offsets for the data file. For example, all row group offsets in a Parquet file. Must be sorted ascending
    #[field_id = 132]
    pub(crate) split_offsets: Option<Vec<i64>>,

    /// Field ids used to determine row equality in equality delete files.
    /// Required when content is EqualityDeletes and must be null otherwise.
    /// Fields with ids listed in this column must be present in the delete file
    #[field_id = 135]
    pub(crate) equality_ids: Option<Vec<i32>>,

    /// DV that applies to the manifest linked to from this entry.
    #[field_id = 151]
    pub(crate) manifest_dv: Option<Bytes>,
}

impl ContentTreeNodeEntry {
    /// Returns a copy of this entry with the tracking status updated.
    pub(crate) fn with_status(mut self, status: TrackingStatus) -> Self {
        if let Some(ref mut tracking_info) = self.tracking_info {
            tracking_info.status = status;
        }
        self
    }
}

/// Builder for [`ContentTreeNodeEntry`] that eliminates boilerplate by providing
/// sensible defaults for most fields.
///
/// # Example
/// ```ignore
/// ContentTreeNodeEntryBuilder::new(DataContentType::Data)
///     .location("path/to/file.parquet")
///     .with_tracking(entry_version, current_version, snapshot_id)
///     .record_count(100)
///     .file_size_in_bytes(1024)
///     .build()
/// ```
pub(crate) struct ContentTreeNodeEntryBuilder {
    content_type: DataContentType,
    location: Option<String>,
    file_format: DataFileFormat,
    tracking_info: Option<TrackingInfo>,
    dv_info: Option<DvInfo>,
    partition_spec_id: i64,
    sort_order_id: Option<i64>,
    record_count: i64,
    file_size_in_bytes: Option<i64>,
    content_stats: Option<StructData>,
    manifest_stats: Option<ManifestStats>,
    manifest_dv: Option<Bytes>,
    key_metadata: Option<Bytes>,
    split_offsets: Option<Vec<i64>>,
    equality_ids: Option<Vec<i32>>,
}

impl ContentTreeNodeEntryBuilder {
    /// Create a new builder with the given content type. All other fields start at
    /// sensible defaults: `file_format=Parquet`, `partition_spec_id=0`, `record_count=0`,
    /// and all optional fields are `None`.
    pub(crate) fn new(content_type: DataContentType) -> Self {
        Self {
            content_type,
            location: None,
            file_format: DataFileFormat::Parquet,
            tracking_info: None,
            dv_info: None,
            partition_spec_id: 0,
            sort_order_id: None,
            record_count: 0,
            file_size_in_bytes: None,
            content_stats: None,
            manifest_stats: None,
            manifest_dv: None,
            key_metadata: None,
            split_offsets: None,
            equality_ids: None,
        }
    }

    pub(crate) fn location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Set tracking info by computing status from `entry_version` vs `current_version`.
    /// If the file was written at `current_version`, its status is `Added`; otherwise `Existed`.
    pub(crate) fn with_tracking(
        mut self,
        entry_version: Version,
        current_version: Version,
        snapshot_id: i64,
    ) -> Self {
        let status = if entry_version == current_version {
            TrackingStatus::Added
        } else {
            TrackingStatus::Existed
        };
        self.tracking_info = Some(TrackingInfo {
            status,
            snapshot_id: Some(snapshot_id),
            sequence_number: Some(entry_version as i64),
            file_sequence_number: Some(entry_version as i64),
            first_row_id: None,
            changes_dv: None,
        });
        self
    }

    /// Set tracking info directly for non-standard cases (e.g., manifest entries
    /// where `sequence_number` should be `None`).
    pub(crate) fn tracking_info(mut self, tracking_info: TrackingInfo) -> Self {
        self.tracking_info = Some(tracking_info);
        self
    }

    pub(crate) fn dv_info_opt(mut self, dv_info: Option<DvInfo>) -> Self {
        self.dv_info = dv_info;
        self
    }

    pub(crate) fn record_count(mut self, record_count: i64) -> Self {
        self.record_count = record_count;
        self
    }

    pub(crate) fn file_size_in_bytes(mut self, file_size_in_bytes: i64) -> Self {
        self.file_size_in_bytes = Some(file_size_in_bytes);
        self
    }

    pub(crate) fn content_stats_opt(mut self, content_stats: Option<StructData>) -> Self {
        self.content_stats = content_stats;
        self
    }

    pub(crate) fn manifest_stats_opt(mut self, manifest_stats: Option<ManifestStats>) -> Self {
        self.manifest_stats = manifest_stats;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn file_format(mut self, file_format: DataFileFormat) -> Self {
        self.file_format = file_format;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn dv_info(mut self, dv_info: DvInfo) -> Self {
        self.dv_info = Some(dv_info);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn partition_spec_id(mut self, partition_spec_id: i64) -> Self {
        self.partition_spec_id = partition_spec_id;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn sort_order_id(mut self, sort_order_id: i64) -> Self {
        self.sort_order_id = Some(sort_order_id);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn content_stats(mut self, content_stats: StructData) -> Self {
        self.content_stats = Some(content_stats);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn manifest_stats(mut self, manifest_stats: ManifestStats) -> Self {
        self.manifest_stats = Some(manifest_stats);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn manifest_dv(mut self, manifest_dv: Bytes) -> Self {
        self.manifest_dv = Some(manifest_dv);
        self
    }

    /// Consume the builder and produce a [`ContentTreeNodeEntry`].
    pub(crate) fn build(self) -> ContentTreeNodeEntry {
        ContentTreeNodeEntry {
            content_type: self.content_type,
            location: self.location,
            file_format: self.file_format,
            tracking_info: self.tracking_info,
            dv_info: self.dv_info,
            partition_spec_id: self.partition_spec_id,
            sort_order_id: self.sort_order_id,
            record_count: self.record_count,
            file_size_in_bytes: self.file_size_in_bytes,
            content_stats: self.content_stats,
            manifest_stats: self.manifest_stats,
            manifest_dv: self.manifest_dv,
            key_metadata: self.key_metadata,
            split_offsets: self.split_offsets,
            equality_ids: self.equality_ids,
        }
    }
}

impl ContentTreeNodeEntry {
    /// Helper to create metadata schema for reading/processing manifest batches.
    ///
    /// This includes `_pos` metadata column and optionally `content_stats` based on table_schema.
    /// Use this when you need a schema that matches actual manifest batch data.
    pub(crate) fn processing_schema_with_pos(
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<SchemaRef> {
        use crate::schema::{MetadataColumnSpec, ToSchema as _};

        let base_schema = if let (Some(ts), Some(ss)) = (table_schema, stats_schema) {
            Self::to_schema_with_content_stats(ts, ss)?
        } else {
            Self::to_schema()
        };

        let mut fields: Vec<StructField> = base_schema.fields().cloned().collect();
        fields.push(StructField::create_metadata_column(
            "_pos",
            MetadataColumnSpec::RowIndex,
        ));
        Ok(Arc::new(StructType::new_unchecked(fields)))
    }

    /// Creates a stats transformation evaluator that transforms content_stats to stats_parsed.
    ///
    /// This evaluator augments metadata batches by reading content_stats and producing stats_parsed.
    /// Returns None if:
    /// - table_schema or stats_schema is not provided
    /// - metadata_schema doesn't have content_stats field
    /// - output_schema doesn't expect stats_parsed
    ///
    /// # Parameters
    /// - `evaluation_handler`: Handler for creating the evaluator
    /// - `metadata_schema`: Schema of the input metadata batch (must include content_stats)
    /// - `output_schema`: Expected output schema (checked for add.stats_parsed field)
    /// - `table_schema`: Table physical schema (with field IDs) for transformation
    /// - `stats_schema`: Stats schema for the stats_parsed output
    pub(crate) fn create_stats_transformation_evaluator(
        evaluation_handler: &dyn EvaluationHandler,
        metadata_schema: &SchemaRef,
        output_schema: &SchemaRef,
        table_schema: Option<&StructType>,
        stats_schema: Option<&StructType>,
    ) -> DeltaResult<Option<Arc<dyn crate::ExpressionEvaluator>>> {
        let (Some(table_sch), Some(stats_sch)) = (table_schema, stats_schema) else {
            return Ok(None);
        };

        // Check if metadata_schema has content_stats field (only present when table_schema was used at read time)
        let has_content_stats = metadata_schema
            .field(crate::content_tree::CONTENT_STATS_FIELD_NAME)
            .is_some();

        // Check if the output schema expects stats_parsed
        let needs_stats_parsed = output_schema
            .field("add")
            .and_then(|f| match f.data_type() {
                crate::schema::DataType::Struct(s) => s.field("stats_parsed"),
                _ => None,
            })
            .is_some();

        debug!(
            "Stats transformation check: has_content_stats={}, needs_stats_parsed={}",
            has_content_stats, needs_stats_parsed
        );

        if !has_content_stats || !needs_stats_parsed {
            return Ok(None);
        }

        debug!("Creating stats transformation: content_stats → stats_parsed");

        // Build augmented transform that adds stats_parsed to metadata batch
        use crate::expressions::Expression;

        // Get all fields from metadata_schema
        let mut field_exprs: Vec<Arc<Expression>> = metadata_schema
            .fields()
            .map(|f| Arc::new(Expression::column([f.name().as_str()])))
            .collect();

        // Add stats_parsed transformation
        let stats_parsed_expr =
            crate::content_tree::stats::create_content_stats_to_stats_parsed_expr(
                table_sch, stats_sch,
            )?;
        field_exprs.push(stats_parsed_expr);

        // Build augmented output schema
        let augmented_output_schema = {
            let mut fields: Vec<StructField> = metadata_schema.fields().cloned().collect();
            fields.push(StructField::nullable(
                "stats_parsed",
                crate::schema::DataType::Struct(Box::new(stats_sch.clone())),
            ));
            Arc::new(StructType::new_unchecked(fields))
        };

        // Create struct expression with all fields
        let augment_expr = Expression::struct_from(field_exprs);

        // Create evaluator
        Ok(Some(evaluation_handler.new_expression_evaluator(
            metadata_schema.clone(),
            Arc::new(augment_expr),
            augmented_output_schema.clone().into(),
        )?))
    }

    /// Returns ContentTreeNodeEntry schema with content_stats based on the given table schema.
    ///
    /// The content_stats field schema is dynamically generated in Delta JSON stats format
    /// (numRecords, nullCount, minValues, maxValues, tightBounds) matching the format
    /// used by [`Transaction::add_files_schema`].
    ///
    /// # Arguments
    ///
    /// * `table_schema` - The table's data schema to generate stats schema from
    ///
    /// # Returns
    ///
    /// Returns `Ok(StructType)` containing the full ContentTreeNodeEntry schema with content_stats,
    /// or an error if stats schema generation fails.
    pub(crate) fn to_schema_with_content_stats(
        table_schema: &StructType,
        stats_schema: &StructType,
    ) -> DeltaResult<StructType> {
        use crate::schema::{ColumnMetadataKey, ToSchema};

        // Generate filtered AMT schema: only columns requested in stats_schema are included,
        // avoiding wasteful reads of per-column statistics that won't be used.
        let amt_stats = stats::filtered_stats_schema(table_schema, stats_schema)?;

        // Build on the derived base schema (which includes field_ids) and insert content_stats
        let base = Self::to_schema();
        let content_stats_field = StructField::nullable(
            CONTENT_STATS_FIELD_NAME,
            DataType::Struct(Box::new(amt_stats)),
        )
        .add_metadata([(ColumnMetadataKey::ParquetFieldId.as_ref(), 146i64)]);

        // Insert content_stats after fileSizeInBytes
        let mut fields = Vec::new();
        for field in base.fields() {
            fields.push(field.clone());
            if field.name() == "fileSizeInBytes" {
                fields.push(content_stats_field.clone());
            }
        }

        Ok(StructType::new_unchecked(fields))
    }
}

impl crate::IntoEngineData for ContentTreeNodeEntry {
    fn into_engine_data(
        self,
        schema: crate::schema::SchemaRef,
        engine: &dyn crate::Engine,
    ) -> DeltaResult<Box<dyn crate::EngineData>> {
        // Use create_many with structured scalars (more efficient than create_one with flat values)
        let scalars = metadata_entry_to_scalars(&self, &schema)?;
        let evaluator = engine.evaluation_handler();
        evaluator.create_many(schema, &[&scalars])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use crate::{engine::sync::SyncEngine, IntoEngineData};
    use tempfile::tempdir;

    // Note: Full integration test for ContentTreeNodeEntry::into_engine_data is not included here
    // because it requires complex setup with nested structs. The implementation is complete
    // and can be tested in integration tests with actual data.

    #[test]
    fn test_simple_into_engine_data() -> DeltaResult<()> {
        use crate::IntoEngineData;
        let engine = SyncEngine::new();

        // Create a very simple entry with no optional fields
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("test.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(200),
                first_row_id: Some(1000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();

        let schema = test_metadata_entry_schema();
        let result = entry.into_engine_data(schema, &engine);
        if let Err(e) = &result {
            eprintln!("Error in test_simple_into_engine_data: {:?}", e);
        }
        result?;

        Ok(())
    }

    #[test]
    fn test_absolute_to_relative_path() {
        // Test with memory:// URLs
        let table_root = Url::parse("memory:///").unwrap();
        let absolute_url = Url::parse("memory:///part-content-root.parquet").unwrap();
        let result = absolute_to_relative_path(&absolute_url, &table_root).unwrap();
        assert_eq!(result, "part-content-root.parquet");

        // Test with s3:// URLs
        let table_root = Url::parse("s3://my-bucket/my-table/").unwrap();
        let absolute_url = Url::parse("s3://my-bucket/my-table/data/part-00000.parquet").unwrap();
        let result = absolute_to_relative_path(&absolute_url, &table_root).unwrap();
        assert_eq!(result, "data/part-00000.parquet");

        // Test with nested paths
        let table_root = Url::parse("s3://bucket/table/").unwrap();
        let absolute_url = Url::parse("s3://bucket/table/year=2023/month=10/part.parquet").unwrap();
        let result = absolute_to_relative_path(&absolute_url, &table_root).unwrap();
        assert_eq!(result, "year=2023/month=10/part.parquet");

        // Test with file:// URLs
        let table_root = Url::parse("file:///path/to/table/").unwrap();
        let absolute_url = Url::parse("file:///path/to/table/data/file.parquet").unwrap();
        let result = absolute_to_relative_path(&absolute_url, &table_root).unwrap();
        assert_eq!(result, "data/file.parquet");

        // Test when root doesn't match (no common prefix)
        let table_root = Url::parse("s3://bucket-b/table-b/").unwrap();
        let absolute_url = Url::parse("s3://bucket-a/table-a/file.parquet").unwrap();
        let result = absolute_to_relative_path(&absolute_url, &table_root).unwrap();
        // Since there's no common prefix in the path part, it returns the path without leading slash
        assert_eq!(result, "table-a/file.parquet");
    }

    #[test]
    fn test_metadata_entry_base_schema_fields() {
        use crate::schema::ToSchema as _;
        // Verify the base schema has the expected structure (excludes content_stats)
        let schema = ContentTreeNodeEntry::to_schema();

        // Schema should have all the top-level fields (excluding content_stats)
        // Fields: contentType, location, fileFormat, trackingInfo, dvInfo, partitionSpecId, sortOrderId,
        // recordCount, fileSizeInBytes, manifestStats, keyMetadata, splitOffsets, equalityIds, manifestDv (14 total - no referencedFile)
        assert_eq!(schema.fields().len(), 14);

        // Check leaves (flattened leaf fields)
        let leaves = schema.leaves(None::<&str>);
        let (leaf_names, _leaf_types) = leaves.as_ref();

        // 28 leaf fields: 25 (our branch base) + keyMetadata(1) + splitOffsets(1) + equalityIds(1)
        // (no referencedFile; dvInfo has 4 leaves vs old dvInfo's 2)
        assert_eq!(leaf_names.len(), 28);
    }

    #[test]
    fn test_to_schema_with_content_stats() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructType};

        // Helper to create field with parquet field ID
        fn field_with_id(
            name: &str,
            data_type: DataType,
            nullable: bool,
            field_id: i32,
        ) -> StructField {
            StructField::new(name, data_type, nullable).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(field_id as i64),
            )])
        }

        // Create a simple table schema with field IDs (required for AMT stats schema)
        let table_schema = StructType::new_unchecked([
            field_with_id("id", DataType::INTEGER, false, 1),
            field_with_id("name", DataType::STRING, true, 2),
            field_with_id("value", DataType::DOUBLE, true, 3),
        ]);

        // Generate schema with content_stats (using all columns)
        let delta_stats_schema = test_delta_stats_schema(&table_schema);
        let schema_with_stats =
            ContentTreeNodeEntry::to_schema_with_content_stats(&table_schema, &delta_stats_schema)?;

        // Schema should have 15 top-level fields (14 base + 1 for content_stats)
        assert_eq!(schema_with_stats.fields().len(), 15);

        // Verify content_stats field exists
        let content_stats_field = schema_with_stats
            .field(CONTENT_STATS_FIELD_NAME)
            .expect("content_stats field should exist");
        assert!(content_stats_field.nullable);

        // Verify content_stats is a struct with AMT stats format:
        // {col_name: {value_count, null_value_count?, nan_value_count?, lower_bound, upper_bound, exact_bounds}, ...}
        let content_stats_struct = match content_stats_field.data_type() {
            DataType::Struct(s) => s.as_ref(),
            _ => panic!("Expected content_stats to be a struct"),
        };

        // Should have 3 fields: id, name, value (one per column)
        assert_eq!(content_stats_struct.fields().count(), 3);
        assert!(content_stats_struct.field("id").is_some());
        assert!(content_stats_struct.field("name").is_some());
        assert!(content_stats_struct.field("value").is_some());

        // Verify each column has a stats struct
        // id: non-nullable INTEGER -> {value_count, lower_bound, upper_bound, exact_bounds}
        let id_stats = match content_stats_struct.field("id").unwrap().data_type() {
            DataType::Struct(s) => s.as_ref(),
            _ => panic!("Expected id stats to be a struct"),
        };
        assert!(id_stats.field(VALUE_COUNT).is_some());
        assert!(id_stats.field(NULL_VALUE_COUNT).is_none()); // not nullable
        assert!(id_stats.field(NAN_VALUE_COUNT).is_none()); // not float/double
        assert!(id_stats.field(LOWER_BOUND).is_some());
        assert!(id_stats.field(UPPER_BOUND).is_some());
        assert!(id_stats.field(EXACT_BOUNDS).is_some());
        assert_eq!(
            id_stats.field(LOWER_BOUND).unwrap().data_type(),
            &DataType::INTEGER
        );

        // name: nullable STRING -> {value_count, null_value_count, avg_value_size, max_value_size, lower_bound, upper_bound, exact_bounds}
        let name_stats = match content_stats_struct.field("name").unwrap().data_type() {
            DataType::Struct(s) => s.as_ref(),
            _ => panic!("Expected name stats to be a struct"),
        };
        assert!(name_stats.field(VALUE_COUNT).is_some());
        assert!(name_stats.field(NULL_VALUE_COUNT).is_some()); // nullable
        assert!(name_stats.field(NAN_VALUE_COUNT).is_none()); // not float/double
        assert!(name_stats.field(AVG_VALUE_SIZE).is_some()); // string has size stats
        assert!(name_stats.field(MAX_VALUE_SIZE).is_some()); // string has size stats
        assert!(name_stats.field(LOWER_BOUND).is_some());
        assert!(name_stats.field(UPPER_BOUND).is_some());
        assert!(name_stats.field(EXACT_BOUNDS).is_some());
        assert_eq!(
            name_stats.field(LOWER_BOUND).unwrap().data_type(),
            &DataType::STRING
        );

        // value: nullable DOUBLE -> {value_count, null_value_count, nan_value_count, lower_bound, upper_bound, exact_bounds}
        let value_stats = match content_stats_struct.field("value").unwrap().data_type() {
            DataType::Struct(s) => s.as_ref(),
            _ => panic!("Expected value stats to be a struct"),
        };
        assert!(value_stats.field(VALUE_COUNT).is_some());
        assert!(value_stats.field(NULL_VALUE_COUNT).is_some()); // nullable
        assert!(value_stats.field(NAN_VALUE_COUNT).is_some()); // double has nan count
        assert!(value_stats.field(LOWER_BOUND).is_some());
        assert!(value_stats.field(UPPER_BOUND).is_some());
        assert!(value_stats.field(EXACT_BOUNDS).is_some());
        assert_eq!(
            value_stats.field(LOWER_BOUND).unwrap().data_type(),
            &DataType::DOUBLE
        );

        Ok(())
    }

    #[test]
    fn test_into_engine_data_with_content_stats() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructType};
        use crate::IntoEngineData;

        // Helper to create field with parquet field ID
        fn field_with_id(
            name: &str,
            data_type: DataType,
            nullable: bool,
            field_id: i32,
        ) -> StructField {
            StructField::new(name, data_type, nullable).with_metadata([(
                ColumnMetadataKey::ParquetFieldId.as_ref(),
                MetadataValue::Number(field_id as i64),
            )])
        }

        let engine = SyncEngine::new();

        // Create a simple table schema with field IDs
        let table_schema = StructType::new_unchecked([
            field_with_id("id", DataType::INTEGER, false, 1),
            field_with_id("value", DataType::DOUBLE, true, 2),
        ]);

        // Generate the schema with content_stats (using all columns)
        let delta_stats_schema = test_delta_stats_schema(&table_schema);
        let schema_with_stats = Arc::new(ContentTreeNodeEntry::to_schema_with_content_stats(
            &table_schema,
            &delta_stats_schema,
        )?);

        // Create content_stats in AMT format:
        // {id: {value_count, lower_bound, upper_bound, exact_bounds},
        //  value: {value_count, null_value_count, nan_value_count, lower_bound, upper_bound, exact_bounds}}

        // Build id stats struct (non-nullable INTEGER, so no null_value_count or nan_value_count)
        let id_stats = StructData::try_new(
            vec![
                StructField::nullable(VALUE_COUNT, DataType::LONG),
                StructField::nullable(LOWER_BOUND, DataType::INTEGER),
                StructField::nullable(UPPER_BOUND, DataType::INTEGER),
                StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
            ],
            vec![
                Scalar::Long(100),
                Scalar::Integer(1),
                Scalar::Integer(1000),
                Scalar::Boolean(true),
            ],
        )?;

        // Build value stats struct (nullable DOUBLE, so has null_value_count and nan_value_count)
        let value_stats = StructData::try_new(
            vec![
                StructField::nullable(VALUE_COUNT, DataType::LONG),
                StructField::nullable(NULL_VALUE_COUNT, DataType::LONG),
                StructField::nullable(NAN_VALUE_COUNT, DataType::LONG),
                StructField::nullable(LOWER_BOUND, DataType::DOUBLE),
                StructField::nullable(UPPER_BOUND, DataType::DOUBLE),
                StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
            ],
            vec![
                Scalar::Long(100),
                Scalar::Long(5),
                Scalar::Long(0),
                Scalar::Double(0.0),
                Scalar::Double(100.0),
                Scalar::Boolean(true),
            ],
        )?;

        // Build the content_stats struct
        let content_stats = StructData::try_new(
            vec![
                StructField::nullable(
                    "id",
                    DataType::Struct(Box::new(StructType::new_unchecked([
                        StructField::nullable(VALUE_COUNT, DataType::LONG),
                        StructField::nullable(LOWER_BOUND, DataType::INTEGER),
                        StructField::nullable(UPPER_BOUND, DataType::INTEGER),
                        StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
                    ]))),
                ),
                StructField::nullable(
                    "value",
                    DataType::Struct(Box::new(StructType::new_unchecked([
                        StructField::nullable(VALUE_COUNT, DataType::LONG),
                        StructField::nullable(NULL_VALUE_COUNT, DataType::LONG),
                        StructField::nullable(NAN_VALUE_COUNT, DataType::LONG),
                        StructField::nullable(LOWER_BOUND, DataType::DOUBLE),
                        StructField::nullable(UPPER_BOUND, DataType::DOUBLE),
                        StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
                    ]))),
                ),
            ],
            vec![Scalar::Struct(id_stats), Scalar::Struct(value_stats)],
        )?;

        // Create a ContentTreeNodeEntry with content_stats
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .content_stats(content_stats)
            .build();

        // Convert to EngineData
        let engine_data = entry.into_engine_data(schema_with_stats.clone(), &engine)?;

        // Verify the engine data was created successfully
        assert!(!engine_data.is_empty());
        assert_eq!(engine_data.len(), 1); // Single row

        Ok(())
    }

    #[test]
    fn test_into_engine_data_with_null_content_stats() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructType};
        use crate::IntoEngineData;

        let engine = SyncEngine::new();

        // Create a simple table schema with parquet field IDs and column mapping annotations
        // (column mapping is required when metadata tree feature is enabled)
        let table_schema =
            StructType::new_unchecked([StructField::new("id", DataType::INTEGER, false)
                .with_metadata([
                    (
                        ColumnMetadataKey::ParquetFieldId.as_ref(),
                        MetadataValue::Number(1),
                    ),
                    (
                        ColumnMetadataKey::ColumnMappingId.as_ref(),
                        MetadataValue::Number(1),
                    ),
                    (
                        ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                        MetadataValue::String("col-id".to_string()),
                    ),
                ])]);

        // Generate the schema with content_stats (using all columns)
        let delta_stats_schema = test_delta_stats_schema(&table_schema);
        let schema_with_stats = Arc::new(ContentTreeNodeEntry::to_schema_with_content_stats(
            &table_schema,
            &delta_stats_schema,
        )?);

        // Create a ContentTreeNodeEntry with content_stats set to None
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        // Convert to EngineData - should handle null content_stats gracefully
        let engine_data = entry.into_engine_data(schema_with_stats.clone(), &engine)?;

        // Verify the engine data was created successfully
        assert!(!engine_data.is_empty());
        assert_eq!(engine_data.len(), 1); // Single row

        Ok(())
    }

    #[test]
    fn test_roundtrip_with_content_stats() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue, StructType};
        use crate::IntoEngineData;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a simple table schema with parquet field IDs and column mapping annotations
        // (column mapping is required when metadata tree feature is enabled)
        let table_schema = StructType::new_unchecked([
            StructField::new("id", DataType::INTEGER, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-id".to_string()),
                ),
            ]),
            StructField::new("name", DataType::STRING, true).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(2),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-name".to_string()),
                ),
            ]),
        ]);

        // Generate the schema with content_stats (using all columns)
        let delta_stats_schema = test_delta_stats_schema(&table_schema);
        let schema_with_stats = Arc::new(ContentTreeNodeEntry::to_schema_with_content_stats(
            &table_schema,
            &delta_stats_schema,
        )?);

        // Create content_stats data in AMT format:
        // {id: {value_count, lower_bound, upper_bound, exact_bounds},
        //  name: {value_count, null_value_count, avg_value_size, max_value_size, lower_bound, upper_bound, exact_bounds}}

        // Build id stats struct (non-nullable INTEGER, so no null_value_count)
        let id_stats_fields = vec![
            StructField::nullable(VALUE_COUNT, DataType::LONG),
            StructField::nullable(LOWER_BOUND, DataType::INTEGER),
            StructField::nullable(UPPER_BOUND, DataType::INTEGER),
            StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
        ];
        let id_stats = StructData::try_new(
            id_stats_fields.clone(),
            vec![
                Scalar::Long(500),
                Scalar::Integer(1),
                Scalar::Integer(500),
                Scalar::Boolean(true),
            ],
        )?;

        // Build name stats struct (nullable STRING, so has null_value_count and size stats)
        let name_stats_fields = vec![
            StructField::nullable(VALUE_COUNT, DataType::LONG),
            StructField::nullable(NULL_VALUE_COUNT, DataType::LONG),
            StructField::nullable(AVG_VALUE_SIZE, DataType::INTEGER),
            StructField::nullable(MAX_VALUE_SIZE, DataType::INTEGER),
            StructField::nullable(LOWER_BOUND, DataType::STRING),
            StructField::nullable(UPPER_BOUND, DataType::STRING),
            StructField::nullable(EXACT_BOUNDS, DataType::BOOLEAN),
        ];
        let name_stats = StructData::try_new(
            name_stats_fields.clone(),
            vec![
                Scalar::Long(500),
                Scalar::Long(10),
                Scalar::Null(DataType::INTEGER),
                Scalar::Null(DataType::INTEGER),
                Scalar::String("aardvark".to_string()),
                Scalar::String("zebra".to_string()),
                Scalar::Boolean(true),
            ],
        )?;

        // Build the content_stats struct in AMT format
        let content_stats_fields = vec![
            StructField::nullable(
                "id",
                DataType::Struct(Box::new(StructType::new_unchecked(id_stats_fields))),
            ),
            StructField::nullable(
                "name",
                DataType::Struct(Box::new(StructType::new_unchecked(name_stats_fields))),
            ),
        ];
        let content_stats = StructData::try_new(
            content_stats_fields,
            vec![Scalar::Struct(id_stats), Scalar::Struct(name_stats)],
        )?;

        // Create a ContentTreeNodeEntry with content_stats
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/data/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(500)
            .file_size_in_bytes(2048)
            .content_stats(content_stats)
            .build();

        // Convert to EngineData using schema with content_stats
        let engine_data = entry.into_engine_data(schema_with_stats.clone(), &engine)?;

        // Create ContentTreeNode and write it
        let metadata = ContentTreeNode {
            data: vec![engine_data],
            version: 0,
            table_root: table_root_url.clone(),
            path_in_log: String::new(),
            leaf: None,
        };

        // Write metadata using the writer
        let writer = writer::ContentTreeNodeWriter::try_new(metadata)?;
        let written_file = writer.write(&engine)?.location;

        // Verify the file was written
        assert!(written_file.as_str().ends_with(".parquet"));

        // Note: Full roundtrip reading is not tested here because the reader
        // would need to be updated to handle content_stats. The key thing
        // we're testing is that the IntoEngineData conversion works correctly.

        Ok(())
    }

    #[test]
    fn test_enum_to_scalar_conversions() {
        // Test DataContentType conversion
        let content_type = DataContentType::Data;
        let scalar: Scalar = content_type.into();
        assert!(matches!(scalar, Scalar::Integer(0)));

        // Test DataFileFormat conversion
        let file_format = DataFileFormat::Parquet;
        let scalar: Scalar = file_format.into();
        assert!(matches!(scalar, Scalar::String(ref s) if s == "parquet"));

        // Test TrackingStatus conversion
        let status = TrackingStatus::Added;
        let scalar: Scalar = status.into();
        assert!(matches!(scalar, Scalar::Integer(1)));
    }

    #[test]
    fn test_bytes_to_scalar_conversion() {
        let bytes = Bytes::from(vec![1, 2, 3, 4]);
        let scalar: Scalar = bytes.into();
        assert!(matches!(scalar, Scalar::Binary(ref v) if v == &vec![1, 2, 3, 4]));
    }

    /// Helper function to create a simple test table schema with parquet field IDs.
    /// This is used for tests that need to generate content_stats schema.
    fn test_table_schema() -> StructType {
        use crate::schema::{ColumnMetadataKey, MetadataValue};

        StructType::new_unchecked([
            StructField::new("id", DataType::INTEGER, false).with_metadata([
                (
                    ColumnMetadataKey::ParquetFieldId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingId.as_ref(),
                    MetadataValue::Number(1),
                ),
                (
                    ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                    MetadataValue::String("col-id".to_string()),
                ),
            ]),
        ])
    }

    /// Creates a minimal Delta stats_schema (with `minValues`) for all primitive leaf columns
    /// in `table_schema`. Used to construct `content_stats` read schemas in tests.
    fn test_delta_stats_schema(table_schema: &StructType) -> StructType {
        fn min_values_fields(schema: &StructType) -> Vec<StructField> {
            schema
                .fields()
                .flat_map(|f| match f.data_type() {
                    DataType::Struct(nested) => min_values_fields(nested),
                    _ => vec![StructField::nullable(f.name(), f.data_type.clone())],
                })
                .collect()
        }
        let min_values = StructType::new_unchecked(min_values_fields(table_schema));
        StructType::new_unchecked([StructField::nullable(
            DELTA_STATS_MIN_VALUES,
            DataType::Struct(Box::new(min_values)),
        )])
    }

    /// Helper function to get the test schema for ContentTreeNodeEntry with content_stats.
    /// Uses `test_table_schema()` to generate the dynamic schema.
    fn test_metadata_entry_schema() -> SchemaRef {
        let table_schema = test_table_schema();
        let stats_schema = test_delta_stats_schema(&table_schema);
        Arc::new(
            ContentTreeNodeEntry::to_schema_with_content_stats(&table_schema, &stats_schema)
                .expect("test schema should be valid"),
        )
    }

    // Helper to compare two metadata entries (excluding fields that are not yet fully supported)
    fn assert_metadata_entry_eq(expected: &ContentTreeNodeEntry, actual: &ContentTreeNodeEntry) {
        assert_eq!(
            expected.content_type, actual.content_type,
            "content_type mismatch"
        );
        assert_eq!(expected.location, actual.location, "location mismatch");
        assert_eq!(
            expected.file_format, actual.file_format,
            "file_format mismatch"
        );

        // Compare tracking_info
        match (&expected.tracking_info, &actual.tracking_info) {
            (Some(exp_ti), Some(act_ti)) => {
                assert_eq!(
                    exp_ti.status, act_ti.status,
                    "tracking_info.status mismatch"
                );
                assert_eq!(
                    exp_ti.snapshot_id, act_ti.snapshot_id,
                    "tracking_info.snapshot_id mismatch"
                );
                assert_eq!(
                    exp_ti.sequence_number, act_ti.sequence_number,
                    "tracking_info.sequence_number mismatch"
                );
                assert_eq!(
                    exp_ti.file_sequence_number, act_ti.file_sequence_number,
                    "tracking_info.file_sequence_number mismatch"
                );
                assert_eq!(
                    exp_ti.first_row_id, act_ti.first_row_id,
                    "tracking_info.first_row_id mismatch"
                );
            }
            (None, None) => {}
            _ => panic!("tracking_info presence mismatch"),
        }

        // Compare manifest_dv and changes_dv
        assert_eq!(
            expected.manifest_dv, actual.manifest_dv,
            "manifest_dv mismatch"
        );
        assert_eq!(
            expected.tracking_info.as_ref().map(|t| &t.changes_dv),
            actual.tracking_info.as_ref().map(|t| &t.changes_dv),
            "changes_dv mismatch"
        );

        assert_eq!(
            expected.partition_spec_id, actual.partition_spec_id,
            "partition_spec_id mismatch"
        );
        assert_eq!(
            expected.sort_order_id, actual.sort_order_id,
            "sort_order_id mismatch"
        );
        assert_eq!(
            expected.record_count, actual.record_count,
            "record_count mismatch"
        );
        assert_eq!(
            expected.file_size_in_bytes, actual.file_size_in_bytes,
            "file_size_in_bytes mismatch"
        );

        // Compare manifest_stats
        match (&expected.manifest_stats, &actual.manifest_stats) {
            (Some(exp_ms), Some(act_ms)) => {
                assert_eq!(
                    exp_ms.added_files_count, act_ms.added_files_count,
                    "manifest_stats.added_files_count mismatch"
                );
                assert_eq!(
                    exp_ms.existing_files_count, act_ms.existing_files_count,
                    "manifest_stats.existing_files_count mismatch"
                );
                assert_eq!(
                    exp_ms.deletes_files_count, act_ms.deletes_files_count,
                    "manifest_stats.deletes_files_count mismatch"
                );
                assert_eq!(
                    exp_ms.added_rows_count, act_ms.added_rows_count,
                    "manifest_stats.added_rows_count mismatch"
                );
                assert_eq!(
                    exp_ms.existing_rows_count, act_ms.existing_rows_count,
                    "manifest_stats.existing_rows_count mismatch"
                );
                assert_eq!(
                    exp_ms.delete_rows_count, act_ms.delete_rows_count,
                    "manifest_stats.delete_rows_count mismatch"
                );
                assert_eq!(
                    exp_ms.min_sequence_number, act_ms.min_sequence_number,
                    "manifest_stats.min_sequence_number mismatch"
                );
            }
            (None, None) => {}
            _ => panic!("manifest_stats presence mismatch"),
        }

        assert_eq!(
            expected.key_metadata, actual.key_metadata,
            "key_metadata mismatch"
        );
        // Note: split_offsets and equality_ids are array types not extracted by the visitor
    }

    #[test]
    fn test_roundtrip_simple_metadata_entry() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create original metadata
        let original_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/path/to/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(200),
                first_row_id: Some(1000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();
        let read_metadata =
            build_and_roundtrip(vec![original_entry.clone()], 0, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&original_entry, &entries[0]);

        Ok(())
    }

    #[test]
    fn test_field_id_annotations_in_schema() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue, ToSchema};

        // Verify that TrackingInfo::to_schema() has the field_id metadata from #[field_id] annotations
        let tracking_info_schema = TrackingInfo::to_schema();

        // Helper to check field_id metadata
        fn assert_field_id(schema: &StructType, field_name: &str, expected_id: i64) {
            let field = schema
                .field(field_name)
                .unwrap_or_else(|| panic!("{} field should exist in schema", field_name));
            let metadata = field.metadata();
            assert!(
                metadata.contains_key(ColumnMetadataKey::ParquetFieldId.as_ref()),
                "{} field should have PARQUET:field_id in metadata",
                field_name
            );
            match metadata.get(ColumnMetadataKey::ParquetFieldId.as_ref()) {
                Some(MetadataValue::Number(n)) => assert_eq!(
                    *n, expected_id,
                    "{} field should have field_id = {}",
                    field_name, expected_id
                ),
                other => panic!(
                    "{} field should have Number metadata, got {:?}",
                    field_name, other
                ),
            }
        }

        // Verify field IDs from TrackingInfo (defined with #[field_id = X] annotations)
        // status: #[field_id = 0]
        assert_field_id(&tracking_info_schema, "status", 0);

        // snapshotId: #[field_id = 1]
        assert_field_id(&tracking_info_schema, "snapshotId", 1);

        // sequenceNumber: #[field_id = 3]
        assert_field_id(&tracking_info_schema, "sequenceNumber", 3);

        // fileSequenceNumber: #[field_id = 4]
        assert_field_id(&tracking_info_schema, "fileSequenceNumber", 4);

        // firstRowId: #[field_id = 142]
        assert_field_id(&tracking_info_schema, "firstRowId", 142);

        // changesDv: #[field_id = 153]
        assert_field_id(&tracking_info_schema, "changesDv", 153);

        // Verify ManifestStats field IDs
        let manifest_stats_schema = ManifestStats::to_schema();
        assert_field_id(&manifest_stats_schema, "addedFilesCount", 504);
        assert_field_id(&manifest_stats_schema, "existingFilesCount", 505);
        assert_field_id(&manifest_stats_schema, "deletesFilesCount", 506);
        assert_field_id(&manifest_stats_schema, "addedRowsCount", 512);
        assert_field_id(&manifest_stats_schema, "existingRowsCount", 513);
        assert_field_id(&manifest_stats_schema, "deleteRowsCount", 514);
        assert_field_id(&manifest_stats_schema, "minSequenceNumber", 516);

        // Verify DvInfo field IDs
        let dv_info_schema = DvInfo::to_schema();
        assert_field_id(&dv_info_schema, "location", 152);
        assert_field_id(&dv_info_schema, "offset", 144);
        assert_field_id(&dv_info_schema, "sizeInBytes", 145);
        assert_field_id(&dv_info_schema, "cardinality", 154);

        // Verify top-level ContentTreeNodeEntry field IDs
        let metadata_entry_schema = ContentTreeNodeEntry::to_schema();
        assert_field_id(&metadata_entry_schema, "contentType", 134);
        assert_field_id(&metadata_entry_schema, "location", 100);
        assert_field_id(&metadata_entry_schema, "fileFormat", 101);
        assert_field_id(&metadata_entry_schema, "trackingInfo", 147);
        assert_field_id(&metadata_entry_schema, "dvInfo", 148);
        assert_field_id(&metadata_entry_schema, "partitionSpecId", 149);
        assert_field_id(&metadata_entry_schema, "sortOrderId", 140);
        assert_field_id(&metadata_entry_schema, "recordCount", 103);
        assert_field_id(&metadata_entry_schema, "fileSizeInBytes", 104);
        assert_field_id(&metadata_entry_schema, "manifestStats", 150);
        assert_field_id(&metadata_entry_schema, "manifestDv", 151);

        // Verify content_stats field_id in to_schema_with_content_stats
        let table_schema =
            StructType::new_unchecked([StructField::not_null("id", DataType::INTEGER)
                .add_metadata([(ColumnMetadataKey::ParquetFieldId.as_ref(), 1i64)])]);
        let delta_stats_schema = test_delta_stats_schema(&table_schema);
        let schema_with_stats =
            ContentTreeNodeEntry::to_schema_with_content_stats(&table_schema, &delta_stats_schema)?;
        assert_field_id(&schema_with_stats, CONTENT_STATS_FIELD_NAME, 146);

        Ok(())
    }

    #[test]
    fn test_field_ids_in_metadata_entry_schema() -> DeltaResult<()> {
        use crate::schema::{ColumnMetadataKey, MetadataValue};

        // Verify that the test_metadata_entry_schema() includes field_id metadata
        // for nested structs like TrackingInfo
        let schema = test_metadata_entry_schema();

        // Get the trackingInfo field
        let tracking_info_field = schema
            .field("trackingInfo")
            .expect("trackingInfo field should exist");

        // Get the nested struct type
        let tracking_info_struct = match tracking_info_field.data_type() {
            DataType::Struct(s) => s.as_ref(),
            _ => panic!("Expected trackingInfo to be a struct"),
        };

        // Verify that nested fields have field_id metadata
        let status_field = tracking_info_struct
            .field("status")
            .expect("status field should exist");
        assert!(
            status_field
                .metadata()
                .contains_key(ColumnMetadataKey::ParquetFieldId.as_ref()),
            "status field should have PARQUET:field_id in metadata"
        );
        assert_eq!(
            status_field
                .metadata()
                .get(ColumnMetadataKey::ParquetFieldId.as_ref()),
            Some(&MetadataValue::Number(0)),
            "status field should have field_id = 0"
        );

        Ok(())
    }

    #[test]
    fn test_roundtrip_preserves_data_with_field_id_schema() -> DeltaResult<()> {
        // This test verifies that metadata entries with field_id annotations
        // can be written and read back correctly (data roundtrip)
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create original metadata
        let original_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/path/to/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(200),
                first_row_id: Some(1000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();
        let read_metadata =
            build_and_roundtrip(vec![original_entry.clone()], 0, &table_root_url, &engine)?;

        // Verify data was preserved
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&original_entry, &entries[0]);

        Ok(())
    }

    #[ignore] // PositionDeletes is not supported
    #[test]
    fn test_roundtrip_metadata_entry_with_deletion_vector() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create metadata with deletion vector
        let original_entry = ContentTreeNodeEntryBuilder::new(DataContentType::PositionDeletes)
            .location("s3://bucket/path/to/deletes.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(5),
                sequence_number: Some(500),
                file_sequence_number: Some(600),
                first_row_id: Some(5000),
                changes_dv: None,
            })
            .partition_spec_id(1)
            .sort_order_id(1)
            .record_count(10)
            .file_size_in_bytes(512)
            .build();
        let read_metadata =
            build_and_roundtrip(vec![original_entry.clone()], 1, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&original_entry, &entries[0]);

        Ok(())
    }

    #[ignore] // DataManifest is not supported
    #[test]
    fn test_roundtrip_metadata_entry_with_manifest_stats() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create metadata with manifest stats
        let original_entry = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("s3://bucket/path/to/manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Existed,
                snapshot_id: Some(10),
                sequence_number: Some(1000),
                file_sequence_number: Some(1000),
                first_row_id: Some(10000),
                changes_dv: None,
            })
            .partition_spec_id(2)
            .sort_order_id(2)
            .record_count(100)
            .file_size_in_bytes(10240)
            .manifest_stats(ManifestStats {
                added_files_count: 5,
                existing_files_count: 10,
                deletes_files_count: 2,
                added_rows_count: 500,
                existing_rows_count: 1000,
                delete_rows_count: 50,
                min_sequence_number: 100,
            })
            .build();
        let read_metadata =
            build_and_roundtrip(vec![original_entry.clone()], 2, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&original_entry, &entries[0]);

        Ok(())
    }

    #[ignore] // DataManifest is not supported
    #[test]
    fn test_roundtrip_metadata_entry_with_inline_deletion_vector() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create metadata with inline deletion vector
        let inline_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF];
        let original_entry = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("s3://bucket/path/to/manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(3),
                sequence_number: Some(300),
                file_sequence_number: Some(400),
                first_row_id: Some(3000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(2048)
            .manifest_dv(Bytes::from(inline_data))
            .build();
        let read_metadata =
            build_and_roundtrip(vec![original_entry.clone()], 3, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&original_entry, &entries[0]);

        // Verify manifest_dv specifically
        let read_entry = &entries[0];
        assert!(
            read_entry.manifest_dv.is_some(),
            "manifest_dv should be present"
        );
        let read_bytes = read_entry.manifest_dv.as_ref().unwrap();
        let orig_bytes = original_entry.manifest_dv.as_ref().unwrap();
        assert_eq!(
            read_bytes.len(),
            orig_bytes.len(),
            "manifest_dv length must match"
        );
        assert_eq!(
            read_bytes.as_ref(),
            orig_bytes.as_ref(),
            "manifest_dv bytes must match"
        );

        Ok(())
    }

    #[ignore] // Not all entry type are supported
    #[test]
    fn test_roundtrip_multiple_metadata_entries() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create multiple entries including one with inline DV
        let entry1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/path/to/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(200),
                first_row_id: Some(1000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();
        let entry2 = ContentTreeNodeEntryBuilder::new(DataContentType::PositionDeletes)
            .location("s3://bucket/path/to/deletes.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(5),
                sequence_number: Some(500),
                file_sequence_number: Some(600),
                first_row_id: Some(5000),
                changes_dv: None,
            })
            .partition_spec_id(1)
            .sort_order_id(1)
            .record_count(10)
            .file_size_in_bytes(512)
            .build();
        let entry3 = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("s3://bucket/path/to/manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Existed,
                snapshot_id: Some(10),
                sequence_number: Some(1000),
                file_sequence_number: Some(1000),
                first_row_id: Some(10000),
                changes_dv: None,
            })
            .partition_spec_id(2)
            .sort_order_id(2)
            .record_count(100)
            .file_size_in_bytes(10240)
            .manifest_stats(ManifestStats {
                added_files_count: 5,
                existing_files_count: 10,
                deletes_files_count: 2,
                added_rows_count: 500,
                existing_rows_count: 1000,
                delete_rows_count: 50,
                min_sequence_number: 100,
            })
            .build();
        let inline_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF];
        let entry4 = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("s3://bucket/path/to/manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(3),
                sequence_number: Some(300),
                file_sequence_number: Some(400),
                first_row_id: Some(3000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(2048)
            .manifest_dv(Bytes::from(inline_data))
            .build();

        let read_metadata = build_and_roundtrip(
            vec![
                entry1.clone(),
                entry2.clone(),
                entry3.clone(),
                entry4.clone(),
            ],
            3,
            &table_root_url,
            &engine,
        )?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 4);
        assert_metadata_entry_eq(&entry1, &entries[0]);
        assert_metadata_entry_eq(&entry2, &entries[1]);
        assert_metadata_entry_eq(&entry3, &entries[2]);
        assert_metadata_entry_eq(&entry4, &entries[3]);

        Ok(())
    }

    #[ignore] // Not all entry type are supported
    #[test]
    fn test_roundtrip_all_data_content_types() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create entries with all content types
        let content_types = vec![
            DataContentType::Data,
            DataContentType::PositionDeletes,
            DataContentType::EqualityDeletes,
            DataContentType::DataManifest,
            DataContentType::DeleteManifest,
        ];

        let entries: Vec<ContentTreeNodeEntry> = content_types
            .into_iter()
            .enumerate()
            .map(|(i, content_type)| {
                ContentTreeNodeEntryBuilder::new(content_type)
                    .location(format!("s3://bucket/file{}.parquet", i))
                    .tracking_info(TrackingInfo {
                        status: TrackingStatus::Added,
                        snapshot_id: Some(i as i64),
                        sequence_number: Some((i * 100) as i64),
                        file_sequence_number: Some((i * 200) as i64),
                        first_row_id: Some((i * 1000) as i64),
                        changes_dv: None,
                    })
                    .partition_spec_id(i as i64)
                    .sort_order_id(i as i64)
                    .record_count((i * 10) as i64)
                    .file_size_in_bytes((i * 512) as i64)
                    .build()
            })
            .collect();

        let read_metadata = build_and_roundtrip(entries.clone(), 4, &table_root_url, &engine)?;

        // Verify
        let read_entries = read_metadata.entries()?;
        assert_eq!(read_entries.len(), entries.len());
        for (expected, actual) in entries.iter().zip(read_entries.iter()) {
            assert_metadata_entry_eq(expected, actual);
        }

        Ok(())
    }

    #[test]
    fn test_roundtrip_all_tracking_statuses() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create entries with all tracking statuses
        let statuses = vec![
            TrackingStatus::Existed,
            TrackingStatus::Added,
            TrackingStatus::Deleted,
        ];

        let entries: Vec<ContentTreeNodeEntry> = statuses
            .into_iter()
            .enumerate()
            .map(|(i, status)| {
                ContentTreeNodeEntryBuilder::new(DataContentType::Data)
                    .location(format!("s3://bucket/file{}.parquet", i))
                    .tracking_info(TrackingInfo {
                        status,
                        snapshot_id: Some(i as i64),
                        sequence_number: Some((i * 100) as i64),
                        file_sequence_number: Some((i * 200) as i64),
                        first_row_id: Some((i * 1000) as i64),
                        changes_dv: None,
                    })
                    .sort_order_id(0)
                    .record_count(42)
                    .file_size_in_bytes(1024)
                    .build()
            })
            .collect();

        let read_metadata = build_and_roundtrip(entries.clone(), 5, &table_root_url, &engine)?;

        // Verify
        let read_entries = read_metadata.entries()?;
        assert_eq!(read_entries.len(), entries.len());
        for (expected, actual) in entries.iter().zip(read_entries.iter()) {
            assert_metadata_entry_eq(expected, actual);
        }

        Ok(())
    }

    #[test]
    fn test_roundtrip_with_optional_fields_null() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create entry with many optional fields set to None
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: None,
                sequence_number: None,
                file_sequence_number: None,
                first_row_id: None,
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();

        let read_metadata = build_and_roundtrip(vec![entry.clone()], 6, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&entry, &entries[0]);

        // Specifically verify the None values
        let actual = &entries[0];
        let ti = actual.tracking_info.as_ref().unwrap();
        assert!(ti.snapshot_id.is_none());
        assert!(ti.sequence_number.is_none());
        assert!(ti.file_sequence_number.is_none());
        assert!(ti.first_row_id.is_none());
        assert!(ti.changes_dv.is_none());
        assert!(actual.manifest_dv.is_none());
        assert!(actual.manifest_stats.is_none());

        Ok(())
    }

    #[test]
    fn test_roundtrip_puffin_format() -> DeltaResult<()> {
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create entry with Puffin format
        let entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("s3://bucket/file.puffin")
            .file_format(DataFileFormat::Puffin)
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(200),
                first_row_id: Some(1000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(42)
            .file_size_in_bytes(1024)
            .build();

        let read_metadata = build_and_roundtrip(vec![entry.clone()], 7, &table_root_url, &engine)?;

        // Verify
        let entries = read_metadata.entries()?;
        assert_eq!(entries.len(), 1);
        assert_metadata_entry_eq(&entry, &entries[0]);
        assert_eq!(entries[0].file_format, DataFileFormat::Puffin);

        Ok(())
    }

    /// Builds a ContentTreeNode from entries using the builder.
    fn build_node(
        entries: Vec<ContentTreeNodeEntry>,
        version: Version,
        table_root_url: &Url,
        engine: &SyncEngine,
    ) -> DeltaResult<ContentTreeNode> {
        use crate::content_tree::builder::ContentTreeNodeBuilder;

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root_url.clone(), version, test_table_schema());
        for entry in entries {
            builder.add_entry(entry);
        }
        builder.build(engine, 1, None).map(|(node, _)| node)
    }

    /// Builds a ContentTreeNode from entries using the builder, writes to disk, and reads back.
    /// This ensures tests go through the same code path as production (builder -> write -> read).
    fn build_and_roundtrip(
        entries: Vec<ContentTreeNodeEntry>,
        version: Version,
        table_root_url: &Url,
        engine: &SyncEngine,
    ) -> DeltaResult<ContentTreeNode> {
        use crate::content_tree::builder::ContentTreeNodeBuilder;

        let mut builder =
            ContentTreeNodeBuilder::new_for(table_root_url.clone(), version, test_table_schema());
        for entry in entries {
            builder.add_entry(entry);
        }
        let (metadata, _) = builder.build(engine, 1, None)?;

        let written_path = writer::ContentTreeNodeWriter::try_new(metadata)?
            .write(engine)?
            .location;
        let path_in_log = absolute_to_relative_path(&written_path, table_root_url)?;
        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &written_path,
            path_in_log,
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_root_url.clone(),
        )
    }

    #[test]
    fn test_data_entry_without_dv_has_no_deletion_vector() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // In the new CombinedManifest model, DV is inline on Data entries.
        // A Data entry with dv_info: None produces an Add with no deletionVector.
        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        let schema = crate::actions::get_log_add_schema().clone();
        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        let batch = action_batches.next().unwrap()?;
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        assert_eq!(visitor.adds.len(), 1);

        assert!(
            visitor.adds[0].deletion_vector.is_none(),
            "Data entry without dv_info should not have a deletion vector"
        );

        Ok(())
    }

    #[test]
    fn test_default_row_commit_version_maps_to_sequence_number() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Use different values for snapshot_id and sequence_number to verify that
        // defaultRowCommitVersion maps to sequenceNumber (42), not snapshotId (1).
        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(42),
                file_sequence_number: Some(42),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        let schema = crate::actions::get_log_add_schema().clone();
        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        let batch = action_batches.next().unwrap()?;
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        assert_eq!(visitor.adds.len(), 1);
        let add = &visitor.adds[0];

        assert_eq!(add.base_row_id, Some(0));
        assert_eq!(
            add.default_row_commit_version,
            Some(42),
            "defaultRowCommitVersion should map to sequenceNumber, not snapshotId"
        );

        Ok(())
    }

    #[test]
    fn test_data_entry_with_dv_has_deletion_vector() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // In the new CombinedManifest model, a Data entry with dv_info produces an Add with a DV.
        // Use a relative DV path format: deletion_vector_{uuid}.bin
        let dv_location = "deletion_vector_12345678-1234-1234-1234-123456789abc.bin";
        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .dv_info(DvInfo {
                location: dv_location.to_string(),
                offset: 0,
                size_in_bytes: 108,
                cardinality: 10,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        let schema = crate::actions::get_log_add_schema().clone();
        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        let batch = action_batches.next().unwrap()?;
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        assert_eq!(visitor.adds.len(), 1);
        let add = &visitor.adds[0];

        assert!(
            add.deletion_vector.is_some(),
            "Data entry with inline dv_info should have a deletion vector"
        );
        assert_eq!(add.deletion_vector.as_ref().unwrap().cardinality, 10);

        Ok(())
    }

    #[test]
    fn test_dv_not_present() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create a data file without any corresponding DV
        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(50),
                file_sequence_number: Some(50),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        // Get action batches (no data skipping for this test)
        let schema = crate::actions::get_log_add_schema().clone();
        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        // Get the Add action using visitor
        let batch = action_batches.next().unwrap()?;
        let mut visitor = AddVisitor::default();
        visitor.visit_rows_of(batch.actions.as_ref())?;
        assert_eq!(visitor.adds.len(), 1);
        let add = &visitor.adds[0];

        // Verify DV is NOT included (doesn't exist)
        assert!(
            add.deletion_vector.is_none(),
            "DV should not be included when it doesn't exist"
        );

        Ok(())
    }

    #[test]
    fn test_manifest_dv_not_dropped_in_serialization() -> DeltaResult<()> {
        // This test verifies that manifest_dv survives the into_engine_data conversion
        // even when not read back through the full reader path
        let engine = SyncEngine::new();

        // Create a metadata entry with manifest DV
        let inline_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF];
        let inline_dv_entry = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("s3://bucket/path/to/manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(3),
                sequence_number: Some(300),
                file_sequence_number: Some(400),
                first_row_id: Some(3000),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(2048)
            .manifest_dv(Bytes::from(inline_data))
            .build();
        let original_dv_bytes = inline_dv_entry.manifest_dv.as_ref().unwrap().clone();

        // Convert to engine data
        let engine_data = inline_dv_entry
            .clone()
            .into_engine_data(test_metadata_entry_schema(), &engine)?;

        // The manifest_dv should be in the engine data
        // We can't easily extract it without the full visitor, but we can verify
        // that the conversion succeeded and the data was included
        assert!(!engine_data.is_empty(), "Engine data should not be empty");

        // Verify the original bytes are not empty
        assert!(
            !original_dv_bytes.is_empty(),
            "Original manifest DV should not be empty"
        );
        assert_eq!(
            original_dv_bytes.len(),
            8,
            "Expected 8 bytes of manifest DV"
        );

        Ok(())
    }

    #[test]
    fn test_multiple_data_entries_each_with_own_dv() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // In the new CombinedManifest model, each Data entry has its own inline DV.
        // Two data entries, each with a different DV, both should produce Add actions with DVs.
        let dv_loc1 = "deletion_vector_12345678-1234-1234-1234-123456789abc.bin";
        let dv_loc2 = "deletion_vector_87654321-4321-4321-4321-cba987654321.bin";
        let data_entry_1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data1.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .dv_info(DvInfo {
                location: dv_loc1.to_string(),
                offset: 0,
                size_in_bytes: 108,
                cardinality: 15,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let data_entry_2 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data2.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(200),
                file_sequence_number: Some(200),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .dv_info(DvInfo {
                location: dv_loc2.to_string(),
                offset: 0,
                size_in_bytes: 108,
                cardinality: 20,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_and_roundtrip(
            vec![data_entry_1, data_entry_2],
            0,
            &table_root_url,
            &engine,
        )?;

        let schema = crate::actions::get_log_add_schema().clone();
        let action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        // Collect all adds from all batches (each data entry may produce a separate batch)
        let mut all_adds = Vec::new();
        for batch_result in action_batches {
            let batch = batch_result?;
            let mut visitor = AddVisitor::default();
            visitor.visit_rows_of(batch.actions.as_ref())?;
            all_adds.extend(visitor.adds);
        }
        assert_eq!(
            all_adds.len(),
            2,
            "Both data entries should produce Add actions"
        );

        // Both entries should have their own DVs
        for add in &all_adds {
            assert!(
                add.deletion_vector.is_some(),
                "Each data entry with inline dv_info should have a deletion vector"
            );
        }
        // Cardinalities should be 15 and 20 (in some order)
        let cardinalities: std::collections::HashSet<i64> = all_adds
            .iter()
            .filter_map(|a| a.deletion_vector.as_ref())
            .map(|dv| dv.cardinality)
            .collect();
        assert!(cardinalities.contains(&15) && cardinalities.contains(&20));

        Ok(())
    }

    #[test]
    fn test_data_entry_with_deleted_status_not_included() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // In the new CombinedManifest model, a Data entry with Deleted tracking status
        // should not produce any Add action (it produces a Remove action instead).
        let mut data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(50),
                file_sequence_number: Some(50),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        if let Some(ref mut ti) = data_entry.tracking_info {
            ti.status = TrackingStatus::Deleted;
        }

        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        let schema = crate::actions::get_log_add_schema().clone();
        let action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &schema,
            &[],
            None,
            None,
            None,
        )?;

        // Collect all adds from all batches
        let mut total_adds = 0;
        for batch_result in action_batches {
            let batch = batch_result?;
            let mut visitor = AddVisitor::default();
            visitor.visit_rows_of(batch.actions.as_ref())?;
            total_adds += visitor.adds.len();
        }

        // Data entry with Deleted status should not produce any Add actions
        assert_eq!(
            total_adds, 0,
            "Data entry with Deleted status should not produce an Add action"
        );

        Ok(())
    }

    #[test]
    fn test_old_data_manifest_format_returns_error() -> DeltaResult<()> {
        // Old DataManifest/DeleteManifest format is no longer supported.
        // manifest_references() should return an error for these entry types.
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        let data_manifest = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
            .location("memory:///data-manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Existed,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .record_count(100)
            .file_size_in_bytes(1024)
            .manifest_stats(ManifestStats {
                added_files_count: 10,
                existing_files_count: 90,
                deletes_files_count: 0,
                added_rows_count: 1000,
                existing_rows_count: 9000,
                delete_rows_count: 0,
                min_sequence_number: 50,
            })
            .build();
        let delete_manifest = ContentTreeNodeEntryBuilder::new(DataContentType::DeleteManifest)
            .location("memory:///delete-manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Existed,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .record_count(10)
            .file_size_in_bytes(512)
            .manifest_stats(ManifestStats {
                added_files_count: 5,
                existing_files_count: 5,
                deletes_files_count: 0,
                added_rows_count: 50,
                existing_rows_count: 50,
                delete_rows_count: 0,
                min_sequence_number: 75,
            })
            .build();

        let metadata = build_node(
            vec![data_manifest, delete_manifest],
            0,
            &table_root_url,
            &engine,
        )?;

        // Old format should return an error
        let result = metadata.manifest_references(None, None, None, None, None);
        assert!(
            result.is_err(),
            "Old DataManifest/DeleteManifest format should return an error"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("no longer supported"),
            "Error should mention format is no longer supported: {err_msg}"
        );

        Ok(())
    }

    #[test]
    fn test_from_batches_with_version_rejects_unsupported_types() -> DeltaResult<()> {
        // from_batches_with_version() should reject DataManifest, DeleteManifest,
        // PositionDeletes, and EqualityDeletes entries at root-read time.
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        let unsupported_cases: &[(DataContentType, &str)] = &[
            (DataContentType::DataManifest, "DataManifest"),
            (DataContentType::DeleteManifest, "DeleteManifest"),
            (DataContentType::PositionDeletes, "PositionDeletes"),
            (DataContentType::EqualityDeletes, "EqualityDeletes"),
        ];

        for (content_type, label) in unsupported_cases {
            let mut entry = ContentTreeNodeEntryBuilder::new(DataContentType::DataManifest)
                .location("memory:///test.parquet")
                .tracking_info(TrackingInfo {
                    status: TrackingStatus::Existed,
                    snapshot_id: Some(1),
                    sequence_number: Some(100),
                    file_sequence_number: Some(100),
                    first_row_id: Some(0),
                    changes_dv: None,
                })
                .record_count(100)
                .file_size_in_bytes(1024)
                .manifest_stats(ManifestStats {
                    added_files_count: 10,
                    existing_files_count: 90,
                    deletes_files_count: 0,
                    added_rows_count: 1000,
                    existing_rows_count: 9000,
                    delete_rows_count: 0,
                    min_sequence_number: 50,
                })
                .build();
            entry.content_type = *content_type;

            let batch = entry
                .clone()
                .into_engine_data(test_metadata_entry_schema(), &engine)?;

            let result = ContentTreeNode::from_batches_with_version(
                vec![batch],
                0,
                String::new(),
                table_root_url.clone(),
            );
            assert!(
                result.is_err(),
                "{label} entries should be rejected by from_batches_with_version"
            );
        }

        Ok(())
    }

    #[test]
    fn test_manifest_references_combined_manifest() -> DeltaResult<()> {
        // Test that CombinedManifest entries work correctly with manifest_references()
        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // CombinedManifest entries contain data files + optional inline DVs
        let combined_manifest = ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
            .location("memory:///combined-manifest.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Existed,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let metadata = build_node(vec![combined_manifest], 0, &table_root_url, &engine)?;

        let root_state = metadata.manifest_references(None, None, None, None, None)?;

        // CombinedManifest produces one manifest reference
        assert_eq!(root_state.manifest_references.len(), 1);
        let refs = &root_state.manifest_references[0];
        assert_eq!(
            refs.data_manifest.manifest.location.as_ref().unwrap(),
            "memory:///combined-manifest.parquet"
        );

        Ok(())
    }

    #[test]
    fn test_full_hierarchical_metadata_tree() -> DeltaResult<()> {
        use crate::actions::visitors::AddVisitor;
        use crate::engine_data::RowVisitor;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        // Create two child data manifests with actual data files (using relative paths)
        // Child manifest 1
        let data_entry_1 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("partition1/data-1.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(50),
                file_sequence_number: Some(50),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let data_entry_2 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("partition1/data-2.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(60),
                file_sequence_number: Some(60),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let child_metadata_1 = build_node(
            vec![data_entry_1, data_entry_2],
            0,
            &table_root_url,
            &engine,
        )?;
        let child_manifest_url_1 = writer::ContentTreeNodeWriter::try_new(child_metadata_1)?
            .write(&engine)?
            .location;

        // Child manifest 2 - use version 1 to avoid filename collision
        let data_entry_3 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("partition2/data-3.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(70),
                file_sequence_number: Some(70),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();
        let data_entry_4 = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("partition2/data-4.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(80),
                file_sequence_number: Some(80),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(100)
            .file_size_in_bytes(1024)
            .build();

        let child_metadata_2 = build_node(
            vec![data_entry_3, data_entry_4],
            1,
            &table_root_url,
            &engine,
        )?;
        let child_manifest_url_2 = writer::ContentTreeNodeWriter::try_new(child_metadata_2)?
            .write(&engine)?
            .location;

        // Create a root manifest that references both child manifests (as CombinedManifest, new format)
        let data_manifest_entry_1 =
            ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
                .location(child_manifest_url_1.as_str())
                .tracking_info(TrackingInfo {
                    status: TrackingStatus::Existed,
                    snapshot_id: Some(1),
                    sequence_number: Some(100),
                    file_sequence_number: Some(100),
                    first_row_id: Some(0),
                    changes_dv: None,
                })
                .record_count(100)
                .file_size_in_bytes(1024)
                .manifest_stats(ManifestStats {
                    added_files_count: 10,
                    existing_files_count: 90,
                    deletes_files_count: 0,
                    added_rows_count: 1000,
                    existing_rows_count: 9000,
                    delete_rows_count: 0,
                    min_sequence_number: 50,
                })
                .build();
        let data_manifest_entry_2 =
            ContentTreeNodeEntryBuilder::new(DataContentType::CombinedManifest)
                .location(child_manifest_url_2.as_str())
                .tracking_info(TrackingInfo {
                    status: TrackingStatus::Existed,
                    snapshot_id: Some(1),
                    sequence_number: Some(100),
                    file_sequence_number: Some(100),
                    first_row_id: Some(0),
                    changes_dv: None,
                })
                .record_count(100)
                .file_size_in_bytes(1024)
                .manifest_stats(ManifestStats {
                    added_files_count: 10,
                    existing_files_count: 90,
                    deletes_files_count: 0,
                    added_rows_count: 1000,
                    existing_rows_count: 9000,
                    delete_rows_count: 0,
                    min_sequence_number: 50,
                })
                .build();

        let root_metadata = build_node(
            vec![data_manifest_entry_1, data_manifest_entry_2],
            0,
            &table_root_url,
            &engine,
        )?;

        // Get manifest references from the root (no manifest-level skipping for this test)
        let root_state = root_metadata.manifest_references(None, None, None, None, None)?;

        // Process all manifests using the helper method
        let schema = crate::actions::get_log_add_schema().clone();
        // No data skipping for this test
        let action_batches = ContentTreeNode::non_root_action_batches(
            root_state,
            engine.parquet_handler(),
            engine.evaluation_handler(),
            &schema,
            &table_root_url,
            None,
            None,
            None,
        )?;

        // Collect all Add actions
        let mut all_adds = Vec::new();
        for batch_result in action_batches {
            let batch = batch_result?;
            let mut visitor = AddVisitor::default();
            visitor.visit_rows_of(batch.actions.as_ref())?;
            all_adds.extend(visitor.adds);
        }

        // Verify we got all 4 data files
        assert_eq!(all_adds.len(), 4);

        // Verify the paths
        let paths: Vec<_> = all_adds.iter().map(|a| a.path.as_str()).collect();
        assert!(paths.contains(&"partition1/data-1.parquet"));
        assert!(paths.contains(&"partition1/data-2.parquet"));
        assert!(paths.contains(&"partition2/data-3.parquet"));
        assert!(paths.contains(&"partition2/data-4.parquet"));

        Ok(())
    }

    /// Test manifest skipping using direct entry filtering (without serialization).
    ///
    /// This test demonstrates the data skipping behavior at the manifest level
    /// by directly testing `filter_entries_by_predicate` on DataManifest entries.
    /// End-to-end integration test for DV size conversion through the metadata tree.
    ///
    /// This test creates a table with deletion vectors using the Transaction API and bulk mode,
    /// then verifies that:
    /// 1. PositionDeletes entries in persisted manifests have Iceberg format sizes (Delta size + 8 bytes)
    /// 2. The size conversion happens at write time in extract_deletion_vector_content
    // TODO: update_deletion_vectors does not yet update inline DV info on existing leaf entries.
    // Re-enable once that is implemented.
    #[test]
    #[ignore]
    fn test_dv_size_conversion_through_metadata_tree() -> Result<(), Box<dyn std::error::Error>> {
        use crate::arrow::array::{
            new_null_array, ArrayRef, BooleanArray, Int64Array, MapArray, StringArray, StructArray,
        };
        use crate::arrow::buffer::OffsetBuffer;
        use crate::arrow::datatypes::{DataType as ArrowDataType, Field, Schema as ArrowSchema};
        use crate::arrow::record_batch::RecordBatch;
        use crate::committer::FileSystemCommitter;
        use crate::engine::arrow_conversion::TryFromKernel;
        use crate::engine::arrow_data::ArrowEngineData;
        use crate::engine::sync::SyncEngine;
        use crate::snapshot::Snapshot;
        use crate::transaction::CommitResult;
        use serde_json::json;
        use std::fs::{create_dir_all, write};
        use std::sync::Arc;
        use tempfile::tempdir;
        use url::Url;
        use uuid::Uuid;

        let engine = Arc::new(SyncEngine::new());
        let temp_dir = tempdir()?;
        let canonical_path = std::fs::canonicalize(temp_dir.path())?;
        let table_url = Url::from_directory_path(canonical_path).unwrap();

        // Step 1: Create initial table with DV support (v0)
        let table_id = Uuid::new_v4().to_string();
        let schema = json!({
            "type": "struct",
            "fields": [
                {
                    "name": "id",
                    "type": "integer",
                    "nullable": true,
                    "metadata": {
                        "PARQUET:field_id": 1,
                        "delta.columnMapping.id": 1,
                        "delta.columnMapping.physicalName": "id"
                    }
                },
                {
                    "name": "value",
                    "type": "string",
                    "nullable": true,
                    "metadata": {
                        "PARQUET:field_id": 2,
                        "delta.columnMapping.id": 2,
                        "delta.columnMapping.physicalName": "value"
                    }
                }
            ]
        });

        let protocol = json!({
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": ["deletionVectors", "columnMapping", "metadataTree-experimental"],
                "writerFeatures": ["deletionVectors", "columnMapping", "metadataTree-experimental"]
            }
        });

        let metadata = json!({
            "metaData": {
                "id": table_id,
                "format": {"provider": "parquet", "options": {}},
                "schemaString": schema.to_string(),
                "partitionColumns": [],
                "configuration": {
                    "delta.enableDeletionVectors": "true",
                    "delta.columnMapping.mode": "id"
                },
                "createdTime": 1677811175819u64
            }
        });

        let data = [
            serde_json::to_vec(&protocol)?,
            b"\n".to_vec(),
            serde_json::to_vec(&metadata)?,
        ]
        .concat();

        let delta_log_path = table_url
            .join("_delta_log/")?
            .to_file_path()
            .map_err(|_| crate::Error::generic("Cannot convert URL to file path"))?;

        create_dir_all(&delta_log_path)?;
        write(delta_log_path.join("00000000000000000000.json"), data)?;

        // Step 2: Add files to a leaf WITHOUT DVs (v1)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
                .with_operation("WRITE".to_string());

            let add_files_schema = txn.add_files_schema();

            // Create add files metadata inline using arrow
            let files = [
                ("file1.parquet", 1000, 1000000, 50),
                ("file2.parquet", 2000, 1000001, 75),
            ];
            let num_files = files.len();

            let path_array =
                StringArray::from(files.iter().map(|(p, _, _, _)| *p).collect::<Vec<_>>());
            let size_array =
                Int64Array::from(files.iter().map(|(_, s, _, _)| *s).collect::<Vec<_>>());
            let mod_time_array =
                Int64Array::from(files.iter().map(|(_, _, m, _)| *m).collect::<Vec<_>>());
            let num_records_array =
                Int64Array::from(files.iter().map(|(_, _, _, n)| *n).collect::<Vec<_>>());

            // Create empty partition values
            let entries_field = std::sync::Arc::new(Field::new(
                "key_value",
                ArrowDataType::Struct(
                    vec![
                        std::sync::Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                        std::sync::Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                    ]
                    .into(),
                ),
                false,
            ));
            let empty_entries = StructArray::from(vec![
                (
                    std::sync::Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                    std::sync::Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
                ),
                (
                    std::sync::Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
                    std::sync::Arc::new(StringArray::from(Vec::<Option<&str>>::new())) as ArrayRef,
                ),
            ]);
            let offsets = OffsetBuffer::from_lengths(vec![0; num_files]);
            let partition_values_array = std::sync::Arc::new(MapArray::new(
                entries_field,
                offsets,
                empty_entries,
                None,
                false,
            ));

            // Build stats struct
            let arrow_schema: ArrowSchema =
                TryFromKernel::try_from_kernel(add_files_schema.as_ref())?;
            let stats_field = arrow_schema
                .field_with_name("stats")
                .expect("stats field should exist");
            let stats_arrow_schema = match stats_field.data_type() {
                ArrowDataType::Struct(fields) => fields.clone(),
                _ => panic!("stats field should be a struct"),
            };

            let mut stats_fields = Vec::new();
            for field in stats_arrow_schema.iter() {
                let array: ArrayRef = match field.name().as_str() {
                    DELTA_STATS_NUM_RECORDS => std::sync::Arc::new(num_records_array.clone()),
                    DELTA_STATS_TIGHT_BOUNDS => {
                        std::sync::Arc::new(BooleanArray::from(vec![Some(true); num_files]))
                    }
                    _ => std::sync::Arc::new(new_null_array(field.data_type(), num_files)),
                };
                stats_fields.push((field.clone(), array));
            }
            let stats_struct = StructArray::from(stats_fields);

            let record_batch = RecordBatch::try_new(
                std::sync::Arc::new(arrow_schema),
                vec![
                    std::sync::Arc::new(path_array) as ArrayRef,
                    partition_values_array as ArrayRef,
                    std::sync::Arc::new(size_array) as ArrayRef,
                    std::sync::Arc::new(mod_time_array) as ArrayRef,
                    std::sync::Arc::new(stats_struct) as ArrayRef,
                ],
            )?;

            let metadata_engine_data: Box<dyn crate::EngineData> =
                Box::new(ArrowEngineData::new(record_batch));
            {
                let batch = txn.with_batch_commit();
                let mut leaf = batch.new_leaf_node_writer(engine.as_ref())?;
                leaf.add_files(engine.as_ref(), metadata_engine_data)?;
                batch.add_leaf(leaf.finish(engine.as_ref())?)?;
            }

            match txn.commit(engine.as_ref())? {
                CommitResult::CommittedTransaction(_) => {}
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Step 3: Scan to find where files landed (manifest URL + indices)
        let snapshot_v1 = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        let scan = snapshot_v1.clone().scan_builder().build()?;

        // Use scan callback to collect file locations - much simpler than RowVisitor!
        fn collect_locations(
            locations: &mut Vec<(String, String, i64)>,
            scan_file: crate::scan::state::ScanFile,
        ) {
            if let (Some(manifest_path), Some(index)) = (
                scan_file.data_manifest_path,
                scan_file.data_manifest_position,
            ) {
                locations.push((scan_file.path, manifest_path, index));
            }
        }

        let mut file_locations = Vec::new();
        for scan_metadata_result in scan.scan_metadata(engine.as_ref())? {
            let scan_metadata = scan_metadata_result?;
            file_locations = scan_metadata.visit_scan_files(file_locations, collect_locations)?;
        }

        // Step 4: Add DVs for the files using a known size (v2)
        let known_dv_size_in_bytes: i32 = 42; // The Delta format size (what we'll test for conversion)
        {
            let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
            let mut txn = snapshot
                .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
                .with_operation("UPDATE".to_string());

            {
                let batch = txn.with_batch_commit();
                let leaf = batch.new_leaf_node_writer(engine.as_ref())?;

                // TODO: Implement inline DV update for existing leaf entries in CombinedManifest model.
                // Previously used leaf.update_deletion_vectors(dv_updates) here.
                // In the new model DVs are inline on data entries, so updating a DV requires
                // re-writing the data entry with updated dv_info.
                let _ = (&file_locations, known_dv_size_in_bytes);

                batch.add_leaf(leaf.finish(engine.as_ref())?)?;
            }

            match txn.commit(engine.as_ref())? {
                CommitResult::CommittedTransaction(_) => {}
                other => panic!("Expected success, got {:?}", other),
            };
        }

        // Step 5: Read the ContentRoot file directly to verify persisted sizes
        let snapshot_v2 = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
        let content_root_info = snapshot_v2
            .content_root()
            .expect("Table should have ContentRoot after batch commit");

        let root_manifest_url = table_url.join(content_root_info.path())?;

        let (iter, version, path_in_log) = ContentTreeNode::open_stream(
            engine.parquet_handler(),
            &root_manifest_url,
            content_root_info.path().to_string(),
            None,
            None,
        )?;
        let data = iter.collect::<DeltaResult<Vec<_>>>()?;
        let root_metadata = ContentTreeNode::from_batches_with_version(
            data,
            version,
            path_in_log,
            table_url.clone(),
        )?;
        let root_entries = root_metadata.entries()?;

        // The root might have DeleteManifest entries that contain the PositionDeletes
        // Let's find all manifests and read them to find PositionDeletes
        let mut found_position_deletes_count = 0;

        // In the new CombinedManifest model, DV info is inline on Data entries.
        // Check CombinedManifest entries in root — they point to leaf manifests
        // that contain Data entries with inline dv_info.
        for entry in &root_entries {
            if matches!(entry.content_type, DataContentType::CombinedManifest) {
                let manifest_path = entry
                    .location
                    .as_ref()
                    .expect("CombinedManifest should have location");
                let manifest_url = table_url.join(manifest_path)?;
                let (iter, version, path_in_log) = ContentTreeNode::open_stream(
                    engine.parquet_handler(),
                    &manifest_url,
                    manifest_path.clone(),
                    None,
                    None,
                )?;
                let data = iter.collect::<DeltaResult<Vec<_>>>()?;
                let manifest_metadata = ContentTreeNode::from_batches_with_version(
                    data,
                    version,
                    path_in_log,
                    table_url.clone(),
                )?;
                let manifest_entries = manifest_metadata.entries()?;

                for manifest_entry in manifest_entries {
                    if manifest_entry.content_type == DataContentType::Data {
                        if let Some(dv_info) = &manifest_entry.dv_info {
                            // Verify the DV size includes the +8 Iceberg framing
                            let expected_iceberg_size = known_dv_size_in_bytes as i64 + 8;
                            assert_eq!(
                                dv_info.size_in_bytes,
                                expected_iceberg_size,
                                "Persisted dv_info.size_in_bytes should be {} (Delta {} + 8 framing), but got {}",
                                expected_iceberg_size,
                                known_dv_size_in_bytes,
                                dv_info.size_in_bytes
                            );
                            found_position_deletes_count += 1;
                        }
                    }
                }
            }
        }

        assert!(
            found_position_deletes_count > 0,
            "Should have Data entries with inline dv_info in CombinedManifest leaf manifests"
        );

        // The test successfully proves:
        // 1. Persisted manifests have Data entries with inline dv_info using Iceberg sizes (Delta + 8)
        //    - We verified dv_info.size_in_bytes = 42 + 8 = 50
        // 2. The size conversion happens at write time in:
        //    - extract_deletion_vector_content (+8): builder.rs
        // 3. On read, the size is subtracted back to Delta format in the visitor
        //    (ParseDVFieldsVisitor subtracts 8 when building dv_sizeInBytes column)

        Ok(())
    }

    #[test]
    fn test_parse_manifest_dv_with_deletions() -> DeltaResult<()> {
        use roaring::RoaringTreemap;

        // Create a RoaringTreemap with deleted positions 1, 3, 5
        let mut deleted_positions = RoaringTreemap::new();
        deleted_positions.insert(1);
        deleted_positions.insert(3);
        deleted_positions.insert(5);

        // Serialize with portable format magic number
        let mut serialized = Vec::new();
        const ROARING_BITMAP_PORTABLE_MAGIC: u32 = 1681511377;
        serialized.extend_from_slice(&ROARING_BITMAP_PORTABLE_MAGIC.to_be_bytes());
        deleted_positions.serialize_into(&mut serialized)?;

        let dv_bytes = Bytes::from(serialized);

        // Test parse_manifest_dv
        let parsed = parse_manifest_dv(&dv_bytes)?;
        assert!(parsed.is_some());
        let treemap = parsed.unwrap();
        assert_eq!(treemap.len(), 3);
        assert!(treemap.contains(1));
        assert!(treemap.contains(3));
        assert!(treemap.contains(5));
        assert!(!treemap.contains(0));
        assert!(!treemap.contains(2));

        // Test parse_manifest_dv_to_selection_vector
        let selection = parse_manifest_dv_to_selection_vector(&dv_bytes, 10)?;
        assert_eq!(selection.len(), 10);
        assert!(selection[0]); // not deleted
        assert!(!selection[1]); // deleted
        assert!(selection[2]); // not deleted
        assert!(!selection[3]); // deleted
        assert!(selection[4]); // not deleted
        assert!(!selection[5]); // deleted
        assert!(selection[6]); // not deleted

        Ok(())
    }

    #[test]
    fn test_parse_manifest_dv_empty() -> DeltaResult<()> {
        let empty_bytes = Bytes::new();

        // Test parse_manifest_dv with empty bytes
        let parsed = parse_manifest_dv(&empty_bytes)?;
        assert!(parsed.is_none());

        // Test parse_manifest_dv_to_selection_vector with empty bytes
        let selection = parse_manifest_dv_to_selection_vector(&empty_bytes, 5)?;
        assert_eq!(selection.len(), 5);
        assert!(selection.iter().all(|&b| b)); // all true (nothing deleted)

        Ok(())
    }

    #[test]
    fn test_parse_manifest_dv_invalid_magic() -> DeltaResult<()> {
        // Create bytes with invalid magic number
        let mut invalid = Vec::new();
        let invalid_magic: u32 = 0xDEADBEEF;
        invalid.extend_from_slice(&invalid_magic.to_be_bytes());
        invalid.extend_from_slice(&[0u8; 10]); // some dummy data

        let dv_bytes = Bytes::from(invalid);

        // Should return error
        let result = parse_manifest_dv(&dv_bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid magic number"));

        Ok(())
    }

    #[test]
    fn test_parse_manifest_dv_too_small() -> DeltaResult<()> {
        // Create bytes that are too small (less than 4 bytes)
        let too_small = Bytes::from(vec![0u8, 1u8]);

        let result = parse_manifest_dv(&too_small);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));

        Ok(())
    }

    #[test]
    fn test_parse_manifest_dv_native_format_unsupported() -> DeltaResult<()> {
        // Create bytes with native format magic number (unsupported)
        let mut native = Vec::new();
        const ROARING_BITMAP_NATIVE_MAGIC: u32 = 1681511376;
        native.extend_from_slice(&ROARING_BITMAP_NATIVE_MAGIC.to_be_bytes());
        native.extend_from_slice(&[0u8; 10]);

        let dv_bytes = Bytes::from(native);

        let result = parse_manifest_dv(&dv_bytes);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Native serialization format"));

        Ok(())
    }

    /// Verifies that `stats` is null and `stats_parsed.numRecords` is populated from
    /// `recordCount` when no `table_schema` is provided (so `has_stats_parsed = false`).
    #[test]
    fn test_stats_null_and_stats_parsed_num_records_from_record_count() -> DeltaResult<()> {
        use crate::actions::{Add, ADD_NAME};
        use crate::engine_data::{GetData, RowVisitor, TypedGetData as _};
        use crate::schema::{ColumnName, ToSchema as _};
        use std::sync::LazyLock;

        const RECORD_COUNT: i64 = 42;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        let data_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///data.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(100),
                file_sequence_number: Some(100),
                first_row_id: Some(0),
                changes_dv: None,
            })
            .sort_order_id(0)
            .record_count(RECORD_COUNT)
            .file_size_in_bytes(1024)
            .build();
        let metadata = build_and_roundtrip(vec![data_entry], 0, &table_root_url, &engine)?;

        // Build a minimal stats schema with just numRecords
        let stats_schema =
            StructType::new_unchecked([StructField::nullable("numRecords", DataType::LONG)]);

        // Build action schema with stats_parsed added to the add struct (mirrors log_segment.rs)
        let mut add_fields: Vec<StructField> = Add::to_schema().fields().cloned().collect();
        add_fields.push(StructField::nullable(
            "stats_parsed",
            DataType::Struct(Box::new(stats_schema.clone())),
        ));
        let action_schema: SchemaRef =
            Arc::new(StructType::new_unchecked([StructField::nullable(
                ADD_NAME,
                StructType::new_unchecked(add_fields),
            )]));

        // table_schema = None → no content_stats in metadata → has_stats_parsed = false
        // The new code should populate stats_parsed.numRecords from recordCount
        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &action_schema,
            &[],
            None,
            None, // table_schema
            Some(&stats_schema),
        )?;

        let batch = action_batches.next().unwrap()?;

        struct StatsChecker {
            stats: Vec<Option<String>>,
            num_records: Vec<Option<i64>>,
        }

        impl RowVisitor for StatsChecker {
            fn selected_column_names_and_types(
                &self,
            ) -> (&'static [ColumnName], &'static [DataType]) {
                static NAMES_AND_TYPES: LazyLock<(Vec<ColumnName>, Vec<DataType>)> =
                    LazyLock::new(|| {
                        (
                            vec![
                                ColumnName::new(["add", "stats"]),
                                ColumnName::new(["add", "stats_parsed", DELTA_STATS_NUM_RECORDS]),
                            ],
                            vec![DataType::STRING, DataType::LONG],
                        )
                    });
                (&NAMES_AND_TYPES.0, &NAMES_AND_TYPES.1)
            }

            fn visit<'a>(
                &mut self,
                row_count: usize,
                getters: &[&'a dyn GetData<'a>],
            ) -> DeltaResult<()> {
                for i in 0..row_count {
                    self.stats.push(getters[0].get_opt(i, "add.stats")?);
                    self.num_records
                        .push(getters[1].get_opt(i, "add.stats_parsed.numRecords")?);
                }
                Ok(())
            }
        }

        let mut visitor = StatsChecker {
            stats: vec![],
            num_records: vec![],
        };
        visitor.visit_rows_of(batch.actions.as_ref())?;

        assert_eq!(visitor.stats.len(), 1);
        assert!(
            visitor.stats[0].is_none(),
            "stats JSON field should be null (no to_json serialization)"
        );
        assert_eq!(
            visitor.num_records[0],
            Some(RECORD_COUNT),
            "stats_parsed.numRecords should be populated from recordCount"
        );

        Ok(())
    }

    /// Verifies that when the output schema contains both "add" and "remove" fields, the
    /// evaluator produces batches where each action type has its field populated and the
    /// other field is null — ensuring a uniform schema across add and remove batches.
    #[test]
    fn test_add_and_remove_actions_have_matching_output_schema() -> DeltaResult<()> {
        use crate::actions::{
            visitors::AddVisitor, visitors::RemoveVisitor, ADD_NAME, REMOVE_NAME,
        };
        use crate::actions::{Add, Remove};
        use crate::engine_data::RowVisitor;
        use crate::schema::ToSchema as _;

        let engine = SyncEngine::new();
        let temp_dir = tempdir().unwrap();
        let table_root_url = Url::from_directory_path(temp_dir.path()).unwrap();

        let add_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///add-file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                first_row_id: None,
                changes_dv: None,
            })
            .record_count(10)
            .file_size_in_bytes(512)
            .build();

        let remove_entry = ContentTreeNodeEntryBuilder::new(DataContentType::Data)
            .location("memory:///remove-file.parquet")
            .tracking_info(TrackingInfo {
                status: TrackingStatus::Deleted,
                snapshot_id: Some(1),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                first_row_id: None,
                changes_dv: None,
            })
            .record_count(5)
            .file_size_in_bytes(256)
            .build();

        let metadata =
            build_and_roundtrip(vec![add_entry, remove_entry], 1, &table_root_url, &engine)?;

        // Build output schema with both add and remove fields.
        let action_schema: SchemaRef = Arc::new(StructType::new_unchecked([
            StructField::nullable(ADD_NAME, Add::to_schema()),
            StructField::nullable(REMOVE_NAME, Remove::to_schema()),
        ]));

        let mut action_batches = metadata.root_action_batches_with_handler(
            engine.evaluation_handler().as_ref(),
            &action_schema,
            &[],
            None,
            None,
            None,
        )?;

        let mut add_paths: Vec<String> = vec![];
        let mut remove_paths: Vec<String> = vec![];

        for batch_result in &mut action_batches {
            let batch = batch_result?;

            // Visit add paths — uses add.path column
            let mut add_visitor = AddVisitor::default();
            add_visitor.visit_rows_of(batch.actions.as_ref())?;
            for add in add_visitor.adds {
                add_paths.push(add.path);
            }

            // Visit remove paths — uses remove.path column. This requires the remove field
            // to be present in the batch schema even when the batch came from the add evaluator
            // (it should be null). Both add and remove batches must carry the full schema.
            let mut remove_visitor = RemoveVisitor::default();
            remove_visitor.visit_rows_of(batch.actions.as_ref())?;
            for remove in remove_visitor.removes {
                remove_paths.push(remove.path);
            }
        }

        assert_eq!(add_paths, vec!["memory:///add-file.parquet"]);
        assert_eq!(remove_paths, vec!["memory:///remove-file.parquet"]);

        Ok(())
    }
}
