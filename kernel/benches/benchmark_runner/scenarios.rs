// Copyright 2025 The Delta Kernel Rust Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Benchmark scenario implementations

use crate::metrics::{BenchmarkMetrics, ScanMetrics, WriteMetrics};
use delta_kernel::arrow::array::{ArrayRef, Int64Array, MapArray, StringArray, StructArray};
use delta_kernel::arrow::buffer::OffsetBuffer;
use delta_kernel::arrow::datatypes::{DataType as ArrowDataType, Field};
use delta_kernel::arrow::record_batch::RecordBatch;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::engine::arrow_conversion::TryFromKernel;
use delta_kernel::engine::arrow_data::ArrowEngineData;
// Note: PredicateRef is still used for scan predicate parameter

use delta_kernel::transaction::Transaction;
use delta_kernel::{DeltaResult, Engine, EngineData, PredicateRef};
use std::sync::Arc;
use std::time::Instant;
use url::Url;

/// Context for counting scan files using the callback pattern
struct ScanFileCounter {
    num_files: usize,
    num_dv_descriptors: usize,
}

/// Callback function to count each scan file
fn count_scan_file(context: &mut ScanFileCounter, scan_file: delta_kernel::scan::state::ScanFile) {
    context.num_files += 1;
    if scan_file.dv_info.has_vector() {
        context.num_dv_descriptors += 1;
    }
}

/// Full table scan - list all scan tasks with no filters
pub fn scan(
    table_url: Url,
    engine: Arc<dyn Engine>,
    predicate: Option<PredicateRef>,
) -> DeltaResult<BenchmarkMetrics> {
    let start = Instant::now();

    // Load snapshot
    let snapshot = {
        let _span = tracing::info_span!("benchmark.scan.load_snapshot").entered();
        delta_kernel::snapshot::Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?
    };

    // Save whether predicate is present before moving it
    let has_predicate = predicate.is_some();

    // Create scan with no filters
    let scan = snapshot.scan_builder().with_predicate(predicate).build()?;

    // Measure time to first task
    let first_task_start = Instant::now();

    let (first_task, time_to_first_task, metadata_iter) = {
        let _span = tracing::info_span!("benchmark.scan.time_to_first_file").entered();
        let mut iter = {
            let _span = tracing::info_span!("benchmark.scan.scan_metadata_setup").entered();
            scan.scan_metadata(engine.as_ref())?
        };
        let first = {
            let _span = tracing::info_span!("benchmark.scan.first_next").entered();
            iter.next()
        };
        let elapsed = first_task_start.elapsed();
        (first, elapsed, iter)
    };

    // Count first task if it exists using visit_scan_files callback
    let mut num_tasks = if first_task.is_some() { 1 } else { 0 };
    let mut context = ScanFileCounter {
        num_files: 0,
        num_dv_descriptors: 0,
    };

    // Visit files in first task using callback pattern
    if let Some(Ok(metadata)) = first_task {
        context = metadata.visit_scan_files(context, count_scan_file)?;
    }

    // Enumerate remaining tasks and visit files using callback pattern
    {
        let _span = tracing::info_span!("benchmark.scan.enumerate_remaining_files").entered();
        for result in metadata_iter {
            let metadata = result?;
            num_tasks += 1;
            context = metadata.visit_scan_files(context, count_scan_file)?;
        }
    }

    let time_to_enumerate_all = first_task_start.elapsed();
    let total_duration = start.elapsed();

    let scan_metrics = ScanMetrics::new(
        time_to_first_task,
        time_to_enumerate_all,
        num_tasks,
        context.num_files,
        context.num_dv_descriptors,
        0, // total_bytes not easily available from scan_metadata
    );

    Ok(BenchmarkMetrics::new(
        format!("scan_with_predicate={}", has_predicate),
        table_url.to_string(),
    )
    .with_scan_metrics(scan_metrics)
    .with_total_duration(total_duration))
}

/// Helper to create a null array of the specified Arrow data type and length
fn new_null_array(data_type: &ArrowDataType, length: usize) -> ArrayRef {
    use delta_kernel::arrow::array::new_null_array as arrow_new_null_array;

    arrow_new_null_array(data_type, length)
}

/// Helper to create add file metadata
fn create_add_files_metadata(
    add_files_schema: &delta_kernel::schema::SchemaRef,
    num_files: usize,
    start_index: usize,
) -> DeltaResult<Box<dyn delta_kernel::EngineData>> {
    // Generate synthetic file metadata with unique file names
    let paths: Vec<String> = (start_index..start_index + num_files)
        .map(|i| format!("part-{:05}.parquet", i))
        .collect();
    let sizes: Vec<i64> = (0..num_files)
        .map(|_| 1024 * 1024) // 1MB per file
        .collect();
    let mod_times: Vec<i64> = (0..num_files)
        .map(|_| 1704067200000) // Fixed timestamp (2024-01-01)
        .collect();
    let num_records: Vec<i64> = (0..num_files)
        .map(|_| 1000) // 1000 records per file
        .collect();

    // Build arrays for each file
    let path_array = StringArray::from(paths);
    let size_array = Int64Array::from(sizes);
    let mod_time_array = Int64Array::from(mod_times);
    let num_records_array = Int64Array::from(num_records);

    // Create empty map for partitionValues (repeated for each file)
    let entries_field = Arc::new(Field::new(
        "key_value",
        ArrowDataType::Struct(
            vec![
                Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
                Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
            ]
            .into(),
        ),
        false,
    ));
    let empty_keys = StringArray::from(Vec::<&str>::new());
    let empty_values = StringArray::from(Vec::<Option<&str>>::new());
    let empty_entries = StructArray::from(vec![
        (
            Arc::new(Field::new("key", ArrowDataType::Utf8, false)),
            Arc::new(empty_keys) as ArrayRef,
        ),
        (
            Arc::new(Field::new("value", ArrowDataType::Utf8, true)),
            Arc::new(empty_values) as ArrayRef,
        ),
    ]);
    let offsets = OffsetBuffer::from_lengths(vec![0; num_files]);
    let partition_values_array = Arc::new(MapArray::new(
        entries_field,
        offsets,
        empty_entries,
        None,
        false,
    ));

    // Convert kernel schema to Arrow schema to get the actual stats field structure
    let arrow_schema: Arc<delta_kernel::arrow::datatypes::Schema> = Arc::new(
        TryFromKernel::try_from_kernel(add_files_schema.as_ref()).map_err(|e| {
            delta_kernel::Error::generic(format!("Failed to convert schema: {}", e))
        })?,
    );

    // Extract the stats field schema from the converted Arrow schema
    let stats_field = arrow_schema
        .field_with_name("stats")
        .map_err(|e| delta_kernel::Error::generic(format!("Failed to find stats field: {}", e)))?;

    // Build the stats struct to match the expected schema
    // The stats field is a struct with fields: numRecords, nullCount, minValues, maxValues, tightBounds
    let stats_struct = if let ArrowDataType::Struct(stats_fields) = stats_field.data_type() {
        let mut field_arrays: Vec<(Arc<Field>, ArrayRef)> = Vec::new();

        for field in stats_fields.iter() {
            match field.name().as_str() {
                "numRecords" => {
                    // Provide actual numRecords values
                    field_arrays.push((
                        field.clone(),
                        Arc::new(num_records_array.clone()) as ArrayRef,
                    ));
                }
                _ => {
                    // For all other fields (nullCount, minValues, maxValues, tightBounds), use null arrays
                    let null_array = new_null_array(field.data_type(), num_files);
                    field_arrays.push((field.clone(), null_array));
                }
            }
        }

        StructArray::from(field_arrays)
    } else {
        return Err(delta_kernel::Error::generic(
            "Stats field is not a struct type",
        ));
    };

    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(path_array) as ArrayRef,
            partition_values_array as ArrayRef,
            Arc::new(size_array) as ArrayRef,
            Arc::new(mod_time_array) as ArrayRef,
            Arc::new(stats_struct) as ArrayRef,
        ],
    )
    .map_err(|e| delta_kernel::Error::generic(format!("Failed to create record batch: {}", e)))?;

    Ok(Box::new(ArrowEngineData::new(batch)))
}

/// Bulk write - insert many files
pub fn write(
    table_url: Url,
    engine: Arc<dyn Engine>,
    num_files: usize,
    batch_size: usize,
    bulk_mode: bool,
) -> DeltaResult<BenchmarkMetrics> {
    let start = Instant::now();

    // Load or create snapshot
    let snapshot = {
        let _span = tracing::info_span!("benchmark.write.load_snapshot").entered();
        match delta_kernel::snapshot::Snapshot::builder_for(table_url.clone())
            .build(engine.as_ref())
        {
            Ok(snapshot) => snapshot,
            Err(_) => {
                return Err(delta_kernel::Error::generic(
                    "Table does not exist. Bulk write to new tables not yet implemented.",
                ));
            }
        }
    };

    let txn_start = Instant::now();

    // Create transaction
    let mut txn = snapshot
        .clone()
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("benchmark-runner")
        .with_operation("WRITE".to_string())
        .with_data_change(true);

    if bulk_mode {
        txn.with_manifest_commit();
    }

    let add_files_schema = txn.add_files_schema();

    let num_batches = num_files / batch_size;
    let batches = {
        let _span =
            tracing::info_span!("benchmark.write.build_batches", num_batches, batch_size).entered();
        let mut batches = Vec::new();
        for batch_idx in 0..num_batches {
            batches.push(create_add_files_metadata(
                add_files_schema,
                batch_size,
                batch_idx * batch_size,
            )?);
        }
        batches
    };

    {
        let _span = tracing::info_span!("benchmark.write.add_to_txn", bulk_mode).entered();
        add_batches_to_txn(&mut txn, batches, bulk_mode, engine.clone())?;
    }

    // Commit transaction
    {
        let _span = tracing::info_span!("benchmark.write.commit").entered();
        match txn.commit(engine.as_ref())? {
            delta_kernel::transaction::CommitResult::CommittedTransaction(_) => {}
            _ => {
                return Err(delta_kernel::Error::generic("Commit failed"));
            }
        }
    }

    let transaction_duration = txn_start.elapsed();
    let total_duration = start.elapsed();

    let write_metrics = WriteMetrics::new(
        transaction_duration,
        num_files,
        num_files as u64 * 1024 * 1024, // 1MB per file
        true,                           // commit succeeded
    );

    Ok(BenchmarkMetrics::new(
        format!(
            "write_{}__{}_files_bulk_mode_{}",
            num_batches, num_files, bulk_mode
        ),
        table_url.to_string(),
    )
    .with_write_metrics(write_metrics)
    .with_total_duration(total_duration))
}

fn add_batches_to_txn(
    txn: &mut Transaction,
    batches: Vec<Box<dyn EngineData>>,
    bulk_mode: bool,
    engine: Arc<dyn Engine>,
) -> DeltaResult<()> {
    if bulk_mode {
        use std::thread;

        let mc = txn.with_manifest_commit();

        // Create leaf writers for each data batch and spawn threads to finish them
        let mut handles = Vec::new();

        for data in batches {
            let mut leaf = mc.new_leaf_node_writer(engine.as_ref())?;
            leaf.add_files(engine.as_ref(), data)?;

            // Clone engine for thread
            let engine_clone = engine.clone();

            // Spawn thread to finish the leaf writer
            let handle = thread::spawn(move || leaf.finish(engine_clone.as_ref()));
            handles.push(handle);
        }

        // Collect results from threads and add to transaction
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| delta_kernel::Error::generic("Thread panicked"))?;
            mc.add_leaf(result?)?;
        }
    } else {
        for data in batches {
            txn.add_files(data);
        }
    }
    Ok(())
}

/// Vacuum/Delete - large-scale delete operations
pub fn vacuum_delete(
    table_url: Url,
    engine: Arc<dyn Engine>,
    partition_threshold: i64,
    bulk_mode: bool,
) -> DeltaResult<BenchmarkMetrics> {
    let start = Instant::now();

    // Load snapshot
    let snapshot = {
        let _span = tracing::info_span!("benchmark.vacuum_delete.load_snapshot").entered();
        delta_kernel::snapshot::Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?
    };

    // Create scan without predicate to get all files
    let scan = snapshot.clone().scan_builder().build()?;

    let mut batches_to_delete = Vec::new();
    let mut files_collected = 0;

    let (total_files, all_batches) = {
        let _span = tracing::info_span!("benchmark.vacuum_delete.scan_all_files").entered();
        let mut total_files = 0;
        let mut all_batches = Vec::new();
        for result in scan.scan_metadata(engine.as_ref())? {
            let metadata = result?;
            total_files += metadata.scan_files.data().len();
            all_batches.push(metadata.scan_files);
        }
        (total_files, all_batches)
    };

    let files_to_delete = (total_files as f64 * partition_threshold as f64 / 100.0) as usize;

    // Collect batches until we reach the target
    for batch in all_batches {
        let num_files = batch.data().len();

        if files_collected + num_files <= files_to_delete {
            // Take the entire batch
            batches_to_delete.push(batch);
            files_collected += num_files;

            if files_collected >= files_to_delete {
                break;
            }
        } else {
            // Take a partial batch using a selection vector
            let remaining = files_to_delete - files_collected;
            if remaining > 0 {
                // Get the existing selection vector from the batch
                let (data, old_selection) = batch.into_parts();

                // Build a new selection vector that respects the old one
                // We need to select only the first 'remaining' SELECTED rows
                let mut new_selection = Vec::new();
                let mut selected_count = 0;

                for i in 0..data.len() {
                    // Check if this row was selected in the original batch
                    let is_selected = if i < old_selection.len() {
                        old_selection[i]
                    } else {
                        // Per FilteredEngineData contract: if selection vector is shorter than data,
                        // remaining rows are assumed to be selected
                        true
                    };

                    // If it was selected and we haven't reached our limit, keep it selected
                    if is_selected && selected_count < remaining {
                        new_selection.push(true);
                        selected_count += 1;
                    } else if is_selected {
                        // It was selected but we've reached our limit, so deselect it
                        new_selection.push(false);
                    } else {
                        // It wasn't selected in the original, keep it unselected
                        new_selection.push(false);
                    }
                }

                // Create a new FilteredEngineData with the combined selection vector
                let filtered_batch =
                    delta_kernel::engine_data::FilteredEngineData::try_new(data, new_selection)?;

                batches_to_delete.push(filtered_batch);
            }
            break;
        }
    }

    let txn_start = Instant::now();

    {
        let _span = tracing::info_span!("benchmark.vacuum_delete.commit", bulk_mode).entered();

        // Create transaction
        let mut txn = snapshot
            .clone()
            .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
            .with_engine_info("benchmark-runner")
            .with_operation("DELETE".to_string())
            .with_data_change(true);

        if bulk_mode {
            txn.with_manifest_commit();
        }

        for batch in batches_to_delete {
            txn.remove_files(batch);
        }

        match txn.commit(engine.as_ref())? {
            delta_kernel::transaction::CommitResult::CommittedTransaction(_) => {}
            _ => {
                return Err(delta_kernel::Error::generic("Commit failed"));
            }
        }
    }

    let transaction_duration = txn_start.elapsed();
    let total_duration = start.elapsed();

    let write_metrics = WriteMetrics::new(transaction_duration, 0, 0, true);

    Ok(BenchmarkMetrics::new(
        format!("vacuum_delete_threshold_{}", partition_threshold),
        table_url.to_string(),
    )
    .with_write_metrics(write_metrics)
    .with_total_duration(total_duration))
}
