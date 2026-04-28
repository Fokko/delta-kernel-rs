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

//! Metrics collection for benchmarks

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Collected benchmark metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Scenario name
    pub scenario: String,

    /// Table path
    pub table_path: String,

    /// Scan metrics (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_metrics: Option<ScanMetrics>,

    /// Write metrics (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_metrics: Option<WriteMetrics>,

    /// Peak memory usage in bytes (estimated)
    pub peak_memory_bytes: Option<u64>,

    /// Total elapsed time
    pub total_duration_ms: u64,
}

/// Metrics for scan operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanMetrics {
    /// Time to first task in milliseconds
    pub time_to_first_task_ms: u64,

    /// Time to enumerate all scan tasks in milliseconds
    pub time_to_enumerate_all_tasks_ms: u64,

    /// Number of tasks enumerated
    pub num_tasks: usize,

    /// Number of files scanned
    pub num_files: usize,

    /// Number of DV descriptors found
    pub num_dv_descriptors: usize,

    /// Total bytes scanned
    pub total_bytes: u64,
}

/// Metrics for write operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteMetrics {
    /// Time from transaction initialization to commit in milliseconds
    pub transaction_duration_ms: u64,

    /// Number of files written
    pub num_files_written: usize,

    /// Total bytes written
    pub total_bytes_written: u64,

    /// Whether commit succeeded
    pub commit_succeeded: bool,
}

impl BenchmarkMetrics {
    pub fn new(scenario: String, table_path: String) -> Self {
        Self {
            scenario,
            table_path,
            scan_metrics: None,
            write_metrics: None,
            peak_memory_bytes: None,
            total_duration_ms: 0,
        }
    }

    pub fn with_scan_metrics(mut self, scan_metrics: ScanMetrics) -> Self {
        self.scan_metrics = Some(scan_metrics);
        self
    }

    pub fn with_write_metrics(mut self, write_metrics: WriteMetrics) -> Self {
        self.write_metrics = Some(write_metrics);
        self
    }

    pub fn with_total_duration(mut self, duration: Duration) -> Self {
        self.total_duration_ms = duration.as_millis() as u64;
        self
    }
}

impl ScanMetrics {
    pub fn new(
        time_to_first_task: Duration,
        time_to_enumerate_all_tasks: Duration,
        num_tasks: usize,
        num_files: usize,
        num_dv_descriptors: usize,
        total_bytes: u64,
    ) -> Self {
        Self {
            time_to_first_task_ms: time_to_first_task.as_millis() as u64,
            time_to_enumerate_all_tasks_ms: time_to_enumerate_all_tasks.as_millis() as u64,
            num_tasks,
            num_files,
            num_dv_descriptors,
            total_bytes,
        }
    }
}

impl WriteMetrics {
    pub fn new(
        transaction_duration: Duration,
        num_files_written: usize,
        total_bytes_written: u64,
        commit_succeeded: bool,
    ) -> Self {
        Self {
            transaction_duration_ms: transaction_duration.as_millis() as u64,
            num_files_written,
            total_bytes_written,
            commit_succeeded,
        }
    }
}
