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

//! Shared utilities for table file operations.
//!
//! This module provides common functionality for working with Delta table files,
//! including cleanup and copy operations used by benchmarking tools.

use std::sync::Arc;

use object_store::{ObjectStore, ObjectStoreExt as _};
use tokio::sync::Semaphore;
use url::Url;

/// Statistics from a cleanup operation
#[derive(Debug, Default)]
pub struct CleanupStats {
    /// Total number of files deleted
    pub files_deleted: usize,
    /// Total bytes deleted
    pub total_bytes_deleted: u64,
    /// Number of _delta_log files deleted
    pub delta_log_files_deleted: usize,
    /// Number of data files deleted
    pub data_files_deleted: usize,
    /// Errors encountered during cleanup (non-fatal)
    pub errors: Vec<String>,
}

impl CleanupStats {
    /// Create a new empty CleanupStats
    pub fn new() -> Self {
        Self::default()
    }
}

/// Remove all files under a table location.
///
/// This function will:
/// - List all files recursively under the table location
/// - Delete each file concurrently (up to 50 concurrent deletions)
/// - Track statistics about what was deleted
/// - Continue on individual file errors (collecting them for later review)
/// - Log each file being removed for transparency
///
/// # Arguments
///
/// * `store` - The object store to use for listing and deleting files
/// * `table_url` - The base URL of the table
/// * `path_prefix` - The path prefix within the object store (empty string for file:// URLs)
///
/// # Returns
///
/// Returns `CleanupStats` with information about the cleanup operation, including
/// any non-fatal errors encountered during deletion.
///
/// # Errors
///
/// Returns an error if:
/// - The initial file listing fails
/// - All deletions fail (individual failures are collected in `errors`)
pub async fn remove_all_table_files(
    store: &Arc<dyn ObjectStore>,
    table_url: &Url,
    path_prefix: &str,
) -> Result<CleanupStats, Box<dyn std::error::Error + Send + Sync>> {
    let mut stats = CleanupStats::new();

    println!("Cleaning table files at: {}", table_url);
    if !path_prefix.is_empty() {
        println!("  Path prefix: {}", path_prefix);
    }

    // List all files under the table location
    let list_prefix = if path_prefix.is_empty() {
        None
    } else {
        Some(object_store::path::Path::from(
            path_prefix.trim_end_matches('/'),
        ))
    };

    use futures::stream::StreamExt;
    let mut file_list = Vec::new();
    let mut skipped_files = 0;
    let mut list_stream = store.list(list_prefix.as_ref());
    while let Some(result) = list_stream.next().await {
        match result {
            Ok(meta) => {
                let path_str = meta.location.to_string();

                // Filter: only include files that are within the table directory
                // Skip files that don't have the path_prefix or are in parent directories
                if !path_prefix.is_empty() && !path_str.starts_with(path_prefix) {
                    skipped_files += 1;
                    continue;
                }

                // Filter: skip any files that look like they're from parent directories
                let relative = if !path_prefix.is_empty() {
                    path_str.strip_prefix(path_prefix).unwrap_or(&path_str)
                } else {
                    &path_str
                };

                // Skip _staged_commits directory
                if relative.contains("_staged_commits") {
                    skipped_files += 1;
                    continue;
                }

                // Only include _delta_log files and .parquet data files
                if relative.contains("_delta_log/")
                    || relative.ends_with(".parquet")
                    || relative.ends_with(".json")
                {
                    file_list.push(meta);
                } else {
                    skipped_files += 1;
                }
            }
            Err(e) => {
                stats.errors.push(format!("Error listing files: {}", e));
            }
        }
    }

    if skipped_files > 0 {
        println!("  Skipped {} files outside table directory", skipped_files);
    }

    if file_list.is_empty() {
        println!("  No files found to delete");
        return Ok(stats);
    }

    println!("  Found {} files to delete", file_list.len());

    // Delete files concurrently with a semaphore to limit concurrency
    let semaphore = Arc::new(Semaphore::new(50));
    let mut delete_tasks = Vec::new();

    for meta in file_list {
        let store = Arc::clone(store);
        let path = meta.location.clone();
        let size = meta.size;
        let sem = Arc::clone(&semaphore);

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            // Determine file type for statistics
            let path_str = path.to_string();
            let is_delta_log = path_str.contains("_delta_log");

            match store.delete(&path).await {
                Ok(_) => {
                    println!("    ✓ Deleted: {}", path);
                    Ok((size, is_delta_log))
                }
                Err(e) => {
                    eprintln!("    ⚠️  Failed to delete {}: {} (continuing...)", path, e);
                    Err(format!("Failed to delete {}: {}", path, e))
                }
            }
        });

        delete_tasks.push(task);
    }

    // Wait for all deletions to complete
    for task in delete_tasks {
        match task.await {
            Ok(Ok((size, is_delta_log))) => {
                stats.files_deleted += 1;
                stats.total_bytes_deleted += size;
                if is_delta_log {
                    stats.delta_log_files_deleted += 1;
                } else {
                    stats.data_files_deleted += 1;
                }
            }
            Ok(Err(e)) => {
                stats.errors.push(e);
            }
            Err(e) => {
                stats.errors.push(format!("Task join error: {}", e));
            }
        }
    }

    println!("  Cleanup complete:");
    println!("    Files deleted: {}", stats.files_deleted);
    println!("    Bytes deleted: {}", stats.total_bytes_deleted);
    println!("    Delta log files: {}", stats.delta_log_files_deleted);
    println!("    Data files: {}", stats.data_files_deleted);

    if !stats.errors.is_empty() {
        println!("    Errors: {}", stats.errors.len());
        for error in &stats.errors {
            eprintln!("      - {}", error);
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cleanup_empty_location() {
        // Create a temporary directory for testing
        let temp_dir = tempfile::tempdir().unwrap();
        let table_path = temp_dir.path().join("test_table");
        std::fs::create_dir_all(&table_path).unwrap();

        let store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&table_path).unwrap());
        let table_url = url::Url::from_directory_path(&table_path).unwrap();

        let stats = remove_all_table_files(&store, &table_url, "")
            .await
            .unwrap();

        assert_eq!(stats.files_deleted, 0);
        assert_eq!(stats.total_bytes_deleted, 0);
        assert_eq!(stats.errors.len(), 0);
    }

    #[tokio::test]
    async fn test_cleanup_with_files() {
        // Create a temporary directory with some test files
        let temp_dir = tempfile::tempdir().unwrap();
        let table_path = temp_dir.path().join("test_table");
        std::fs::create_dir_all(&table_path).unwrap();
        std::fs::create_dir_all(table_path.join("_delta_log")).unwrap();

        // Create some test files
        std::fs::write(
            table_path.join("_delta_log/00000000000000000000.json"),
            "test",
        )
        .unwrap();
        std::fs::write(table_path.join("data-file.parquet"), "data").unwrap();

        let store: Arc<dyn ObjectStore> =
            Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&table_path).unwrap());
        let table_url = url::Url::from_directory_path(&table_path).unwrap();

        let stats = remove_all_table_files(&store, &table_url, "")
            .await
            .unwrap();

        assert_eq!(stats.files_deleted, 2);
        assert!(stats.total_bytes_deleted > 0);
        assert_eq!(stats.delta_log_files_deleted, 1);
        assert_eq!(stats.data_files_deleted, 1);
        assert_eq!(stats.errors.len(), 0);

        // Verify files are gone
        assert!(!table_path
            .join("_delta_log/00000000000000000000.json")
            .exists());
        assert!(!table_path.join("data-file.parquet").exists());
    }
}
