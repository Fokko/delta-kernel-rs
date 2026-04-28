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

//! Unity Catalog Table Copy Tool
//!
//! This tool copies Delta tables at the file level using Unity Catalog credentials.
//! It uses separate engines with Read and ReadWrite credentials because UC scopes
//! credentials per table and operation type.

use std::sync::Arc;

use bench_utils::{table_utils, uc_support};
use clap::Parser;
use futures::stream::StreamExt;
use object_store::ObjectStoreExt as _;
use tokio::sync::Semaphore;
use unity_catalog_delta_client_api::Operation;

#[derive(Parser)]
#[command(name = "uc-table-copy")]
#[command(about = "Copy Unity Catalog Delta tables at the file level")]
#[command(version)]
struct Args {
    /// Source table name (catalog.schema.table)
    #[arg(short = 's', long)]
    source_table: String,

    /// Destination table name (catalog.schema.table)
    #[arg(short = 'd', long)]
    dest_table: String,

    /// Unity Catalog endpoint URL (e.g., `https://uc.example.com`)
    #[arg(long)]
    uc_endpoint: String,

    /// Unity Catalog authentication token
    #[arg(long)]
    uc_token: String,

    /// Clear destination table before copying
    #[arg(long, default_value_t = false)]
    clear_dest: bool,
}

/// Statistics from a copy operation
#[derive(Debug, Default)]
struct CopyStats {
    files_copied: usize,
    bytes_copied: u64,
    delta_log_files: usize,
    data_files: usize,
    errors: Vec<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("Unity Catalog Table Copy Tool");
    println!("==============================");
    println!();
    println!("Source table: {}", args.source_table);
    println!("Destination table: {}", args.dest_table);
    println!("UC Endpoint: {}", args.uc_endpoint);
    println!("Clear destination: {}", args.clear_dest);
    println!();

    if let Err(e) = run(&args).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    println!("\n✓ Table copy complete!");
}

async fn run(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let uc_config = uc_support::UCConfig {
        endpoint: args.uc_endpoint.clone(),
        token: args.uc_token.clone(),
    };

    // Step 1: Get source table info and create engine with Read credentials
    println!("1. Getting source table information...");
    let source_info = uc_support::get_table_info(&uc_config, &args.source_table).await?;
    println!();

    println!("2. Creating source engine with Read credentials...");
    let (_, _source_url, source_prefix, source_store) =
        uc_support::create_engine_with_uc_credentials(&uc_config, &source_info, Operation::Read)
            .await?;
    println!();

    // Step 2: Get destination table info and create engine with ReadWrite credentials
    println!("3. Getting destination table information...");
    let dest_info = uc_support::get_table_info(&uc_config, &args.dest_table).await?;
    println!();

    println!("4. Creating destination engine with ReadWrite credentials...");
    let (_, dest_url, dest_prefix, dest_store) =
        uc_support::create_engine_with_uc_credentials(&uc_config, &dest_info, Operation::ReadWrite)
            .await?;
    println!();

    // Step 3: Clear destination if requested
    if args.clear_dest {
        println!("5. Clearing destination table...");
        let cleanup_stats =
            table_utils::remove_all_table_files(&dest_store, &dest_url, &dest_prefix).await?;
        if !cleanup_stats.errors.is_empty() {
            println!(
                "  WARNING: {} errors occurred during cleanup",
                cleanup_stats.errors.len()
            );
        }
        println!();
    }

    // Step 4: Copy all files from source to destination
    let step_num = if args.clear_dest { 6 } else { 5 };
    println!("{}. Copying files from source to destination...", step_num);
    let copy_stats =
        copy_table_files(source_store, &source_prefix, dest_store, &dest_prefix).await?;

    println!();
    println!("Copy statistics:");
    println!("  Files copied: {}", copy_stats.files_copied);
    println!("  Bytes copied: {}", copy_stats.bytes_copied);
    println!("  Delta log files: {}", copy_stats.delta_log_files);
    println!("  Data files: {}", copy_stats.data_files);

    if !copy_stats.errors.is_empty() {
        println!(
            "  ⚠️  Warnings: {} files failed to copy",
            copy_stats.errors.len()
        );
        for error in &copy_stats.errors {
            eprintln!("    - {}", error);
        }
        println!("  Note: Failures are non-fatal. Continuing...");
    }

    Ok(())
}

/// Copy all files from source to destination table
async fn copy_table_files(
    source_store: Arc<dyn object_store::ObjectStore>,
    source_prefix: &str,
    dest_store: Arc<dyn object_store::ObjectStore>,
    dest_prefix: &str,
) -> Result<CopyStats, Box<dyn std::error::Error + Send + Sync>> {
    let mut stats = CopyStats::default();

    // List all files from source
    let list_prefix = if source_prefix.is_empty() {
        None
    } else {
        Some(object_store::path::Path::from(
            source_prefix.trim_end_matches('/'),
        ))
    };

    let mut file_list = Vec::new();
    let mut skipped_files = 0;
    let mut list_stream = source_store.list(list_prefix.as_ref());

    while let Some(result) = list_stream.next().await {
        match result {
            Ok(meta) => {
                let path_str = meta.location.to_string();

                // Only include _delta_log files and .parquet data files
                if path_str.contains("_delta_log/")
                    || path_str.ends_with(".parquet")
                    || path_str.ends_with(".json")
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
        println!("  No files found to copy");
        return Ok(stats);
    }

    println!("  Found {} files to copy", file_list.len());

    // Copy files concurrently with a semaphore to limit concurrency
    let semaphore = Arc::new(Semaphore::new(50));
    let mut copy_tasks = Vec::new();

    for meta in file_list {
        let source_store = Arc::clone(&source_store);
        let dest_store = Arc::clone(&dest_store);
        let source_path = meta.location.clone();
        let size = meta.size;
        let sem = Arc::clone(&semaphore);

        let source_prefix = source_prefix.to_string();
        let dest_prefix = dest_prefix.to_string();

        let task = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();

            // Determine relative path within the table
            let source_path_str = source_path.to_string();
            let relative_path = if !source_prefix.is_empty() {
                source_path_str
                    .strip_prefix(&source_prefix)
                    .unwrap_or(&source_path_str)
            } else {
                &source_path_str
            };

            // Build destination path
            let dest_path_str = if !dest_prefix.is_empty() {
                format!("{}{}", dest_prefix, relative_path)
            } else {
                relative_path.to_string()
            };
            let dest_path = object_store::path::Path::from(dest_path_str);

            // Determine file type for statistics
            let is_delta_log = source_path_str.contains("_delta_log");

            // Get file from source and put to destination
            match source_store.get(&source_path).await {
                Ok(get_result) => match get_result.bytes().await {
                    Ok(bytes) => match dest_store.put(&dest_path, bytes.into()).await {
                        Ok(_) => {
                            println!("    ✓ Copied: {} -> {}", source_path, dest_path);
                            Ok((size, is_delta_log))
                        }
                        Err(e) => {
                            eprintln!("    ⚠️  Failed to put {}: {} (continuing...)", dest_path, e);
                            Err(format!("Failed to put {}: {}", dest_path, e))
                        }
                    },
                    Err(e) => {
                        eprintln!(
                            "    ⚠️  Failed to read bytes from {}: {} (continuing...)",
                            source_path, e
                        );
                        Err(format!("Failed to read bytes from {}: {}", source_path, e))
                    }
                },
                Err(e) => {
                    eprintln!(
                        "    ⚠️  Failed to get {}: {} (continuing...)",
                        source_path, e
                    );
                    Err(format!("Failed to get {}: {}", source_path, e))
                }
            }
        });

        copy_tasks.push(task);
    }

    // Wait for all copies to complete
    for task in copy_tasks {
        match task.await {
            Ok(Ok((size, is_delta_log))) => {
                stats.files_copied += 1;
                stats.bytes_copied += size;
                if is_delta_log {
                    stats.delta_log_files += 1;
                } else {
                    stats.data_files += 1;
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

    Ok(stats)
}
