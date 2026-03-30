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

//! Benchmark Runner
//!
//! A tool for running performance benchmarks on Delta tables with different
//! configurations and scenarios.

mod metrics;
mod output;
mod scenarios;

use clap::{Parser, Subcommand};
use delta_kernel::expressions::{column_expr, Expression, Scalar};
use std::process;
use std::sync::Arc;
use url::Url;

#[path = "../uc_support.rs"]
mod uc_support;

#[derive(Parser)]
#[command(name = "benchmark-runner")]
#[command(about = "Run Delta table performance benchmarks")]
#[command(version)]
struct Args {
    /// Path to the Delta table (or Unity Catalog table name if using UC options)
    #[arg(short = 't', long)]
    table_path: String,

    /// Unity Catalog endpoint URL (e.g., <https://uc.example.com>)
    #[arg(long)]
    uc_endpoint: Option<String>,

    /// Unity Catalog authentication token
    #[arg(long)]
    uc_token: Option<String>,

    /// Benchmark scenario to run
    #[command(subcommand)]
    scenario: Scenario,

    /// Output format
    #[arg(short = 'o', long, default_value = "json")]
    output_format: OutputFormat,

    /// Path to write Chrome trace spans JSON (requires trace-spans feature)
    #[arg(long)]
    trace_file: Option<String>,
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Json,
    Human,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "human" => Ok(OutputFormat::Human),
            _ => Err(format!("Unknown output format: {}", s)),
        }
    }
}

#[derive(Subcommand)]
enum Scenario {
    /// Full table scan - enumerate all scan tasks
    FullTableScan,

    /// Needle-in-a-haystack - selective query with filter
    NeedleInHaystack {
        /// Partition ID to filter (e.g., 1)
        #[arg(short = 'p', long)]
        partition_id: i64,
    },

    /// Bulk write - insert many files
    BulkWrite {
        /// Number of files to write
        #[arg(short = 'n', long, default_value_t = 100000)]
        num_files: usize,

        /// Batch size for commits (1 = commit all at once)
        #[arg(short = 'b', long, default_value_t = 20000)]
        batch_size: usize,

        /// Bulk mode: true = manifest commit, false = commit each batch
        #[arg(short = 'm', default_value_t = false)]
        bulk_mode: bool,
    },

    /// Small write - delete a few rows
    SmallWrite {
        /// Number of rows to delete
        #[arg(short = 'n', long, default_value_t = 5)]
        num_files: usize,

        /// Write method: delta or content-root
        #[arg(short = 'm', default_value_t = false)]
        bulk_mode: bool,
    },

    /// Vacuum/Delete - large-scale delete operations
    VacuumDelete {
        /// Partition threshold for deletion (delete where partition_value < threshold)
        #[arg(short = 'p', long)]
        partition_threshold: i64,

        /// Delete method: delta vs content root
        #[arg(short = 'm', default_value_t = false)]
        bulk_mode: bool,
    },
}

#[cfg(feature = "trace-spans")]
fn setup_span_capture(path: &str) -> impl Drop {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::{filter::LevelFilter, prelude::*};
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(path).build();
    tracing_subscriber::registry()
        .with(chrome_layer.with_filter(LevelFilter::INFO))
        .init();
    guard
}

/// Set up the table URL and engine, either from Unity Catalog or direct path
async fn setup_table_and_engine(
    args: &Args,
) -> Result<(Url, Arc<dyn delta_kernel::Engine>), Box<dyn std::error::Error + Send + Sync>> {
    // Determine operation type based on scenario
    let operation = match args.scenario {
        Scenario::FullTableScan | Scenario::NeedleInHaystack { .. } => {
            uc_client::prelude::Operation::ReadWrite
        }
        Scenario::BulkWrite { .. }
        | Scenario::SmallWrite { .. }
        | Scenario::VacuumDelete { .. } => uc_client::prelude::Operation::ReadWrite,
    };

    // Use the common setup function
    let setup = uc_support::setup_table_access(
        &args.table_path,
        args.uc_endpoint.as_deref(),
        args.uc_token.as_deref(),
        operation,
    )
    .await?;

    Ok((setup.table_url, setup.engine))
}

/// Run the specified benchmark scenario
fn run_scenario(
    scenario: &Scenario,
    table_url: Url,
    engine: Arc<dyn delta_kernel::Engine>,
) -> delta_kernel::DeltaResult<crate::metrics::BenchmarkMetrics> {
    match scenario {
        Scenario::FullTableScan => scenarios::scan(table_url, engine, /*predicate=*/ None),
        Scenario::NeedleInHaystack { partition_id } => {
            let predicate = Some(Arc::new(
                column_expr!("id").eq(Expression::Literal(Scalar::Long(*partition_id))),
            ));
            scenarios::scan(table_url, engine, predicate)
        }
        Scenario::BulkWrite {
            num_files,
            batch_size,
            bulk_mode,
        } => scenarios::write(table_url, engine, *num_files, *batch_size, *bulk_mode),
        Scenario::SmallWrite {
            num_files,
            bulk_mode,
        } => scenarios::write(table_url, engine, *num_files, *num_files, *bulk_mode),
        Scenario::VacuumDelete {
            partition_threshold,
            bulk_mode,
        } => scenarios::vacuum_delete(table_url, engine, *partition_threshold, *bulk_mode),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    #[cfg(feature = "trace-spans")]
    let _trace_guard = args.trace_file.as_deref().map(setup_span_capture);

    // Set up table and engine
    let (table_url, engine) = match setup_table_and_engine(&args).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Failed to set up table and engine: {}", e);
            process::exit(1);
        }
    };

    println!();

    // Run the benchmark scenario
    let result = run_scenario(&args.scenario, table_url, engine);

    match result {
        Ok(metrics) => {
            // Output results
            match args.output_format {
                OutputFormat::Json => {
                    println!("{}", output::to_json(&metrics));
                }
                OutputFormat::Human => {
                    output::print_human(&metrics);
                }
            }
        }
        Err(e) => {
            eprintln!("Benchmark failed: {}", e);
            process::exit(1);
        }
    }
}
