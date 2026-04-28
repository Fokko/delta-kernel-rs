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

//! Add Action Generator
//!
//! A standalone tool for generating random Add actions with configurable statistics
//! and deletion vectors, then writing them to parquet files in Delta checkpoint format
//! with structured (parsed) stats.
//!
//! This tool is designed for performance benchmarking and testing.

mod analyze;
mod deletion_vector;
mod generator;
mod stats;
mod writer;

use std::process;

use clap::Parser;

#[derive(Parser)]
#[command(name = "add-action-generator")]
#[command(about = "Generate random Add actions for Delta benchmarking")]
#[command(version)]
struct Args {
    /// Number of Add actions to generate
    #[arg(short = 'n', long, default_value_t = 100)]
    num_actions: usize,

    /// Probability of generating deletion vectors (0.0-1.0)
    #[arg(short = 'd', long, default_value_t = 0.0, value_parser = validate_probability)]
    dv_probability: f64,

    /// Output parquet file path
    #[arg(short = 'o', long)]
    output: String,

    /// Random seed for reproducibility
    #[arg(short = 's', long)]
    seed: Option<u64>,

    /// Starting value for deterministic column
    #[arg(long, default_value_t = 0)]
    deterministic_start: i64,

    /// Analyze an existing parquet file instead of generating
    #[arg(short = 'a', long)]
    analyze: bool,
}

fn validate_probability(s: &str) -> Result<f64, String> {
    let prob: f64 = s.parse().map_err(|_| "Invalid number")?;
    if (0.0..=1.0).contains(&prob) {
        Ok(prob)
    } else {
        Err("Probability must be between 0.0 and 1.0".to_string())
    }
}

fn main() {
    let args = Args::parse();

    // If analyze mode, just analyze and exit
    if args.analyze {
        analyze::analyze_parquet_file(&args.output);
        return;
    }

    println!("Add Action Generator");
    println!("====================");
    println!("Number of actions: {}", args.num_actions);
    println!("DV probability: {}", args.dv_probability);
    println!("Output file: {}", args.output);
    if let Some(seed) = args.seed {
        println!("Random seed: {}", seed);
    }
    println!("Deterministic start: {}", args.deterministic_start);
    println!();

    // Generate Add actions
    println!("Generating {} Add actions...", args.num_actions);
    let actions = generator::generate_add_actions(
        args.num_actions,
        args.dv_probability,
        args.deterministic_start,
        args.seed,
    );

    // Write to parquet
    println!("Writing to parquet file: {}", args.output);
    println!("Compression: ZSTD");
    match writer::write_checkpoint_parquet(actions, &args.output) {
        Ok(()) => {
            println!("Successfully wrote checkpoint parquet file!");
            println!("File: {}", args.output);
        }
        Err(e) => {
            eprintln!("Error writing parquet file: {}", e);
            process::exit(1);
        }
    }
}
