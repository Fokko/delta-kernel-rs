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

//! Parquet file analyzer to show size breakdown by column

use std::fs::File;

use delta_kernel::parquet::file::reader::{FileReader, SerializedFileReader};

pub fn analyze_parquet_file(path: &str) {
    let file = File::open(path).expect("Failed to open file");
    let reader = SerializedFileReader::new(file).expect("Failed to create reader");
    let metadata = reader.metadata();

    println!("\n=== Parquet File Analysis ===");
    println!("File: {}", path);
    println!("Number of rows: {}", metadata.file_metadata().num_rows());
    println!("Number of row groups: {}", metadata.num_row_groups());
    println!();

    // Collect column sizes across all row groups
    let mut column_sizes: std::collections::HashMap<String, (u64, u64)> =
        std::collections::HashMap::new();

    for i in 0..metadata.num_row_groups() {
        let rg_metadata = metadata.row_group(i);
        for j in 0..rg_metadata.num_columns() {
            let col_metadata = rg_metadata.column(j);
            let col_path = col_metadata.column_path().string();
            let compressed_size = col_metadata.compressed_size() as u64;
            let uncompressed_size = col_metadata.uncompressed_size() as u64;

            let entry = column_sizes.entry(col_path.clone()).or_insert((0, 0));
            entry.0 += compressed_size;
            entry.1 += uncompressed_size;
        }
    }

    // Sort by compressed size
    let mut columns: Vec<_> = column_sizes.into_iter().collect();
    columns.sort_by_key(|col| std::cmp::Reverse(col.1 .0));

    println!("Column Size Breakdown (sorted by compressed size):");
    println!(
        "{:<60} {:>15} {:>15} {:>10}",
        "Column Path", "Compressed", "Uncompressed", "Ratio"
    );
    println!("{}", "-".repeat(110));

    let mut total_compressed = 0u64;
    let mut total_uncompressed = 0u64;

    for (col_path, (compressed, uncompressed)) in &columns {
        total_compressed += compressed;
        total_uncompressed += uncompressed;

        let ratio = if *uncompressed > 0 {
            (*compressed as f64 / *uncompressed as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<60} {:>12} MB {:>12} MB {:>9.1}%",
            col_path,
            *compressed as f64 / (1024.0 * 1024.0),
            *uncompressed as f64 / (1024.0 * 1024.0),
            ratio
        );
    }

    println!("{}", "-".repeat(110));
    println!(
        "{:<60} {:>12} MB {:>12} MB {:>9.1}%",
        "TOTAL",
        total_compressed as f64 / (1024.0 * 1024.0),
        total_uncompressed as f64 / (1024.0 * 1024.0),
        (total_compressed as f64 / total_uncompressed as f64) * 100.0
    );
    println!();
}
