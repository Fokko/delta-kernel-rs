//! Backfill Delta Table with Snapshot V2 and Sidecar Files
//!
//! This script creates a Delta table from scratch with:
//! - Commit 0 containing Metadata and Protocol actions
//! - Checkpoint at version 0
//! - 20 sidecar files with 50K Add actions each
//! - Configurable deletion vector percentage

use std::collections::HashMap;
#[allow(unused_imports)] // Used in file:// URL path, may appear unused in some build configs
use std::fs;
use std::process;

use clap::Parser;
use delta_kernel::object_store::ObjectStoreExt as _;
use serde::{Deserialize, Serialize};
use serde_json::json;
use unity_catalog_delta_client_api::Operation;

// Share the generator modules from the same directory
#[path = "deletion_vector.rs"]
mod deletion_vector;
#[path = "generator.rs"]
mod generator;
#[path = "stats.rs"]
mod stats;
#[path = "writer.rs"]
mod writer;

// Shared benchmark utilities (using path-based modules to avoid cyclic dependencies)
#[path = "../table_utils.rs"]
mod table_utils;
#[path = "../uc_support.rs"]
mod uc_support;

// Use the shared modules
use generator::generate_add_actions;
use writer::write_checkpoint_parquet;

#[derive(Parser)]
#[command(name = "backfill-delta-table")]
#[command(about = "Backfill a Delta table with Snapshot V2 and sidecar files")]
#[command(version)]
struct Args {
    /// Target table directory path (or Unity Catalog table name if using UC options)
    #[arg(short = 't', long)]
    table_dir: String,

    /// Unity Catalog endpoint URL (e.g., <https://uc.example.com>)
    #[arg(long)]
    uc_endpoint: Option<String>,

    /// Unity Catalog authentication token
    #[arg(long)]
    uc_token: Option<String>,

    /// Percentage of deletion vectors (0-100)
    #[arg(short = 'd', long, default_value_t = 30.0, value_parser = validate_percentage)]
    dv_percentage: f64,

    /// Random seed for reproducibility
    #[arg(short = 's', long, default_value_t = 42)]
    seed: u64,

    /// Number of sidecar files to generate
    #[arg(short = 'n', long, default_value_t = 20)]
    num_sidecars: usize,

    /// Number of actions per sidecar file
    #[arg(short = 'a', long, default_value_t = 50000)]
    actions_per_sidecar: usize,

    /// Generate content root representation
    #[arg(short = 'c', long, default_value_t = false)]
    generate_content_root: bool,

    /// Batch size for content root leaves (number of IDs per leaf).
    /// If not specified, defaults to actions_per_sidecar to align leaf partitioning with sidecars.
    #[arg(short = 'b', long)]
    batch_size: Option<usize>,

    /// Number of incremental commits to generate after the initial checkpoint
    #[arg(long, default_value_t = 0)]
    num_incremental_commits: usize,

    /// Number of add actions per incremental commit
    #[arg(long, default_value_t = 200)]
    actions_per_commit: usize,

    /// Remove all existing files before backfilling the table
    #[arg(long, default_value_t = false)]
    clean_before_backfill: bool,
}

fn validate_percentage(s: &str) -> Result<f64, String> {
    let pct: f64 = s.parse().map_err(|_| "Invalid number")?;
    if (0.0..=100.0).contains(&pct) {
        Ok(pct)
    } else {
        Err("Percentage must be between 0.0 and 100.0".to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Format {
    provider: String,
    options: HashMap<String, String>,
}

impl Default for Format {
    fn default() -> Self {
        Self {
            provider: "parquet".to_string(),
            options: HashMap::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Metadata {
    id: String,
    name: Option<String>,
    description: Option<String>,
    format: Format,
    schema_string: String,
    partition_columns: Vec<String>,
    created_time: Option<i64>,
    configuration: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Protocol {
    min_reader_version: i32,
    min_writer_version: i32,
    reader_features: Option<Vec<String>>,
    writer_features: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommitInfo {
    timestamp: i64,
    operation: String,
    operation_parameters: HashMap<String, String>,
    is_blind_append: Option<bool>,
    engine_info: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sidecar {
    path: String,
    size_in_bytes: i64,
    modification_time: i64,
    tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointMetadata {
    version: i64,
    tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LastCheckpoint {
    version: i64,
    size: i64,
    parts: Option<i32>,
    #[serde(rename = "sizeInBytes")]
    size_in_bytes: Option<i64>,
    #[serde(rename = "numOfAddFiles")]
    num_of_add_files: Option<i64>,
    #[serde(rename = "checkpointSchema")]
    checkpoint_schema: Option<serde_json::Value>,
    #[serde(rename = "checksum")]
    checksum: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("Delta Table Backfill Tool");
    println!("=========================");

    // Use the common setup function to get table location and engine
    let setup = match uc_support::setup_table_access(
        &args.table_dir,
        args.uc_endpoint.as_deref(),
        args.uc_token.as_deref(),
        Operation::ReadWrite,
    )
    .await
    {
        Ok(setup) => setup,
        Err(e) => {
            eprintln!("Failed to set up table access: {}", e);
            process::exit(1);
        }
    };

    let engine = setup.engine;

    println!("Table URL: {}", setup.table_url);
    println!("DV percentage: {}%", args.dv_percentage);
    println!("Random seed: {}", args.seed);
    println!("Number of sidecar files: {}", args.num_sidecars);
    println!("Actions per sidecar: {}", args.actions_per_sidecar);
    println!("Generate content root: {}", args.generate_content_root);

    // Default batch_size to actions_per_sidecar for aligned partitioning
    let batch_size = args.batch_size.unwrap_or(args.actions_per_sidecar);
    if args.generate_content_root {
        println!("Batch size: {}", batch_size);
    }

    println!("Incremental commits: {}", args.num_incremental_commits);
    println!("Actions per commit: {}", args.actions_per_commit);
    println!("Clean before backfill: {}", args.clean_before_backfill);
    println!();

    // Clean existing files if requested
    if args.clean_before_backfill {
        println!("Cleaning existing table files before backfill...");
        let store_for_cleanup = engine
            .get_object_store_for_url(&setup.table_url)
            .expect("Failed to get object store for URL");

        let path_prefix_for_cleanup = if setup.table_url.scheme() == "file" {
            String::new()
        } else {
            setup.path_prefix.clone()
        };

        match table_utils::remove_all_table_files(
            &store_for_cleanup,
            &setup.table_url,
            &path_prefix_for_cleanup,
        )
        .await
        {
            Ok(stats) => {
                println!("  ✓ Cleanup complete:");
                println!("    Files deleted: {}", stats.files_deleted);
                println!("    Bytes deleted: {}", stats.total_bytes_deleted);
                if !stats.errors.is_empty() {
                    println!(
                        "    WARNING: {} errors occurred during cleanup",
                        stats.errors.len()
                    );
                }
            }
            Err(e) => {
                eprintln!("  WARNING: Cleanup failed: {}", e);
                eprintln!("  Continuing with backfill anyway...");
            }
        }
        println!();
    }

    // Get the object store from the engine
    let mut store = engine
        .get_object_store_for_url(&setup.table_url)
        .expect("Failed to get object store for URL");

    // For file:// URLs, we need a prefixed store for writing
    // The engine's store is rooted at /, but we need it rooted at the table directory
    let path_prefix = if setup.table_url.scheme() == "file" {
        let path = setup
            .table_url
            .to_file_path()
            .expect("Failed to convert file:// URL to path");

        // Create the directory if it doesn't exist
        std::fs::create_dir_all(&path).expect("Failed to create table directory");

        store = std::sync::Arc::new(
            delta_kernel::object_store::local::LocalFileSystem::new_with_prefix(&path)
                .expect("Failed to create prefixed file system"),
        );
        println!(
            "Using prefixed local filesystem for writing to: {}",
            path.display()
        );
        String::new() // No prefix needed since store is already prefixed
    } else {
        // For S3/UC, use the path prefix from the setup
        if !setup.path_prefix.is_empty() {
            println!("Using path prefix for S3 writes: {}", setup.path_prefix);
        }
        setup.path_prefix
    };

    if let Err(e) = run(&args, setup.table_url, engine.clone(), store, path_prefix).await {
        eprintln!("Error: {}", e);
        process::exit(1);
    }

    println!("\n✓ Delta table backfill complete!");
}

async fn run(
    args: &Args,
    table_url: url::Url,
    engine: std::sync::Arc<dyn delta_kernel::Engine>,
    store: std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Setting up table structure...");
    // Object stores are flat - no need to create directories
    println!("   ✓ Ready to write to {}", table_url);

    // Generate commit 0
    println!("\n2. Generating commit 0 (metadata + protocol)...");
    generate_commit_0(&table_url, &store, &path_prefix, args.generate_content_root).await?;
    println!("   ✓ Written: 00000000000000000000.json");

    // Generate sidecar files
    println!("\n3. Generating {} sidecar files...", args.num_sidecars);
    let sidecars = generate_sidecars(
        &table_url,
        &store,
        &path_prefix,
        args.num_sidecars,
        args.actions_per_sidecar,
        args.dv_percentage / 100.0,
        args.seed,
    )
    .await?;
    println!("   ✓ Generated {} sidecar files", sidecars.len());

    // Generate checkpoint
    println!("\n4. Generating checkpoint at version 0...");
    generate_checkpoint(&table_url, &store, &path_prefix, &sidecars).await?;

    // Generate _last_checkpoint
    println!("\n5. Generating _last_checkpoint file...");
    let total_actions: i64 = (args.num_sidecars * args.actions_per_sidecar) as i64;
    generate_last_checkpoint(&table_url, &store, &path_prefix, &sidecars, total_actions).await?;
    println!("   ✓ Written: _last_checkpoint");

    // Generate content root if requested
    let mut current_version = 0;
    if args.generate_content_root {
        // Default batch_size to actions_per_sidecar for aligned partitioning
        let batch_size = args.batch_size.unwrap_or(args.actions_per_sidecar);
        println!("\n6. Generating content root representation...");
        generate_content_root(&table_url, &engine, batch_size).await?;
        println!("   ✓ Content root generated");
        current_version = 1; // Commit 0 (with metadataTree-experimental), Commit 1 (content root)
    }

    // Generate incremental commits
    if args.num_incremental_commits > 0 {
        let starting_id = (args.num_sidecars * args.actions_per_sidecar) as i64;
        let step_num = if args.generate_content_root { 7 } else { 6 };
        println!(
            "\n{}. Generating {} incremental commits...",
            step_num, args.num_incremental_commits
        );
        generate_incremental_commits(
            &table_url,
            &store,
            &path_prefix,
            IncrementalCommitConfig {
                num_commits: args.num_incremental_commits,
                actions_per_commit: args.actions_per_commit,
                starting_id,
                dv_probability: args.dv_percentage / 100.0,
                seed: args.seed,
                starting_version: current_version,
            },
        )
        .await?;
        println!(
            "   ✓ Generated {} incremental commits",
            args.num_incremental_commits
        );
    }

    Ok(())
}

async fn generate_commit_0(
    _table_url: &url::Url,
    store: &std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: &str,
    enable_metadata_tree: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = chrono::Utc::now().timestamp_millis();

    // Create table schema (matching the add_action_generator schema)
    // Include column mapping metadata for each field (required for future column mapping
    // enablement)
    let schema = json!({
        "type": "struct",
        "fields": [
            {"name": "phonetic", "type": "string", "nullable": true, "metadata": {
                "delta.columnMapping.id": 1,
                "delta.columnMapping.physicalName": "phonetic"
            }},
            {"name": "city", "type": "string", "nullable": true, "metadata": {
                "delta.columnMapping.id": 2,
                "delta.columnMapping.physicalName": "city"
            }},
            {"name": "state", "type": "string", "nullable": true, "metadata": {
                "delta.columnMapping.id": 3,
                "delta.columnMapping.physicalName": "state"
            }},
            {"name": "num1", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 4,
                "delta.columnMapping.physicalName": "num1"
            }},
            {"name": "num2", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 5,
                "delta.columnMapping.physicalName": "num2"
            }},
            {"name": "num3", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 6,
                "delta.columnMapping.physicalName": "num3"
            }},
            {"name": "num4", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 7,
                "delta.columnMapping.physicalName": "num4"
            }},
            {"name": "num5", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 8,
                "delta.columnMapping.physicalName": "num5"
            }},
            {"name": "num6", "type": "double", "nullable": true, "metadata": {
                "delta.columnMapping.id": 9,
                "delta.columnMapping.physicalName": "num6"
            }},
            {"name": "num7", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 10,
                "delta.columnMapping.physicalName": "num7"
            }},
            {"name": "num8", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 11,
                "delta.columnMapping.physicalName": "num8"
            }},
            {"name": "num9", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 12,
                "delta.columnMapping.physicalName": "num9"
            }},
            {"name": "num10", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 13,
                "delta.columnMapping.physicalName": "num10"
            }},
            {"name": "num11", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 14,
                "delta.columnMapping.physicalName": "num11"
            }},
            {"name": "num12", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 15,
                "delta.columnMapping.physicalName": "num12"
            }},
            {"name": "num13", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 16,
                "delta.columnMapping.physicalName": "num13"
            }},
            {"name": "num14", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 17,
                "delta.columnMapping.physicalName": "num14"
            }},
            {"name": "num15", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 18,
                "delta.columnMapping.physicalName": "num15"
            }},
            {"name": "num16", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 19,
                "delta.columnMapping.physicalName": "num16"
            }},
            {"name": "id", "type": "long", "nullable": true, "metadata": {
                "delta.columnMapping.id": 20,
                "delta.columnMapping.physicalName": "id"
            }},
        ]
    });

    // Create metadata action with column mapping enabled (Id mode)
    let mut configuration = HashMap::new();
    configuration.insert("delta.columnMapping.mode".to_string(), "id".to_string());

    let metadata = Metadata {
        id: uuid::Uuid::new_v4().to_string(),
        name: Some("benchmark_table".to_string()),
        description: Some("Backfilled Delta table with snapshot v2".to_string()),
        format: Format::default(),
        schema_string: serde_json::to_string(&schema)?,
        partition_columns: vec![],
        created_time: Some(timestamp),
        configuration,
    };

    // Create protocol action for V2 checkpoint with column mapping (reader version 3, writer
    // version 7) Optionally include metadataTree-experimental if content root will be generated
    let mut reader_features = vec!["v2Checkpoint".to_string(), "columnMapping".to_string()];
    let mut writer_features = vec!["v2Checkpoint".to_string(), "columnMapping".to_string()];
    if enable_metadata_tree {
        reader_features.push("metadataTree-experimental".to_string());
        writer_features.push("metadataTree-experimental".to_string());
    }
    let protocol = Protocol {
        min_reader_version: 3,
        min_writer_version: 7,
        reader_features: Some(reader_features),
        writer_features: Some(writer_features),
    };

    // Create commit info
    let commit_info = CommitInfo {
        timestamp,
        operation: "BACKFILL".to_string(),
        operation_parameters: HashMap::from([
            ("mode".to_string(), "backfill".to_string()),
            ("tool".to_string(), "backfill_delta_table".to_string()),
        ]),
        is_blind_append: Some(true),
        engine_info: Some("delta-kernel-rust backfill tool".to_string()),
    };

    // Write commit 0
    let mut content = String::new();
    content.push_str(&serde_json::to_string(&json!({"commitInfo": commit_info}))?);
    content.push('\n');
    content.push_str(&serde_json::to_string(&json!({"metaData": metadata}))?);
    content.push('\n');
    content.push_str(&serde_json::to_string(&json!({"protocol": protocol}))?);
    content.push('\n');

    let commit_path = delta_kernel::object_store::path::Path::from(format!(
        "{}_delta_log/00000000000000000000.json",
        path_prefix
    ));
    store.put(&commit_path, content.into()).await?;

    Ok(())
}

async fn generate_sidecars(
    _table_url: &url::Url,
    store: &std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: &str,
    num_sidecars: usize,
    actions_per_sidecar: usize,
    dv_probability: f64,
    seed: u64,
) -> Result<Vec<Sidecar>, Box<dyn std::error::Error>> {
    let mut sidecars = Vec::new();

    println!("   Generating sidecars using add_action_generator library...");

    for i in 0..num_sidecars {
        let sidecar_name = format!("sidecar-{:05}.parquet", i);

        // Calculate deterministic start for this sidecar to avoid ID overlap
        let deterministic_start = (i * actions_per_sidecar) as i64;

        // Use a different seed for each sidecar file for variety
        let file_seed = seed + (i as u64);

        println!(
            "   Generating sidecar {}/{}: {} (start_id={}, seed={})",
            i + 1,
            num_sidecars,
            sidecar_name,
            deterministic_start,
            file_seed
        );

        // Use the library directly - much cleaner and faster than subprocess!
        let actions = generate_add_actions(
            actions_per_sidecar,
            dv_probability,
            deterministic_start,
            Some(file_seed),
        );

        // Write to a temporary file first, then upload
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("sidecar-{}.parquet", uuid::Uuid::new_v4()));
        write_checkpoint_parquet(actions, temp_path.to_str().unwrap())
            .map_err(|e| format!("Failed to generate sidecar {}: {}", sidecar_name, e))?;

        // Read the file and upload to object store
        let file_bytes = tokio::fs::read(&temp_path).await?;
        let size_in_bytes = file_bytes.len() as i64;
        let modification_time = chrono::Utc::now().timestamp_millis();

        let sidecar_path = delta_kernel::object_store::path::Path::from(format!(
            "{}_delta_log/_sidecars/{}",
            path_prefix, sidecar_name
        ));
        store.put(&sidecar_path, file_bytes.into()).await?;

        // Clean up temp file
        tokio::fs::remove_file(&temp_path).await.ok();

        sidecars.push(Sidecar {
            path: sidecar_name,
            size_in_bytes,
            modification_time,
            tags: None,
        });
    }

    Ok(sidecars)
}

async fn generate_checkpoint(
    _table_url: &url::Url,
    store: &std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: &str,
    sidecars: &[Sidecar],
) -> Result<(), Box<dyn std::error::Error>> {
    // V2 checkpoints can be JSON when using UUID naming format:
    // 00000000000000000000.checkpoint.{uuid}.json
    let uuid = uuid::Uuid::new_v4().to_string();
    let checkpoint_filename = format!("00000000000000000000.checkpoint.{}.json", uuid);

    println!(
        "   Writing V2 checkpoint with {} sidecar references...",
        sidecars.len()
    );

    // Read metadata and protocol from commit 0
    let commit_0_path = delta_kernel::object_store::path::Path::from(format!(
        "{}_delta_log/00000000000000000000.json",
        path_prefix
    ));
    let commit_0_bytes = store.get(&commit_0_path).await?.bytes().await?;
    let commit_0_content = String::from_utf8(commit_0_bytes.to_vec())?;

    let mut metadata: Option<Metadata> = None;
    let mut protocol: Option<Protocol> = None;

    for line in commit_0_content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let action: serde_json::Value = serde_json::from_str(line)?;
        if let Some(m) = action.get("metaData") {
            metadata = Some(serde_json::from_value(m.clone())?);
        }
        if let Some(p) = action.get("protocol") {
            protocol = Some(serde_json::from_value(p.clone())?);
        }
    }

    let metadata = metadata.ok_or("No metadata found in commit 0")?;
    let protocol = protocol.ok_or("No protocol found in commit 0")?;

    // Create checkpoint file content
    let mut content = String::new();

    // 1. Write CheckpointMetadata action (this indicates it's a V2 checkpoint)
    let checkpoint_metadata = CheckpointMetadata {
        version: 0,
        tags: None,
    };
    content.push_str(&serde_json::to_string(
        &json!({"checkpointMetadata": checkpoint_metadata}),
    )?);
    content.push('\n');

    // 2. Write Metadata action
    content.push_str(&serde_json::to_string(&json!({"metaData": metadata}))?);
    content.push('\n');

    // 3. Write Protocol action
    content.push_str(&serde_json::to_string(&json!({"protocol": protocol}))?);
    content.push('\n');

    // 4. Write Sidecar actions
    for sidecar in sidecars {
        content.push_str(&serde_json::to_string(&json!({"sidecar": sidecar}))?);
        content.push('\n');
    }

    let checkpoint_path = delta_kernel::object_store::path::Path::from(format!(
        "{}_delta_log/{}",
        path_prefix, checkpoint_filename
    ));
    store.put(&checkpoint_path, content.into()).await?;

    println!(
        "   ✓ Written V2 checkpoint with {} actions:",
        3 + sidecars.len()
    );
    println!("     - 1 CheckpointMetadata action");
    println!("     - 1 Metadata action");
    println!("     - 1 Protocol action");
    println!("     - {} Sidecar actions", sidecars.len());
    println!();
    println!("   UUID-named V2 checkpoint: {}", checkpoint_filename);

    Ok(())
}

async fn generate_last_checkpoint(
    _table_url: &url::Url,
    store: &std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: &str,
    sidecars: &[Sidecar],
    total_actions: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate total size of all sidecars
    let total_size_in_bytes: i64 = sidecars.iter().map(|s| s.size_in_bytes).sum();

    let last_checkpoint = LastCheckpoint {
        version: 0,
        size: total_actions,
        parts: None, // Single-file checkpoint
        size_in_bytes: Some(total_size_in_bytes),
        num_of_add_files: Some(total_actions),
        checkpoint_schema: None,
        checksum: None,
    };

    let json_str = serde_json::to_string_pretty(&last_checkpoint)?;
    let last_checkpoint_path = delta_kernel::object_store::path::Path::from(format!(
        "{}_delta_log/_last_checkpoint",
        path_prefix
    ));
    store.put(&last_checkpoint_path, json_str.into()).await?;

    println!("   Total actions: {}", total_actions);
    println!("   Total size: {} bytes", total_size_in_bytes);

    Ok(())
}

struct IncrementalCommitConfig {
    num_commits: usize,
    actions_per_commit: usize,
    starting_id: i64,
    dv_probability: f64,
    seed: u64,
    starting_version: usize,
}

async fn generate_incremental_commits(
    _table_url: &url::Url,
    store: &std::sync::Arc<dyn delta_kernel::object_store::ObjectStore>,
    path_prefix: &str,
    config: IncrementalCommitConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    for commit_idx in 0..config.num_commits {
        let version = config.starting_version + commit_idx + 1;

        // Calculate the starting ID for this commit
        let commit_start_id = config.starting_id + (commit_idx * config.actions_per_commit) as i64;

        // Use a different seed for each commit for variety
        let commit_seed = config.seed + 1000 + (commit_idx as u64);

        println!(
            "   Generating commit {} (version {}) with {} actions (start_id={})...",
            commit_idx + 1,
            version,
            config.actions_per_commit,
            commit_start_id
        );

        // Generate add actions for this commit
        let actions = generate_add_actions(
            config.actions_per_commit,
            config.dv_probability,
            commit_start_id,
            Some(commit_seed),
        );

        // Build commit file content
        let mut content = String::new();

        // Write commit info
        let timestamp = chrono::Utc::now().timestamp_millis();
        let commit_info = CommitInfo {
            timestamp,
            operation: "WRITE".to_string(),
            operation_parameters: HashMap::from([
                ("mode".to_string(), "Append".to_string()),
                (
                    "numFiles".to_string(),
                    config.actions_per_commit.to_string(),
                ),
            ]),
            is_blind_append: Some(true),
            engine_info: Some("delta-kernel-rust backfill tool".to_string()),
        };
        content.push_str(&serde_json::to_string(&json!({"commitInfo": commit_info}))?);
        content.push('\n');

        // Write add actions
        for action in actions {
            // Convert stats to JSON string format
            let stats_json = json!({
                "numRecords": action.stats.num_records,
                "minValues": {
                    "phonetic": action.stats.phonetic_min,
                    "city": action.stats.city_min,
                    "state": action.stats.state_min,
                    "num1": action.stats.num1_min,
                    "num2": action.stats.num2_min,
                    "num3": action.stats.num3_min,
                    "num4": action.stats.num4_min,
                    "num5": action.stats.num5_min,
                    "num6": action.stats.num6_min,
                    "num7": action.stats.num7_min,
                    "num8": action.stats.num8_min,
                    "num9": action.stats.num9_min,
                    "num10": action.stats.num10_min,
                    "num11": action.stats.num11_min,
                    "num12": action.stats.num12_min,
                    "num13": action.stats.num13_min,
                    "num14": action.stats.num14_min,
                    "num15": action.stats.num15_min,
                    "num16": action.stats.num16_min,
                    "id": action.stats.id_value
                },
                "maxValues": {
                    "phonetic": action.stats.phonetic_max,
                    "city": action.stats.city_max,
                    "state": action.stats.state_max,
                    "num1": action.stats.num1_max,
                    "num2": action.stats.num2_max,
                    "num3": action.stats.num3_max,
                    "num4": action.stats.num4_max,
                    "num5": action.stats.num5_max,
                    "num6": action.stats.num6_max,
                    "num7": action.stats.num7_max,
                    "num8": action.stats.num8_max,
                    "num9": action.stats.num9_max,
                    "num10": action.stats.num10_max,
                    "num11": action.stats.num11_max,
                    "num12": action.stats.num12_max,
                    "num13": action.stats.num13_max,
                    "num14": action.stats.num14_max,
                    "num15": action.stats.num15_max,
                    "num16": action.stats.num16_max,
                    "id": action.stats.id_value
                },
                "nullCount": {
                    "phonetic": 0,
                    "city": 0,
                    "state": 0,
                    "num1": 0,
                    "num2": 0,
                    "num3": 0,
                    "num4": 0,
                    "num5": 0,
                    "num6": 0,
                    "num7": 0,
                    "num8": 0,
                    "num9": 0,
                    "num10": 0,
                    "num11": 0,
                    "num12": 0,
                    "num13": 0,
                    "num14": 0,
                    "num15": 0,
                    "num16": 0,
                    "id": 0
                }
            });
            let stats_string = serde_json::to_string(&stats_json)?;

            // Convert AddActionMetadata to JSON-serializable format
            let add_action = json!({
                "path": action.path,
                "size": action.size,
                "modificationTime": action.modification_time,
                "dataChange": true,
                "stats": stats_string,
                "deletionVector": action.deletion_vector.as_ref().map(|dv| {
                    json!({
                        "storageType": dv.storage_type.to_string(),
                        "pathOrInlineDv": dv.path_or_inline_dv,
                        "offset": dv.offset,
                        "sizeInBytes": dv.size_in_bytes,
                        "cardinality": dv.cardinality
                    })
                }),
                "partitionValues": {},
                "tags": null
            });

            content.push_str(&serde_json::to_string(&json!({"add": add_action}))?);
            content.push('\n');
        }

        // Write to object store
        let commit_filename = format!("{:020}.json", version);
        let commit_path = delta_kernel::object_store::path::Path::from(format!(
            "{}_delta_log/{}",
            path_prefix, commit_filename
        ));
        store.put(&commit_path, content.into()).await?;

        if (commit_idx + 1) % 5 == 0 {
            println!("      ✓ Completed {} commits", commit_idx + 1);
        }
    }

    Ok(())
}

async fn generate_content_root(
    table_url: &url::Url,
    engine: &std::sync::Arc<dyn delta_kernel::Engine>,
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use delta_kernel::committer::FileSystemCommitter;
    use delta_kernel::Snapshot;

    // Note: metadataTree-experimental feature is already enabled in commit 0
    println!("   Step 6a: Creating transaction...");

    // Open the table using the provided engine (which has correct path handling)
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;

    println!("      ✓ Opened table at version {}", snapshot.version());

    // Create transaction with manifest_commit mode
    println!("      Creating transaction...");

    let committer = Box::new(FileSystemCommitter::new());
    let mut txn = snapshot.transaction(committer, engine.as_ref())?;

    println!("      ✓ Transaction created in manifest_commit mode");

    println!("   Step 6b: Scanning existing actions...");

    // Release root and delta actions (no predicate needed for counting approach)
    let scan = {
        let mc = txn.with_manifest_commit();
        mc.release_root_and_delta_actions()?
    };

    println!("      ✓ Released root and delta actions");

    println!(
        "   Step 6c: Partitioning actions into leaves (every {} actions)...",
        batch_size
    );

    // Process the scan and partition actions into leaves
    let leaf_count = partition_actions_into_leaves(&mut txn, scan, engine.as_ref(), batch_size)?;

    println!("      ✓ Created {} leaf manifests", leaf_count);

    println!("   Step 6d: Committing transaction...");

    // Commit the transaction
    use delta_kernel::transaction::CommitResult;
    let commit_result = txn.commit(engine.as_ref())?;

    match commit_result {
        CommitResult::CommittedTransaction(committed) => {
            println!(
                "      ✓ Committed at version {}",
                committed.commit_version()
            );
        }
        CommitResult::ConflictedTransaction(_) => {
            return Err("Transaction conflicted during commit".into());
        }
        CommitResult::RetryableTransaction(_) => {
            return Err("Transaction failed with retryable error".into());
        }
    }

    Ok(())
}

fn partition_actions_into_leaves(
    txn: &mut delta_kernel::transaction::Transaction,
    scan: delta_kernel::scan::Scan,
    engine: &dyn delta_kernel::Engine,
    batch_size: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    use delta_kernel::engine::arrow_data::ArrowEngineData;

    // TODO: Use stats_parsed.minValues.id to partition actions by actual ID values for better
    // data skipping. Currently we use a simple counting approach: create a new leaf for every
    // N actions seen. If a batch would span the N-action boundary, we finish the current leaf
    // and start a new one with the entire batch (we don't split batches across leaves).

    println!("      Scanning and partitioning actions...");
    println!(
        "      Creating a new leaf for approximately every {} actions",
        batch_size
    );

    let mut current_leaf_writer: Option<delta_kernel::transaction::leaf_writer::LeafNodeWriter> =
        None;
    let mut actions_in_current_leaf: usize = 0;
    let mut leaf_count: usize = 0;

    let mc = txn.with_manifest_commit();

    // Scan metadata and count actions
    let scan_iter = scan.scan_metadata(engine)?;

    for scan_metadata_result in scan_iter {
        let scan_metadata = scan_metadata_result?;

        // Get the Arrow data to determine how many actions are in this batch
        let arrow_data = scan_metadata
            .scan_files
            .data()
            .any_ref()
            .downcast_ref::<ArrowEngineData>()
            .ok_or_else(|| delta_kernel::Error::generic("Expected ArrowEngineData"))?;
        let record_batch = arrow_data.record_batch();

        let row_count = record_batch.num_rows();
        let original_selection = scan_metadata.scan_files.selection_vector();

        // Count how many actions are actually selected in this batch
        let selected_count = if original_selection.is_empty() {
            row_count
        } else {
            original_selection.iter().filter(|&&x| x).count()
        };

        // Check if this batch would exceed the target batch size
        if actions_in_current_leaf > 0 && actions_in_current_leaf + selected_count > batch_size {
            // Finish the current leaf before adding this batch
            let leaf_result = current_leaf_writer.take().unwrap().finish(engine)?;
            mc.add_leaf(leaf_result)?;
            leaf_count += 1;
            actions_in_current_leaf = 0;
        }

        // Add this batch to the current (or new) leaf
        if current_leaf_writer.is_none() {
            current_leaf_writer = Some(mc.new_leaf_node_writer(engine)?);
        }

        let leaf_writer = current_leaf_writer.as_mut().unwrap();
        leaf_writer.add_existing_actions(engine, scan_metadata.scan_files)?;
        actions_in_current_leaf += selected_count;

        // If we've reached or exceeded batch_size, finish this leaf
        if actions_in_current_leaf >= batch_size {
            let leaf_result = current_leaf_writer.take().unwrap().finish(engine)?;
            mc.add_leaf(leaf_result)?;
            leaf_count += 1;
            actions_in_current_leaf = 0;

            if leaf_count.is_multiple_of(10) {
                println!("         Finished {} leaves...", leaf_count);
            }
        }
    }

    // Finish any remaining leaf
    if let Some(writer) = current_leaf_writer {
        mc.add_leaf(writer.finish(engine)?)?;
        leaf_count += 1;
    }

    println!("      ✓ Partitioned actions into {} leaves", leaf_count);

    Ok(leaf_count)
}
