//! End-to-end test: create a v4 table with icebergNativeV4, write data,
//! and verify that metadata.json is automatically generated on disk.

use std::collections::HashSet;
use std::sync::Arc;

use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::schema::{DataType, StructField, StructType};
use delta_kernel::snapshot::Snapshot;
use delta_kernel::transaction::create_table::create_table;
use iceberg::spec as iceberg_spec;
use test_utils::{collect_file_paths, create_add_files_metadata};

/// Reads the latest metadata.json from the iceberg metadata directory, validates file count
/// and version prefix, and returns the parsed TableMetadata for further assertions.
fn read_and_validate_iceberg_metadata(
    iceberg_metadata_dir: &std::path::Path,
    expected_file_count: usize,
    expected_version: u64,
) -> iceberg_spec::TableMetadata {
    let mut files: Vec<_> = std::fs::read_dir(iceberg_metadata_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".metadata.json"))
        .collect();
    files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());

    assert_eq!(
        files.len(),
        expected_file_count,
        "Expected {} metadata.json files, got {}",
        expected_file_count,
        files.len()
    );

    let latest = files.last().unwrap();
    let latest_name = latest.file_name().to_string_lossy().to_string();
    let expected_prefix = format!("v{}-", expected_version);
    assert!(
        latest_name.starts_with(&expected_prefix),
        "Latest metadata.json should start with '{}', got: {}",
        expected_prefix,
        latest_name
    );

    let content = std::fs::read_to_string(latest.path()).unwrap();
    serde_json::from_str(&content).unwrap()
}

#[tokio::test]
async fn test_iceberg_metadata_json_generated_on_manifest_commit(
) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let table_path = temp_dir.path().to_str().unwrap();
    eprintln!("Table path: {}", table_path);
    let table_url = url::Url::from_directory_path(table_path).unwrap();
    let engine = test_utils::create_default_engine(&table_url)?;
    let schema = Arc::new(StructType::try_new(vec![
        StructField::new("id", DataType::INTEGER, false),
        StructField::new("name", DataType::STRING, true),
    ])?);

    // Step 1: Create the table (version 0)
    let txn = create_table(table_path, schema.clone(), "TestEngine/1.0")
        .with_table_properties([
            ("delta.columnMapping.mode", "id"),
            ("delta.feature.metadataTree-experimental", "supported"),
            ("delta.feature.domainMetadata", "supported"),
            ("delta.enableRowTracking", "true"),
            ("delta.enableIcebergNativeV4Experimental", "true"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 0);

    // Verify: CREATE TABLE should generate metadata.json (no snapshot, schema only)
    let table_dir = std::path::Path::new(&table_path);
    let iceberg_metadata_dir = table_dir.join("__iceberg").join("metadata");
    let create_metadata = read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 1, 0);
    assert_eq!(create_metadata.snapshots().len(), 0);
    assert_eq!(
        create_metadata.current_schema().as_struct().fields().len(),
        2
    );

    // Step 2: Write data via manifest commit (version 1)
    let table_url = delta_kernel::try_parse_uri(table_path)?;
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("iceberg e2e test")
        .with_data_change(true);
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![
            ("part-00000.parquet", 1024, 1_000_000, 10),
            ("part-00001.parquet", 2048, 1_000_001, 20),
        ],
    )?);

    // Commit — should automatically generate metadata.json
    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 1);
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 2, 1); // v0 + v1

    // Verify the table is readable
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    assert_eq!(snapshot.version(), 1);
    assert!(
        snapshot.checkpoint_action().is_some(),
        "Manifest commit should produce a content root"
    );

    let paths = collect_file_paths(snapshot, engine.as_ref())?;
    let expected: HashSet<String> = ["part-00000.parquet", "part-00001.parquet"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(paths, expected, "Scan should return the written files");

    // Validate metadata.json content for version 1 (with snapshot)
    let table_metadata = read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 2, 1);
    assert_eq!(
        table_metadata.format_version(),
        iceberg_spec::FormatVersion::V2
    );
    assert_eq!(
        table_metadata.current_schema().as_struct().fields().len(),
        2
    );
    assert!(table_metadata.current_snapshot_id().is_some());
    let snapshot = table_metadata.current_snapshot().unwrap();
    assert!(
        snapshot.manifest_list().contains(".content."),
        "manifest-list should point to a .content. parquet file"
    );
    assert!(table_metadata.properties().contains_key("delta-version"));

    // Verify IcebergMetadataDomain is in the Delta commit JSON
    let commit_path = table_dir
        .join("_delta_log")
        .join("00000000000000000001.json");
    let commit_content = std::fs::read_to_string(&commit_path)?;
    println!("\n=== Delta commit JSON ===");
    println!("{}", commit_content);

    // Parse commit JSON lines and find the Iceberg domainMetadata action
    let domain_action = commit_content
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|action| {
            action
                .get("domainMetadata")
                .and_then(|dm| dm.get("domain"))
                .and_then(|d| d.as_str())
                == Some("com.databricks.iceberg.metadata")
        })
        .expect("Commit should contain an Iceberg domainMetadata action");

    let dm = &domain_action["domainMetadata"];
    assert_eq!(dm["removed"], false);

    // Parse the configuration JSON and verify fields
    let config: serde_json::Value = serde_json::from_str(dm["configuration"].as_str().unwrap())?;
    println!("\n=== IcebergMetadataDomain configuration ===");
    println!("{}", serde_json::to_string_pretty(&config)?);

    assert_eq!(config["deltaCommitVersion"], 1);
    assert!(config["currentSnapshotId"].as_i64().is_some());
    assert!(!config["newSnapshotIds"].as_array().unwrap().is_empty());
    assert!(config["metadataLocation"]
        .as_str()
        .unwrap()
        .contains("metadata.json"));
    assert_eq!(
        config["icebergPartitionSpecJson"],
        r#"{"spec-id":0,"fields":[]}"#
    );
    assert_eq!(config["domainName"], "com.databricks.iceberg.metadata");

    println!("\n=== SUCCESS: Iceberg metadata.json + DomainMetadata verified ===");

    // ===================================================================
    // Step 3: Write more data (version 2) — verifies incremental metadata
    // ===================================================================
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("iceberg e2e test v2")
        .with_data_change(true);
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![("part-00002.parquet", 3072, 1_000_002, 30)],
    )?);

    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 2);
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 3, 2); // v0 + v1 + v2

    // ===================================================================
    // Step 4: Write even more data (version 3)
    // ===================================================================
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("iceberg e2e test v3")
        .with_data_change(true);
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![("part-00003.parquet", 4096, 1_000_003, 40)],
    )?);

    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 3);
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 4, 3); // v0 + v1 + v2 + v3

    // ===================================================================
    // Verify: metadata.json at version 3 should contain snapshot history
    // ===================================================================
    let mut metadata_files: Vec<_> = std::fs::read_dir(&iceberg_metadata_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".metadata.json"))
        .collect();
    // Sort by modification time to get the latest
    metadata_files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());

    let latest_metadata_path = metadata_files.last().unwrap().path();
    let latest_content = std::fs::read_to_string(&latest_metadata_path)?;
    let latest_metadata: iceberg_spec::TableMetadata = serde_json::from_str(&latest_content)?;

    println!("\n=== Latest metadata.json (version 3) ===");
    println!("{}", latest_content);

    // Should have 3 snapshots (version 1, 2, 3)
    let snapshot_count = latest_metadata.snapshots().len();
    println!("\nSnapshot count: {}", snapshot_count);
    assert_eq!(
        snapshot_count, 3,
        "After 3 commits, metadata.json should have 3 snapshots, got {}",
        snapshot_count
    );

    // TODO: Update delta-version property during incremental builds.
    // Currently the incremental builder preserves properties from the first metadata.json.
    // For now, verify the current snapshot ID is set.
    assert!(
        latest_metadata.current_snapshot_id().is_some(),
        "Latest metadata should have a current snapshot"
    );

    // Verify snapshot log has 3 entries
    let snapshot_log = latest_metadata.history();
    println!("Snapshot log entries: {}", snapshot_log.len());
    assert_eq!(snapshot_log.len(), 3, "Snapshot log should have 3 entries");

    // Verify metadata log tracks previous metadata files
    let metadata_log_count = latest_metadata.metadata_log().len();
    println!("Metadata log entries: {}", metadata_log_count);
    assert!(
        metadata_log_count >= 1,
        "Metadata log should track at least 1 previous metadata file"
    );

    // Print all files
    println!("\n=== All files after 3 commits ===");
    for entry in walkdir::WalkDir::new(table_dir)
        .min_depth(1)
        .sort_by_file_name()
        .into_iter()
        .flatten()
    {
        if entry.file_type().is_file() {
            println!(
                "  {}",
                entry.path().strip_prefix(table_dir).unwrap().display()
            );
        }
    }

    println!("\n=== SUCCESS: Snapshot history preserved across 3 commits ===");
    Ok(())
}

/// When a client (e.g. table service / IRC) provides its own IcebergMetadataDomain via
/// `with_domain_metadata`, the kernel should skip auto-generating metadata.json.
#[tokio::test]
async fn test_client_provided_iceberg_domain_skips_auto_generation(
) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let table_path = temp_dir.path().to_str().unwrap();
    let table_url = url::Url::from_directory_path(table_path).unwrap();
    let engine = test_utils::create_default_engine(&table_url)?;
    let schema = Arc::new(StructType::try_new(vec![
        StructField::new("id", DataType::INTEGER, false),
        StructField::new("name", DataType::STRING, true),
    ])?);

    // Step 1: Create the table (version 0) — kernel auto-generates metadata.json
    let txn = create_table(table_path, schema.clone(), "TestEngine/1.0")
        .with_table_properties([
            ("delta.columnMapping.mode", "id"),
            ("delta.feature.metadataTree-experimental", "supported"),
            ("delta.feature.domainMetadata", "supported"),
            ("delta.enableRowTracking", "true"),
            ("delta.enableIcebergNativeV4Experimental", "true"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

    assert_eq!(
        txn.commit(engine.as_ref())?
            .unwrap_committed()
            .commit_version(),
        0
    );

    let table_dir = std::path::Path::new(table_path);
    let iceberg_metadata_dir = table_dir.join("__iceberg").join("metadata");
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 1, 0);

    // Step 2: Write data WITH client-provided IcebergMetadataDomain
    // Simulates table service / IRC providing its own domain metadata.
    let table_url = delta_kernel::try_parse_uri(table_path)?;
    let snapshot = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("table-service-test")
        .with_data_change(true)
        .with_domain_metadata(
            "com.databricks.iceberg.metadata".to_string(),
            serde_json::json!({
                "deltaCommitVersion": 1,
                "currentSnapshotId": 9999,
                "newSnapshotIds": [9999],
                "metadataLocation": "s3://bucket/metadata/client-provided.metadata.json",
                "icebergPartitionSpecJson": "{\"spec-id\":0,\"fields\":[]}",
                "domainName": "com.databricks.iceberg.metadata"
            })
            .to_string(),
        );
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![("part-00000.parquet", 1024, 1_000_000, 10)],
    )?);

    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 1);

    // Verify: kernel should NOT have generated a new metadata.json — count stays at 1
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 1, 0);

    // Verify: the client-provided domain metadata IS in the Delta commit
    let commit_path = table_dir
        .join("_delta_log")
        .join("00000000000000000001.json");
    let commit_content = std::fs::read_to_string(&commit_path)?;
    let domain_action = commit_content
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|action| {
            action
                .get("domainMetadata")
                .and_then(|dm| dm.get("domain"))
                .and_then(|d| d.as_str())
                == Some("com.databricks.iceberg.metadata")
        })
        .expect("Commit should contain client-provided Iceberg domainMetadata");

    let config: serde_json::Value = serde_json::from_str(
        domain_action["domainMetadata"]["configuration"]
            .as_str()
            .unwrap(),
    )?;
    assert_eq!(
        config["metadataLocation"],
        "s3://bucket/metadata/client-provided.metadata.json"
    );
    assert_eq!(config["currentSnapshotId"], 9999);

    println!("\n=== SUCCESS: Client-provided domain skipped auto-generation ===");
    Ok(())
}
/// CTAS (CREATE TABLE AS SELECT): create table with data in one commit.
/// Verifies metadata.json is generated with a snapshot (not empty like pure CREATE TABLE).
#[tokio::test]
async fn test_ctas_generates_metadata_json_with_snapshot() -> Result<(), Box<dyn std::error::Error>>
{
    let temp_dir = tempfile::tempdir()?;
    let table_path = temp_dir.path().to_str().unwrap();
    let table_url = url::Url::from_directory_path(table_path).unwrap();
    let engine = test_utils::create_default_engine(&table_url)?;
    let schema = Arc::new(StructType::try_new(vec![
        StructField::new("id", DataType::INTEGER, false),
        StructField::new("value", DataType::STRING, true),
    ])?);

    // CTAS: create table + add files in one commit
    let mut txn = create_table(table_path, schema, "TestEngine/1.0")
        .with_table_properties([
            ("delta.columnMapping.mode", "id"),
            ("delta.feature.metadataTree-experimental", "supported"),
            ("delta.feature.domainMetadata", "supported"),
            ("delta.enableRowTracking", "true"),
            ("delta.enableIcebergNativeV4Experimental", "true"),
        ])
        .build(engine.as_ref(), Box::new(FileSystemCommitter::new()))?;

    // Add files to make this a CTAS
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![
            ("ctas-part-00000.parquet", 1024, 1_000_000, 100),
            ("ctas-part-00001.parquet", 2048, 1_000_001, 200),
        ],
    )?);

    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 0);

    // Verify metadata.json was generated
    let table_dir = std::path::Path::new(table_path);
    let iceberg_metadata_dir = table_dir.join("__iceberg").join("metadata");
    assert!(
        iceberg_metadata_dir.exists(),
        "CTAS should generate __iceberg/metadata/ directory"
    );

    let metadata_files: Vec<_> = std::fs::read_dir(&iceberg_metadata_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".metadata.json"))
        .collect();

    // CTAS should produce exactly 1 metadata.json (with snapshot, not the empty create-table one)
    assert_eq!(
        metadata_files.len(),
        1,
        "CTAS should produce exactly 1 metadata.json"
    );

    let filename = metadata_files[0].file_name().to_string_lossy().to_string();
    assert!(
        filename.starts_with("v0-"),
        "CTAS metadata.json should be v0-<uuid>.metadata.json, got: {}",
        filename
    );

    // Verify it has a snapshot (unlike pure CREATE TABLE which has 0)
    let content = std::fs::read_to_string(metadata_files[0].path())?;
    let table_metadata: iceberg_spec::TableMetadata = serde_json::from_str(&content)?;

    println!("\n=== CTAS metadata.json ===");
    println!("{}", content);

    assert_eq!(
        table_metadata.snapshots().len(),
        1,
        "CTAS metadata.json should have 1 snapshot"
    );
    assert!(
        table_metadata.current_snapshot_id().is_some(),
        "CTAS should have a current snapshot"
    );

    let snapshot = table_metadata.current_snapshot().unwrap();
    assert!(
        snapshot.manifest_list().contains(".content."),
        "Snapshot should point to a .content. parquet file, got: {}",
        snapshot.manifest_list()
    );

    // Verify Iceberg schema is readable and matches the Delta schema
    let iceberg_schema = table_metadata.current_schema();
    let fields = iceberg_schema.as_struct().fields();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "id");
    assert_eq!(
        *fields[0].field_type,
        iceberg_spec::Type::Primitive(iceberg_spec::PrimitiveType::Int)
    );
    assert_eq!(fields[1].name, "value");
    assert_eq!(
        *fields[1].field_type,
        iceberg_spec::Type::Primitive(iceberg_spec::PrimitiveType::String)
    );

    // Verify the table is readable
    let table_url = delta_kernel::try_parse_uri(table_path)?;
    let snapshot_view = Snapshot::builder_for(table_url).build(engine.as_ref())?;
    let paths = collect_file_paths(snapshot_view, engine.as_ref())?;
    let expected: HashSet<String> = ["ctas-part-00000.parquet", "ctas-part-00001.parquet"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(paths, expected, "CTAS files should be readable");

    // ===================================================================
    // Follow-up commits after CTAS — verify incremental metadata works
    // ===================================================================
    let table_url = delta_kernel::try_parse_uri(table_path)?;
    let iceberg_metadata_dir = table_dir.join("__iceberg").join("metadata");

    // Version 1: INSERT more data
    let snapshot_view = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot_view
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("ctas followup v1")
        .with_data_change(true);
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![("insert-part-00000.parquet", 3072, 1_000_002, 50)],
    )?);
    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 1);
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 2, 1); // v0 (CTAS) + v1

    // Version 2: INSERT more data
    let snapshot_view = Snapshot::builder_for(table_url.clone()).build(engine.as_ref())?;
    let mut txn = snapshot_view
        .transaction(Box::new(FileSystemCommitter::new()), engine.as_ref())?
        .with_engine_info("ctas followup v2")
        .with_data_change(true);
    let add_files_schema = txn.add_files_schema();
    txn.add_files(create_add_files_metadata(
        add_files_schema,
        vec![("insert-part-00001.parquet", 4096, 1_000_003, 60)],
    )?);
    let committed = txn.commit(engine.as_ref())?.unwrap_committed();
    assert_eq!(committed.commit_version(), 2);
    read_and_validate_iceberg_metadata(&iceberg_metadata_dir, 3, 2); // v0 + v1 + v2

    // Verify final metadata.json has 3 snapshots with history
    let mut final_files: Vec<_> = std::fs::read_dir(&iceberg_metadata_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".metadata.json"))
        .collect();
    final_files.sort_by_key(|e| e.metadata().unwrap().modified().unwrap());
    let final_content = std::fs::read_to_string(final_files.last().unwrap().path())?;
    let final_metadata: iceberg_spec::TableMetadata = serde_json::from_str(&final_content)?;

    assert_eq!(
        final_metadata.snapshots().len(),
        3,
        "After CTAS + 2 inserts, should have 3 snapshots"
    );
    assert_eq!(
        final_metadata.history().len(),
        3,
        "Snapshot log should have 3 entries"
    );

    // Verify parent chain: v2 -> v1 -> v0 (CTAS)
    let current = final_metadata.current_snapshot().unwrap();
    assert!(
        current.parent_snapshot_id().is_some(),
        "v2 snapshot should have a parent"
    );

    // Verify all files are readable
    let snapshot_view = Snapshot::builder_for(table_url).build(engine.as_ref())?;
    let all_paths = collect_file_paths(snapshot_view, engine.as_ref())?;
    let expected_all: HashSet<String> = [
        "ctas-part-00000.parquet",
        "ctas-part-00001.parquet",
        "insert-part-00000.parquet",
        "insert-part-00001.parquet",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(all_paths, expected_all, "All files should be readable");

    println!("\n=== SUCCESS: CTAS + 2 follow-up commits with snapshot history ===");
    Ok(())
}
