// Tests for checkpoint action validation and optimization
use super::*;
use crate::engine::default::DefaultEngine;
use crate::object_store::memory::InMemory;
use crate::object_store::path::Path;
use crate::object_store::ObjectStore;
use crate::DeltaResult;
use std::sync::Arc;
use url::Url;

/// Shared P+M JSON suffix for checkpoint action test payloads.
const CHECKPOINT_PM_SUFFIX: &str = concat!(
    r#","protocol":{"minReaderVersion":3,"minWriterVersion":7,"#,
    r#""readerFeatures":["metadataTree-experimental"],"#,
    r#""writerFeatures":["metadataTree-experimental"]},"#,
    r#""metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"#,
    r#""schemaString":"{\"type\":\"struct\",\"fields\":[]}","#,
    r#""partitionColumns":[],"configuration":{}}"#,
);

/// Build a checkpoint action JSON string with embedded P+M.
fn checkpoint_json(version: u64, path: &str, size: u64) -> String {
    format!(
        r#"{{"checkpoint":{{"version":{version},"contentRoot":{{"path":"{path}","sizeInBytes":{size}}}{CHECKPOINT_PM_SUFFIX}}}}}"#,
    )
}

/// Helper to create an in-memory store and log root
fn new_in_memory_store() -> (Arc<InMemory>, Url) {
    let store = Arc::new(InMemory::new());
    let log_root = Url::parse("memory:///_delta_log/").unwrap();
    (store, log_root)
}

#[test]
fn test_nested_pm_overrides_when_top_level_protocol_lacks_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITHOUT MetadataTreeExperimental + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":[],"writerFeatures":[]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action with nested P+M (includes metadataTree-experimental)
    let commit1_content = checkpoint_json(1, "root.parquet", 1024);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 1,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                1,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    // Log replay processes v1 first (most recent). The checkpoint action's nested P+M
    // fill the protocol and metadata via fill_missing_pm. The v0 protocol (without feature)
    // is never reached because P+M are already populated.
    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    assert!(metadata.is_some(), "Should find metadata from nested P+M");
    let protocol = protocol.expect("Should find protocol from nested P+M");
    assert!(
        protocol.has_reader_feature(&crate::table_features::TableFeature::MetadataTreeExperimental),
        "Protocol should come from checkpoint action's nested P+M with the feature"
    );
    assert!(checkpoint_action.is_some(), "Should find checkpoint action");

    Ok(())
}

#[test]
fn test_skip_search_when_existing_protocol_lacks_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Create existing protocol without feature
    let existing_protocol =
        Protocol::try_new(3, 7, Some(Vec::<String>::new()), Some(Vec::<String>::new()))?;

    // Commit 0: Just metadata (no protocol change)
    let commit0_content = r#"{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action with valid P+M (should not be searched because
    // existing protocol lacks feature, but must have P+M for valid schema parsing)
    let commit1_content = checkpoint_json(1, "root.parquet", 1024);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 1,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                1,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    // Pass existing protocol - should skip checkpoint action search entirely
    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(
            engine.as_ref(),
            Some(&existing_protocol),
            &LazyCrc::new(None),
        )?;

    assert!(metadata.is_some(), "Should find metadata");
    assert!(protocol.is_none(), "Should not find new protocol");
    assert!(
        checkpoint_action.is_none(),
        "Should NOT search for checkpoint action when existing protocol lacks feature"
    );

    Ok(())
}

#[test]
fn test_find_checkpoint_action_when_protocol_has_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH MetadataTreeExperimental + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action with nested P+M
    let commit1_content = checkpoint_json(1, "root.parquet", 1024);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 1,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                1,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    assert!(metadata.is_some(), "Should find metadata");
    assert!(protocol.is_some(), "Should find protocol");
    assert!(
        checkpoint_action.is_some(),
        "Should find checkpoint action when protocol supports it"
    );

    let checkpoint_action = checkpoint_action.unwrap();
    assert_eq!(checkpoint_action.content_root.path, "root.parquet");
    assert_eq!(checkpoint_action.content_root.size_in_bytes, 1024);

    Ok(())
}

#[test]
fn test_early_termination_when_feature_enabled_in_later_commit() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITHOUT feature + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":[],"writerFeatures":[]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Protocol WITH feature (feature turned on)
    let commit1_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}"#;

    // Commit 2: Some other action (should not be read due to early termination)
    let commit2_content = r#"{"add":{"path":"file.parquet"}}"#;

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000002.json"),
                commit2_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 2,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    2,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                2,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    assert!(metadata.is_some(), "Should find metadata");
    assert!(protocol.is_some(), "Should find protocol");
    assert!(
        checkpoint_action.is_none(),
        "Should NOT find checkpoint action (feature just enabled, none written yet)"
    );

    // Verify we found the upgraded protocol
    let protocol = protocol.unwrap();
    assert!(
        protocol.has_reader_feature(&crate::table_features::TableFeature::MetadataTreeExperimental)
    );

    Ok(())
}

#[test]
fn test_continue_searching_when_started_optimistically() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH feature + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Some other action
    let commit1_content = r#"{"add":{"path":"file.parquet"}}"#;

    // Commit 2: Checkpoint action with nested P+M (should be found - we must keep searching)
    let commit2_content = checkpoint_json(2, "root.parquet", 1024);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000002.json"),
                commit2_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 2,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    2,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                2,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    assert!(metadata.is_some(), "Should find metadata");
    assert!(protocol.is_some(), "Should find protocol");
    assert!(
        checkpoint_action.is_some(),
        "Should find checkpoint action (must keep searching until found)"
    );

    let checkpoint_action = checkpoint_action.unwrap();
    assert_eq!(checkpoint_action.content_root.path, "root.parquet");
    assert_eq!(checkpoint_action.version, 2);

    Ok(())
}

#[test]
fn test_continue_searching_when_existing_protocol_has_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Create existing protocol WITH feature
    let existing_protocol = Protocol::try_new(
        3,
        7,
        Some(vec!["metadataTree-experimental".to_string()]),
        Some(vec!["metadataTree-experimental".to_string()]),
    )?;

    // Commit 0: Just metadata (no protocol)
    let commit0_content = r#"{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Some other action
    let commit1_content = r#"{"add":{"path":"file.parquet"}}"#;

    // Commit 2: Checkpoint action with nested P+M (should be found)
    let commit2_content = checkpoint_json(2, "root.parquet", 1024);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000002.json"),
                commit2_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 2,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    2,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                2,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(
            engine.as_ref(),
            Some(&existing_protocol),
            &LazyCrc::new(None),
        )?;

    assert!(metadata.is_some(), "Should find metadata");
    // Protocol is extracted from the checkpoint action's nested P+M via fill_missing_pm
    assert!(
        protocol.is_some(),
        "Should find protocol from checkpoint action's nested P+M"
    );
    assert!(
        checkpoint_action.is_some(),
        "Should find checkpoint action (existing protocol has feature)"
    );

    let checkpoint_action = checkpoint_action.unwrap();
    assert_eq!(checkpoint_action.content_root.path, "root.parquet");
    assert_eq!(checkpoint_action.version, 2);

    Ok(())
}

#[test]
fn test_multiple_checkpoint_actions_returns_most_recent() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH feature + Metadata (no checkpoint action yet)
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: First checkpoint action
    let commit1_content = checkpoint_json(1, "first.parquet", 1024);

    // Commit 2: Second checkpoint action (should be ignored - first one wins)
    let commit2_content = checkpoint_json(2, "second.parquet", 2048);

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000000.json"),
                commit0_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000001.json"),
                commit1_content.into(),
            )
            .await
    })
    .unwrap();

    futures::executor::block_on(async {
        store
            .put(
                &Path::from("_delta_log/00000000000000000002.json"),
                commit2_content.into(),
            )
            .await
    })
    .unwrap();

    let log_segment = LogSegment {
        end_version: 2,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root: Url::parse("memory:///").unwrap(),
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: vec![
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    0,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    1,
                ),
                ParsedLogPath::create_parsed_published_commit(
                    &Url::parse("memory:///").unwrap(),
                    2,
                ),
            ],
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: Some(ParsedLogPath::create_parsed_published_commit(
                &Url::parse("memory:///").unwrap(),
                2,
            )),
            max_published_version: None,
        },
        checkpoint_schema: None,
    };

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    assert!(metadata.is_some(), "Should find metadata");
    assert!(protocol.is_some(), "Should find protocol");
    assert!(checkpoint_action.is_some(), "Should find checkpoint action");

    // Currently returns the MOST RECENT checkpoint action (from commit 2)
    // This is because the implementation continues searching and try_new_from_data
    // returns the last one it encounters
    let checkpoint_action = checkpoint_action.unwrap();
    assert_eq!(
        checkpoint_action.content_root.path, "second.parquet",
        "Returns most recent checkpoint action"
    );
    assert_eq!(checkpoint_action.content_root.size_in_bytes, 2048);
    assert_eq!(checkpoint_action.version, 2);

    Ok(())
}

/// Helper to store a commit JSON in the in-memory store.
fn put_commit(store: &InMemory, version: u64, content: &str) {
    let payload: bytes::Bytes = content.to_owned().into();
    futures::executor::block_on(async {
        store
            .put(
                &Path::from(format!("_delta_log/{:020}.json", version)),
                payload.into(),
            )
            .await
    })
    .unwrap();
}

/// Helper to build a LogSegment for versions 0..=end_version.
fn make_log_segment(log_root: &Url, end_version: u64) -> LogSegment {
    let table_root = Url::parse("memory:///").unwrap();
    let commit_files: Vec<_> = (0..=end_version)
        .map(|v| ParsedLogPath::create_parsed_published_commit(&table_root, v))
        .collect();
    LogSegment {
        end_version,
        checkpoint_version: None,
        log_root: log_root.clone(),
        table_root,
        listed: crate::log_segment_files::LogSegmentFiles {
            ascending_commit_files: commit_files.clone(),
            ascending_compaction_files: vec![],
            checkpoint_parts: vec![],
            latest_crc_file: None,
            latest_commit_file: commit_files.last().cloned(),
            max_published_version: None,
        },
        checkpoint_schema: None,
    }
}

#[test]
fn test_nested_pm_extracted_from_manifest_commit() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // A manifest commit: checkpoint action with nested P+M, no top-level P+M
    let commit0 = concat!(
        r#"{"checkpoint":{"version":0,"contentRoot":{"path":"root.parquet","sizeInBytes":1024},"#,
        r#""protocol":{"minReaderVersion":3,"minWriterVersion":7,"#,
        r#""readerFeatures":["metadataTree-experimental"],"#,
        r#""writerFeatures":["metadataTree-experimental"]},"#,
        r#""metaData":{"id":"nested-id","format":{"provider":"parquet","options":{}},"#,
        r#""schemaString":"{\"type\":\"struct\",\"fields\":[]}","#,
        r#""partitionColumns":[],"configuration":{},"createdTime":1677811175819}}}"#,
    );

    put_commit(&store, 0, commit0);
    let log_segment = make_log_segment(&log_root, 0);

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    // P+M should be extracted from the checkpoint action's nested fields
    let metadata = metadata.expect("metadata should be extracted from nested P+M");
    assert_eq!(metadata.id(), "nested-id");

    let protocol = protocol.expect("protocol should be extracted from nested P+M");
    assert!(
        protocol.has_reader_feature(&crate::table_features::TableFeature::MetadataTreeExperimental)
    );

    // Checkpoint action should have the nested P+M populated
    let ca = checkpoint_action.expect("should find checkpoint action");
    assert_eq!(ca.content_root.path, "root.parquet");

    Ok(())
}

#[test]
fn test_top_level_pm_takes_precedence_over_nested() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Top-level protocol + metadata on separate lines, then checkpoint with different nested P+M
    let commit0 = concat!(
        r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"#,
        r#""readerFeatures":["metadataTree-experimental"],"#,
        r#""writerFeatures":["metadataTree-experimental"]}}"#,
        "\n",
        r#"{"metaData":{"id":"top-level-id","format":{"provider":"parquet","options":{}},"#,
        r#""schemaString":"{\"type\":\"struct\",\"fields\":[]}","#,
        r#""partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#,
        "\n",
        r#"{"checkpoint":{"version":0,"contentRoot":{"path":"root.parquet","sizeInBytes":1024},"#,
        r#""protocol":{"minReaderVersion":3,"minWriterVersion":7,"#,
        r#""readerFeatures":["metadataTree-experimental"],"#,
        r#""writerFeatures":["metadataTree-experimental"]},"#,
        r#""metaData":{"id":"nested-id","format":{"provider":"parquet","options":{}},"#,
        r#""schemaString":"{\"type\":\"struct\",\"fields\":[]}","#,
        r#""partitionColumns":[],"configuration":{},"createdTime":1677811175819}}}"#,
    );

    put_commit(&store, 0, commit0);
    let log_segment = make_log_segment(&log_root, 0);

    let (metadata, protocol, checkpoint_action) = log_segment
        .protocol_and_metadata_and_checkpoint(engine.as_ref(), None, &LazyCrc::new(None))?;

    // Top-level metadata should win over nested
    let metadata = metadata.expect("should find metadata");
    assert_eq!(
        metadata.id(),
        "top-level-id",
        "top-level metadata should take precedence over nested"
    );

    assert!(protocol.is_some(), "should find protocol");
    assert!(checkpoint_action.is_some(), "should find checkpoint action");

    Ok(())
}
