// Tests for checkpoint action validation and optimization
use super::*;
use crate::engine::default::DefaultEngine;
use crate::object_store::memory::InMemory;
use crate::object_store::path::Path;
use crate::object_store::ObjectStore;
use crate::DeltaResult;
use std::sync::Arc;
use url::Url;

/// Helper to create an in-memory store and log root
fn new_in_memory_store() -> (Arc<InMemory>, Url) {
    let store = Arc::new(InMemory::new());
    let log_root = Url::parse("memory:///_delta_log/").unwrap();
    (store, log_root)
}

/// Test error when protocol lacks feature but checkpoint action exists
///
/// Scenario: Protocol without feature, then checkpoint action in later commit
/// Expected: Should error - invalid table state
#[test]
fn test_error_when_protocol_lacks_feature_but_checkpoint_action_exists() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol without MetadataTreeExperimental + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":[],"writerFeatures":[]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action (invalid - protocol doesn't support it)
    let commit1_content =
        r#"{"checkpoint":{"version":1,"contentRoot":{"path":"root.parquet","sizeInBytes":1024}}}"#;

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

    // Should error because checkpoint action exists but protocol doesn't support it
    let result = log_segment.protocol_and_metadata_and_checkpoint(
        engine.as_ref(),
        None,
        &LazyCrc::new(None),
    );

    assert!(
        result.is_err(),
        "Should error when checkpoint action exists without protocol support"
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("checkpoint action")
            && err.to_string().contains("MetadataTreeExperimental"),
        "Error should mention checkpoint action and MetadataTreeExperimental, got: {}",
        err
    );

    Ok(())
}

/// Test correctness: skip checkpoint action search when existing protocol lacks feature
///
/// Scenario: Existing protocol without feature passed in
/// Expected: Should not search for checkpoint action (feature not supported)
#[test]
fn test_skip_search_when_existing_protocol_lacks_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Create existing protocol without feature
    let existing_protocol =
        Protocol::try_new(3, 7, Some(Vec::<String>::new()), Some(Vec::<String>::new()))?;

    // Commit 0: Just metadata (no protocol change)
    let commit0_content = r#"{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action (should not be searched because existing protocol lacks feature)
    let commit1_content =
        r#"{"checkpoint":{"version":1,"contentRoot":{"path":"root.parquet","sizeInBytes":1024}}}"#;

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

/// Test happy path: find checkpoint action when protocol supports it
///
/// Scenario: Protocol with feature, checkpoint action in later commit
/// Expected: Should find metadata, protocol, and checkpoint action
#[test]
fn test_find_checkpoint_action_when_protocol_has_feature() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH MetadataTreeExperimental + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Checkpoint action
    let commit1_content =
        r#"{"checkpoint":{"version":1,"contentRoot":{"path":"root.parquet","sizeInBytes":1024}}}"#;

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

/// Test early termination when feature is enabled in later commit
///
/// Scenario: Started without searching (protocol lacks feature), then protocol
///          upgraded to add feature
/// Expected: Should terminate early once new protocol is found (feature was just
///          turned on, no checkpoint action written yet)
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

/// Test continued searching when started with searching enabled
///
/// Scenario: Started optimistically (no existing protocol), feature enabled,
///          checkpoint action in later commit
/// Expected: Should keep searching until checkpoint action is found
#[test]
fn test_continue_searching_when_started_optimistically() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH feature + Metadata
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: Some other action
    let commit1_content = r#"{"add":{"path":"file.parquet"}}"#;

    // Commit 2: Checkpoint action (should be found - we must keep searching)
    let commit2_content =
        r#"{"checkpoint":{"version":2,"contentRoot":{"path":"root.parquet","sizeInBytes":1024}}}"#;

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

/// Test continued searching when existing protocol has feature
///
/// Scenario: Existing protocol with feature, checkpoint action in later commit
/// Expected: Should search and find checkpoint action
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

    // Commit 2: Checkpoint action (should be found)
    let commit2_content =
        r#"{"checkpoint":{"version":2,"contentRoot":{"path":"root.parquet","sizeInBytes":1024}}}"#;

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
    assert!(
        protocol.is_none(),
        "Should not find new protocol (using existing)"
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

/// Test behavior when multiple checkpoint actions exist
///
/// Scenario: Multiple commits with checkpoint actions
/// Expected: Currently returns most recent one (implementation continues searching)
/// Note: This test documents current behavior - multiple checkpoint actions is unusual
#[test]
fn test_multiple_checkpoint_actions_returns_most_recent() -> DeltaResult<()> {
    let (store, log_root) = new_in_memory_store();
    let engine = Arc::new(DefaultEngine::new(store.clone()));

    // Commit 0: Protocol WITH feature + Metadata (no checkpoint action yet)
    let commit0_content = r#"{"protocol":{"minReaderVersion":3,"minWriterVersion":7,"readerFeatures":["metadataTree-experimental"],"writerFeatures":["metadataTree-experimental"]}}
{"metaData":{"id":"test-id","format":{"provider":"parquet","options":{}},"schemaString":"{\"type\":\"struct\",\"fields\":[]}","partitionColumns":[],"configuration":{},"createdTime":1677811175819}}"#;

    // Commit 1: First checkpoint action
    let commit1_content =
        r#"{"checkpoint":{"version":1,"contentRoot":{"path":"first.parquet","sizeInBytes":1024}}}"#;

    // Commit 2: Second checkpoint action (should be ignored - first one wins)
    let commit2_content = r#"{"checkpoint":{"version":2,"contentRoot":{"path":"second.parquet","sizeInBytes":2048}}}"#;

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
