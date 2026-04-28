use tracing::instrument;
use url::Url;

use crate::content_tree::ContentTreeNode;
use crate::path::ParsedLogPath;
use crate::{DeltaResult, Engine, ParquetCompression, ParquetWriterConfig};

/// Orchestrates the process of creating a V3 checkpoint for a table.
pub(crate) struct ContentTreeNodeWriter {
    pub(crate) metadata: ContentTreeNode,
}

/// The result of writing a content tree node to a parquet file.
pub(crate) struct ContentTreeWriteResult {
    pub(crate) location: Url,
    pub(crate) size_in_bytes: u64,
}

impl ContentTreeNodeWriter {
    /// Creates a new [`ContentTreeNodeWriter`] for given content root metadata.
    pub(crate) fn try_new(metadata: ContentTreeNode) -> DeltaResult<Self> {
        Ok(Self { metadata })
    }

    /// Returns the URL where the content metadata file should be written.
    ///
    /// This method generates the checkpoint path based on the table's content root and the version
    /// of the underlying snapshot being checkpointed. The resulting path follows the
    /// Delta checkpoint naming convention (where the version is zero-padded to 20 digits):
    ///
    /// For root manifests (no leaf UUID):
    ///   `<table_root>/_delta_log/<version>.content.parquet`
    ///
    /// For leaf manifests (with leaf UUID):
    ///   `<table_root>/_delta_log/<version>.content.<uuid>.parquet`
    ///
    /// For example, if the table root is `s3://bucket/path` and the version is `10`:
    /// - Root: `s3://bucket/path/_delta_log/00000000000000000010.content.parquet`
    /// - Leaf: `s3://bucket/path/_delta_log/00000000000000000010.content.
    ///   550e8400-e29b-41d4-a716-446655440000.parquet`
    fn checkpoint_path(&self) -> DeltaResult<Url> {
        ParsedLogPath::new_content_metadata_path(
            &self.metadata.table_root,
            self.metadata.version,
            self.metadata.leaf(),
        )
        .map(|parsed| parsed.location)
    }

    #[instrument(
        name = "content_tree.write_manifest",
        skip_all,
        fields(is_leaf = self.metadata.leaf().is_some()),
        err
    )]
    pub(crate) fn write(self, engine: &dyn Engine) -> DeltaResult<ContentTreeWriteResult> {
        let path = self.checkpoint_path()?;
        let data_iter = self.metadata.data.into_iter().map(Ok);

        let write_config = ParquetWriterConfig {
            compression: ParquetCompression::Zstd,
        };

        let size_in_bytes = {
            let _parquet_span = tracing::info_span!("content_tree.write_parquet").entered();
            engine
                .parquet_handler()
                .write_parquet_file(path.clone(), Box::new(data_iter), &write_config)?
                .size_in_bytes
        };

        Ok(ContentTreeWriteResult {
            location: path,
            size_in_bytes,
        })
    }
}
