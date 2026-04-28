//! Test-only [`Engine`] that rejects Parquet writes (for manifest explicit-root commits).

use std::sync::Arc;

use delta_kernel::engine::default::executor::TaskExecutor;
use delta_kernel::engine::default::DefaultEngine;
use delta_kernel::schema::SchemaRef;
use delta_kernel::{
    DeltaResult, Engine, EngineData, Error, EvaluationHandler, FileDataReadResultIterator,
    FileMeta, JsonHandler, MetricsReporter, ParquetFooter, ParquetHandler, ParquetWriteResult,
    ParquetWriterConfig, PredicateRef, StorageHandler,
};
use url::Url;

/// Forwards reads to the default engine but rejects any Parquet write.
pub struct ForbidParquetWritesHandler {
    inner: Arc<dyn ParquetHandler>,
    explicit_root_relative_path: String,
}

impl ParquetHandler for ForbidParquetWritesHandler {
    fn read_parquet_files(
        &self,
        files: &[FileMeta],
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<FileDataReadResultIterator> {
        self.inner
            .read_parquet_files(files, physical_schema, predicate)
    }

    fn read_parquet_file_groups(
        &self,
        file_groups: Vec<Vec<FileMeta>>,
        physical_schema: SchemaRef,
        predicate: Option<PredicateRef>,
    ) -> DeltaResult<Vec<FileDataReadResultIterator>> {
        self.inner
            .read_parquet_file_groups(file_groups, physical_schema, predicate)
    }

    fn write_parquet_file(
        &self,
        location: Url,
        _data: Box<dyn Iterator<Item = DeltaResult<Box<dyn EngineData>>> + Send>,
        _write_config: &ParquetWriterConfig,
    ) -> DeltaResult<ParquetWriteResult> {
        Err(Error::generic(format!(
            "unexpected Parquet write while committing with explicit root manifest {:?} at {}",
            self.explicit_root_relative_path, location
        )))
    }

    fn read_parquet_footer(&self, file: &FileMeta) -> DeltaResult<ParquetFooter> {
        self.inner.read_parquet_footer(file)
    }
}

/// Delegates to [`DefaultEngine`] except [`Engine::parquet_handler`] uses
/// [`ForbidParquetWritesHandler`].
pub struct ForbidParquetWritesEngine<E: TaskExecutor> {
    inner: Arc<DefaultEngine<E>>,
    parquet: Arc<ForbidParquetWritesHandler>,
}

impl<E: TaskExecutor + 'static> Engine for ForbidParquetWritesEngine<E> {
    fn evaluation_handler(&self) -> Arc<dyn EvaluationHandler> {
        self.inner.evaluation_handler()
    }

    fn storage_handler(&self) -> Arc<dyn StorageHandler> {
        self.inner.storage_handler()
    }

    fn json_handler(&self) -> Arc<dyn JsonHandler> {
        self.inner.json_handler()
    }

    fn parquet_handler(&self) -> Arc<dyn ParquetHandler> {
        self.parquet.clone()
    }

    fn get_metrics_reporter(&self) -> Option<Arc<dyn MetricsReporter>> {
        self.inner.get_metrics_reporter()
    }
}

/// Wrap `inner` so any `write_parquet_file` during commit fails with a message naming
/// `explicit_root_relative_path`.
pub fn engine_forbid_parquet_writes<E: TaskExecutor + 'static>(
    inner: Arc<DefaultEngine<E>>,
    explicit_root_relative_path: String,
) -> Arc<ForbidParquetWritesEngine<E>> {
    let parquet_inner = inner.parquet_handler();
    Arc::new(ForbidParquetWritesEngine {
        inner,
        parquet: Arc::new(ForbidParquetWritesHandler {
            inner: parquet_inner,
            explicit_root_relative_path,
        }),
    })
}
