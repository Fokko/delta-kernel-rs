//! Code to get an engine that talks to the file-system service

use std::sync::Arc;

use delta_kernel::engine::default::file_api_http_client::FilesApiHttpClient;
use delta_kernel::engine::default::DefaultEngineBuilder;
use delta_kernel::{DeltaResult, Engine};

use crate::error::{AllocateErrorFn, ExternResult, IntoExternResult};
use crate::handle::Handle;
use crate::{
    engine_to_handle, KernelStringSlice, MultithreadedExecutorConfig, SharedExternEngine,
    TryFromStringSlice,
};

/// # Safety
///
/// Caller is responsible for passing a valid string slices for all arguments, and a valid
/// allocate_error function
#[no_mangle]
pub unsafe extern "C" fn get_filesystem_service_engine(
    workspace_url: KernelStringSlice,
    user_id: KernelStringSlice,
    user_name: KernelStringSlice,
    org_id: KernelStringSlice,
    account_id: KernelStringSlice,
    bearer_token: KernelStringSlice,
    allocate_error: AllocateErrorFn,
) -> ExternResult<Handle<SharedExternEngine>> {
    get_fs_service_engine_impl(
        None,
        workspace_url,
        user_id,
        user_name,
        org_id,
        account_id,
        bearer_token,
        allocate_error,
    )
    .into_extern_result(&allocate_error)
}

/// Build a default engine that talks to the filesytem service
///
/// If `executor_config` is `Some`, uses a multi-threaded executor that owns its runtime. Otherwise,
/// uses the default single-threaded background executor.
#[allow(clippy::too_many_arguments)]
fn get_fs_service_engine_impl(
    executor_config: Option<MultithreadedExecutorConfig>,
    workspace_url: KernelStringSlice,
    user_id: KernelStringSlice,
    user_name: KernelStringSlice,
    org_id: KernelStringSlice,
    account_id: KernelStringSlice,
    bearer_token: KernelStringSlice,
    allocate_error: AllocateErrorFn,
) -> DeltaResult<Handle<SharedExternEngine>> {
    let workspace_url: &str = unsafe { TryFromStringSlice::try_from_slice(&workspace_url) }?;
    let user_id: &str = unsafe { TryFromStringSlice::try_from_slice(&user_id) }?;
    let user_name: &str = unsafe { TryFromStringSlice::try_from_slice(&user_name) }?;
    let org_id: &str = unsafe { TryFromStringSlice::try_from_slice(&org_id) }?;
    let account_id: &str = unsafe { TryFromStringSlice::try_from_slice(&account_id) }?;
    let bearer_token: &str = unsafe { TryFromStringSlice::try_from_slice(&bearer_token) }?;

    let fs_client = Arc::new(FilesApiHttpClient::try_new(
        workspace_url,
        user_id,
        user_name,
        org_id,
        account_id,
        bearer_token,
    )?);

    let engine: Arc<dyn Engine> = if let Some(config) = executor_config {
        use delta_kernel::engine::default::executor::tokio::TokioMultiThreadExecutor;

        let executor = TokioMultiThreadExecutor::new_owned_runtime(
            config.worker_threads,
            config.max_blocking_threads,
        )?;
        Arc::new(
            DefaultEngineBuilder::new(fs_client)
                .with_task_executor(Arc::new(executor))
                .build(),
        )
    } else {
        Arc::new(DefaultEngineBuilder::new(fs_client).build())
    };

    Ok(engine_to_handle(engine, allocate_error))
}
