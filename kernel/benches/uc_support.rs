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

//! Unity Catalog support utilities for benchmarking tools.
//!
//! This module provides shared functionality for interacting with Unity Catalog
//! to get table locations and credentials for S3 access.

use std::sync::Arc;
use uc_client::prelude::*;
use url::Url;

/// Configuration for Unity Catalog access
pub struct UCConfig {
    pub endpoint: String,
    pub token: String,
}

/// Result from Unity Catalog table lookup
pub struct UCTableInfo {
    pub table_id: String,
    pub table_uri: String,
}

/// Setup result containing table location and engine
pub struct TableSetup {
    /// The table URL for use with Delta Kernel
    pub table_url: Url,
    /// Engine with appropriate credentials/configuration
    pub engine: Arc<
        delta_kernel::engine::default::DefaultEngine<
            delta_kernel::engine::default::executor::tokio::TokioMultiThreadExecutor,
        >,
    >,
    /// The path prefix within the object store (e.g., "uuid1/tables/table-uuid/" for S3)
    /// Empty for local file:// URLs where the store is already prefixed
    /// Note: Only used by write operations (backfill), not read operations (benchmark runner)
    #[allow(dead_code)]
    pub path_prefix: String,
}

/// Get table information from Unity Catalog
pub async fn get_table_info(
    config: &UCConfig,
    table_name: &str,
) -> Result<UCTableInfo, Box<dyn std::error::Error + Send + Sync>> {
    let client_config = uc_client::ClientConfig::build(&config.endpoint, &config.token).build()?;
    let uc_client = UCClient::new(client_config)?;

    let res = uc_client.get_table(table_name).await?;
    let table_id = res.table_id;
    let table_uri = res.storage_location;

    println!("Unity Catalog table info:");
    println!("  Table ID: {}", table_id);
    println!("  Table URI: {}", table_uri);

    Ok(UCTableInfo {
        table_id,
        table_uri,
    })
}

/// Create an engine with Unity Catalog credentials for the specified operation
/// Returns (engine, table_url, path_prefix) where path_prefix is the path within the object store
pub async fn create_engine_with_uc_credentials(
    config: &UCConfig,
    table_info: &UCTableInfo,
    operation: Operation,
) -> Result<
    (
        Arc<
            delta_kernel::engine::default::DefaultEngine<
                delta_kernel::engine::default::executor::tokio::TokioMultiThreadExecutor,
            >,
        >,
        Url,
        String,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let client_config = uc_client::ClientConfig::build(&config.endpoint, &config.token).build()?;
    let uc_client = UCClient::new(client_config)?;

    // Get credentials for the operation
    let creds = uc_client
        .get_credentials(&table_info.table_id, operation)
        .await
        .map_err(|e| format!("Failed to get credentials: {}", e))?;

    // TODO: support non-AWS
    let creds = creds
        .aws_temp_credentials
        .ok_or("No AWS temporary credentials found")?;

    println!("Retrieved AWS temporary credentials from Unity Catalog");

    // Parse table URL and create object store with credentials
    println!("Constructing table URL from storage_location and table_id...");
    // Unity Catalog's storage_location may not include the table_id in the path.
    // We need to append the table_id to get the full table path.
    let table_uri_with_id = if table_info.table_uri.contains(&table_info.table_id) {
        // table_id is already in the path, use as-is
        table_info.table_uri.clone()
    } else {
        // Append table_id to the storage_location
        let base = table_info.table_uri.trim_end_matches('/');
        format!("{}/{}", base, table_info.table_id)
    };

    // Ensure trailing slash for proper path resolution
    let table_uri_with_slash = if table_uri_with_id.ends_with('/') {
        table_uri_with_id
    } else {
        table_uri_with_id + "/"
    };

    let table_url = Url::parse(&table_uri_with_slash)?;
    println!("Final table URL: {}", table_url);

    let options = [
        ("region", "us-west-2"),
        ("access_key_id", &creds.access_key_id),
        ("secret_access_key", &creds.secret_access_key),
        ("session_token", &creds.session_token),
    ];

    let (store, path) = delta_kernel::object_store::parse_url_opts(&table_url, options)?;

    // Use multi-threaded executor for better parallel IO performance
    let executor = Arc::new(
        delta_kernel::engine::default::executor::tokio::TokioMultiThreadExecutor::new(
            tokio::runtime::Handle::current(),
        ),
    );
    let engine = Arc::new(
        delta_kernel::engine::default::DefaultEngine::new_with_executor(store.into(), executor),
    );

    // Return both engine and table_url, plus the path prefix for writing
    // The path is what object_store strips from the URL
    // Ensure it ends with "/" for easy concatenation
    let path_prefix = if path.as_ref().is_empty() {
        String::new()
    } else {
        format!("{}/", path.as_ref())
    };

    Ok((engine, table_url, path_prefix))
}

/// Set up table access for either Unity Catalog or direct path
///
/// This function provides a unified interface for setting up table access in both
/// UC and direct path modes. It's designed to be used by both the benchmark runner
/// and backfill tools.
///
/// # Parameters
///
/// - `table_path`: The table name (for UC) or direct path
/// - `uc_endpoint`: Optional Unity Catalog endpoint URL
/// - `uc_token`: Optional Unity Catalog authentication token
/// - `operation`: The operation type (Read, Write, ReadWrite) for UC credentials
/// # Returns
///
/// A `TableSetup` struct containing:
/// - `table_url`: A parsed URL object for use with Delta Kernel
/// - `engine`: An engine configured with appropriate credentials
pub async fn setup_table_access(
    table_path: &str,
    uc_endpoint: Option<&str>,
    uc_token: Option<&str>,
    operation: Operation,
) -> Result<TableSetup, Box<dyn std::error::Error + Send + Sync>> {
    let using_uc = uc_endpoint.is_some() && uc_token.is_some();

    if using_uc {
        // Unity Catalog mode
        let uc_endpoint = uc_endpoint.unwrap();
        let uc_token = uc_token.unwrap();

        println!("Using Unity Catalog:");
        println!("  Endpoint: {}", uc_endpoint);
        println!("  Table: {}", table_path);
        println!();

        let uc_config = UCConfig {
            endpoint: uc_endpoint.to_string(),
            token: uc_token.to_string(),
        };

        // Get table info from UC
        let table_info = get_table_info(&uc_config, table_path).await?;

        println!();

        // Create engine with UC credentials
        let (engine, table_url, path_prefix) =
            create_engine_with_uc_credentials(&uc_config, &table_info, operation).await?;

        Ok(TableSetup {
            table_url,
            engine,
            path_prefix,
        })
    } else {
        // Direct path mode
        // Try to parse as URL first, if that fails, try as a local file path

        // Use multi-threaded executor for better parallel IO performance
        let executor = Arc::new(
            delta_kernel::engine::default::executor::tokio::TokioMultiThreadExecutor::new(
                tokio::runtime::Handle::current(),
            ),
        );

        let build_engine = |store, executor: Arc<_>| {
            Arc::new(
                delta_kernel::engine::default::DefaultEngine::new_with_executor(store, executor),
            )
        };

        let (table_url, engine) = if let Ok(url) = Url::parse(table_path) {
            // Valid URL - use store_from_url for all URLs
            let store = delta_kernel::engine::default::storage::store_from_url(&url)?;
            (url.clone(), build_engine(store, executor))
        } else {
            // Try to parse as a local file path
            let path = std::path::Path::new(table_path);
            let abs_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            let table_url = Url::from_directory_path(&abs_path)
                .map_err(|_| format!("Failed to parse as URL or file path: {}", table_path))?;

            // Use store_from_url for consistency
            let store = delta_kernel::engine::default::storage::store_from_url(&table_url)?;
            (table_url.clone(), build_engine(store, executor))
        };

        // For direct paths, the store is already rooted correctly, so path_prefix is empty
        Ok(TableSetup {
            table_url,
            engine,
            path_prefix: String::new(),
        })
    }
}
