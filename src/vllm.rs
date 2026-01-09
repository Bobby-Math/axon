//! vLLM backend implementation
//!
//! This module provides an Axon backend for vLLM, the open-source LLM serving
//! engine with PagedAttention and continuous batching.

pub mod process;
pub mod client;
pub mod config;

use crate::backend::{BackendMetrics, HealthStatus, InferenceBackend};
use crate::error::{AxonError, Result};
use crate::types::{InferenceRequest, InferenceResponse, ModelConfig};

use process::VllmProcess;
use client::VllmClient;
use config::VllmConfig;

/// vLLM backend for Axon
///
/// Spawns and manages a vLLM server process, communicating via its
/// OpenAI-compatible HTTP API.
///
/// # Example
///
/// ```rust,no_run
/// use axon::{InferenceBackend, vllm::VllmBackend, ModelConfig, InferenceRequest};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut backend = VllmBackend::new();
///
/// backend.load_model(ModelConfig {
///     model_name: "meta-llama/Llama-2-7b-hf".to_string(),
///     ..Default::default()
/// }).await?;
///
/// let response = backend.infer(InferenceRequest {
///     prompt: "Explain Rust in one sentence.".to_string(),
///     ..Default::default()
/// }).await?;
///
/// println!("{}", response.text);
/// # Ok(())
/// # }
/// ```
pub struct VllmBackend {
    /// The vLLM process (if spawned by Axon)
    process: Option<VllmProcess>,

    /// HTTP client for communicating with vLLM API
    client: Option<VllmClient>,

    /// Whether this backend spawned its own vLLM process
    owns_process: bool,

    /// Current model configuration
    current_model: Option<String>,

    /// Metrics tracker
    metrics: BackendMetrics,
}

impl VllmBackend {
    /// Create a new vLLM backend that will spawn its own process
    pub fn new() -> Self {
        Self {
            process: None,
            client: None,
            owns_process: true,
            current_model: None,
            metrics: BackendMetrics::new(),
        }
    }

    /// Create a vLLM backend that connects to an existing vLLM server
    ///
    /// Use this when vLLM is already running (e.g., in a separate container or process).
    ///
    /// # Arguments
    ///
    /// * `base_url` - The base URL of the running vLLM server (e.g., "http://localhost:8000")
    pub fn connect_to(base_url: String) -> Self {
        Self {
            process: None,
            client: Some(VllmClient::new(base_url)),
            owns_process: false,
            current_model: None,
            metrics: BackendMetrics::new(),
        }
    }

    /// Get a reference to the HTTP client
    fn client(&self) -> Option<&VllmClient> {
        self.client.as_ref()
    }

    /// Check if the process is still running
    async fn check_process(&self) -> Result<bool> {
        if let Some(process) = &self.process {
            Ok(process.is_running())
        } else {
            Ok(false)
        }
    }
}

impl Default for VllmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for VllmBackend {
    async fn load_model(&mut self, config: ModelConfig) -> Result<()> {
        // Validate configuration
        if config.model_name.is_empty() {
            return Err(AxonError::InvalidConfig("model_name cannot be empty".into()));
        }

        // If we own the process, spawn vLLM
        if self.owns_process {
            let vllm_config = VllmConfig::from_model_config(config.clone());
            let process = VllmProcess::spawn(vllm_config).await?;

            // Wait for vLLM to be ready
            process.wait_until_ready().await?;

            let base_url = format!("http://{}:{}", config.host.as_deref().unwrap_or("127.0.0.1"), config.port.unwrap_or(8000));
            self.client = Some(VllmClient::new(base_url));
            self.process = Some(process);
        }

        // Verify the server is responding
        if let Some(client) = self.client() {
            client.health_check().await?;
        }

        self.current_model = Some(config.model_name);
        Ok(())
    }

    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let client = self.client.as_ref()
            .ok_or_else(|| AxonError::BackendNotRunning)?;

        // Check process health if we own it
        if self.owns_process && !self.check_process().await? {
            return Err(AxonError::BackendNotRunning);
        }

        client.infer(request).await
    }

    async fn health_check(&self) -> HealthStatus {
        // If we own the process, check if it's running
        if self.owns_process {
            if let Some(process) = &self.process {
                if !process.is_running() {
                    return HealthStatus::Failed;
                }
            }
        }

        // Check the HTTP API
        if let Some(client) = self.client() {
            match client.health_check().await {
                Ok(_) => HealthStatus::Healthy,
                Err(_) => HealthStatus::Degraded,
            }
        } else {
            HealthStatus::Starting
        }
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.clone()
    }

    async fn shutdown(&mut self) -> Result<()> {
        // Shutdown the process if we own it
        if let Some(process) = self.process.take() {
            process.terminate().await?;
        }

        self.client = None;
        self.current_model = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_backend_new() {
        let backend = VllmBackend::new();
        assert!(backend.owns_process);
        assert!(backend.client.is_none());
    }

    #[test]
    fn test_vllm_backend_connect_to() {
        let backend = VllmBackend::connect_to("http://localhost:8000".to_string());
        assert!(!backend.owns_process);
        assert!(backend.client.is_some());
    }

    #[test]
    fn test_vllm_backend_default() {
        let backend = VllmBackend::default();
        assert!(backend.owns_process);
    }

    #[tokio::test]
    async fn test_health_check_no_client() {
        let backend = VllmBackend::new();
        assert_eq!(backend.health_check().await, HealthStatus::Starting);
    }
}
