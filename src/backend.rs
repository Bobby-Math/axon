//! Core backend abstraction trait
//!
//! All inference backends must implement the `InferenceBackend` trait,
//! providing a unified interface regardless of the underlying engine.

use crate::error::Result;
use crate::types::{InferenceRequest, InferenceResponse, ModelConfig};

/// Health status of a backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Backend is healthy and ready to serve requests
    Healthy,
    /// Backend is starting up or loading a model
    Starting,
    /// Backend is unhealthy but may recover
    Degraded,
    /// Backend has failed and cannot recover
    Failed,
}

/// Metrics reported by a backend
#[derive(Debug, Clone)]
pub struct BackendMetrics {
    /// Number of requests currently being processed
    pub pending_requests: u64,

    /// Total requests processed since startup
    pub total_requests: u64,

    /// Number of failed requests
    pub failed_requests: u64,

    /// Average tokens per second
    pub average_tps: f32,

    /// Current memory usage as percentage (0-100)
    pub memory_usage_percent: Option<f32>,

    /// GPU utilization percentage (0-100)
    pub gpu_utilization_percent: Option<f32>,
}

impl BackendMetrics {
    /// Create empty metrics
    pub fn new() -> Self {
        Self {
            pending_requests: 0,
            total_requests: 0,
            failed_requests: 0,
            average_tps: 0.0,
            memory_usage_percent: None,
            gpu_utilization_percent: None,
        }
    }
}

impl Default for BackendMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified interface for all LLM inference backends
///
/// This trait abstracts over different inference engines (vLLM, TGI, TensorRT-LLM),
/// allowing applications to switch backends via configuration without code changes.
///
/// # Lifecycle
///
/// 1. Create backend instance
/// 2. Call `load_model()` to initialize the model
/// 3. Call `infer()` to process requests
/// 4. Call `health_check()` to monitor backend status
/// 5. Call `shutdown()` to cleanly terminate
///
/// # Example
///
/// ```rust,no_run
/// use axon::{InferenceBackend, VllmBackend, ModelConfig, InferenceRequest};
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
///     prompt: "Hello, world!".to_string(),
///     ..Default::default()
/// }).await?;
///
/// backend.shutdown().await?;
/// # Ok(())
/// # }
/// ```
pub trait InferenceBackend: Send + Sync {
    /// Load a model with the given configuration
    ///
    /// This method may:
    /// - Spawn a new process (for out-of-process backends like vLLM)
    /// - Load model weights into memory
    /// - Initialize GPU resources
    /// - Take significant time for large models
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model files cannot be found
    /// - Insufficient GPU memory
    /// - Backend process fails to start
    /// - Invalid configuration
    async fn load_model(&mut self, config: ModelConfig) -> Result<()>;

    /// Run inference on a single request
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Backend is not ready (model not loaded)
    /// - Request is invalid
    /// - Backend fails during inference
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;

    /// Check if the backend is healthy and ready
    ///
    /// Returns `HealthStatus::Healthy` if the backend can serve requests.
    /// Other statuses indicate varying degrees of unhealthiness.
    async fn health_check(&self) -> HealthStatus;

    /// Get current metrics from the backend
    ///
    /// Metrics are best-effort and may not be available from all backends.
    fn metrics(&self) -> BackendMetrics;

    /// Gracefully shutdown the backend
    ///
    /// This method should:
    /// - Finish in-flight requests
    /// - Release GPU resources
    /// - Terminate any spawned processes
    /// - Clean up temporary files
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails, but attempts best-effort cleanup.
    async fn shutdown(&mut self) -> Result<()>;
}
