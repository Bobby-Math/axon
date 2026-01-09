//! # Axon
//!
//! Rust abstraction layer for LLM inference engines.
//!
//! Axon provides a unified interface to multiple inference backends
//! (vLLM, TGI, TensorRT-LLM) while handling lifecycle management,
//! health monitoring, and failure recovery.

#![warn(missing_docs)]

/// Backend abstraction trait
pub mod backend;

/// Error types for all Axon operations
pub mod error;

/// vLLM backend implementation
pub mod vllm;

/// Re-export the backend trait and common types
pub use backend::{InferenceBackend, HealthStatus, BackendMetrics};

/// Re-export vLLM backend for convenience
pub use vllm::VllmBackend;

/// Re-export error types
pub use error::{AxonError, Result};

/// Common request/response types shared across backends
pub mod types {
    /// Configuration for loading a model
    #[derive(Debug, Clone)]
    pub struct ModelConfig {
        /// Name or path of the model (e.g., "meta-llama/Llama-2-7b-hf")
        pub model_name: String,

        /// Number of GPUs for tensor parallelism
        pub tensor_parallel_size: Option<usize>,

        /// Maximum number of tokens in a batch
        pub max_batch_size: Option<usize>,

        /// Maximum sequence length
        pub max_sequence_length: Option<usize>,

        /// Whether to use half precision (FP16/BF16)
        pub dtype: Option<String>,

        /// Host to bind the inference server to
        pub host: Option<String>,

        /// Port for the inference server API
        pub port: Option<u16>,

        /// Backend-specific configuration options
        pub extra_options: Vec<(String, String)>,
    }

    impl Default for ModelConfig {
        fn default() -> Self {
            Self {
                model_name: String::new(),
                tensor_parallel_size: Some(1),
                max_batch_size: Some(256),
                max_sequence_length: Some(2048),
                dtype: Some("auto".to_string()),
                host: Some("127.0.0.1".to_string()),
                port: Some(8000),
                extra_options: Vec::new(),
            }
        }
    }

    /// A single inference request
    #[derive(Debug, Clone)]
    pub struct InferenceRequest {
        /// The input prompt(s)
        pub prompt: String,

        /// Sampling strategy
        pub sampling: SamplingParams,

        /// Optional request ID for tracing
        pub request_id: Option<String>,
    }

    impl Default for InferenceRequest {
        fn default() -> Self {
            Self {
                prompt: String::new(),
                sampling: SamplingParams::default(),
                request_id: None,
            }
        }
    }

    /// Parameters controlling generation behavior
    #[derive(Debug, Clone)]
    pub struct SamplingParams {
        /// Maximum tokens to generate
        pub max_tokens: u32,

        /// Sampling temperature (0.0 = greedy, higher = more random)
        pub temperature: f32,

        /// Nucleus sampling threshold
        pub top_p: Option<f32>,

        /// Top-k sampling
        pub top_k: Option<u32>,

        /// Presence penalty
        pub presence_penalty: Option<f32>,

        /// Frequency penalty
        pub frequency_penalty: Option<f32>,

        /// Stop sequences
        pub stop_sequences: Vec<String>,
    }

    impl Default for SamplingParams {
        fn default() -> Self {
            Self {
                max_tokens: 100,
                temperature: 1.0,
                top_p: Some(1.0),
                top_k: None,
                presence_penalty: Some(0.0),
                frequency_penalty: Some(0.0),
                stop_sequences: Vec::new(),
            }
        }
    }

    /// Response from an inference request
    #[derive(Debug, Clone)]
    pub struct InferenceResponse {
        /// Generated text
        pub text: String,

        /// Number of tokens generated
        pub tokens_generated: usize,

        /// Time taken for inference (seconds)
        pub inference_time: f64,

        /// Tokens per second
        pub tokens_per_second: f32,

        /// Finish reason ("length", "stop", or "error")
        pub finish_reason: String,

        /// Optional request ID (echoed back if provided)
        pub request_id: Option<String>,
    }

    /// Streaming inference response chunk
    #[derive(Debug, Clone)]
    pub struct InferenceChunk {
        /// Generated text delta
        pub text_delta: String,

        /// Whether this is the final chunk
        pub finished: bool,

        /// Finish reason (only if finished)
        pub finish_reason: Option<String>,
    }
}

// Re-export common types
pub use types::{InferenceRequest, InferenceResponse, InferenceChunk, ModelConfig, SamplingParams};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.tensor_parallel_size, Some(1));
        assert_eq!(config.max_batch_size, Some(256));
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.max_tokens, 100);
        assert_eq!(params.temperature, 1.0);
    }

    #[test]
    fn test_inference_request_builder_pattern() {
        let request = InferenceRequest {
            prompt: "Hello, world!".to_string(),
            sampling: SamplingParams {
                max_tokens: 50,
                temperature: 0.7,
                ..Default::default()
            },
            request_id: Some("test-123".to_string()),
        };

        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(request.sampling.max_tokens, 50);
        assert_eq!(request.request_id, Some("test-123".to_string()));
    }
}
