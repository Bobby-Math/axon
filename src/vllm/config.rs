//! vLLM-specific configuration

use crate::types::ModelConfig;

/// vLLM-specific configuration derived from ModelConfig
#[derive(Debug, Clone)]
pub struct VllmConfig {
    /// Model name or path
    pub model_name: String,

    /// Host to bind to
    pub host: String,

    /// Port to bind to
    pub port: u16,

    /// Tensor parallel size (multi-GPU)
    pub tensor_parallel_size: Option<usize>,

    /// Maximum sequence length
    pub max_sequence_length: Option<usize>,

    /// Data type (auto, half, bfloat16, float32)
    pub dtype: Option<String>,
}

impl VllmConfig {
    /// Create vLLM config from generic ModelConfig
    pub fn from_model_config(config: ModelConfig) -> Self {
        Self {
            model_name: config.model_name,
            host: config.host.unwrap_or_else(|| "127.0.0.1".to_string()),
            port: config.port.unwrap_or(8000),
            tensor_parallel_size: config.tensor_parallel_size,
            max_sequence_length: config.max_sequence_length,
            dtype: config.dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_model_config() {
        let model_config = ModelConfig {
            model_name: "meta-llama/Llama-2-7b".to_string(),
            host: Some("0.0.0.0".to_string()),
            port: Some(8080),
            tensor_parallel_size: Some(2),
            max_sequence_length: Some(4096),
            dtype: Some("bfloat16".to_string()),
            ..Default::default()
        };

        let vllm_config = VllmConfig::from_model_config(model_config);

        assert_eq!(vllm_config.model_name, "meta-llama/Llama-2-7b");
        assert_eq!(vllm_config.host, "0.0.0.0");
        assert_eq!(vllm_config.port, 8080);
        assert_eq!(vllm_config.tensor_parallel_size, Some(2));
    }

    #[test]
    fn test_from_model_config_defaults() {
        let model_config = ModelConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        };

        let vllm_config = VllmConfig::from_model_config(model_config);

        assert_eq!(vllm_config.host, "127.0.0.1");
        assert_eq!(vllm_config.port, 8000);
    }
}
