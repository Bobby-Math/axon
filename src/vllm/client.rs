//! HTTP client for vLLM's OpenAI-compatible API

use crate::error::{AxonError, Result};
use crate::types::{InferenceRequest, InferenceResponse};
use serde::{Deserialize, Serialize};

/// HTTP client for communicating with vLLM
pub struct VllmClient {
    /// Base URL of the vLLM server
    base_url: String,

    /// HTTP client
    client: reqwest::Client,
}

impl VllmClient {
    /// Create a new vLLM client
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap();

        Self { base_url, client }
    }

    /// Check if the vLLM server is healthy
    pub async fn health_check(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let resp = self.client.get(&url).send().await?;

        if resp.status().is_success() {
            Ok(())
        } else {
            Err(AxonError::Unhealthy(format!("Status: {}", resp.status())))
        }
    }

    /// Run inference on a single prompt
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let url = format!("{}/v1/completions", self.base_url);

        let vllm_req = VllmCompletionRequest {
            model: "default".to_string(),
            prompt: request.prompt.clone(),
            max_tokens: request.sampling.max_tokens,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            top_k: request.sampling.top_k,
            presence_penalty: request.sampling.presence_penalty,
            frequency_penalty: request.sampling.frequency_penalty,
            stop: if request.sampling.stop_sequences.is_empty() {
                None
            } else {
                Some(request.sampling.stop_sequences)
            },
        };

        let start = std::time::Instant::now();
        let resp = self.client.post(&url).json(&vllm_req).send().await?;
        let elapsed = start.elapsed();

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AxonError::InferenceFailed(format!("{}: {}", status, text)));
        }

        let vllm_resp: VllmCompletionResponse = resp.json().await?;

        let choice = vllm_resp.choices.first()
            .ok_or_else(|| AxonError::InferenceFailed("No choices in response".into()))?;

        Ok(InferenceResponse {
            text: choice.text.clone(),
            tokens_generated: choice.text_tokens.unwrap_or(0),
            inference_time: elapsed.as_secs_f64(),
            tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                (choice.text_tokens.unwrap_or(0) as f32) / (elapsed.as_secs_f64() as f32)
            } else {
                0.0
            },
            finish_reason: choice.finish_reason.clone(),
            request_id: request.request_id,
        })
    }
}

/// vLLM completion request format (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct VllmCompletionRequest {
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// vLLM completion response format (OpenAI-compatible)
#[derive(Debug, Deserialize)]
struct VllmCompletionResponse {
    id: String,
    choices: Vec<VllmChoice>,
    usage: VllmUsage,
}

#[derive(Debug, Deserialize)]
struct VllmChoice {
    text: String,
    #[serde(rename = "finish_reason")]
    finish_reason: String,
    #[serde(rename = "text_tokens")]
    text_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct VllmUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_client_new() {
        let client = VllmClient::new("http://localhost:8000".to_string());
        assert_eq!(client.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_completion_request_serialization() {
        let req = VllmCompletionRequest {
            model: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            presence_penalty: Some(0.0),
            frequency_penalty: Some(0.0),
            stop: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"temperature\":0.7"));
    }
}
