//! Basic inference example using Axon with vLLM
//!
//! This example shows how to use Axon to run inference with a vLLM backend.
//! It assumes vLLM is already running on http://localhost:8000

use axon::{InferenceBackend, VllmBackend, InferenceRequest, SamplingParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Create a backend that connects to an existing vLLM server
    let backend = VllmBackend::connect_to("http://localhost:8000".to_string());

    // Or spawn a new vLLM process:
    // let mut backend = VllmBackend::new();
    // backend.load_model(ModelConfig {
    //     model_name: "meta-llama/Llama-2-7b-hf".to_string(),
    //     ..Default::default()
    // }).await?;

    println!("Backend created, running inference...");

    // Create a simple inference request
    let request = InferenceRequest {
        prompt: "Explain quantum computing in one sentence.".to_string(),
        sampling: SamplingParams {
            max_tokens: 100,
            temperature: 0.7,
            ..Default::default()
        },
        request_id: Some("example-001".to_string()),
    };

    // Run inference
    match backend.infer(request).await {
        Ok(response) => {
            println!("\nGenerated text:");
            println!("{}", response.text);
            println!("\nMetrics:");
            println!("  Tokens generated: {}", response.tokens_generated);
            println!("  Inference time: {:.2}s", response.inference_time);
            println!("  Tokens/sec: {:.1}", response.tokens_per_second);
            println!("  Finish reason: {}", response.finish_reason);
        }
        Err(e) => {
            eprintln!("Inference failed: {}", e);

            // Hint for common errors
            if e.to_string().contains("connect") {
                eprintln!("\nHint: Make sure vLLM is running on http://localhost:8000");
                eprintln!("Start vLLM with:");
                eprintln!("  python -m vllm.entrypoints.openai.api_server --model <MODEL_NAME>");
            }
        }
    }

    // Check backend health
    let health = backend.health_check().await;
    println!("\nBackend health: {:?}", health);

    // Clean shutdown (only needed if we spawned the process)
    // backend.shutdown().await?;

    Ok(())
}
