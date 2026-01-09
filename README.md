# Axon

**Rust Abstraction Layer for LLM Inference Engines**

Axon provides a unified, type-safe Rust interface to multiple LLM inference backends (vLLM, TGI, TensorRT-LLM). It handles process lifecycle management, health monitoring, and failure recovery—letting you focus on your application, not inference engine plumbing.

## Motivation

Running LLMs from Rust typically means choosing one inference engine and tightly coupling your codebase to it. If you later want to switch from vLLM to TGI or TensorRT-LLM, you're rewriting significant portions of your infrastructure.

Axon solves this by providing a **unified abstraction** over multiple inference engines:

- **Single API** – Use the same code regardless of backend
- **Backend swapping** – Switch engines via configuration, no code changes
- **Process management** – Spawning, health checks, graceful shutdown handled for you
- **Production-ready** – Failure recovery, retries, and observability built-in

## Supported Backends

| Backend | Status | Best For |
|---------|--------|----------|
| **vLLM** | 🚧 Target for Phase 1 | General-purpose, best open-source |
| **TGI** | 📅 Phase 3 | HuggingFace ecosystem integration |
| **TensorRT-LLM** | 📅 Phase 3 | Maximum performance on NVIDIA GPUs |
| **Custom CUDA** | 📅 Experimental | Specialized models via Synapse |

## Quick Start

```rust
use axon::{InferenceBackend, VllmBackend, ModelConfig, InferenceRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create backend - Axon handles process spawning
    let mut backend = VllmBackend::new();

    // Load model
    backend.load_model(ModelConfig {
        model_name: "meta-llama/Llama-2-7b-hf".to_string(),
        tensor_parallel_size: 1,
        ..Default::default()
    }).await?;

    // Run inference
    let response = backend.infer(InferenceRequest {
        prompt: "Explain quantum computing in one sentence.",
        max_tokens: 100,
        temperature: 0.7,
    }).await?;

    println!("{}", response.text);

    // Clean shutdown
    backend.shutdown().await?;

    Ok(())
}
```

## Use Cases

- **Distributed inference systems** – Swap backends per deployment region
- **A/B testing** – Compare inference engines without code changes
- **Multi-cloud deployments** – Use different backends on different providers
- **Spot instance orchestration** – Perfect companion for systems like [Synkti](https://github.com/bobby-math/synkti)

## Architecture

```
Your Application
    │
    ▼
┌─────────────────────────────────────────┐
│              Axon                        │
│  ┌────────────────────────────────────┐ │
│  │       InferenceBackend Trait       │ │
│  └────────────────────────────────────┘ │
│                                         │
│  ┌────────┐  ┌────────┐  ┌───────────┐ │
│  │ vLLM   │  │  TGI   │  │TensorRT-LLM│ │
│  └────────┘  └────────┘  └───────────┘ │
└─────────────────────────────────────────┘
    │            │            │
    ▼            ▼            ▼
vLLM proc     TGI proc    TensorRT proc
(Python)      (Python)      (C++/CUDA)
```

## Why Not Just Build Another Inference Server?

Building a production-grade LLM inference server means replicating years of work:
- PagedAttention (KV cache management)
- Continuous batching
- CUDA graphs and kernel fusion
- Multi-GPU tensor parallelism

Rather than competing with vLLM, Axon **embraces it**—providing the Rust-native integration layer that the ecosystem lacks.

## Status

🚧 **Early Development** – Actively transitioning to vLLM integration layer.

See [CLAUDE.md](CLAUDE.md) for detailed development roadmap.

## Used By

- [Synkti](https://github.com/bobby-math/synkti) – Distributed orchestration for spot instances

## License

Business Source License 1.1 (BSL-1.1)

Axon is licensed under the Business Source License 1.1. The license allows free use for non-production purposes and production use that doesn't compete with commercial LLM inference offerings.

On **2029-11-27** (four years from initial publication), the license automatically converts to the MIT License, making Axon fully open source.

See the [LICENSE](LICENSE) file for complete terms.
