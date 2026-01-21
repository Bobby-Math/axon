# Axon

**Rust Abstraction Layer for LLM Inference Engines**

Axon provides a Rust interface to LLM inference backends (vLLM, TGI, TensorRT-LLM), handling process lifecycle management and health monitoring.

## Current Status: Exploratory

Axon is currently a **thin HTTP wrapper** around vLLM's OpenAI-compatible API. It spawns vLLM, monitors health, and forwards requests.

**Honest assessment:** For simple use cases (spawn vLLM, health check, shutdown), Axon may be unnecessary overhead. Direct container management is often sufficient.

**When Axon becomes valuable:** If you need request routing based on:
- Budget constraints (cap daily spend)
- Data residency / jurisdiction (EU-only routing)
- Cost/quality tradeoffs (cheap model vs expensive model)
- Heterogeneous chips (GPU, TPU, LPU selection)

The routing layer is designed but not yet built. We're waiting for real usage patterns before committing to the abstraction.

## What Axon Does

- **Process management** – Spawn inference engines, monitor health, graceful shutdown
- **Unified API** – Same interface regardless of backend (vLLM, TGI, TensorRT-LLM)
- **Backend swapping** – Switch engines via configuration

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

## Potential Use Cases

These are explored but not yet validated - waiting for real usage patterns:

- **Multi-cloud deployments** – Unified interface across AWS, GCP, Azure backends. This is the clearest justification for Axon's abstraction layer.
- **Budget-capped inference** – Enforce daily spend limits
- **Jurisdiction-aware routing** – Keep EU data in EU
- **Cost/quality routing** – Cheap model for low-stakes, expensive for high-stakes
- **Heterogeneous chip routing** – Route to GPU, TPU, or LPU based on workload
- **Cross-cloud failover** – AWS spot preempted → route to GCP while migrating

**Multi-cloud trigger:** If [Synkti](https://github.com/bobby-math/synkti) expands beyond AWS to support GCP preemptible VMs, Azure spot, etc., Axon becomes the abstraction layer that unifies inference across clouds. Until then, direct container management is simpler.

See [claude/axon-directions.md](claude/axon-directions.md) for detailed analysis.

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

🔬 **Exploratory** – Basic vLLM wrapper exists. Routing features designed but not built.

**Strategic direction:** Wait for real usage patterns before expanding. The routing layer may become a feature of [Synkti's](https://github.com/bobby-math/synkti) control plane, a client SDK, or a separate service - we don't know yet.

See [CLAUDE.md](CLAUDE.md) for strategic context.

## Used By

- [Synkti](https://github.com/bobby-math/synkti) – Distributed orchestration for spot instances

## License

Business Source License 1.1 (BSL-1.1)

Axon is licensed under the Business Source License 1.1. The license allows free use for non-production purposes and production use that doesn't compete with commercial LLM inference offerings.

On **2029-11-27** (four years from initial publication), the license automatically converts to the MIT License, making Axon fully open source.

See the [LICENSE](LICENSE) file for complete terms.
