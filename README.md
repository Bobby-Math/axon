# Axon

**High-performance ML inference server in Rust**

Axon is a production-ready ML inference server optimized for GPU workloads, featuring dynamic batching, model caching, and sub-10ms p99 latency (target).

## Features

- ✅ Async Rust (Tokio) architecture
- ✅ GPU-accelerated inference via [Synapse](https://github.com/yourname/synapse)
- ✅ Dynamic batching for improved throughput (planned)
- ✅ Intelligent model caching (planned)
- ✅ Prometheus metrics and observability (planned)

## Use Cases

- **Embedded/edge deployment** - Lightweight, Rust-native
- **Safety-critical systems** - Memory safety guarantees
- **Rust-native stacks** - Zero Python overhead
- **Custom CUDA integration** - Direct control over GPU operations

## Quick Start

```rust
use axon::server::InferenceServer;

#[tokio::main]
async fn main() -> Result<(), String> {
    let server = InferenceServer::load("model.onnx").await?;
    let result = server.infer(&input_tensor).await?;
    Ok(())
}
```

## Architecture

```
Client Requests
    ↓
Axon Server (batching, caching, routing)
    ↓
Model Execution (ONNX Runtime - planned)
    ↓
Synapse (GPU operations)
    ↓
CUDA Kernels
```

## Not Competing with vLLM

Axon is designed for specialized use cases where Python-based frameworks don't fit:

- Embedded/edge deployments (lightweight)
- Safety-critical applications (memory safety)
- Rust-native infrastructure (no Python runtime)
- Custom CUDA kernel integration

**For mainstream LLM serving in Python environments, use [vLLM](https://github.com/vllm-project/vllm).**

## Status

🚧 **Early Development** - Core architecture in place. ONNX Runtime integration in progress.

## Used By

- [Tessera](https://github.com/yourname/tessera) - Distributed GPU orchestration (private)

## License

Business Source License 1.1 (BSL-1.1)

Axon is licensed under the Business Source License 1.1. The license allows free use for non-production purposes and production use that doesn't compete with commercial ML inference offerings.

On **2029-11-27** (four years from initial publication), the license automatically converts to the MIT License, making Axon fully open source.

See the [LICENSE](LICENSE) file for complete terms.
