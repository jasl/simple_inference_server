# Simple Inference Server

All-in-one OpenAI-compatible inference server for edge deployment.
Run embeddings, chat, vision, and audio models together on a single device.

Built for edge AI scenarios where you need multiple models running concurrently with low latency—think AI-powered NAS, local RAG pipelines, or self-hosted inference for privacy-sensitive applications.

## Why This Project

- **Multi-model, single device**: Run embeddings, chat, vision (Qwen3-VL), and audio (Whisper) models simultaneously without juggling multiple services
- **Edge-optimized**: Designed for <4B parameter models with smart batching and GPU resource management for high throughput on limited hardware
- **OpenAI-compatible**: Drop-in replacement for OpenAI APIs—your existing code just works
- **Unified gateway**: Proxy to upstream services (OpenAI, vLLM) through the same endpoint for models that need more compute

## Features

| Capability | Endpoints | Models |
|------------|-----------|--------|
| **Embeddings** | `POST /v1/embeddings` | BGE-M3, Qwen3-Embedding, Gemma |
| **Chat** | `POST /v1/chat/completions` | Qwen3, Llama 3.2 |
| **Vision** | `POST /v1/chat/completions` | Qwen3-VL (image inputs) |
| **Audio** | `POST /v1/audio/transcriptions`, `/translations` | Whisper variants |
| **Rerank** | `POST /v1/rerank` | Custom rerank models |

**Production-ready features:**
- Micro-batching for embeddings and chat (configurable windows)
- Per-capability concurrency limits and queue management
- LRU embedding cache, warmup on startup
- Prometheus metrics at `/metrics`
- Request tracing via `X-Request-ID`

## Quick Start

**Requirements:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install dependencies
uv sync

# 2. Run the server (models download automatically)
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 uv run python scripts/run_dev.py

# Or with Whisper for audio:
MODELS=BAAI/bge-m3,openai/whisper-tiny uv run python scripts/run_dev.py
```

**Test it:**

```bash
# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":"hello world"}'

# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"Hi!"}],"max_tokens":64}'

# Audio transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=openai/whisper-tiny" \
  -F "file=@sample.wav"
```

## Configuration Essentials

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | *(required)* | Comma-separated model IDs to load |
| `MODEL_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:<idx>`, `mps`, or `auto` |
| `MAX_CONCURRENT` | `4` | Max concurrent model forward passes |
| `MAX_QUEUE_SIZE` | `64` | Request queue capacity |

Copy `env` to `.env` for local configuration. See [Configuration Reference](docs/configuration.md) for all options.

## Documentation

| Document | Description |
|----------|-------------|
| [Configuration Reference](docs/configuration.md) | All environment variables and settings |
| [API Reference](docs/api-reference.md) | Endpoint documentation with examples |
| [Models Guide](docs/models.md) | Model catalog and how to add custom models |
| [Performance Tuning](docs/perf_testing.md) | Tuning checklist, load testing, and monitoring |
| [Upstream Proxy](docs/upstream_proxy.md) | Forward requests to OpenAI/vLLM |

## Benchmarking

```bash
# Embeddings
uv run python scripts/benchmark_embeddings.py --models BAAI/bge-m3 --n-requests 50 --concurrency 8

# Chat
uv run python scripts/benchmark_chat.py --model-name Qwen/Qwen3-4B-Instruct-2507 --n-requests 40 --concurrency 8

# Audio
MODEL_NAME=openai/whisper-tiny uv run python scripts/benchmark_audio.py -- --n-requests 20 --concurrency 4
```

## License

MIT License. See [LICENSE](LICENSE) for details.
