# Simple Inference Server

A straightforward OpenAI-compatible inference API server for hosting multiple small models at the edge.
Built with FastAPI, PyTorch, and Hugging Face Transformers.

## Features

- OpenAI-compatible `POST /v1/embeddings`
- Health check `GET /health`
- Optional Prometheus metrics at `/metrics`
- Preloaded, offline models (no runtime downloads)
- In-memory LRU cache for recent embeddings (configurable)
- Bounded concurrency with 429 backpressure
- Model list `GET /v1/models` (OpenAI-compatible)
- Batch and input guards: `MAX_BATCH_SIZE` (default 32), `MAX_TEXT_CHARS` (default 20000)
- Usage accounting: prompt/total tokens when tokenizer is available

## Built-in models (catalog)

All supported models are defined in `configs/model_config.yaml` (kept as the catalog). Pick which ones to load at runtime via `MODELS` / `--models`.

| id (`model` param) | HF repo | Handler |
| --- | --- | --- |
| `bge-m3` | `BAAI/bge-m3` | `app.models.bge_m3.BgeM3Embedding` |
| `embedding-gemma-300m` | `google/embeddinggemma-300m` | `app.models.embedding_gemma.EmbeddingGemmaEmbedding` |

## API endpoints

- `POST /v1/embeddings`: OpenAI-compatible embeddings API. Body:
  - `model` (string): one of the names in `configs/model_config.yaml`
  - `input` (string or array of strings): text to embed
  - `encoding_format` (optional, default `"float"`): only `"float"` is supported
- `GET /v1/models`: List loaded models with id, owner, and embedding dimensions.
- `GET /health`: Liveness/readiness check; returns 503 if the registry is not ready.
- `GET /metrics`: Prometheus metrics (enabled by default; toggle via `ENABLE_METRICS`).

## Quick start (dev)

```bash
uv sync
```

```bash
MODELS= uv run python scripts/run_dev.py --config configs/model_config.yaml --device auto
# MODELS must be set (comma-separated). Alternatively:
# uv run python scripts/run_dev.py --models bge-m3,embedding-gemma-300m
```

Default model cache is locked to the repo-local `models/` directory. Pre-download models via `scripts/download_models.py` (always writes to `models/`) before building or running the service. For private/licensed models, set `HF_TOKEN` only when running the download script; the runtime uses local files only.

Environment variables can be kept in a `.env` file (see `.env.example`) and are loaded on startup without overriding existing variables. Startup performs an optional warmup for each model (toggle via `ENABLE_WARMUP`, default on): it runs a batch through every executor worker to initialize per-thread tokenizers and compile kernels. Control batch and repetitions with `WARMUP_BATCH_SIZE` and `WARMUP_STEPS`.

Request batching: by default the server can micro-batch concurrent embedding requests. Configure via `ENABLE_BATCHING` (default on), `BATCH_WINDOW_MS` (collection window), and `BATCH_WINDOW_MAX_SIZE` (max combined batch). Set `BATCH_WINDOW_MS=0` to effectively disable coalescing.

Embedding cache: repeated inputs are served from an in-memory LRU keyed by the full text. Control size with `EMBEDDING_CACHE_SIZE` (default `256` entries per model instance); set to `0` to disable. Prometheus counters `embedding_cache_hits_total` / `embedding_cache_misses_total` expose effectiveness per model.

## Performance tuning (quick checklist)

- **Concurrency gate**: `MAX_CONCURRENT` caps in-flight forwards and also sets threadpool size. On a single GPU/MPS start with 1–2; raise only if throughput improves while p99 stays acceptable.
- **Micro-batching**: keep `ENABLE_BATCHING=1`; tune `BATCH_WINDOW_MS` (e.g., 4–10 ms) and `BATCH_WINDOW_MAX_SIZE` (8–16) to trade a few ms of queueing for higher throughput. Set `BATCH_WINDOW_MS=0` to disable coalescing.
- **Queueing**: `MAX_QUEUE_SIZE` controls how many requests can wait. Too large increases tail latency; too small yields 429s. Set per your SLA.
- **Warmup**: keep `ENABLE_WARMUP=1`; for heavier models raise `WARMUP_STEPS` / `WARMUP_BATCH_SIZE` (e.g., 2–3 steps, batch 4–8). Restart after changing models or devices.
- **Device choice**: set `MODEL_DEVICE` (`cuda`, `cuda:<idx>`, `mps`, `cpu`). On macOS MPS, performance varies with temperature/power; on CUDA you can try `MAX_CONCURRENT=2–4`.
- **Batch/input guards**: `MAX_BATCH_SIZE` and `MAX_TEXT_CHARS` protect the API—keep them within what the model can handle.
- **Metrics & logs**: `ENABLE_METRICS` exposes `/metrics`; logs include `runtime_config`, `warmup_ok`, and `embedding_request` to observe tuning impact.

### GPU tuning (example starting points)

- **Single RTX-class GPU (e.g., RTX 6000 Ada/Pro 6000)**: `MODEL_DEVICE=cuda`, `MAX_CONCURRENT=1`, `BATCH_WINDOW_MS=4–6`, `BATCH_WINDOW_MAX_SIZE=16`, `MAX_QUEUE_SIZE=128`, `ENABLE_WARMUP=1`, `WARMUP_STEPS=2`, `WARMUP_BATCH_SIZE=8`. Increase `MAX_CONCURRENT` only if p99 stays flat.
- **Multi-GPU**: pin with `CUDA_VISIBLE_DEVICES` and run multiple replicas, one per GPU. Keep `MAX_CONCURRENT` per replica low (1–2) and rely on batching for utilization.
- **Latency-sensitive**: set `BATCH_WINDOW_MS=0` to disable coalescing; keep `MAX_CONCURRENT=1` to minimize tail.

## cURL example

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "model": "bge-m3",
        "input": ["hello", "world"]
      }'
```

Expected response shape (truncated):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [ ... ]},
    {"object": "embedding", "index": 1, "embedding": [ ... ]}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": null}
}
```

## Adding a new model

1. **Implement a handler**: Create `app/models/<your_model>.py` implementing `EmbeddingModel` (see `bge_m3.py` / `embedding_gemma.py` for reference). Set `capabilities` on the handler (e.g., `["text-embedding"]`). Use `cache_dir` pointing to `models/` (or `HF_HOME` fallback) and `local_files_only=True`.
2. **Add config entry**: Append to `configs/model_config.yaml` with fields `name`, `hf_repo_id`, and `handler` (fully qualified import path, e.g., `app.models.my_model.MyModelEmbedding`). Keep all supported models in this file; it serves as the catalog.
3. **Select models to load at runtime**: Set `MODELS` (comma-separated) or pass `--models` to `scripts/run_dev.py`. This is required; the server will exit if not provided.
4. **Pre-download weights**: Run `uv run python scripts/download_models.py` (requires `MODELS` set) to populate `models/`, then rebuild/restart the service. The download script honors `MODELS` to fetch only selected models. Startup will fail if any requested model cannot be loaded.

Unsupported or unregistered `model` values return `404 Model not found`. Be sure to add both the config entry and handler, then restart the service.

## Device selection

- Pass `--device` when starting the service (e.g., `uv run python scripts/run_dev.py --device cuda:1`).
- Supported values: `cpu`, `mps`, `cuda`, or `cuda:<index>` for multi-GPU setups.
- Default is `auto` (prefer CUDA, then MPS, otherwise CPU).
- For entrypoints that don't accept custom flags (e.g., `uvicorn app.main:app`), set environment variable `MODEL_DEVICE` instead.

## TODO

- Llama.cpp-style APIs
  - `POST /embedding`: generate embedding of a given text
  - `POST /embeddings`: non-OpenAI-compatible embeddings API
  - `POST /reranking`: rerank documents according to a given query
- Request/trace IDs: include a per-request ID in logs and responses for easier tracing.

## License

MIT License. See `LICENSE` for details.
