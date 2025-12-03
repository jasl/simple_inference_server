# Simple Inference Server

OpenAI-compatible inference API for small/edge models. Ships ready-to-run with FastAPI + PyTorch + Hugging Face, supporting embeddings and chat (text + vision for Qwen3-VL).

## What you can do

- Serve multiple local HF models behind OpenAI-style endpoints.
- Get embeddings via `/v1/embeddings`.
- Chat via `/v1/chat/completions` (Qwen3-VL supports image inputs).
- Transcribe or translate audio via `/v1/audio/transcriptions` and `/v1/audio/translations` (Whisper-compatible).
- Observe with `/metrics`, guard with batching/backpressure, and list models with `/v1/models`.

## Requirements

- Python ≥ 3.12
- Local model weights (downloaded ahead of time)
- For FP8 models or `device_map=auto`: `accelerate` (already in deps)
- For Whisper audio: system `ffmpeg` (or torchaudio-compatible codecs) installed and on PATH
- Startup will fail hard if any requested model fails to download or load (auto-download is on by default; disable with `AUTO_DOWNLOAD_MODELS=0` if you want prefetch-only).

## TODO / Future Work

- Provide high-performance handlers per model type:
  - Whisper via faster-whisper/CT2 for lower latency.
  - Chat via vLLM/TGI/llama.cpp backends while keeping the same `ChatModel` interface.
  - Embeddings via ONNX/TensorRT (e.g., bge, gemma) to cut CPU/GPU latency.
- Chat continuous batching phase 2: streaming + user abort (phase 1 is in place; ideas recorded in `docs/continuous_batching.md`).
- Add optional remote inference handler (HTTP/gRPC) implementing the same protocols for easy swapping.
- Expand benchmarks to compare reference HF vs. high-performance variants under identical prompts/audio.

## Quick start

1) Install deps  

   ```bash
   uv sync
   ```

2) Download the models you plan to load (examples) — optional if you keep AUTO_DOWNLOAD_MODELS=1  

   ```bash
   MODELS=BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct uv run python scripts/download_models.py
   ```

   Audio (Whisper) quick demo:  

   ```bash
   MODELS=openai/whisper-tiny uv run python scripts/download_models.py
   ```

3) Run the server  

   ```bash
   MODELS=BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct uv run python scripts/run_dev.py --device auto
   ```

   Run only Whisper for ASR/translation (auto-download will fetch weights if missing):  

   ```bash
   MODELS=openai/whisper-tiny uv run python scripts/run_dev.py --device auto
   ```

4) Call the API  
   - Embedding:

     ```bash
     curl -X POST http://localhost:8000/v1/embeddings \
       -H "Content-Type: application/json" \
       -d '{"model":"BAAI/bge-m3","input":"hello world"}'
     ```

   - Chat (text only, small model):

     ```bash
     curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":128}'
     ```

   - Chat with image (Qwen3-VL):

     ```bash
     curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
             "model": "Qwen/Qwen3-VL-4B-Instruct",
             "messages": [
               {"role": "user", "content": [
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...truncated..."}},
                 {"type": "text", "text": "Describe the cat."}
               ]}
             ],
             "max_tokens": 128
           }'
     ```

   - Audio transcription (Whisper, OpenAI-compatible):

     ```bash
     curl -X POST http://localhost:8000/v1/audio/transcriptions \
       -F "model=openai/whisper-tiny" \
       -F "file=@/path/to/sample.wav" \
       -F "response_format=text"
     ```

## Configuration highlights

- `MODELS` (required): comma-separated model IDs from `configs/model_config.yaml`.
- `MODEL_DEVICE`: `cpu` | `mps` | `cuda` | `cuda:<idx>` | `auto` (default).
- `AUTO_DOWNLOAD_MODELS` (default `1`): download selected models on startup; set to `0` to require pre-downloaded weights. Startup exits on download/load failure.
- Chat generation defaults: per-model `defaults` (temperature/top_p/max_tokens) in the config; request args override.
- Chat batching (text-only): `ENABLE_CHAT_BATCHING` (default `1`), `CHAT_BATCH_WINDOW_MS` (default `10` ms), `CHAT_BATCH_MAX_SIZE` (default `8`), `CHAT_MAX_PROMPT_TOKENS` (default `4096`), `CHAT_MAX_NEW_TOKENS` (default `2048`), `CHAT_BATCH_ALLOW_VISION` (default `0` keeps vision models on legacy path).
- Vision fetch safety (Qwen3-VL): `ALLOW_REMOTE_IMAGES=0` (default), `REMOTE_IMAGE_TIMEOUT=5`, `MAX_REMOTE_IMAGE_BYTES=5242880`.
- FP8 models need `accelerate`; non-FP8 variants avoid this dependency.

## Features

- OpenAI-compatible embeddings and chat endpoints (non-streaming)
- Vision input for Qwen3-VL (optional remote image fetch, see envs below)
- Prometheus metrics, health checks, model listing
- Micro-batching for embeddings, bounded concurrency and request guards
- Offline-first: loads only local weights; HF cache under `./models` by default

## Built-in models (catalog)

All supported models are defined in `configs/model_config.yaml` (kept as the catalog). Pick which ones to load at runtime via `MODELS` / `--models`.

| id (`model` param) | HF repo | Handler |
| --- | --- | --- |
| `BAAI/bge-m3` | `BAAI/bge-m3` | `app.models.bge_m3.BgeM3Embedding` |
| `google/embeddinggemma-300m` | `google/embeddinggemma-300m` | `app.models.embedding_gemma.EmbeddingGemmaEmbedding` |
| `Qwen/Qwen3-VL-4B-Instruct-FP8` | `Qwen/Qwen3-VL-4B-Instruct-FP8` | `app.models.qwen_vl.QwenVLChat` |
| `Qwen/Qwen3-VL-2B-Instruct-FP8` | `Qwen/Qwen3-VL-2B-Instruct-FP8` | `app.models.qwen_vl.QwenVLChat` |
| `Qwen/Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Instruct` | `app.models.qwen_vl.QwenVLChat` |
| `Qwen/Qwen3-VL-2B-Instruct` | `Qwen/Qwen3-VL-2B-Instruct` | `app.models.qwen_vl.QwenVLChat` |
| `Qwen/Qwen3-4B-Instruct-2507` | `Qwen/Qwen3-4B-Instruct-2507` | `app.models.text_chat.TextChatModel` |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` | `Qwen/Qwen3-4B-Instruct-2507-FP8` | `app.models.text_chat.TextChatModel` |
| `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` | `app.models.text_chat.TextChatModel` |
| `meta-llama/Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` | `app.models.text_chat.TextChatModel` |
| `jethrowang/whisper-tiny-chinese` | `jethrowang/whisper-tiny-chinese` | `app.models.whisper.WhisperASR` |
| `Ivydata/whisper-small-japanese` | `Ivydata/whisper-small-japanese` | `app.models.whisper.WhisperASR` |
| `BELLE-2/Belle-whisper-large-v3-zh` | `BELLE-2/Belle-whisper-large-v3-zh` | `app.models.whisper.WhisperASR` |
| `whisper-large-v3-japanese-4k-steps` | `whisper-large-v3-japanese-4k-steps` | `app.models.whisper.WhisperASR` |
| `openai/whisper-large-v2` | `openai/whisper-large-v2` | `app.models.whisper.WhisperASR` |
| `openai/whisper-medium` | `openai/whisper-medium` | `app.models.whisper.WhisperASR` |
| `openai/whisper-medium.en` | `openai/whisper-medium.en` | `app.models.whisper.WhisperASR` |
| `openai/whisper-small` | `openai/whisper-small` | `app.models.whisper.WhisperASR` |
| `openai/whisper-small.en` | `openai/whisper-small.en` | `app.models.whisper.WhisperASR` |
| `openai/whisper-tiny` | `openai/whisper-tiny` | `app.models.whisper.WhisperASR` |
| `openai/whisper-tiny.en` | `openai/whisper-tiny.en` | `app.models.whisper.WhisperASR` |

FP8 Qwen3-VL models require `accelerate` (added to dependencies). If you prefer non-quantized weights, swap the repos to `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-2B-Instruct` in `configs/model_config.yaml`; all other logic remains the same.

Per-model generation defaults (temperature / top_p / max_tokens) can be set in `configs/model_config.yaml` under a `defaults` block; request parameters override these defaults.

## API endpoints

- `POST /v1/embeddings`: OpenAI-compatible embeddings API. Body:
  - `model` (string): one of the names in `configs/model_config.yaml`
  - `input` (string or array of strings): text to embed
  - `encoding_format` (optional, default `"float"`): only `"float"` is supported
- `POST /v1/chat/completions`: OpenAI-compatible chat API (non-streaming). Body:
  - `model` (string): chat-capable model id
  - `messages` (array): OpenAI messages; supports multi-modal `image_url` parts for Qwen3-VL models
  - `max_tokens` (optional; falls back to per-model default then env `MAX_NEW_TOKENS` → 512)
  - `temperature`, `top_p`, `stop`, `user` as in OpenAI; `n` must be 1; `stream` is not yet supported
  - `top_k` is intentionally unsupported for OpenAI compatibility
- `POST /v1/audio/transcriptions`: OpenAI Whisper-compatible ASR. Multipart form fields:
  - `file` (required): audio file (e.g., wav/m4a/mp3)
  - `model` (string): Whisper model id (e.g., `openai/whisper-tiny`)
  - `language` (optional): ISO code to skip auto-detect (e.g., `ja`, `zh`)
  - `prompt`, `temperature` (optional): bias decoding / adjust randomness
  - `response_format` (default `json`): `json` | `text` | `srt` | `vtt` | `verbose_json`
  - `timestamp_granularities` (optional): `segment` or `word` to include timestamps when supported
- `POST /v1/audio/translations`: same as transcriptions but forces English output; accepts `prompt`, `temperature`, `response_format`, and `timestamp_granularities`.
- `GET /v1/models`: List loaded models with id, owner, and embedding dimensions.
- `GET /health`: Liveness/readiness check; returns 503 if the registry is not ready.
- `GET /metrics`: Prometheus metrics (enabled by default; toggle via `ENABLE_METRICS`).

## Quick start (dev)

```bash
uv sync
```

```bash
MODELS=BAAI/bge-m3 uv run python scripts/run_dev.py --device auto
# MODELS must be set (comma-separated). Alternatively:
# uv run python scripts/run_dev.py --models BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct
```

Default model cache is locked to the repo-local `models/` directory. Pre-download models via `scripts/download_models.py` (always writes to `models/`) before building or running the service. For private/licensed models, set `HF_TOKEN` only when running the download script; the runtime uses local files only.

Environment variables can be kept in a `.env` file (see `.env.example`) and are loaded on startup without overriding existing variables. Startup performs an optional warmup for each model (toggle via `ENABLE_WARMUP`, default on): it runs a batch through every executor worker to initialize per-thread tokenizers and compile kernels. Control batch and repetitions with `WARMUP_BATCH_SIZE` and `WARMUP_STEPS`.

Request batching: by default the server can micro-batch concurrent embedding requests. Configure via `ENABLE_BATCHING` (default on), `BATCH_WINDOW_MS` (collection window), and `BATCH_WINDOW_MAX_SIZE` (max combined batch). Set `BATCH_WINDOW_MS=0` to effectively disable coalescing.

Embedding cache: repeated inputs are served from an in-memory LRU keyed by the full text. Control size with `EMBEDDING_CACHE_SIZE` (default `256` entries per model instance); set to `0` to disable. Prometheus counters `embedding_cache_hits_total` / `embedding_cache_misses_total` expose effectiveness per model.

## Performance tuning (quick checklist)

- **Concurrency gate**: `MAX_CONCURRENT` caps in-flight forwards and also sets threadpool size. On a single GPU/MPS start with 1–2; raise only if throughput improves while p99 stays acceptable.
- **Micro-batching**: keep `ENABLE_BATCHING=1`; tune `BATCH_WINDOW_MS` (e.g., 4–10 ms) and `BATCH_WINDOW_MAX_SIZE` (8–16) to trade a few ms of queueing for higher throughput. Set `BATCH_WINDOW_MS=0` to disable coalescing.
- **Chat batching (text-only)**: `ENABLE_CHAT_BATCHING=1` by default; tune `CHAT_BATCH_WINDOW_MS` (e.g., 4–10 ms) and `CHAT_BATCH_MAX_SIZE` (4–8). Guards: `CHAT_MAX_PROMPT_TOKENS` (default 4096) and `CHAT_MAX_NEW_TOKENS` (default 2048). Vision models stay on the legacy path unless `CHAT_BATCH_ALLOW_VISION=1`.
- **Queueing**: `MAX_QUEUE_SIZE` controls how many requests can wait. Too large increases tail latency; too small yields 429s. Set per your SLA.
- **Warmup**: keep `ENABLE_WARMUP=1`; for heavier models raise `WARMUP_STEPS` / `WARMUP_BATCH_SIZE` (e.g., 2–3 steps, batch 4–8). Restart after changing models or devices.
- **Device choice**: set `MODEL_DEVICE` (`cuda`, `cuda:<idx>`, `mps`, `cpu`). On macOS MPS, performance varies with temperature/power; on CUDA you can try `MAX_CONCURRENT=2–4`.
- **Batch/input guards**: `MAX_BATCH_SIZE` and `MAX_TEXT_CHARS` protect the API—keep them within what the model can handle.
- **Audio guard**: `MAX_AUDIO_BYTES` (default 25MB) limits upload size for Whisper endpoints.
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
        "model": "BAAI/bge-m3",
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
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 0, "total_tokens": 0, "completion_tokens": null}
}
```

Chat completion with an image (Qwen3-VL):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "messages": [
          {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...truncated..."}},
            {"type": "text", "text": "Describe this cat in detail."}
          ]}
        ],
        "max_tokens": 128
      }'
```

Basic chat (text only):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
          {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 128,
        "temperature": 0.7
      }'
```

## Benchmarking

- Embeddings: `uv run python scripts/benchmark_embeddings.py --models BAAI/bge-m3 --n-requests 50 --concurrency 8`
- Chat: `uv run python scripts/benchmark_chat.py --model-name meta-llama/Llama-3.2-1B-Instruct --prompt "Explain FP8 quantization" --n-requests 40 --concurrency 8`
- Audio (Whisper): `BASE_URL=http://localhost:8000 MODEL_NAME=openai/whisper-tiny uv run python scripts/benchmark_audio.py -- --n-requests 20 --concurrency 4`

All benchmark scripts accept environment overrides (e.g., `BASE_URL`, `MODEL_NAME`, `PROMPT`, `N_REQUESTS`, `CONCURRENCY`).

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
- OpenAI-style streaming chat completions (SSE/chunked responses)
- Request/trace IDs: include a per-request ID in logs and responses for easier tracing.

## License

MIT License. See `LICENSE` for details.
