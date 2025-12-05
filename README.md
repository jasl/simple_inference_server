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
- Optional non-PyTorch backends for embeddings / intent / rerank models (e.g., ONNX or C++ runtimes) that plug into the existing `EmbeddingModel`-style handlers without adding hard dependencies to the core server.
- Chat continuous batching phase 2: streaming + user abort (phase 1 is in place; ideas recorded in `docs/continuous_batching.md`).
- Deepen cancellation support: propagate cancel signals into backend kernels (e.g., vLLM/TGI/llama.cpp/faster-whisper) once their APIs expose cooperative interrupts.
- Add optional remote inference handler (HTTP/gRPC) implementing the same protocols for easy swapping.
- Expand benchmarks to compare reference HF vs. high-performance variants under identical prompts/audio.
- Keep default handlers dependency-light, but add TODO hooks for swapping in faster-whisper/CT2 and ONNX/TensorRT embeddings as optional backends.

## Quick start

1) Install deps  

   ```bash
   uv sync
   ```

2) Download the models you plan to load (examples) — optional if you keep AUTO_DOWNLOAD_MODELS=1  

   ```bash
   MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 uv run python scripts/download_models.py
   ```

   Audio (Whisper) quick demo:  

   ```bash
   MODELS=openai/whisper-tiny uv run python scripts/download_models.py
   ```

3) Run the server  

   ```bash
   MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 uv run python scripts/run_dev.py --device auto
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
       -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":128}'
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
   - Manual smoke (embeddings/chat/audio):

     ```bash
     uv run python scripts/manual_smoke.py \
       --base-url http://localhost:8000 \
       --embed-model BAAI/bge-m3 \
       --chat-model Qwen/Qwen3-4B-Instruct-2507 \
       --audio-model openai/whisper-tiny
     ```

## Configuration highlights

- `MODELS` (required): comma-separated model IDs from `configs/model_config.yaml`.
- `MODEL_DEVICE`: `cpu` | `mps` | `cuda` | `cuda:<idx>` | `auto` (default).
- `AUTO_DOWNLOAD_MODELS` (default `1`): download selected models on startup; set to `0` to require pre-downloaded weights. Startup exits on download/load failure.
- **Warmup behavior**: `ENABLE_WARMUP` (default `1`) enables a multi-capability warmup pass across embeddings / chat / vision / audio. When warmup is enabled the server **always fails fast** if any model warmup fails; use `WARMUP_ALLOWLIST` / `WARMUP_SKIPLIST` to scope coverage instead of turning off fail-fast.
- Chat generation defaults: per-model `defaults` (temperature/top_p/max_tokens) in the config; request args override.
- Embedding batching queue: `EMBEDDING_BATCH_QUEUE_SIZE` (default `64`, falls back to `MAX_QUEUE_SIZE`) bounds the per-model micro-batch queue to prevent unbounded RAM growth.
- Audio path isolation: `AUDIO_MAX_CONCURRENT` / `AUDIO_MAX_QUEUE_SIZE` / `AUDIO_QUEUE_TIMEOUT_SEC` plus `AUDIO_MAX_WORKERS` size a dedicated limiter + thread pool for Whisper so it will not block chat/embedding traffic; default worker count is **1** and Whisper currently serializes work with a lock, so bumping workers only helps if you fork per-worker pipelines.
  - Handlers must be thread-safe if `AUDIO_MAX_WORKERS` > 1; otherwise wrap shared state with locks (Whisper handler already guards its pipeline but still serializes by default).
- Request timeouts: `EMBEDDING_GENERATE_TIMEOUT_SEC` (default `60`) and `AUDIO_PROCESS_TIMEOUT_SEC` (default `180`) bound executor work to keep queues from clogging under hung models.
- **Cancellation and status codes** (best-effort): for embeddings, chat, and audio we race executor work against client disconnect and a hard timeout. Timeouts surface as `504 Gateway Timeout`; client-side aborts (disconnect or cancel) surface as `499 Client Closed Request`. Model kernels are not preempted—timeouts/cancellations only cut off responses and free queue capacity—so keep `MAX_CONCURRENT` / timeouts conservative.
- Chat batching (text-only): `ENABLE_CHAT_BATCHING` (default `1`), `CHAT_BATCH_WINDOW_MS` (default `10` ms), `CHAT_BATCH_MAX_SIZE` (default `8`), `CHAT_BATCH_QUEUE_SIZE` (default `64`), `CHAT_MAX_PROMPT_TOKENS` (default `4096`), `CHAT_MAX_NEW_TOKENS` (default `2048`), `CHAT_BATCH_ALLOW_VISION` (default `0` keeps vision models on an unbatched path).
- Chat scheduler tuning: `CHAT_COUNT_MAX_WORKERS` (token-count threads, default `2`), `CHAT_REQUEUE_RETRIES` (default `3`) and `CHAT_REQUEUE_BASE_DELAY_MS` (default `5`) control requeue backoff to降低峰值 429。
- Chat cancellation: the chat path now shares the same cancellation helper as embeddings/audio, so `499` / `504` behavior and metrics are aligned across all three capabilities. As with other paths, cancellation is cooperative at the application level only.
- Embedding usage token counting: by default the embeddings endpoint performs a second tokenizer pass per request to populate `usage.prompt_tokens` in the OpenAI-style response. Set `EMBEDDING_USAGE_DISABLE_TOKEN_COUNT=1` to skip this work (usage fields will report `prompt_tokens=0`) in high-QPS scenarios where you do not need per-request token accounting.
- Vision fetch safety (Qwen3-VL): `ALLOW_REMOTE_IMAGES=0` (default), `REMOTE_IMAGE_TIMEOUT=5`, `MAX_REMOTE_IMAGE_BYTES=5242880`. Remote HTTP fetch stays **disabled by default** to avoid SSRF/large downloads; enable only with trusted sources **and** set `REMOTE_IMAGE_HOST_ALLOWLIST` (comma-separated domains) or the request will be rejected. Private/loopback IPs are blocked even if allowlisted.
  - When the HTTP client for remote images is first created, its `timeout` and connection limits are logged so misconfigurations are visible in logs.
- TODO: add bandwidth/throughput metrics for remote image fetch when enabled.
- FP8 models need `accelerate`; non-FP8 variants avoid this dependency.

## Features

- OpenAI-compatible embeddings and chat endpoints (non-streaming)
- Vision input for Qwen3-VL (optional remote image fetch, see envs below)
- Prometheus metrics, health checks, model listing
- Micro-batching for embeddings, bounded concurrency and request guards
- Offline-first: loads only local weights; HF cache under `./models` by default
- Queue wait histograms for embeddings/chat/audio plus batch wait metrics to tune backpressure.

## Built-in models (catalog)

All supported models are defined in `configs/model_config.yaml` (kept as the catalog). Pick which ones to load at runtime via `MODELS` / `--models`.

| id (`model` param) | HF repo | Handler |
| --- | --- | --- |
| `BAAI/bge-m3` | `BAAI/bge-m3` | `app.models.hf_embedding.HFEmbeddingModel` |
| `google/embeddinggemma-300m` | `google/embeddinggemma-300m` | `app.models.hf_embedding.HFEmbeddingModel` |
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
  - Includes warmup status per model and capability so operators can spot partial warmups.
- `GET /metrics`: Prometheus metrics (enabled by default; toggle via `ENABLE_METRICS`).
  - Key histograms: `embedding_request_latency_seconds`, `embedding_request_queue_wait_seconds{model}`, `embedding_batch_wait_seconds{model}`, `chat_request_queue_wait_seconds{model}`, `chat_batch_wait_seconds{model}`, `audio_request_queue_wait_seconds{model}`.

## Quick start (dev)

```bash
uv sync
```

```bash
MODELS=BAAI/bge-m3 uv run python scripts/run_dev.py --device auto
# MODELS must be set (comma-separated). Alternatively:
# uv run python scripts/run_dev.py --models BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507
```

Default model cache is locked to the repo-local `models/` directory. Pre-download models via `scripts/download_models.py` (always writes to `models/`) before building or running the service. For private/licensed models, set `HF_TOKEN` only when running the download script; the runtime uses local files only.

Environment variables can be kept in a `.env` file (see `.env.example`) and are loaded on startup without overriding existing variables. The example file ships with a conservative `MODEL_DEVICE=auto` + single-GPU friendly concurrency profile; for best results you should tune `MAX_CONCURRENT`, per-capability `*_MAX_WORKERS`, and batching windows based on your actual hardware (CUDA, MPS, or CPU-only). Startup performs an optional warmup for each model (toggle via `ENABLE_WARMUP`, default on): it runs a batch through every executor worker across supported capabilities (embeddings, chat, vision, audio) to initialize per-thread tokenizers/pipelines and compile kernels.

Warmup controls:
- `WARMUP_BATCH_SIZE` / `WARMUP_STEPS`: adjust batch and repetitions.
- `WARMUP_INFERENCE_MODE` (default `1`): use `torch.inference_mode()` / `no_grad` during warmup.
- `WARMUP_VRAM_BUDGET_MB` / `WARMUP_VRAM_PER_WORKER_MB`: limit concurrent warmup workers based on memory headroom.
- `WARMUP_ALLOWLIST` / `WARMUP_SKIPLIST`: include or skip specific models.
- **Fail-fast guarantee**: when `ENABLE_WARMUP=1`, startup will always fail if any model warmup fails; turn warmup off entirely with `ENABLE_WARMUP=0` if you prefer to skip this guardrail.

Warmup worker selection (VRAM budgeting):

- **Base workers**: per capability, warmup starts from `min(executor_max_workers, MAX_CONCURRENT)`.
- **VRAM budget**: on CUDA devices, if `WARMUP_VRAM_BUDGET_MB > 0` and `WARMUP_VRAM_PER_WORKER_MB > 0`, warmup further caps workers to `floor(budget / per_worker)`; if `WARMUP_VRAM_BUDGET_MB=0`, the budget defaults to the runtime free VRAM reported by `torch.cuda.mem_get_info`.
- **Effective workers**: `max(1, min(base_workers, floor(budget / per_worker)))`.

Example: with `MAX_CONCURRENT=4`, an executor sized to 8 workers, `WARMUP_VRAM_BUDGET_MB=4096` and `WARMUP_VRAM_PER_WORKER_MB=1024`, warmup will fan out to 4 workers for that capability.

Request batching: by default the server can micro-batch concurrent embedding requests. Configure via `ENABLE_EMBEDDING_BATCHING` (default on), `EMBEDDING_BATCH_WINDOW_MS` (collection window, default `6` ms), and `EMBEDDING_BATCH_WINDOW_MAX_SIZE` (max combined batch). Set `EMBEDDING_BATCH_WINDOW_MS=0` to effectively disable coalescing.

Embedding cache: repeated inputs are served from an in-memory LRU keyed by the full text. Control size with `EMBEDDING_CACHE_SIZE` (default `256` entries per model instance); set to `0` to disable. Prometheus counters `embedding_cache_hits_total` / `embedding_cache_misses_total` expose effectiveness per model.

## Performance tuning (quick checklist)

- **Concurrency gate**: `MAX_CONCURRENT` caps how many requests may run model forwards at once via the global limiter. Per-capability worker counts (`EMBEDDING_MAX_WORKERS`, `CHAT_MAX_WORKERS`, `VISION_MAX_WORKERS`, `AUDIO_MAX_WORKERS`) size the underlying thread pools but do **not** bypass the limiter. For most deployments, keep each `*_MAX_WORKERS` ≤ `MAX_CONCURRENT` to avoid oversubscribing CPU/GPU threads; on a single GPU/MPS start with `MAX_CONCURRENT=1–2` and small worker pools, and only raise them if throughput improves while p99 stays acceptable.
- **Micro-batching**: keep `ENABLE_EMBEDDING_BATCHING=1`; tune `EMBEDDING_BATCH_WINDOW_MS` (e.g., 4–10 ms; default `6` ms) and `EMBEDDING_BATCH_WINDOW_MAX_SIZE` (8–16) to trade a few ms of queueing for higher throughput. Set `EMBEDDING_BATCH_WINDOW_MS=0` to disable coalescing.
- **Chat batching (text-only)**: `ENABLE_CHAT_BATCHING=1` by default; tune `CHAT_BATCH_WINDOW_MS` (e.g., 4–10 ms) and `CHAT_BATCH_MAX_SIZE` (4–8). Guards: `CHAT_MAX_PROMPT_TOKENS` (default 4096) and `CHAT_MAX_NEW_TOKENS` (default 2048). Vision models stay on the unbatched path unless `CHAT_BATCH_ALLOW_VISION=1`.
- **Chat token counting**: defaults to a dedicated pool (`CHAT_COUNT_USE_CHAT_EXECUTOR=0`). Flip to `1` only if you want counting to share chat worker threads and can tolerate possible head-of-line blocking.
- **Queueing**: `MAX_QUEUE_SIZE` controls how many requests can wait. Too large increases tail latency; too small yields 429s. Set per your SLA.
- **Warmup**: keep `ENABLE_WARMUP=1`; for heavier models raise `WARMUP_STEPS` / `WARMUP_BATCH_SIZE` (e.g., 2–3 steps, batch 4–8). Use `WARMUP_VRAM_BUDGET_MB` / `WARMUP_VRAM_PER_WORKER_MB` to bound warmup fan-out on tighter GPUs; restart after changing models or devices.
- **Device choice**: set `MODEL_DEVICE` (`cuda`, `cuda:<idx>`, `mps`, `cpu`). On macOS MPS, performance varies with temperature/power; on CUDA you can try `MAX_CONCURRENT=2–4`.
- **Batch/input guards**: `MAX_BATCH_SIZE` and `MAX_TEXT_CHARS` protect the API—keep them within what the model can handle.
- **Audio guard**: `MAX_AUDIO_BYTES` (default 25MB) limits upload size for Whisper endpoints.
- **Metrics & logs**: `ENABLE_METRICS` exposes `/metrics`; logs include `runtime_config`, `warmup_ok`, and `embedding_request` to observe tuning impact.

### GPU tuning (example starting points)

- **Single RTX-class GPU (e.g., RTX 6000 Ada/Pro 6000)**: `MODEL_DEVICE=cuda`, `MAX_CONCURRENT=1`, `EMBEDDING_BATCH_WINDOW_MS=4–6`, `EMBEDDING_BATCH_WINDOW_MAX_SIZE=16`, `MAX_QUEUE_SIZE=128`, `ENABLE_WARMUP=1`, `WARMUP_STEPS=2`, `WARMUP_BATCH_SIZE=8`. Increase `MAX_CONCURRENT` only if p99 stays flat.
- **Multi-GPU**: pin with `CUDA_VISIBLE_DEVICES` and run multiple replicas, one per GPU. Keep `MAX_CONCURRENT` per replica low (1–2) and rely on batching for utilization.
- **Latency-sensitive**: set `EMBEDDING_BATCH_WINDOW_MS=0` to disable coalescing; keep `MAX_CONCURRENT=1` to minimize tail.
 - **CPU-only edge device (e.g., small VM/NAS)**: `MODEL_DEVICE=cpu`, `MAX_CONCURRENT=1`, `EMBEDDING_MAX_WORKERS=1–2`, `CHAT_MAX_WORKERS=1–2`, `AUDIO_MAX_CONCURRENT=1`. Prefer slightly larger batch windows over higher concurrency to keep tail latency controllable.

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
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "messages": [
          {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 128,
        "temperature": 0.7
      }'
```

## Benchmarking

- Embeddings: `uv run python scripts/benchmark_embeddings.py --models BAAI/bge-m3 --n-requests 50 --concurrency 8`
- Chat: `uv run python scripts/benchmark_chat.py --model-name Qwen/Qwen3-4B-Instruct-2507 --prompt "Explain FP8 quantization" --n-requests 40 --concurrency 8`
- Audio (Whisper): `BASE_URL=http://localhost:8000 MODEL_NAME=openai/whisper-tiny uv run python scripts/benchmark_audio.py -- --n-requests 20 --concurrency 4`

All benchmark scripts accept environment overrides (e.g., `BASE_URL`, `MODEL_NAME`, `PROMPT`, `N_REQUESTS`, `CONCURRENCY`).

## Adding a new model

1. **Implement a handler**: Create `app/models/<your_model>.py` implementing `EmbeddingModel` (see `hf_embedding.py` for reference). Set `capabilities` on the handler (e.g., `["text-embedding"]`). Use `cache_dir` pointing to `models/` (or `HF_HOME` fallback) and `local_files_only=True`.
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
- OpenAI-style streaming chat completions (SSE/chunked responses)
- Request/trace IDs: include a per-request ID in logs and responses for easier tracing.
- Remote image telemetry / tuning: add bandwidth/throughput metrics for remote image fetch and finer-grained limits; remote fetch is already guardrailed with host allowlists, private IP blocking, MIME sniffing, size caps, and redirect checks.
- Rerank: implement lightweight rerank handler/endpoint once a backend is chosen (current codepath removed).
- Test additions: vision chat path, non-batch chat serialization, prompt-length guard, warmup OOM surfacing.

## License

MIT License. See `LICENSE` for details.
