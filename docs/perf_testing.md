# Performance Testing & Tuning Guide

This document covers performance tuning, load testing, and monitoring for Simple Inference Server.

## Quick Tuning Checklist

Before diving into detailed load testing, use this checklist to configure your deployment:

### Concurrency Gate

- **`MAX_CONCURRENT`** caps how many requests may run model forward passes simultaneously
- Per-capability worker counts (`EMBEDDING_MAX_WORKERS`, `CHAT_MAX_WORKERS`, etc.) size thread pools but do **not** bypass the limiter
- Keep each `*_MAX_WORKERS` ≤ `MAX_CONCURRENT` to avoid oversubscribing CPU/GPU threads
- On a single GPU/MPS, start with `MAX_CONCURRENT=1–2` and tune batching instead of raising concurrency

### Micro-Batching

- Keep `ENABLE_EMBEDDING_BATCHING=1`; tune `EMBEDDING_BATCH_WINDOW_MS` (4–10 ms) and `EMBEDDING_BATCH_WINDOW_MAX_SIZE` (8–16)
- For chat: `ENABLE_CHAT_BATCHING=1`; tune `CHAT_BATCH_WINDOW_MS` (4–10 ms) and `CHAT_BATCH_MAX_SIZE` (4–8)
- Enable `CHAT_BATCH_PROMPT_BUCKETING=1` to reduce padding waste for heterogeneous prompts

### Queueing

- `MAX_QUEUE_SIZE` controls waiting requests; too large increases tail latency, too small yields 429s
- Set per your SLA requirements

### Warmup

- Keep `ENABLE_WARMUP=1` for production deployments
- For heavier models, raise `WARMUP_STEPS` (2–3) and `WARMUP_BATCH_SIZE` (4–8)
- Use `WARMUP_VRAM_BUDGET_MB` / `WARMUP_VRAM_PER_WORKER_MB` to bound warmup on tighter GPUs

### Device Selection

- Set `MODEL_DEVICE` to `cuda`, `cuda:<idx>`, `mps`, or `cpu`
- On macOS MPS, performance varies with temperature/power

### Input Guards

- `MAX_BATCH_SIZE` and `MAX_TEXT_CHARS` protect the API from oversized requests
- `MAX_AUDIO_BYTES` (default 25MB) limits Whisper upload size

### GPU Tuning Examples

| Scenario | Configuration |
|----------|---------------|
| **Single RTX-class GPU** | `MODEL_DEVICE=cuda`, `MAX_CONCURRENT=1`, `EMBEDDING_BATCH_WINDOW_MS=4–6`, `EMBEDDING_BATCH_WINDOW_MAX_SIZE=16`, `MAX_QUEUE_SIZE=128`, `ENABLE_WARMUP=1` |
| **Multi-GPU** | Pin with `CUDA_VISIBLE_DEVICES`, run one replica per GPU, keep `MAX_CONCURRENT=1–2` per replica |
| **Latency-sensitive** | `EMBEDDING_BATCH_WINDOW_MS=0` (disable coalescing), `MAX_CONCURRENT=1` |
| **CPU-only edge** | `MODEL_DEVICE=cpu`, `MAX_CONCURRENT=1`, `EMBEDDING_MAX_WORKERS=1–2`, `CHAT_MAX_WORKERS=1–2`, `AUDIO_MAX_CONCURRENT=1` |

---

## Load Testing Guide

### Overview

This section provides guidance for high-concurrency load testing and monitoring. Main objectives:

- Verify **throughput, latency, and error rates** for embeddings / chat / audio under high concurrency on the target hardware.
- Use `/metrics` and `/health` to observe **rate limiting/queuing, batching, caching, and warmup** status.
- Identify configuration issues, model overload, or potential resource problems (CPU/GPU saturation, memory/thread anomalies) early.

This guide describes methods and suggested commands that you can execute as needed.

---

## Environment Setup

### Starting the Service

1. Install dependencies:

```bash
uv sync
```

2. Pre-download models (optional if `AUTO_DOWNLOAD_MODELS=1` is kept):

```bash
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 uv run python scripts/download_models.py
MODELS=openai/whisper-tiny uv run python scripts/download_models.py
```

3. Start the service (example):

```bash
# Embeddings + chat only
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 \
MODEL_DEVICE=auto \
MAX_CONCURRENT=2 \
uv run python scripts/run_dev.py --device auto

# With Whisper audio
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507,openai/whisper-tiny \
MODEL_DEVICE=auto \
MAX_CONCURRENT=2 \
AUDIO_MAX_CONCURRENT=1 \
uv run python scripts/run_dev.py --device auto
```

### Basic Health Check

- `GET /health`: Check model list, warmup completion, and failed models.
- `GET /metrics`: Prometheus metrics endpoint (all metric names in this document come from here).

---

### Concurrency Model Quick Reference

- Global rate limiting: `MAX_CONCURRENT` + `MAX_QUEUE_SIZE` control **how many requests can execute model forward passes simultaneously** and queue length via limiter, preventing unbounded queuing.
- Thread pool size: `EMBEDDING_MAX_WORKERS`, `CHAT_MAX_WORKERS`, `VISION_MAX_WORKERS`, `AUDIO_MAX_WORKERS` only determine the thread count in each `ThreadPoolExecutor` and do not bypass the limiter.
- Recommended practices:
  - In most scenarios, set `*_MAX_WORKERS <= MAX_CONCURRENT` to avoid excessive CPU/GPU thread contention on the device.
  - Single machine, single GPU: Start with `MAX_CONCURRENT=1` or `2` plus a small thread pool, and improve throughput by tuning `EMBEDDING_BATCH_WINDOW_MS` / `CHAT_BATCH_WINDOW_MS` rather than blindly increasing concurrency.
  - Audio path: `AUDIO_MAX_CONCURRENT` controls Whisper concurrency separately; it's usually recommended to keep it less than or equal to `MAX_CONCURRENT` to prevent audio tasks from slowing down embeddings/chat.

## Load Test Scenarios

### 1. Embeddings Load Test (Core Path for High QPS, High-Frequency Calls)

**Recommended script**: `scripts/benchmark_embeddings.py`  
**Target**: Observe throughput and p95/p99 under different `MAX_CONCURRENT`, `EMBEDDING_BATCH_WINDOW_MS`, `EMBEDDING_BATCH_WINDOW_MAX_SIZE` settings.

Example command:

```bash
# Single model, concurrency 8, 200 requests
uv run python scripts/benchmark_embeddings.py \
  --models BAAI/bge-m3 \
  --n-requests 200 \
  --concurrency 8 \
  --base-url http://localhost:8000
```

**Dimensions to try**:

- `MAX_CONCURRENT = 1 / 2 / 4`
- `EMBEDDING_BATCH_WINDOW_MS = 0 / 4 / 6 / 10`
- `EMBEDDING_BATCH_WINDOW_MAX_SIZE = 8 / 16 / 32`

**Key metrics to watch**:

- Whether throughput increases significantly with `MAX_CONCURRENT` and batch window changes.
- Whether 429 (Too Many Requests) errors appear frequently (indicates queue configuration is too small or model is too slow).
- Whether tail latency (p95/p99) is within acceptable range.

---

### 2. Chat Text Load Test (LLM Short Conversations)

**Recommended script**: `scripts/benchmark_chat.py`  
**Target**: Verify batching effectiveness under `ENABLE_CHAT_BATCHING`, and prompt limits/fallback behavior.

Example command:

```bash
uv run python scripts/benchmark_chat.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --prompt "Explain FP8 quantization" \
  --n-requests 100 \
  --concurrency 8 \
  --base-url http://localhost:8000
```

**Dimensions to try**:

- `CHAT_BATCH_WINDOW_MS = 4 / 8 / 10`
- `CHAT_BATCH_MAX_SIZE = 4 / 8`
- Whether `CHAT_MAX_PROMPT_TOKENS` is large enough; too small will cause 400 errors.

**Key metrics to watch**:

- Whether noticeably higher throughput is observed when batching is enabled.
- Whether 429 errors mainly come from the chat batch queue (rather than global limiter); this can be distinguished via metrics.
- Whether overly long prompts return a friendly 400 error.

---

### 3. Chat + Vision (Qwen-VL) Functional Validation

These calls are typically heavier; high concurrency or extended load testing is not recommended. Just need:

- Low concurrency (e.g., 2–4).
- Few requests (e.g., 10–20), to verify:

  - When remote images are disabled (`ALLOW_REMOTE_IMAGES=0`), whether `data:` / local path inputs work stably.
  - When remote images are enabled (`ALLOW_REMOTE_IMAGES=1` with `REMOTE_IMAGE_HOST_ALLOWLIST` configured), whether large images or unauthorized hosts are properly rejected. The remote HTTP fetch includes a series of security measures: only domains explicitly in the allowlist are allowed, private/loopback IPs are rejected, single response size is limited (`MAX_REMOTE_IMAGE_BYTES`), MIME type and redirect count are validated, and `timeout` and connection limits are logged when the HTTP client is created to help troubleshoot configuration issues.

Sample curl commands (refer to Qwen-VL examples in README) are sufficient; no need to write additional scripts.

---

### 4. Whisper Audio Load Test

**Recommended script**: `scripts/benchmark_audio.py`  
**Target**: Verify the audio path doesn't overwhelm embeddings/chat, and whether `AUDIO_MAX_CONCURRENT` and `AUDIO_MAX_QUEUE_SIZE` are appropriate.

Example command:

```bash
BASE_URL=http://localhost:8000 \
MODEL_NAME=openai/whisper-tiny \
uv run python scripts/benchmark_audio.py \
  -- --n-requests 40 --concurrency 4
```

**Key metrics to watch**:

- For CPU-only or low-performance GPU, start with `AUDIO_MAX_CONCURRENT=1`.
- Audio requests should not significantly increase queue wait times for embeddings/chat.

---

## Key Monitoring Metrics & Expectations

### 1. Request-Level Metrics

**Embeddings**

- `embedding_requests_total{model,status}`
- `embedding_request_latency_seconds{model}`
- `embedding_request_queue_wait_seconds{model}`

**Chat**

- `chat_requests_total{model,status}`
- `chat_request_latency_seconds{model}`
- `chat_request_queue_wait_seconds{model}`

**Audio**

- `audio_requests_total{model,status}`
- `audio_request_latency_seconds{model}`
- `audio_request_queue_wait_seconds{model}`

**Cancellation & Timeouts (Unified Semantics)**

- All three main paths (embeddings / chat / audio) use a unified helper to handle the race between "model execution + client disconnect + hard timeout":
  - When `EMBEDDING_GENERATE_TIMEOUT_SEC` / `CHAT_GENERATE_TIMEOUT_SEC` / `AUDIO_PROCESS_TIMEOUT_SEC` is reached, returns `504 Gateway Timeout` and counts toward `*_requests_total{status="504"}`.
  - When client actively disconnects or is treated as cancelled on server side, returns `499 Client Closed Request`.
- These cancellations are **best effort**: they don't preempt the underlying kernel, just stop sending data to the client and release queue/concurrency slots. When tuning, you can use the ratio of 499/504 to distinguish "client-side cancellation" from "server-side timeout".

**Global Queue Rejections**

- `embedding_queue_rejections_total` (shared by limiter / audio_limiter)

**Recommended checks**:

- Whether status=200 count matches successful requests reported by the load testing tool.
- Whether status=429 / 503 rate is acceptable (some during short parameter tuning is OK; should minimize during long-term operation).
- Whether queue wait time histograms show many samples >0.1s or >1s.

---

### 2. Batching & Caching Metrics

**Chat Batching**

- `chat_batch_queue_size{model}`: Current chat batch queue depth
- `chat_batch_size{model}`: Distribution of requests per batch
- `chat_batch_wait_seconds{model}`: Wait time from enqueue to batch execution
- `chat_batch_oom_retries_total{model}`: Batch OOM retry count
- `chat_batch_queue_rejections_total{model}`: Requests rejected due to queue limits
- `chat_batch_requeues_total{model}`: Re-queue count due to incompatible configuration/fallback
- `chat_count_pool_size`: Token counting thread pool size

When chat queue ages significantly and requests are dropped by `_QUEUE_MAX_WAIT_SEC` or `_REQUEUE_MAX_WAIT_SEC`, logs will show lines like `chat_batch_items_dropped_due_to_queue_wait` with `model`, `dropped` (count dropped this time), `queue_size`, `max_queue_size`, and `queue_max_wait_sec`. Combined with `chat_batch_queue_rejections_total{model}`, you can determine whether the batch queue itself is too small / window too long, or if the model's actual compute capacity is insufficient.

**Embedding Batching & Caching**

- `embedding_batch_wait_seconds{model}`: Embedding batch wait time
- `embedding_cache_hits_total{model}`
- `embedding_cache_misses_total{model}`

**Recommended checks**:

- Whether chat batch size is concentrated in a reasonable range (e.g., 2–8).
- Whether batch wait time roughly corresponds to `CHAT_BATCH_WINDOW_MS` and doesn't frequently exceed the window significantly.
- Whether cache hit/miss ratio meets expectations: high-repetition content scenarios expect high hit rate; purely random text will have mostly misses.

---

### 3. Warmup & Health Status

**Warmup Metrics**

- `warmup_pool_ready_workers{model,capability,executor}`

**Health Info (`GET /health`)**

- `status`: `ok` / `unhealthy`
- `warmup.ok_models`, `warmup.failures`
- `warmup.capabilities[model][capability]`
- `chat_batch_queues` / `embedding_batch_queues`: Queue depth and max size
- `runtime_config`: Snapshot of current effective key parameters

**Recommended checks**:

- After startup, whether warmup marks expected models and capabilities as `True`.
- If warmup is disabled or restricted (via `ENABLE_WARMUP=0` or allowlist/skiplist), whether `/health` output matches the configuration.
- During load testing, whether `chat_batch_queues` / `embedding_batch_queues` are consistently near their limits (if so, rate limiting/batch window/model bottleneck needs adjustment).

---

## Common Issues & Tuning Strategies (Checklist)

1. **Heavy 429 (Too Many Requests)**
   - Check `embedding_queue_rejections_total`, `chat_batch_queue_rejections_total`:
     - If caused by limiter → appropriately increase `MAX_QUEUE_SIZE` or `MAX_CONCURRENT`, while monitoring p99.
     - If caused by chat batch queue → increase `CHAT_BATCH_QUEUE_SIZE`, or reduce batch window/batch size to avoid backlog.

2. **High p99 latency but GPU/CPU utilization not high**
   - Check:
     - Whether `EMBEDDING_BATCH_WINDOW_MS` is set too large.
     - Whether chat/embedding queue wait histograms are elongated.
   - Tuning strategy:
     - Reduce `EMBEDDING_BATCH_WINDOW_MS` / `CHAT_BATCH_WINDOW_MS`.
     - On single-machine setups, try `MAX_CONCURRENT=1` or 2, and improve throughput through batching rather than simply increasing concurrency.

3. **Audio requests slowing overall response**
   - Check:
     - Whether `audio_request_queue_wait_seconds` has increased significantly.
     - Whether embeddings/chat queue wait has deteriorated simultaneously.
   - Adjustments:
     - Reduce `AUDIO_MAX_CONCURRENT`, and ensure `MAX_CONCURRENT` primarily serves embeddings/chat.
     - If necessary, deploy audio on a separate instance.

4. **Warmup phase takes too long or occasionally triggers OOM**
   - Reduce:
     - `WARMUP_BATCH_SIZE`, `WARMUP_STEPS`.
   - Or limit:
     - `WARMUP_VRAM_BUDGET_MB`, `WARMUP_VRAM_PER_WORKER_MB` to make warmup use more conservative worker counts. Actual worker count is roughly `min(executor_max_workers, MAX_CONCURRENT, floor(budget / per_worker))`; when `WARMUP_VRAM_BUDGET_MB=0`, the current device's available VRAM is used as budget.
   - Verify warmup coverage and failed models via `/health` and `warmup_pool_ready_workers`.

---

## Appendix: Quick Reference Commands

- List models:

```bash
curl http://localhost:8000/v1/models
```

- Health check:

```bash
curl http://localhost:8000/health | jq .
```

- Fetch partial Metrics (manual inspection):

```bash
curl http://localhost:8000/metrics | grep -E "embedding_request_latency|chat_request_latency|audio_request_latency"
```

- Quick Embeddings manual test:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["hello","world"]}'
```

Follow the steps in this document when you need to perform load testing, executing commands and checking monitoring results accordingly.
