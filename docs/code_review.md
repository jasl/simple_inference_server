# Codebase assessment

This document captures the current state of the codebase after the refactoring and optimization pass. It highlights strengths, design decisions, and important implementation details.

## What works well

- **Startup contracts and fail-fast behavior**: The main startup path enforces model allowlisting, downloads when enabled, and validates dependencies like `ffmpeg`; optional warmup now fans out across all capabilities (embeddings, chat, vision) with per-worker execution and VRAM budgeting to avoid oversubscription. Warmup module state is now thread-safe with explicit locking.【F:app/main.py†L14-L91】【F:app/warmup.py†L98-L182】
- **Per-capability backpressure**: Dedicated limiters for embedding, chat, and audio traffic prevent one capability from starving another. Each limiter couples a bounded queue with a semaphore so requests either acquire capacity or receive clear 429/503-style errors without unbounded buffering; chat batching adds its own bounded queue and prompt-length guard before scheduling heavy work.【F:app/concurrency/limiter.py†L21-L97】【F:app/chat_batching.py†L47-L175】
- **Batching and caching for high-frequency paths**: Embedding and chat handlers share configurable executors sized to `MAX_CONCURRENT`, leverage micro-batching with windowed coalescing, and embedder models include per-request no-grad guards plus an LRU cache (with fast xxhash keys) to avoid redundant computation.【F:app/threadpool.py†L1-L46】【F:app/batching.py†L1-L180】【F:app/embedding_cache.py†L1-L102】
- **OOM graceful degradation**: Chat batching automatically halves the max batch size on OOM and recovers after a configurable cooldown (`CHAT_OOM_COOLDOWN_SEC`, default 5 minutes). This prevents cascading failures while allowing the service to self-heal.【F:app/chat_batching.py†L154-L190】
- **Shared utilities**: Common logic for OOM handling and device resolution is extracted to `app/models/generation_utils.py`, ensuring consistent behavior across text and vision chat models.【F:app/models/generation_utils.py†L1-L60】

## Local dev: running tests and linters

This repo uses `uv` to manage the environment. When running tests or tools locally, prefer:

- Tests:
  - `uv run pytest`
  - Or for a subset: `uv run pytest tests/test_embeddings_api.py tests/test_audio_api.py`
- Ruff:
  - `uv run ruff check app tests`
- mypy:
  - `uv run mypy app tests`

Avoid calling `pytest`, `ruff`, or `mypy` directly without `uv run` to ensure the correct environment and dependencies are used.

## Engineering notes / conventions

- When adding internal tasks that use `embedding_limiter`, `chat_limiter`, or `audio_limiter`, always set a queue label (model or task name) via `set_queue_label` / `set_audio_queue_label` so queue-wait metrics stay attributable instead of falling back to `generic`.
- Handler authors may expose a boolean `thread_safe` attribute; when `True` the handler is expected to be safe under up to `*_MAX_WORKERS` concurrent calls from the shared executor for that capability, without extra locking at the call site. When `False` the handler must serialize internal shared state (tokenizers, pipelines, HTTP clients) with its own locks; the server will still use the shared executor but will emit warnings if the corresponding worker count is >1 so operators know concurrency is effectively 1.
- **HTTP client lifecycle**: Vision model handlers (e.g., `QwenVLChat`) manage their own `httpx.Client` instance bound to the model lifecycle rather than using a module-level global. This prevents connection leaks in hot-reload or testing scenarios and allows clean resource cleanup.
- **Requeue path on chat batching**: When batching splits by generation parameters, leftover items are now requeued with bounded exponential backoff and a per-item deadline. This smooths burstiness without unbounded retries; items that cannot be requeued in time surface as 429s and are counted via `CHAT_BATCH_REQUEUES` / `CHAT_BATCH_QUEUE_REJECTIONS` metrics.【F:app/chat_batching.py†L171-L365】
- **Visibility of warmup coverage**: Warmup metrics record pool readiness and `/health` exposes `warmup` details including per-model capability success plus `warmup_failures`. Operators can see which models/capabilities skipped or failed warmup without inspecting logs.【F:app/api.py†L1076-L1136】【F:app/warmup.py†L140-L185】
- **Model handlers**: Embedding models can use `HFEmbeddingModel` directly from `app.models.hf_embedding`; custom embedding handlers should inherit from this class or implement the `EmbeddingModel` protocol. The empty shell classes `BgeM3Embedding` and `EmbeddingGemmaEmbedding` have been removed in favor of direct handler references in `model_config.yaml`.

## Future-facing structure: lightweight pools and protocols

- Protocols for embeddings/chat/audio now live in `app/models/base.py`, and placeholder `RerankModel` / `IntentModel` protocols are defined in the same style. Future rerank/intent handlers should implement these interfaces so they can plug into batching/limiters consistently with existing model types.
- A future “lightweight model pool” limiter is expected for embeddings/intent/rerank: conceptually a dedicated limiter + small thread pool that:
  - gatekeeps inexpensive models (embeddings/intent/rerank) independently from the heavier chat/vision/audio pools;
  - exposes its own `{light_model}_request_queue_wait_seconds` metrics and `*_queue_rejections_total` counters;
  - allows different `MAX_CONCURRENT_LIGHT` / `LIGHT_MAX_QUEUE_SIZE` settings tuned for high-QPS, low-latency workloads.
- The current design already keeps chat/audio on dedicated executors and an audio-specific limiter; the lightweight pool can be introduced alongside these without changing existing behavior, then endpoints for embeddings/intent/rerank can be gradually migrated to it.

## Documentation alignment

- README now describes warmup across all capabilities plus configuration toggles for budgets, allow/skip lists, and fail-fast behavior so operators can keep eager-loading guarantees consistent with runtime flags.【F:README.md†L180-L210】
