# Continuous batching plan (chat)

Scope and priorities
- Phase 1 (implemented): HF-backed continuous batching for text-only chat models to raise throughput and lower p99 without changing the OpenAI-compatible API. Responses stay non-streaming; no user-initiated abort yet.
- Phase 2 (later): add streaming responses plus user-side abort to reclaim KV cache mid-generation. Keep the ideas here so we do not forget the details.

Why
- A queue + scheduler lets concurrent chat requests share prefill/decoding kernels, improving GPU/CPU utilization and reducing per-request overhead.
- Transformers >=4.46 ships continuous batching primitives, and our pin (4.57.3) already includes them, so we can rely on upstream helpers rather than a custom engine.

Phase 1 behavior (non-streaming, no abort)
- Queue + scheduler: per-chat-model worker batches compatible requests (same decoding params) within `CHAT_BATCH_WINDOW_MS`, capped by `CHAT_BATCH_MAX_SIZE`. Only text-only chat models are opted in by default.
- Execution: uses model `batched_generate` when available (TextChatModel) and falls back to per-request `generate` otherwise. Single worker per model keeps model usage thread-safe.
- Limits: prompt length guard (`CHAT_MAX_PROMPT_TOKENS`) and `CHAT_MAX_NEW_TOKENS` ceiling reject oversize requests early to avoid OOM. CUDA OOM retries once after clearing cache; then falls back to sequential generation for that batch.
- Compatibility: CPU/MPS/CUDA all supported; vision models stay on the legacy path unless `CHAT_BATCH_ALLOW_VISION=1`.

Phase 2 notes (future TODOs)
- Streaming: push partial generations every `stream_interval` tokens to cut tail latency while sharing batches.
- User abort: accept `request_id`, track in the scheduler, and add an `/abort` endpoint (or header) to drop a live sequence and free KV cache promptly.
- Both should be config-gated so existing clients remain compatible.

Testing outline
- Add unit tests for the scheduler (batch windowing, config grouping, overflow rejections).
- Add integration tests that send concurrent chat requests and assert improved throughput vs. single-request path and that fallbacks still work when batching is disabled.
