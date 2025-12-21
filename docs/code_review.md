# Codebase assessment (living)

This document is a **high-level, human-readable** assessment of the current codebase state. It is intentionally *not* a line-numbered audit report.

## Project health snapshot

- **Static checks**: `ruff` and `mypy` are clean.
- **Tests**: the Python test suite is green.
- **Operational posture**: the server defaults to “fail fast” on startup when models / dependencies are missing, and it protects itself under load via bounded queues + semaphores.

## Architecture highlights (what’s working well)

- **Clear lifecycle**: `FastAPI` `lifespan` performs startup in a background thread (to keep Ctrl-C responsive), then cleanly drains work and shuts down batchers, model resources, and executors on exit.
- **Per-capability isolation**:
  - Dedicated **limiters**: `embedding` / `chat` / `vision` / `audio` + upstream proxy limiters.
  - Dedicated **thread pools**: `embedding` / `chat` / `vision` / `audio` (so one heavy path can’t starve others).
- **Backpressure by construction**: every high-frequency entry point has a bounded queue somewhere in the chain (limiter queue and/or batch queue), preventing unbounded RAM growth under bursty traffic.
- **Cooperative cancellation (best-effort)**:
  - API layer races “executor work” vs “client disconnect” vs “hard timeout”.
  - Models receive `cancel_event` where supported and check it between expensive steps.
  - Whisper has an optional **subprocess kill mode** for hard cancellation.
- **OOM resilience**: chat batching detects CUDA OOM and degrades batch size (cooldown-based recovery) instead of spiraling into repeated failures.
- **Security posture for vision inputs**:
  - Remote image fetching is disabled by default.
  - When enabled, it is guard-railed via host allowlist + private IP rejection + MIME allowlist + byte budget + redirect checks.

## Thread safety & resource management assessment

### What looks solid

- **Thread pool creation is locked** and reused process-wide to avoid accidental executor leaks.
- **Thread-unsafe handlers are supported** (`thread_safe = False` + internal locks).
- **Caches and queues are bounded**:
  - Embedding LRU cache is bounded by entry count.
  - Batch queues are bounded per model.
- **Temporary file handling**:
  - Audio uploads are streamed to a tempfile with a hard size limit and unlinked in `finally`.

### Remaining risks / sharp edges (worth tracking)

- **Context propagation into executor threads**:
  - Request IDs are stored in `contextvars`, but context is not propagated into thread pool workers by default.
  - Outcome: logs emitted *inside* model handlers may not include `request_id` even though API logs do.
- **Best-effort cancellation**:
  - Cancelling the asyncio `Future` does not preempt a running kernel in PyTorch/Transformers; it only stops waiting and frees queue/limiter capacity.
  - Under sustained timeouts, work can still accumulate inside worker threads until they complete.
- **Daemon-thread executors**:
  - Daemon threads improve shutdown responsiveness, but they are a deliberate tradeoff: in a hard interpreter exit, in-flight work may be dropped.

## Optimization opportunities (high ROI)

- **Propagate tracing context to executors** (`contextvars.copy_context()` wrapper around `run_in_executor`) so handler logs can include `request_id`.
- **Cache “signature inspection” results** in the chat route (whether a model accepts `cancel_event`) to avoid per-request `inspect.signature()` overhead.
- **Optional: prompt-length bucketing** in chat batching to reduce padding waste for heterogeneous prompt lengths.
- **Optional: rerank batching** if rerank becomes high-QPS (it currently runs as a straightforward executor call).

## Documentation alignment

- `README.md`, `docs/perf_testing.md`, and `docs/upstream_proxy.md` reflect the current runtime behavior and configuration knobs.
- Environment example: the repo ships `env.example` (copy to `.env`) to keep configuration discoverable without committing secrets.

If you make meaningful architectural changes (limiters, batching semantics, cancellation behavior), please update this document and the README together so operators don’t have to “learn by reading code.”
