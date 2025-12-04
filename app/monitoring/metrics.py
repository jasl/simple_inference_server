import os
from contextlib import suppress

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from starlette.applications import Starlette

REQUEST_COUNT = Counter(
    "embedding_requests_total",
    "Total number of embedding requests",
    labelnames=("model", "status"),
)

REQUEST_LATENCY = Histogram(
    "embedding_request_latency_seconds",
    "Embedding request latency in seconds",
    labelnames=("model",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUEST_QUEUE_WAIT = Histogram(
    "embedding_request_queue_wait_seconds",
    "Time spent waiting for embedding worker",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

CHAT_REQUEST_COUNT = Counter(
    "chat_requests_total",
    "Total number of chat/completion requests",
    labelnames=("model", "status"),
)

CHAT_REQUEST_LATENCY = Histogram(
    "chat_request_latency_seconds",
    "Chat/completion request latency in seconds",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
)

CHAT_REQUEST_QUEUE_WAIT = Histogram(
    "chat_request_queue_wait_seconds",
    "Time spent waiting for chat worker",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

CHAT_BATCH_QUEUE = Gauge(
    "chat_batch_queue_size",
    "Current queue size for chat batching",
    labelnames=("model",),
)

CHAT_BATCH_SIZE = Histogram(
    "chat_batch_size",
    "Number of requests in a chat batch",
    labelnames=("model",),
    buckets=(1, 2, 4, 6, 8, 12, 16, 24, 32),
)

CHAT_BATCH_WAIT = Histogram(
    "chat_batch_wait_seconds",
    "Time from enqueue to batch execution for chat",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

CHAT_BATCH_OOM_RETRIES = Counter(
    "chat_batch_oom_retries_total",
    "Number of OOM retries triggered by chat batching",
    labelnames=("model",),
)

CHAT_BATCH_QUEUE_REJECTIONS = Counter(
    "chat_batch_queue_rejections_total",
    "Requests rejected due to chat batch queue limits",
    labelnames=("model",),
)

CHAT_COUNT_POOL_SIZE = Gauge(
    "chat_count_pool_size",
    "Worker count of the chat token counting executor",
)

AUDIO_REQUEST_COUNT = Counter(
    "audio_requests_total",
    "Total number of audio transcription/translation requests",
    labelnames=("model", "status"),
)

AUDIO_REQUEST_LATENCY = Histogram(
    "audio_request_latency_seconds",
    "Audio request latency in seconds",
    labelnames=("model",),
    buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0),
)

AUDIO_REQUEST_QUEUE_WAIT = Histogram(
    "audio_request_queue_wait_seconds",
    "Time spent waiting for audio worker",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
)

QUEUE_REJECTIONS = Counter(
    "embedding_queue_rejections_total",
    "Requests rejected due to queue limits",
)

CACHE_HITS = Counter(
    "embedding_cache_hits_total",
    "Cache hits when serving embeddings",
    labelnames=("model",),
)

CACHE_MISSES = Counter(
    "embedding_cache_misses_total",
    "Cache misses when serving embeddings",
    labelnames=("model",),
)

EMBED_BATCH_WAIT = Histogram(
    "embedding_batch_wait_seconds",
    "Time from enqueue to batch execution for embeddings",
    labelnames=("model",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

WARMUP_POOL_READY = Gauge(
    "warmup_pool_ready_workers",
    "Number of executor workers warmed up per model/capability",
    labelnames=("model", "capability", "executor"),
)


def setup_metrics(app: Starlette) -> None:
    if os.getenv("ENABLE_METRICS", "1") == "0":
        return
    # Mount Prometheus ASGI app at /metrics
    app.mount("/metrics", make_asgi_app())


def record_request(model: str, status: str) -> None:
    with suppress(Exception):
        REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        REQUEST_LATENCY.labels(model=model).observe(seconds)


def observe_queue_wait(model: str, seconds: float) -> None:
    with suppress(Exception):
        REQUEST_QUEUE_WAIT.labels(model=model).observe(seconds)


def record_chat_request(model: str, status: str) -> None:
    with suppress(Exception):
        CHAT_REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_chat_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        CHAT_REQUEST_LATENCY.labels(model=model).observe(seconds)


def observe_chat_queue_wait(model: str, seconds: float) -> None:
    with suppress(Exception):
        CHAT_REQUEST_QUEUE_WAIT.labels(model=model).observe(seconds)


def record_chat_batch_queue(model: str, size: int) -> None:
    with suppress(Exception):
        CHAT_BATCH_QUEUE.labels(model=model).set(size)


def observe_chat_batch_size(model: str, size: int) -> None:
    with suppress(Exception):
        CHAT_BATCH_SIZE.labels(model=model).observe(size)


def observe_chat_batch_wait(model: str, seconds: float) -> None:
    with suppress(Exception):
        CHAT_BATCH_WAIT.labels(model=model).observe(seconds)


def record_chat_batch_oom_retry(model: str) -> None:
    with suppress(Exception):
        CHAT_BATCH_OOM_RETRIES.labels(model=model).inc()


def record_chat_batch_queue_rejection(model: str) -> None:
    with suppress(Exception):
        CHAT_BATCH_QUEUE_REJECTIONS.labels(model=model).inc()


def record_chat_count_pool_size(workers: int) -> None:
    with suppress(Exception):
        CHAT_COUNT_POOL_SIZE.set(workers)


def record_queue_rejection() -> None:
    with suppress(Exception):
        QUEUE_REJECTIONS.inc()


def record_cache_usage(model: str, hits: int, misses: int) -> None:
    """Record cache hits/misses for observability."""

    with suppress(Exception):
        if hits:
            CACHE_HITS.labels(model=model).inc(hits)
        if misses:
            CACHE_MISSES.labels(model=model).inc(misses)


def observe_embedding_batch_wait(model: str, seconds: float) -> None:
    with suppress(Exception):
        EMBED_BATCH_WAIT.labels(model=model).observe(seconds)


def record_audio_request(model: str, status: str) -> None:
    with suppress(Exception):
        AUDIO_REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_audio_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        AUDIO_REQUEST_LATENCY.labels(model=model).observe(seconds)


def observe_audio_queue_wait(model: str, seconds: float) -> None:
    with suppress(Exception):
        AUDIO_REQUEST_QUEUE_WAIT.labels(model=model).observe(seconds)


def record_warmup_pool_ready(model: str, capability: str, executor: str, workers: int) -> None:
    with suppress(Exception):
        WARMUP_POOL_READY.labels(model=model, capability=capability, executor=executor).set(workers)
AUDIO_GENERIC_LABEL_WARN = Counter(
    "audio_queue_generic_label_warn_total",
    "Audio limiter used generic label instead of model/task",
)

GENERIC_LABEL_WARN = Counter(
    "queue_generic_label_warn_total",
    "Limiter used generic label instead of model/task",
)
