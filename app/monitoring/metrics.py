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


def record_chat_request(model: str, status: str) -> None:
    with suppress(Exception):
        CHAT_REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_chat_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        CHAT_REQUEST_LATENCY.labels(model=model).observe(seconds)


def record_chat_batch_queue(model: str, size: int) -> None:
    with suppress(Exception):
        CHAT_BATCH_QUEUE.labels(model=model).set(size)


def observe_chat_batch_size(model: str, size: int) -> None:
    with suppress(Exception):
        CHAT_BATCH_SIZE.labels(model=model).observe(size)


def record_chat_batch_oom_retry(model: str) -> None:
    with suppress(Exception):
        CHAT_BATCH_OOM_RETRIES.labels(model=model).inc()


def record_chat_batch_queue_rejection(model: str) -> None:
    with suppress(Exception):
        CHAT_BATCH_QUEUE_REJECTIONS.labels(model=model).inc()


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


def record_audio_request(model: str, status: str) -> None:
    with suppress(Exception):
        AUDIO_REQUEST_COUNT.labels(model=model, status=status).inc()


def observe_audio_latency(model: str, seconds: float) -> None:
    with suppress(Exception):
        AUDIO_REQUEST_LATENCY.labels(model=model).observe(seconds)
