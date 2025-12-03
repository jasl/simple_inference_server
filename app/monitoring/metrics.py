import os
from contextlib import suppress

from prometheus_client import Counter, Histogram, make_asgi_app
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
