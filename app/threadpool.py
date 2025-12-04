from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from app.concurrency.limiter import MAX_CONCURRENT

# Allow per-capability sizing; defaults are decoupled from the global limiter but still bounded >=1.
EMBEDDING_MAX_WORKERS = max(1, int(os.getenv("EMBEDDING_MAX_WORKERS", "4")))
CHAT_MAX_WORKERS = max(1, int(os.getenv("CHAT_MAX_WORKERS", "4")))
VISION_MAX_WORKERS = max(1, int(os.getenv("VISION_MAX_WORKERS", "2")))
RERANK_MAX_WORKERS = max(1, int(os.getenv("RERANK_MAX_WORKERS", "2")))
AUDIO_MAX_WORKERS = max(
    1,
    int(os.getenv("AUDIO_MAX_WORKERS", os.getenv("AUDIO_MAX_CONCURRENT", "1"))),
)

_state: dict[str, ThreadPoolExecutor | None] = {
    "embedding_executor": None,
    "chat_executor": None,
    "vision_executor": None,
    "rerank_executor": None,
    "audio_executor": None,
}


def _get_executor(kind: str, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
    executor = _state[kind]
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        _state[kind] = executor
    return executor


def get_embedding_executor() -> ThreadPoolExecutor:
    return _get_executor("embedding_executor", EMBEDDING_MAX_WORKERS, "embed-worker")


def get_chat_executor() -> ThreadPoolExecutor:
    return _get_executor("chat_executor", CHAT_MAX_WORKERS, "chat-worker")


def get_vision_executor() -> ThreadPoolExecutor:
    return _get_executor("vision_executor", VISION_MAX_WORKERS, "vision-worker")


def get_rerank_executor() -> ThreadPoolExecutor:
    return _get_executor("rerank_executor", RERANK_MAX_WORKERS, "rerank-worker")


def get_audio_executor() -> ThreadPoolExecutor:
    return _get_executor("audio_executor", AUDIO_MAX_WORKERS, "audio-worker")


def shutdown_embedding_executor() -> None:
    executor = _state.get("embedding_executor")
    _state["embedding_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_chat_executor() -> None:
    executor = _state.get("chat_executor")
    _state["chat_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_vision_executor() -> None:
    executor = _state.get("vision_executor")
    _state["vision_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_rerank_executor() -> None:
    executor = _state.get("rerank_executor")
    _state["rerank_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_audio_executor() -> None:
    executor = _state.get("audio_executor")
    _state["audio_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_executors() -> None:
    shutdown_embedding_executor()
    shutdown_chat_executor()
    shutdown_vision_executor()
    shutdown_rerank_executor()
    shutdown_audio_executor()
