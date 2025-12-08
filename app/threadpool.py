from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from app.config import settings


class _ExecutorState(TypedDict):
    """Type definition for executor state including both executor and max_workers."""

    executor: ThreadPoolExecutor | None
    max_workers: int


_state_lock = threading.Lock()
_state: dict[str, _ExecutorState] = {
    "embedding": {"executor": None, "max_workers": max(1, settings.embedding_max_workers)},
    "embedding_count": {"executor": None, "max_workers": max(1, settings.embedding_count_max_workers)},
    "chat": {"executor": None, "max_workers": max(1, settings.chat_max_workers)},
    "vision": {"executor": None, "max_workers": max(1, settings.vision_max_workers)},
    "audio": {"executor": None, "max_workers": max(1, settings.audio_max_workers)},
}

# Module-level constants for backward compatibility with external code
EMBEDDING_MAX_WORKERS = _state["embedding"]["max_workers"]
CHAT_MAX_WORKERS = _state["chat"]["max_workers"]
VISION_MAX_WORKERS = _state["vision"]["max_workers"]
AUDIO_MAX_WORKERS = _state["audio"]["max_workers"]


def _get_executor(kind: str, thread_name_prefix: str) -> ThreadPoolExecutor:
    """Return (and lazily create) a shared ThreadPoolExecutor of the given kind.

    Creation is guarded by a process-wide lock so that concurrent first use from
    multiple requests cannot accidentally create and leak multiple executors.
    """
    with _state_lock:
        entry = _state.get(kind)
        if entry is None:
            raise ValueError(f"Unknown executor kind: {kind}")
        executor = entry["executor"]
        if executor is None:
            max_workers = entry["max_workers"]
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
            entry["executor"] = executor
        return executor


def enforce_single_worker(kind: str) -> int:
    """Force a given executor kind to a single worker.

    Returns the previous configured max_workers to aid logging.
    """
    kind_normalized = kind.strip().lower()
    with _state_lock:
        entry = _state.get(kind_normalized)
        if entry is None:
            raise ValueError(f"Unknown executor kind: {kind}")
        previous = entry["max_workers"]
        entry["max_workers"] = 1
        executor = entry["executor"]
        entry["executor"] = None

    if executor is not None:
        executor.shutdown(wait=True)

    return previous


def get_embedding_executor() -> ThreadPoolExecutor:
    return _get_executor("embedding", "embed-worker")


def get_embedding_count_executor() -> ThreadPoolExecutor:
    return _get_executor("embedding_count", "embed-count")


def get_chat_executor() -> ThreadPoolExecutor:
    return _get_executor("chat", "chat-worker")


def get_vision_executor() -> ThreadPoolExecutor:
    return _get_executor("vision", "vision-worker")


def get_audio_executor() -> ThreadPoolExecutor:
    return _get_executor("audio", "audio-worker")


def _shutdown_executor(kind: str) -> None:
    """Shutdown and clear an executor of the given kind."""
    with _state_lock:
        entry = _state.get(kind)
        if entry is None:
            return
        executor = entry["executor"]
        entry["executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_embedding_executor() -> None:
    _shutdown_executor("embedding")


def shutdown_embedding_count_executor() -> None:
    _shutdown_executor("embedding_count")


def shutdown_chat_executor() -> None:
    _shutdown_executor("chat")


def shutdown_vision_executor() -> None:
    _shutdown_executor("vision")


def shutdown_audio_executor() -> None:
    _shutdown_executor("audio")


def shutdown_executors() -> None:
    shutdown_embedding_executor()
    shutdown_embedding_count_executor()
    shutdown_chat_executor()
    shutdown_vision_executor()
    shutdown_audio_executor()
