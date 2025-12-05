from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Allow per-capability sizing; defaults are decoupled from the global limiter but still bounded >=1.
EMBEDDING_MAX_WORKERS = max(1, int(os.getenv("EMBEDDING_MAX_WORKERS", "4")))
CHAT_MAX_WORKERS = max(1, int(os.getenv("CHAT_MAX_WORKERS", "4")))
VISION_MAX_WORKERS = max(1, int(os.getenv("VISION_MAX_WORKERS", "2")))
AUDIO_MAX_WORKERS = max(
    1,
    int(os.getenv("AUDIO_MAX_WORKERS", os.getenv("AUDIO_MAX_CONCURRENT", "1"))),
)
EMBEDDING_COUNT_MAX_WORKERS = max(1, int(os.getenv("EMBEDDING_COUNT_MAX_WORKERS", "2")))

_state_lock = threading.Lock()
_state: dict[str, ThreadPoolExecutor | None] = {
    "embedding_executor": None,
    "embedding_count_executor": None,
    "chat_executor": None,
    "vision_executor": None,
    "audio_executor": None,
}


def _get_executor(kind: str, max_workers: int, thread_name_prefix: str) -> ThreadPoolExecutor:
    """Return (and lazily create) a shared ThreadPoolExecutor of the given kind.

    Creation is guarded by a process-wide lock so that concurrent first use from
    multiple requests cannot accidentally create and leak multiple executors.
    """
    with _state_lock:
        executor = _state.get(kind)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
            _state[kind] = executor
        return executor


def enforce_single_worker(kind: str) -> int:
    """Force a given executor kind to a single worker.

    Returns the previous configured max_workers to aid logging.
    """

    kind_normalized = kind.strip().lower()
    max_attr = f"{kind_normalized.upper()}_MAX_WORKERS"
    state_key = f"{kind_normalized}_executor"
    if max_attr not in globals() or state_key not in _state:
        raise ValueError(f"Unknown executor kind: {kind}")

    with _state_lock:
        previous = int(globals().get(max_attr, 1))
        globals()[max_attr] = 1
        executor = _state.get(state_key)
        _state[state_key] = None

    if executor is not None:
        executor.shutdown(wait=True)

    return previous


def get_embedding_executor() -> ThreadPoolExecutor:
    return _get_executor("embedding_executor", EMBEDDING_MAX_WORKERS, "embed-worker")


def get_embedding_count_executor() -> ThreadPoolExecutor:
    return _get_executor("embedding_count_executor", EMBEDDING_COUNT_MAX_WORKERS, "embed-count")


def get_chat_executor() -> ThreadPoolExecutor:
    return _get_executor("chat_executor", CHAT_MAX_WORKERS, "chat-worker")


def get_vision_executor() -> ThreadPoolExecutor:
    return _get_executor("vision_executor", VISION_MAX_WORKERS, "vision-worker")


def get_audio_executor() -> ThreadPoolExecutor:
    return _get_executor("audio_executor", AUDIO_MAX_WORKERS, "audio-worker")


def shutdown_embedding_executor() -> None:
    with _state_lock:
        executor = _state.get("embedding_executor")
        _state["embedding_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_embedding_count_executor() -> None:
    with _state_lock:
        executor = _state.get("embedding_count_executor")
        _state["embedding_count_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_chat_executor() -> None:
    with _state_lock:
        executor = _state.get("chat_executor")
        _state["chat_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_vision_executor() -> None:
    with _state_lock:
        executor = _state.get("vision_executor")
        _state["vision_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_audio_executor() -> None:
    with _state_lock:
        executor = _state.get("audio_executor")
        _state["audio_executor"] = None
    if executor is not None:
        executor.shutdown(wait=True)


def shutdown_executors() -> None:
    shutdown_embedding_executor()
    shutdown_embedding_count_executor()
    shutdown_chat_executor()
    shutdown_vision_executor()
    shutdown_audio_executor()
