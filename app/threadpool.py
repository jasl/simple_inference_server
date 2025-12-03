from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from app.concurrency.limiter import MAX_CONCURRENT

EMBEDDING_MAX_WORKERS = max(1, int(os.getenv("EMBEDDING_MAX_WORKERS", str(MAX_CONCURRENT))))
CHAT_MAX_WORKERS = max(1, int(os.getenv("CHAT_MAX_WORKERS", str(MAX_CONCURRENT))))

_state: dict[str, ThreadPoolExecutor | None] = {"embedding_executor": None, "chat_executor": None}


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


def shutdown_executors() -> None:
    shutdown_embedding_executor()
    shutdown_chat_executor()
