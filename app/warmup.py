from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch

from app.concurrency.limiter import MAX_CONCURRENT
from app.models.base import EmbeddingModel
from app.models.registry import ModelRegistry
from app.threadpool import get_embedding_executor

logger = logging.getLogger(__name__)

_failed_models: set[str] = set()


def _should_sync(device: object) -> bool:
    return torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda"


def warm_up_models(registry: ModelRegistry) -> None:
    """Warm each model (and each worker thread) to smooth first-request latency.

    Runs one batch through every configured model on every executor worker to ensure
    per-thread tokenizers are initialized and CUDA kernels are compiled.
    Records failures instead of aborting startup so operations can inspect health data.
    """

    batch_size = int(os.getenv("WARMUP_BATCH_SIZE", "1"))
    steps = int(os.getenv("WARMUP_STEPS", "1"))
    use_inference_mode = os.getenv("WARMUP_INFERENCE_MODE", "1") != "0"
    vram_budget_mb = float(os.getenv("WARMUP_VRAM_BUDGET_MB", "0"))
    per_worker_vram_mb = float(os.getenv("WARMUP_VRAM_PER_WORKER_MB", "1024"))
    texts = ["hello world"] * batch_size

    executor = get_embedding_executor()
    workers = max(1, getattr(executor, "_max_workers", 1))

    _failed_models.clear()
    for name in registry.list_models():
        model = registry.get(name)
        if "text-embedding" not in getattr(model, "capabilities", []):
            continue
        device = getattr(model, "device", None)
        allowed_workers = _select_worker_count(
            device=device,
            executor_workers=workers,
            per_worker_vram_mb=per_worker_vram_mb,
            vram_budget_mb=vram_budget_mb,
        )
        logger.info(
            "warmup_plan",
            extra={
                "model": name,
                "steps": steps,
                "batch_size": batch_size,
                "workers": allowed_workers,
                "device": str(device),
                "inference_mode": use_inference_mode,
            },
        )

        current_batch = batch_size
        current_workers = allowed_workers
        failed = False

        for step in range(steps):
            step_complete = False
            while not step_complete:
                start = time.perf_counter()
                try:
                    _run_warmup_step(
                        model=model,
                        texts=texts[:current_batch],
                        workers=current_workers,
                        use_inference_mode=use_inference_mode,
                        executor=executor,
                    )
                    if _should_sync(device):
                        torch.cuda.synchronize(device)
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    logger.info(
                        "warmup_ok",
                        extra={
                            "model": name,
                            "step": step + 1,
                            "latency_ms": duration_ms,
                            "batch_size": current_batch,
                            "workers": current_workers,
                        },
                    )
                    step_complete = True
                except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - defensive
                    torch.cuda.empty_cache()
                    current_batch, current_workers, retry = _handle_oom(
                        model_name=name,
                        batch=current_batch,
                        workers=current_workers,
                        exc=exc,
                    )
                    if not retry:
                        _failed_models.add(name)
                        failed = True
                        step_complete = True
                except Exception:  # pragma: no cover - startup guardrail
                    logger.exception("warmup_failed", extra={"model": name, "step": step + 1})
                    _failed_models.add(name)
                    failed = True
                    step_complete = True
            if failed:
                break

    if _failed_models:
        logger.warning("warmup_failed_models", extra={"models": sorted(_failed_models)})
    else:
        logger.info("warmup_completed")


def _run_warmup_step(
    model: EmbeddingModel,
    texts: Iterable[str],
    workers: int,
    use_inference_mode: bool,
    executor: ThreadPoolExecutor,
) -> None:
    context = _inference_context(enabled=use_inference_mode)
    text_batch = list(texts)
    if workers <= 1:
        with context:
            model.embed(text_batch)
        return

    futures = [
        executor.submit(_embed_once, model, text_batch, use_inference_mode)
        for _ in range(workers)
    ]
    wait(futures)
    for fut in futures:
        fut.result()


def _embed_once(model: EmbeddingModel, texts: list[str], use_inference_mode: bool) -> None:
    with _inference_context(enabled=use_inference_mode):
        model.embed(texts)


def _inference_context(enabled: bool) -> AbstractContextManager[None]:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "inference_mode"):
        return cast(AbstractContextManager[None], torch.inference_mode())
    return cast(AbstractContextManager[None], torch.no_grad())


def _select_worker_count(
    device: object | None,
    executor_workers: int,
    per_worker_vram_mb: float,
    vram_budget_mb: float,
) -> int:
    base = max(1, min(executor_workers, MAX_CONCURRENT))
    if not _is_cuda_device(device):
        return base

    available_mb = _available_vram_mb(device)
    budget_mb = vram_budget_mb if vram_budget_mb > 0 else available_mb
    if budget_mb <= 0 or per_worker_vram_mb <= 0:
        return max(1, base)

    allowed_by_budget = max(1, int(budget_mb // per_worker_vram_mb))
    return max(1, min(base, allowed_by_budget))


def _available_vram_mb(device: object | None) -> float:
    if not _is_cuda_device(device):
        return 0.0
    try:
        index = None
        if isinstance(device, torch.device):
            index = device.index
        if isinstance(device, str) and ":" in device:
            _, idx = device.split(":", 1)
            if idx.isdigit():
                index = int(idx)
        if index is None:
            index = torch.cuda.current_device()
        free, _ = torch.cuda.mem_get_info(index)
        return free / (1024 * 1024)
    except Exception:  # pragma: no cover - best-effort
        return 0.0


def _is_cuda_device(device: object | None) -> bool:
    if not torch.cuda.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return isinstance(device, str) and device.startswith("cuda")


def _handle_oom(model_name: str, batch: int, workers: int, exc: Exception) -> tuple[int, int, bool]:
    """Return updated (batch, workers, retry?)."""

    if batch > 1:
        new_batch = max(1, batch // 2)
        logger.warning(
            "warmup_oom_retry",
            extra={"model": model_name, "batch_size": new_batch, "prev_batch": batch},
        )
        return new_batch, 1, True

    if workers > 1:
        logger.warning(
            "warmup_oom_retry",
            extra={"model": model_name, "workers": 1, "prev_workers": workers},
        )
        return batch, 1, True

    logger.error(
        "warmup_oom_give_up",
        extra={"model": model_name, "error": str(exc)},
    )
    return batch, workers, False


def get_failed_warmups() -> list[str]:
    return sorted(_failed_models)
