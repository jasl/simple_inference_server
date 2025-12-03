from __future__ import annotations

import logging
import os
import time
from concurrent.futures import wait

import torch

from app.models.registry import ModelRegistry
from app.threadpool import get_embedding_executor

logger = logging.getLogger(__name__)


def _should_sync(device: object) -> bool:
    return torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda"


def warm_up_models(registry: ModelRegistry) -> None:
    """Warm each model (and each worker thread) to smooth first-request latency.

    Runs one batch through every configured model on every executor worker to ensure
    per-thread tokenizers are initialized and CUDA kernels are compiled.
    Raises on failure so startup can abort rather than serving a broken model.
    """

    batch_size = int(os.getenv("WARMUP_BATCH_SIZE", "1"))
    steps = int(os.getenv("WARMUP_STEPS", "1"))
    texts = ["hello world"] * batch_size

    executor = get_embedding_executor()
    workers = max(1, getattr(executor, "_max_workers", 1))

    for name in registry.list_models():
        model = registry.get(name)
        if "text-embedding" not in getattr(model, "capabilities", []):
            continue
        device = getattr(model, "device", None)

        try:
            for step in range(steps):
                start = time.perf_counter()
                futures = [executor.submit(model.embed, texts) for _ in range(workers)]
                wait(futures)
                if _should_sync(device):
                    torch.cuda.synchronize(device)
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "warmup_ok",
                    extra={
                        "model": name,
                        "step": step + 1,
                        "latency_ms": duration_ms,
                        "batch_size": batch_size,
                        "workers": workers,
                    },
                )
        except Exception:  # pragma: no cover - startup guardrail
            logger.exception("warmup_failed", extra={"model": name})
            raise
