from __future__ import annotations

import base64
import io
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard, cast, runtime_checkable

import torch
from PIL import Image

from app.concurrency.limiter import MAX_CONCURRENT
from app.models.base import EmbeddingModel
from app.models.registry import ModelRegistry
from app.threadpool import get_embedding_executor

logger = logging.getLogger(__name__)

type DeviceLike = torch.device | str | int | None
type CudaDeviceLike = torch.device | str

_failed_models: set[str] = set()


@dataclass
class WarmupConfig:
    batch_size: int
    steps: int
    use_inference_mode: bool
    vram_budget_mb: float
    per_worker_vram_mb: float
    texts: list[str]
    executor: ThreadPoolExecutor


@runtime_checkable
class Generates(Protocol):
    name: str
    device: DeviceLike
    capabilities: list[str]

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> Any:
        ...


@runtime_checkable
class RerankModel(Protocol):
    name: str
    device: DeviceLike
    capabilities: list[str]

    def rerank(self, *, query: str, documents: list[str]) -> Any:
        ...


@runtime_checkable
class Embeds(Protocol):
    name: str
    device: DeviceLike
    capabilities: list[str]

    def embed(self, texts: list[str]) -> Any:
        ...


def _should_sync(device: DeviceLike) -> bool:
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
    allowlist = _parse_list_env("WARMUP_ALLOWLIST")
    skiplist = _parse_list_env("WARMUP_SKIPLIST")
    texts = ["hello world"] * batch_size

    executor = get_embedding_executor()

    _failed_models.clear()
    config = WarmupConfig(
        batch_size=batch_size,
        steps=steps,
        use_inference_mode=use_inference_mode,
        vram_budget_mb=vram_budget_mb,
        per_worker_vram_mb=per_worker_vram_mb,
        texts=texts,
        executor=executor,
    )

    for name in registry.list_models():
        if allowlist is not None and name not in allowlist:
            logger.info("warmup_skip", extra={"model": name, "reason": "not_in_allowlist"})
            continue
        if skiplist is not None and name in skiplist:
            logger.info("warmup_skip", extra={"model": name, "reason": "skiplist"})
            continue

        model = registry.get(name)
        capabilities = getattr(model, "capabilities", [])
        device = cast(DeviceLike, getattr(model, "device", None))
        plan = {
            "model": name,
            "steps": steps,
            "batch_size": batch_size,
            "device": str(device),
            "inference_mode": use_inference_mode,
        }
        logger.info("warmup_plan", extra=plan)

        capability_results: dict[str, bool] = {}
        for capability in capabilities:
            warmer = _CAPABILITY_WARMERS.get(capability)
            if warmer is None:
                logger.debug("warmup_noop", extra={"model": name, "capability": capability})
                continue

            ok = warmer(model, device, config)
            capability_results[capability] = ok
            log_extra = {"model": name, "capability": capability}
            if ok:
                logger.info("warmup_ok", extra=log_extra)
            else:
                logger.warning("warmup_failed", extra=log_extra)

        if capability_results and not all(capability_results.values()):
            _failed_models.add(name)

    if _failed_models:
        logger.warning("warmup_failed_models", extra={"models": sorted(_failed_models)})
    else:
        logger.info("warmup_completed")


def _warmup_embedding_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, Embeds):
        logger.warning("warmup_no_embed", extra={"model": getattr(model, "name", "unknown")})
        return False

    typed_model = cast(EmbeddingModel, model)
    workers = max(1, getattr(config.executor, "_max_workers", 1))
    allowed_workers = _select_worker_count(
        device=device,
        executor_workers=workers,
        per_worker_vram_mb=config.per_worker_vram_mb,
        vram_budget_mb=config.vram_budget_mb,
    )

    current_batch = config.batch_size
    current_workers = allowed_workers

    for step in range(config.steps):
        step_complete = False
        while not step_complete:
            start = time.perf_counter()
            try:
                _run_warmup_step(
                    model=typed_model,
                    texts=config.texts[: current_batch],
                    workers=current_workers,
                    use_inference_mode=config.use_inference_mode,
                    executor=config.executor,
                )
                if _should_sync(device):
                    torch.cuda.synchronize(device)
                duration_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "warmup_step_ok",
                    extra={
                        "model": getattr(model, "name", "unknown"),
                        "step": step + 1,
                        "latency_ms": duration_ms,
                        "batch_size": current_batch,
                        "workers": current_workers,
                        "capability": "text-embedding",
                    },
                )
                step_complete = True
            except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - defensive
                torch.cuda.empty_cache()
                current_batch, current_workers, retry = _handle_oom(
                    model_name=getattr(model, "name", "unknown"),
                    batch=current_batch,
                    workers=current_workers,
                    exc=exc,
                )
                if not retry:
                    return False
            except Exception:  # pragma: no cover - startup guardrail
                logger.exception(
                    "warmup_failed",
                    extra={"model": getattr(model, "name", "unknown"), "step": step + 1},
                )
                return False
        if current_workers <= 0:
            break

    return True


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
    device: DeviceLike,
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


def _available_vram_mb(device: DeviceLike) -> float:
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


def _is_cuda_device(device: DeviceLike) -> TypeGuard[CudaDeviceLike]:
    if not torch.cuda.is_available():
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return isinstance(device, str) and device.startswith("cuda")


def _warmup_chat_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, Generates):
        logger.warning(
            "warmup_no_generator", extra={"model": getattr(model, "name", "unknown")}
        )
        return False

    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    context = _inference_context(enabled=config.use_inference_mode)
    try:
        start = time.perf_counter()
        with context:
            model.generate(
                messages,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            )
        if _should_sync(device):
            torch.cuda.synchronize(device)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "warmup_step_ok",
            extra={
                "model": getattr(model, "name", "unknown"),
                "capability": "chat-completion",
                "latency_ms": duration_ms,
            },
        )
        return True
    except Exception:  # pragma: no cover - startup guardrail
        logger.exception(
            "warmup_failed",
            extra={"model": getattr(model, "name", "unknown"), "capability": "chat-completion"},
        )
        return False


def _warmup_vision_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, Generates):
        logger.warning(
            "warmup_no_generator", extra={"model": getattr(model, "name", "unknown")}
        )
        return False

    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{encoded}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "vision warmup"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]
    context = _inference_context(enabled=config.use_inference_mode)
    try:
        start = time.perf_counter()
        with context:
            model.generate(
                messages,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            )
        if _should_sync(device):
            torch.cuda.synchronize(device)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "warmup_step_ok",
            extra={
                "model": getattr(model, "name", "unknown"),
                "capability": "vision",
                "latency_ms": duration_ms,
            },
        )
        return True
    except Exception:  # pragma: no cover - startup guardrail
        logger.exception(
            "warmup_failed",
            extra={"model": getattr(model, "name", "unknown"), "capability": "vision"},
        )
        return False


def _warmup_rerank_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    # Placeholder implementation for rerank-capable models.
    try:
        if not isinstance(model, RerankModel):
            logger.warning("warmup_no_rerank", extra={"model": getattr(model, "name", "unknown")})
            return False
        with _inference_context(enabled=config.use_inference_mode):
            model.rerank(query="warmup query", documents=["doc one", "doc two"])
        if _should_sync(device):
            torch.cuda.synchronize(device)
        return True
    except Exception:  # pragma: no cover - startup guardrail
        logger.exception(
            "warmup_failed",
            extra={"model": getattr(model, "name", "unknown"), "capability": "rerank"},
        )
        return False


def _parse_list_env(var: str) -> set[str] | None:
    raw = os.getenv(var)
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values if values else None


WarmupFn = Callable[[object, DeviceLike, WarmupConfig], bool]


_CAPABILITY_WARMERS: dict[str, WarmupFn] = {
    "text-embedding": _warmup_embedding_model,
    "chat-completion": _warmup_chat_model,
    "vision": _warmup_vision_model,
    "rerank": _warmup_rerank_model,
}


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
