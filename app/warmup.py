from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeGuard, cast, runtime_checkable

import torch
from PIL import Image

from app.concurrency.limiter import MAX_CONCURRENT
from app.models.base import EmbeddingModel, SpeechModel
from app.models.registry import ModelRegistry
from app.monitoring.metrics import record_warmup_pool_ready
from app.threadpool import (
    get_audio_executor,
    get_chat_executor,
    get_embedding_executor,
    get_vision_executor,
)

logger = logging.getLogger(__name__)

type DeviceLike = torch.device | str | int | None
type CudaDeviceLike = torch.device | str

# Module-level state for warmup status, protected by _warmup_lock
_warmup_lock = threading.Lock()
_failed_models: set[str] = set()
_capability_status: dict[str, dict[str, bool]] = {}


@dataclass
class WarmupConfig:
    batch_size: int
    steps: int
    use_inference_mode: bool
    vram_budget_mb: float
    per_worker_vram_mb: float
    texts: list[str]
    executors: dict[str, ThreadPoolExecutor]


@dataclass(frozen=True, slots=True)
class WarmupContext:
    model_name: str
    capability: str
    device: DeviceLike
    config: WarmupConfig
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
class Embeds(Protocol):
    name: str
    device: DeviceLike
    capabilities: list[str]

    def embed(self, texts: list[str]) -> Any:
        ...


def _should_sync(device: DeviceLike) -> bool:
    return torch.cuda.is_available() and isinstance(device, torch.device) and device.type == "cuda"


def _make_silence_wav(duration_sec: float = 0.2, sample_rate: int = 16000) -> str:
    frames = max(1, int(sample_rate * duration_sec))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * frames)
        return tmp.name


def _build_warmup_config() -> tuple[WarmupConfig, set[str] | None, set[str] | None]:
    """Build warmup configuration from environment variables."""
    batch_size = int(os.getenv("WARMUP_BATCH_SIZE", "1"))
    steps = int(os.getenv("WARMUP_STEPS", "1"))
    use_inference_mode = os.getenv("WARMUP_INFERENCE_MODE", "1") != "0"
    vram_budget_mb = float(os.getenv("WARMUP_VRAM_BUDGET_MB", "0"))
    per_worker_vram_mb = float(os.getenv("WARMUP_VRAM_PER_WORKER_MB", "1024"))
    allowlist = _parse_list_env("WARMUP_ALLOWLIST")
    skiplist = _parse_list_env("WARMUP_SKIPLIST")
    texts = ["hello world"] * batch_size

    executors = {
        "text-embedding": get_embedding_executor(),
        "chat-completion": get_chat_executor(),
        "vision": get_vision_executor(),
        "audio": get_audio_executor(),
    }

    config = WarmupConfig(
        batch_size=batch_size,
        steps=steps,
        use_inference_mode=use_inference_mode,
        vram_budget_mb=vram_budget_mb,
        per_worker_vram_mb=per_worker_vram_mb,
        texts=texts,
        executors=executors,
    )
    return config, allowlist, skiplist


def _warmup_single_model(
    name: str,
    model: object,
    config: WarmupConfig,
) -> dict[str, bool]:
    """Run warmup for a single model across all its capabilities."""
    capabilities = getattr(model, "capabilities", [])
    device = cast(DeviceLike, getattr(model, "device", None))
    logger.info(
        "warmup_plan",
        extra={
            "model": name,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "device": str(device),
            "inference_mode": config.use_inference_mode,
        },
    )

    capability_results: dict[str, bool] = {}
    for capability in capabilities:
        warmer = _CAPABILITY_WARMERS.get(capability)
        if warmer is None:
            logger.debug("warmup_noop", extra={"model": name, "capability": capability})
            continue

        ok = warmer(model, device, config)
        capability_results[capability] = ok
        executor = config.executors.get(capability)
        log_extra: dict[str, str] = {"model": name, "capability": capability}
        if executor is not None:
            log_extra["executor"] = _executor_label(executor)
        if ok:
            logger.info("warmup_ok", extra=log_extra)
        else:
            logger.warning("warmup_failed", extra=log_extra)

    return capability_results


def warm_up_models(registry: ModelRegistry) -> list[str]:
    """Warm each model (and each worker thread) to smooth first-request latency.

    Runs one batch through every configured model on every executor worker to ensure
    per-thread tokenizers are initialized and CUDA kernels are compiled.
    Records failures instead of aborting startup so operations can inspect health data.
    """
    config, allowlist, skiplist = _build_warmup_config()

    with _warmup_lock:
        _failed_models.clear()
        _capability_status.clear()

    for name in registry.list_models():
        if allowlist is not None and name not in allowlist:
            logger.info("warmup_skip", extra={"model": name, "reason": "not_in_allowlist"})
            continue
        if skiplist is not None and name in skiplist:
            logger.info("warmup_skip", extra={"model": name, "reason": "skiplist"})
            continue

        model = registry.get(name)
        capability_results = _warmup_single_model(name, model, config)

        with _warmup_lock:
            if capability_results:
                _capability_status[name] = capability_results
            if capability_results and not all(capability_results.values()):
                _failed_models.add(name)

    with _warmup_lock:
        has_failures = bool(_failed_models)
        failed_list = sorted(_failed_models) if has_failures else []

    if has_failures:
        logger.warning("warmup_failed_models", extra={"models": failed_list})
    else:
        logger.info("warmup_completed")

    return get_failed_warmups()


def _warmup_embedding_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, Embeds):
        _record_pool_readiness(
            model_name=getattr(model, "name", "unknown"),
            capability="text-embedding",
            executor=_executor_for_capability("text-embedding", config),
            workers=0,
            ready=False,
        )
        logger.warning("warmup_no_embed", extra={"model": getattr(model, "name", "unknown")})
        return False

    executor = _executor_for_capability("text-embedding", config)
    typed_model = cast(EmbeddingModel, model)

    context = WarmupContext(
        model_name=getattr(model, "name", "unknown"),
        capability="text-embedding",
        device=device,
        config=config,
        executor=executor,
    )

    def _run_once() -> None:
        with _inference_context(enabled=config.use_inference_mode):
            typed_model.embed(config.texts[: config.batch_size])

    # Reuse the generic warmup harness so worker count and VRAM budgeting are
    # handled consistently with other capabilities.
    return _warmup_with_executor(
        context=context,
        run_once=_run_once,
        step_extra={"batch_size": config.batch_size},
    )


def _inference_context(enabled: bool) -> AbstractContextManager[None]:
    if not enabled:
        return nullcontext()
    if hasattr(torch, "inference_mode"):
        return cast(AbstractContextManager[None], torch.inference_mode())
    return cast(AbstractContextManager[None], torch.no_grad())


def _executor_for_capability(capability: str, config: WarmupConfig) -> ThreadPoolExecutor:
    executor = config.executors.get(capability)
    if executor is None:
        if capability.startswith("audio"):
            fallback = config.executors.get("audio")
            if fallback is not None:
                config.executors[capability] = fallback
                return fallback
        fallback = config.executors.get("text-embedding") or get_embedding_executor()
        config.executors[capability] = fallback
        return fallback
    return executor


def _executor_label(executor: ThreadPoolExecutor) -> str:
    return getattr(executor, "_thread_name_prefix", "executor")


def _executor_workers(executor: ThreadPoolExecutor) -> int:
    return max(1, getattr(executor, "_max_workers", 1))


def _run_worker_tasks(executor: ThreadPoolExecutor, workers: int, func: Callable[[], None]) -> None:
    if workers <= 1:
        func()
        return
    futures = [executor.submit(func) for _ in range(workers)]
    wait(futures)
    for fut in futures:
        fut.result()


def _record_pool_readiness(
    *,
    model_name: str,
    capability: str,
    executor: ThreadPoolExecutor,
    workers: int,
    ready: bool,
) -> None:
    status = "warmup_pool_ready" if ready else "warmup_pool_unready"
    ready_workers = workers if ready else 0
    log_extra = {
        "model": model_name,
        "capability": capability,
        "executor": _executor_label(executor),
        "ready_workers": ready_workers,
    }
    if ready:
        logger.info(status, extra=log_extra)
    else:
        logger.warning(status, extra=log_extra)
    record_warmup_pool_ready(
        model=model_name,
        capability=capability,
        executor=_executor_label(executor),
        workers=ready_workers,
    )


def _warmup_with_executor(
    *,
    context: WarmupContext,
    run_once: Callable[[], None],
    step_extra: dict[str, float | int | str] | None = None,
) -> bool:
    workers = _select_worker_count(
        device=context.device,
        executor_workers=_executor_workers(context.executor),
        per_worker_vram_mb=context.config.per_worker_vram_mb,
        vram_budget_mb=context.config.vram_budget_mb,
    )

    for step in range(context.config.steps):
        start = time.perf_counter()
        try:
            _run_worker_tasks(context.executor, workers, run_once)
            if _should_sync(context.device):
                torch.cuda.synchronize(context.device)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log_extra = {
                "model": context.model_name,
                "capability": context.capability,
                "executor": _executor_label(context.executor),
                "latency_ms": duration_ms,
                "step": step + 1,
                "workers": workers,
            }
            if step_extra:
                log_extra.update(step_extra)
            logger.info("warmup_step_ok", extra=log_extra)
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - defensive
            # Warmup failures are treated as a startup guardrail; callers decide
            # whether to fail-fast the process or surface partial readiness in
            # health checks.
            logger.exception(
                "warmup_oom",
                extra={
                    "model": context.model_name,
                    "capability": context.capability,
                    "executor": _executor_label(context.executor),
                    "step": step + 1,
                },
            )
            raise
        except Exception:  # pragma: no cover - startup guardrail
            logger.exception(
                "warmup_failed",
                extra={
                    "model": context.model_name,
                    "capability": context.capability,
                    "executor": _executor_label(context.executor),
                    "step": step + 1,
                },
            )
            _record_pool_readiness(
                model_name=context.model_name,
                capability=context.capability,
                executor=context.executor,
                workers=workers,
                ready=False,
            )
            return False

    _record_pool_readiness(
        model_name=context.model_name,
        capability=context.capability,
        executor=context.executor,
        workers=workers,
        ready=True,
    )
    return True


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
        _record_pool_readiness(
            model_name=getattr(model, "name", "unknown"),
            capability="chat-completion",
            executor=_executor_for_capability("chat-completion", config),
            workers=0,
            ready=False,
        )
        logger.warning(
            "warmup_no_generator", extra={"model": getattr(model, "name", "unknown")}
        )
        return False

    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    executor = _executor_for_capability("chat-completion", config)

    def _generate_once() -> None:
        with _inference_context(enabled=config.use_inference_mode):
            model.generate(
                messages,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            )

    context = WarmupContext(
        model_name=getattr(model, "name", "unknown"),
        capability="chat-completion",
        device=device,
        config=config,
        executor=executor,
    )

    return _warmup_with_executor(
        context=context,
        run_once=_generate_once,
    )


def _warmup_audio_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, SpeechModel):
        _record_pool_readiness(
            model_name=getattr(model, "name", "unknown"),
            capability="audio-transcription",
            executor=_executor_for_capability("audio", config),
            workers=0,
            ready=False,
        )
        logger.warning(
            "warmup_no_audio", extra={"model": getattr(model, "name", "unknown")}
        )
        return False

    sample_path = _make_silence_wav()
    executor = _executor_for_capability("audio", config)

    def _run_once() -> None:
        model.transcribe(
            sample_path,
            language="en",
            prompt=None,
            temperature=0.0,
            task="transcribe",
            timestamp_granularity=None,
            cancel_event=None,
        )

    context = WarmupContext(
        model_name=getattr(model, "name", "unknown"),
        capability="audio-transcription",
        device=device,
        config=config,
        executor=executor,
    )

    try:
        return _warmup_with_executor(
            context=context,
            run_once=_run_once,
        )
    finally:
        Path(sample_path).unlink(missing_ok=True)


def _warmup_vision_model(model: object, device: DeviceLike, config: WarmupConfig) -> bool:
    if not isinstance(model, Generates):
        _record_pool_readiness(
            model_name=getattr(model, "name", "unknown"),
            capability="vision",
            executor=_executor_for_capability("vision", config),
            workers=0,
            ready=False,
        )
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
    executor = _executor_for_capability("vision", config)

    def _generate_once() -> None:
        with _inference_context(enabled=config.use_inference_mode):
            model.generate(
                messages,
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            )

    context = WarmupContext(
        model_name=getattr(model, "name", "unknown"),
        capability="vision",
        device=device,
        config=config,
        executor=executor,
    )

    return _warmup_with_executor(
        context=context,
        run_once=_generate_once,
    )


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
    "audio-transcription": _warmup_audio_model,
    "audio-translation": _warmup_audio_model,
}


def get_failed_warmups() -> list[str]:
    with _warmup_lock:
        return sorted(_failed_models)


def get_capability_status() -> dict[str, dict[str, bool]]:
    with _warmup_lock:
        return {model: caps.copy() for model, caps in _capability_status.items()}
