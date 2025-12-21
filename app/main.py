import asyncio
import importlib.util
import logging
import os
import shutil
import signal
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import huggingface_hub as hf
import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download

from app import state, warmup
from app.api import router as api_router
from app.batching import BatchingService
from app.chat_batching import ChatBatchingService, shutdown_count_executor
from app.concurrency import audio_limiter as audio_limits, limiter
from app.concurrency.audio_limiter import (
    start_accepting as start_accepting_audio,
    stop_accepting as stop_accepting_audio,
    wait_for_drain as wait_for_drain_audio,
)
from app.concurrency.limiter import start_accepting, stop_accepting, wait_for_drain
from app.config import settings
from app.logging_config import setup_logging
from app.middleware import RequestIDMiddleware
from app.model_config import load_model_config
from app.models.registry import ModelRegistry
from app.monitoring.metrics import record_device_memory, setup_metrics
from app.state import WarmupStatus
from app.threadpool import (
    AUDIO_MAX_WORKERS,
    CHAT_MAX_WORKERS,
    EMBEDDING_MAX_WORKERS,
    VISION_MAX_WORKERS,
    get_audio_executor,
    get_chat_executor,
    get_embedding_count_executor,
    get_embedding_executor,
    get_vision_executor,
    shutdown_executors,
)
from app.warmup import warm_up_models

logger = logging.getLogger(__name__)

# Mapping from capability to executor kind for thread-safety enforcement
_CAPABILITY_EXECUTOR_MAP: dict[str, str] = {
    "audio-transcription": "audio",
    "audio-translation": "audio",
    "vision": "vision",
    "chat-completion": "chat",
    "text-embedding": "embedding",
}


async def _run_in_daemon_thread[T](func: Callable[[], T], *, name: str) -> T:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[T] = loop.create_future()

    def _set_result(result: T) -> None:
        if fut.done():
            return
        fut.set_result(result)

    def _set_exc(exc: BaseException) -> None:
        if fut.done():
            return
        fut.set_exception(exc)

    def _target() -> None:
        try:
            result = func()
        except BaseException as exc:  # noqa: BLE001 - propagate SystemExit/KeyboardInterrupt too
            try:
                loop.call_soon_threadsafe(_set_exc, exc)
            except RuntimeError:
                return
        else:
            try:
                loop.call_soon_threadsafe(_set_result, result)
            except RuntimeError:
                return

    threading.Thread(target=_target, name=name, daemon=True).start()
    return await fut


async def _startup_in_background() -> tuple[ModelRegistry, BatchingService, ChatBatchingService]:
    sigint_handler: Any | None = None
    if threading.current_thread() is threading.main_thread():
        sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return await _run_in_daemon_thread(startup, name="model-startup")
    finally:
        if sigint_handler is not None:
            signal.signal(signal.SIGINT, sigint_handler)


def _load_model_config() -> tuple[str, list[str] | None, str | None]:
    """Load base configuration paths and allowlists."""
    # Prefer local ./models; if HF_HOME already set, keep it for fallback use
    hf_home = settings.hf_home or str(Path.cwd() / "models")
    os.environ.setdefault("HF_HOME", hf_home)

    config_path = settings.model_config_path
    model_allowlist = [m.strip() for m in settings.models.split(",") if m.strip()] or None

    if model_allowlist is None:
        raise SystemExit("No models specified. Set MODELS env (comma-separated) before starting the service.")
    device_override = settings.model_device if settings.model_device != "auto" else None
    return config_path, model_allowlist, device_override


def _init_batching(registry: ModelRegistry) -> tuple[BatchingService, ChatBatchingService, dict[str, Any]]:
    """Initialize batching services and return runtime config components."""
    batching_service = BatchingService(
        registry,
        enabled=settings.enable_embedding_batching,
        max_batch_size=settings.effective_embedding_batch_max_size,
        window_ms=settings.embedding_batch_window_ms,
        queue_size=settings.effective_embedding_batch_queue_size,
    )

    chat_batching_service = ChatBatchingService(
        registry,
        enabled=settings.enable_chat_batching,
        max_batch_size=settings.chat_batch_max_size,
        window_ms=settings.chat_batch_window_ms,
        max_prompt_tokens=settings.chat_max_prompt_tokens,
        max_new_tokens_ceiling=settings.chat_max_new_tokens,
        queue_size=settings.chat_batch_queue_size,
        allow_vision=settings.chat_batch_allow_vision,
    )

    # Partial runtime config derived from batching settings
    config = {
        "max_batch_size": settings.max_batch_size,
        "max_text_chars": settings.max_text_chars,
        "enable_batching": settings.enable_embedding_batching,
        "batch_window_ms": settings.embedding_batch_window_ms,
        "batch_max_size": settings.effective_embedding_batch_max_size,
        "embedding_batch_queue_size": settings.effective_embedding_batch_queue_size,
        "enable_chat_batching": settings.enable_chat_batching,
        "chat_batch_window_ms": settings.chat_batch_window_ms,
        "chat_batch_max_size": settings.chat_batch_max_size,
        "chat_max_prompt_tokens": settings.chat_max_prompt_tokens,
        "chat_max_new_tokens": settings.chat_max_new_tokens,
        "chat_batch_queue_size": settings.chat_batch_queue_size,
        "chat_queue_max_wait_ms": settings.chat_queue_max_wait_ms,
        "chat_requeue_max_wait_ms": settings.chat_requeue_max_wait_ms,
        "chat_requeue_max_tasks": settings.chat_requeue_max_tasks,
        "embedding_generate_timeout_sec": settings.embedding_generate_timeout_sec,
    }
    return batching_service, chat_batching_service, config


def startup() -> tuple[ModelRegistry, BatchingService, ChatBatchingService]:
    setup_logging()
    start_accepting()
    start_accepting_audio()

    config_path, model_allowlist, device_override = _load_model_config()
    cache_dir = os.environ["HF_HOME"]

    _download_models_if_enabled(config_path, model_allowlist, cache_dir)
    _warn_if_accelerate_missing(config_path, model_allowlist)

    try:
        registry = ModelRegistry(
            config_path,
            device=device_override,
            allowed_models=model_allowlist,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load models during startup", extra={"config_path": config_path})
        raise SystemExit(1) from exc

    _enforce_thread_safety(registry)
    _validate_ffmpeg_for_audio(registry)

    batching_service, chat_batching_service, batch_cfg = _init_batching(registry)

    state.batching_service = batching_service
    state.chat_batching_service = chat_batching_service

    # Build full runtime config
    embedding_executor = get_embedding_executor()
    embedding_count_executor = get_embedding_count_executor()
    chat_executor = get_chat_executor()
    vision_executor = get_vision_executor()
    audio_executor = get_audio_executor()

    runtime_cfg = {
        "model_config": config_path,
        "model_device": device_override or "auto",
        "model_allowlist": model_allowlist,
        "max_concurrent": limiter.MAX_CONCURRENT,
        "max_queue_size": limiter.MAX_QUEUE_SIZE,
        "queue_timeout_sec": limiter.QUEUE_TIMEOUT_SEC,
        "embedding_max_concurrent": limiter.EMBEDDING_MAX_CONCURRENT,
        "embedding_max_queue_size": limiter.EMBEDDING_MAX_QUEUE_SIZE,
        "embedding_queue_timeout_sec": limiter.EMBEDDING_QUEUE_TIMEOUT_SEC,
        "chat_max_concurrent": limiter.CHAT_MAX_CONCURRENT,
        "chat_max_queue_size": limiter.CHAT_MAX_QUEUE_SIZE,
        "chat_queue_timeout_sec": limiter.CHAT_QUEUE_TIMEOUT_SEC,
        "vision_max_concurrent": limiter.VISION_MAX_CONCURRENT,
        "vision_max_queue_size": limiter.VISION_MAX_QUEUE_SIZE,
        "vision_queue_timeout_sec": limiter.VISION_QUEUE_TIMEOUT_SEC,
        "embedding_max_workers": getattr(embedding_executor, "_max_workers", EMBEDDING_MAX_WORKERS),
        "embedding_count_max_workers": getattr(embedding_count_executor, "_max_workers", None),
        "chat_max_workers": getattr(chat_executor, "_max_workers", CHAT_MAX_WORKERS),
        "vision_max_workers": getattr(vision_executor, "_max_workers", VISION_MAX_WORKERS),
        "audio_max_workers": getattr(audio_executor, "_max_workers", AUDIO_MAX_WORKERS),
        "audio_max_concurrent": audio_limits.MAX_CONCURRENT,
        "audio_max_queue_size": audio_limits.MAX_QUEUE_SIZE,
        "audio_queue_timeout_sec": audio_limits.QUEUE_TIMEOUT_SEC,
        "audio_process_timeout_sec": settings.audio_process_timeout_sec,
        **batch_cfg,
    }

    logger.info(
        "Loaded models",
        extra={
            "models": registry.list_models(),
            "device": registry.device,
            "devices": {name: getattr(model, "device", "unknown") for name, model in registry.models.items()},
        },
    )
    logger.info("runtime_config", extra=runtime_cfg)

    warmup_required = settings.enable_warmup
    warmup_failures: list[str] = []
    warmup_completed = not warmup_required
    if warmup_required:
        warmup_failures = warm_up_models(registry)
        warmup_completed = True
        if warmup_failures:
            failed_list = ", ".join(sorted(warmup_failures))
            raise SystemExit(f"Warmup failed for model(s): {failed_list}")

    failed_set = set(warmup_failures)
    ok_models = [name for name in registry.list_models() if name not in failed_set]
    warmup_status = WarmupStatus(
        required=warmup_required,
        completed=warmup_completed,
        failures=warmup_failures,
        ok_models=ok_models,
        capabilities=warmup.get_capability_status(),
    )
    state.warmup_status = warmup_status
    state.runtime_config = runtime_cfg

    record_device_memory(registry.device)

    return registry, batching_service, chat_batching_service


def _warn_if_accelerate_missing(config_path: str, allowlist: list[str] | None) -> None:
    """Pre-flight warning if FP8 models are configured but accelerate is absent."""

    try:
        cfg = load_model_config(config_path)
    except Exception:  # pragma: no cover - non-critical
        return

    requested = set(allowlist) if allowlist else None
    fp8_models: list[str] = []
    for item in cfg.get("models", []):
        name = item.get("name") or item.get("hf_repo_id")
        if not name:
            continue
        if requested is not None and name not in requested:
            continue
        repo = str(item.get("hf_repo_id", "")).lower()
        if "fp8" in repo:
            fp8_models.append(name)

    if fp8_models:
        if importlib.util.find_spec("accelerate") is None:
            raise SystemExit(f"FP8 models {fp8_models} require 'accelerate'. Install it or select non-FP8 repos.")
        if not torch.cuda.is_available() and not getattr(torch, "xpu", None):
            raise SystemExit(
                f"FP8 models {fp8_models} require a GPU/XPU runtime. Select non-FP8 variants or run on GPU/XPU."
            )


def _enforce_thread_safety(registry: ModelRegistry) -> None:
    """Warn when thread-unsafe models may be invoked concurrently.

    Handlers can mark themselves thread-unsafe (thread_safe=False) to signal they
    must serialize internal state (tokenizers, pipelines, clients). We do not
    globally force executors down to a single worker because that would
    penalize unrelated models sharing the same executor.
    """

    executor_workers = {
        "embedding": getattr(get_embedding_executor(), "_max_workers", EMBEDDING_MAX_WORKERS),
        "chat": getattr(get_chat_executor(), "_max_workers", CHAT_MAX_WORKERS),
        "vision": getattr(get_vision_executor(), "_max_workers", VISION_MAX_WORKERS),
        "audio": getattr(get_audio_executor(), "_max_workers", AUDIO_MAX_WORKERS),
    }
    for name in registry.list_models():
        model = registry.get(name)
        if getattr(model, "thread_safe", True):
            continue

        caps = getattr(model, "capabilities", [])
        for cap in caps:
            executor_kind = _CAPABILITY_EXECUTOR_MAP.get(cap)
            if executor_kind is None:
                continue
            workers = executor_workers.get(executor_kind, 1)
            if workers <= 1:
                continue
            logger.warning(
                "thread_unsafe_model_with_multiple_workers",
                extra={
                    "model": name,
                    "capability": cap,
                    "executor": executor_kind,
                    "workers": workers,
                },
            )


def _validate_ffmpeg_for_audio(registry: ModelRegistry) -> None:
    """Ensure ffmpeg is available when audio/Whisper models are loaded."""

    has_audio_model = any("audio-transcription" in getattr(m, "capabilities", []) for m in registry.models.values())
    if not has_audio_model:
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Whisper/audio models require ffmpeg for decoding. Install ffmpeg and restart."
        )


def _download_models_if_enabled(config_path: str, allowlist: list[str] | None, cache_dir: str | None) -> None:
    """Optionally download requested models before startup; exit on failure."""

    if not settings.auto_download_models:
        logger.info("Auto download disabled; assuming models are pre-fetched")
        return
    if not hasattr(hf, "snapshot_download"):
        raise SystemExit("huggingface_hub is required for auto download")

    try:
        cfg = load_model_config(config_path)
    except Exception as exc:  # pragma: no cover - startup guardrail
        raise SystemExit(f"Failed to read model config at {config_path}") from exc

    target_dir = Path(cache_dir) if cache_dir else Path.cwd() / "models"
    target_dir.mkdir(parents=True, exist_ok=True)

    requested = set(allowlist) if allowlist else None
    downloaded: list[str] = []
    for item in cfg.get("models", []):
        repo_id = item.get("hf_repo_id")
        name = item.get("name") or repo_id
        handler = item.get("handler")
        if not repo_id or not handler:
            raise SystemExit("Each model requires 'hf_repo_id' and 'handler' in config")
        if requested is not None and name not in requested:
            continue
        logger.info("Downloading model %s (%s) to %s", name, repo_id, target_dir)
        try:
            snapshot_download(repo_id=repo_id, cache_dir=target_dir, local_dir_use_symlinks=False)
            downloaded.append(name)
        except Exception as exc:  # pragma: no cover - network/runtime failure
            raise SystemExit(f"Failed to download model {name} ({repo_id})") from exc

    if requested is not None:
        missing = requested - set(downloaded)
        if missing:
            raise SystemExit(f"Requested model(s) not found in config: {', '.join(sorted(missing))}")


async def shutdown(
    batching_service: BatchingService | None,
    chat_batching_service: ChatBatchingService | None,
    registry: ModelRegistry | None = None,
) -> None:
    stop_accepting()
    stop_accepting_audio()
    await wait_for_drain()
    await wait_for_drain_audio()
    if batching_service is not None:
        await batching_service.stop()
        state.batching_service = None
    if chat_batching_service is not None:
        await chat_batching_service.stop()
        state.chat_batching_service = None
    if registry is not None:
        for model in registry.models.values():
            close_fn = getattr(model, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:  # pragma: no cover - best-effort shutdown
                    logger.warning("model_close_failed", extra={"model": getattr(model, "name", "unknown")})
    state.warmup_status = WarmupStatus()
    shutdown_count_executor()
    shutdown_executors()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = None
    batching_service = None
    chat_batching_service = None
    try:
        registry, batching_service, chat_batching_service = await _startup_in_background()
        app.state.model_registry = registry
        app.state.batching_service = batching_service
        app.state.chat_batching_service = chat_batching_service
        app.state.warmup_status = state.warmup_status
        app.state.runtime_config = state.runtime_config
        setup_metrics(app)

        # Start batcher workers before accepting requests.
        # This ensures the async worker tasks are running and ready,
        # avoiding lazy initialization latency on the first request.
        if batching_service is not None:
            await batching_service.start()
            logger.info("embedding_batchers_started")
        if chat_batching_service is not None:
            await chat_batching_service.start()
            logger.info("chat_batchers_started")

        yield
    finally:
        await shutdown(batching_service, chat_batching_service, registry)


app = FastAPI(title="Inference Service", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)
