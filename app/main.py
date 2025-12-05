import importlib.util
import logging
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import huggingface_hub as hf
import torch
import yaml
from fastapi import FastAPI
from huggingface_hub import snapshot_download

from app import state, warmup
from app.api import router as api_router
from app.batching import BatchingService
from app.chat_batching import ChatBatchingService, shutdown_count_executor
from app.concurrency import audio_limiter as audio_limits, limiter
from app.concurrency.audio_limiter import stop_accepting as stop_accepting_audio, wait_for_drain as wait_for_drain_audio
from app.concurrency.limiter import stop_accepting, wait_for_drain
from app.logging_config import setup_logging
from app.models.registry import ModelRegistry
from app.monitoring.metrics import setup_metrics
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


def startup() -> tuple[ModelRegistry, BatchingService, ChatBatchingService]:  # noqa: PLR0915
    setup_logging()

    # Prefer local ./models; if HF_HOME already set, keep it for fallback use
    os.environ.setdefault("HF_HOME", str(Path.cwd() / "models"))
    cache_dir = os.environ["HF_HOME"]

    config_path = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
    device_override = os.getenv("MODEL_DEVICE") or None
    models_env = os.getenv("MODELS")
    model_allowlist = [m.strip() for m in (models_env or "").split(",") if m.strip()] or None
    if model_allowlist is None:
        raise SystemExit(
            "No models specified. Set MODELS env (comma-separated) before starting the service."
        )
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
        # Hard exit to avoid serving traffic with missing/bad models
        raise SystemExit(1) from exc
    _warn_thread_unsafe_models(registry)
    _validate_ffmpeg_for_audio(registry)

    batching_enabled = os.getenv("ENABLE_EMBEDDING_BATCHING", "1") != "0"
    batch_window_ms = float(os.getenv("EMBEDDING_BATCH_WINDOW_MS", "6"))
    batch_max_size = int(
        os.getenv("EMBEDDING_BATCH_WINDOW_MAX_SIZE", os.getenv("MAX_BATCH_SIZE", "32"))
    )
    batch_queue_size = int(os.getenv("EMBEDDING_BATCH_QUEUE_SIZE", os.getenv("MAX_QUEUE_SIZE", "64")))
    batching_service = BatchingService(
        registry,
        enabled=batching_enabled,
        max_batch_size=batch_max_size,
        window_ms=batch_window_ms,
        queue_size=batch_queue_size,
    )
    chat_batching_enabled = os.getenv("ENABLE_CHAT_BATCHING", "1") != "0"
    chat_batch_window_ms = float(os.getenv("CHAT_BATCH_WINDOW_MS", "10"))
    chat_batch_max_size = int(os.getenv("CHAT_BATCH_MAX_SIZE", "8"))
    chat_max_prompt_tokens = int(os.getenv("CHAT_MAX_PROMPT_TOKENS", "4096"))
    chat_max_new_tokens = int(os.getenv("CHAT_MAX_NEW_TOKENS", "2048"))
    chat_batch_queue_size = int(os.getenv("CHAT_BATCH_QUEUE_SIZE", "64"))
    chat_allow_vision = os.getenv("CHAT_BATCH_ALLOW_VISION", "0") != "0"
    chat_batching_service = ChatBatchingService(
        registry,
        enabled=chat_batching_enabled,
        max_batch_size=chat_batch_max_size,
        window_ms=chat_batch_window_ms,
        max_prompt_tokens=chat_max_prompt_tokens,
        max_new_tokens_ceiling=chat_max_new_tokens,
        queue_size=chat_batch_queue_size,
        allow_vision=chat_allow_vision,
    )

    # Record process-wide services for non-request contexts (e.g., warmup status).
    state.batching_service = batching_service
    state.chat_batching_service = chat_batching_service

    runtime_cfg = _build_runtime_config(
        config_path=config_path,
        device_override=device_override,
        model_allowlist=model_allowlist,
        batching_enabled=batching_enabled,
        batch_window_ms=batch_window_ms,
        batch_max_size=batch_max_size,
        batch_queue_size=batch_queue_size,
        chat_batching_enabled=chat_batching_enabled,
        chat_batch_window_ms=chat_batch_window_ms,
        chat_batch_max_size=chat_batch_max_size,
        chat_max_prompt_tokens=chat_max_prompt_tokens,
        chat_max_new_tokens=chat_max_new_tokens,
        chat_batch_queue_size=chat_batch_queue_size,
    )
    logger.info(
        "Loaded models",
        extra={
            "models": registry.list_models(),
            "device": registry.device,
            "devices": {name: getattr(model, "device", "unknown") for name, model in registry.models.items()},
        },
    )
    logger.info("runtime_config", extra=runtime_cfg)

    warmup_required = os.getenv("ENABLE_WARMUP", "1") != "0"
    warmup_failures: list[str] = []
    warmup_completed = not warmup_required
    if warmup_required:
        warmup_failures = warm_up_models(registry)
        warmup_completed = True
        if warmup_failures:
            failed_list = ", ".join(sorted(warmup_failures))
            # Fail-fast is the only behavior when warmup is enabled: if startup
            # cannot successfully exercise the loaded models, serving traffic is
            # likely to fail as well.
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

    return registry, batching_service, chat_batching_service


def _warn_if_accelerate_missing(config_path: str, allowlist: list[str] | None) -> None:
    """Pre-flight warning if FP8 models are configured but accelerate is absent."""

    try:
        with Path(config_path).open() as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:  # pragma: no cover - non-critical
        return

    requested = set(allowlist) if allowlist else None
    fp8_models: list[str] = []
    for item in cfg.get("models", []):
        name = item.get("name")
        if not name:
            continue
        if requested is not None and name not in requested:
            continue
        repo = str(item.get("hf_repo_id", "")).lower()
        if "fp8" in repo:
            fp8_models.append(name)

    if fp8_models:
        if importlib.util.find_spec("accelerate") is None:
            raise SystemExit(
                f"FP8 models {fp8_models} require 'accelerate'. Install it or select non-FP8 repos."
            )
        if not torch.cuda.is_available() and not getattr(torch, "xpu", None):
            raise SystemExit(
                f"FP8 models {fp8_models} require a GPU/XPU runtime. Select non-FP8 variants or run on GPU/XPU."
            )


def _warn_thread_unsafe_models(registry: ModelRegistry) -> None:
    """Warn if non-thread-safe handlers are paired with worker>1 executors."""

    for name in registry.list_models():
        model = registry.get(name)
        thread_safe = getattr(model, "thread_safe", True)
        if thread_safe:
            continue

        caps = getattr(model, "capabilities", [])
        warn = False
        workers = None
        if "audio-transcription" in caps or "audio-translation" in caps:
            warn = AUDIO_MAX_WORKERS > 1
            workers = AUDIO_MAX_WORKERS
        elif "vision" in caps:
            warn = VISION_MAX_WORKERS > 1
            workers = VISION_MAX_WORKERS
        elif "chat-completion" in caps:
            warn = CHAT_MAX_WORKERS > 1
            workers = CHAT_MAX_WORKERS
        elif "text-embedding" in caps:
            warn = EMBEDDING_MAX_WORKERS > 1
            workers = EMBEDDING_MAX_WORKERS

        if warn:
            logger.warning(
                "thread_unsafe_model_with_multiple_workers",
                extra={"model": name, "capabilities": caps, "workers": workers},
            )


def _build_runtime_config(  # noqa: PLR0913
    *,
    config_path: str,
    device_override: str | None,
    model_allowlist: list[str] | None,
    batching_enabled: bool,
    batch_window_ms: float,
    batch_max_size: int,
    batch_queue_size: int,
    chat_batching_enabled: bool,
    chat_batch_window_ms: float,
    chat_batch_max_size: int,
    chat_max_prompt_tokens: int,
    chat_max_new_tokens: int,
    chat_batch_queue_size: int,
) -> dict[str, Any]:
    """Assemble a snapshot of the runtime configuration from environment and defaults."""

    # Eagerly initialize shared executors so thread pools and their worker counts
    # are fixed during startup and visible in the runtime configuration snapshot.
    embedding_executor = get_embedding_executor()
    embedding_count_executor = get_embedding_count_executor()
    chat_executor = get_chat_executor()
    vision_executor = get_vision_executor()
    audio_executor = get_audio_executor()

    runtime_cfg: dict[str, Any] = {
        "model_config": config_path,
        "model_device": device_override or "auto",
        "model_allowlist": model_allowlist,
        "max_concurrent": limiter.MAX_CONCURRENT,
        "max_queue_size": limiter.MAX_QUEUE_SIZE,
        "queue_timeout_sec": limiter.QUEUE_TIMEOUT_SEC,
        "embedding_max_workers": getattr(embedding_executor, "_max_workers", EMBEDDING_MAX_WORKERS),
        "embedding_count_max_workers": getattr(
            embedding_count_executor,
            "_max_workers",
            None,
        ),
        "chat_max_workers": getattr(chat_executor, "_max_workers", CHAT_MAX_WORKERS),
        "vision_max_workers": getattr(vision_executor, "_max_workers", VISION_MAX_WORKERS),
        "audio_max_workers": getattr(audio_executor, "_max_workers", AUDIO_MAX_WORKERS),
        "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "32")),
        "max_text_chars": int(os.getenv("MAX_TEXT_CHARS", "20000")),
        "enable_batching": batching_enabled,
        "batch_window_ms": batch_window_ms,
        "batch_max_size": batch_max_size,
        "embedding_batch_queue_size": batch_queue_size,
        "enable_chat_batching": chat_batching_enabled,
        "chat_batch_window_ms": chat_batch_window_ms,
        "chat_batch_max_size": chat_batch_max_size,
        "chat_max_prompt_tokens": chat_max_prompt_tokens,
        "chat_max_new_tokens": chat_max_new_tokens,
        "chat_batch_queue_size": chat_batch_queue_size,
        "chat_queue_max_wait_ms": float(os.getenv("CHAT_QUEUE_MAX_WAIT_MS", "2000")),
        "chat_requeue_max_wait_ms": float(os.getenv("CHAT_REQUEUE_MAX_WAIT_MS", "2000")),
        "chat_requeue_max_tasks": int(os.getenv("CHAT_REQUEUE_MAX_TASKS", "64")),
        "audio_max_concurrent": audio_limits.MAX_CONCURRENT,
        "audio_max_queue_size": audio_limits.MAX_QUEUE_SIZE,
        "audio_queue_timeout_sec": audio_limits.QUEUE_TIMEOUT_SEC,
        "embedding_generate_timeout_sec": float(os.getenv("EMBEDDING_GENERATE_TIMEOUT_SEC", "60")),
        "audio_process_timeout_sec": float(os.getenv("AUDIO_PROCESS_TIMEOUT_SEC", "180")),
    }

    # Log when audio concurrency/queue defaults are derived from global settings
    # rather than explicit AUDIO_* overrides, for easier ops debugging.
    if "AUDIO_MAX_CONCURRENT" not in os.environ and audio_limits.MAX_CONCURRENT == limiter.MAX_CONCURRENT:
        logger.debug(
            "audio_max_concurrent_derived_from_global",
            extra={
                "audio_max_concurrent": audio_limits.MAX_CONCURRENT,
                "max_concurrent": limiter.MAX_CONCURRENT,
            },
        )
    if (
        "AUDIO_MAX_QUEUE_SIZE" not in os.environ
        and audio_limits.MAX_QUEUE_SIZE == limiter.MAX_QUEUE_SIZE
    ):
        logger.debug(
            "audio_max_queue_size_derived_from_global",
            extra={
                "audio_max_queue_size": audio_limits.MAX_QUEUE_SIZE,
                "max_queue_size": limiter.MAX_QUEUE_SIZE,
            },
        )
    if (
        "AUDIO_QUEUE_TIMEOUT_SEC" not in os.environ
        and audio_limits.QUEUE_TIMEOUT_SEC == limiter.QUEUE_TIMEOUT_SEC
    ):
        logger.debug(
            "audio_queue_timeout_sec_derived_from_global",
            extra={
                "audio_queue_timeout_sec": audio_limits.QUEUE_TIMEOUT_SEC,
                "queue_timeout_sec": limiter.QUEUE_TIMEOUT_SEC,
            },
        )

    return runtime_cfg


def _validate_ffmpeg_for_audio(registry: ModelRegistry) -> None:
    """Ensure ffmpeg is available when audio/Whisper models are loaded."""

    has_audio_model = any("audio-transcription" in getattr(m, "capabilities", []) for m in registry.models.values())
    if not has_audio_model:
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Whisper/audio models require ffmpeg for decoding. "
            "Install ffmpeg and restart."
        )


def _download_models_if_enabled(config_path: str, allowlist: list[str] | None, cache_dir: str | None) -> None:
    """Optionally download requested models before startup; exit on failure."""

    if os.getenv("AUTO_DOWNLOAD_MODELS", "1") == "0":
        logger.info("Auto download disabled; assuming models are pre-fetched")
        return
    if not hasattr(hf, "snapshot_download"):
        raise SystemExit("huggingface_hub is required for auto download")

    try:
        with Path(config_path).open() as f:
            cfg = yaml.safe_load(f) or {}
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
    state.warmup_status = WarmupStatus()
    shutdown_count_executor()
    shutdown_executors()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = None
    batching_service = None
    chat_batching_service = None
    try:
        registry, batching_service, chat_batching_service = startup()
        app.state.model_registry = registry
        app.state.batching_service = batching_service
        app.state.chat_batching_service = chat_batching_service
        app.state.warmup_status = state.warmup_status
        app.state.runtime_config = state.runtime_config
        setup_metrics(app)
        yield
    finally:
        await shutdown(batching_service, chat_batching_service)


app = FastAPI(title="Inference Service", lifespan=lifespan)
app.include_router(api_router)
