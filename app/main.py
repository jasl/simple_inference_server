import importlib.util
import logging
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import huggingface_hub as hf
import torch
import yaml
from app import warmup
from fastapi import FastAPI
from huggingface_hub import snapshot_download

from app import state
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
from app.threadpool import shutdown_executors
from app.warmup import warm_up_models
from app.threadpool import (
    EMBEDDING_MAX_WORKERS,
    CHAT_MAX_WORKERS,
    VISION_MAX_WORKERS,
    AUDIO_MAX_WORKERS,
)

logger = logging.getLogger(__name__)


def startup() -> tuple[ModelRegistry, BatchingService, ChatBatchingService]:
    setup_logging()

    # Prefer local ./models; if HF_HOME already set, keep it for fallback use
    os.environ.setdefault("HF_HOME", str(Path.cwd() / "models"))
    cache_dir = os.environ["HF_HOME"]

    config_path = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
    device_override = os.getenv("MODEL_DEVICE") or None
    models_env = os.getenv("MODELS") or os.getenv("MODEL_NAMES")  # MODEL_NAMES kept for backward compat
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

    batching_enabled = os.getenv("ENABLE_BATCHING", "1") != "0"
    batch_window_ms = float(os.getenv("BATCH_WINDOW_MS", "6"))
    batch_max_size = int(
        os.getenv("BATCH_WINDOW_MAX_SIZE", os.getenv("BATCH_MAX_SIZE", os.getenv("MAX_BATCH_SIZE", "32")))
    )
    batch_queue_size = int(os.getenv("EMBEDDING_BATCH_QUEUE_SIZE", os.getenv("BATCH_QUEUE_SIZE", os.getenv("MAX_QUEUE_SIZE", "64"))))
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

    # Set both module-global and FastAPI state for robustness
    state.model_registry = registry
    state.batching_service = batching_service
    state.chat_batching_service = chat_batching_service

    runtime_cfg = {
        "model_config": config_path,
        "model_device": device_override or "auto",
        "model_allowlist": model_allowlist,
        "max_concurrent": limiter.MAX_CONCURRENT,
        "max_queue_size": limiter.MAX_QUEUE_SIZE,
        "queue_timeout_sec": limiter.QUEUE_TIMEOUT_SEC,
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
        "audio_max_concurrent": audio_limits.MAX_CONCURRENT,
        "audio_max_queue_size": audio_limits.MAX_QUEUE_SIZE,
        "audio_queue_timeout_sec": audio_limits.QUEUE_TIMEOUT_SEC,
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

    warmup_required = os.getenv("ENABLE_WARMUP", "1") != "0"
    # Backward compatible: WARMUP_FAIL_FAST overrides REQUIRE_WARMUP_SUCCESS (default on).
    require_warmup_success = os.getenv(
        "WARMUP_FAIL_FAST",
        os.getenv("REQUIRE_WARMUP_SUCCESS", "1"),
    ) != "0"
    warmup_failures: list[str] = []
    warmup_completed = not warmup_required
    if warmup_required:
        warmup_failures = warm_up_models(registry)
        warmup_completed = True
        if require_warmup_success and warmup_failures:
            failed_list = ", ".join(sorted(warmup_failures))
            raise SystemExit(f"Warmup failed for model(s): {failed_list}")

    ok_models = [] if warmup_failures else registry.list_models()
    warmup_status = WarmupStatus(
        required=warmup_required,
        completed=warmup_completed,
        failures=warmup_failures,
        ok_models=ok_models,
        capabilities=warmup.get_capability_status(),
    )
    state.warmup_status = warmup_status

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
    state.model_registry = None
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
        setup_metrics(app)
        yield
    finally:
        await shutdown(batching_service, chat_batching_service)


app = FastAPI(title="Inference Service", lifespan=lifespan)
app.include_router(api_router)
