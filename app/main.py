import importlib.util
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI

from app import state
from app.api import router as api_router
from app.batching import BatchingService
from app.concurrency import limiter
from app.concurrency.limiter import stop_accepting, wait_for_drain
from app.logging_config import setup_logging
from app.models.registry import ModelRegistry
from app.monitoring.metrics import setup_metrics
from app.threadpool import shutdown_executor
from app.warmup import warm_up_models

logger = logging.getLogger(__name__)

# Load environment variables from a .env file if present (does not override existing env).
load_dotenv(override=False)


def startup() -> tuple[ModelRegistry, BatchingService]:
    setup_logging()

    # Prefer local ./models; if HF_HOME already set, keep it for fallback use
    os.environ.setdefault("HF_HOME", str(Path.cwd() / "models"))

    config_path = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
    device_override = os.getenv("MODEL_DEVICE") or None
    models_env = os.getenv("MODELS") or os.getenv("MODEL_NAMES")  # MODEL_NAMES kept for backward compat
    model_allowlist = [m.strip() for m in (models_env or "").split(",") if m.strip()] or None
    if model_allowlist is None:
        raise SystemExit(
            "No models specified. Set MODELS env (comma-separated) before starting the service."
        )
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

    batching_enabled = os.getenv("ENABLE_BATCHING", "1") != "0"
    batch_window_ms = float(os.getenv("BATCH_WINDOW_MS", "0"))
    batch_max_size = int(
        os.getenv("BATCH_WINDOW_MAX_SIZE", os.getenv("BATCH_MAX_SIZE", os.getenv("MAX_BATCH_SIZE", "32")))
    )
    batching_service = BatchingService(
        registry,
        enabled=batching_enabled,
        max_batch_size=batch_max_size,
        window_ms=batch_window_ms,
    )

    # Set both module-global and FastAPI state for robustness
    state.model_registry = registry
    state.batching_service = batching_service

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

    if os.getenv("ENABLE_WARMUP", "1") != "0":
        warm_up_models(registry)

    return registry, batching_service


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
            logger.warning(
                "FP8 models %s require 'accelerate'. Install it or switch to non-FP8 repos.",
                fp8_models,
            )
        if not torch.cuda.is_available() and not getattr(torch, "xpu", None):
            logger.warning(
                "FP8 models %s require a GPU/XPU runtime. Select non-FP8 variants or run on GPU.",
                fp8_models,
            )


async def shutdown(batching_service: BatchingService | None) -> None:
    stop_accepting()
    await wait_for_drain()
    state.model_registry = None
    if batching_service is not None:
        await batching_service.stop()
        state.batching_service = None
    shutdown_executor()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    registry = None
    batching_service = None
    try:
        registry, batching_service = startup()
        app.state.model_registry = registry
        app.state.batching_service = batching_service
        setup_metrics(app)
        yield
    finally:
        await shutdown(batching_service)


app = FastAPI(title="Inference Service", lifespan=lifespan)
app.include_router(api_router)
