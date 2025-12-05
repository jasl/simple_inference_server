from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

from app.models.base import ChatModel, EmbeddingModel
from app.monitoring.metrics import record_device_memory
from app.utils.device import resolve_device

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(
        self,
        config_path: str,
        device: str | None = None,
        allowed_models: Iterable[str] | None = None,
    ) -> None:
        self.models: dict[str, Any] = {}
        # Prefer CLI/env provided device; fall back to auto-detection.
        self.device_preference: str = (device or os.getenv("MODEL_DEVICE") or "auto")
        self.device = resolve_device(self.device_preference, validate=True)
        self.allowed_models = {m.strip() for m in allowed_models or [] if m.strip()} or None
        self._load_from_config(config_path)
        record_device_memory(self.device)

    def _load_from_config(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model config not found: {path}")

        with path_obj.open() as f:
            cfg = yaml.safe_load(f)
        models_cfg = cfg.get("models", [])
        if not models_cfg:
            raise ValueError("No models configured")

        requested = set(self.allowed_models) if self.allowed_models is not None else None
        loaded: set[str] = set()
        for item in models_cfg:
            repo = item["hf_repo_id"]
            name = item.get("name") or repo
            if requested is not None and name not in requested:
                continue
            handler_path = item.get("handler")
            gen_defaults = item.get("defaults") or {}
            if "fp8" in str(repo).lower() and not self._has_fp8_hardware():
                raise RuntimeError(
                    f"Model '{name}' uses FP8 weights but no CUDA/XPU device is available. "
                    "Use a non-FP8 repo (e.g., the BF16/FP16 variant) or run on GPU/XPU hardware."
                )

            if not handler_path:
                raise ValueError(f"Model '{name}' is missing a handler in config")
            handler_factory = self._import_handler(handler_path)

            model = handler_factory(repo, self.device)
            # Optional per-model generation defaults (e.g., temperature, top_p, max_tokens) for chat-capable models only.
            if gen_defaults and "chat-completion" in getattr(model, "capabilities", []):
                setattr(model, "generation_defaults", gen_defaults)
            elif gen_defaults:
                logger.debug("Ignoring generation defaults for non-chat model %s", name)

            self.models[name] = model
            loaded.add(name)

        if requested is not None:
            missing = requested - loaded
            if missing:
                raise ValueError(f"Requested model(s) not found in config: {', '.join(sorted(missing))}")

    def _import_handler(self, dotted_path: str) -> Callable[[str, str], EmbeddingModel | ChatModel | Any]:
        if "." not in dotted_path:
            raise ValueError(f"Handler path must be module.Class, got: {dotted_path}")
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        try:
            handler = getattr(module, class_name)
            return handler
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ImportError(f"Handler class {class_name} not found in {module_path}") from exc

    def get(self, name: str) -> Any:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not loaded")
        return self.models[name]

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    @staticmethod
    def _has_fp8_hardware() -> bool:
        has_cuda = torch.cuda.is_available()
        has_xpu = getattr(torch, "xpu", None) and torch.xpu.is_available()
        return bool(has_cuda or has_xpu)
