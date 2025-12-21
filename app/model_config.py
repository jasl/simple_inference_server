from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

YamlDict = dict[str, Any]


def _local_overlay_path(base_path: Path) -> Path:
    """Return the sibling `*.local.{yaml|yml}` overlay path for a given base config."""

    suffix = base_path.suffix
    if suffix in {".yaml", ".yml"}:
        # model_config.yaml -> model_config.local.yaml
        return base_path.with_name(f"{base_path.stem}.local{suffix}")
    # Fallback: append ".local" to the full filename
    return base_path.with_name(f"{base_path.name}.local")


def _ensure_mapping(value: Any, *, label: str) -> YamlDict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _extract_models(cfg: YamlDict, *, label: str) -> list[YamlDict]:
    value = cfg.get("models", [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label}.models must be a list")
    models: list[YamlDict] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}.models[{idx}] must be a mapping")
        models.append(dict(item))
    return models


def _deep_merge(base: YamlDict, override: YamlDict) -> YamlDict:
    """Deep-merge `override` into `base` (recursing into nested dicts)."""

    merged: YamlDict = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(YamlDict, base_value), cast(YamlDict, value))
        else:
            merged[key] = value
    return merged


def _merge_models(base_models: list[YamlDict], overlay_models: list[YamlDict]) -> list[YamlDict]:
    """Merge model lists, overriding entries by `name` (preferred) or `hf_repo_id`.

    - If an overlay entry matches an existing model (same `name` or `hf_repo_id`), the entry is deep-merged.
    - Otherwise, the overlay entry is appended.
    """

    merged = [dict(item) for item in base_models]

    def _build_indices(models: list[YamlDict]) -> tuple[dict[str, int], dict[str, int]]:
        by_name: dict[str, int] = {}
        by_repo: dict[str, int] = {}
        for idx, item in enumerate(models):
            name = item.get("name")
            if isinstance(name, str) and name:
                by_name[name] = idx
            repo = item.get("hf_repo_id")
            if isinstance(repo, str) and repo:
                by_repo[repo] = idx
        return by_name, by_repo

    by_name, by_repo = _build_indices(merged)

    for overlay in overlay_models:
        overlay_name = overlay.get("name")
        overlay_repo = overlay.get("hf_repo_id")

        match_idx: int | None = None
        if isinstance(overlay_name, str) and overlay_name and overlay_name in by_name:
            match_idx = by_name[overlay_name]
        elif isinstance(overlay_repo, str) and overlay_repo and overlay_repo in by_repo:
            match_idx = by_repo[overlay_repo]

        if match_idx is None:
            merged.append(dict(overlay))
        else:
            merged[match_idx] = _deep_merge(merged[match_idx], dict(overlay))

        # Overlay may introduce/modify identifiers; keep indices consistent.
        by_name, by_repo = _build_indices(merged)

    return merged


def load_model_config(config_path: str | Path) -> YamlDict:
    """Load the model config and apply an optional `*.local.yaml` overlay.

    If `{base}.local.{yaml|yml}` exists, it is merged on top of the base config so
    you can override existing models and/or add new ones without modifying the
    git-tracked base file.
    """

    base_path = Path(config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Model config not found: {base_path}")

    with base_path.open() as f:
        base_raw = yaml.safe_load(f)
    base_cfg = _ensure_mapping(base_raw, label=str(base_path))
    base_models = _extract_models(base_cfg, label=str(base_path))

    overlay_path = _local_overlay_path(base_path)
    if not overlay_path.exists():
        # Normalize the shape so callers can rely on list-of-mappings.
        base_cfg["models"] = base_models
        return base_cfg

    with overlay_path.open() as f:
        overlay_raw = yaml.safe_load(f)
    overlay_cfg = _ensure_mapping(overlay_raw, label=str(overlay_path))
    overlay_models = _extract_models(overlay_cfg, label=str(overlay_path))

    merged_cfg = _deep_merge(base_cfg, overlay_cfg)
    merged_cfg["models"] = _merge_models(base_models, overlay_models)
    return merged_cfg
