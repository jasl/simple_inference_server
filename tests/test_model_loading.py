import textwrap
from pathlib import Path

import numpy as np
import pytest

from app.models import registry


class StubModel:
    def __init__(self, repo: str, device: str) -> None:
        self.repo = repo
        self.device = device
        self.name = repo  # enough for testing
        self.dim = 4
        self.capabilities = ["text-embedding"]

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim))


def test_registry_loads_from_config(tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            models:
              - hf_repo_id: "BAAI/bge-m3"
                handler: "tests.test_model_loading.StubModel"
              - hf_repo_id: "google/embeddinggemma-300m"
                handler: "tests.test_model_loading.StubModel"
            """
        )
    )

    reg = registry.ModelRegistry(str(cfg), device="cpu")
    assert set(reg.list_models()) == {"BAAI/bge-m3", "google/embeddinggemma-300m"}
    model = reg.get("BAAI/bge-m3")
    vecs = model.embed(["hello"])
    assert vecs.shape == (1, 4)
    assert model.capabilities == ["text-embedding"]


def test_registry_applies_local_overlay(tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            models:
              - hf_repo_id: "repo/a"
                handler: "tests.test_model_loading.StubModel"
                supports_structured_outputs: false
            """
        )
    )
    overlay = tmp_path / "model_config.local.yaml"
    overlay.write_text(
        textwrap.dedent(
            """
            models:
              - hf_repo_id: "repo/a"
                supports_structured_outputs: true
              - hf_repo_id: "repo/b"
                handler: "tests.test_model_loading.StubModel"
            """
        )
    )

    reg = registry.ModelRegistry(str(cfg), device="cpu")
    assert set(reg.list_models()) == {"repo/a", "repo/b"}
    assert reg.get("repo/a").supports_structured_outputs is True


def test_registry_accepts_specific_cuda_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            models:
              - hf_repo_id: "BAAI/bge-m3"
                handler: "tests.test_model_loading.StubModel"
            """
        )
    )

    monkeypatch.setattr(registry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(registry.torch.cuda, "device_count", lambda: 2)

    reg = registry.ModelRegistry(str(cfg), device="cuda:1")

    assert reg.device == "cuda:1"
    assert reg.get("BAAI/bge-m3").device == "cuda:1"


def test_registry_respects_allowlist(tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            models:
              - hf_repo_id: "repo/a"
                handler: "tests.test_model_loading.StubModel"
              - hf_repo_id: "repo/b"
                handler: "tests.test_model_loading.StubModel"
            """
        )
    )

    reg = registry.ModelRegistry(str(cfg), device="cpu", allowed_models=["repo/b"])
    assert set(reg.list_models()) == {"repo/b"}
    assert reg.get("repo/b").name == "repo/b"

    with pytest.raises(ValueError):
        registry.ModelRegistry(str(cfg), device="cpu", allowed_models=["repo/c"])
