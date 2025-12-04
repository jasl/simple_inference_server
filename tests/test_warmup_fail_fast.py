import os
import types

import pytest

from app import main
from app.models.base import EmbeddingModel


class FailingModel(EmbeddingModel):
    name = "fail"
    dim = 1
    device = "cpu"
    capabilities = ["text-embedding"]

    def embed(self, texts: list[str]):  # pragma: no cover - not called
        raise RuntimeError("should not run")

    def count_tokens(self, texts: list[str]) -> int:  # pragma: no cover - not called
        return 0


def test_warmup_fail_fast_exits(tmp_path, monkeypatch):
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        """
models:
  - hf_repo_id: "fail/repo"
    handler: "tests.test_warmup_fail_fast.FailingModel"
"""
    )

    monkeypatch.setenv("MODEL_CONFIG", str(cfg))
    monkeypatch.setenv("MODELS", "fail/repo")
    monkeypatch.setenv("ENABLE_WARMUP", "1")
    monkeypatch.setenv("WARMUP_FAIL_FAST", "1")

    # stub download to no-op
    monkeypatch.setattr(main, "snapshot_download", lambda **kwargs: None)

    # stub warmup to raise
    def _fail(*_args, **_kwargs):
        return ["fail/repo"]

    monkeypatch.setattr(main, "warm_up_models", _fail)

    with pytest.raises(SystemExit):
        main.startup()

