
import os
import threading
from pathlib import Path
from typing import Literal, cast

import pytest

from app import main
from app.models.base import SpeechModel, SpeechResult
from app.models.registry import ModelRegistry


def test_startup_requires_models_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure both new and legacy env vars are absent
    monkeypatch.delenv("MODELS", raising=False)
    monkeypatch.delenv("MODEL_NAMES", raising=False)

    with pytest.raises(SystemExit):
        main.startup()


def test_download_runs_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        """
models:
  - hf_repo_id: "repo/a"
    handler: "tests.test_model_loading.StubModel"
"""
    )
    called: list[str] = []

    class DummyHub:
        @staticmethod
        def snapshot_download(repo_id: str, cache_dir: Path, local_dir_use_symlinks: bool) -> None:
            called.append(repo_id)

    monkeypatch.setenv("AUTO_DOWNLOAD_MODELS", "1")
    monkeypatch.setenv("MODEL_CONFIG", str(cfg))
    monkeypatch.setenv("MODELS", "repo/a")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setattr(main, "snapshot_download", DummyHub.snapshot_download)

    # Use private helper directly to avoid full startup side effects.
    main._download_models_if_enabled(str(cfg), ["repo/a"], cache_dir=os.environ["HF_HOME"])

    assert called == ["repo/a"]


def test_download_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        """
models:
  - hf_repo_id: "repo/a"
    handler: "tests.test_model_loading.StubModel"
"""
    )
    monkeypatch.setenv("AUTO_DOWNLOAD_MODELS", "0")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))

    called: list[str] = []

    def fake_snapshot_download(**kwargs: object) -> None:
        called.append("called")

    monkeypatch.setattr(main, "snapshot_download", fake_snapshot_download, raising=False)
    main._download_models_if_enabled(str(cfg), ["repo/a"], cache_dir=os.environ["HF_HOME"])
    assert called == []


def test_download_missing_model_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "model_config.yaml"
    cfg.write_text(
        """
models:
  - hf_repo_id: "repo/a"
    handler: "tests.test_model_loading.StubModel"
"""
    )
    monkeypatch.setenv("AUTO_DOWNLOAD_MODELS", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))

    called: list[str] = []

    def fake_snapshot_download(**kwargs: object) -> None:
        called.append("called")

    monkeypatch.setattr(main, "snapshot_download", fake_snapshot_download)

    with pytest.raises(SystemExit):
        main._download_models_if_enabled(str(cfg), ["repo/missing"], cache_dir=os.environ["HF_HOME"])
    assert called == []


def test_validate_ffmpeg_for_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAudio(SpeechModel):
        name = "audio"
        capabilities = ["audio-transcription"]
        device = "cpu"

        def transcribe(  # pragma: no cover - not used  # noqa: PLR0913
            self,
            audio_path: str,
            *,
            language: str | None,
            prompt: str | None,
            temperature: float | None,
            task: Literal["transcribe", "translate"],
            timestamp_granularity: Literal["word", "segment"] | None,
            cancel_event: threading.Event | None = None,
        ) -> SpeechResult:
            return SpeechResult(text="")

    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(SystemExit):
        dummy_registry = cast(ModelRegistry, type("R", (), {"models": {"audio": DummyAudio()}})())
        main._validate_ffmpeg_for_audio(dummy_registry)
