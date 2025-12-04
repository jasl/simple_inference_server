
import threading
from typing import Any, cast

from app.warmup import get_capability_status, warm_up_models


class DummySpeechModel:
    def __init__(self) -> None:
        self.name = "dummy-audio"
        self.device = "cpu"
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.calls = 0

    def transcribe(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str,
        timestamp_granularity: str | None,
        cancel_event: threading.Event | None = None,
    ) -> object:
        self.calls += 1
        return object()


class DummyRegistry:
    def __init__(self, model: DummySpeechModel) -> None:
        self._model = model

    def list_models(self) -> list[str]:
        return ["dummy-audio"]

    def get(self, name: str) -> DummySpeechModel:
        if name != "dummy-audio":
            raise KeyError
        return self._model


def test_audio_warmup_runs_once(monkeypatch: Any) -> None:
    model = DummySpeechModel()
    registry = DummyRegistry(model)

    # keep warmup light
    monkeypatch.setenv("WARMUP_STEPS", "1")
    monkeypatch.setenv("WARMUP_BATCH_SIZE", "1")
    # allowlist to avoid touching other models if present
    monkeypatch.setenv("WARMUP_ALLOWLIST", "dummy-audio")

    failures = warm_up_models(cast(Any, registry))
    assert failures == []
    assert model.calls >= 1

    status = get_capability_status()
    assert status["dummy-audio"]["audio-transcription"] is True
