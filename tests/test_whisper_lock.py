# ruff: noqa: E402
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Literal, cast

# Stub transformers to avoid heavyweight imports during test collection.
transformers_stub = cast(Any, sys.modules.get("transformers") or types.ModuleType("transformers"))


class _Dummy:  # pragma: no cover - stub
    @staticmethod
    def from_pretrained(*args: object, **kwargs: object) -> None:
        return None


def _pipeline(*args: object, **kwargs: object) -> None:  # pragma: no cover - stub
    return None


transformers_stub.WhisperForConditionalGeneration = getattr(transformers_stub, "WhisperForConditionalGeneration", _Dummy)
transformers_stub.WhisperProcessor = getattr(transformers_stub, "WhisperProcessor", _Dummy)
transformers_stub.pipeline = getattr(transformers_stub, "pipeline", _pipeline)
sys.modules["transformers"] = transformers_stub

from app.models.base import SpeechResult
from app.models.whisper import WhisperASR


class DummyWhisper(WhisperASR):
    def __init__(self) -> None:
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = SimpleNamespace(type="cpu")
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def _build_generate_kwargs(self, *args: object, **kwargs: object) -> dict:  # noqa: D401
        return {}

    def pipeline(self, *args: object, **kwargs: object) -> dict:
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        time.sleep(0.05)
        self._active -= 1
        return {"text": "ok"}

    def transcribe(  # noqa: PLR0913 - signature mirrors production handler
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
        return super().transcribe(
            audio_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            task=task,
            timestamp_granularity=timestamp_granularity,
            cancel_event=cancel_event,
        )


def test_whisper_transcribe_is_serialized() -> None:
    model = DummyWhisper()

    def _run() -> None:
        result = model.transcribe(
            "dummy.wav",
            language=None,
            prompt=None,
            temperature=None,
            task="transcribe",
            timestamp_granularity=None,
        )
        assert isinstance(result, SpeechResult)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(_run)
        pool.submit(_run)

    assert model.max_active == 1
