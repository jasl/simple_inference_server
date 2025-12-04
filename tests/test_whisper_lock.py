import threading
import time
import sys
import types
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

# Stub transformers to avoid heavyweight imports during test collection.
transformers_stub = sys.modules.get("transformers") or types.ModuleType("transformers")


class _Dummy:  # pragma: no cover - stub
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return None


def _pipeline(*args, **kwargs):  # pragma: no cover - stub
    return None


transformers_stub.WhisperForConditionalGeneration = getattr(transformers_stub, "WhisperForConditionalGeneration", _Dummy)
transformers_stub.WhisperProcessor = getattr(transformers_stub, "WhisperProcessor", _Dummy)
transformers_stub.pipeline = getattr(transformers_stub, "pipeline", _pipeline)
sys.modules["transformers"] = transformers_stub

from app.models.base import SpeechResult
from app.models.whisper import WhisperASR


class DummyWhisper(WhisperASR):
    def __init__(self) -> None:  # type: ignore[super-init-not-called]
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = SimpleNamespace(type="cpu")
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def _build_generate_kwargs(self, *args, **kwargs):  # noqa: D401
        return {}

    def pipeline(self, *args, **kwargs):  # type: ignore[override]
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        time.sleep(0.05)
        self._active -= 1
        return {"text": "ok"}

    def transcribe(self, *args, **kwargs):  # type: ignore[override]
        return super().transcribe(*args, **kwargs)


def test_whisper_transcribe_is_serialized() -> None:
    model = DummyWhisper()

    def _run():
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
