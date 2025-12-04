import io
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient


class _DummyCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:  # pragma: no cover - noop for tests
        return None


_torch_stub: Any = types.ModuleType("torch")
_torch_stub.cuda = _DummyCuda
_torch_stub.OutOfMemoryError = _DummyCuda.OutOfMemoryError
sys.modules.setdefault("torch", _torch_stub)

_torchaudio_stub: Any = types.ModuleType("torchaudio")
_torchaudio_stub.info = lambda _path: types.SimpleNamespace(num_frames=0, sample_rate=0)
sys.modules.setdefault("torchaudio", _torchaudio_stub)

from app import api  # noqa: E402
from app.dependencies import get_model_registry  # noqa: E402
from app.models.base import SpeechResult, SpeechSegment  # noqa: E402

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


class DummySpeechModel:
    def __init__(self) -> None:
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = "cpu"

    def transcribe(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: str,
        timestamp_granularity: str | None,
    ) -> SpeechResult:
        return SpeechResult(
            text="hello audio",
            language=language or "en",
            duration=1.0,
            segments=[
                SpeechSegment(id=0, start=0.0, end=1.0, text="hello audio")
            ],
        )


class DummyRegistry:
    def __init__(self, models: dict[str, object]) -> None:
        self._models = models

    def get(self, name: str) -> object:
        if name not in self._models:
            raise KeyError
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def create_app(models: dict[str, object]) -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry(models)
    return app


def test_transcription_json() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.json()["text"] == "hello audio"


def test_transcription_text_response_format() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny", "response_format": "text"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.text.strip() == "hello audio"


def test_transcription_verbose_json_includes_segments() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "openai/whisper-tiny", "response_format": "verbose_json"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["language"] == "en"
    assert payload["segments"][0]["text"] == "hello audio"


def test_translation_endpoint() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/translations",
        data={"model": "openai/whisper-tiny"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_OK
    assert resp.json()["text"] == "hello audio"


def test_audio_model_not_found() -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "missing"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_NOT_FOUND


def test_audio_capability_required() -> None:
    class NoAudioModel:
        capabilities: list[str] = ["chat-completion"]
        device = "cpu"

    client = TestClient(create_app({"text-model": NoAudioModel()}))
    wav_bytes = _make_wav_bytes()
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "text-model"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_BAD_REQUEST


def test_audio_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app({"openai/whisper-tiny": DummySpeechModel()}))
    wav_bytes = _make_wav_bytes()

    original_save = api._save_upload

    async def tiny_save(file: UploadFile) -> tuple[str, int]:
        return await original_save(file, max_bytes=10)

    monkeypatch.setattr(api, "_save_upload", tiny_save)

    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-dummy"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
    )
    assert resp.status_code == HTTP_BAD_REQUEST


@pytest.mark.asyncio
async def test_save_upload_streams_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"stream-me" * 4
    upload = UploadFile(filename="audio.wav", file=io.BytesIO(data))

    monkeypatch.setattr(api, "UPLOAD_CHUNK_BYTES", 4)

    path, size = await api._save_upload(upload, max_bytes=len(data) + 10)
    assert size == len(data)

    saved = Path(path).read_bytes()
    Path(path).unlink()

    assert saved == data


@pytest.mark.asyncio
async def test_save_upload_removes_temp_on_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"0123456789"
    upload = UploadFile(filename="audio.wav", file=io.BytesIO(data))

    recorded_paths: list[str] = []
    original_tempfile = tempfile.NamedTemporaryFile

    def tracking_tempfile(*args: Any, **kwargs: Any) -> tempfile._TemporaryFileWrapper:
        tmp = original_tempfile(*args, **kwargs)
        recorded_paths.append(tmp.name)
        return tmp

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_tempfile)
    monkeypatch.setattr(api, "UPLOAD_CHUNK_BYTES", 4)

    with pytest.raises(HTTPException) as excinfo:
        await api._save_upload(upload, max_bytes=5)

    assert excinfo.value.status_code == HTTP_BAD_REQUEST
    for path in recorded_paths:
        assert not Path(path).exists()
