from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Literal

import torch
from transformers import (
    StoppingCriteriaList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from app.models.whisper_worker import _worker_loop
from app.monitoring.metrics import (
    record_whisper_init_failure,
    record_whisper_kill,
    record_whisper_restart,
)

from app.models.base import SpeechModel, SpeechResult, SpeechSegment
from app.models.generation_utils import StopOnCancel
from app.utils.device import resolve_torch_device

logger = logging.getLogger(__name__)


class WhisperASR(SpeechModel):
    """Whisper speech-to-text handler with OpenAI-style behavior."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.hf_repo_id = hf_repo_id
        self.name = hf_repo_id  # expose repo id for clarity in logs/metrics
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = resolve_torch_device(device, validate=False)
        # Serialize access to the underlying pipeline so that a single handler
        # instance is safe even when the shared audio executor has >1 workers.
        self._lock = threading.Lock()
        self.thread_safe = False

        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        cache_dir = str(models_dir) if models_dir.exists() else os.environ.get("HF_HOME")

        self.processor = WhisperProcessor.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=cache_dir,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=cache_dir,
        )
        # Move to device and prefer half precision on CUDA.
        if self.device.type == "cuda":
            self.model.to(self.device)
            self.model.half()
        else:
            self.model.to(self.device)
        self.model.eval()

        self.pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self._pipeline_device_arg(),
            torch_dtype=self.model.dtype,
        )

        # Optional hard-kill path: run transcribe inside a dedicated worker process.
        self._use_subprocess = os.getenv("WHISPER_USE_SUBPROCESS", "0") != "0"
        self._proc_ctx = mp.get_context("spawn") if self._use_subprocess else None
        self._worker_proc: mp.process.BaseProcess | None = None
        self._parent_conn: mp.connection.Connection | None = None
        self._proc_lock = threading.Lock()
        self._last_used = time.monotonic()

    def transcribe(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        timestamp_granularity: Literal["word", "segment", None],
        cancel_event: threading.Event | None = None,
    ) -> SpeechResult:
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        use_subprocess = getattr(self, "_use_subprocess", False)

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Transcription cancelled")

        if use_subprocess:
            return self._transcribe_subprocess(
                audio_path,
                language=language,
                prompt=prompt,
                temperature=temperature,
                task=task,
                timestamp_granularity=timestamp_granularity,
                cancel_event=cancel_event,
            )

        generate_kwargs = self._build_generate_kwargs(language, prompt, temperature, task, cancel_event)
        return_ts: bool | str = False
        if timestamp_granularity == "word":
            return_ts = "word"
        elif timestamp_granularity == "segment":
            return_ts = True

        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Transcription cancelled")
            result = self.pipeline(
                audio_path,
                return_timestamps=return_ts,
                generate_kwargs=generate_kwargs,
            )

        return self._to_speech_result(result, language)

    def _build_generate_kwargs(
        self,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        cancel_event: threading.Event | None,
    ) -> dict:
        kwargs: dict = {"task": task}
        target_lang = language or ("en" if task == "translate" else None)
        if target_lang:
            kwargs["language"] = target_lang

        if prompt:
            try:
                prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors="pt")
                kwargs["prompt_ids"] = prompt_ids
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("failed_to_encode_prompt", extra={"repo": self.hf_repo_id}, exc_info=exc)

        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        if cancel_event is not None:
            stopper = StopOnCancel(cancel_event)
            kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])

        return kwargs


    def _to_speech_result(self, result: dict, language: str | None) -> SpeechResult:
        text = (result.get("text") or "").strip()
        language_out = result.get("language") or language

        segments: list[SpeechSegment] = []
        for idx, chunk in enumerate(result.get("chunks") or []):
            ts = chunk.get("timestamp")
            if not ts or len(ts) != 2 or ts[0] is None or ts[1] is None:
                continue
            segments.append(
                SpeechSegment(
                    id=idx,
                    start=float(ts[0]),
                    end=float(ts[1]),
                    text=(chunk.get("text") or "").strip(),
                )
            )

        return SpeechResult(
            text=text,
            language=language_out,
            segments=segments or None,
        )

    def _pipeline_device_arg(self) -> int | str | torch.device:
        if self.device.type == "cpu":
            return -1
        return self.device

    # ---------- subprocess-based cancellation path ---------------------
    def _device_str(self) -> str:
        return str(self.device)

    def _ensure_worker(self) -> None:
        if not self._use_subprocess:
            return
        idle_secs = float(os.getenv("WHISPER_SUBPROCESS_IDLE_SEC", "0"))
        if self._worker_proc is not None and self._worker_proc.is_alive():
            if idle_secs > 0 and (time.monotonic() - self._last_used) > idle_secs:
                self._kill_worker(log_reason="idle_timeout")
            else:
                return

        if self._worker_proc is not None:
            try:
                self._worker_proc.kill()
            except Exception:
                pass

        if self._proc_ctx is None:
            raise RuntimeError("Subprocess mode not initialized")

        parent_conn, child_conn = self._proc_ctx.Pipe(duplex=True)
        proc = self._proc_ctx.Process(
            target=_worker_loop,
            args=(child_conn, self.hf_repo_id, self._device_str()),
            daemon=True,
        )
        proc.start()
        record_whisper_restart(self.hf_repo_id)
        self._parent_conn = parent_conn
        self._worker_proc = proc
        self._last_used = time.monotonic()

    def _kill_worker(self, log_reason: str | None = None) -> None:
        if self._worker_proc is not None:
            with contextlib.suppress(Exception):
                if self._parent_conn is not None:
                    try:
                        self._parent_conn.send({"cmd": "stop"})
                    except Exception:
                        pass
                    try:
                        self._parent_conn.close()
                    except Exception:
                        pass
            self._worker_proc.join(timeout=1.0)
            if self._worker_proc.is_alive():
                with contextlib.suppress(Exception):
                    self._worker_proc.kill()
                self._worker_proc.join(timeout=1.0)
            record_whisper_kill(self.hf_repo_id)
            if log_reason:
                logging.getLogger(__name__).warning(
                    "whisper_subprocess_killed",
                    extra={"model": self.hf_repo_id, "reason": log_reason},
                )
        self._worker_proc = None
        self._parent_conn = None

    def _transcribe_subprocess(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        timestamp_granularity: Literal["word", "segment", None],
        cancel_event: threading.Event | None,
    ) -> SpeechResult:
        poll_interval = float(os.getenv("WHISPER_SUBPROCESS_POLL_INTERVAL_SEC", "0.05"))

        if not hasattr(self, "_proc_lock"):
            self._proc_lock = threading.Lock()

        with self._proc_lock:
            self._ensure_worker()
            if self._parent_conn is None:
                raise RuntimeError("Whisper subprocess unavailable")

            msg = {
                "cmd": "transcribe",
                "audio_path": audio_path,
                "language": language,
                "prompt": prompt,
                "temperature": temperature,
                "task": task,
                "ts_granularity": timestamp_granularity,
            }
            try:
                self._parent_conn.send(msg)
            except Exception as exc:
                self._kill_worker()
                raise RuntimeError("Failed to dispatch to Whisper subprocess") from exc

            start = time.monotonic()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._kill_worker()
                    raise RuntimeError("Transcription cancelled")

                if self._parent_conn.poll(poll_interval):
                    resp = self._parent_conn.recv()
                    if isinstance(resp, dict) and "err" in resp:
                        raise RuntimeError(resp.get("err") or "Whisper subprocess failed")
                    return self._to_speech_result(resp, language)

                if self._worker_proc is not None and not self._worker_proc.is_alive():
                    self._kill_worker()
                    raise RuntimeError("Whisper subprocess died")

                # Optional max wall time if caller relies solely on cancellation.
                max_wall = os.getenv("WHISPER_SUBPROCESS_MAX_WALL_SEC")
                if max_wall is not None:
                    try:
                        max_wall_f = float(max_wall)
                        if time.monotonic() - start > max_wall_f:
                            self._kill_worker()
                            raise RuntimeError("Whisper subprocess timed out")
                    except ValueError:
                        pass

    def close(self) -> None:  # override to also drop worker
        self._kill_worker()
        with contextlib.suppress(Exception):
            super_close = getattr(self.pipeline, "close", None)
            if callable(super_close):
                super_close()
