from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from app.models.base import SpeechModel, SpeechResult, SpeechSegment

logger = logging.getLogger(__name__)


class WhisperASR(SpeechModel):
    """Whisper speech-to-text handler with OpenAI-style behavior."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.hf_repo_id = hf_repo_id
        self.name = hf_repo_id  # expose repo id for clarity in logs/metrics
        self.capabilities = ["audio-transcription", "audio-translation"]
        self.device = self._resolve_device(device)
        self._lock = threading.Lock()
        self.thread_safe = True

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

    def transcribe(  # noqa: PLR0913
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        timestamp_granularity: Literal["word", "segment", None],
    ) -> SpeechResult:
        generate_kwargs = self._build_generate_kwargs(language, prompt, temperature, task)
        return_ts: bool | str = False
        if timestamp_granularity == "word":
            return_ts = "word"
        elif timestamp_granularity == "segment":
            return_ts = True

        with self._lock:
            result = self.pipeline(
                audio_path,
                return_timestamps=return_ts,
                generate_kwargs=generate_kwargs,
            )

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

    def _build_generate_kwargs(
        self,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
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

        return kwargs

    def _resolve_device(self, preference: str) -> torch.device:
        pref = (preference or "auto").lower()
        if pref == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(pref)

    def _pipeline_device_arg(self) -> int | str | torch.device:
        if self.device.type == "cpu":
            return -1
        return self.device
