from __future__ import annotations

import base64
import importlib
import io
import logging
import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    __version__ as TRANSFORMERS_VERSION,
)

from app.models.base import ChatGeneration, ChatModel

logger = logging.getLogger(__name__)


class _StopOnTokens(StoppingCriteria):
    """Stop generation when any of the provided token sequences is produced."""

    def __init__(self, stop_token_ids: list[list[int]]) -> None:
        super().__init__()
        # Keep only non-empty stop sequences
        self.stop_token_ids = [ids for ids in stop_token_ids if ids]
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self.stop_token_ids:
            return False
        generated = input_ids[0].tolist()
        for ids in self.stop_token_ids:
            if len(ids) <= len(generated) and generated[-len(ids) :] == ids:
                self.triggered = True
                return True
        return False


class QwenVLChat(ChatModel):
    """Chat handler for Qwen3-VL models with OpenAI-compatible inputs."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id.split("/")[-1]
        self.capabilities = ["chat-completion", "vision"]
        self.device = device
        self.hf_repo_id = hf_repo_id
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.cache_dir = str(models_dir) if models_dir.exists() else os.environ.get("HF_HOME")

        device_map = self._resolve_device_map(device)

        if "fp8" in hf_repo_id.lower() and not (torch.cuda.is_available() or getattr(torch, "xpu", None)):
            raise RuntimeError(
                "FP8 quantized Qwen3-VL requires a GPU/XPU. "
                "Use a non-FP8 repo (e.g., Qwen/Qwen3-VL-4B-Instruct) or run on a GPU-enabled machine."
            )

        self.processor = AutoProcessor.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=self.cache_dir,
        )
        # Qwen3-VL has a dedicated class in recent transformers; fall back to AutoModel when absent.
        model_cls: Any = self._resolve_model_cls()
        self.model = model_cls.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=self.cache_dir,
            device_map=device_map,
            dtype="auto",
        )
        # If we didn't use device_map auto, place on requested device.
        if device_map is None and device != "auto":
            self.model.to(self.device)
        self.model.eval()

    # ----------- Public API -------------------------------------------------
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> ChatGeneration:
        qwen_messages = self._to_qwen_messages(messages)
        stop_criteria, stop_flag = self._build_stop_criteria(stop)

        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop_criteria is not None:
            generation_kwargs["stopping_criteria"] = stop_criteria

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]
        completion_tokens = int(generated_ids.shape[1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        stop_hit = bool(stop_flag and stop_flag.triggered)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text, trimmed = self._trim_with_stop(text, stop)
        if trimmed:
            stop_hit = True

        return ChatGeneration(
            text=text.strip(),
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            finish_reason="stop" if stop_hit else finish_reason,
        )

    # ----------- Helpers ----------------------------------------------------
    def _resolve_model_cls(self) -> Any:
        """Use dedicated Qwen3-VL class if available; otherwise fall back to AutoModel."""

        try:
            module = importlib.import_module("transformers")
            cls = getattr(module, "Qwen3VLForConditionalGeneration", None)
            if cls is not None:
                return cls
        except Exception as exc:  # pragma: no cover - best-effort import
            logger.debug("Failed to import Qwen3VLForConditionalGeneration: %s", exc)
        # Warn only when we have to fall back; users on >=4.51.0 should have the class.
        logger.warning(
            "Qwen3VLForConditionalGeneration not found in transformers %s; "
            "falling back to AutoModelForCausalLM with trust_remote_code. "
            "Consider upgrading transformers to >=4.51.0 (pyproject pins 4.57.3).",
            TRANSFORMERS_VERSION,
        )
        return AutoModelForCausalLM

    def _resolve_device_map(self, device_pref: str) -> str | dict[str, Any] | None:
        """Use device_map='auto' only if accelerate is installed; otherwise fall back."""

        if device_pref != "auto":
            return None

        try:
            importlib.import_module("accelerate")
            return "auto"
        except ImportError:
            logger.warning(
                "accelerate not installed; cannot use device_map='auto'. "
                "Model will load on default device and then .to(%s). "
                "Install accelerate to enable sharded/auto placement.",
                device_pref,
            )
            return None

    def count_tokens(self, messages: Sequence[dict[str, Any]]) -> int:
        qwen_messages = self._to_qwen_messages(messages)
        encoded = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    # ----------- Helpers ----------------------------------------------------
    def _build_stop_criteria(
        self, stop: list[str] | None
    ) -> tuple[StoppingCriteriaList | None, _StopOnTokens | None]:
        if not stop:
            return None, None
        stop_token_ids = [
            self.processor.tokenizer.encode(s, add_special_tokens=False) for s in stop if s
        ]
        stopper = _StopOnTokens(stop_token_ids)
        return StoppingCriteriaList([stopper]), stopper

    def _to_qwen_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        qwen_messages: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            qwen_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": self._normalize_content(content),
                }
            )
        return qwen_messages

    def _normalize_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        parts: list[dict[str, Any]] = []
        if isinstance(content, Sequence):
            for part in content:
                p_type = part.get("type")
                if p_type == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif p_type == "image_url":
                    image_url = (part.get("image_url") or {}).get("url")
                    if not image_url:
                        raise ValueError("image_url content missing 'url'")
                    parts.append({"type": "image", "image": self._load_image(image_url)})
                else:
                    raise ValueError(f"Unsupported content part type: {p_type}")
        else:
            raise ValueError("content must be a string or list of content parts")
        return parts

    @staticmethod
    def _trim_with_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
        """Trim the generated text at the earliest occurrence of any stop string."""

        if not stop:
            return text, False

        earliest_idx: int | None = None
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx

        if earliest_idx is None:
            return text, False

        return text[:earliest_idx].rstrip(), True

    def _load_image(self, source: str) -> Image.Image:
        """Load an image from a data URI, remote URL, or local path."""

        max_bytes = int(os.getenv("MAX_REMOTE_IMAGE_BYTES", str(5 * 1024 * 1024)))
        remote_timeout = float(os.getenv("REMOTE_IMAGE_TIMEOUT", "5"))
        allow_remote = os.getenv("ALLOW_REMOTE_IMAGES", "0") != "0"

        if source.startswith("data:"):
            try:
                _, b64_data = source.split(",", 1)
            except ValueError as exc:
                raise ValueError("Invalid data URI for image_url") from exc
            data = base64.b64decode(b64_data)
            if max_bytes and len(data) > max_bytes:
                raise ValueError("Image too large")
            return Image.open(io.BytesIO(data)).convert("RGB")

        if source.startswith("http://") or source.startswith("https://"):
            if not allow_remote:
                raise ValueError("Remote image URLs are disabled (set ALLOW_REMOTE_IMAGES=1 to enable)")
            with urllib.request.urlopen(source, timeout=remote_timeout) as resp:  # noqa: S310
                data = resp.read(max_bytes + 1)
            if max_bytes and len(data) > max_bytes:
                raise ValueError("Remote image too large")
            return Image.open(io.BytesIO(data)).convert("RGB")

        # Assume local file path
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {source}")
        return Image.open(path).convert("RGB")
