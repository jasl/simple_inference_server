from __future__ import annotations

import base64
import importlib
import io
import ipaddress
import logging
import os
import socket
import threading
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import torch
from PIL import Image
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList

from app.models.base import ChatGeneration, ChatModel
from app.models.generation_utils import (
    StopOnCancel,
    StopOnTokens,
    handle_oom,
    resolve_runtime_device,
    trim_with_stop,
)

logger = logging.getLogger(__name__)


class QwenVLChat(ChatModel):
    """Chat handler for Qwen3-VL models with OpenAI-compatible inputs."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id
        self.capabilities = ["chat-completion", "vision"]
        self.device = resolve_runtime_device(device)
        self.hf_repo_id = hf_repo_id
        # Generation is serialized via _gen_lock so a single handler instance
        # is safe even when the shared chat executor has >1 workers.
        self._gen_lock = threading.Lock()
        self.thread_safe = False
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
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        # HTTP client for remote image fetching, bound to instance lifecycle
        self._http_client: httpx.Client | None = None
        self._http_client_lock = threading.Lock()

    def close(self) -> None:
        """Close any resources held by this model instance."""
        with self._http_client_lock:
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None

    def _get_http_client(self, *, timeout: float) -> httpx.Client:
        """Return a lazily-created httpx client bound to this instance.

        The client is created on first use and closed when the instance is closed.
        Access is guarded by a lock to remain safe under multi-threaded callers.
        """
        if self._http_client is not None:
            return self._http_client

        with self._http_client_lock:
            if self._http_client is None:
                limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
                self._http_client = httpx.Client(
                    timeout=timeout,
                    limits=limits,
                    follow_redirects=True,
                )
                logger.info(
                    "qwen_vl_http_client_created",
                    extra={
                        "model": self.hf_repo_id,
                        "timeout": timeout,
                        "max_keepalive_connections": limits.max_keepalive_connections,
                        "max_connections": limits.max_connections,
                    },
                )
            return self._http_client

    # ----------- Public API -------------------------------------------------
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration:
        with self._gen_lock:
            stop_criteria, stop_flag = self._build_stop_criteria(stop, cancel_event)
            prepared_inputs, prompt_len = self.prepare_inputs(messages, add_generation_prompt=True)
            inputs = {k: v.to(self.model.device) for k, v in prepared_inputs.items() if k != "_prompt_len"}

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
                handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

            generated_ids = output_ids[:, prompt_len:]
            completion_tokens = int(generated_ids.shape[1])
            finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
            stop_hit = bool(stop_flag and stop_flag.triggered)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text, trimmed = trim_with_stop(text, stop)
            if trimmed:
                stop_hit = True

            return ChatGeneration(
                text=text.strip(),
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                finish_reason="stop" if stop_hit else finish_reason,
            )

    def generate_prepared(
        self,
        prepared: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration:
        with self._gen_lock:
            stop_criteria, stop_flag = self._build_stop_criteria(stop, cancel_event)
            prompt_len = int(prepared.get("_prompt_len") or prepared["input_ids"].shape[1])
            inputs = {k: v.to(self.model.device) for k, v in prepared.items() if k != "_prompt_len"}

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop_criteria is not None:
                generation_kwargs["stopping_criteria"] = stop_criteria

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
                handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

            generated_ids = output_ids[:, prompt_len:]
            completion_tokens = int(generated_ids.shape[1])
            finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
            stop_hit = bool(stop_flag and stop_flag.triggered)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text, trimmed = trim_with_stop(text, stop)
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
        """Resolve the dedicated Qwen3-VL class and fail hard if unavailable.

        This removes the older fallback to AutoModelForCausalLM so that we only
        ever run against the intended Qwen3-VL architecture. If the installed
        transformers build does not expose Qwen3VLForConditionalGeneration,
        startup should fail rather than silently degrading behaviour.
        """

        try:
            module = importlib.import_module("transformers")
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError("transformers is required for Qwen3-VL models") from exc

        cls = getattr(module, "Qwen3VLForConditionalGeneration", None)
        if cls is None:
            raise ImportError(
                "Qwen3VLForConditionalGeneration not found in the installed transformers package. "
                "Upgrade transformers to a build that includes Qwen3-VL (pyproject pins 4.57.3)."
            )
        return cls

    def _resolve_device_map(self, device_pref: str) -> str | None:
        """Return 'auto' for device_map when using auto device preference.

        Accelerate is a hard dependency, so we don't need a fallback branch.
        """
        if device_pref != "auto":
            return None
        return "auto"

    def count_tokens(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool = True
    ) -> int:
        """Count tokens in a message sequence.

        By default, includes the generation prompt to match what will actually
        be sent to the model during generation. Set add_generation_prompt=False
        for raw message counting.
        """
        qwen_messages = self._to_qwen_messages(messages)
        encoded = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    def batched_generate_prepared(
        self,
        prepared_list: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_events: list[threading.Event] | None = None,
    ) -> list[ChatGeneration]:
        # Vision models are not batched by default; fall back to sequential generation.
        with self._gen_lock:
            cancel_events = cancel_events or [threading.Event() for _ in prepared_list]
            generations: list[ChatGeneration] = []
            for prepared, cancel_event in zip(prepared_list, cancel_events, strict=False):
                generations.append(
                    self.generate_prepared(
                        prepared,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        cancel_event=cancel_event,
                    )
                )
            return generations

    def prepare_inputs(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[dict[str, Any], int]:
        qwen_messages = self._to_qwen_messages(messages)
        raw_inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._normalize_chat_template_output(raw_inputs)
        prompt_len = int(inputs["input_ids"].shape[1])
        inputs["_prompt_len"] = prompt_len
        return inputs, prompt_len

    # ----------- Helpers ----------------------------------------------------
    def _build_stop_criteria(
        self, stop: list[str] | None, cancel_event: threading.Event | None
    ) -> tuple[StoppingCriteriaList | None, StopOnTokens | None]:
        criteria: list[StoppingCriteria] = []
        stopper = None
        if stop:
            stop_token_ids = [
                self.processor.tokenizer.encode(s, add_special_tokens=False) for s in stop if s
            ]
            stopper = StopOnTokens(stop_token_ids)
            criteria.append(stopper)
        if cancel_event is not None:
            criteria.append(StopOnCancel(cancel_event))
        if not criteria:
            return None, stopper
        return StoppingCriteriaList(criteria), stopper

    def _normalize_chat_template_output(self, raw_inputs: Any) -> dict[str, torch.Tensor]:
        """Normalize processor outputs to a plain dict of tensors.

        Qwen's AutoProcessor may return a BatchEncoding, a plain dict, or a
        namespace-like object. In addition, some fields can be non-tensor
        metadata (e.g., original text) which must not be passed to
        ``.to(device)`` in generate_prepared. This helper mirrors the
        TextChatModel behavior but preserves higher-rank tensors (e.g.,
        pixel_values for vision inputs) by avoiding any reshaping.
        """

        if isinstance(raw_inputs, dict):
            source = raw_inputs
        elif hasattr(raw_inputs, "data") and isinstance(raw_inputs.data, dict):
            source = raw_inputs.data  # BatchEncoding-style
        elif hasattr(raw_inputs, "input_ids"):
            source = {
                "input_ids": raw_inputs.input_ids,
                "attention_mask": getattr(raw_inputs, "attention_mask", None),
            }
        else:
            raise ValueError("Processor returned unexpected format for chat template")

        normalized: dict[str, torch.Tensor] = {}
        for key, value in source.items():
            if value is None:
                continue
            if not torch.is_tensor(value):
                logger.debug(
                    "Dropping non-tensor chat template field %s (%s)", key, type(value).__name__
                )
                continue
            normalized[key] = value

        if "input_ids" not in normalized:
            raise ValueError("Processor output missing 'input_ids' tensor")

        return normalized

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

    def _load_image(self, source: str) -> Image.Image:
        """Load an image from a data URI, remote URL, or local path.

        Remote fetch is disabled by default and, when enabled, uses a pooled HTTP client
        plus head checks for MIME/length and streaming with a byte budget to avoid
        blocking executor threads.
        """

        max_bytes = int(os.getenv("MAX_REMOTE_IMAGE_BYTES", str(5 * 1024 * 1024)))
        remote_timeout = float(os.getenv("REMOTE_IMAGE_TIMEOUT", "5"))
        allow_remote = os.getenv("ALLOW_REMOTE_IMAGES", "0") != "0"
        host_allowlist = {h.strip() for h in os.getenv("REMOTE_IMAGE_HOST_ALLOWLIST", "").split(",") if h.strip()}
        mime_allowlist = {m.strip().lower() for m in os.getenv("REMOTE_IMAGE_MIME_ALLOWLIST", "image/png,image/jpeg,image/webp,image/gif").split(",") if m.strip()}

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

            parsed = urllib.parse.urlparse(source)
            if not parsed.hostname:
                raise ValueError("Remote image host missing")
            if not host_allowlist:
                # TODO: tighten remote fetch safety (private ranges, content sniffing) if remote images are enabled.
                raise ValueError("Remote image host allowlist is empty; set REMOTE_IMAGE_HOST_ALLOWLIST to enable remote fetch")
            if parsed.hostname not in host_allowlist:
                raise ValueError("Remote image host not allowed")
            _reject_private_ip(parsed.hostname)

            client = self._get_http_client(timeout=remote_timeout)

            # HEAD for MIME/length validation
            head_resp = client.head(source, follow_redirects=True, timeout=remote_timeout)
            try:
                _ensure_public_url(head_resp.url)
                content_length = int(head_resp.headers.get("content-length", "0") or 0)
                if max_bytes and content_length and content_length > max_bytes:
                    raise ValueError("Remote image too large")
                content_type = (head_resp.headers.get("content-type") or "").split(";")[0].lower()
                if mime_allowlist and content_type and content_type not in mime_allowlist:
                    raise ValueError("Remote image MIME not allowed")
            finally:
                head_resp.close()

            with client.stream("GET", source, follow_redirects=True, timeout=remote_timeout) as resp:
                _ensure_public_url(resp.url)
                resp.raise_for_status()
                buf = io.BytesIO()
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    if not chunk:
                        break
                    buf.write(chunk)
                    if max_bytes and buf.tell() > max_bytes:
                        raise ValueError("Remote image too large")
                buf.seek(0)
                return Image.open(buf).convert("RGB")

        # Assume local file path
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {source}")
        return Image.open(path).convert("RGB")


def _reject_private_ip(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to resolve remote image host") from exc

    for info in infos:
        addr = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError("Remote image host not allowed (private address)")


def _ensure_public_url(url: httpx.URL) -> None:
    host = url.host
    if host is None:
        raise ValueError("Remote image host missing")
    _reject_private_ip(host)
