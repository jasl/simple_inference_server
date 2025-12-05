from __future__ import annotations

import importlib.util as importlib_util
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from app.models.base import ChatGeneration, ChatModel
from app.models.generation_utils import (
    StopOnCancel,
    StopOnCancelAny,
    StopOnTokens,
    handle_oom,
    resolve_runtime_device,
    trim_with_stop,
)

logger = logging.getLogger(__name__)


class TextChatModel(ChatModel):
    """Generic text-only chat handler using HF chat template."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id
        self.capabilities = ["chat-completion"]
        self.device = resolve_runtime_device(device)
        self.hf_repo_id = hf_repo_id
        self.thread_safe = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
        )
        # Ensure padding is defined for batched generation; fall back to EOS when absent.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the latest tokens aligned across a batch.
        self.tokenizer.padding_side = "left"

        device_map = self._resolve_device_map(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
            device_map=device_map,
            dtype="auto",
        )
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

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
        stop_criteria, stop_flag = self._build_stop_criteria(stop, cancel_event)
        prepared_inputs, prompt_len = self.prepare_inputs(messages, add_generation_prompt=True)
        inputs = self._move_to_device({k: v for k, v in prepared_inputs.items() if k != "_prompt_len"})

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "use_cache": True,  # keep KV cache enabled for speed
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if stop_criteria is not None:
            gen_kwargs["stopping_criteria"] = stop_criteria

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
            handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

        generated_ids = output_ids[:, prompt_len:]
        completion_tokens = int(generated_ids.shape[1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        stop_hit = bool(stop_flag and stop_flag.triggered)

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
        stop_criteria, stop_flag = self._build_stop_criteria(stop, cancel_event)
        prompt_len = int(prepared.get("_prompt_len") or prepared["input_ids"].shape[1])
        inputs = self._move_to_device({k: v for k, v in prepared.items() if k != "_prompt_len"})

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if stop_criteria is not None:
            gen_kwargs["stopping_criteria"] = stop_criteria

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
            handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

        generated_ids = output_ids[:, prompt_len:]
        completion_tokens = int(generated_ids.shape[1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        stop_hit = bool(stop_flag and stop_flag.triggered)

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text, trimmed = trim_with_stop(text, stop)
        if trimmed:
            stop_hit = True

        return ChatGeneration(
            text=text.strip(),
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            finish_reason="stop" if stop_hit else finish_reason,
        )

    def batched_generate(
        self,
        batch_messages: list[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_events: list[threading.Event] | None = None,
    ) -> list[ChatGeneration]:
        """Batched generation for compatible requests (shared decoding params)."""
        stop = stop or []
        cancel_events = cancel_events or [threading.Event() for _ in batch_messages]
        encodings = []
        prompt_lengths: list[int] = []
        for msgs in batch_messages:
            prepared, prompt_len = self.prepare_inputs(msgs, add_generation_prompt=True)
            encodings.append({k: v for k, v in prepared.items() if k != "_prompt_len"})
            prompt_lengths.append(prompt_len)

        padded = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        inputs = self._move_to_device(self._normalize_chat_template_output(padded))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        criteria: list[StoppingCriteria] = []
        if stop:
            criteria.append(StopOnTokens([self.tokenizer.encode(s, add_special_tokens=False) for s in stop if s]))
        if any(ev is not None for ev in cancel_events):
            criteria.append(StopOnCancelAny([ev for ev in cancel_events if ev is not None]))
        if criteria:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
            handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

        generations: list[ChatGeneration] = []
        for idx, prompt_len in enumerate(prompt_lengths):
            generated_ids = output_ids[idx, prompt_len:]
            completion_tokens = int(generated_ids.shape[0])
            finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            text, trimmed = trim_with_stop(text, stop)
            stop_hit = trimmed

            generations.append(
                ChatGeneration(
                    text=text.strip(),
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    finish_reason="stop" if stop_hit else finish_reason,
                )
            )
        return generations

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
        stop = stop or []
        cancel_events = cancel_events or [threading.Event() for _ in prepared_list]
        encodings = []
        prompt_lengths: list[int] = []
        for prepared in prepared_list:
            encodings.append({k: v for k, v in prepared.items() if k != "_prompt_len"})
            prompt_lengths.append(int(prepared.get("_prompt_len") or prepared["input_ids"].shape[1]))

        padded = self.tokenizer.pad(encodings, padding=True, return_tensors="pt")
        inputs = self._move_to_device(self._normalize_chat_template_output(padded))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        criteria: list[StoppingCriteria] = []
        if stop:
            criteria.append(StopOnTokens([self.tokenizer.encode(s, add_special_tokens=False) for s in stop if s]))
        if any(ev is not None for ev in cancel_events):
            criteria.append(StopOnCancelAny([ev for ev in cancel_events if ev is not None]))
        if criteria:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
            handle_oom(exc, self.hf_repo_id, getattr(self.model, "device", None))

        generations: list[ChatGeneration] = []
        for idx, prompt_len in enumerate(prompt_lengths):
            generated_ids = output_ids[idx, prompt_len:]
            completion_tokens = int(generated_ids.shape[0])
            finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            text, trimmed = trim_with_stop(text, stop)
            stop_hit = trimmed

            generations.append(
                ChatGeneration(
                    text=text.strip(),
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    finish_reason="stop" if stop_hit else finish_reason,
                )
            )
        return generations

    def _normalize_chat_template_output(self, raw_inputs: Any) -> dict[str, torch.Tensor]:
        """Handle tokenizer outputs that may be dict, BatchEncoding, or tuple/list."""

        if isinstance(raw_inputs, dict):
            return self._ensure_2d(raw_inputs)
        # BatchEncoding behaves like dict for .items()
        if hasattr(raw_inputs, "data") and isinstance(raw_inputs.data, dict):
            return self._ensure_2d(cast(dict[str, torch.Tensor], raw_inputs.data))
        if hasattr(raw_inputs, "input_ids"):
            # Some tokenizers may return namespace-like objects
            return self._ensure_2d(
                {
                    "input_ids": raw_inputs.input_ids,
                    "attention_mask": getattr(raw_inputs, "attention_mask", None),
                }
            )
        raise ValueError("Tokenizer returned unexpected format for chat template")

    def _move_to_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move tensors to the model device, using pinned memory + non_blocking when on CUDA."""

        use_pinned = self._can_pin_memory()
        device = self.model.device

        moved: dict[str, torch.Tensor] = {}
        for key, tensor in inputs.items():
            if tensor is None:
                continue
            pinned = tensor
            if use_pinned and tensor.device.type == "cpu" and hasattr(tensor, "pin_memory"):
                try:
                    pinned = tensor.pin_memory()
                except RuntimeError:
                    pinned = tensor  # pinning not supported (e.g., no CUDA build)
            moved[key] = pinned.to(device, non_blocking=use_pinned)
        return moved

    def _can_pin_memory(self) -> bool:
        device = getattr(self.model, "device", None)
        return bool(torch.cuda.is_available() and device is not None and str(device).startswith("cuda"))

    @staticmethod
    def _ensure_2d(inputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
        """Guarantee input_ids/attention_mask are 2D as expected by generate()."""

        fixed: dict[str, torch.Tensor] = {}
        for key, tensor in inputs.items():
            if tensor is None:
                continue
            if not torch.is_tensor(tensor):
                raise ValueError(f"{key} is not a tensor")
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2:
                tensor = tensor.view(tensor.shape[0], -1)
            fixed[key] = tensor
        return fixed

    def count_tokens(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool = True
    ) -> int:
        """Count tokens in a message sequence.

        By default, includes the generation prompt to match what will actually
        be sent to the model during generation. Set add_generation_prompt=False
        for raw message counting.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    def prepare_inputs(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[dict[str, Any], int]:
        raw_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        normalized = self._normalize_chat_template_output(raw_inputs)
        prompt_len = int(normalized["input_ids"].shape[1])
        normalized["_prompt_len"] = prompt_len
        return normalized, prompt_len

    def _build_stop_criteria(
        self, stop: list[str] | None, cancel_event: threading.Event | None
    ) -> tuple[StoppingCriteriaList | None, StopOnTokens | None]:
        criteria: list[StoppingCriteria] = []
        stopper = None
        if stop:
            stop_token_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in stop if s]
            stopper = StopOnTokens(stop_token_ids)
            criteria.append(stopper)
        if cancel_event is not None:
            criteria.append(StopOnCancel(cancel_event))
        if not criteria:
            return None, stopper
        return StoppingCriteriaList(criteria), stopper

    def _resolve_device_map(self, device_pref: str) -> str | None:
        if device_pref != "auto":
            return None
        if importlib_util.find_spec("accelerate") is not None:
            return "auto"
        logger.debug("accelerate not installed; loading %s without device_map", self.hf_repo_id)
        return None
