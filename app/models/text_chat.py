from __future__ import annotations

import importlib
import logging
import os
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

logger = logging.getLogger(__name__)


class _StopOnTokens(StoppingCriteria):
    """Stop generation when any of the provided token sequences is produced."""

    def __init__(self, stop_token_ids: list[list[int]]) -> None:
        super().__init__()
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


class TextChatModel(ChatModel):
    """Generic text-only chat handler using HF chat template."""

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id.split("/")[-1]
        self.capabilities = ["chat-completion"]
        self.device = device
        self.hf_repo_id = hf_repo_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
        )

        device_map = self._resolve_device_map(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
            device_map=device_map,
            dtype="auto",
        )
        if device_map is None and device != "auto":
            self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> ChatGeneration:
        stop_criteria, stop_flag = self._build_stop_criteria(stop)
        raw_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if not isinstance(raw_inputs, dict):
            raise ValueError("Tokenizer returned unexpected format for chat template")
        inputs = {k: v.to(self.model.device) for k, v in raw_inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop_criteria is not None:
            gen_kwargs["stopping_criteria"] = stop_criteria

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_len:]
        completion_tokens = int(generated_ids.shape[1])
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"
        stop_hit = bool(stop_flag and stop_flag.triggered)

        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text, trimmed = self._trim_with_stop(text, stop)
        if trimmed:
            stop_hit = True

        return ChatGeneration(
            text=text.strip(),
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            finish_reason="stop" if stop_hit else finish_reason,
        )

    def count_tokens(self, messages: Sequence[dict[str, Any]]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return int(encoded["input_ids"].shape[1])

    def _build_stop_criteria(
        self, stop: list[str] | None
    ) -> tuple[StoppingCriteriaList | None, _StopOnTokens | None]:
        if not stop:
            return None, None
        stop_token_ids = [
            self.tokenizer.encode(s, add_special_tokens=False) for s in stop if s
        ]
        stopper = _StopOnTokens(stop_token_ids)
        return StoppingCriteriaList([stopper]), stopper

    @staticmethod
    def _trim_with_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
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

    def _resolve_device_map(self, device_pref: str) -> str | None:
        if device_pref != "auto":
            return None
        if importlib.util.find_spec("accelerate") is not None:
            return "auto"
        logger.debug("accelerate not installed; loading %s without device_map", self.hf_repo_id)
        return None
