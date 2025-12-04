from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import torch


class EmbeddingModel(Protocol):
    name: str
    dim: int
    device: str | torch.device
    # Capabilities advertised by the handler, e.g., ["text-embedding"].
    capabilities: list[str]

    def embed(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray: ...

    def count_tokens(self, texts: list[str]) -> int: ...


@dataclass
class ChatGeneration:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = "stop"


class ChatModel(Protocol):
    name: str
    device: str | torch.device
    # Capabilities advertised by the handler, e.g., ["chat-completion", "vision"].
    capabilities: list[str]

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration: ...

    def prepare_inputs(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[dict[str, Any], int]: ...

    def generate_prepared(
        self,
        prepared: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration: ...

    def batched_generate_prepared(
        self,
        prepared_list: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_events: list[threading.Event] | None = None,
    ) -> list[ChatGeneration]: ...

    def count_tokens(self, messages: Sequence[dict[str, Any]]) -> int: ...


@dataclass
class SpeechSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass
class SpeechResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[SpeechSegment] | None = None


@runtime_checkable
class SpeechModel(Protocol):
    name: str
    device: str | torch.device
    capabilities: list[str]

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        timestamp_granularity: Literal["word", "segment", None],
        cancel_event: threading.Event | None = None,
    ) -> SpeechResult: ...
