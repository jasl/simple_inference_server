from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


class EmbeddingModel(Protocol):
    name: str
    dim: int
    device: str | torch.device
    # Capabilities advertised by the handler, e.g., ["text-embedding"].
    capabilities: list[str]

    def embed(self, texts: list[str]) -> np.ndarray: ...

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
    ) -> ChatGeneration: ...

    def count_tokens(self, messages: Sequence[dict[str, Any]]) -> int: ...
