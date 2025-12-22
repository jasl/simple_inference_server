from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings
from app.embedding_cache import EmbeddingCache, embed_with_cache
from app.models.base import EmbeddingModel
from app.utils.remote_code import require_trust_remote_code


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with an attention mask."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
    masked = last_hidden * mask
    sum_hidden = masked.sum(dim=1)  # (B, H)
    lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
    return sum_hidden / lengths


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor, *, padding_side: str) -> torch.Tensor:
    """Pool by selecting the last non-padding token embedding.

    For decoder-only models, Hugging Face tokenizers often use left padding to
    keep the latest tokens aligned across a batch. In that case, the last
    position is always a real token and we can take `[:, -1]` directly.

    With right padding, we need to use `attention_mask` to locate the final
    non-padding token per row.
    """
    if padding_side == "left":
        return last_hidden[:, -1, :]

    # Right padding (or unknown): find last valid token index per sequence.
    seq_lens = attention_mask.sum(dim=1).to(torch.long) - 1  # (B,)
    seq_lens = seq_lens.clamp(min=0)
    batch = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_lens]


def _inference_context() -> Any:
    """Return a best-effort no-grad context manager.

    In production, PyTorch exposes `torch.inference_mode()` / `torch.no_grad()`.
    Some unit tests stub torch with a lightweight module where `no_grad` may be a
    context manager object rather than a callable.
    """
    inference_mode = getattr(torch, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()

    no_grad = getattr(torch, "no_grad", None)
    if callable(no_grad):
        return no_grad()
    if no_grad is not None:
        return no_grad
    return nullcontext()


class HFEmbeddingModel(EmbeddingModel):
    """Shared Hugging Face embedding implementation with configurable pooling.

    Defaults to L2-normalized mean pooling. For Qwen3 Embedding models, we
    default to last-token pooling per the model README.
    """

    def __init__(self, hf_repo_id: str, device: str = "cuda", config: dict[str, Any] | None = None) -> None:
        self.name = hf_repo_id
        self.capabilities = ["text-embedding"]
        self.device = torch.device(device)
        self.hf_repo_id = hf_repo_id
        self._tokenizer_local = threading.local()
        self.thread_safe = True
        self._config = dict(config or {})

        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.cache_dir = str(models_dir) if models_dir.exists() else os.environ.get("HF_HOME")

        self._cache = EmbeddingCache(max_size=max(settings.embedding_cache_size, 0))

        # Remote code execution is gated behind an explicit allowlist to avoid
        # accidentally running arbitrary model code at startup.
        model_name = str(self._config.get("name") or hf_repo_id)
        trust_remote_code = require_trust_remote_code(hf_repo_id, model_name=model_name)

        pooling_cfg = str(self._config.get("pooling") or "").strip().lower()
        if pooling_cfg and pooling_cfg not in {"mean", "last_token"}:
            raise ValueError(f"Unsupported pooling strategy for {hf_repo_id}: {pooling_cfg!r}")
        if pooling_cfg:
            self.pooling = pooling_cfg
        else:
            # Qwen3 Embedding recommends pooling on the final token hidden state.
            self.pooling = "last_token" if "qwen3-embedding" in hf_repo_id.lower() else "mean"

        # Warm a tokenizer for the creating thread; other threads lazily init their own.
        self._tokenizer_local.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            cache_dir=self.cache_dir,
        )
        # Ensure padding exists for batched tokenization; fall back to EOS when absent.
        tok = self._tokenizer_local.tokenizer
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token

        self.model = AutoModel.from_pretrained(
            hf_repo_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def _encode(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        """Encode texts with cooperative cancellation between steps.

        Checks cancel_event between tokenization, forward pass, and normalization
        to allow early exit when a request is cancelled.
        """
        tokenizer = self._get_tokenizer()
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Check after tokenization
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Embedding cancelled")

        with _inference_context():
            outputs = self.model(**batch)

        # Check after forward pass
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Embedding cancelled")

        last_hidden = outputs.last_hidden_state  # (B, T, H)
        attention_mask = batch["attention_mask"]  # (B, T)
        if self.pooling == "last_token":
            embeddings = _last_token_pool(
                last_hidden,
                attention_mask,
                padding_side=str(getattr(tokenizer, "padding_side", "right") or "right"),
            )
        else:
            embeddings = _mean_pool(last_hidden, attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Embedding cancelled")

        # Pass cancel_event to _encode for cooperative cancellation
        vectors = embed_with_cache(
            texts,
            lambda t: self._encode(t, cancel_event=cancel_event),
            self._cache,
            self.name,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Embedding cancelled")
        return vectors

    def count_tokens(self, texts: list[str]) -> int:
        # Token counting uses the tokenizer on CPU; no gradient required. Note
        # that encoding and counting currently run in different executors, so
        # any future caching between _encode() and count_tokens() must take
        # care to remain thread-safe and ideally co-locate work in the same
        # worker to avoid cross-thread state.
        tokenized = self._get_tokenizer()(texts, add_special_tokens=True)
        return sum(len(ids) for ids in tokenized["input_ids"])

    def _get_tokenizer(self) -> Any:
        tok = getattr(self._tokenizer_local, "tokenizer", None)
        if tok is None:
            model_name = str(self._config.get("name") or self.hf_repo_id)
            trust_remote_code = require_trust_remote_code(self.hf_repo_id, model_name=model_name)
            tok = AutoTokenizer.from_pretrained(
                self.hf_repo_id,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
                cache_dir=self.cache_dir,
            )
            if getattr(tok, "pad_token_id", None) is None:
                tok.pad_token = tok.eos_token
            self._tokenizer_local.tokenizer = tok
        return tok
