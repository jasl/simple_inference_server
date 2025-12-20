from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings
from app.embedding_cache import EmbeddingCache, embed_with_cache
from app.models.base import EmbeddingModel


class HFEmbeddingModel(EmbeddingModel):
    """Shared Hugging Face embedding implementation with L2-normalized mean pooling."""

    def __init__(self, hf_repo_id: str, device: str = "cuda") -> None:
        self.name = hf_repo_id
        self.capabilities = ["text-embedding"]
        self.device = torch.device(device)
        self.hf_repo_id = hf_repo_id
        self._tokenizer_local = threading.local()
        self.thread_safe = True

        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.cache_dir = str(models_dir) if models_dir.exists() else os.environ.get("HF_HOME")

        self._cache = EmbeddingCache(max_size=max(settings.embedding_cache_size, 0))

        # Warm a tokenizer for the creating thread; other threads lazily init their own.
        self._tokenizer_local.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
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

        outputs = self.model(**batch)

        # Check after forward pass
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Embedding cancelled")

        last_hidden = outputs.last_hidden_state  # (B, T, H)
        attention_mask = batch["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = last_hidden * attention_mask
        sum_hidden = masked.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        embeddings = sum_hidden / lengths
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
            tok = AutoTokenizer.from_pretrained(
                self.hf_repo_id,
                local_files_only=True,
                cache_dir=self.cache_dir,
            )
            self._tokenizer_local.tokenizer = tok
        return tok
