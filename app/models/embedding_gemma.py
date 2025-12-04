import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.embedding_cache import EmbeddingCache, embed_with_cache
from app.models.base import EmbeddingModel


class EmbeddingGemmaEmbedding(EmbeddingModel):
    def __init__(self, hf_repo_id: str, device: str = "cuda") -> None:
        self.name = hf_repo_id
        self.capabilities = ["text-embedding"]
        self.device = torch.device(device)
        self.hf_repo_id = hf_repo_id
        self._tokenizer_local = threading.local()
        self.thread_safe = True
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.cache_dir = str(models_dir) if models_dir.exists() else os.environ.get("HF_HOME")
        cache_size = int(os.getenv("EMBEDDING_CACHE_SIZE", "256"))
        self._cache = EmbeddingCache(max_size=max(cache_size, 0))
        self._tokenizer_local.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id, local_files_only=True, cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            hf_repo_id, local_files_only=True, cache_dir=self.cache_dir
        ).to(self.device)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        tokenizer = self._get_tokenizer()
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**batch)
        last_hidden = outputs.last_hidden_state
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        masked = last_hidden * attention_mask
        sum_hidden = masked.sum(dim=1)
        lengths = attention_mask.sum(dim=1)
        embeddings = sum_hidden / lengths
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed(self, texts: list[str]) -> np.ndarray:
        return embed_with_cache(texts, self._encode, self._cache, self.name)

    def count_tokens(self, texts: list[str]) -> int:
        tokenized = self._get_tokenizer()(texts, add_special_tokens=True)
        return sum(len(ids) for ids in tokenized["input_ids"])

    def _get_tokenizer(self) -> Any:
        tok = getattr(self._tokenizer_local, "tokenizer", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(
                self.hf_repo_id, local_files_only=True, cache_dir=self.cache_dir
            )
            self._tokenizer_local.tokenizer = tok
        return tok
