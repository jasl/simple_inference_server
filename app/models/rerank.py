from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.models.base import RerankResult
from app.models.generation_utils import resolve_runtime_device
from app.utils.remote_code import require_trust_remote_code

logger = logging.getLogger(__name__)


class RerankHandler:
    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id
        self.capabilities = ["rerank"]
        self.device = resolve_runtime_device(device)
        self.hf_repo_id = hf_repo_id
        self.thread_safe = True
        trust_remote_code = require_trust_remote_code(hf_repo_id, model_name=hf_repo_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_repo_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
        ).to(self.device)
        self.model.eval()

    def predict(
        self,
        pairs: list[tuple[str, str]],
        cancel_event: threading.Event | None = None,
    ) -> list[float]:
        if not pairs:
            return []

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")

        # Tokenize
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.model.device)

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")

        with torch.inference_mode():
            outputs = self.model(**features)
            scores = outputs.logits

            # Handle different output shapes
            if scores.dim() == 1:
                return scores.tolist()
            if scores.shape[1] == 1:
                return scores.view(-1).tolist()

            # If multiple outputs, assume the last one is the "positive" class (common for NLI-based rerankers)
            # or just take the first one if it's a regressor.
            # For standard cross-encoders, usually it's a single score.
            # But some might be entailment.
            # BGE-reranker: 1 output.
            return scores[:, 0].tolist()

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[RerankResult]:
        # Fallback for non-batched calls
        pairs = [(query, str(doc)) for doc in documents]
        scores = self.predict(pairs, cancel_event=cancel_event)

        indexed_scores = [
            (idx, float(score), str(documents[idx]))
            for idx, score in enumerate(scores)
        ]

        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            RerankResult(index=idx, relevance_score=score, document=doc)
            for idx, score, doc in ranked
        ]

