from __future__ import annotations

import types
from typing import Any

from app.config import get_settings
from app.models import hf_embedding


def test_hf_embedding_defaults_to_last_token_pooling_for_qwen3_embedding(monkeypatch: Any) -> None:
    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token = "<eos>"  # noqa: S105 - test fixture token string
            self.padding_side = "right"

    class DummyModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(hidden_size=8)

        def to(self, _device: Any) -> DummyModel:
            return self

        def eval(self) -> DummyModel:
            return self

    def fake_tok_from_pretrained(_repo: str, **_kwargs: Any) -> DummyTokenizer:
        return DummyTokenizer()

    def fake_model_from_pretrained(_repo: str, **_kwargs: Any) -> DummyModel:
        return DummyModel()

    monkeypatch.setattr(hf_embedding.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    monkeypatch.setattr(hf_embedding.AutoModel, "from_pretrained", fake_model_from_pretrained)

    model = hf_embedding.HFEmbeddingModel("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    assert model.pooling == "last_token"


def test_hf_embedding_trust_remote_code_is_gated_by_allowlist(monkeypatch: Any) -> None:
    calls: list[tuple[str, bool]] = []

    class DummyTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token = "<eos>"  # noqa: S105 - test fixture token string
            self.padding_side = "right"

    class DummyModel:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(hidden_size=8)

        def to(self, _device: Any) -> DummyModel:
            return self

        def eval(self) -> DummyModel:
            return self

    def fake_tok_from_pretrained(_repo: str, **kwargs: Any) -> DummyTokenizer:
        calls.append(("tokenizer", bool(kwargs.get("trust_remote_code", False))))
        return DummyTokenizer()

    def fake_model_from_pretrained(_repo: str, **kwargs: Any) -> DummyModel:
        calls.append(("model", bool(kwargs.get("trust_remote_code", False))))
        return DummyModel()

    monkeypatch.setattr(hf_embedding.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    monkeypatch.setattr(hf_embedding.AutoModel, "from_pretrained", fake_model_from_pretrained)

    # Default: remote code disabled
    monkeypatch.setenv("TRUST_REMOTE_CODE_ALLOWLIST", "")
    get_settings.cache_clear()
    calls.clear()
    hf_embedding.HFEmbeddingModel("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    assert calls == [("tokenizer", False), ("model", False)]

    # Allowlist: remote code enabled
    monkeypatch.setenv("TRUST_REMOTE_CODE_ALLOWLIST", "Qwen/Qwen3-Embedding-0.6B")
    get_settings.cache_clear()
    calls.clear()
    hf_embedding.HFEmbeddingModel("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    assert calls == [("tokenizer", True), ("model", True)]
