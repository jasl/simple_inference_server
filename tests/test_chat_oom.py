from __future__ import annotations

import threading
from typing import Any

import pytest

from app.models import qwen_vl, text_chat


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        return [1] if s else []

    def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
        return [""]

    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> Any:  # pragma: no cover - not used in these tests
        return {"input_ids": None}


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()

    def batch_decode(self, ids: Any, skip_special_tokens: bool = True) -> list[str]:
        return [""]


class _OOM(Exception):
    """Synthetic OOM used for testing."""


def _patch_torch_oom(torch_mod: Any) -> None:
    # Ensure the stub torch exposes the same attributes our handlers catch.
    if not hasattr(torch_mod, "OutOfMemoryError"):
        torch_mod.OutOfMemoryError = _OOM
    if not hasattr(torch_mod.cuda, "OutOfMemoryError"):
        torch_mod.cuda.OutOfMemoryError = _OOM


def test_text_chat_generate_prepared_oom(monkeypatch: pytest.MonkeyPatch, mock_torch: None) -> None:
    torch_mod = text_chat.torch
    _patch_torch_oom(torch_mod)

    class DummyModel:
        device = "cpu"

        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise _OOM("oom")

    obj: Any = text_chat.TextChatModel.__new__(text_chat.TextChatModel)
    obj.hf_repo_id = "dummy-text"
    obj.tokenizer = _DummyTokenizer()
    obj.model = DummyModel()

    prepared = {
        "input_ids": torch_mod.ones((1, 3)),
        "attention_mask": torch_mod.ones((1, 3)),
        "_prompt_len": 3,
    }

    calls: list[BaseException] = []

    def fake_handle(exc: BaseException, model_name: str, device: Any) -> None:
        calls.append(exc)
        raise exc

    # Patch the shared handle_oom function imported in text_chat module
    monkeypatch.setattr(text_chat, "handle_oom", fake_handle)

    with pytest.raises(_OOM):
        obj.generate_prepared(
            prepared,
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            cancel_event=None,
        )

    assert len(calls) == 1


def test_qwen_vl_generate_prepared_oom(monkeypatch: pytest.MonkeyPatch, mock_torch: None) -> None:
    torch_mod = qwen_vl.torch
    _patch_torch_oom(torch_mod)

    class DummyModel:
        device = "cpu"

        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise _OOM("oom")

    obj: Any = qwen_vl.QwenVLChat.__new__(qwen_vl.QwenVLChat)
    obj.hf_repo_id = "dummy-qwen-vl"
    obj.processor = _DummyProcessor()
    obj.model = DummyModel()
    obj._gen_lock = threading.Lock()

    prepared = {
        "input_ids": torch_mod.ones((1, 3)),
        "attention_mask": torch_mod.ones((1, 3)),
        "_prompt_len": 3,
    }

    calls: list[BaseException] = []

    def fake_handle(exc: BaseException, model_name: str, device: Any) -> None:
        calls.append(exc)
        raise exc

    # Patch the shared handle_oom function imported in qwen_vl module
    monkeypatch.setattr(qwen_vl, "handle_oom", fake_handle)

    with pytest.raises(_OOM):
        obj.generate_prepared(
            prepared,
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            stop=None,
            cancel_event=None,
        )

    assert len(calls) == 1
