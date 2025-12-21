import asyncio
import threading
from typing import Any

import pytest

from app.chat_batching import ChatBatcher, _ChatBatchItem

BUCKET_SIZE_TOKENS = 10
MAX_BATCH = 4
MAX_PROMPT_TOKENS = 100
MAX_NEW_TOKENS_CEILING = 32
QUEUE_SIZE = 16


class _DummyModel:
    name = "dummy"

def _make_item(loop: asyncio.AbstractEventLoop, prompt_tokens: int) -> _ChatBatchItem:
    fut: asyncio.Future[Any] = loop.create_future()
    return _ChatBatchItem(
        messages=[{"role": "user", "content": "hi"}],
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        stop=(),
        prompt_tokens=prompt_tokens,
        prepared_inputs=None,
        future=fut,
        enqueue_time=0.0,
        deadline=None,
        cancel_event=threading.Event(),
    )


@pytest.mark.asyncio
async def test_prompt_bucketing_selects_largest_bucket() -> None:
    batcher = ChatBatcher(
        _DummyModel(),
        max_batch=MAX_BATCH,
        window_ms=0,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens_ceiling=MAX_NEW_TOKENS_CEILING,
        queue_size=QUEUE_SIZE,
    )
    batcher._prompt_bucketing = True
    batcher._prompt_bucket_size_tokens = BUCKET_SIZE_TOKENS

    loop = asyncio.get_running_loop()
    small_1 = _make_item(loop, 5)
    small_2 = _make_item(loop, 6)
    large = _make_item(loop, 25)

    selected, leftover = batcher._bucket_by_prompt_length([small_1, small_2, large])
    assert selected == [small_1, small_2]
    assert leftover == [large]


@pytest.mark.asyncio
async def test_prompt_bucketing_noop_when_no_bucket_has_multiple_items() -> None:
    batcher = ChatBatcher(
        _DummyModel(),
        max_batch=MAX_BATCH,
        window_ms=0,
        max_prompt_tokens=MAX_PROMPT_TOKENS,
        max_new_tokens_ceiling=MAX_NEW_TOKENS_CEILING,
        queue_size=QUEUE_SIZE,
    )
    batcher._prompt_bucketing = True
    batcher._prompt_bucket_size_tokens = BUCKET_SIZE_TOKENS

    loop = asyncio.get_running_loop()
    a = _make_item(loop, 1)
    b = _make_item(loop, 20)

    selected, leftover = batcher._bucket_by_prompt_length([a, b])
    assert selected == [a, b]
    assert leftover == []

