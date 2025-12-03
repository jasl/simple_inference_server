import pytest

from app.models.qwen_vl import QwenVLChat


@pytest.mark.parametrize(
    ("text", "stop", "expected", "hit"),
    [
        ("hello stop there", ["stop"], "hello", True),
        ("no match here", ["stop"], "no match here", False),
        ("abc END xyz", ["END", "STOP"], "abc", True),
        ("abc END xyz stop", ["stop"], "abc END xyz", True),
    ],
)
def test_trim_with_stop(text: str, stop: list[str], expected: str, hit: bool) -> None:
    trimmed, got_hit = QwenVLChat._trim_with_stop(text, stop)
    assert trimmed == expected
    assert got_hit is hit
