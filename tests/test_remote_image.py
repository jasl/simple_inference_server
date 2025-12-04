import pytest

from app.models.qwen_vl import QwenVLChat


def test_remote_image_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "0")
    obj = QwenVLChat.__new__(QwenVLChat)

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/image.png")

