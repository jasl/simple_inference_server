import pytest

from app.models.qwen_vl import QwenVLChat


def test_remote_image_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "0")
    obj = QwenVLChat.__new__(QwenVLChat)

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/image.png")


def test_remote_image_private_ip_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "1")
    monkeypatch.setenv("REMOTE_IMAGE_HOST_ALLOWLIST", "example.com")
    obj = QwenVLChat.__new__(QwenVLChat)

    # Force DNS to resolve to a private address
    monkeypatch.setattr("socket.getaddrinfo", lambda *_args, **_kwargs: [("family", "type", "proto", "canon", ("127.0.0.1", 0))])

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/image.png")
