#!/usr/bin/env -S uv run --script

"""
Lightweight manual smoke tester for the inference server.

Usage:
  uv run python scripts/manual_smoke.py \
    --base-url http://localhost:8000 \
    --embed-model BAAI/bge-m3 \
    --chat-model meta-llama/Llama-3.2-1B-Instruct \
    --audio-model openai/whisper-tiny
"""

import argparse
import asyncio
import tempfile
import wave
from pathlib import Path

import httpx


def _make_wav(duration_sec: float = 0.5, samplerate: int = 16000) -> str:
    """Create a short silent wav file on disk and return the path."""

    frames = int(duration_sec * samplerate)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        path = tmp.name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(b"\x00\x00" * frames)
    return path


async def check_embeddings(client: httpx.AsyncClient, base_url: str, model: str, text: str) -> None:
    resp = await client.post(f"{base_url}/v1/embeddings", json={"model": model, "input": text})
    resp.raise_for_status()
    data = resp.json()
    dim = len(data["data"][0]["embedding"]) if data.get("data") else "?"
    print(f"[embeddings] {model} ok; dim={dim}")


async def check_chat(client: httpx.AsyncClient, base_url: str, model: str, prompt: str) -> None:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]["content"]
    print(f"[chat] {model} ok; sample='{msg[:60]}...'")


async def check_audio(client: httpx.AsyncClient, base_url: str, model: str) -> None:
    wav_path = _make_wav()
    with Path(wav_path).open("rb") as fh:
        files = {"file": ("sample.wav", fh, "audio/wav")}
        data = {"model": model, "response_format": "text"}
        resp = await client.post(f"{base_url}/v1/audio/transcriptions", files=files, data=data)
    resp.raise_for_status()
    text = resp.text.strip()
    print(f"[audio] {model} ok; text='{text[:60]}...'")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke test for the inference server")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--embed-model", help="Embedding model id to test")
    parser.add_argument("--chat-model", help="Chat model id to test")
    parser.add_argument("--audio-model", help="Whisper model id to test")
    parser.add_argument("--text", default="hello world", help="Text prompt for tests")
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=60) as client:
        if args.embed_model:
            await check_embeddings(client, args.base_url, args.embed_model, args.text)
        if args.chat_model:
            await check_chat(client, args.base_url, args.chat_model, args.text)
        if args.audio_model:
            await check_audio(client, args.base_url, args.audio_model)


if __name__ == "__main__":
    asyncio.run(main())
