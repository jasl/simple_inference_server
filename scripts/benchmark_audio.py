#!/usr/bin/env -S uv run --script

import asyncio
import os
import time
import wave
from pathlib import Path

import httpx
import numpy as np

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/whisper-tiny")
AUDIO_FILE = os.getenv("AUDIO_FILE", "sample.wav")
N_REQUESTS = int(os.getenv("N_REQUESTS", "10"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "0") != "0"
TIMEOUT = float(os.getenv("TIMEOUT", "60"))
RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "json")
VERBOSE = os.getenv("VERBOSE", "0") != "0"


def _ensure_audio(path: str) -> bytes:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)  # 1s silence
    return p.read_bytes()


async def worker(queue: asyncio.Queue, client: httpx.AsyncClient, audio: bytes, results: list[float], errors: list[str]) -> None:
    while True:
        try:
            _ = await queue.get()
        except asyncio.CancelledError:
            break
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/audio/transcriptions",
                data={"model": MODEL_NAME, "response_format": RESPONSE_FORMAT},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
            resp.raise_for_status()
            results.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            queue.task_done()


async def run() -> None:
    audio_bytes = _ensure_audio(AUDIO_FILE)
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(N_REQUESTS):
        queue.put_nowait(1)

    results: list[float] = []
    errors: list[str] = []
    async with httpx.AsyncClient(verify=VERIFY_SSL, timeout=TIMEOUT) as client:
        tasks = [asyncio.create_task(worker(queue, client, audio_bytes, results, errors)) for _ in range(CONCURRENCY)]
        await queue.join()
        for t in tasks:
            t.cancel()

    successes = len(results)
    failures = len(errors)
    print(f"Requests: {N_REQUESTS}, Concurrency: {CONCURRENCY}, Successes: {successes}, Failures: {failures}")

    if results:
        arr = np.array(results)
        p50, p90, p99 = np.percentile(arr, [50, 90, 99])
        print(f"Latency p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s")
        print(f"Throughput ≈ {len(results) / arr.sum():.2f} req/s")
    if errors:
        print(f"First error: {errors[0]}")
        if VERBOSE:
            for err in errors:
                print(f"- {err}")


if __name__ == "__main__":
    asyncio.run(run())
