#!/usr/bin/env -S uv run --script

import asyncio
import os
import time
from typing import cast

import httpx
import numpy as np

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
N_REQUESTS = int(os.getenv("N_REQUESTS", "20"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
TEXT = os.getenv("TEXT", "hello world")
VERIFY_SSL = os.getenv("VERIFY_SSL", "0") != "0"
TIMEOUT = float(os.getenv("TIMEOUT", "30"))
VERBOSE = os.getenv("VERBOSE", "0") != "0"


async def worker(
    queue: asyncio.Queue, client: httpx.AsyncClient, results: list[float], errors: list[str]
) -> None:
    while True:
        try:
            _ = await queue.get()
        except asyncio.CancelledError:
            break
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{BASE_URL}/v1/embeddings",
                json={"model": MODEL_NAME, "input": TEXT},
            )
            resp.raise_for_status()
            latency = time.perf_counter() - start
            results.append(latency)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            queue.task_done()


async def run() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(N_REQUESTS):
        queue.put_nowait(1)

    results: list[float] = []
    errors: list[str] = []
    async with httpx.AsyncClient(verify=VERIFY_SSL, timeout=TIMEOUT) as client:
        tasks = [
            asyncio.create_task(worker(queue, client, results, errors))
            for _ in range(CONCURRENCY)
        ]
        await queue.join()
        for t in tasks:
            t.cancel()

    if not results:
        print("No results")
        return

    successes = len(results)
    failures = len(errors)
    print(f"Requests: {N_REQUESTS}, Concurrency: {CONCURRENCY}, Successes: {successes}, Failures: {failures}")

    if results:
        arr = np.array(results)
        percentiles = cast(list[float], np.percentile(arr, [50, 90, 99]).tolist())
        p50, p90, p99 = percentiles
        print(f"Latency p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s")
        print(f"Throughput ≈ {len(results) / arr.sum():.2f} req/s")
    if errors:
        print(f"First error: {errors[0]}")
        if VERBOSE:
            for err in errors:
                print(f"- {err}")


if __name__ == "__main__":
    asyncio.run(run())
