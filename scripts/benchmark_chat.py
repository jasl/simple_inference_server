#!/usr/bin/env -S uv run --script

"""
Quick chat benchmark against the local OpenAI-compatible endpoint.

Env overrides (also mirrored as CLI flags):
  BASE_URL      - server base URL (default http://localhost:8000)
  MODEL_NAME    - chat model name (default Qwen/Qwen3-VL-2B-Instruct)
  PROMPT        - user prompt (default "Hello, how are you?")
  MAX_TOKENS    - max new tokens to request (default 128)
  TEMPERATURE   - temperature (default 0.7)
  TOP_P         - top_p (default 0.9)
  N_REQUESTS    - total requests to send (default 20)
  CONCURRENCY   - number of concurrent workers (default 5)
  TIMEOUT       - HTTP timeout seconds (default 30)
  VERIFY_SSL    - set 0 to disable TLS verification (default 1)
  VERBOSE       - set 1 to print all errors (default 0)
"""

import argparse
import asyncio
import os
import time
from typing import cast

import httpx
import numpy as np


async def worker(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    results: list[float],
    errors: list[str],
    args: argparse.Namespace,
) -> None:
    while True:
        try:
            _ = await queue.get()
        except asyncio.CancelledError:
            break
        start = time.perf_counter()
        try:
            resp = await client.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": args.prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            )
            resp.raise_for_status()
            latency = time.perf_counter() - start
            results.append(latency)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
        finally:
            queue.task_done()


async def run(args: argparse.Namespace) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(args.n_requests):
        queue.put_nowait(1)

    results: list[float] = []
    errors: list[str] = []
    async with httpx.AsyncClient(verify=args.verify_ssl, timeout=args.timeout) as client:
        tasks = [
            asyncio.create_task(
                worker(
                    queue,
                    client,
                    results,
                    errors,
                    args,
                )
            )
            for _ in range(args.concurrency)
        ]
        await queue.join()
        for t in tasks:
            t.cancel()

    successes = len(results)
    failures = len(errors)
    print(
        f"Requests: {args.n_requests}, Concurrency: {args.concurrency}, "
        f"Successes: {successes}, Failures: {failures}"
    )

    if results:
        arr = np.array(results)
        percentiles = cast(list[float], np.percentile(arr, [50, 90, 99]).tolist())
        p50, p90, p99 = percentiles
        print(f"Latency p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s")
        total_time = arr.sum()
        if total_time > 0:
            print(f"Throughput ≈ {len(results) / total_time:.2f} req/s")
    if errors:
        print(f"First error: {errors[0]}")
        if args.verbose:
            for err in errors:
                print(f"- {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /v1/chat/completions endpoint.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--prompt", default=os.getenv("PROMPT", "Hello, how are you?"))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "128")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "0.9")))
    parser.add_argument("--n-requests", type=int, default=int(os.getenv("N_REQUESTS", "20")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "5")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("TIMEOUT", "30")))
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        default=os.getenv("VERIFY_SSL", "1") != "0",
        help="Enable TLS verification (default on; use --no-verify-ssl to disable)",
    )
    parser.add_argument(
        "--no-verify-ssl",
        dest="verify_ssl",
        action="store_false",
        help="Disable TLS verification",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=os.getenv("VERBOSE", "0") != "0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
