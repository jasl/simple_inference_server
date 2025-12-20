from __future__ import annotations

import contextlib
import multiprocessing.connection
import os
import traceback
from typing import Any

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


def _worker_loop(conn: multiprocessing.connection.Connection, hf_repo_id: str, device: str) -> None:
    """Subprocess loop owning a Whisper pipeline.

    Receives dict messages over a Pipe; expected shape:
      {"cmd": "transcribe", "audio_path": str, "language": str|None, "prompt": str|None,
       "temperature": float|None, "task": str, "ts_granularity": str|None}
    Replies with either a result dict (pipeline output) or {"err", "trace"} on failure.
    """

    try:
        processor = WhisperProcessor.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            hf_repo_id,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME"),
        )
        dev = torch.device(device)
        if dev.type == "cuda":
            model.to(dev)
            model.half()
        else:
            model.to(dev)
        model.eval()

        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1 if dev.type == "cpu" else dev,
            dtype=model.dtype,
        )
    except Exception as exc:  # pragma: no cover - defensive
        conn.send({"err": f"init_failed: {exc}", "trace": traceback.format_exc()})
        return

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break

        if not isinstance(msg, dict):
            continue
        if msg.get("cmd") == "stop":
            break
        if msg.get("cmd") != "transcribe":
            continue

        try:
            generate_kwargs: dict[str, Any] = {"task": msg.get("task")}
            if msg.get("language"):
                generate_kwargs["language"] = msg.get("language")
            if msg.get("prompt"):
                with contextlib.suppress(Exception):
                    prompt_ids = processor.get_prompt_ids(msg.get("prompt"), return_tensors="pt")
                    generate_kwargs["prompt_ids"] = prompt_ids
            temperature = msg.get("temperature")
            if temperature is not None:
                generate_kwargs["temperature"] = float(temperature)

            return_ts: bool | str = False
            gran = msg.get("ts_granularity")
            if gran == "word":
                return_ts = "word"
            elif gran == "segment":
                return_ts = True

            result = pipe(
                msg.get("audio_path"),
                return_timestamps=return_ts,
                generate_kwargs=generate_kwargs,
            )

            conn.send(result)
        except Exception as exc:  # pragma: no cover - runtime failure
            conn.send({"err": str(exc), "trace": traceback.format_exc()})
