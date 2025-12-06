#!/usr/bin/env -S uv run --script

import argparse
import os
from typing import Final

import uvicorn

DEFAULT_MODEL_CONFIG: Final = "configs/model_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the embedding API locally with reload")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "auto"), help="Device to load models on (cpu|mps|cuda[:idx]|auto)")
    parser.add_argument(
        "--config",
        default=os.getenv("MODEL_CONFIG", DEFAULT_MODEL_CONFIG),
        help="Path to model config file",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model names to load (default: all defined in config)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_env = args.models or os.getenv("MODELS")
    if not models_env:
        raise SystemExit("No models specified. Set --models or MODELS env (comma-separated).")
    os.environ["MODEL_CONFIG"] = args.config
    os.environ["MODEL_DEVICE"] = args.device
    os.environ["MODELS"] = models_env
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)  # noqa: S104


if __name__ == "__main__":
    main()
