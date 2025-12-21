#!/usr/bin/env -S uv run --script

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from app.model_config import load_model_config

CONFIG_PATH = os.getenv("MODEL_CONFIG", "configs/model_config.yaml")
MODELS = os.getenv("MODELS")
# Hard lock cache to repo-local models directory to ensure Docker COPY works.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    if not MODELS:
        raise SystemExit("No models specified. Set MODELS env (comma-separated) to download.")

    config_path = Path(CONFIG_PATH)
    cfg = load_model_config(config_path)

    target_dir = DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    requested = {m.strip() for m in MODELS.split(",") if m.strip()} if MODELS else None
    downloaded: set[str] = set()
    for item in cfg.get("models", []):
        repo_id = item["hf_repo_id"]
        name = item.get("name") or repo_id
        if requested is not None and name not in requested:
            continue
        print(f"Downloading {name} ({repo_id}) to {target_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        downloaded.add(name)

    if requested is not None:
        missing = requested - downloaded
        if missing:
            raise SystemExit(f"Requested model(s) not found in config: {', '.join(sorted(missing))}")

    print("All models downloaded.")


if __name__ == "__main__":
    main()
