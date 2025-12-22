# Models Guide

This document covers the built-in model catalog, how to select models at runtime, and how to add custom models.

## Model Selection

Models to load are specified via the `MODELS` environment variable (comma-separated):

```bash
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507,openai/whisper-tiny uv run python scripts/run_dev.py
```

Only models listed in `MODELS` are loaded at startup. This keeps memory usage minimal.

## Built-in Model Catalog

All supported models are defined in `models.yaml`. Below is the current catalog:

### Embedding Models

| Model ID | Handler | Description |
|----------|---------|-------------|
| `BAAI/bge-m3` | `HFEmbeddingModel` | Multilingual embedding model, 1024 dimensions |
| `google/embeddinggemma-300m` | `HFEmbeddingModel` | Lightweight Gemma-based embeddings |
| `Qwen/Qwen3-Embedding-0.6B` | `HFEmbeddingModel` | Qwen3 embedding (last-token pooling) |
| `Qwen/Qwen3-Embedding-4B` | `HFEmbeddingModel` | Larger Qwen3 embedding (last-token pooling) |

### Chat Models (Text-Only)

| Model ID | Handler | Structured Outputs |
|----------|---------|-------------------|
| `Qwen/Qwen3-4B-Instruct-2507` | `TextChatModel` | No |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` | `TextChatModel` | No |
| `meta-llama/Llama-3.2-1B-Instruct` | `TextChatModel` | No |
| `meta-llama/Llama-3.2-3B-Instruct` | `TextChatModel` | No |

### Vision-Language Models

| Model ID | Handler | Notes |
|----------|---------|-------|
| `Qwen/Qwen3-VL-4B-Instruct` | `QwenVLChat` | Standard precision |
| `Qwen/Qwen3-VL-2B-Instruct` | `QwenVLChat` | Standard precision |
| `Qwen/Qwen3-VL-4B-Instruct-FP8` | `QwenVLChat` | FP8 quantized (requires `accelerate`) |
| `Qwen/Qwen3-VL-2B-Instruct-FP8` | `QwenVLChat` | FP8 quantized (requires `accelerate`) |

### Audio Models (Whisper)

| Model ID | Handler | Notes |
|----------|---------|-------|
| `openai/whisper-tiny` | `WhisperASR` | Fastest, lowest accuracy |
| `openai/whisper-tiny.en` | `WhisperASR` | English-only tiny |
| `openai/whisper-small` | `WhisperASR` | Good balance |
| `openai/whisper-small.en` | `WhisperASR` | English-only small |
| `openai/whisper-medium` | `WhisperASR` | Higher accuracy |
| `openai/whisper-medium.en` | `WhisperASR` | English-only medium |
| `openai/whisper-large-v2` | `WhisperASR` | Best accuracy |
| `jethrowang/whisper-tiny-chinese` | `WhisperASR` | Chinese-tuned |
| `Ivydata/whisper-small-japanese` | `WhisperASR` | Japanese-tuned |
| `BELLE-2/Belle-whisper-large-v3-zh` | `WhisperASR` | Chinese large model |

---

## Model Configuration

### Configuration Files

- **`models.yaml`**: Git-tracked catalog of all supported models
- **`models.local.yaml`** (or `.yml`): Local overrides, gitignored

The local file takes precedence. Entries with matching `name` or `hf_repo_id` override the catalog; new entries are appended.

### Model Entry Schema

```yaml
models:
  - name: "my-custom-name"           # Optional: API-facing model ID (defaults to hf_repo_id)
    hf_repo_id: "org/model-name"     # Required: Hugging Face repo ID
    handler: "app.models.handler.ClassName"  # Required: Handler class path
    supports_structured_outputs: false       # Optional: Enable response_format for chat
    pooling: "mean"                          # Optional: Embedding pooling strategy
    defaults:                                # Optional: Generation defaults
      temperature: 0.7
      top_p: 0.9
      max_tokens: 512
    # Proxy model fields (see upstream_proxy.md):
    # upstream_base_url: "http://..."
    # upstream_api_key_env: "MY_API_KEY"
    # skip_download: true
```

### Per-Model Generation Defaults

Chat models can specify default generation parameters in `models.yaml`:

```yaml
- hf_repo_id: "Qwen/Qwen3-4B-Instruct-2507"
  handler: "app.models.text_chat.TextChatModel"
  defaults:
    temperature: 0.7
    top_p: 0.9
    max_tokens: 512
```

Request parameters always override these defaults.

---

## Pre-downloading Models

Download models before deployment to avoid startup delays:

```bash
# Download specific models
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507 uv run python scripts/download_models.py

# Download with custom cache directory
HF_HOME=./models MODELS=openai/whisper-tiny uv run python scripts/download_models.py
```

Set `AUTO_DOWNLOAD_MODELS=0` to require pre-downloaded weights (startup fails if models are missing).

---

## Adding a Custom Model

### Step 1: Implement a Handler

Create a new file in `app/models/` implementing the appropriate protocol.

**Embedding Handler Example:**

```python
# app/models/my_embedding.py
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.models.base import EmbeddingModel


class MyEmbeddingModel:
    """Custom embedding handler."""

    name: str
    dim: int
    device: str | torch.device
    capabilities = ["text-embedding"]

    def __init__(
        self,
        hf_repo_id: str,
        device: str | torch.device,
        cache_dir: str | None = None,
        **kwargs: Any,
    ):
        self.name = kwargs.get("name", hf_repo_id)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            hf_repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to(device).eval()

        # Set embedding dimension from model config
        self.dim = self.model.config.hidden_size

    def embed(
        self,
        texts: list[str],
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(self.tokenizer.encode(t)) for t in texts)
```

**Chat Handler Example:**

```python
# app/models/my_chat.py
from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models.base import ChatGeneration, ChatModel


class MyChatModel:
    """Custom chat handler."""

    name: str
    device: str | torch.device
    capabilities = ["chat-completion"]

    def __init__(
        self,
        hf_repo_id: str,
        device: str | torch.device,
        cache_dir: str | None = None,
        **kwargs: Any,
    ):
        self.name = kwargs.get("name", hf_repo_id)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_repo_id, cache_dir=cache_dir, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_repo_id, cache_dir=cache_dir, local_files_only=True
        ).to(device).eval()

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration:
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return ChatGeneration(
            text=generated,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=len(outputs[0]) - inputs["input_ids"].shape[1],
        )

    def count_tokens(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return len(self.tokenizer.encode(text))
```

### Step 2: Add Configuration Entry

Add to `models.local.yaml`:

```yaml
models:
  - name: "my-embedding"
    hf_repo_id: "my-org/my-embedding-model"
    handler: "app.models.my_embedding.MyEmbeddingModel"

  - name: "my-chat"
    hf_repo_id: "my-org/my-chat-model"
    handler: "app.models.my_chat.MyChatModel"
    supports_structured_outputs: false
    defaults:
      temperature: 0.7
      max_tokens: 512
```

### Step 3: Load the Model

```bash
MODELS=my-embedding,my-chat uv run python scripts/run_dev.py
```

---

## Handler Protocol Reference

### EmbeddingModel

```python
class EmbeddingModel(Protocol):
    name: str
    dim: int
    device: str | torch.device
    capabilities: list[str]  # ["text-embedding"]

    def embed(self, texts: list[str], cancel_event: threading.Event | None = None) -> np.ndarray: ...
    def count_tokens(self, texts: list[str]) -> int: ...
```

### ChatModel

```python
class ChatModel(Protocol):
    name: str
    device: str | torch.device
    capabilities: list[str]  # ["chat-completion"] or ["chat-completion", "vision"]

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ChatGeneration: ...

    def count_tokens(self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool = True) -> int: ...
```

### SpeechModel

```python
class SpeechModel(Protocol):
    name: str
    device: str | torch.device
    capabilities: list[str]  # ["audio-transcription"]

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        task: Literal["transcribe", "translate"],
        timestamp_granularity: Literal["word", "segment", None],
        cancel_event: threading.Event | None = None,
    ) -> SpeechResult: ...
```

### RerankModel

```python
class RerankModel(Protocol):
    name: str
    device: str | torch.device
    capabilities: list[str]  # ["rerank"]

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[RerankResult]: ...
```

---

## Handler Attributes

Optional attributes that affect server behavior:

| Attribute | Type | Description |
|-----------|------|-------------|
| `thread_safe` | `bool` | If `True`, handler can be called concurrently. If `False`, handler should use internal locks. |
| `max_parallelism` | `int \| None` | Cap in-flight calls per handler instance (useful for handlers with internal locks). |
| `generation_defaults` | `dict` | Default generation parameters (temperature, top_p, max_tokens). |
| `supports_structured_outputs` | `bool` | Enable `response_format` for structured JSON output. |
| `owned_by` | `str` | For `/v1/models` response: `"local"`, `"openai"`, or `"vllm"`. |

---

## Upstream Proxy Models

You can also add proxy models that forward requests to upstream services (OpenAI, vLLM, etc.). See [upstream_proxy.md](upstream_proxy.md) for details.

```yaml
models:
  - name: "gpt-4o"
    hf_repo_id: "gpt-4o"
    handler: "app.models.openai_proxy.OpenAIChatProxyModel"
    skip_download: true
```

