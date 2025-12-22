# Configuration Reference

This document covers all environment variables and configuration options for Simple Inference Server.

## Configuration Files

The server loads configuration from environment variables with the following precedence (highest to lowest):

1. **Exported environment variables** (e.g., `MODELS=... uv run ...`)
2. **`.env` file** (gitignored, for local overrides)
3. **`env` file** (repo default, safe to commit)

Copy `env` to `.env` to create local overrides without modifying the tracked file.

---

## Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | *(required)* | Comma-separated list of model IDs to load (from `models.yaml`) |
| `MODEL_DEVICE` | `auto` | Device for inference: `cpu`, `cuda`, `cuda:<idx>`, `mps`, or `auto` |
| `MODEL_CONFIG_PATH` | `models.yaml` | Path to model catalog file |
| `AUTO_DOWNLOAD_MODELS` | `1` | Download missing models on startup; set `0` to require pre-downloaded weights |
| `HF_HOME` | `./models` | Hugging Face cache directory for model weights |

### Model Config YAML

Models are defined in `models.yaml` (catalog) with optional overrides in `models.local.yaml` (gitignored).

Each model entry supports:

```yaml
models:
  - name: "my-model"              # Model ID for API requests (optional; defaults to hf_repo_id)
    hf_repo_id: "org/model-name"  # Hugging Face repo or upstream model ID for proxies
    handler: "app.models.handler.ClassName"  # Handler class
    supports_structured_outputs: false        # Enable response_format for chat (optional)
    defaults:                                 # Per-model generation defaults (optional)
      temperature: 0.7
      top_p: 0.9
      max_tokens: 512
```

---

## Global Concurrency

These settings control the global rate limiter that protects all inference paths.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT` | `4` | Maximum concurrent model forward passes across all capabilities |
| `MAX_QUEUE_SIZE` | `64` | Maximum requests waiting in queue before rejecting with 429 |
| `QUEUE_TIMEOUT_SEC` | `2.0` | How long a request can wait in queue before timeout |

---

## Embeddings

### Concurrency and Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent embedding requests |
| `EMBEDDING_MAX_QUEUE_SIZE` | *(falls back to `MAX_QUEUE_SIZE`)* | Embedding queue capacity |
| `EMBEDDING_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Embedding queue timeout |
| `EMBEDDING_MAX_WORKERS` | `4` | Thread pool size for embedding executor |
| `EMBEDDING_COUNT_MAX_WORKERS` | `2` | Thread pool size for token counting |
| `EMBEDDING_GENERATE_TIMEOUT_SEC` | `60` | Hard timeout for embedding generation |

### Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_EMBEDDING_BATCHING` | `1` | Enable micro-batching for embeddings |
| `EMBEDDING_BATCH_WINDOW_MS` | `6` | Collection window for batching (milliseconds) |
| `EMBEDDING_BATCH_WINDOW_MAX_SIZE` | *(falls back to `MAX_BATCH_SIZE`)* | Maximum batch size |
| `EMBEDDING_BATCH_QUEUE_SIZE` | `64` | Per-model batch queue capacity |
| `EMBEDDING_BATCH_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Batch queue timeout |

### Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CACHE_SIZE` | `256` | LRU cache entries per model; set `0` to disable |
| `EMBEDDING_USAGE_DISABLE_TOKEN_COUNT` | `0` | Skip per-request token counting for high-QPS scenarios |

---

## Chat (Text and Multimodal)

### Concurrency and Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent chat requests |
| `CHAT_MAX_QUEUE_SIZE` | *(falls back to `MAX_QUEUE_SIZE`)* | Chat queue capacity |
| `CHAT_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Chat queue timeout |
| `CHAT_MAX_WORKERS` | `4` | Thread pool size for chat executor |
| `CHAT_COUNT_MAX_WORKERS` | `2` | Thread pool size for token counting |
| `CHAT_COUNT_USE_CHAT_EXECUTOR` | `0` | Use chat executor for token counting (may cause head-of-line blocking) |

### Batching

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CHAT_BATCHING` | `1` | Enable request batching for text-only chat |
| `CHAT_BATCH_WINDOW_MS` | `10` | Collection window for batching (milliseconds) |
| `CHAT_BATCH_MAX_SIZE` | `8` | Maximum requests per batch |
| `CHAT_BATCH_QUEUE_SIZE` | `64` | Per-model batch queue capacity |
| `CHAT_BATCH_ALLOW_VISION` | `0` | Include vision models in batching (experimental) |

### Prompt Bucketing

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_BATCH_PROMPT_BUCKETING` | `0` | Group requests by prompt length to reduce padding waste |
| `CHAT_BATCH_PROMPT_BUCKET_SIZE_TOKENS` | `256` | Bucket width in tokens |

### Limits and Timeouts

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_MAX_PROMPT_TOKENS` | `4096` | Maximum input tokens (rejects longer prompts) |
| `CHAT_MAX_NEW_TOKENS` | `2048` | Maximum output tokens per request |
| `CHAT_PREPARE_TIMEOUT_SEC` | `10` | Timeout for request preparation |
| `CHAT_GENERATE_TIMEOUT_SEC` | `60` | Hard timeout for generation |
| `CHAT_OOM_COOLDOWN_SEC` | `300` | Cooldown after OOM before restoring batch size |

### Queue Management

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_QUEUE_MAX_WAIT_MS` | `2000` | Maximum wait time in batch queue |
| `CHAT_REQUEUE_RETRIES` | `3` | Retries for incompatible batch requeue |
| `CHAT_REQUEUE_BASE_DELAY_MS` | `5` | Base delay between requeue attempts |
| `CHAT_REQUEUE_MAX_DELAY_MS` | `100` | Maximum requeue delay |
| `CHAT_REQUEUE_MAX_WAIT_MS` | `2000` | Maximum total requeue wait time |
| `CHAT_REQUEUE_MAX_TASKS` | `64` | Maximum pending requeue tasks |

### Structured Outputs

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_STRUCTURED_OUTPUT_MAX_RETRIES` | `1` | Retries for invalid JSON output |
| `CHAT_STRUCTURED_OUTPUT_WARN_ONLY` | `0` | Return raw output on validation failure instead of 5xx |

---

## Audio (Whisper)

### Concurrency and Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent audio requests |
| `AUDIO_MAX_QUEUE_SIZE` | `64` | Audio queue capacity |
| `AUDIO_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Audio queue timeout |
| `AUDIO_MAX_WORKERS` | `1` | Thread pool size for audio executor |
| `AUDIO_PROCESS_TIMEOUT_SEC` | `180` | Hard timeout for audio processing |
| `MAX_AUDIO_BYTES` | `26214400` (25MB) | Maximum upload size |

### Subprocess Mode

Optional subprocess-based Whisper for hard cancellation support.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_USE_SUBPROCESS` | `0` | Enable subprocess mode for hard kill on cancel/timeout |
| `WHISPER_SUBPROCESS_POLL_INTERVAL_SEC` | `0.05` | Poll interval for subprocess communication |
| `WHISPER_SUBPROCESS_MAX_WALL_SEC` | `0` | Maximum wall time (0 = disabled) |
| `WHISPER_SUBPROCESS_IDLE_SEC` | `0` | Idle timeout before subprocess termination |

---

## Vision (Qwen VL)

### Concurrency and Workers

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent vision requests |
| `VISION_MAX_QUEUE_SIZE` | `64` | Vision queue capacity |
| `VISION_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Vision queue timeout |
| `VISION_MAX_WORKERS` | `2` | Thread pool size for vision executor |

### Remote Image Fetching

Remote image fetching is **disabled by default** for security (SSRF prevention).

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOW_REMOTE_IMAGES` | `0` | Enable HTTP fetching of remote images |
| `REMOTE_IMAGE_HOST_ALLOWLIST` | *(empty)* | Comma-separated allowed domains (required when enabled) |
| `REMOTE_IMAGE_TIMEOUT` | `5` | Fetch timeout in seconds |
| `MAX_REMOTE_IMAGE_BYTES` | `5242880` (5MB) | Maximum image size to fetch |
| `REMOTE_IMAGE_MIME_ALLOWLIST` | `image/png,image/jpeg,image/webp,image/gif` | Allowed MIME types |

**Security notes:**
- Private/loopback IPs are blocked even if the host is allowlisted
- MIME type is validated from response headers
- Redirects are limited and validated

---

## Upstream Proxy (OpenAI / vLLM)

Forward requests to upstream OpenAI-compatible services. See [upstream_proxy.md](upstream_proxy.md) for detailed configuration.

### OpenAI Upstream

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `OPENAI_API_KEY` | *(empty)* | API key (if empty, may forward inbound Authorization) |
| `OPENAI_PROXY_TIMEOUT_SEC` | `60` | Upstream request timeout |
| `OPENAI_PROXY_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent proxy requests |
| `OPENAI_PROXY_MAX_QUEUE_SIZE` | *(falls back to `MAX_QUEUE_SIZE`)* | Proxy queue capacity |
| `OPENAI_PROXY_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Proxy queue timeout |

### vLLM Upstream

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | *(required for vLLM proxies)* | vLLM server base URL |
| `VLLM_API_KEY` | *(empty)* | API key (optional) |
| `VLLM_PROXY_TIMEOUT_SEC` | `60` | Upstream request timeout |
| `VLLM_PROXY_MAX_CONCURRENT` | *(falls back to `MAX_CONCURRENT`)* | Concurrent proxy requests |
| `VLLM_PROXY_MAX_QUEUE_SIZE` | *(falls back to `MAX_QUEUE_SIZE`)* | Proxy queue capacity |
| `VLLM_PROXY_QUEUE_TIMEOUT_SEC` | *(falls back to `QUEUE_TIMEOUT_SEC`)* | Proxy queue timeout |

---

## Request Guards

Global limits to protect the API from oversized requests.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_BATCH_SIZE` | `32` | Maximum items per batch request |
| `MAX_TEXT_CHARS` | `20000` | Maximum characters per text input |
| `MAX_NEW_TOKENS` | `512` | Default max tokens for generation (per-model defaults override) |
| `EXECUTOR_GRACE_PERIOD_SEC` | `2.0` | Grace period for executor shutdown |

---

## Warmup

Warmup runs a test batch through each model on startup to initialize tokenizers and compile kernels.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_WARMUP` | `1` | Enable warmup pass on startup |
| `WARMUP_BATCH_SIZE` | `1` | Batch size for warmup |
| `WARMUP_STEPS` | `1` | Number of warmup iterations |
| `WARMUP_INFERENCE_MODE` | `1` | Use `torch.inference_mode()` during warmup |
| `WARMUP_VRAM_BUDGET_MB` | `0` | VRAM budget for warmup workers (0 = use available) |
| `WARMUP_VRAM_PER_WORKER_MB` | `1024` | Estimated VRAM per worker |
| `WARMUP_ALLOWLIST` | *(empty)* | Comma-separated models to include (empty = all) |
| `WARMUP_SKIPLIST` | *(empty)* | Comma-separated models to skip |

**Fail-fast behavior:** When `ENABLE_WARMUP=1`, the server exits immediately if any model warmup fails. Use `WARMUP_SKIPLIST` to exclude problematic models rather than disabling warmup entirely.

---

## Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_METRICS` | `1` | Enable Prometheus metrics at `/metrics` |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Security

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUST_REMOTE_CODE_ALLOWLIST` | *(empty)* | Comma-separated HF repo IDs allowed to use `trust_remote_code=True` |

Models requiring custom code (e.g., some Qwen variants) will fail to load unless explicitly allowlisted.

---

## Example Configuration

### Single GPU, Embeddings + Chat

```bash
# .env
MODEL_DEVICE=cuda
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507
MAX_CONCURRENT=1
EMBEDDING_BATCH_WINDOW_MS=4
EMBEDDING_BATCH_WINDOW_MAX_SIZE=16
ENABLE_WARMUP=1
```

### CPU-Only Edge Device

```bash
# .env
MODEL_DEVICE=cpu
MODELS=BAAI/bge-m3
MAX_CONCURRENT=1
EMBEDDING_MAX_WORKERS=1
CHAT_MAX_WORKERS=1
AUDIO_MAX_CONCURRENT=1
```

### Multi-Model with Whisper

```bash
# .env
MODEL_DEVICE=auto
MODELS=BAAI/bge-m3,Qwen/Qwen3-4B-Instruct-2507,openai/whisper-tiny
MAX_CONCURRENT=2
AUDIO_MAX_CONCURRENT=1
AUDIO_MAX_WORKERS=1
```

