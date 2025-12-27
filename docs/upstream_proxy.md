# Upstream Proxy Models (OpenAI / vLLM)

This service can act as a **single OpenAI-compatible gateway**: your application only needs to configure one inference endpoint (this service), while specific `model` requests can be forwarded to an upstream OpenAI-compatible service (e.g. **vLLM** or **OpenAI**). Proxy traffic and local traffic are **rate-limited and queued independently**.

## What you get

- **Explicit routing**: only entries declared as “proxy models” in `model_config` will be forwarded.
- **Pass-through**: the proxy path does not enforce strict field validation. Fields like `tools`, `tool` role, vendor extensions, etc. are preserved, and upstream responses are returned as-is.
- **Isolated flow control**: OpenAI proxy and vLLM proxy each has its own limiter (does not interfere with local chat/embedding/vision/audio limiting).
- **Streaming pass-through**: proxied chat supports `stream=true` and forwards upstream SSE (`text/event-stream`) directly to the client (local models still do not support streaming).

## Quick start

1. Add proxy model entries to `models.local.yaml` (or `models.local.yml`) in the repo root (recommended as an overlay; this file is ignored by `.gitignore`).
2. Put the proxy model **name** in `MODELS` (if `name` is not set, `hf_repo_id` is used as the model id).
3. Configure the upstream URL / auth (via environment variables or per-model overrides), then start the service.

## Key concept: `name` vs `hf_repo_id` (very important)

When this service loads models:

- **`name` (optional)**: the model ID your client/app uses (the request `model` field) and the ID used in `MODELS`.
- **`hf_repo_id` (required)**:
  - For local models: this is the Hugging Face repo id.
  - For proxy models: we reuse this field as the **upstream model id** (i.e. the `model` we send to the upstream service).

In other words, a typical proxy model configuration is:

- Client/app uses: `model: "<name>"` (e.g. `proxy-chat`)
- Gateway forwards upstream using: `model: "<hf_repo_id>"` (e.g. `gpt-4o-mini`)

## Supported endpoints

### `POST /v1/chat/completions`

- **Local models**: keep current behavior (no streaming; stricter request schema; structured outputs may be validated/retried locally).
- **Proxy models**: pass through to upstream `/v1/chat/completions`.
  - If `stream=true`, the gateway forwards upstream SSE to the client.

### `POST /v1/embeddings`

- **Local models**: keep current behavior.
- **Proxy models**: pass through to upstream `/v1/embeddings` (non-streaming).

## Configure proxy models (YAML)

Recommended: add entries in `models.local.yaml` (or `models.local.yml`) in the repo root.

### OpenAI chat proxy

```yaml
models:
  - name: "proxy-chat"
    hf_repo_id: "gpt-4o-mini"  # upstream model id
    handler: "app.models.openai_proxy.OpenAIChatProxyModel"
    # Optional per-model overrides:
    # upstream_base_url: "https://api.openai.com/v1"
    # upstream_api_key_env: "OPENAI_API_KEY"
    # upstream_timeout_sec: 60
    # upstream_headers:
    #   X-My-Header: "foo"
```

### vLLM chat proxy

```yaml
models:
  - name: "heavy-qwen"
    hf_repo_id: "Qwen3-32B-Instruct"
    handler: "app.models.vllm_proxy.VLLMChatProxyModel"
    upstream_base_url: "http://127.0.0.1:8001/v1"
```

At startup, set:

- `MODELS=proxy-chat,heavy-qwen,...`

## Proxy handler list

- **OpenAI upstream**:
  - `app.models.openai_proxy.OpenAIChatProxyModel`
  - `app.models.openai_proxy.OpenAIEmbeddingProxyModel`
- **vLLM upstream**:
  - `app.models.vllm_proxy.VLLMChatProxyModel`
  - `app.models.vllm_proxy.VLLMEmbeddingProxyModel`

## Per-model fields (passed to the handler via `config=`)

- **`upstream_base_url`**: upstream base URL. You may specify `http://host` or `http://host/v1`; the gateway will normalize it to `/v1`.
  - For vLLM proxy models, if this field is not set, you must provide `VLLM_BASE_URL`.
- **`upstream_timeout_sec`**: upstream request timeout (seconds).
- **`upstream_api_key_env`**: which environment variable to read the API key from (recommended; avoids writing secrets into YAML).
- **`upstream_api_key`**: inline API key (supported but not recommended to commit).
- **`upstream_headers`**: additional custom headers (e.g. for ingress/routing).
- **`skip_download`**: when true, `AUTO_DOWNLOAD_MODELS` and `scripts/download_models.py` will skip this entry.
  - Proxy handlers are skipped by default anyway; this field is mainly for non-HF custom handlers.

## Environment variables

### OpenAI

- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `OPENAI_API_KEY` (optional)
- `OPENAI_PROXY_TIMEOUT_SEC` (default `60`)
- `OPENAI_PROXY_MAX_CONCURRENT` / `OPENAI_PROXY_MAX_QUEUE_SIZE` / `OPENAI_PROXY_QUEUE_TIMEOUT_SEC` (optional; if unset, falls back to global `MAX_CONCURRENT` / `MAX_QUEUE_SIZE` / `QUEUE_TIMEOUT_SEC`)

### vLLM

- `VLLM_BASE_URL` (required if `upstream_base_url` is not set per-model)
- `VLLM_API_KEY` (optional)
- `VLLM_PROXY_TIMEOUT_SEC` (default `60`)
- `VLLM_PROXY_MAX_CONCURRENT` / `VLLM_PROXY_MAX_QUEUE_SIZE` / `VLLM_PROXY_QUEUE_TIMEOUT_SEC` (optional; if unset, falls back to global values)

## Authentication behavior (important)

Gateway `Authorization` behavior for upstream requests:

- If the gateway is configured with a key (e.g. `OPENAI_API_KEY`, or per-model `upstream_api_key_env` / `upstream_api_key`), it will send:
  - `Authorization: Bearer <key>`
- Otherwise, if the client request includes `Authorization`, the gateway will **forward that header** to the upstream.

Security recommendation:

- If you do not want clients to “bring their own upstream key”, configure `OPENAI_API_KEY` on the gateway side (and add your own auth/network isolation in front). Do not expose the gateway directly to the public Internet.

## Independent limiting & common status codes

Proxy traffic uses separate limiters:

- OpenAI proxy: `OPENAI_PROXY_*`
- vLLM proxy: `VLLM_PROXY_*`

Common responses:

- `429 Too Many Requests` (queue full or queue wait timeout), with `Retry-After`
- `503 Service Unavailable` (service is shutting down / draining)
- `504 Gateway Timeout` (upstream timeout)
- `502 Bad Gateway` (upstream HTTP-layer failure)

## Warmup & download behavior

- Proxy models are **automatically skipped for downloads** (so `gpt-4o-mini` is not treated as a Hugging Face repo to download).
- Proxy models are **skipped by default in warmup** (so startup does not depend on upstream availability/keys).
- If you add a proxy model to `WARMUP_ALLOWLIST`, warmup will send a minimal request to the upstream. If warmup is enabled and fails, startup may fail-fast (same behavior as local models).

## Observability

- `GET /v1/models` returns `owned_by` (`local` / `openai` / `vllm`)
- Logs: `chat_proxy_request` and `embedding_proxy_request` record upstream type, status code, and latency
