# API Reference

Simple Inference Server exposes OpenAI-compatible REST APIs for embeddings, chat completions, audio transcription/translation, and model management.

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication is required by default. For upstream proxy models, see [upstream_proxy.md](upstream_proxy.md) for auth configuration.

## Request Tracing

Every request is assigned a unique ID (UUID hex). Pass `X-Request-ID` header to use your own; the same ID is echoed in the response header and included in logs for distributed tracing.

---

## Endpoints

### POST /v1/embeddings

Generate embeddings for text input.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model ID from `models.yaml` |
| `input` | string \| string[] | Yes | Text to embed (single string or array) |
| `encoding_format` | string | No | Only `"float"` is supported (default) |
| `user` | string | No | OpenAI compatibility placeholder |

**Limits:**
- Maximum batch size: `MAX_BATCH_SIZE` (default 32)
- Maximum text length: `MAX_TEXT_CHARS` (default 20,000 characters)

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-m3",
    "input": ["hello", "world"]
  }'
```

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.123, -0.456, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.789, -0.012, ...]}
  ],
  "model": "BAAI/bge-m3",
  "usage": {"prompt_tokens": 2, "total_tokens": 2, "completion_tokens": null}
}
```

---

### POST /v1/chat/completions

Generate chat completions. Supports text-only and vision (image) inputs for compatible models.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model ID with `chat-completion` capability |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | No | Max tokens to generate (uses model default) |
| `temperature` | float | No | Sampling temperature (default: model-specific) |
| `top_p` | float | No | Nucleus sampling (default: model-specific) |
| `n` | integer | No | Must be 1 (only value supported) |
| `stream` | boolean | No | Local models: not supported. Proxy models: pass-through |
| `stop` | string \| string[] | No | Stop sequences |
| `response_format` | object | No | Structured output format (see below) |
| `user` | string | No | OpenAI compatibility placeholder |

**Message Format:**

```json
{
  "role": "system" | "user" | "assistant",
  "content": "string" | [content_parts]
}
```

**Content Parts (for vision):**

```json
[
  {"type": "text", "text": "Describe this image"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]
```

**Structured Output (response_format):**

Requires `supports_structured_outputs: true` in model config.

```json
// JSON object mode
{"type": "json_object"}

// JSON Schema mode
{
  "type": "json_schema",
  "json_schema": {
    "name": "my_schema",
    "schema": {"type": "object", "properties": {...}},
    "strict": true
  }
}
```

**Limits:**
- Maximum prompt tokens: `CHAT_MAX_PROMPT_TOKENS` (default 4096)
- Maximum output tokens: `CHAT_MAX_NEW_TOKENS` (default 2048)

**Example Request (text):**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Who are you?"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**Example Request (vision):**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

**Example Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "I am Qwen, a large language model..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}
}
```

---

### POST /v1/audio/transcriptions

Transcribe audio to text (Whisper-compatible).

**Request Body (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file (wav, mp3, m4a, etc.) |
| `model` | string | Yes | Whisper model ID |
| `language` | string | No | ISO language code (e.g., `ja`, `zh`) to skip auto-detect |
| `prompt` | string | No | Prompt to guide transcription |
| `temperature` | float | No | Sampling temperature (default: 0) |
| `response_format` | string | No | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities` | string[] | No | `segment` or `word` for timestamps |

**Limits:**
- Maximum file size: `MAX_AUDIO_BYTES` (default 25MB)

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=openai/whisper-tiny" \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=text"
```

**Example Response (json):**

```json
{"text": "Hello, this is a transcription test."}
```

**Example Response (verbose_json):**

```json
{
  "text": "Hello, this is a transcription test.",
  "language": "en",
  "duration": 5.2,
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello, this is"},
    {"id": 1, "start": 2.5, "end": 5.2, "text": "a transcription test."}
  ]
}
```

---

### POST /v1/audio/translations

Translate audio to English (Whisper-compatible). Same parameters as transcriptions, but always outputs English.

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "model=openai/whisper-tiny" \
  -F "file=@/path/to/japanese_audio.wav" \
  -F "response_format=text"
```

---

### POST /v1/rerank

Rerank documents by relevance to a query.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model ID with `rerank` capability |
| `query` | string | Yes | Query to rank documents against |
| `documents` | string[] | Yes | List of documents to rerank |
| `top_n` | integer | No | Return only top N results |

**Example Request:**

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-rerank-model",
    "query": "What is machine learning?",
    "documents": ["ML is a subset of AI...", "Football is a sport...", "Neural networks learn..."],
    "top_n": 2
  }'
```

**Example Response:**

```json
{
  "model": "your-rerank-model",
  "results": [
    {"index": 0, "relevance_score": 0.95, "document": "ML is a subset of AI..."},
    {"index": 2, "relevance_score": 0.87, "document": "Neural networks learn..."}
  ],
  "usage": {"total_tokens": 0, "prompt_tokens": 0}
}
```

---

### GET /v1/models

List all loaded models.

**Example Request:**

```bash
curl http://localhost:8000/v1/models
```

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {"id": "BAAI/bge-m3", "object": "model", "owned_by": "local", "embedding_dimensions": 1024},
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "object": "model", "owned_by": "local", "embedding_dimensions": null},
    {"id": "proxy-chat", "object": "model", "owned_by": "openai", "embedding_dimensions": null}
  ]
}
```

---

### GET /health

Health check endpoint for liveness/readiness probes.

**Example Request:**

```bash
curl http://localhost:8000/health
```

**Example Response (healthy):**

```json
{
  "status": "ok",
  "models": ["BAAI/bge-m3", "Qwen/Qwen3-4B-Instruct-2507"],
  "warmup": {
    "required": true,
    "completed": true,
    "ok_models": ["BAAI/bge-m3", "Qwen/Qwen3-4B-Instruct-2507"],
    "capabilities": {
      "BAAI/bge-m3": {"text-embedding": true},
      "Qwen/Qwen3-4B-Instruct-2507": {"chat-completion": true}
    }
  },
  "chat_batch_queues": [{"model": "Qwen/Qwen3-4B-Instruct-2507", "size": 0, "max_size": 64}],
  "embedding_batch_queues": [{"model": "BAAI/bge-m3", "size": 2, "max_size": 64}],
  "runtime_config": {...}
}
```

**Status Codes:**
- `200 OK`: Service healthy
- `503 Service Unavailable`: Warmup incomplete or failures detected

---

### GET /metrics

Prometheus metrics endpoint.

**Key Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `embedding_requests_total{model,status}` | Counter | Total embedding requests |
| `embedding_request_latency_seconds{model}` | Histogram | Embedding latency |
| `embedding_request_queue_wait_seconds{model}` | Histogram | Queue wait time |
| `embedding_cache_hits_total{model}` | Counter | Cache hits |
| `embedding_cache_misses_total{model}` | Counter | Cache misses |
| `chat_requests_total{model,status}` | Counter | Total chat requests |
| `chat_request_latency_seconds{model}` | Histogram | Chat latency |
| `chat_batch_size{model}` | Histogram | Batch size distribution |
| `audio_requests_total{model,status}` | Counter | Total audio requests |
| `audio_request_latency_seconds{model}` | Histogram | Audio latency |
| `remote_image_rejections_total{reason}` | Counter | Rejected remote images |

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

**Common Status Codes:**

| Code | Description |
|------|-------------|
| `400 Bad Request` | Invalid request parameters or unsupported operation |
| `404 Not Found` | Model not found |
| `422 Unprocessable Entity` | Request validation failed |
| `429 Too Many Requests` | Queue full or timeout; includes `Retry-After` header |
| `499 Client Closed Request` | Client disconnected before response |
| `500 Internal Server Error` | Unexpected server error |
| `503 Service Unavailable` | Service shutting down or unhealthy |
| `504 Gateway Timeout` | Request processing timed out |

---

## Upstream Proxy

For proxy model endpoints (`owned_by: openai` or `owned_by: vllm`), additional status codes:

| Code | Description |
|------|-------------|
| `502 Bad Gateway` | Upstream HTTP error |
| `504 Gateway Timeout` | Upstream request timed out |

See [upstream_proxy.md](upstream_proxy.md) for detailed proxy configuration.

