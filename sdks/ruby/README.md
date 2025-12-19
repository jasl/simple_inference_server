## simple_inference Ruby SDK

Fiber-friendly Ruby client for the Simple Inference Server APIs (chat, embeddings, audio, rerank, health), designed to work well inside Rails apps and background jobs.

### Installation

Add the gem to your Rails application's `Gemfile`, pointing at this repository path:

```ruby
gem "simple_inference", path: "sdks/ruby"
```

Then run:

```bash
bundle install
```

### Configuration

You can configure the client via environment variables:

- `SIMPLE_INFERENCE_BASE_URL`: e.g. `http://localhost:8000`
- `SIMPLE_INFERENCE_API_KEY`: optional, if your deployment requires auth (sent as `Authorization: Bearer <token>`).
- `SIMPLE_INFERENCE_TIMEOUT`, `SIMPLE_INFERENCE_OPEN_TIMEOUT`, `SIMPLE_INFERENCE_READ_TIMEOUT` (seconds).
- `SIMPLE_INFERENCE_RAISE_ON_ERROR`: `true`/`false` (default `true`).

Or explicitly when constructing a client:

```ruby
client = SimpleInference::Client.new(
  base_url: "http://localhost:8000",
  api_key:  ENV["SIMPLE_INFERENCE_API_KEY"],
  timeout:  30.0
)
```

For convenience, you can also use the module constructor:

```ruby
client = SimpleInference.new(base_url: "http://localhost:8000")
```

### Rails integration example

Create an initializer, for example `config/initializers/simple_inference.rb`:

```ruby
SIMPLE_INFERENCE_CLIENT = SimpleInference::Client.new(
  base_url: ENV.fetch("SIMPLE_INFERENCE_BASE_URL", "http://localhost:8000"),
  api_key:  ENV["SIMPLE_INFERENCE_API_KEY"]
)
```

Then in a controller:

```ruby
class ChatsController < ApplicationController
  def create
    result = SIMPLE_INFERENCE_CLIENT.chat_completions(
      model:    "local-llm",
      messages: [
        { "role" => "user", "content" => params[:prompt] }
      ]
    )

    render json: result[:body], status: result[:status]
  end
end
```

You can also use the client in background jobs:

```ruby
class EmbedJob < ApplicationJob
  queue_as :default

  def perform(text)
    result = SIMPLE_INFERENCE_CLIENT.embeddings(
      model: "bge-m3",
      input: text
    )

    vector = result[:body]["data"].first["embedding"]
    # TODO: persist the vector (e.g. in DB or a vector store)
  end
end
```

And for health checks / maintenance tasks:

```ruby
if SIMPLE_INFERENCE_CLIENT.healthy?
  Rails.logger.info("Inference server is healthy")
else
  Rails.logger.warn("Inference server is unhealthy")
end

models = SIMPLE_INFERENCE_CLIENT.list_models
Rails.logger.info("Available models: #{models[:body].inspect}")
```

### API methods

- `client.chat_completions(params)` → `POST /v1/chat/completions`
- `client.embeddings(params)` → `POST /v1/embeddings`
- `client.rerank(params)` → `POST /v1/rerank`
- `client.list_models` → `GET /v1/models`
- `client.health` → `GET /health`
- `client.healthy?` → boolean helper based on `/health`
- `client.audio_transcriptions(params)` → `POST /v1/audio/transcriptions`
- `client.audio_translations(params)` → `POST /v1/audio/translations`

All methods follow a Receive-an-Object / Return-an-Object style:

- Input: a Ruby `Hash` (keys can be strings or symbols).
- Output: a `Hash` with keys:
  - `:status` – HTTP status code
  - `:headers` – response headers (lowercased keys)
  - `:body` – parsed JSON (Ruby `Hash`) when the response is JSON, or a `String` for text bodies.

### Error handling

By default (`raise_on_error: true`) non-2xx HTTP responses raise:

- `SimpleInference::Errors::HTTPError` – wraps status, headers and raw body.

Network and parsing errors are mapped to:

- `SimpleInference::Errors::TimeoutError`
- `SimpleInference::Errors::ConnectionError`
- `SimpleInference::Errors::DecodeError`

If you prefer to handle HTTP error codes manually, disable raising:

```ruby
client = SimpleInference::Client.new(
  base_url: "http://localhost:8000",
  raise_on_error: false
)

response = client.embeddings(model: "local-embed", input: "hello")
if response[:status] == 200
  # happy path
else
  Rails.logger.warn("Embedding call failed: #{response[:status]} #{response[:body].inspect}")
end
```

### Using with OpenAI and compatible services

Because this SDK follows the OpenAI-style HTTP paths (`/v1/chat/completions`, `/v1/embeddings`, etc.), you can also point it directly at OpenAI or other compatible inference services.

#### Connect to OpenAI

```ruby
client = SimpleInference::Client.new(
  base_url: "https://api.openai.com",
  api_key:  ENV["OPENAI_API_KEY"]
)

response = client.chat_completions(
  model:    "gpt-4.1-mini",
  messages: [{ "role" => "user", "content" => "Hello" }]
)

pp response[:body]
```

#### Streaming chat completions (SSE)

For OpenAI-style streaming (`text/event-stream`), use `chat_completions_stream`. It yields parsed JSON events (Ruby `Hash`), so you can consume deltas incrementally:

```ruby
client.chat_completions_stream(
  model: "gpt-4.1-mini",
  messages: [{ "role" => "user", "content" => "Hello" }]
) do |event|
  delta = event.dig("choices", 0, "delta", "content")
  print delta if delta
end
puts
```

If you prefer, it also returns an Enumerator:

```ruby
client.chat_completions_stream(model: "gpt-4.1-mini", messages: [...]).each do |event|
  # ...
end
```

Fallback behavior:

- If the upstream service does **not** support streaming (for example, this repo's server currently returns `400` with `{"detail":"Streaming responses are not supported yet"}`), the SDK will **retry non-streaming** and yield a **single synthetic chunk** so your streaming consumer code can still run.

#### Connect to any OpenAI-compatible endpoint

For services that expose an OpenAI-compatible API (same paths and payloads), point `base_url` at that service and provide the correct token:

```ruby
client = SimpleInference::Client.new(
  base_url: "https://my-openai-compatible.example.com",
  api_key:  ENV["MY_SERVICE_TOKEN"]
)
```

If the service uses a non-standard header instead of `Authorization: Bearer`, you can omit `api_key` and pass headers explicitly:

```ruby
client = SimpleInference::Client.new(
  base_url: "https://my-service.example.com",
  headers: {
    "x-api-key" => ENV["MY_SERVICE_KEY"]
  }
)
```

### Puma vs Falcon (Fiber / Async) usage

The default HTTP adapter uses Ruby's `Net::HTTP` and is safe to use under Puma's multithreaded model:

- No global mutable state
- Per-client configuration only
- Blocking IO that integrates with Ruby 3 Fiber scheduler

For Falcon / async environments, you can keep the default adapter, or use the optional HTTPX adapter (requires the `httpx` gem):

```ruby
gem "httpx" # optional, only required when using the HTTPX adapter
```

You can then use the optional HTTPX adapter shipped with this gem:

```ruby
adapter = SimpleInference::HTTPAdapter::HTTPX.new(timeout: 30.0)

SIMPLE_INFERENCE_CLIENT =
  SimpleInference::Client.new(
    base_url: ENV.fetch("SIMPLE_INFERENCE_BASE_URL", "http://localhost:8000"),
    api_key:  ENV["SIMPLE_INFERENCE_API_KEY"],
    adapter:  adapter
  )
```
