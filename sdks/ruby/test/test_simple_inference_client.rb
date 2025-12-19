# frozen_string_literal: true

require "json"
require "test_helper"

class TestSimpleInferenceClient < Minitest::Test
  def test_chat_completions_sends_to_openai_path
    adapter = Class.new do
      attr_reader :last_request

      def call(env)
        @last_request = env
        {
          status: 200,
          headers: { "content-type" => "application/json" },
          body: '{"ok":true}',
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", api_key: "secret", adapter: adapter)
    client.chat_completions(model: "foo", messages: [])

    assert_equal :post, adapter.last_request[:method]
    assert_equal "http://example.com/v1/chat/completions", adapter.last_request[:url]
    body = JSON.parse(adapter.last_request[:body])
    assert_equal "foo", body["model"]
  end

  def test_parses_json_responses_into_hashes
    adapter = Class.new do
      def call(_env)
        {
          status: 200,
          headers: { "content-type" => "application/json" },
          body: '{"ok":true}',
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", api_key: "secret", adapter: adapter)
    response = client.embeddings(model: "foo", input: "bar")

    assert_equal 200, response[:status]
    assert_equal({ "ok" => true }, response[:body])
  end

  def test_healthy_helper
    adapter = Class.new do
      def call(_env)
        {
          status: 200,
          headers: { "content-type" => "application/json" },
          body: '{"status":"ok"}',
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", adapter: adapter)
    assert_equal true, client.healthy?
  end

  def test_raises_http_error_on_non_2xx_when_raise_on_error_true
    adapter = Class.new do
      def call(_env)
        {
          status: 500,
          headers: { "content-type" => "application/json" },
          body: '{"error":"boom"}',
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", adapter: adapter)

    error = assert_raises(SimpleInference::Errors::HTTPError) do
      client.embeddings(model: "foo", input: "bar")
    end
    assert_equal 500, error.status
    assert_includes error.message, "boom"
  end

  def test_chat_completions_stream_yields_parsed_sse_events
    adapter = Class.new do
      attr_reader :last_request

      def call_stream(env)
        @last_request = env

        # Intentionally chunked in odd boundaries to exercise buffering.
        sse = +""
        sse << %(data: {"id":"evt1","choices":[{"delta":{"content":"Hel"}}]}\n\n)
        sse << %(data: {"id":"evt1","choices":[{"delta":{"content":"lo"}}]}\n\n)
        sse << "data: [DONE]\n\n"

        chunks = [
          sse[0, 7],
          sse[7, 13],
          sse[20..],
        ]

        chunks.compact.each do |chunk|
          yield chunk
        end

        {
          status: 200,
          headers: { "content-type" => "text/event-stream" },
          body: nil,
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", adapter: adapter)
    events = client.chat_completions_stream(model: "foo", messages: []).to_a

    assert_equal(
      [
        { "id" => "evt1", "choices" => [{ "delta" => { "content" => "Hel" } }] },
        { "id" => "evt1", "choices" => [{ "delta" => { "content" => "lo" } }] },
      ],
      events
    )

    req_body = JSON.parse(adapter.last_request[:body])
    assert_equal true, req_body["stream"]
    assert_includes adapter.last_request[:headers]["Accept"], "text/event-stream"
  end

  def test_chat_completions_stream_skips_empty_data_events
    adapter = Class.new do
      def call_stream(_env)
        sse = +""
        sse << "data:\n\n"
        sse << %(data: {"id":"evt1","choices":[{"delta":{"content":"ok"}}]}\n\n)
        sse << "data: [DONE]\n\n"

        yield sse

        {
          status: 200,
          headers: { "content-type" => "text/event-stream" },
          body: nil,
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", adapter: adapter)
    events = client.chat_completions_stream(model: "foo", messages: []).to_a

    assert_equal(
      [
        { "id" => "evt1", "choices" => [{ "delta" => { "content" => "ok" } }] },
      ],
      events
    )
  end

  def test_chat_completions_stream_falls_back_when_streaming_unsupported
    adapter = Class.new do
      attr_reader :stream_request, :requests

      def initialize
        @requests = []
      end

      def call_stream(env)
        @stream_request = env
        {
          status: 400,
          headers: { "content-type" => "application/json" },
          body: '{"detail":"Streaming responses are not supported yet"}',
        }
      end

      def call(env)
        @requests << env
        {
          status: 200,
          headers: { "content-type" => "application/json" },
          body: JSON.generate(
            {
              id: "chatcmpl-1",
              object: "chat.completion",
              created: 123,
              model: "foo",
              choices: [
                {
                  index: 0,
                  message: { role: "assistant", content: "hello" },
                  finish_reason: "stop",
                },
              ],
              usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }
          ),
        }
      end
    end.new

    client = SimpleInference::Client.new(base_url: "http://example.com", adapter: adapter)

    yielded = []
    response =
      client.chat_completions_stream(model: "foo", messages: []) do |event|
        yielded << event
      end

    assert_equal 200, response[:status]
    assert_equal 1, yielded.length
    assert_equal "chat.completion.chunk", yielded[0]["object"]
    assert_equal "hello", yielded[0].dig("choices", 0, "delta", "content")

    stream_body = JSON.parse(adapter.stream_request[:body])
    assert_equal true, stream_body["stream"]

    fallback_body = JSON.parse(adapter.requests.first[:body])
    assert_equal false, fallback_body.key?("stream")
  end
end
