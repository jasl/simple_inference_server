# frozen_string_literal: true

require "json"
require_relative "spec_helper"

RSpec.describe SimpleInference::Client do
  let(:adapter_class) do
    Class.new do
      attr_reader :last_request
      attr_accessor :response

      def initialize
        @response = {
          status: 200,
          headers: { "content-type" => "application/json" },
          body: '{"ok":true}',
        }
      end

      def call(env)
        @last_request = env
        @response
      end
    end
  end

  let(:adapter) { adapter_class.new }

  let(:client) do
    described_class.new(
      base_url: "http://example.com",
      api_key: "secret",
      adapter: adapter
    )
  end

  it "sends chat_completions to /v1/chat/completions" do
    client.chat_completions(model: "foo", messages: [])

    expect(adapter.last_request[:method]).to eq(:post)
    expect(adapter.last_request[:url]).to eq("http://example.com/v1/chat/completions")
    body = JSON.parse(adapter.last_request[:body])
    expect(body["model"]).to eq("foo")
  end

  it "parses JSON responses into Ruby hashes" do
    response = client.embeddings(model: "foo", input: "bar")

    expect(response[:status]).to eq(200)
    expect(response[:body]).to eq("ok" => true)
  end

  it "handles health.healthy? helper" do
    adapter.response = {
      status: 200,
      headers: { "content-type" => "application/json" },
      body: '{"status":"ok"}',
    }

    expect(client.healthy?).to eq(true)
  end

  it "raises HTTPError on non-2xx when raise_on_error is true" do
    adapter.response = {
      status: 500,
      headers: { "content-type" => "application/json" },
      body: '{"error":"boom"}',
    }

    expect do
      client.embeddings(model: "foo", input: "bar")
    end.to raise_error(SimpleInference::Errors::HTTPError) do |error|
      expect(error.status).to eq(500)
      expect(error.message).to include("boom")
    end
  end
end


