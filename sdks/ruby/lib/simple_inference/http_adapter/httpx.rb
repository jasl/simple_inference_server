# frozen_string_literal: true

begin
  require "httpx"
rescue LoadError => e
  raise LoadError,
        "httpx gem is required for SimpleInference::HTTPAdapter::HTTPX (add `gem \"httpx\"`)",
        cause: e
end

module SimpleInference
  module HTTPAdapter
    # Fiber-friendly HTTP adapter built on HTTPX.
    #
    # NOTE: This adapter intentionally does NOT implement `#call_stream`.
    # Streaming consumers will still work via the SDK's full-body SSE parsing
    # fallback path (see SimpleInference::Client#handle_stream_response).
    class HTTPX
      def initialize(timeout: nil)
        @timeout = timeout
      end

      def call(request)
        method = request.fetch(:method).to_s.downcase.to_sym
        url = request.fetch(:url)
        headers = request[:headers] || {}
        body = request[:body]

        client = ::HTTPX

        # Mirror the SDK's timeout semantics:
        # - `:timeout` is the overall request deadline (maps to HTTPX `request_timeout`)
        # - `:open_timeout` and `:read_timeout` override connect/read deadlines
        timeout = request[:timeout] || @timeout
        open_timeout = request[:open_timeout] || timeout
        read_timeout = request[:read_timeout] || timeout

        timeout_opts = {}
        timeout_opts[:request_timeout] = timeout.to_f if timeout
        timeout_opts[:connect_timeout] = open_timeout.to_f if open_timeout
        timeout_opts[:read_timeout] = read_timeout.to_f if read_timeout

        unless timeout_opts.empty?
          client = client.with(timeout: timeout_opts)
        end

        response = client.request(method, url, headers: headers, body: body)

        # HTTPX may return an error response object instead of raising.
        if response.respond_to?(:status) && response.status.to_i == 0
          err = response.respond_to?(:error) ? response.error : nil
          raise Errors::ConnectionError, (err ? err.message : "HTTPX request failed")
        end

        response_headers =
          response.headers.to_h.each_with_object({}) do |(k, v), out|
            out[k.to_s] = v.is_a?(Array) ? v.join(", ") : v.to_s
          end

        {
          status: response.status.to_i,
          headers: response_headers,
          body: response.body.to_s,
        }
      rescue ::HTTPX::TimeoutError => e
        raise Errors::TimeoutError, e.message
      rescue ::HTTPX::Error, IOError, SystemCallError => e
        raise Errors::ConnectionError, e.message
      end
    end
  end
end
