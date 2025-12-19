# frozen_string_literal: true

require "net/http"
require "uri"

module SimpleInference
  module HTTPAdapter
    # Optional adapters are lazily loaded so the SDK has no hard runtime deps.
    autoload :HTTPX, "simple_inference/http_adapter/httpx"

    # Default synchronous HTTP adapter built on Net::HTTP.
    # It is compatible with Ruby 3 Fiber scheduler and keeps the interface
    # minimal so it can be swapped out for custom adapters (HTTPX, async-http, etc.).
    class Default
      def call(request)
        uri = URI.parse(request.fetch(:url))

        http = Net::HTTP.new(uri.host, uri.port)
        http.use_ssl = uri.scheme == "https"

        timeout = request[:timeout]
        open_timeout = request[:open_timeout] || timeout
        read_timeout = request[:read_timeout] || timeout

        http.open_timeout = open_timeout if open_timeout
        http.read_timeout = read_timeout if read_timeout

        klass = http_class_for(request[:method])
        req = klass.new(uri.request_uri)

        headers = request[:headers] || {}
        headers.each do |key, value|
          req[key.to_s] = value
        end

        body = request[:body]
        req.body = body if body

        response = http.request(req)

        {
          status: Integer(response.code),
          headers: response.each_header.to_h,
          body: response.body.to_s,
        }
      end

      # Streaming-capable request helper.
      #
      # When the response is `text/event-stream` (and 2xx), it yields raw body chunks
      # as they arrive via the given block, and returns a response hash with `body: nil`.
      #
      # For non-streaming responses, it behaves like `#call` and returns the full body.
      def call_stream(request)
        return call(request) unless block_given?

        uri = URI.parse(request.fetch(:url))

        http = Net::HTTP.new(uri.host, uri.port)
        http.use_ssl = uri.scheme == "https"

        timeout = request[:timeout]
        open_timeout = request[:open_timeout] || timeout
        read_timeout = request[:read_timeout] || timeout

        http.open_timeout = open_timeout if open_timeout
        http.read_timeout = read_timeout if read_timeout

        klass = http_class_for(request[:method])
        req = klass.new(uri.request_uri)

        headers = request[:headers] || {}
        headers.each do |key, value|
          req[key.to_s] = value
        end

        body = request[:body]
        req.body = body if body

        status = nil
        response_headers = {}
        response_body = +""

        http.request(req) do |response|
          status = Integer(response.code)
          response_headers = response.each_header.to_h

          headers_lc = response_headers.transform_keys { |k| k.to_s.downcase }
          content_type = headers_lc["content-type"]

          if status >= 200 && status < 300 && content_type&.include?("text/event-stream")
            response.read_body do |chunk|
              yield chunk
            end
            response_body = nil
          else
            response_body = response.body.to_s
          end
        end

        {
          status: Integer(status),
          headers: response_headers,
          body: response_body,
        }
      end

      private

      def http_class_for(method)
        case method.to_s.upcase
        when "GET"    then Net::HTTP::Get
        when "POST"   then Net::HTTP::Post
        when "PUT"    then Net::HTTP::Put
        when "PATCH"  then Net::HTTP::Patch
        when "DELETE" then Net::HTTP::Delete
        else
          raise ArgumentError, "Unsupported HTTP method: #{method.inspect}"
        end
      end
    end
  end
end
