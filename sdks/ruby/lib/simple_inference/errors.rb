# frozen_string_literal: true

module SimpleInference
  module Errors
    class Error < StandardError; end

    class ConfigurationError < Error; end

    class HTTPError < Error
      attr_reader :status, :headers, :body

      def initialize(message, status:, headers:, body:)
        super(message)
        @status = status
        @headers = headers
        @body = body
      end
    end

    class TimeoutError < Error; end
    class ConnectionError < Error; end
    class DecodeError < Error; end
  end
end


