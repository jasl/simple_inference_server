# frozen_string_literal: true

require "json"
require "securerandom"
require "uri"
require "timeout"
require "socket"

module SimpleInference
  class Client
    attr_reader :config, :adapter

    def initialize(options = {})
      @config = Config.new(options || {})
      @adapter = @config.adapter || HTTPAdapter::Default.new
    end

    # POST /v1/chat/completions
    # params: { model: "model-name", messages: [...], ... }
    def chat_completions(params)
      post_json("/v1/chat/completions", params)
    end

    # POST /v1/embeddings
    def embeddings(params)
      post_json("/v1/embeddings", params)
    end

    # POST /v1/rerank
    def rerank(params)
      post_json("/v1/rerank", params)
    end

    # GET /v1/models
    def list_models
      get_json("/v1/models")
    end

    # GET /health
    def health
      get_json("/health")
    end

    # Returns true when service is healthy, false otherwise.
    def healthy?
      response = get_json("/health", raise_on_http_error: false)
      status_ok = response[:status] == 200
      body_status_ok = response.dig(:body, "status") == "ok"
      status_ok && body_status_ok
    rescue Errors::Error
      false
    end

    # POST /v1/audio/transcriptions
    # params: { file: io_or_hash, model: "model-name", **audio_options }
    def audio_transcriptions(params)
      post_multipart("/v1/audio/transcriptions", params)
    end

    # POST /v1/audio/translations
    def audio_translations(params)
      post_multipart("/v1/audio/translations", params)
    end

    private

    def base_url
      config.base_url
    end

    def get_json(path, params: nil, raise_on_http_error: nil)
      full_path = with_query(path, params)
      request_json(
        method: :get,
        path: full_path,
        body: nil,
        expect_json: true,
        raise_on_http_error: raise_on_http_error
      )
    end

    def post_json(path, body, raise_on_http_error: nil)
      request_json(
        method: :post,
        path: path,
        body: body,
        expect_json: true,
        raise_on_http_error: raise_on_http_error
      )
    end

    def request_json(method:, path:, body:, expect_json:, raise_on_http_error:)
      if base_url.nil? || base_url.empty?
        raise Errors::ConfigurationError, "base_url is required"
      end

      url = "#{base_url}#{path}"

      headers = config.headers.merge("Content-Type" => "application/json")
      payload = body.nil? ? nil : JSON.generate(body)

      request_env = {
        method: method,
        url: url,
        headers: headers,
        body: payload,
        timeout: config.timeout,
        open_timeout: config.open_timeout,
        read_timeout: config.read_timeout,
      }

      handle_response(
        request_env,
        expect_json: expect_json,
        raise_on_http_error: raise_on_http_error
      )
    end

    def with_query(path, params)
      return path if params.nil? || params.empty?

      query = URI.encode_www_form(params)
      separator = path.include?("?") ? "&" : "?"
      "#{path}#{separator}#{query}"
    end

    def post_multipart(path, params)
      file_value = params[:file] || params["file"]
      model = params[:model] || params["model"]

      raise Errors::ConfigurationError, "file is required" if file_value.nil?
      raise Errors::ConfigurationError, "model is required" if model.nil? || model.to_s.empty?

      io, filename = normalize_upload(file_value)

      form_fields = {
        "model" => model.to_s,
      }

      # Optional scalar fields
      %i[language prompt response_format temperature].each do |key|
        value = params[key] || params[key.to_s]
        next if value.nil?

        form_fields[key.to_s] = value.to_s
      end

      # timestamp_granularities can be an array or single value
      tgs = params[:timestamp_granularities] || params["timestamp_granularities"]
      if tgs && !tgs.empty?
        Array(tgs).each_with_index do |value, index|
          form_fields["timestamp_granularities[#{index}]"] = value.to_s
        end
      end

      body, headers = build_multipart_body(io, filename, form_fields)

      request_env = {
        method: :post,
        url: "#{base_url}#{path}",
        headers: config.headers.merge(headers),
        body: body,
        timeout: config.timeout,
        open_timeout: config.open_timeout,
        read_timeout: config.read_timeout,
      }

      handle_response(
        request_env,
        expect_json: nil, # auto-detect based on Content-Type
        raise_on_http_error: nil
      )
    ensure
      if io && io.respond_to?(:close)
        begin
          io.close unless io.closed?
        rescue StandardError
          # ignore close errors
        end
      end
    end

    def normalize_upload(file)
      if file.is_a?(Hash)
        io = file[:io] || file["io"]
        filename = file[:filename] || file["filename"] || "audio.wav"
      elsif file.respond_to?(:read)
        io = file
        filename =
          if file.respond_to?(:path) && file.path
            File.basename(file.path)
          else
            "audio.wav"
          end
      else
        raise Errors::ConfigurationError,
              "file must be an IO object or a hash with :io and :filename keys"
      end

      raise Errors::ConfigurationError, "file IO is required" if io.nil?

      [io, filename]
    end

    def build_multipart_body(io, filename, fields)
      boundary = "simple-inference-ruby-#{SecureRandom.hex(12)}"

      headers = {
        "Content-Type" => "multipart/form-data; boundary=#{boundary}",
      }

      body = +""

      fields.each do |name, value|
        body << "--#{boundary}\r\n"
        body << %(Content-Disposition: form-data; name="#{name}"\r\n\r\n)
        body << value.to_s
        body << "\r\n"
      end

      body << "--#{boundary}\r\n"
      body << %(Content-Disposition: form-data; name="file"; filename="#{filename}"\r\n)
      body << "Content-Type: application/octet-stream\r\n\r\n"

      while (chunk = io.read(16_384))
        body << chunk
      end

      body << "\r\n--#{boundary}--\r\n"

      [body, headers]
    end

    def handle_response(request_env, expect_json:, raise_on_http_error:)
      response = @adapter.call(request_env)

      status = response[:status]
      headers = (response[:headers] || {}).transform_keys { |k| k.to_s.downcase }
      body = response[:body].to_s

      # Decide whether to raise on HTTP errors
      raise_on =
        if raise_on_http_error.nil?
          config.raise_on_error
        else
          !!raise_on_http_error
        end

      if raise_on && (status < 200 || status >= 300)
        message = "HTTP #{status}"

        begin
          error_body = JSON.parse(body)
          message = error_body["error"] || error_body["message"] || message
        rescue JSON::ParserError
          # fall back to generic message
        end

        raise Errors::HTTPError.new(
          message,
          status: status,
          headers: headers,
          body: body
        )
      end

      should_parse_json =
        if expect_json.nil?
          content_type = headers["content-type"]
          content_type && content_type.include?("json")
        else
          expect_json
        end

      parsed_body =
        if should_parse_json
          parse_json(body)
        else
          body
        end

      {
        status: status,
        headers: headers,
        body: parsed_body,
      }
    rescue Timeout::Error => e
      raise Errors::TimeoutError, e.message
    rescue SocketError, SystemCallError => e
      raise Errors::ConnectionError, e.message
    end

    def parse_json(body)
      return nil if body.nil? || body.empty?

      JSON.parse(body)
    rescue JSON::ParserError => e
      raise Errors::DecodeError, "Failed to parse JSON response: #{e.message}"
    end
  end
end


