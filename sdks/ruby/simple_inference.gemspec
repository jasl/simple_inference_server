#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "lib/simple_inference/version"

Gem::Specification.new do |spec|
  spec.name          = "simple_inference_sdk"
  spec.version       = SimpleInference::VERSION
  spec.summary       = "Ruby client for the Simple Inference Server"
  spec.description   = "Fiber-friendly Ruby SDK for the Simple Inference Server APIs (chat, embeddings, audio, rerank, health)."
  spec.authors       = ["simple_inference_server maintainers"]
  spec.email         = ["dev@example.com"]

  spec.files         = Dir["lib/**/*", "README.md"]
  spec.require_paths = ["lib"]

  spec.homepage      = "https://github.com/your-org/simple_inference_server"
  spec.license       = "MIT"

  spec.metadata["homepage_uri"]    = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage

  spec.required_ruby_version = ">= 3.2"

  spec.add_development_dependency "rspec", "~> 3.13"
end


