"""Centralized configuration management using Pydantic Settings.

All environment variables are defined here with their defaults and validation.
Import `settings` singleton from this module instead of using os.getenv() directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    model_config_path: str = "configs/model_config.yaml"
    models: str = ""  # Comma-separated list of model IDs to load
    model_device: str = "auto"  # cpu, cuda, cuda:<idx>, mps, auto
    auto_download_models: bool = True
    hf_home: str = ""  # Falls back to ./models if not set

    # -------------------------------------------------------------------------
    # Global concurrency settings
    # -------------------------------------------------------------------------
    max_concurrent: int = 4
    max_queue_size: int = 64
    queue_timeout_sec: float = 2.0

    # -------------------------------------------------------------------------
    # Per-capability concurrency (falls back to global if not set)
    # -------------------------------------------------------------------------
    # Embedding
    embedding_max_concurrent: int | None = None
    embedding_max_queue_size: int | None = None
    embedding_queue_timeout_sec: float | None = None
    embedding_max_workers: int = 4
    embedding_count_max_workers: int = 2

    # Chat
    chat_max_concurrent: int | None = None
    chat_max_queue_size: int | None = None
    chat_queue_timeout_sec: float | None = None
    chat_max_workers: int = 4
    chat_count_max_workers: int = 2
    chat_count_use_chat_executor: bool = False

    # Vision
    vision_max_workers: int = 2

    # Audio
    audio_max_concurrent: int | None = None
    audio_max_queue_size: int | None = None
    audio_queue_timeout_sec: float | None = None
    audio_max_workers: int = 1

    # -------------------------------------------------------------------------
    # Embedding batching
    # -------------------------------------------------------------------------
    enable_embedding_batching: bool = True
    embedding_batch_window_ms: float = 6.0
    embedding_batch_window_max_size: int | None = None  # Falls back to max_batch_size
    embedding_batch_queue_size: int | None = None  # Falls back to max_queue_size
    embedding_batch_queue_timeout_sec: float | None = None  # Falls back to queue_timeout_sec
    embedding_cache_size: int = 256
    embedding_usage_disable_token_count: bool = False
    embedding_generate_timeout_sec: float = 60.0

    # -------------------------------------------------------------------------
    # Chat batching
    # -------------------------------------------------------------------------
    enable_chat_batching: bool = True
    chat_batch_window_ms: float = 10.0
    chat_batch_max_size: int = 8
    chat_batch_queue_size: int = 64
    chat_batch_allow_vision: bool = False
    chat_max_prompt_tokens: int = 4096
    chat_max_new_tokens: int = 2048

    # Chat batch queue management
    chat_queue_max_wait_ms: float = 2000.0
    chat_requeue_retries: int = 3
    chat_requeue_base_delay_ms: float = 5.0
    chat_requeue_max_delay_ms: float = 100.0
    chat_requeue_max_wait_ms: float = 2000.0
    chat_requeue_max_tasks: int = 64

    # Chat timeouts
    chat_prepare_timeout_sec: float = 10.0
    chat_generate_timeout_sec: float = 60.0
    chat_oom_cooldown_sec: float = 300.0  # 5 minutes

    # -------------------------------------------------------------------------
    # Input/output limits
    # -------------------------------------------------------------------------
    max_batch_size: int = 32
    max_text_chars: int = 20000
    max_new_tokens: int = 512
    max_audio_bytes: int = 25 * 1024 * 1024  # 25MB

    # -------------------------------------------------------------------------
    # Audio/Whisper settings
    # -------------------------------------------------------------------------
    audio_process_timeout_sec: float = 180.0
    whisper_use_subprocess: bool = False
    whisper_subprocess_idle_sec: float = 0.0
    whisper_subprocess_poll_interval_sec: float = 0.05
    whisper_subprocess_max_wall_sec: float | None = None

    # -------------------------------------------------------------------------
    # Vision/remote image settings
    # -------------------------------------------------------------------------
    allow_remote_images: bool = False
    remote_image_timeout: float = 5.0
    max_remote_image_bytes: int = 5 * 1024 * 1024  # 5MB
    remote_image_host_allowlist: str = ""  # Comma-separated
    remote_image_mime_allowlist: str = "image/png,image/jpeg,image/webp,image/gif"

    # -------------------------------------------------------------------------
    # Warmup settings
    # -------------------------------------------------------------------------
    enable_warmup: bool = True
    warmup_batch_size: int = 1
    warmup_steps: int = 1
    warmup_inference_mode: bool = True
    warmup_vram_budget_mb: float = 0.0
    warmup_vram_per_worker_mb: float = 1024.0
    warmup_allowlist: str = ""  # Comma-separated
    warmup_skiplist: str = ""  # Comma-separated

    # -------------------------------------------------------------------------
    # Executor and timeout settings
    # -------------------------------------------------------------------------
    executor_grace_period_sec: float = 2.0

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------
    enable_metrics: bool = True
    log_level: str = "INFO"

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    trust_remote_code_allowlist: str = ""  # Comma-separated list of repo IDs or model names

    # -------------------------------------------------------------------------
    # Computed properties with fallbacks
    # -------------------------------------------------------------------------
    @property
    def effective_embedding_max_concurrent(self) -> int:
        return self.embedding_max_concurrent if self.embedding_max_concurrent is not None else self.max_concurrent

    @property
    def effective_embedding_max_queue_size(self) -> int:
        return self.embedding_max_queue_size if self.embedding_max_queue_size is not None else self.max_queue_size

    @property
    def effective_embedding_queue_timeout_sec(self) -> float:
        return self.embedding_queue_timeout_sec if self.embedding_queue_timeout_sec is not None else self.queue_timeout_sec

    @property
    def effective_chat_max_concurrent(self) -> int:
        return self.chat_max_concurrent if self.chat_max_concurrent is not None else self.max_concurrent

    @property
    def effective_chat_max_queue_size(self) -> int:
        return self.chat_max_queue_size if self.chat_max_queue_size is not None else self.max_queue_size

    @property
    def effective_chat_queue_timeout_sec(self) -> float:
        return self.chat_queue_timeout_sec if self.chat_queue_timeout_sec is not None else self.queue_timeout_sec

    @property
    def effective_audio_max_concurrent(self) -> int:
        return self.audio_max_concurrent if self.audio_max_concurrent is not None else self.max_concurrent

    @property
    def effective_audio_max_queue_size(self) -> int:
        return self.audio_max_queue_size if self.audio_max_queue_size is not None else self.max_queue_size

    @property
    def effective_audio_queue_timeout_sec(self) -> float:
        return self.audio_queue_timeout_sec if self.audio_queue_timeout_sec is not None else self.queue_timeout_sec

    @property
    def effective_embedding_batch_max_size(self) -> int:
        return self.embedding_batch_window_max_size if self.embedding_batch_window_max_size is not None else self.max_batch_size

    @property
    def effective_embedding_batch_queue_size(self) -> int:
        return self.embedding_batch_queue_size if self.embedding_batch_queue_size is not None else self.max_queue_size

    @property
    def effective_embedding_batch_queue_timeout_sec(self) -> float:
        return self.embedding_batch_queue_timeout_sec if self.embedding_batch_queue_timeout_sec is not None else self.queue_timeout_sec

    @property
    def remote_image_host_allowlist_set(self) -> set[str]:
        return {h.strip() for h in self.remote_image_host_allowlist.split(",") if h.strip()}

    @property
    def remote_image_mime_allowlist_set(self) -> set[str]:
        return {m.strip().lower() for m in self.remote_image_mime_allowlist.split(",") if m.strip()}

    @property
    def trust_remote_code_allowlist_set(self) -> set[str]:
        return {item.strip() for item in self.trust_remote_code_allowlist.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance.

    The settings are loaded once and cached for the lifetime of the process.
    Call get_settings.cache_clear() to reload settings from environment.
    """
    return Settings()


class _SettingsProxy:
    """Proxy object that delegates to get_settings() for lazy access.

    This allows tests to modify environment variables and clear the cache
    without needing to reload modules that imported `settings`.

    Type hints are provided via __class_getitem__ and actual delegation
    happens via __getattr__.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> object:
        return getattr(get_settings(), name)


# Cast the proxy to Settings type for static analysis
# At runtime, all attribute access goes through __getattr__ to get_settings()
settings: Settings = _SettingsProxy()  # type: ignore[assignment]
