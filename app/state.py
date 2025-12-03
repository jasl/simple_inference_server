from __future__ import annotations

from app.batching import BatchingService
from app.chat_batching import ChatBatchingService
from app.models.registry import ModelRegistry

# Global holder for the loaded ModelRegistry instance.
model_registry: ModelRegistry | None = None
batching_service: BatchingService | None = None
chat_batching_service: ChatBatchingService | None = None
