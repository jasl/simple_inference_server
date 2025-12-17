from __future__ import annotations

import contextlib
import logging
import threading
from typing import TYPE_CHECKING, Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from app.utils.device import resolve_device

if TYPE_CHECKING:
    pass  # For future type-only imports

logger = logging.getLogger(__name__)


def resolve_runtime_device(preference: str) -> str:
    """Resolve device preference to a concrete device string.

    Args:
        preference: Device preference string ("auto", "cpu", "cuda", "mps", etc.)

    Returns:
        Resolved device string

        Note:
        This function delegates to app.utils.device.resolve_device. Unlike the
        legacy behavior, non-auto values are validated so misconfigurations fail
        fast instead of silently falling back to CPU.
    """
    pref = preference or "auto"
    return resolve_device(pref, validate=pref != "auto")


def handle_oom(exc: BaseException, model_name: str, device: Any) -> None:
    """Handle CUDA OOM in a consistent, recoverable way.

    Logs the exception, clears CUDA cache if available, and re-raises.
    Chat batching and API layers will interpret the re-raised exception as a
    generic generation failure and surface a 500 error.

    Args:
        exc: The OOM exception
        model_name: Name of the model that triggered OOM
        device: Device the model is running on

    Raises:
        The original exception after cleanup
    """
    logger.exception(
        "chat_generate_oom",
        extra={
            "model": model_name,
            "device": str(device) if device else "unknown",
        },
    )
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    raise exc


class StopOnTokens(StoppingCriteria):
    """Stop generation when any of the provided token sequences is produced."""

    def __init__(self, stop_token_ids: list[list[int]]) -> None:
        super().__init__()
        # Keep only non-empty stop sequences
        self.stop_token_ids = [ids for ids in stop_token_ids if ids]
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        if not self.stop_token_ids:
            return False
        # `transformers.generate` runs stopping criteria at the batch level: returning
        # True stops generation for the *entire* batch. To avoid truncating other
        # requests in a batched call, only stop early when *all* sequences have
        # ended with a stop token sequence (single-item batches behave as usual).
        sequences: list[list[int]] = [input_ids.tolist()] if input_ids.dim() == 1 else input_ids.tolist()

        def _ends_with_any_stop(seq: list[int]) -> bool:
            return any(len(ids) <= len(seq) and seq[-len(ids) :] == ids for ids in self.stop_token_ids)

        if all(_ends_with_any_stop(seq) for seq in sequences):
            self.triggered = True
            return True
        return False


class StopOnCancel(StoppingCriteria):
    """Stop generation when a cancellation event is set."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__()
        self.event = event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        return self.event.is_set()


class StopOnCancelAny(StoppingCriteria):
    """Stop batched generation when any cancellation event is set."""

    def __init__(self, events: list[threading.Event]) -> None:
        super().__init__()
        self.events = events

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: D401
        return any(ev.is_set() for ev in self.events)


def trim_with_stop(text: str, stop: list[str] | None) -> tuple[str, bool]:
    """Trim the generated text at the earliest occurrence of any stop string."""

    if not stop:
        return text, False

    earliest_idx: int | None = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx

    if earliest_idx is None:
        return text, False

    return text[:earliest_idx].rstrip(), True


def build_stop_criteria(
    stop_token_ids: list[list[int]],
    cancel_event: threading.Event | None,
) -> tuple[StoppingCriteriaList | None, StopOnTokens | None]:
    """Build stopping criteria from stop token IDs and an optional cancel event.

    This is a shared helper used by both text and vision chat models.

    Args:
        stop_token_ids: List of token ID sequences to stop on
        cancel_event: Optional threading event to check for cancellation

    Returns:
        Tuple of (StoppingCriteriaList or None, StopOnTokens stopper or None)
    """
    criteria: list[StoppingCriteria] = []
    stopper: StopOnTokens | None = None

    if stop_token_ids:
        stopper = StopOnTokens(stop_token_ids)
        criteria.append(stopper)

    if cancel_event is not None:
        criteria.append(StopOnCancel(cancel_event))

    if not criteria:
        return None, stopper

    return StoppingCriteriaList(criteria), stopper


def normalize_chat_template_output(
    raw_inputs: Any,
    *,
    ensure_2d: bool = False,
    drop_non_tensor: bool = True,
) -> dict[str, torch.Tensor]:
    """Normalize tokenizer/processor outputs to a plain dict of tensors.

    Handles various output formats from tokenizers/processors:
    - Plain dict
    - BatchEncoding (has .data attribute)
    - Namespace-like objects (has .input_ids attribute)

    Args:
        raw_inputs: Output from tokenizer/processor apply_chat_template
        ensure_2d: If True, ensure input_ids/attention_mask are 2D
        drop_non_tensor: If True, drop non-tensor fields

    Returns:
        Dict of tensor values ready for model input
    """
    # Extract the source dict from various formats
    if isinstance(raw_inputs, dict):
        source = raw_inputs
    elif hasattr(raw_inputs, "data") and isinstance(raw_inputs.data, dict):
        # BatchEncoding-style
        source = raw_inputs.data
    elif hasattr(raw_inputs, "input_ids"):
        # Namespace-like objects
        source = {
            "input_ids": raw_inputs.input_ids,
            "attention_mask": getattr(raw_inputs, "attention_mask", None),
        }
    else:
        raise ValueError("Tokenizer/processor returned unexpected format for chat template")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if value is None:
            continue

        if not torch.is_tensor(value):
            if drop_non_tensor:
                logger.debug(
                    "Dropping non-tensor chat template field %s (%s)", key, type(value).__name__
                )
                continue
            raise ValueError(f"Field {key} is not a tensor")

        tensor = value
        if ensure_2d:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() > 2:
                tensor = tensor.view(tensor.shape[0], -1)

        normalized[key] = tensor

    if "input_ids" not in normalized:
        raise ValueError("Output missing 'input_ids' tensor")

    return normalized


