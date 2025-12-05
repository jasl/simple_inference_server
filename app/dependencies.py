from fastapi import HTTPException, Request

from app.models.registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    """Return the per-app ModelRegistry instance.

    All services are expected to run under FastAPI, so we only read from
    request.app.state and no longer fall back to module-level globals.
    """
    registry = getattr(request.app.state, "model_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialized")
    return registry
