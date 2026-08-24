"""
src/api/routes/health.py — Liveness and Readiness
==================================================

WHY: A platform health check that returns 200 because the process is running,
while the model failed to load, is worse than no check at all — it hides the
one fact an operator needs. `status` is "ok" only when predictions can
actually be served.
"""

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.api.service import state

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether this instance can serve predictions."""
    model_loaded = state.predictor is not None
    dataset_loaded = bool(state.store and state.store.is_loaded)

    return HealthResponse(
        status="ok" if state.is_ready else "degraded",
        model_loaded=model_loaded,
        model_name=state.settings.MODEL_NAME,
        dataset_loaded=dataset_loaded,
        machines_known=len(state.store.machine_ids) if state.store else 0,
        threshold=state.settings.PREDICTION_THRESHOLD,
        version=state.settings.APP_VERSION,
    )
