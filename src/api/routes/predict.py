"""
src/api/routes/predict.py — Prediction Endpoints
=================================================

WHY: The fast path. Everything here must stay in the low hundreds of
milliseconds, which is why the service slices to one machine before scoring —
see `src/api/service.py` for the measurement that forces it.

No endpoint in this module calls an LLM. Report generation lives in
routes/reports.py precisely so a slow model can never delay a prediction.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas import (
    ExplainedPrediction,
    FleetSummaryResponse,
    PredictionResponse,
    PredictRequest,
)
from src.api.service import PredictionService, state

router = APIRouter(tags=["predictions"])


def _require_service() -> "PredictionService":
    """
    Return the service, or 503.

    Also narrows the type: `state.is_ready` guarantees the service and store
    are present, but it is a property and a type checker cannot follow it, so
    the check is spelled out here once instead of at every call site.
    """
    if not state.is_ready or state.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The model or dataset is not loaded. Check /health. "
                f"{state.model_error or ''}".strip()
            ),
        )
    return state.service


@router.post("/predict", response_model=PredictionResponse)
def predict_from_readings(request: PredictRequest) -> PredictionResponse:
    """
    Score a machine from readings supplied in the request.

    This is the endpoint a real plant uses: its own sensors are the source of
    truth, not a CSV on the API host.
    """
    if state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded. {state.model_error or ''}".strip(),
        )

    from src.api.service import MachineDataStore, PredictionService

    # Scoring caller-supplied readings needs no dataset, so this endpoint
    # works even when the store failed to load — hence the empty fallback
    # rather than requiring a ready service.
    service = state.service or PredictionService(
        state.predictor, state.store or MachineDataStore({})
    )
    return PredictionResponse(**service.predict_from_readings(request))


@router.get("/machines/{machine_id}/predict", response_model=PredictionResponse)
def predict_stored_machine(machine_id: int) -> PredictionResponse:
    """Score one machine from the dataset this instance has loaded."""
    service = _require_service()
    return PredictionResponse(**service.predict_machine(machine_id))


@router.get("/machines/{machine_id}/explain", response_model=ExplainedPrediction)
def explain_stored_machine(
    machine_id: int,
    history_hours: int = Query(24, ge=0, le=168),
) -> ExplainedPrediction:
    """
    Score one machine and return the evidence behind the score.

    This is the payload the GenAI layer and the dashboard both consume — the
    sensor readings, their baselines, and which of them are actually
    concerning.
    """
    service = _require_service()
    record = service.explain_machine(machine_id, history_hours=history_hours)
    context = record.pop("context", {})
    context.pop("recent_readings", None)
    return ExplainedPrediction(**record, **context)


@router.get("/machines/{machine_id}/history", response_model=list)
def machine_history(
    machine_id: int,
    hours: int = Query(48, ge=1, le=720),
) -> list:
    """Recent hourly sensor readings for one machine."""
    service = _require_service()
    store = service.store
    store.require_machine(machine_id)

    telemetry = store.dataset["telemetry"]
    rows = (
        telemetry[telemetry["machine_id"] == machine_id]
        .sort_values("datetime")
        .tail(hours)
    )
    return [
        {
            "datetime": row["datetime"].isoformat(),
            "voltage": round(float(row["voltage"]), 2),
            "rotation": round(float(row["rotation"]), 2),
            "pressure": round(float(row["pressure"]), 2),
            "vibration": round(float(row["vibration"]), 2),
        }
        for _, row in rows.iterrows()
    ]


@router.get("/fleet", response_model=FleetSummaryResponse)
def fleet_status(
    alerts_only: bool = Query(False, description="Only machines at or above threshold"),
    refresh: bool = Query(False, description="Bypass the cache"),
) -> FleetSummaryResponse:
    """
    Score every machine, most urgent first.

    Cached for 5 minutes: scoring 100 machines takes ~16 s, which is too slow
    per request and too useful to leave out.
    """
    service = _require_service()
    results = service.fleet(force=refresh)
    alerting = sum(1 for r in results if r["will_fail"])

    shown = [r for r in results if r["will_fail"]] if alerts_only else results
    return FleetSummaryResponse(
        machines_assessed=len(results),
        machines_alerting=alerting,
        threshold=state.settings.PREDICTION_THRESHOLD,
        generated_at=datetime.now(timezone.utc),
        predictions=[PredictionResponse(**r) for r in shown],
    )
