"""
src/api/routes/reports.py — LLM-Backed Endpoints
=================================================

WHY THIS IS A SEPARATE MODULE:
    An LLM call takes seconds; a prediction takes ~160 ms. Mixing them in one
    handler means the fast path inherits the slow path's latency and failure
    modes. Keeping them apart is what lets /predict stay fast and stay up
    while the model provider is down.

TWO THINGS THIS FILE GETS RIGHT, DELIBERATELY:

  1. **The LLM call runs in a threadpool.** LangChain's `.invoke()` is
     synchronous and blocking. Awaited directly inside an async handler it
     would block the event loop for the whole generation — freezing every
     other request on the worker, including /health. `run_in_threadpool`
     keeps the loop free.

  2. **A failed report still returns the prediction.** The prediction decides
     whether a technician is dispatched; the narrative is a convenience over
     it. A provider outage degrades this endpoint to 502 *with the prediction
     attached*, never to a bare error.
"""

import asyncio

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import PredictionResponse, ReportRequest, ReportResponse
from src.api.service import state
from src.utils.exceptions import LLMConnectionError, ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["reports"])

# An unresponsive provider must not hold a worker indefinitely. Local models
# are slow but finite; beyond this the caller is better served by an error and
# the prediction than by waiting.
REPORT_TIMEOUT_SECONDS = 120.0


def _require_service():
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model or dataset not loaded. {state.model_error or ''}".strip(),
        )
    return state.service


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Write a maintenance report for one machine, or answer a question about it.

    Returns 502 if the LLM provider is unreachable — with the prediction
    included in the error detail, so the caller still gets the number that
    matters.
    """
    service = _require_service()

    # Prediction first, and it is fast. If the LLM half fails, this is what
    # gets handed back.
    record = await run_in_threadpool(service.explain_machine, request.machine_id)
    prediction = PredictionResponse(
        **{k: v for k, v in record.items() if k != "context"}
    )

    from src.genai import ReportGenerator

    def _generate() -> str:
        llm_kwargs = {"model": request.model} if request.model else {}
        generator = ReportGenerator(provider=request.provider, **llm_kwargs)
        if request.question:
            return generator.answer_question(record, request.question)
        return generator.generate_report(record)

    try:
        # Threadpool so the blocking LangChain call cannot stall the event
        # loop, plus a ceiling so one hung provider cannot hold a worker.
        report = await asyncio.wait_for(
            run_in_threadpool(_generate), timeout=REPORT_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(
            f"Report generation timed out after {REPORT_TIMEOUT_SECONDS:.0f}s "
            f"for machine {request.machine_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"The language model did not respond within "
                f"{REPORT_TIMEOUT_SECONDS:.0f}s. The prediction is unaffected: "
                f"machine {prediction.machine_id} probability "
                f"{prediction.failure_probability:.4f} ({prediction.risk_level})."
            ),
        )
    except LLMConnectionError as e:
        logger.error(f"LLM unavailable for machine {request.machine_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"The language model is unavailable, so no written report could "
                f"be produced. The prediction stands: machine "
                f"{prediction.machine_id} probability "
                f"{prediction.failure_probability:.4f} ({prediction.risk_level}). "
                f"Provider error: {e}"
            ),
        )
    except ReportGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    return ReportResponse(
        machine_id=request.machine_id,
        report=report,
        prediction=prediction,
        provider=request.provider,
        model=request.model,
    )
