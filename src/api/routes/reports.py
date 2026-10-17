"""
src/api/routes/reports.py — LLM-Backed Endpoints
=================================================

WHY THIS IS A SEPARATE MODULE:
    An LLM call takes seconds; a prediction takes ~160 ms. Mixing them in one
    handler means the fast path inherits the slow path's latency and failure
    modes. Keeping them apart is what lets /predict stay fast and stay up
    while the model provider is down.

TWO THINGS THIS FILE GETS RIGHT, DELIBERATELY:

  1. **The LLM call runs on a worker thread, and the number of those is
     capped.** LangChain's `.invoke()` is synchronous and blocking. Awaited
     directly inside an async handler it would block the event loop for the
     whole generation — freezing every other request on the worker,
     including /health. A thread keeps the loop free, and a fixed number of
     slots keeps a hung provider from piling threads up (see
     `_start_in_report_slot`).

  2. **A failed report still returns the prediction.** The prediction decides
     whether a technician is dispatched; the narrative is a convenience over
     it. A provider outage degrades this endpoint to 502 *with the prediction
     attached*, never to a bare error.
"""

import asyncio
import threading
from concurrent.futures import Future
from functools import partial
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import HTTP_422, PredictionResponse, ReportRequest, ReportResponse
from src.api.service import state
from src.utils.exceptions import LLMConnectionError, ReportGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["reports"])

# An unresponsive provider must not hold a worker indefinitely. Local models
# are slow but finite; beyond this the caller is better served by an error and
# the prediction than by waiting.
REPORT_TIMEOUT_SECONDS = 120.0

# At most this many provider calls run at once. A call holds its slot until
# its thread actually returns, which can be after the 504: asyncio.wait_for
# stops waiting but cannot stop a thread, and run_in_threadpool hands its
# capacity token back when the wait is cancelled while the thread runs on.
# Measured before this: with a limit of one, four timed-out reports left four
# live threads. Once every slot is held, new reports get 503 until one frees.
REPORT_MAX_CONCURRENT = 4
_report_slots = threading.BoundedSemaphore(REPORT_MAX_CONCURRENT)


def _start_in_report_slot(fn: Callable[[], str]) -> Optional["asyncio.Future[str]"]:
    """
    Run `fn` on its own thread if a report slot is free; None if none is.

    The slot is released by the thread when `fn` returns or raises, never by
    the caller, so a call the caller gave up on still counts against the
    cap. The thread is a daemon so a provider call that never returns cannot
    hold up process exit.
    """
    slots = _report_slots  # the one acquired is the one released
    if not slots.acquire(blocking=False):
        return None
    future: "Future[str]" = Future()
    # Running from the start, so cancelling the wait cannot cancel the record
    # of work that is still going.
    future.set_running_or_notify_cancel()

    def run() -> None:
        # Released before the outcome is set, so a caller that sees this call
        # finish also sees its slot free.
        try:
            result = fn()
        except BaseException as e:
            slots.release()
            future.set_exception(e)
        else:
            slots.release()
            future.set_result(result)

    try:
        threading.Thread(target=run, name="report", daemon=True).start()
    except BaseException:
        slots.release()
        raise
    return asyncio.wrap_future(future)


def _require_service():
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model or dataset not loaded. "
                f"{state.model_error or ''} {state.data_error or ''}".strip()
            ),
        )
    return state.service


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Write a maintenance report for one machine, or answer a question about it.

    Returns 502 if the LLM provider is unreachable, 504 if it is too slow,
    and 503 if every report slot is busy — each with the prediction included
    in the error detail, so the caller still gets the number that matters.
    """
    service = _require_service()

    # Prediction first, and it is fast. If the LLM half fails, this is what
    # gets handed back.
    record = await run_in_threadpool(
        partial(service.explain_machine, request.machine_id, as_of=request.as_of)
    )
    prediction = PredictionResponse(
        **{k: v for k, v in record.items() if k != "context"}
    )

    from src.genai import ReportGenerator

    def _generate() -> str:
        # Passed by keyword rather than splatted: **{"model": ...} lines up
        # against ReportGenerator's positional parameters and a type checker
        # reads it as overriding `llm` and `temperature`.
        #
        # `timeout` is passed to the provider client so the thread itself
        # can stop, not just the wait_for below. How exactly it stops depends
        # on the client (see get_llm): OpenAI and langchain-ollama cut the
        # socket at the deadline; Google and the community Ollama client may
        # run past it. Either way the thread holds its slot until it returns.
        if request.model:
            generator = ReportGenerator(
                provider=request.provider,
                model=request.model,
                timeout=REPORT_TIMEOUT_SECONDS,
            )
        else:
            generator = ReportGenerator(
                provider=request.provider, timeout=REPORT_TIMEOUT_SECONDS
            )
        if request.question:
            return generator.answer_question(record, request.question)
        return generator.generate_report(record)

    work = _start_in_report_slot(_generate)
    if work is None:
        logger.warning(
            f"Report for machine {request.machine_id} refused: all "
            f"{REPORT_MAX_CONCURRENT} report slots are busy"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"All {REPORT_MAX_CONCURRENT} report slots are busy; try again "
                f"later. The prediction is unaffected: machine "
                f"{prediction.machine_id} probability "
                f"{prediction.failure_probability:.4f} ({prediction.risk_level})."
            ),
        )

    try:
        report = await asyncio.wait_for(work, timeout=REPORT_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        # Stopped waiting — not stopped working. The thread keeps its slot
        # until it returns, and a hosted provider may finish (and bill) the
        # generation regardless.
        logger.error(
            f"Report generation timed out after {REPORT_TIMEOUT_SECONDS:.0f}s "
            f"for machine {request.machine_id}; its provider call keeps a "
            f"report slot until it returns"
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
        raise HTTPException(status_code=HTTP_422, detail=str(e))

    return ReportResponse(
        machine_id=request.machine_id,
        report=report,
        prediction=prediction,
        provider=request.provider,
        model=request.model,
    )
