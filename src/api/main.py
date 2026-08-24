"""
src/api/main.py — FastAPI Application
======================================

WHY THIS FILE EXISTS:
    Everything built so far — the model, the inference pipeline, the report
    generator — is a Python library. This makes it a service.

HOW IT WORKS:
    A lifespan handler loads the model and dataset ONCE at startup. Loading
    per request would add ~2 s to every call for the model alone, and the
    dataset is 876,000 rows.

    The exception hierarchy from `src/utils/exceptions.py` finally earns its
    keep here. It was built on Day 1 so that failures could be caught by
    architectural layer, and this is the payoff: one handler per layer maps
    to the right status code, with no string matching on error messages and
    no stack trace crossing the boundary.

      DataValidationError    -> 422  the caller sent something wrong
      PredictionError        -> 422  the input could not be scored
      ResourceNotFoundError  -> 404  no such machine
      ModelNotFoundError     -> 503  this instance cannot serve
      LLMConnectionError     -> 502  an upstream provider is down
      ReportGenerationError  -> 422  the report input was wrong
      anything else          -> 500  generic body, correlation id in the logs

    That last line is the security-relevant one: an unexpected exception
    returns an opaque message and a correlation id. The detail goes to the
    logs, where an operator can find it, and not to a client who may be
    hostile.
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from src.api.routes import health, machines, predict, reports
from src.api.schemas import HTTP_422, ErrorResponse
from src.api.service import state
from src.utils.exceptions import (
    DataValidationError,
    LLMConnectionError,
    ModelNotFoundError,
    PredictionError,
    PredMaintenanceError,
    ReportGenerationError,
    ResourceNotFoundError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and dataset once, and release them on shutdown."""
    logger.info("API starting up — loading model and dataset...")
    state.startup()
    if state.is_ready and state.store is not None:
        logger.info(f"API ready — {len(state.store.machine_ids)} machines.")
    else:
        logger.warning("API started DEGRADED — see /health.")
    yield
    logger.info("API shutting down.")
    state.shutdown()


app = FastAPI(
    title="Predictive Maintenance + GenAI API",
    description=(
        "Predicts equipment failure 24 hours ahead from sensor telemetry, and "
        "turns those predictions into plain-English maintenance reports.\n\n"
        "**Prediction endpoints are fast (~160 ms) and never call an LLM.** "
        "Report generation is isolated in `/report` so a slow or unavailable "
        "language model can never delay or break a prediction."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Restricted to the dashboard origin rather than "*": the API returns
# operational data about physical equipment, and a wildcard would let any
# page a technician visits read it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{settings.DASHBOARD_PORT}",
        f"http://127.0.0.1:{settings.DASHBOARD_PORT}",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _error(status_code: int, detail: str, error_type: str, cid=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            detail=detail, error_type=error_type, correlation_id=cid
        ).model_dump(),
    )


@app.exception_handler(ResourceNotFoundError)
async def _not_found(request: Request, exc: ResourceNotFoundError):
    return _error(status.HTTP_404_NOT_FOUND, str(exc), "ResourceNotFoundError")


@app.exception_handler(ModelNotFoundError)
async def _model_missing(request: Request, exc: ModelNotFoundError):
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc), "ModelNotFoundError")


@app.exception_handler(LLMConnectionError)
async def _llm_down(request: Request, exc: LLMConnectionError):
    # 502, not 500: the failure is upstream of us, and it is retryable.
    return _error(status.HTTP_502_BAD_GATEWAY, str(exc), "LLMConnectionError")


@app.exception_handler(DataValidationError)
async def _bad_data(request: Request, exc: DataValidationError):
    return _error(HTTP_422, str(exc), "DataValidationError")


@app.exception_handler(PredictionError)
async def _bad_prediction(request: Request, exc: PredictionError):
    return _error(HTTP_422, str(exc), "PredictionError")


@app.exception_handler(ReportGenerationError)
async def _bad_report(request: Request, exc: ReportGenerationError):
    return _error(HTTP_422, str(exc), "ReportGenerationError")


@app.exception_handler(PredMaintenanceError)
async def _catch_all_ours(request: Request, exc: PredMaintenanceError):
    """Any of our own errors not handled above."""
    logger.error(f"Unhandled project error on {request.url.path}: {exc}")
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc), type(exc).__name__)


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    """
    Anything we did not anticipate.

    The client gets an opaque message and an id; the logs get the detail. A
    stack trace in an HTTP response tells an attacker about file paths,
    library versions, and internal structure.
    """
    correlation_id = str(uuid.uuid4())[:8]
    logger.error(
        f"[{correlation_id}] Unexpected error on {request.method} "
        f"{request.url.path}: {exc}",
        exc_info=True,
    )
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An internal error occurred. Quote the correlation id when reporting it.",
        "InternalServerError",
        cid=correlation_id,
    )


app.include_router(health.router)
app.include_router(machines.router)
app.include_router(predict.router)
app.include_router(reports.router)


@app.get("/", include_in_schema=False)
def root():
    """Point a browser at something useful."""
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
