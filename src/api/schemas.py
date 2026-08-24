"""
src/api/schemas.py — API Request/Response Models
=================================================

WHY THIS FILE EXISTS:
    The boundary where untrusted input becomes typed data. Every field here
    is validated before it reaches the model, so a malformed payload returns
    422 with a precise complaint rather than a 500 from somewhere deep in
    pandas.

    Keeping the schemas in one module also means the OpenAPI documentation at
    /docs is generated from the same definitions the code validates against —
    the docs cannot drift from the behaviour.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Physical bounds from the data dictionary. A reading outside these is a
# broken sensor or a unit mix-up, and either way should be rejected at the
# door rather than scored — the model has never seen values like that and its
# output would be meaningless rather than merely wrong.
SENSOR_BOUNDS = {
    "voltage": (0.0, 500.0),
    "rotation": (0.0, 1500.0),
    "pressure": (0.0, 400.0),
    "vibration": (0.0, 300.0),
}


class SensorReading(BaseModel):
    """One hourly reading from one machine."""

    datetime: datetime
    voltage: float = Field(..., description="Volts")
    rotation: float = Field(..., description="RPM")
    pressure: float = Field(..., description="PSI")
    vibration: float = Field(..., description="mm/s")

    @field_validator("voltage", "rotation", "pressure", "vibration")
    @classmethod
    def _within_physical_bounds(cls, value: float, info) -> float:
        low, high = SENSOR_BOUNDS[info.field_name]
        if not low <= value <= high:
            raise ValueError(
                f"{info.field_name}={value} is outside the plausible range "
                f"[{low}, {high}]. Check the sensor and the units."
            )
        return value


class PredictRequest(BaseModel):
    """Score a machine from readings supplied by the caller."""

    machine_id: int
    readings: List[SensorReading] = Field(
        ...,
        description=(
            "Consecutive hourly readings, oldest first. At least 48 are "
            "needed: feature engineering consumes the first 24 for rolling "
            "and lag windows, and the LSTM needs 24 more to form a sequence."
        ),
    )
    model: Optional[str] = Field(
        None, description="Machine model (e.g. model3). Improves accuracy if known."
    )
    age: Optional[int] = Field(None, ge=0, le=100, description="Machine age in years")

    @field_validator("readings")
    @classmethod
    def _enough_history(cls, readings: List[SensorReading]) -> List[SensorReading]:
        if len(readings) < 48:
            raise ValueError(
                f"{len(readings)} readings supplied; at least 48 hours of history "
                "are required to build a single prediction window."
            )
        return readings


class SensorEvidence(BaseModel):
    """What one sensor is doing, and whether it matters."""

    current: float
    baseline_24h: float
    change_24h: float
    volatility_24h: float
    deviation_sigma: float
    unit: str
    direction: str
    is_concerning: bool
    typical_cause: Optional[str] = None


class PredictionResponse(BaseModel):
    """A failure prediction for one machine."""

    machine_id: int
    datetime: str
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    will_fail: bool = Field(
        ..., description="True when the probability is at or above the alert threshold"
    )
    threshold: float


class ExplainedPrediction(PredictionResponse):
    """A prediction with the evidence behind it."""

    age_years: Optional[int] = None
    errors_last_24h: int = 0
    hours_since_maintenance: Dict[str, int] = Field(default_factory=dict)
    sensors: Dict[str, SensorEvidence] = Field(default_factory=dict)
    most_deviant_sensors: List[str] = Field(default_factory=list)


class FleetSummaryResponse(BaseModel):
    """Fleet status, most urgent first."""

    machines_assessed: int
    machines_alerting: int
    threshold: float
    generated_at: datetime
    predictions: List[PredictionResponse]


class ReportRequest(BaseModel):
    """Ask for a written maintenance report."""

    machine_id: int
    provider: Optional[Literal["openai", "google", "ollama"]] = None
    model: Optional[str] = Field(
        None,
        description=(
            "Override the provider's model, e.g. an Ollama tag you have pulled. "
            "Without this the provider's configured default is used, which is a "
            "404 if that model is not installed."
        ),
    )
    question: Optional[str] = Field(
        None,
        description="Ask a specific question instead of generating a full report.",
    )


class ReportResponse(BaseModel):
    """A generated report, with the prediction it was written from."""

    machine_id: int
    report: str
    prediction: PredictionResponse
    provider: Optional[str] = None
    model: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness and readiness.

    `status` is "ok" only when the model is loaded and the dataset is
    available. A process that is running but cannot serve predictions is not
    healthy, and reporting otherwise defeats the point of the check.
    """

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    dataset_loaded: bool
    machines_known: int
    threshold: float
    version: str


class MachineInfo(BaseModel):
    """Static facts about one machine."""

    machine_id: int
    model: Optional[str] = None
    age: Optional[int] = None
    readings_available: int
    first_reading: Optional[datetime] = None
    last_reading: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """
    A failure, described without leaking internals.

    `detail` is safe to show a user; the correlation id is what an operator
    greps the logs for. Stack traces never cross this boundary.
    """

    detail: str
    error_type: str
    correlation_id: Optional[str] = None
