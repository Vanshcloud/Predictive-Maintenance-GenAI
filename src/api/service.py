"""
src/api/service.py — Shared Application State
==============================================

WHY THIS FILE EXISTS:
    Two things must happen exactly once for the API to be usable, and both
    are too slow to do per request: loading the model (~2 s) and loading the
    dataset (876,000 rows). This module owns both, so routes are thin and
    testable and nothing reloads a 4 GB artifact inside a request handler.

THE SLICING CONSTRAINT — measured, and the reason this module exists:
    `Predictor.explain_machine()` runs `merge_tables` and `engineer_features`
    over whatever dataset it is handed. Handed the whole fleet to score ONE
    machine, that is **over two minutes**. Sliced to that machine and a recent
    window first, the same call is **~160 ms** — a factor of roughly 800.

    So the API never passes the full dataset to the predictor. `slice_for()`
    narrows to one machine and the last N hours before anything is computed.
    That is not an optimisation; without it the endpoint cannot exist.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import get_settings
from src.prediction import Predictor
from src.utils.exceptions import ResourceNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_TABLES = ("telemetry", "machines", "errors", "maintenance")

# Hours of history sliced per request. Feature engineering needs 24 for
# rolling/lag windows and the LSTM needs 24 more; 200 leaves generous margin
# while keeping the per-request cost flat.
DEFAULT_WINDOW_HOURS = 200

# Fleet scoring touches every machine, so it is cached. Sensor data arrives
# hourly, making a 5-minute TTL a reasonable staleness budget.
FLEET_CACHE_TTL_SECONDS = 300


class MachineDataStore:
    """Holds the dataset in memory and slices it per machine."""

    def __init__(self, dataset: Dict[str, pd.DataFrame]) -> None:
        self.dataset = dataset
        machines = dataset.get("machines")
        self.machine_ids: List[int] = (
            sorted(int(m) for m in machines["machine_id"].unique())
            if machines is not None and not machines.empty
            else []
        )

    @classmethod
    def load(cls, data_dir) -> "MachineDataStore":
        """Read the raw tables once, parsing datetimes up front."""
        dataset = {}
        for name in RAW_TABLES:
            path = data_dir / f"{name}.csv"
            if not path.exists():
                logger.warning(f"{path} not found — the API will start degraded.")
                continue
            frame = pd.read_csv(path)
            if "datetime" in frame.columns:
                # Parsed once here rather than on every slice: over 876,000
                # rows this is the difference between a fast slice and a slow one.
                frame["datetime"] = pd.to_datetime(frame["datetime"])
            dataset[name] = frame
        return cls(dataset)

    @property
    def is_loaded(self) -> bool:
        return "telemetry" in self.dataset and "machines" in self.dataset

    def require_machine(self, machine_id: int) -> None:
        if machine_id not in self.machine_ids:
            raise ResourceNotFoundError(
                f"Machine {machine_id} is not in the dataset. "
                f"Known machines: {self.machine_ids[:5]}"
                f"{'...' if len(self.machine_ids) > 5 else ''}"
            )

    def slice_for(
        self, machine_id: int, window_hours: int = DEFAULT_WINDOW_HOURS
    ) -> Dict[str, pd.DataFrame]:
        """
        Narrow the dataset to one machine and its most recent window.

        See the module docstring: this is what makes prediction servable.
        """
        self.require_machine(machine_id)

        telemetry = self.dataset["telemetry"]
        telemetry = (
            telemetry[telemetry["machine_id"] == machine_id]
            .sort_values("datetime")
            .tail(window_hours)
        )

        sliced = {
            "telemetry": telemetry,
            "machines": self.dataset["machines"][
                self.dataset["machines"]["machine_id"] == machine_id
            ],
        }
        for name in ("errors", "maintenance"):
            frame = self.dataset.get(name)
            if frame is not None:
                sliced[name] = frame[frame["machine_id"] == machine_id]
        return sliced

    def machine_info(self, machine_id: int) -> Dict[str, Any]:
        """Static facts plus data coverage for one machine."""
        self.require_machine(machine_id)

        row = self.dataset["machines"]
        row = row[row["machine_id"] == machine_id].iloc[0]
        telemetry = self.dataset["telemetry"]
        readings = telemetry[telemetry["machine_id"] == machine_id]

        return {
            "machine_id": int(machine_id),
            "model": str(row["model"]) if "model" in row else None,
            "age": int(row["age"]) if "age" in row else None,
            "readings_available": int(len(readings)),
            "first_reading": readings["datetime"].min() if len(readings) else None,
            "last_reading": readings["datetime"].max() if len(readings) else None,
        }


class PredictionService:
    """Wraps the Predictor with per-machine slicing and a fleet cache."""

    def __init__(self, predictor: Predictor, store: MachineDataStore) -> None:
        self.predictor = predictor
        self.store = store
        self._fleet_cache: Optional[List[Dict[str, Any]]] = None
        self._fleet_cached_at: float = 0.0

    def predict_machine(
        self, machine_id: int, window_hours: int = DEFAULT_WINDOW_HOURS
    ) -> Dict[str, Any]:
        """Score one machine from stored data."""
        sliced = self.store.slice_for(machine_id, window_hours)
        return self.predictor.predict_machine(sliced, machine_id)

    def explain_machine(
        self,
        machine_id: int,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        history_hours: int = 24,
    ) -> Dict[str, Any]:
        """Score one machine and return the evidence behind it."""
        sliced = self.store.slice_for(machine_id, window_hours)
        return self.predictor.explain_machine(
            sliced, machine_id, history_hours=history_hours
        )

    def predict_from_readings(self, request: Any) -> Dict[str, Any]:
        """
        Score readings supplied by the caller rather than from stored data.

        This is the endpoint a real plant would use — its sensors are the
        source of truth, not a CSV sitting on the API host.
        """
        telemetry = pd.DataFrame(
            [
                {
                    "datetime": r.datetime,
                    "machine_id": request.machine_id,
                    "voltage": r.voltage,
                    "rotation": r.rotation,
                    "pressure": r.pressure,
                    "vibration": r.vibration,
                }
                for r in request.readings
            ]
        )
        machines = pd.DataFrame(
            [
                {
                    "machine_id": request.machine_id,
                    "model": request.model or "model1",
                    "age": request.age if request.age is not None else 0,
                }
            ]
        )
        empty_errors = pd.DataFrame(columns=["datetime", "machine_id", "error_id"])
        empty_maint = pd.DataFrame(columns=["datetime", "machine_id", "comp"])

        dataset = {
            "telemetry": telemetry,
            "machines": machines,
            "errors": empty_errors,
            "maintenance": empty_maint,
        }
        return self.predictor.predict_machine(dataset, request.machine_id)

    def fleet(self, force: bool = False) -> List[Dict[str, Any]]:
        """
        Score every machine, cached.

        Scoring the fleet one machine at a time costs roughly 160 ms each, so
        a 100-machine fleet is ~16 s — far too slow to recompute per request,
        and far too useful to omit.
        """
        age = time.monotonic() - self._fleet_cached_at
        if (
            not force
            and self._fleet_cache is not None
            and age < FLEET_CACHE_TTL_SECONDS
        ):
            logger.debug(f"Fleet cache hit ({age:.0f}s old)")
            return self._fleet_cache

        started = time.perf_counter()
        results = []
        for machine_id in self.store.machine_ids:
            try:
                results.append(self.predict_machine(machine_id))
            except Exception as e:
                # One unscoreable machine must not blank the whole fleet view.
                logger.warning(f"Skipping machine {machine_id}: {e}")

        results.sort(key=lambda r: r["failure_probability"], reverse=True)
        self._fleet_cache = results
        self._fleet_cached_at = time.monotonic()

        alerting = sum(1 for r in results if r["will_fail"])
        logger.info(
            f"Scored fleet of {len(results)} machines in "
            f"{time.perf_counter() - started:.1f}s — {alerting} alerting"
        )
        return results

    def invalidate_fleet_cache(self) -> None:
        self._fleet_cache = None
        self._fleet_cached_at = 0.0


class AppState:
    """Everything the routes need, built once at startup."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.predictor: Optional[Predictor] = None
        self.store: Optional[MachineDataStore] = None
        self.service: Optional[PredictionService] = None
        self.started_at: Optional[datetime] = None
        self.model_error: Optional[str] = None

    def startup(self) -> None:
        """
        Load the model and dataset.

        Deliberately does NOT raise. An API that refuses to start because the
        model is missing cannot serve /health, which is exactly what an
        operator needs in order to find out that the model is missing. Failures
        are recorded and reported as "degraded" instead.
        """
        self.started_at = datetime.now(timezone.utc)

        try:
            self.predictor = Predictor()
            logger.info("Model loaded.")
        except Exception as e:
            self.model_error = str(e)
            logger.error(f"Model unavailable — API will run degraded: {e}")

        self.store = MachineDataStore.load(self.settings.raw_data_path)
        if self.store.is_loaded:
            logger.info(f"Dataset loaded: {len(self.store.machine_ids)} machines.")
        else:
            logger.warning("Dataset unavailable — machine endpoints will 503.")

        if self.predictor and self.store.is_loaded:
            self.service = PredictionService(self.predictor, self.store)

    def shutdown(self) -> None:
        self.predictor = None
        self.store = None
        self.service = None

    @property
    def is_ready(self) -> bool:
        return self.service is not None


# Module-level singleton, populated by the app's lifespan handler.
state = AppState()
