"""
tests/unit/test_api.py
=======================
Tests for the REST API.

These run against the real app with a **stubbed** predictor and dataset.
Loading the genuine model takes ~2 s and the genuine dataset is 876,000 rows;
a test suite that pays that cost is a test suite nobody runs before
committing. What is being tested here is the HTTP layer — validation, status
codes, error shaping, and the isolation of the slow path — none of which needs
a real LSTM.

The error-handling tests matter most. A 500 that leaks a stack trace, or a
provider outage reported as an internal error, are the kind of defects that
survive to production because the happy path looks fine.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import service as service_module
from src.api.main import app
from src.utils.exceptions import (
    LLMConnectionError,
    PredictionError,
    ResourceNotFoundError,
)


class StubStore:
    """Stands in for MachineDataStore without loading 876,000 rows."""

    is_loaded = True

    def __init__(self, machine_ids=(1, 2, 51)):
        self.machine_ids = list(machine_ids)
        self.dataset = {"telemetry": _telemetry_frame(self.machine_ids)}

    def require_machine(self, machine_id):
        if machine_id not in self.machine_ids:
            raise ResourceNotFoundError(f"Machine {machine_id} is not in the dataset.")

    def slice_for(self, machine_id, window_hours=200, as_of=None):
        self.require_machine(machine_id)
        return {"telemetry": self.dataset["telemetry"]}

    @property
    def data_range(self):
        col = self.dataset["telemetry"]["datetime"]
        return col.min(), col.max()

    def machine_info(self, machine_id):
        self.require_machine(machine_id)
        return {
            "machine_id": machine_id,
            "model": "model3",
            "age": 12,
            "readings_available": 200,
            "first_reading": datetime(2024, 1, 1),
            "last_reading": datetime(2024, 12, 30),
        }


def _telemetry_frame(machine_ids):
    import pandas as pd

    start = datetime(2024, 12, 1)
    rows = [
        {
            "datetime": start + timedelta(hours=h),
            "machine_id": mid,
            "voltage": 170.0 + h,
            "rotation": 450.0,
            "pressure": 100.0,
            "vibration": 40.0,
        }
        for mid in machine_ids
        for h in range(60)
    ]
    return pd.DataFrame(rows)


def _record(machine_id=51, probability=0.87, will_fail=True):
    return {
        "machine_id": machine_id,
        "datetime": "2024-12-30 23:00:00",
        "failure_probability": probability,
        "risk_level": "critical" if will_fail else "low",
        "will_fail": will_fail,
        "threshold": 0.6678,
    }


class StubService:
    """Stands in for PredictionService."""

    def __init__(self, store):
        self.store = store
        self.fleet_calls = 0
        # Records (method, machine_id, as_of) so tests can assert the
        # timestamp actually reaches the service rather than being dropped.
        self.calls = []

    def predict_machine(self, machine_id, window_hours=200, as_of=None):
        self.store.require_machine(machine_id)
        self.calls.append(("predict_machine", machine_id, as_of))
        return _record(machine_id, 0.87 if machine_id == 51 else 0.01, machine_id == 51)

    def explain_machine(
        self, machine_id, window_hours=200, history_hours=24, as_of=None
    ):
        self.store.require_machine(machine_id)
        self.calls.append(("explain_machine", machine_id, as_of))
        record = self.predict_machine(machine_id, as_of=as_of)
        record["context"] = {
            "age_years": 17,
            "errors_last_24h": 2,
            "hours_since_maintenance": {"comp1": 412},
            "sensors": {
                "vibration": {
                    "current": 62.27,
                    "baseline_24h": 46.36,
                    "change_24h": 24.04,
                    "volatility_24h": 8.99,
                    "deviation_sigma": 1.77,
                    "unit": "mm/s",
                    "direction": "above",
                    "is_concerning": True,
                    "typical_cause": "components loosening",
                }
            },
            "most_deviant_sensors": ["vibration"],
            "recent_readings": [],
        }
        return record

    def predict_from_readings(self, request):
        return _record(request.machine_id, 0.12, False)

    def fleet(self, force=False, as_of=None):
        self.fleet_calls += 1
        self.calls.append(("fleet", None, as_of))
        return [_record(51, 0.87, True), _record(1, 0.01, False)]


def _client_with(monkeypatch, *, predictor, store, service, model_error):
    """
    Build a TestClient whose lifespan does NOT load the real artifacts.

    Entering the TestClient context runs the app's lifespan, and that calls
    state.startup(), which loads the genuine model and 876,000-row dataset —
    overwriting whatever stubs were installed beforehand and adding ~40 s to
    the suite. Neutering startup is what makes these tests fast and
    deterministic; the stubs are installed in its place.
    """
    state = service_module.state
    monkeypatch.setattr(state, "startup", lambda: None)
    monkeypatch.setattr(state, "shutdown", lambda: None)
    monkeypatch.setattr(state, "predictor", predictor)
    monkeypatch.setattr(state, "store", store)
    monkeypatch.setattr(state, "service", service)
    monkeypatch.setattr(state, "model_error", model_error)
    # raise_server_exceptions=False so our exception handlers run, rather than
    # the test client re-raising and hiding the response the client would see.
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(monkeypatch):
    """A ready API with stubs in place of the model and dataset."""
    store = StubStore()
    with _client_with(
        monkeypatch,
        predictor=object(),
        store=store,
        service=StubService(store),
        model_error=None,
    ) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(monkeypatch):
    """An API that started without a model — the realistic failure."""
    with _client_with(
        monkeypatch,
        predictor=None,
        store=None,
        service=None,
        model_error="no model artifact at models/...",
    ) as test_client:
        yield test_client


class TestHealth:
    def test_healthy_when_model_and_data_are_loaded(self, client):
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["dataset_loaded"] is True
        assert body["machines_known"] == 3

    def test_degraded_when_the_model_is_missing(self, degraded_client):
        """
        A process that is up but cannot predict is NOT healthy.

        Returning "ok" here would hide the one fact an operator needs, and a
        platform health check would happily keep routing traffic to it.
        """
        response = degraded_client.get("/health")

        assert response.status_code == 200  # the endpoint itself still works
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False


class TestMachines:
    def test_lists_machines(self, client):
        body = client.get("/machines").json()

        assert len(body) == 3
        assert body[0]["machine_id"] == 1
        assert body[0]["model"] == "model3"

    def test_unknown_machine_is_404_not_500(self, client):
        response = client.get("/machines/9999")

        assert response.status_code == 404
        assert response.json()["error_type"] == "ResourceNotFoundError"

    def test_machine_endpoints_are_503_when_degraded(self, degraded_client):
        assert degraded_client.get("/machines").status_code == 503


class TestPredictionEndpoints:
    def test_predicts_a_stored_machine(self, client):
        body = client.get("/machines/51/predict").json()

        assert body["machine_id"] == 51
        assert body["will_fail"] is True
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert body["risk_level"] == "critical"

    def test_explain_returns_the_evidence(self, client):
        body = client.get("/machines/51/explain").json()

        assert body["sensors"]["vibration"]["deviation_sigma"] == 1.77
        assert body["sensors"]["vibration"]["is_concerning"] is True
        assert body["most_deviant_sensors"] == ["vibration"]

    def test_history_returns_readings(self, client):
        body = client.get("/machines/51/history?hours=10").json()

        assert len(body) == 10
        assert "vibration" in body[0]

    def test_history_rejects_an_absurd_window(self, client):
        assert client.get("/machines/51/history?hours=99999").status_code == 422

    def test_fleet_is_sorted_and_counts_alerts(self, client):
        body = client.get("/fleet").json()

        assert body["machines_assessed"] == 2
        assert body["machines_alerting"] == 1
        probabilities = [p["failure_probability"] for p in body["predictions"]]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_fleet_alerts_only_filters(self, client):
        body = client.get("/fleet?alerts_only=true").json()

        assert len(body["predictions"]) == 1
        assert body["predictions"][0]["machine_id"] == 51
        # The totals still describe the whole fleet, not the filtered view.
        assert body["machines_assessed"] == 2


class TestRequestValidation:
    """Bad input must be rejected at the door with a precise complaint."""

    def _readings(self, count=48, **overrides):
        start = datetime(2024, 12, 1)
        return [
            {
                "datetime": (start + timedelta(hours=h)).isoformat(),
                "voltage": 170.0,
                "rotation": 450.0,
                "pressure": 100.0,
                "vibration": 40.0,
                **overrides,
            }
            for h in range(count)
        ]

    def test_accepts_a_well_formed_request(self, client):
        response = client.post(
            "/predict",
            json={"machine_id": 7, "readings": self._readings(), "model": "model2"},
        )
        assert response.status_code == 200
        assert response.json()["machine_id"] == 7

    def test_too_few_readings_is_422_with_a_useful_message(self, client):
        response = client.post(
            "/predict", json={"machine_id": 7, "readings": self._readings(10)}
        )

        assert response.status_code == 422
        assert "48" in str(response.json())

    def test_impossible_sensor_value_is_rejected(self, client):
        """
        A voltage of 99999 is a broken sensor or a unit mix-up.

        Scoring it would produce a confident number from an input the model
        has never seen anything like — worse than refusing.
        """
        response = client.post(
            "/predict",
            json={"machine_id": 7, "readings": self._readings(voltage=99999.0)},
        )

        assert response.status_code == 422
        assert "outside the plausible range" in str(response.json())

    def test_missing_field_is_rejected(self, client):
        response = client.post("/predict", json={"machine_id": 7})
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "timeline",
        [
            pytest.param([datetime(2024, 12, 1)] * 48, id="duplicated"),
            pytest.param(
                [datetime(2024, 12, 1) + timedelta(days=d) for d in range(48)],
                id="daily",
            ),
            pytest.param(
                [datetime(2024, 12, 1) + timedelta(hours=h) for h in range(48)][::-1],
                id="newest-first",
            ),
            pytest.param(
                [
                    datetime(2024, 12, 1) + timedelta(hours=h)
                    for h in range(49)
                    if h != 30
                ],
                id="one-hour-gap",
            ),
        ],
    )
    def test_readings_must_be_consecutive_hours(self, client, timeline):
        """
        Feature windows count rows, so 48 rows are 48 hours only if they are.

        Each of these used to be accepted and scored as hourly history.
        """
        readings = self._readings(len(timeline))
        for reading, moment in zip(readings, timeline):
            reading["datetime"] = moment.isoformat()

        response = client.post("/predict", json={"machine_id": 7, "readings": readings})

        assert response.status_code == 422
        assert "consecutive hourly" in str(response.json())

    def test_timezone_aware_readings_are_accepted(self, client):
        """The same consecutive hours, written with an offset, are still valid."""
        readings = self._readings()
        for reading in readings:
            reading["datetime"] += "+05:30"

        response = client.post("/predict", json={"machine_id": 7, "readings": readings})

        assert response.status_code == 200


class TestReportEndpoint:
    """The slow path, and its isolation from the fast one."""

    def test_generates_a_report(self, client, monkeypatch):
        class StubGenerator:
            def __init__(self, provider=None, **kwargs):
                pass

            def generate_report(self, record):
                return "ASSESSMENT\nMachine 51 is at critical risk."

            def answer_question(self, record, question):
                return f"Answer to: {question}"

        import src.genai as genai_module

        monkeypatch.setattr(genai_module, "ReportGenerator", StubGenerator)

        response = client.post("/report", json={"machine_id": 51})

        assert response.status_code == 200
        body = response.json()
        assert body["report"].startswith("ASSESSMENT")
        # The prediction travels with the report.
        assert body["prediction"]["machine_id"] == 51

    def test_the_provider_client_gets_the_report_deadline(self, client, monkeypatch):
        """
        `asyncio.wait_for` only stops waiting; the blocking provider call kept
        its thread until the client's own default (600 s per attempt for
        OpenAI) expired. The deadline has to reach the client itself.
        """
        from src.api.routes import reports

        seen = {}

        class StubGenerator:
            def __init__(self, provider=None, **kwargs):
                seen.update(kwargs)

            def generate_report(self, record):
                return "ok"

        import src.genai as genai_module

        monkeypatch.setattr(genai_module, "ReportGenerator", StubGenerator)

        response = client.post("/report", json={"machine_id": 51, "model": "m"})

        assert response.status_code == 200
        assert seen == {"model": "m", "timeout": reports.REPORT_TIMEOUT_SECONDS}

    def test_a_question_is_answered_instead_of_a_report(self, client, monkeypatch):
        class StubGenerator:
            def __init__(self, provider=None, **kwargs):
                pass

            def answer_question(self, record, question):
                return f"Answer to: {question}"

        import src.genai as genai_module

        monkeypatch.setattr(genai_module, "ReportGenerator", StubGenerator)

        body = client.post(
            "/report", json={"machine_id": 51, "question": "Why?"}
        ).json()
        assert body["report"] == "Answer to: Why?"

    def test_llm_outage_is_502_and_still_reports_the_prediction(
        self, client, monkeypatch
    ):
        """
        The central promise of this layer.

        The prediction decides whether a technician is dispatched; the report
        is a convenience over it. An LLM outage must degrade the endpoint, not
        erase the number.
        """

        class DeadGenerator:
            def __init__(self, provider=None, **kwargs):
                pass

            def generate_report(self, record):
                raise LLMConnectionError("Connection refused")

        import src.genai as genai_module

        monkeypatch.setattr(genai_module, "ReportGenerator", DeadGenerator)

        response = client.post("/report", json={"machine_id": 51})

        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "0.8700" in detail, "the prediction must survive an LLM outage"
        assert "critical" in detail

    def test_unknown_machine_in_report_is_404(self, client):
        assert client.post("/report", json={"machine_id": 9999}).status_code == 404


class TestReportCapacity:
    """
    A timed-out provider call keeps its report slot until its thread returns.

    Measured before the fix: wait_for + run_in_threadpool returned 504 and
    handed the threadpool token back while the thread ran on, so with a
    limit of one, four timed-out reports left four live threads.
    """

    @pytest.fixture
    def blocked(self, monkeypatch):
        """A provider that hangs until released; counts its live threads."""
        import src.genai as genai_module
        from src.api.routes import reports

        release = threading.Event()
        live, started = [], []

        class BlockedGenerator:
            def __init__(self, provider=None, **kwargs):
                pass

            def generate_report(self, record):
                started.append(1)
                live.append(1)
                try:
                    release.wait(30)
                    return "late report"
                finally:
                    live.pop()

        monkeypatch.setattr(genai_module, "ReportGenerator", BlockedGenerator)
        monkeypatch.setattr(reports, "REPORT_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(reports, "_report_slots", threading.BoundedSemaphore(1))
        yield release, live, started
        release.set()

    @staticmethod
    def _wait_until(condition, seconds=5.0):
        stop = time.monotonic() + seconds
        while not condition() and time.monotonic() < stop:
            time.sleep(0.01)
        return condition()

    def test_timed_out_workers_stay_counted(self, client, blocked):
        release, live, started = blocked

        codes = [client.post("/report", json={"machine_id": 51}) for _ in range(4)]

        assert [r.status_code for r in codes] == [504, 503, 503, 503]
        assert len(started) == 1 and len(live) == 1, "a refused call still ran"
        assert "0.8700" in codes[1].json()["detail"], "503 must keep the prediction"
        # The rest of the API is not starved while the slot is held.
        assert client.get("/health").status_code == 200

        release.set()
        assert self._wait_until(lambda: not live), "the worker never finished"
        assert client.post("/report", json={"machine_id": 51}).status_code == 200

    def test_concurrent_requests_cannot_exceed_the_cap(self, client, blocked):
        release, live, started = blocked
        with ThreadPoolExecutor(8) as pool:
            codes = sorted(
                r.status_code
                for r in pool.map(
                    lambda _: client.post("/report", json={"machine_id": 51}),
                    range(8),
                )
            )

        assert codes == [503] * 7 + [504]
        assert len(started) == 1

    def test_success_and_failure_both_free_the_slot(self, client, monkeypatch):
        import src.genai as genai_module
        from src.api.routes import reports

        outcomes = iter([LLMConnectionError("down"), "ok", "ok"])

        class Generator:
            def __init__(self, provider=None, **kwargs):
                pass

            def generate_report(self, record):
                outcome = next(outcomes)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        monkeypatch.setattr(genai_module, "ReportGenerator", Generator)
        monkeypatch.setattr(reports, "_report_slots", threading.BoundedSemaphore(1))

        codes = [
            client.post("/report", json={"machine_id": 51}).status_code
            for _ in range(3)
        ]

        assert codes == [502, 200, 200]


class TestErrorShaping:
    def test_internal_errors_do_not_leak_details(self, client, monkeypatch):
        """
        An unexpected exception must not put a stack trace on the wire.

        Paths, library versions, and internal structure are exactly what an
        attacker wants, and exactly what a default traceback provides.
        """

        def explode(*args, **kwargs):
            raise RuntimeError("psycopg2 connection to 10.0.0.5 failed: bad password")

        monkeypatch.setattr(service_module.state.service, "fleet", explode)

        response = client.get("/fleet")

        assert response.status_code == 500
        body = response.json()
        assert "psycopg2" not in str(body)
        assert "10.0.0.5" not in str(body)
        assert "password" not in str(body)
        # But an operator can find it in the logs.
        assert body["correlation_id"]
        assert body["error_type"] == "InternalServerError"

    def test_prediction_errors_are_422_not_500(self, client, monkeypatch):
        def bad_input(*args, **kwargs):
            raise PredictionError("not enough history to build a window")

        monkeypatch.setattr(service_module.state.service, "predict_machine", bad_input)

        response = client.get("/machines/51/predict")

        assert response.status_code == 422
        assert response.json()["error_type"] == "PredictionError"

    def test_error_responses_have_a_consistent_shape(self, client):
        body = client.get("/machines/9999").json()

        assert set(body) == {"detail", "error_type", "correlation_id"}


class TestDocumentation:
    def test_openapi_schema_covers_every_endpoint(self, client):
        paths = client.get("/openapi.json").json()["paths"]

        for expected in (
            "/health",
            "/machines",
            "/machines/{machine_id}",
            "/machines/{machine_id}/predict",
            "/machines/{machine_id}/explain",
            "/machines/{machine_id}/history",
            "/fleet",
            "/predict",
            "/report",
        ):
            assert expected in paths, f"{expected} missing from the OpenAPI schema"

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"


class TestTimeTravel:
    """
    `as_of` rewinds the assessment to an earlier hour.

    The property that matters is not "the parameter is accepted" but that
    everything after the chosen moment genuinely disappears. A historical
    assessment that could still see the failure it is meant to predict — or the
    maintenance visit that followed it — would look impressive and mean
    nothing.
    """

    @staticmethod
    def _store():
        """A two-machine store spanning three hours, built without file I/O."""
        hours = pd.date_range("2024-06-01 00:00", periods=3, freq="h")
        telemetry = pd.DataFrame(
            {
                "datetime": list(hours) * 2,
                "machine_id": [1] * 3 + [2] * 3,
                "voltage": [170.0, 171.0, 172.0, 180.0, 181.0, 182.0],
            }
        )
        return service_module.MachineDataStore(
            {
                "telemetry": telemetry,
                "machines": pd.DataFrame(
                    {"machine_id": [1, 2], "model": ["model1", "model2"], "age": [5, 9]}
                ),
                "errors": pd.DataFrame(
                    {
                        "datetime": hours,
                        "machine_id": [1, 1, 1],
                        "error_id": ["error1", "error2", "error3"],
                    }
                ),
                "maintenance": pd.DataFrame(
                    {
                        "datetime": hours,
                        "machine_id": [1, 1, 1],
                        "component": ["comp1", "comp2", "comp3"],
                    }
                ),
            }
        )

    def test_data_range_reports_the_available_window(self):
        lo, hi = self._store().data_range
        assert str(lo) == "2024-06-01 00:00:00"
        assert str(hi) == "2024-06-01 02:00:00"

    def test_telemetry_after_as_of_is_hidden(self):
        sliced = self._store().slice_for(1, as_of=pd.Timestamp("2024-06-01 01:00"))
        assert list(sliced["telemetry"]["voltage"]) == [170.0, 171.0]

    def test_as_of_is_inclusive_of_its_own_hour(self):
        """The chosen hour has already happened, so its reading is evidence."""
        sliced = self._store().slice_for(1, as_of=pd.Timestamp("2024-06-01 00:00"))
        assert list(sliced["telemetry"]["voltage"]) == [170.0]

    def test_errors_and_maintenance_are_hidden_too(self):
        """
        Telemetry alone is not enough.

        `errors_last_24h` and `hours_since_maintenance` are model features. If
        only telemetry were filtered, a historical prediction would be made
        knowing about breakdowns that had not yet occurred — leakage, wearing
        the clothes of a feature.
        """
        sliced = self._store().slice_for(1, as_of=pd.Timestamp("2024-06-01 01:00"))

        assert list(sliced["errors"]["error_id"]) == ["error1", "error2"]
        assert list(sliced["maintenance"]["component"]) == ["comp1", "comp2"]

    def test_omitting_as_of_returns_everything(self):
        sliced = self._store().slice_for(1)
        assert len(sliced["telemetry"]) == 3
        assert len(sliced["errors"]) == 3

    def test_health_publishes_the_range_so_a_ui_can_bound_its_picker(self, client):
        body = client.get("/health").json()

        assert body["data_start"] is not None
        assert body["data_end"] is not None
        assert body["data_start"] <= body["data_end"]

    @pytest.mark.parametrize(
        "path",
        [
            "/machines/51/predict?as_of=2024-10-30T12:00:00",
            "/machines/51/explain?as_of=2024-10-30T12:00:00",
            "/fleet?as_of=2024-10-30T12:00:00",
        ],
    )
    def test_the_timestamp_reaches_the_service(self, client, path):
        """
        A query parameter that is parsed and then dropped is the failure this
        guards: the endpoint answers 200 with a present-day assessment while
        the UI believes it is showing the past.
        """
        assert client.get(path).status_code == 200

        service = service_module.state.service
        assert service.calls, "the route never called the service"
        _, _, as_of = service.calls[0]
        assert as_of == datetime(2024, 10, 30, 12, 0)

    @pytest.mark.parametrize(
        "path",
        [
            "/machines/51/predict?as_of=2024-10-30T17:30:00%2B05:30",
            "/machines/51/explain?as_of=2024-10-30T12:00:00Z",
            "/fleet?as_of=2024-10-30T17:30:00%2B05:30",
        ],
    )
    def test_an_aware_timestamp_reaches_the_service_as_utc(self, client, path):
        """
        The stored telemetry is naive, and pandas refuses to compare it with an
        aware timestamp — `?as_of=...Z` used to be a 500. An offset must be
        honoured, not dropped: 17:30+05:30 is 12:00 UTC.
        """
        assert client.get(path).status_code == 200

        _, _, as_of = service_module.state.service.calls[0]
        assert as_of == datetime(2024, 10, 30, 12, 0)
        assert as_of.tzinfo is None

    def test_an_aware_timestamp_can_slice_the_store(self):
        """End to end through the real store, where the 500 was raised."""
        from src.api.schemas import ReportRequest

        request = ReportRequest(machine_id=1, as_of="2024-06-01T01:00:00Z")
        sliced = self._store().slice_for(1, as_of=request.as_of)

        assert list(sliced["telemetry"]["voltage"]) == [170.0, 171.0]

    def test_an_unparseable_timestamp_is_rejected(self, client):
        assert client.get("/machines/51/predict?as_of=last-tuesday").status_code == 422

    def test_the_fleet_cache_is_keyed_by_as_of(self, client):
        """
        The cache made the fleet endpoint viable (13.4 s cold, 1.6 ms warm),
        and a single shared slot would now serve a cached present-day answer to
        a request that asked about October.
        """
        service = service_module.state.service

        client.get("/fleet")
        client.get("/fleet?as_of=2024-10-30T12:00:00")
        client.get("/fleet")

        assert [c[2] for c in service.calls] == [
            None,
            datetime(2024, 10, 30, 12, 0),
            None,
        ]


class TestFleetCacheIsBounded:
    """
    The cache is keyed by `as_of`, a caller-supplied query parameter, and
    nothing evicted: the TTL was only consulted on a hit, so every distinct
    timestamp added ~100 prediction records that were never freed. The
    dashboard's rewind control is a date picker plus an hour slider, so
    ordinary use walks thousands of keys in one session.
    """

    @staticmethod
    def _service():
        class Store:
            machine_ids = [1]

            def slice_for(self, machine_id, window_hours=200, as_of=None):
                return {}

        class Predictor:
            def predict_machine(self, sliced, machine_id):
                return {
                    "machine_id": machine_id,
                    "failure_probability": 0.1,
                    "will_fail": False,
                }

        return service_module.PredictionService(Predictor(), Store())

    def test_distinct_as_of_values_do_not_grow_the_cache_without_bound(self):
        service = self._service()
        for hour in range(service_module.FLEET_CACHE_MAX_ENTRIES * 4):
            service.fleet(as_of=datetime(2024, 10, 30) + timedelta(hours=hour))

        assert len(service._fleet_cache) <= service_module.FLEET_CACHE_MAX_ENTRIES
        # The bookkeeping dict must be evicted alongside it, or the leak simply
        # moves from one dict to the other.
        assert len(service._fleet_cached_at) <= service_module.FLEET_CACHE_MAX_ENTRIES

    def test_the_oldest_key_is_the_one_dropped(self):
        service = self._service()
        first = datetime(2024, 10, 30)
        for hour in range(service_module.FLEET_CACHE_MAX_ENTRIES + 1):
            service.fleet(as_of=first + timedelta(hours=hour))

        assert pd.Timestamp(first) not in service._fleet_cache
        assert pd.Timestamp(first) not in service._fleet_cached_at

    def test_a_repeatedly_read_key_survives_eviction(self):
        """LRU, not FIFO: the view an operator keeps returning to is the one
        worth keeping, and re-inserting it would re-pay the ~16 s scoring cost."""
        service = self._service()
        hot = datetime(2024, 10, 30)
        service.fleet(as_of=hot)
        for hour in range(1, service_module.FLEET_CACHE_MAX_ENTRIES):
            service.fleet(as_of=hot + timedelta(hours=hour))
            service.fleet(as_of=hot)  # a cache hit, which must refresh recency

        service.fleet(as_of=hot + timedelta(days=99))
        assert pd.Timestamp(hot) in service._fleet_cache


class TestUnassessableFleet:
    """
    A machine that could not be scored has UNKNOWN status, not a healthy one.

    Before the data starts, or within its first 48 hours, machines lack
    history. Each was skipped silently: an entirely unscoreable fleet came
    back as a successful "0 alerting", and a partial one simply omitted the
    machines it could not score — which read the same as "fine".
    """

    @staticmethod
    def _service(machine_ids=(1, 2, 3), unscoreable=()):
        class Store:
            def __init__(self):
                self.machine_ids = list(machine_ids)

            def require_machine(self, machine_id):
                pass

            def slice_for(self, machine_id, window_hours=200, as_of=None):
                return {}

        class Predictor:
            def predict_machine(self, sliced, machine_id):
                if machine_id in unscoreable:
                    raise PredictionError("No complete sequences could be built.")
                return _record(machine_id, 0.01, False)

        return service_module.PredictionService(Predictor(), Store())

    def _get(self, monkeypatch, service, path):
        with _client_with(
            monkeypatch,
            predictor=object(),
            store=service.store,
            service=service,
            model_error=None,
        ) as test_client:
            return test_client.get(path)

    def test_a_fully_assessed_fleet_has_no_unknowns(self, monkeypatch):
        body = self._get(monkeypatch, self._service(), "/fleet").json()

        assert body["machines_assessed"] == 3
        assert body["unassessed_machine_ids"] == []

    def test_a_partial_fleet_names_the_machines_it_could_not_assess(self, monkeypatch):
        service = self._service(unscoreable=(2,))

        body = self._get(monkeypatch, service, "/fleet").json()

        assert body["machines_assessed"] == 2
        assert body["unassessed_machine_ids"] == [2]
        assert 2 not in [p["machine_id"] for p in body["predictions"]]

    def test_alerts_only_still_reports_the_unknowns(self, monkeypatch):
        """An unknown machine could be one that should be alerting."""
        service = self._service(unscoreable=(2,))

        body = self._get(monkeypatch, service, "/fleet?alerts_only=true").json()

        assert body["predictions"] == []
        assert body["unassessed_machine_ids"] == [2]

    def test_the_service_refuses_to_report_an_empty_assessment(self):
        service = self._service(machine_ids=(1, 2), unscoreable=(1, 2))
        with pytest.raises(PredictionError, match="none of the 2 machines"):
            service.fleet(as_of=datetime(2024, 1, 1))

    def test_the_route_reports_insufficient_data_not_a_healthy_fleet(self, monkeypatch):
        service = self._service(machine_ids=(1, 2), unscoreable=(1, 2))

        response = self._get(monkeypatch, service, "/fleet?as_of=2024-01-01T00:00:00")

        assert response.status_code == 422
        assert "could be assessed" in response.json()["detail"]


class TestFleetCacheUnderConcurrency:
    """
    Route handlers are `def`, not `async def`, so FastAPI runs them in a
    threadpool and concurrency here is real. Checking the cache and filling it
    used to be unsynchronised, so every request arriving for an uncached
    `as_of` while another was mid-computation missed too, and recomputed the
    same answer.

    Measured against the running API before the fix: four concurrent requests
    for one cold timestamp produced four full fleet scorings and made every
    caller wait 58 s for work that takes ~14 s once.
    """

    @staticmethod
    def _service(scorings, barrier=None):
        """A service whose scoring is slow enough to overlap, and counted."""

        class Store:
            machine_ids = [1]

            def slice_for(self, machine_id, window_hours=200, as_of=None):
                return {}

        class Predictor:
            def predict_machine(self, sliced, machine_id):
                scorings.append(machine_id)
                if barrier is not None:
                    # Every worker is inside the scoring body at once — the
                    # exact interleaving the old code got wrong. Without the
                    # lock this returns; with it, only one thread arrives and
                    # the barrier times out, which is the point.
                    try:
                        barrier.wait(timeout=0.5)
                    except threading.BrokenBarrierError:
                        pass
                time.sleep(0.05)
                return {
                    "machine_id": machine_id,
                    "failure_probability": 0.1,
                    "will_fail": False,
                }

        return service_module.PredictionService(Predictor(), Store())

    def test_concurrent_requests_for_one_cold_key_score_the_fleet_once(self):
        scorings = []
        service = self._service(scorings)
        as_of = datetime(2024, 10, 31, 6)

        results = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for r in pool.map(lambda _: service.fleet(as_of=as_of), range(6)):
                results.append(r)

        # One scoring, not six. This is the whole fix.
        assert len(scorings) == 1, (
            f"fleet() scored {len(scorings)} times for one cold key — "
            "concurrent callers are stampeding the cache instead of waiting"
        )
        # And every caller still got the answer.
        assert all(r == results[0] for r in results)
        assert len(results) == 6

    def test_a_cache_hit_does_not_block_behind_a_cold_computation(self):
        """
        The compute lock must not be held for reads. An operator refreshing a
        cached view should not wait ~14 s because someone else rewound to an
        uncached hour.
        """
        scorings = []
        service = self._service(scorings)
        warm = datetime(2024, 10, 31, 6)
        service.fleet(as_of=warm)  # populate
        assert len(scorings) == 1

        cold = datetime(2024, 11, 14)
        started = threading.Event()

        def slow_cold():
            started.set()
            service.fleet(as_of=cold)

        worker = threading.Thread(target=slow_cold, daemon=True)
        worker.start()
        started.wait(timeout=2)

        # Hit the warm key while the cold scoring is in flight.
        elapsed = time.perf_counter()
        service.fleet(as_of=warm)
        elapsed = time.perf_counter() - elapsed

        worker.join(timeout=10)
        assert elapsed < 0.05, (
            f"a cache hit took {elapsed:.3f}s while another key was computing "
            "— reads are blocking on the compute lock"
        )

    def test_concurrent_distinct_keys_all_get_stored(self):
        """Serialising computation must not lose anyone's result."""
        scorings = []
        service = self._service(scorings)
        stamps = [datetime(2024, 10, 31, h) for h in range(6)]

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda t: service.fleet(as_of=t), stamps))

        assert len(scorings) == 6  # six distinct keys, six scorings
        for t in stamps:
            assert pd.Timestamp(t) in service._fleet_cache


class TestDatasetAvailability:
    """
    startup() promises never to raise and to report problems as "degraded".
    An unreadable CSV raised straight through it, so the API never came up;
    a missing event log was not a problem at all — /health said "ok" while
    every stored prediction filled errors and maintenance with defaults.
    """

    from config.settings import PROJECT_ROOT

    SAMPLE = PROJECT_ROOT / "data" / "sample"

    def _copy_sample(self, tmp_path):
        import shutil

        for name in service_module.RAW_TABLES:
            shutil.copy(self.SAMPLE / f"{name}.csv", tmp_path / f"{name}.csv")
        return tmp_path

    def test_a_complete_dataset_loads(self, tmp_path):
        store = service_module.MachineDataStore.load(self._copy_sample(tmp_path))

        assert store.is_loaded
        assert store.problems == []

    def test_an_unreadable_table_degrades_instead_of_crashing(self, tmp_path):
        data_dir = self._copy_sample(tmp_path)
        (data_dir / "errors.csv").write_text("")

        store = service_module.MachineDataStore.load(data_dir)

        assert not store.is_loaded
        assert "errors.csv could not be read" in " ".join(store.problems)

    @pytest.mark.parametrize("table", ["errors", "maintenance"])
    def test_a_missing_event_log_is_not_a_loaded_dataset(self, tmp_path, table):
        data_dir = self._copy_sample(tmp_path)
        (data_dir / f"{table}.csv").unlink()

        store = service_module.MachineDataStore.load(data_dir)

        assert not store.is_loaded
        assert f"{table}.csv not found" in " ".join(store.problems)

    def test_startup_survives_and_the_503_says_why(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        data_dir = self._copy_sample(tmp_path)
        (data_dir / "maintenance.csv").write_text("")
        monkeypatch.setattr(service_module, "Predictor", lambda: object())
        app_state = service_module.AppState()
        app_state.settings = SimpleNamespace(raw_data_path=data_dir)

        app_state.startup()  # must not raise

        assert not app_state.is_ready
        assert "maintenance.csv" in app_state.data_error

        with _client_with(
            monkeypatch,
            predictor=app_state.predictor,
            store=app_state.store,
            service=None,
            model_error=None,
        ) as test_client:
            monkeypatch.setattr(
                service_module.state, "data_error", app_state.data_error
            )
            response = test_client.get("/machines/1/predict")

        assert response.status_code == 503
        assert "maintenance.csv" in response.json()["detail"]

    @pytest.mark.parametrize(
        "table,column",
        [
            ("machines", "machine_id"),
            ("telemetry", "voltage"),
            ("telemetry", "datetime"),
            ("errors", "error_id"),
            ("maintenance", "comp"),
        ],
    )
    def test_a_table_missing_a_required_column_is_not_loaded(
        self, tmp_path, table, column
    ):
        # machines without machine_id raised KeyError out of startup; telemetry
        # without voltage was "loaded" and /health said ok while every
        # prediction failed for want of the feature.
        data_dir = self._copy_sample(tmp_path)
        path = data_dir / f"{table}.csv"
        pd.read_csv(path).drop(columns=[column]).to_csv(path, index=False)

        store = service_module.MachineDataStore.load(data_dir)  # must not raise

        assert not store.is_loaded
        assert f"{table}.csv is invalid" in " ".join(store.problems)
        assert column in " ".join(store.problems)

    def test_a_sensor_column_read_as_text_is_not_loaded(self, tmp_path):
        data_dir = self._copy_sample(tmp_path)
        path = data_dir / "telemetry.csv"
        frame = pd.read_csv(path)
        frame["vibration"] = frame["vibration"].astype(str)
        frame.loc[0, "vibration"] = "sensor fault"
        frame.to_csv(path, index=False)

        store = service_module.MachineDataStore.load(data_dir)

        assert not store.is_loaded
        assert "vibration" in " ".join(store.problems)

    def test_an_unparseable_timestamp_is_not_loaded(self, tmp_path):
        data_dir = self._copy_sample(tmp_path)
        path = data_dir / "errors.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "datetime"] = "not a date"
        frame.to_csv(path, index=False)

        store = service_module.MachineDataStore.load(data_dir)

        assert not store.is_loaded
        assert "errors.csv could not be read" in " ".join(store.problems)

    def test_an_empty_machines_table_is_not_loaded(self, tmp_path):
        data_dir = self._copy_sample(tmp_path)
        (data_dir / "machines.csv").write_text("machine_id,model,age\n")

        store = service_module.MachineDataStore.load(data_dir)

        assert not store.is_loaded
        assert "machines.csv is invalid" in " ".join(store.problems)

    def test_an_event_log_with_no_events_still_loads(self, tmp_path):
        # Zero events is a real observation, not a schema failure.
        data_dir = self._copy_sample(tmp_path)
        (data_dir / "errors.csv").write_text("datetime,machine_id,error_id\n")

        store = service_module.MachineDataStore.load(data_dir)

        assert store.is_loaded
        assert store.problems == []

    def test_health_and_endpoints_agree_on_an_invalid_dataset(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        data_dir = self._copy_sample(tmp_path)
        path = data_dir / "telemetry.csv"
        pd.read_csv(path).drop(columns=["voltage"]).to_csv(path, index=False)
        monkeypatch.setattr(service_module, "Predictor", lambda: object())
        app_state = service_module.AppState()
        app_state.settings = SimpleNamespace(raw_data_path=data_dir)

        app_state.startup()

        assert not app_state.is_ready
        with _client_with(
            monkeypatch,
            predictor=app_state.predictor,
            store=app_state.store,
            service=app_state.service,
            model_error=None,
        ) as test_client:
            monkeypatch.setattr(
                service_module.state, "data_error", app_state.data_error
            )
            health = test_client.get("/health").json()
            machines = test_client.get("/machines")
            predict = test_client.get("/machines/1/predict")

        assert health["status"] == "degraded"
        assert health["dataset_loaded"] is False
        assert health["machines_known"] == 0
        assert machines.status_code == 503
        assert predict.status_code == 503
        assert "voltage" in predict.json()["detail"]
