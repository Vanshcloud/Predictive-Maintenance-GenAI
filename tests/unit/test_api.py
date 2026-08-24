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

from datetime import datetime, timedelta

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

    def slice_for(self, machine_id, window_hours=200):
        self.require_machine(machine_id)
        return {"telemetry": self.dataset["telemetry"]}

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

    def predict_machine(self, machine_id, window_hours=200):
        self.store.require_machine(machine_id)
        return _record(machine_id, 0.87 if machine_id == 51 else 0.01, machine_id == 51)

    def explain_machine(self, machine_id, window_hours=200, history_hours=24):
        self.store.require_machine(machine_id)
        record = self.predict_machine(machine_id)
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

    def fleet(self, force=False):
        self.fleet_calls += 1
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


class TestReportEndpoint:
    """The slow path, and its isolation from the fast one."""

    def test_generates_a_report(self, client, monkeypatch):
        class StubGenerator:
            def __init__(self, provider=None):
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

    def test_a_question_is_answered_instead_of_a_report(self, client, monkeypatch):
        class StubGenerator:
            def __init__(self, provider=None):
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
            def __init__(self, provider=None):
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
