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
