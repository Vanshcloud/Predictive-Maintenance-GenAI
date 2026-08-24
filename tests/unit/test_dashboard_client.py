"""
tests/unit/test_dashboard_client.py
====================================
Tests for the dashboard's API client.

The Streamlit app itself is mostly layout, and testing layout is low value.
The logic worth protecting lives here: distinguishing the three ways the API
can fail, and turning error bodies into something a human can act on.

That distinction is the whole point. "Something went wrong" is useless to a
supervisor; "the API is not running, start it with make run-api" is not.
"""

import ast
import sys
from pathlib import Path

import pytest
import requests

DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"
sys.path.insert(0, str(DASHBOARD))

from api_client import (  # noqa: E402
    REPORT_TIMEOUT,
    APIClient,
    APIDegraded,
    APIError,
    APIUnavailable,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def client():
    return APIClient("http://testhost:8000")


def patch_request(monkeypatch, handler):
    """Replace requests.request, capturing what the client sent."""
    calls = []

    def fake(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return handler(method, url, **kwargs)

    monkeypatch.setattr(requests, "request", fake)
    return calls


class TestFailureModes:
    """
    Three distinct failures need three distinct messages.

    Collapsing them would produce a dashboard that says "error" to a user who
    could have been told exactly what to restart.
    """

    def test_connection_refused_becomes_api_unavailable(self, client, monkeypatch):
        def refuse(*args, **kwargs):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "request", refuse)

        with pytest.raises(APIUnavailable) as exc:
            client.health()

        # The message must name the URL and the fix.
        assert "http://testhost:8000" in str(exc.value)
        assert "make run-api" in str(exc.value)

    def test_timeout_becomes_api_unavailable(self, client, monkeypatch):
        def slow(*args, **kwargs):
            raise requests.exceptions.Timeout("too slow")

        monkeypatch.setattr(requests, "request", slow)

        with pytest.raises(APIUnavailable, match="did not respond"):
            client.health()

    def test_degraded_health_is_rejected_by_require_ready(self, client, monkeypatch):
        """
        A 200 response does not mean the API can predict.

        require_ready() exists so the UI can say "the model is not loaded"
        up front, instead of surfacing a 503 from three calls deeper.
        """
        patch_request(
            monkeypatch,
            lambda *a, **k: FakeResponse(
                200,
                {"status": "degraded", "model_loaded": False, "dataset_loaded": True},
            ),
        )

        with pytest.raises(APIDegraded) as exc:
            client.require_ready()
        assert "model_loaded=False" in str(exc.value)

    def test_ready_health_passes_through(self, client, monkeypatch):
        payload = {"status": "ok", "model_loaded": True, "dataset_loaded": True}
        patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, payload))

        assert client.require_ready() == payload

    def test_error_status_becomes_api_error_with_code(self, client, monkeypatch):
        patch_request(
            monkeypatch,
            lambda *a, **k: FakeResponse(
                404, {"detail": "Machine 9999 is not in the dataset."}
            ),
        )

        with pytest.raises(APIError) as exc:
            client.predict(9999)

        assert exc.value.status_code == 404
        assert "9999" in str(exc.value)


class TestErrorDescription:
    """The server's message must survive to the screen."""

    def test_structured_detail_is_used(self, client, monkeypatch):
        patch_request(
            monkeypatch,
            lambda *a, **k: FakeResponse(503, {"detail": "Model not loaded."}),
        )

        with pytest.raises(APIError, match="Model not loaded."):
            client.machines()

    def test_pydantic_validation_errors_are_flattened(self, client, monkeypatch):
        """
        FastAPI's validation errors use a different shape from ours.

        Rendering the raw list would put a nested JSON blob on screen; the
        client flattens it to field-and-reason.
        """
        patch_request(
            monkeypatch,
            lambda *a, **k: FakeResponse(
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "readings", 0, "voltage"],
                            "msg": "Value error, voltage=99999.0 is outside "
                            "the plausible range",
                        }
                    ]
                },
            ),
        )

        with pytest.raises(APIError) as exc:
            client.predict(1)

        message = str(exc.value)
        assert "voltage" in message
        assert "outside the plausible range" in message

    def test_non_json_error_body_does_not_crash(self, client, monkeypatch):
        """A proxy returning HTML must not produce a JSONDecodeError."""
        patch_request(
            monkeypatch,
            lambda *a, **k: FakeResponse(502, payload=None, text="<html>Bad Gateway"),
        )

        with pytest.raises(APIError, match="HTTP 502"):
            client.machines()


class TestRequestConstruction:
    def test_endpoints_hit_the_expected_paths(self, client, monkeypatch):
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))

        client.health()
        client.machines()
        client.predict(51)
        client.explain(51)
        client.history(51, hours=72)
        client.fleet(alerts_only=True)

        paths = [c["url"].replace("http://testhost:8000", "") for c in calls]
        assert paths == [
            "/health",
            "/machines",
            "/machines/51/predict",
            "/machines/51/explain",
            "/machines/51/history",
            "/fleet",
        ]
        assert calls[4]["params"]["hours"] == 72
        assert calls[5]["params"]["alerts_only"] is True

    def test_as_of_travels_on_every_endpoint_that_accepts_it(self, client, monkeypatch):
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))
        stamp = "2024-10-30T12:00:00"

        client.predict(51, as_of=stamp)
        client.explain(51, as_of=stamp)
        client.history(51, as_of=stamp)
        client.fleet(as_of=stamp)

        assert [c["params"]["as_of"] for c in calls] == [stamp] * 4

    def test_as_of_is_omitted_rather_than_sent_as_none(self, client, monkeypatch):
        """
        `params={"as_of": None}` is dropped by requests, but relying on that
        silently is how a literal "None" ends up in a query string if the
        transport is ever swapped. Asserted so the swap breaks a test.
        """
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))

        client.fleet()

        assert calls[0]["params"]["as_of"] is None

    def test_report_omits_empty_optional_fields(self, client, monkeypatch):
        """
        An explicit null is not the same as an absent field.

        Sending {"provider": null} would override the API's own default
        selection logic with nothing.
        """
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))

        client.report(51)
        assert calls[0]["json"] == {"machine_id": 51}

        client.report(51, question="why?", provider="ollama", model="qwen")
        assert calls[1]["json"] == {
            "machine_id": 51,
            "question": "why?",
            "provider": "ollama",
            "model": "qwen",
        }

    def test_report_waits_longer_than_the_api_ceiling(self, client, monkeypatch):
        """
        The API caps generation at 120 s. A client timing out sooner would
        abandon work that was about to succeed, and report a failure that did
        not happen.
        """
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))

        client.report(51)

        assert calls[0]["timeout"] == REPORT_TIMEOUT
        assert REPORT_TIMEOUT > 120.0

    def test_base_url_trailing_slash_is_normalised(self, monkeypatch):
        client = APIClient("http://testhost:8000/")
        calls = patch_request(monkeypatch, lambda *a, **k: FakeResponse(200, {}))

        client.health()
        assert calls[0]["url"] == "http://testhost:8000/health"


def _imported_modules(path: Path) -> set:
    """
    Every top-level module name the file imports.

    Parsed rather than grepped: a docstring that mentions "tensorflow" to
    explain that it is NOT used would fail a substring check, which is
    exactly what happened when this test was first written.
    """
    tree = ast.parse(path.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


class TestDashboardIndependence:
    """
    The dashboard must stay a pure HTTP client.

    If it imported src.prediction, containerising it would drag in TensorFlow
    and the model file just to render charts — and the UI could no longer be
    deployed, scaled, or restarted separately from the API.
    """

    FORBIDDEN = {"src", "config", "tensorflow", "keras", "sklearn", "joblib"}

    def test_the_client_imports_nothing_from_the_ml_stack(self):
        assert not (_imported_modules(DASHBOARD / "api_client.py") & self.FORBIDDEN)

    def test_the_app_imports_nothing_from_the_ml_stack(self):
        leaked = _imported_modules(DASHBOARD / "app.py") & self.FORBIDDEN
        assert not leaked, f"dashboard leaked ML-stack imports: {sorted(leaked)}"
