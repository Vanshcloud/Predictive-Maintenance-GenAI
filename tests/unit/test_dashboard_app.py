"""
tests/unit/test_dashboard_app.py
=================================
Tests that the Streamlit app actually renders.

`api_client` tests cover the logic; these cover the thing that logic tests
cannot: whether the page executes end to end without raising. A Streamlit app
that throws still serves HTTP 200 — the exception surfaces inside the session,
not in the response — so "the server started" proves nothing about whether a
user sees a dashboard or a stack trace.

Streamlit's `AppTest` runs the script in-process with a fake session, which is
what makes this checkable at all. The API is stubbed: these assert the UI's
behaviour, not the model's.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"
APP = DASHBOARD / "app.py"

# The dashboard is standalone by design — it is not a package under src/, so
# its modules are only importable once its directory is on the path. The app
# does this for itself at runtime; the tests must do it to patch the client.
sys.path.insert(0, str(DASHBOARD))

HEALTH_OK = {
    "status": "ok",
    "model_loaded": True,
    "model_name": "lstm_predictive_maintenance",
    "dataset_loaded": True,
    "machines_known": 100,
    "threshold": 0.6678,
    "version": "0.1.0",
}

MACHINES = [
    {
        "machine_id": 51,
        "model": "model4",
        "age": 17,
        "readings_available": 8760,
        "first_reading": "2024-01-01T00:00:00",
        "last_reading": "2024-12-30T23:00:00",
    }
]

FLEET = {
    "machines_assessed": 2,
    "machines_alerting": 1,
    "threshold": 0.6678,
    "generated_at": "2024-12-30T23:00:00",
    "predictions": [
        {
            "machine_id": 51,
            "datetime": "2024-12-30 23:00:00",
            "failure_probability": 0.8731,
            "risk_level": "critical",
            "will_fail": True,
            "threshold": 0.6678,
        },
        {
            "machine_id": 1,
            "datetime": "2024-12-30 23:00:00",
            "failure_probability": 0.0001,
            "risk_level": "low",
            "will_fail": False,
            "threshold": 0.6678,
        },
    ],
}

EXPLAINED = {
    "machine_id": 51,
    "datetime": "2024-12-30 23:00:00",
    "failure_probability": 0.8731,
    "risk_level": "critical",
    "will_fail": True,
    "threshold": 0.6678,
    "age_years": 17,
    "errors_last_24h": 3,
    "hours_since_maintenance": {"comp1": 412, "comp2": 9999},
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
            "typical_cause": "components loosening or bearings worn",
        },
        "rotation": {
            "current": 400.0,
            "baseline_24h": 418.3,
            "change_24h": -33.8,
            "volatility_24h": 56.0,
            "deviation_sigma": -0.33,
            "unit": "RPM",
            "direction": "below",
            "is_concerning": False,
            "typical_cause": None,
        },
    },
    "most_deviant_sensors": ["vibration"],
}

HISTORY = [
    {
        "datetime": f"2024-12-30T{h:02d}:00:00",
        "voltage": 170.0,
        "rotation": 450.0,
        "pressure": 100.0,
        "vibration": 40.0 + h,
    }
    for h in range(24)
]


def make_client(**overrides):
    """A stub APIClient with sensible defaults."""
    client = MagicMock()
    client.health.return_value = overrides.get("health", HEALTH_OK)
    client.machines.return_value = overrides.get("machines", MACHINES)
    client.fleet.return_value = overrides.get("fleet", FLEET)
    client.explain.return_value = overrides.get("explained", EXPLAINED)
    client.history.return_value = overrides.get("history", HISTORY)
    client.report.return_value = overrides.get("report", {})
    for name, error in overrides.get("raises", {}).items():
        getattr(client, name).side_effect = error
    return client


def run_app(client, page="Fleet overview"):
    """
    Execute the dashboard with a stubbed client and return the result.

    The caches must be cleared first. `get_client()` is decorated with
    @st.cache_resource keyed on the API URL, which is correct in production —
    one client per URL, reused across reruns — but in-process it means the
    stub installed by the FIRST test is handed to every later one, and the
    failure-path tests silently exercise the happy path instead.
    """
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()

    app = AppTest.from_file(str(APP), default_timeout=30)
    with patch("api_client.APIClient", return_value=client):
        app.run()
        if app.radio and page != "Fleet overview":
            app.radio[0].set_value(page).run()
    return app


class TestFleetOverview:
    def test_renders_without_error(self):
        app = run_app(make_client())
        assert not app.exception, f"dashboard raised: {app.exception}"

    def test_shows_the_fleet_metrics(self):
        app = run_app(make_client())
        values = [m.value for m in app.metric]

        assert "2" in values  # machines assessed
        assert "1" in values  # alerting
        assert "0.6678" in values  # threshold

    def test_healthy_fleet_says_nothing_needs_action(self):
        """
        Zero alerts is the normal state and must read as reassurance.

        An empty table with no message looks like a broken query.
        """
        quiet = {
            **FLEET,
            "machines_alerting": 0,
            "predictions": FLEET["predictions"][1:],
        }
        app = run_app(make_client(fleet=quiet))

        assert any("Nothing needs action" in s.value for s in app.success)


class TestMachineDetail:
    def test_renders_without_error(self):
        app = run_app(make_client(), page="Machine detail")
        assert not app.exception, f"dashboard raised: {app.exception}"

    def test_shows_the_probability_and_alert_state(self):
        app = run_app(make_client(), page="Machine detail")
        values = [m.value for m in app.metric]

        assert "0.8731" in values
        assert "YES" in values


class TestFailureRendering:
    """
    The failure paths are what a user actually hits, and each needs different
    instructions. A traceback on screen helps nobody.
    """

    def test_unreachable_api_explains_how_to_start_it(self):
        import api_client

        app = run_app(
            make_client(raises={"health": api_client.APIUnavailable("refused")})
        )

        assert not app.exception, "the UI must render the problem, not crash"
        rendered = " ".join(e.value for e in app.error) + " ".join(
            m.value for m in app.markdown
        )
        assert "Cannot reach the API" in rendered
        assert "make run-api" in rendered

    def test_degraded_api_explains_the_missing_artifacts(self):
        degraded = {**HEALTH_OK, "status": "degraded", "model_loaded": False}
        app = run_app(make_client(health=degraded))

        assert not app.exception
        rendered = " ".join(e.value for e in app.error) + " ".join(
            m.value for m in app.markdown
        )
        assert "cannot serve predictions" in rendered
        assert "train_model.py" in rendered

    def test_endpoint_error_does_not_crash_the_page(self):
        import api_client

        app = run_app(make_client(raises={"fleet": api_client.APIError("boom", 500)}))

        assert not app.exception
        assert any("returned an error" in e.value for e in app.error)


class TestRiskConsistency:
    def test_risk_colours_are_keyed_off_the_api_level_only(self):
        """
        The dashboard must never recompute risk from the probability.

        If it applied its own thresholds it could show "medium" for a machine
        the API is alerting on. That inconsistency destroys trust and survives
        for months because both halves look individually correct.
        """
        source = APP.read_text()

        assert "RISK_COLOURS" in source
        # Colour lookup is by level name...
        assert "RISK_COLOURS.get(level" in source
        # ...and no threshold constants are defined here to compare against.
        for invented in ("RISK_BAND", "> 0.9", ">= 0.9", "> 0.6", ">= 0.6"):
            assert invented not in source, (
                f"dashboard appears to compute its own risk bands ({invented!r}); "
                "it must use the risk_level the API assigned"
            )


@pytest.mark.parametrize("page", ["Fleet overview", "Machine detail", "AI report"])
def test_every_page_renders(page):
    app = run_app(make_client(), page=page)
    assert not app.exception, f"page {page!r} raised: {app.exception}"
