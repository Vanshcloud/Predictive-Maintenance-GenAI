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
RISK_MODULE = DASHBOARD / "risk.py"

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
    "data_start": "2024-01-01T00:00:00",
    "data_end": "2024-12-30T23:00:00",
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
        # Both halves of the dashboard, since the palette moved to risk.py.
        source = APP.read_text(encoding="utf-8") + RISK_MODULE.read_text(
            encoding="utf-8"
        )

        assert "RISK_COLOURS" in source
        # Colour lookup is by level name...
        assert "RISK_COLOURS.get(level" in source
        # ...and no threshold constants are defined anywhere to compare against.
        for invented in ("RISK_BAND", "> 0.9", ">= 0.9", "> 0.6", ">= 0.6"):
            assert invented not in source, (
                f"dashboard appears to compute its own risk bands ({invented!r}); "
                "it must use the risk_level the API assigned"
            )


@pytest.mark.parametrize("page", ["Fleet overview", "Machine detail", "AI report"])
def test_every_page_renders(page):
    app = run_app(make_client(), page=page)
    assert not app.exception, f"page {page!r} raised: {app.exception}"


class TestRewind:
    """
    The rewind control is the only way to see the model do anything.

    The dataset's final hour has no machine inside a pre-failure window, so a
    dashboard pinned to "now" always reports zero alerts. These assert the
    control exists, is off by default, and that turning it on actually changes
    what is asked of the API — a picker whose value is dropped would leave the
    page looking identical while claiming to show October.
    """

    @staticmethod
    def _rewind(app):
        return next((t for t in app.toggle if t.label == "Rewind"), None)

    def test_hidden_when_the_api_does_not_publish_a_range(self):
        """
        An older API, or one running degraded, has no range to offer. Showing a
        picker with invented bounds would produce empty assessments.
        """
        no_range = {k: v for k, v in HEALTH_OK.items() if not k.startswith("data_")}
        app = run_app(make_client(health=no_range))

        assert not app.exception
        assert self._rewind(app) is None

    def test_off_by_default_so_the_page_asks_about_now(self):
        client = make_client()
        app = run_app(client)

        assert self._rewind(app) is not None, "no rewind control was rendered"
        assert client.fleet.call_args.kwargs["as_of"] is None

    def test_turning_it_on_sends_a_timestamp(self):
        client = make_client()
        app = run_app(client)
        self._rewind(app).set_value(True).run()

        assert not app.exception, f"dashboard raised: {app.exception}"
        as_of = client.fleet.call_args.kwargs["as_of"]
        assert as_of is not None
        # Defaults to the end of the data, so switching it on changes nothing
        # visible until the user actually moves the picker.
        assert as_of.startswith("2024-12-30T23:00")

    def test_the_picker_is_bounded_by_the_published_range(self):
        """
        Outside the range every assessment is empty or identical. The bounds
        are the API's to state, never the dashboard's to assume.
        """
        app = run_app(make_client())
        self._rewind(app).set_value(True).run()

        picker = next(d for d in app.date_input if d.label == "As of date")
        assert str(picker.min) == "2024-01-01"
        assert str(picker.max) == "2024-12-30"

    def test_machine_detail_scores_and_charts_the_same_moment(self):
        """
        The prediction and the sensor chart must share a timestamp. Scoring
        October while plotting December would read as a model that ignores its
        own inputs.
        """
        client = make_client()
        app = run_app(client, page="Machine detail")
        self._rewind(app).set_value(True).run()

        assert not app.exception
        as_of = client.explain.call_args.kwargs["as_of"]
        assert as_of is not None, "rewind did not reach the prediction call"
        assert client.history.call_args.kwargs["as_of"] == as_of


class TestBadgeContrast:
    """
    The risk badge is white text on the risk colour, and it is the thing a
    supervisor scans the fleet table for. Two of the four colours once sat at
    3.19:1 and 2.94:1 — readable on a good monitor, which is exactly why it
    survived review, and not readable in a lit workshop or with low vision.

    Contrast is a number, so it gets checked like one rather than eyeballed.

    These import `risk`, never `app`: importing app.py *runs* it, and the
    sidebar reaches the network on its first line. An earlier version of this
    class did import it, passed locally against a stray container on port 8000,
    and failed in CI where nothing answers.
    """

    @staticmethod
    def _contrast(hex_a: str, hex_b: str) -> float:
        """WCAG 2.1 relative-luminance contrast ratio."""

        def luminance(value: str) -> float:
            channels = [int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [
                c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        light, dark = sorted((luminance(hex_a), luminance(hex_b)), reverse=True)
        return (light + 0.05) / (dark + 0.05)

    def test_every_risk_colour_carries_white_text_at_wcag_aa(self):
        import risk

        for level, colour in risk.RISK_COLOURS.items():
            ratio = self._contrast("#ffffff", colour)
            # Badge text is 0.8em bold ~= 12.8px: normal text under WCAG 2.1,
            # so the 3:1 large-text allowance does not apply.
            assert ratio >= 4.5, (
                f"risk level {level!r} ({colour}) gives {ratio:.2f}:1 against "
                f"white badge text; WCAG 2.1 AA requires 4.5:1"
            )

    def test_the_unknown_level_fallback_is_also_readable(self):
        import risk

        # An unrecognised level still renders a badge; it must not render an
        # unreadable one.
        assert self._contrast("#ffffff", risk.UNKNOWN_COLOUR) >= 4.5

    def test_badge_escapes_the_level_it_is_given(self):
        """
        The badge is rendered with unsafe_allow_html, and the level comes from
        whatever host the sidebar's API URL points at — api_client returns
        response.json() without validating it.
        """
        import risk

        badge = risk.risk_badge("<img src=x onerror=alert(1)>")
        # The payload is upper-cased by the badge, so match the escaped form.
        assert "&lt;IMG SRC=X ONERROR=ALERT(1)&gt;" in badge
        assert "<img" not in badge.lower()

    def test_an_unknown_level_still_renders_a_badge(self):
        import risk

        assert risk.UNKNOWN_COLOUR in risk.risk_badge("bananas")
