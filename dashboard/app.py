"""
dashboard/app.py — Streamlit Dashboard
=======================================

WHY THIS FILE EXISTS:
    The audience for this project is a maintenance supervisor, not a data
    scientist. A REST API with OpenAPI docs is not a product for them; this
    is.

HOW IT WORKS:
    A **pure HTTP client** of the API. It imports nothing from `src/` — no
    TensorFlow, no model file, no preprocessing. Everything on screen came
    over the wire.

    Two consequences that matter:

      1. **Risk colours are keyed off the `risk_level` string the API
         assigned**, never recomputed from the probability here. If the
         dashboard applied its own thresholds it could show "medium" for a
         machine the API is alerting on, and that inconsistency is
         trust-destroying and survives for months.

      2. **Report generation is visibly slow and labelled as such.** It calls
         a language model and takes ~20 s. A spinner that says why is the
         difference between "thinking" and "broken".
"""

import html
import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import APIClient, APIDegraded, APIError, APIUnavailable  # noqa: E402

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
)

# Keyed by the level name the API assigns, so the two cannot disagree.
#
# Every fill here carries white badge text at 12.8px bold — normal text under
# WCAG 2.1, so it needs 4.5:1. The amber and yellow this started with (#d97706
# and #ca8a04) measured 3.19:1 and 2.94:1: legible to most people, unreadable
# in sunlight or with low vision, and exactly the two levels a supervisor scans
# for. The darker shades below clear 4.5:1 while keeping the red-orange-yellow-
# green severity ramp, and all four also clear the 3:1 that WCAG 1.4.11 asks of
# the bar chart's fills against the white page.
RISK_COLOURS = {
    "critical": "#b3202c",  # 6.65:1
    "high": "#c2410c",  # 5.18:1
    "medium": "#a16207",  # 4.92:1
    "low": "#15803d",  # 5.02:1
}
RISK_ORDER = ["critical", "high", "medium", "low"]


def risk_badge(level: str) -> str:
    # `level` is interpolated into markup rendered with unsafe_allow_html, and
    # it arrives from whatever host the sidebar's API URL points at — api_client
    # returns `response.json()` unvalidated, so the Literal in the API's schema
    # constrains our server and nothing else. Escaping costs one stdlib call.
    # The colour lookup is already safe: an unknown level falls back to grey
    # rather than reaching the style attribute.
    colour = RISK_COLOURS.get(level, "#6b7280")
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8em;font-weight:600'>"
        f"{html.escape(level.upper())}</span>"
    )


@st.cache_resource
def get_client(base_url: str) -> APIClient:
    return APIClient(base_url)


def show_api_problem(error: Exception) -> None:
    """
    Explain what is wrong and what to do, rather than showing a traceback.

    The three failure modes need different instructions, which is why the
    client raises three exception types instead of one.
    """
    if isinstance(error, APIUnavailable):
        st.error("**Cannot reach the API.**")
        st.markdown(
            "The dashboard is a client of the prediction API and cannot do "
            "anything without it.\n\n"
            "```bash\nmake run-api\n```\n\n"
            "If it is running elsewhere, set the URL in the sidebar."
        )
    elif isinstance(error, APIDegraded):
        st.error("**The API is running but cannot serve predictions.**")
        st.markdown(
            f"{error}\n\nThis usually means the trained model or the dataset "
            "is missing on the API host:\n\n"
            "```bash\npython scripts/run_preprocessing.py\n"
            "python scripts/train_model.py\n```"
        )
    else:
        st.error(f"**The API returned an error.** {error}")
    st.caption(str(error))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔧 Predictive Maintenance")
# Read from the environment so the container can be pointed at the API by
# service name. Inside docker-compose "localhost" is the dashboard itself, so
# a hardcoded default would leave the packaged app unable to reach anything.
DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
base_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
client = get_client(base_url)

try:
    health = client.health()
    if health.get("status") == "ok":
        st.sidebar.success(
            f"API ok — {health['machines_known']} machines\n\n"
            f"model `{health['model_name']}` · v{health['version']}"
        )
    else:
        st.sidebar.warning("API degraded — see the main panel.")
    api_reachable = True
except Exception as e:  # noqa: BLE001 — the UI must render whatever went wrong
    st.sidebar.error("API unreachable")
    health, api_reachable, startup_error = {}, False, e

page = st.sidebar.radio("View", ["Fleet overview", "Machine detail", "AI report"])

# ---------------------------------------------------------------------------
# Point in time
# ---------------------------------------------------------------------------
# Without this the dashboard can only ever assess the dataset's final hour, and
# on a fleet of 100 machines with 47 failures across a year that hour is almost
# always quiet — the demo shows "0 alerting" and looks broken rather than calm.
# Rewinding to a chosen hour is also what an operator genuinely wants ("what did
# this look like last Tuesday, before it failed?"), so it earns its place.
#
# The API hides everything after the chosen timestamp, so a past assessment is
# made on the evidence available at the time. It cannot see the failure coming
# because it has already happened.
as_of: Optional[str] = None
data_start, data_end = health.get("data_start"), health.get("data_end")

if data_start and data_end:
    lo = datetime.fromisoformat(data_start)
    hi = datetime.fromisoformat(data_end)

    st.sidebar.divider()
    if st.sidebar.toggle(
        "Rewind",
        value=False,
        help="Assess the fleet as it looked at an earlier hour.",
    ):
        picked = st.sidebar.date_input(
            "As of date",
            value=hi.date(),
            min_value=lo.date(),
            max_value=hi.date(),
        )
        hour = st.sidebar.slider("Hour", 0, 23, value=hi.hour)
        as_of = datetime.combine(picked, time(hour=hour)).isoformat()
        st.sidebar.caption(f"Everything after **{as_of.replace('T', ' ')}** is hidden.")
        # Verified against data/raw/failures.csv: machine 51 fails at
        # 2024-10-31 12:00 and machine 96 at 2024-11-14 00:00. The second pair
        # of dates is the point — 36 h out the model is silent, because it was
        # only ever trained to see 24 h ahead.
        st.sidebar.caption(
            "**Try:** 2024-10-31 hour 6 (machine 51) or 2024-11-13 hour 12 "
            "(machine 96) — both hours before a real failure. Then rewind to "
            "2024-10-30 and machine 51 goes quiet: that is the 24-hour "
            "horizon, not a bug."
        )

st.sidebar.divider()
st.sidebar.caption(
    "The dashboard talks only to the REST API. It holds no model and does no "
    "scoring of its own."
)

if not api_reachable:
    st.title("Predictive Maintenance")
    show_api_problem(startup_error)
    st.stop()

if health.get("status") != "ok":
    st.title("Predictive Maintenance")
    show_api_problem(
        APIDegraded(
            f"model_loaded={health.get('model_loaded')}, "
            f"dataset_loaded={health.get('dataset_loaded')}"
        )
    )
    st.stop()


# ---------------------------------------------------------------------------
# Fleet overview
# ---------------------------------------------------------------------------

if page == "Fleet overview":
    st.title("Fleet overview")

    col_a, col_b = st.columns([1, 4])
    alerts_only = col_a.toggle("Alerts only", value=False)
    if col_b.button("Refresh", help="Bypass the API's 5-minute fleet cache"):
        st.cache_data.clear()
        refresh = True
    else:
        refresh = False

    try:
        with st.spinner("Scoring the fleet…"):
            fleet = client.fleet(alerts_only=alerts_only, refresh=refresh, as_of=as_of)
    except (APIUnavailable, APIDegraded, APIError) as e:
        show_api_problem(e)
        st.stop()

    predictions = fleet["predictions"]
    counts = {level: 0 for level in RISK_ORDER}
    for p in predictions:
        counts[p["risk_level"]] = counts.get(p["risk_level"], 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Machines assessed", fleet["machines_assessed"])
    m2.metric("Alerting", fleet["machines_alerting"])
    m3.metric("Alert threshold", f"{fleet['threshold']:.4f}")
    m4.metric("Critical + high", counts["critical"] + counts["high"])

    if fleet["machines_alerting"] == 0:
        st.success(
            "No machine is at or above the alert threshold. "
            "Nothing needs action right now."
        )

    if not predictions:
        st.info("Nothing to show with the current filter.")
        st.stop()

    frame = pd.DataFrame(predictions)
    frame["probability"] = frame["failure_probability"].map(lambda p: f"{p:.4f}")

    st.header("Machines, most urgent first")
    st.dataframe(
        frame[["machine_id", "probability", "risk_level", "will_fail", "datetime"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "machine_id": "Machine",
            "probability": "P(failure ≤24h)",
            "risk_level": "Risk",
            "will_fail": st.column_config.CheckboxColumn("Alert"),
            "datetime": "Assessed at",
        },
    )

    st.header("Risk distribution")
    dist = pd.DataFrame(
        {"risk": RISK_ORDER, "machines": [counts[r] for r in RISK_ORDER]}
    )
    st.altair_chart(
        alt.Chart(dist)
        .mark_bar()
        .encode(
            x=alt.X("machines:Q", title="Machines"),
            y=alt.Y("risk:N", sort=RISK_ORDER, title=None),
            color=alt.Color(
                "risk:N",
                scale=alt.Scale(
                    domain=RISK_ORDER, range=[RISK_COLOURS[r] for r in RISK_ORDER]
                ),
                legend=None,
            ),
        )
        .properties(height=180),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Machine detail
# ---------------------------------------------------------------------------

elif page == "Machine detail":
    st.title("Machine detail")

    try:
        machines = client.machines()
    except (APIUnavailable, APIDegraded, APIError) as e:
        show_api_problem(e)
        st.stop()

    ids = [m["machine_id"] for m in machines]
    machine_id = st.selectbox("Machine", ids)
    hours = st.slider("Hours of history", 24, 336, 72, step=24)

    try:
        with st.spinner(f"Scoring machine {machine_id}…"):
            explained = client.explain(machine_id, as_of=as_of)
            readings = client.history(machine_id, hours=hours, as_of=as_of)
    except (APIUnavailable, APIDegraded, APIError) as e:
        show_api_problem(e)
        st.stop()

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(
        f"## Machine {machine_id} &nbsp; {risk_badge(explained['risk_level'])}",
        unsafe_allow_html=True,
    )
    c2.metric("P(failure ≤24h)", f"{explained['failure_probability']:.4f}")
    c3.metric("Alerting", "YES" if explained["will_fail"] else "no")

    info = next((m for m in machines if m["machine_id"] == machine_id), {})
    st.caption(
        f"model {info.get('model', '?')} · age {info.get('age', '?')} years · "
        f"{explained.get('errors_last_24h', 0)} errors in last 24h · "
        f"assessed at {explained['datetime']}"
    )

    st.header("Sensor evidence")
    sensors = explained.get("sensors", {})
    if not sensors:
        st.info("No sensor evidence returned.")
    else:
        rows = []
        for name in explained.get("most_deviant_sensors", []) + [
            s for s in sensors if s not in explained.get("most_deviant_sensors", [])
        ]:
            s = sensors.get(name)
            if not s:
                continue
            rows.append(
                {
                    "Sensor": name,
                    "Current": f"{s['current']} {s['unit']}",
                    "24h baseline": f"{s['baseline_24h']} {s['unit']}",
                    "Change 24h": f"{s['change_24h']:+} {s['unit']}",
                    "Deviation": f"{abs(s['deviation_sigma'])}σ {s['direction']}",
                    # The API decides what is concerning; this only renders it.
                    "Verdict": (
                        f"⚠️ {s['typical_cause']}"
                        if s["is_concerning"]
                        else "within normal variation"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    maintenance = explained.get("hours_since_maintenance") or {}
    if maintenance:
        st.caption(
            "Hours since last replacement — "
            + " · ".join(
                f"{comp}: {'no record' if h >= 9999 else f'{h}h'}"
                for comp, h in sorted(maintenance.items())
            )
        )

    st.header(f"Sensor readings, last {hours}h")
    if not readings:
        st.info("No readings available.")
    else:
        history = pd.DataFrame(readings)
        history["datetime"] = pd.to_datetime(history["datetime"])
        long = history.melt(
            id_vars="datetime",
            value_vars=["voltage", "rotation", "pressure", "vibration"],
            var_name="sensor",
            value_name="value",
        )
        st.altair_chart(
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("datetime:T", title=None),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("sensor:N", legend=None),
            )
            .properties(height=110)
            .facet(row=alt.Row("sensor:N", title=None, sort=list(sensors)))
            .resolve_scale(y="independent"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# AI report
# ---------------------------------------------------------------------------

elif page == "AI report":
    st.title("AI maintenance report")
    st.caption(
        "Generated by a language model from the prediction and its sensor "
        "evidence. Every figure it quotes comes from the API — it is given no "
        "other information to work from."
    )

    try:
        machines = client.machines()
    except (APIUnavailable, APIDegraded, APIError) as e:
        show_api_problem(e)
        st.stop()

    col1, col2 = st.columns([1, 2])
    machine_id = col1.selectbox("Machine", [m["machine_id"] for m in machines])
    question = col2.text_input(
        "Ask a specific question (optional)",
        placeholder="e.g. Is this machine safe to run through the next shift?",
    )

    adv1, adv2 = st.columns(2)
    provider = adv1.selectbox("Provider", ["(default)", "openai", "google", "ollama"])
    model = adv2.text_input(
        "Model override (optional)",
        placeholder="e.g. qwen2.5-coder:7b",
        help="Needed if the provider's configured default is not installed.",
    )

    if st.button("Generate", type="primary"):
        try:
            # Labelled honestly: this is the one action that calls a language
            # model, and it takes tens of seconds. A bare spinner reads as a
            # hang.
            with st.spinner(
                "Calling the language model — this usually takes 20–30 seconds…"
            ):
                result = client.report(
                    machine_id,
                    question=question or None,
                    provider=None if provider == "(default)" else provider,
                    model=model or None,
                    as_of=as_of,
                )
        except APIError as e:
            if e.status_code == 502:
                # The API's contract: the prediction survives an LLM outage.
                st.warning("**The language model is unavailable.**")
                st.markdown(
                    "The prediction is unaffected — only the written report "
                    "could not be produced."
                )
                st.code(str(e), language=None)
            elif e.status_code == 504:
                st.warning(f"**The language model timed out.** {e}")
            else:
                show_api_problem(e)
            st.stop()
        except (APIUnavailable, APIDegraded) as e:
            show_api_problem(e)
            st.stop()

        prediction = result["prediction"]
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(
            f"## Machine {prediction['machine_id']} &nbsp; "
            f"{risk_badge(prediction['risk_level'])}",
            unsafe_allow_html=True,
        )
        c2.metric("P(failure ≤24h)", f"{prediction['failure_probability']:.4f}")
        c3.metric("Alerting", "YES" if prediction["will_fail"] else "no")

        st.markdown("---")
        st.markdown(result["report"])

        if result.get("provider") or result.get("model"):
            st.caption(
                f"generated by {result.get('provider') or 'default provider'}"
                + (f" · {result['model']}" if result.get("model") else "")
            )
