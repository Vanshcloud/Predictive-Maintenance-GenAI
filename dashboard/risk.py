"""
dashboard/risk.py — Risk Presentation
======================================

WHY THIS FILE EXISTS:
    `app.py` is a Streamlit *script*: importing it executes it, which reaches
    the network on the first line of the sidebar. That makes the pure
    presentation logic inside it untestable without spinning up the whole app,
    and a test that reached for it by importing the module talked to whatever
    happened to be listening on port 8000 — passing locally against a stray
    container and failing in CI, where nothing answers.

    So the two pure things live here instead. No Streamlit import, no I/O,
    nothing that runs on import.

HOW IT WORKS:
    One palette and one formatter. The invariant they exist to protect is that
    **the dashboard never derives a risk band itself** — `risk_level` is a
    string the API assigned, and everything here is keyed off that string. If
    the dashboard applied its own thresholds it could show "medium" for a
    machine the API is alerting on, and that inconsistency looks correct from
    either side while destroying trust in both.
"""

import html

# Every fill carries white badge text at 0.8em bold (~12.8px) — normal text
# under WCAG 2.1, so it needs 4.5:1, not the 3:1 large-text allowance. The
# amber and yellow this started with (#d97706, #ca8a04) measured 3.19:1 and
# 2.94:1: fine on a good monitor in a dim room, unreadable in a lit workshop,
# and they were the two levels a supervisor scans the fleet table for. The
# ratios below are asserted by tests/unit/test_dashboard_app.py, because
# contrast is a number and shipped broken once by being eyeballed instead.
RISK_COLOURS = {
    "critical": "#b3202c",  # 6.65:1
    "high": "#c2410c",  # 5.18:1
    "medium": "#a16207",  # 4.92:1
    "low": "#15803d",  # 5.02:1
}

# Worst first: the order the fleet table sorts by and the chart stacks in.
RISK_ORDER = ["critical", "high", "medium", "low"]

# Any level the API invents that we do not know about still renders a badge —
# a readable grey one, rather than an unstyled or invisible sliver.
UNKNOWN_COLOUR = "#6b7280"  # 4.83:1


def risk_badge(level: str) -> str:
    """Render `level` as a coloured pill, safe to pass to unsafe_allow_html."""
    # `level` reaches this from whatever host the sidebar's API URL points at:
    # api_client returns response.json() unvalidated, so the Literal in the
    # API's own schema constrains our server and nothing else. The colour
    # lookup is already safe — an unknown level falls back to grey rather than
    # reaching the style attribute — but the text is interpolated into markup,
    # so it gets escaped.
    colour = RISK_COLOURS.get(level, UNKNOWN_COLOUR)
    return (
        f"<span style='background:{colour};color:white;padding:2px 10px;"
        f"border-radius:10px;font-size:0.8em;font-weight:600'>"
        f"{html.escape(level.upper())}</span>"
    )
