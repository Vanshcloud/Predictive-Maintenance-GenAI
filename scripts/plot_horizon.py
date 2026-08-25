#!/usr/bin/env python3
"""
scripts/plot_horizon.py — The 24-Hour Horizon, Drawn

WHY THIS FILE EXISTS:
    The project's central claim is "predicts failures 24 hours in advance."
    Everywhere else that claim is a sentence, a metric, or something you have
    to install TensorFlow and train a model to see. This draws it.

    It scores ONE machine at every hour of the two days before it actually
    failed, using the API's `as_of` parameter so each point is made only on
    the evidence available at that hour. The result is the model waking up:
    flat at zero, then a near-vertical climb through the alert threshold about
    sixteen hours out.

    The image is committed for the README, and this script is committed so it
    can be regenerated rather than trusted. A chart nobody can reproduce is
    the same problem as a status badge nobody can falsify.

HOW IT WORKS:
    Needs the stack running (`make docker-up` or `make run-api`). Every point
    is one GET against /machines/{id}/predict?as_of=...; nothing is computed
    here. Defaults describe machine 51, whose failure at 2024-10-31 12:00 is
    recorded in data/raw/failures.csv.

    python scripts/plot_horizon.py
    python scripts/plot_horizon.py --machine 96 --failure 2024-11-14T00:00:00
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a build machine
import matplotlib.pyplot as plt  # noqa: E402

# Validated with the dataviz palette checker against a white surface: all six
# checks pass, worst-case CVD separation ΔE 27.6 (protan). The red is the
# project's own `critical` risk colour, so the chart and the dashboard agree.
LINE = "#1d4ed8"
CRITICAL = "#b3202c"
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"


def score(api: str, machine: int, when: datetime) -> dict:
    # quote(), because isoformat() emits a literal "+" for a timezone-aware
    # timestamp and a "+" in a query string decodes to a space — which the
    # API then rejects as an invalid datetime on every single point.
    as_of = urllib.parse.quote(when.isoformat(), safe="")
    url = f"{api}/machines/{machine}/predict?as_of={as_of}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fail(message: str) -> int:
    """Print a diagnosis and a fix, never a traceback."""
    print(f"\n{message}", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--machine", type=int, default=51)
    p.add_argument("--failure", default="2024-10-31T12:00:00")
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--out", default="docs/images/horizon.png")
    args = p.parse_args()

    failure = datetime.fromisoformat(args.failure)

    try:
        with urllib.request.urlopen(f"{args.api}/health", timeout=30) as response:
            health = json.load(response)
    except urllib.error.URLError as e:
        return fail(
            f"cannot reach the API at {args.api}: {e}\n"
            f"Start it with: make docker-up"
        )

    # /health answers 200 whether or not the model loaded — "degraded" is a
    # field, not a status code, which is deliberate (an operator needs the
    # check to answer in order to learn the model is missing). So it has to be
    # read. On a fresh clone models/ and data/ are gitignored and merely
    # mounted, so this is the normal first-run state, and without the check
    # the script sailed past the friendly branch above and died on the first
    # scoring call with an unhandled 503 traceback.
    if health.get("status") != "ok":
        return fail(
            f"the API is up but {health.get('status')}: "
            f"model_loaded={health.get('model_loaded')}, "
            f"dataset_loaded={health.get('dataset_loaded')}.\n"
            f"It has nothing to score with. Generate the data and train first:\n"
            f"  python scripts/generate_data.py\n"
            f"  python scripts/run_preprocessing.py\n"
            f"  python scripts/train_model.py"
        )
    threshold = health["threshold"]

    # The window must sit inside the loaded data. Asking for a failure near
    # the start of the dataset silently produced an empty telemetry slice and
    # a PredictionError 20 hourly rows into the run, with no chart written.
    window_start = failure - timedelta(hours=args.hours)
    start, end = health.get("data_start"), health.get("data_end")
    if start and end:
        first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if window_start < first or failure > last:
            return fail(
                f"--failure {args.failure} with --hours {args.hours} needs data "
                f"from {window_start} to {failure}, but this API has loaded "
                f"{first} to {last}.\n"
                f"Pick a failure at least {args.hours}h after the start of the data."
            )

    hours, probs = [], []
    for back in range(args.hours, -1, -1):
        try:
            record = score(args.api, args.machine, failure - timedelta(hours=back))
        except urllib.error.HTTPError as e:
            # 404 for an unknown --machine, 503 for an unscoreable window.
            # Either way the run is over; say which hour and why rather than
            # unwinding a traceback on top of 20 rows of printed output.
            detail = e.read().decode("utf-8", "replace")[:300]
            return fail(
                f"the API returned {e.code} scoring machine {args.machine} "
                f"at -{back}h:\n{detail}"
            )
        hours.append(-back)
        probs.append(record["failure_probability"])
        print(f"-{back:02d}h  {record['failure_probability']:.4f}", flush=True)

    # The hour the model first commits — the single number this chart exists
    # to show. Found from the data rather than hardcoded, so it stays true if
    # the model is retrained.
    crossing = next((h for h, v in zip(hours, probs) if v >= threshold), None)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Alerting region: shaded rather than outlined, so it recedes behind the line.
    if crossing is not None:
        ax.axvspan(crossing, 0, color=CRITICAL, alpha=0.06, zorder=0)

    ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(
        -args.hours + 0.5,
        threshold + 0.025,
        f"alert threshold  {threshold:.4f}",
        color=MUTED,
        fontsize=9,
    )

    ax.plot(hours, probs, color=LINE, linewidth=2, zorder=3)
    ax.axvline(0, color=CRITICAL, linewidth=2, zorder=2)
    ax.text(
        -0.7,
        0.5,
        "FAILURE",
        color=CRITICAL,
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="right",
    )

    # Selective direct labels only — never a number on every point.
    if crossing is not None:
        label_x = max(crossing - args.hours * 0.44, -args.hours + 0.5)
        ax.plot(
            [crossing],
            [dict(zip(hours, probs))[crossing]],
            "o",
            color=LINE,
            ms=9,
            zorder=4,
        )
        ax.annotate(
            f"first hour at or above the threshold\n"
            f"— {abs(crossing)} hours before failure",
            xy=(crossing, dict(zip(hours, probs))[crossing]),
            # Clear of the dashed threshold line; the earlier placement sat
            # directly on it and both became harder to read. Offset as a
            # fraction of the axis rather than a fixed 21 hours: the x-limits
            # follow --hours, so the constant put the label off-figure at any
            # window shorter than the default 48.
            xytext=(label_x, 0.40),
            color=INK,
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2),
        )
    ax.annotate(
        "silent — nothing in the sensors yet",
        xy=(-args.hours + 6, 0.02),
        xytext=(-args.hours + 6, 0.16),
        color=MUTED,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.2),
    )

    ax.set_title(
        f"Machine {args.machine}: probability of failure within 24h,\n"
        f"scored hour by hour on the evidence available at that hour",
        fontsize=12,
        color=INK,
        loc="left",
        pad=14,
    )
    ax.set_xlabel("hours before the actual failure", fontsize=10, color=INK)
    ax.set_ylabel("P(failure ≤ 24h)", fontsize=10, color=INK)
    ax.set_xlim(-args.hours, 1.5)
    ax.set_ylim(-0.03, 1.06)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}  (threshold crossed at {crossing}h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
