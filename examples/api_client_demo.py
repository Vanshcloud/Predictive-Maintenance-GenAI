#!/usr/bin/env python3
"""
examples/api_client_demo.py — Drive the API end to end.

WHY THIS FILE EXISTS:
    The API reference shows request and response shapes; this shows the order
    you actually call things in, and — more usefully — what each failure mode
    looks like when it happens rather than when it is described.

    It reuses `dashboard.api_client`, which already distinguishes the three
    failures a client must tell apart, instead of writing a fourth HTTP
    wrapper.

HOW IT WORKS:
    Six steps, each printing what it learned:

      1. readiness      — refuse to continue against a degraded instance
      2. inventory      — what machines exist
      3. fleet          — everything, most urgent first
      4. point-in-time  — the same fleet as it looked before a known failure
      5. evidence       — why one machine scored the way it did
      6. report         — the narrative layer, and its graceful degradation

USAGE:
    make docker-up-d                      # or: make run-api
    python examples/api_client_demo.py
    python examples/api_client_demo.py --machine 96 --as-of 2024-11-13T12:00:00
    python examples/api_client_demo.py --no-report      # skip the slow step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The dashboard package is a sibling of this directory, not an installed one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.api_client import (  # noqa: E402
    APIClient,
    APIDegraded,
    APIError,
    APIUnavailable,
)

# Machine 51 fails at 2024-10-31 12:00 in the seeded dataset, so six hours
# earlier is a moment the model should already be alerting on. Verifiable
# against data/raw/failures.csv.
DEFAULT_MACHINE = 51
DEFAULT_AS_OF = "2024-10-31T06:00:00"


def rule(title: str) -> None:
    print(f"\n{'─' * 68}\n{title}\n{'─' * 68}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--machine", type=int, default=DEFAULT_MACHINE)
    parser.add_argument(
        "--as-of",
        default=DEFAULT_AS_OF,
        help="Assess as of this timestamp. Pass 'none' for the latest reading.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip step 6 — it calls a language model and takes ~21 s.",
    )
    args = parser.parse_args()

    as_of = None if args.as_of.lower() == "none" else args.as_of
    client = APIClient(args.url)

    # ---- 1. Readiness ----------------------------------------------------
    # require_ready() raises rather than letting five later calls fail one at
    # a time with a 503 each.
    rule("1. Is this instance able to serve predictions?")
    try:
        health = client.require_ready()
    except APIUnavailable as e:
        print(f"✗ Nothing answered at {args.url}\n  {e}")
        print("\n  Start it: make run-api")
        return 1
    except APIDegraded as e:
        print(f"✗ Up, but cannot predict.\n  {e}")
        print("\n  Usually a missing model. See docs/troubleshooting.md")
        return 1

    print(f"  status      {health['status']}")
    print(f"  version     {health['version']}")
    print(f"  model       {health['model_name']}")
    print(f"  machines    {health['machines_known']}")
    print(f"  threshold   {health['threshold']}")
    print(f"  data range  {health['data_start']} → {health['data_end']}")

    # ---- 2. Inventory ----------------------------------------------------
    rule("2. What is in the fleet?")
    machines = client.machines()
    print(f"  {len(machines)} machines")
    for m in machines[:3]:
        print(
            f"    #{m['machine_id']:<4} {m['model']:<8} "
            f"age {m['age']:>2}  {m['readings_available']:,} readings"
        )
    if len(machines) > 3:
        print(f"    … and {len(machines) - 3} more")

    # ---- 3. Fleet now ----------------------------------------------------
    rule("3. Fleet status at the latest reading")
    fleet = client.fleet()
    print(
        f"  assessed {fleet['machines_assessed']}   "
        f"alerting {fleet['machines_alerting']}"
    )
    if fleet["machines_alerting"] == 0:
        print("\n  Nothing is alerting — which is usually correct. The dataset's")
        print("  final hour is a quiet one, so this view alone makes the model")
        print("  look inert. That is what step 4 exists to show.")

    # ---- 4. Point-in-time ------------------------------------------------
    if as_of:
        rule(f"4. The same fleet, as of {as_of}")
        print("  Everything after that moment is hidden — telemetry, errors, and")
        print("  maintenance alike, because errors_last_24h and")
        print("  hours_since_maintenance are model features.\n")
        past = client.fleet(as_of=as_of)
        print(
            f"  assessed {past['machines_assessed']}   "
            f"alerting {past['machines_alerting']}"
        )
        for p in past["predictions"][:5]:
            flag = "◀ ALERT" if p["will_fail"] else ""
            print(
                f"    #{p['machine_id']:<4} {p['failure_probability']:.6f}  "
                f"{p['risk_level']:<9}{flag}"
            )

    # ---- 5. Evidence -----------------------------------------------------
    rule(f"5. Why did machine {args.machine} score that way?")
    try:
        record = client.explain(args.machine, as_of=as_of)
    except APIError as e:
        print(f"✗ {e}")
        return 1

    print(f"  probability {record['failure_probability']:.6f}")
    print(f"  risk        {record['risk_level']}")
    print(f"  alerting    {'yes' if record['will_fail'] else 'no'}")
    print(f"  errors 24h  {record.get('errors_last_24h', 0)}\n")

    # Every figure below is read from the engineered features the model
    # consumed. None is derived for presentation.
    for name in record.get("most_deviant_sensors", []):
        s = record["sensors"][name]
        print(
            f"  {name:<10} {s['current']:>9.2f} {s['unit']:<5} "
            f"(24h baseline {s['baseline_24h']:.2f}, "
            f"{abs(s['deviation_sigma']):.2f}σ {s['direction']})"
        )
        verdict = (
            f"→ ABNORMAL; typically indicates {s['typical_cause']}"
            if s["is_concerning"]
            else "→ within normal variation"
        )
        print(f"  {'':<10} {verdict}")

    # ---- 6. Narrative ----------------------------------------------------
    if args.no_report:
        print("\n(step 6 skipped)")
        return 0

    rule(f"6. A written report for machine {args.machine}")
    print("  Calling a language model — this takes ~21 s locally.\n")
    try:
        result = client.report(args.machine, as_of=as_of)
    except APIError as e:
        # The contract worth demonstrating: the prediction survives an LLM
        # failure. A 502 carries it in the error detail.
        if e.status_code in (502, 504):
            print(f"  The language model is unavailable ({e.status_code}).")
            print("  The prediction is unaffected — that is the design:\n")
            print(f"    {e}")
            print("\n  Keyless: pip install -e '.[ollama]' && ollama pull llama3")
            return 0
        print(f"✗ {e}")
        return 1

    print(result["report"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
