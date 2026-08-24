#!/usr/bin/env python3
"""
scripts/predict.py — Score Machines With the Trained Model
===========================================================

WHY THIS FILE EXISTS:
    The Predictor is a library class. This is the way to actually use it
    without writing Python — the command a maintenance engineer runs, and
    the thing Day 9's API will wrap.

HOW IT WORKS:
    Loads the raw tables from a directory, scores every machine, and prints
    a fleet view sorted most-urgent-first. `--latest` gives one row per
    machine (current status); without it you get every hourly window, which
    is what you want for plotting a risk timeline.

USAGE:
    python scripts/predict.py                          # current fleet status
    python scripts/predict.py --data-dir data/sample   # against the fixture
    python scripts/predict.py --machine 47             # one machine, as JSON
    python scripts/predict.py --alerts-only -o out.csv # only what needs action
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure src can be found if running from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TensorFlow must load before pandas/scikit-learn pull in Arrow — see
# src/models/__init__.py. Importing Predictor first guarantees the order.
from src.prediction import Predictor  # noqa: E402  isort:skip

from config.settings import get_settings  # noqa: E402
from src.utils.exceptions import PredMaintenanceError  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

RAW_TABLES = ("telemetry", "machines", "errors", "maintenance")

RISK_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def load_tables(data_dir: Path) -> dict:
    """Load the raw CSV tables the predictor needs."""
    dataset = {}
    for name in RAW_TABLES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            logger.error(
                f"Missing {path}. Generate it with `python scripts/generate_data.py`."
            )
            sys.exit(1)
        dataset[name] = pd.read_csv(path)
        logger.info(f"  loaded {name}: {len(dataset[name]):,} rows")
    return dataset


def render_table(df: pd.DataFrame, limit: int) -> str:
    """Format the fleet view for a terminal."""
    view = df.head(limit).copy()
    view["failure_probability"] = view["failure_probability"].map(lambda p: f"{p:.4f}")
    view["will_fail"] = view["will_fail"].map({True: "ALERT", False: ""})
    return view.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description="Score machines for failure risk")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory holding the raw CSV tables (default: settings raw data path)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="One row per machine — current fleet status (default: every window)",
    )
    parser.add_argument(
        "--machine", default=None, help="Score a single machine and print JSON"
    )
    parser.add_argument(
        "--alerts-only",
        action="store_true",
        help="Show only machines at or above the alert threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the alert threshold (default: settings.PREDICTION_THRESHOLD)",
    )
    parser.add_argument(
        "--limit", type=int, default=25, help="Rows to print (default: 25)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write full results to CSV"
    )
    args = parser.parse_args()

    settings = get_settings()
    data_dir = args.data_dir or settings.raw_data_path

    logger.info(f"Loading raw tables from {data_dir}")
    dataset = load_tables(data_dir)

    predictor = Predictor(threshold=args.threshold)

    # A single machine is a different question with a different answer shape:
    # one record, JSON, for a human or an API to consume directly.
    if args.machine is not None:
        machine_id = args.machine
        ids = dataset["machines"]["machine_id"]
        if pd.api.types.is_numeric_dtype(ids):
            try:
                machine_id = type(ids.iloc[0])(args.machine)
            except (TypeError, ValueError):
                pass
        print(json.dumps(predictor.predict_machine(dataset, machine_id), indent=2))
        return

    result = predictor.predict(dataset, latest_only=args.latest)

    if args.alerts_only:
        result = result[result["will_fail"]]
        if result.empty:
            logger.info(
                f"No machine is at or above the alert threshold "
                f"({predictor.threshold:.4f}). Nothing needs action."
            )
            return

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        logger.info(f"Wrote {len(result):,} rows to {args.output}")

    counts = result["risk_level"].value_counts().to_dict()
    ordered = {k: counts[k] for k in sorted(counts, key=lambda r: RISK_ORDER[r])}

    print()
    print(render_table(result, args.limit))
    if len(result) > args.limit:
        print(f"... {len(result) - args.limit:,} more rows (use --limit or -o)")
    print()
    print(f"risk levels: {ordered}")
    print(
        f"{int(result['will_fail'].sum()):,} of {len(result):,} rows at or above "
        f"threshold {predictor.threshold:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user.")
    except PredMaintenanceError as e:
        # Our own errors already carry actionable context — no stack trace needed.
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction script failed: {e}", exc_info=True)
        sys.exit(1)
