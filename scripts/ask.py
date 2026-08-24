#!/usr/bin/env python3
"""
scripts/ask.py — Interactive Maintenance Q&A
=============================================

WHY THIS FILE EXISTS:
    A report answers the questions its author thought to ask. A technician at
    the machine has others, and each one depends on the last. This is the
    conversational front end to that.

HOW IT WORKS:
    Scores one machine, opens a session pinned to it, and reads questions
    until you stop. The machine's data is fixed for the session — every answer
    comes from the same facts, so the conversation cannot drift onto a
    different machine or a re-scored probability halfway through.

USAGE:
    python scripts/ask.py --machine 51
    python scripts/ask.py --machine 51 --ask "Has vibration been rising?"
    python scripts/ask.py --machine 3 --data-dir data/sample --show-facts

    In the session:  /facts  show the grounded data
                     /reset  clear the conversation, keep the machine
                     /quit   exit
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src can be found if running from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TensorFlow must load before pandas/scikit-learn pull in Arrow — see
# src/models/__init__.py. Importing Predictor first guarantees the order.
from src.prediction import Predictor  # noqa: E402  isort:skip

from config.settings import get_settings  # noqa: E402
from src.genai import MaintenanceAssistant, format_machine_facts  # noqa: E402
from src.utils.exceptions import LLMConnectionError, PredMaintenanceError  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

RAW_TABLES = ("telemetry", "machines", "errors", "maintenance")


def load_tables(data_dir: Path) -> dict:
    dataset = {}
    for name in RAW_TABLES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            logger.error(f"Missing {path}. Run `python scripts/generate_data.py`.")
            sys.exit(1)
        dataset[name] = pd.read_csv(path)
    return dataset


def coerce_machine_id(raw: str, dataset: dict):
    ids = dataset["machines"]["machine_id"]
    if pd.api.types.is_numeric_dtype(ids):
        try:
            return type(ids.iloc[0])(raw)
        except (TypeError, ValueError):
            pass
    return raw


def main():
    parser = argparse.ArgumentParser(description="Ask questions about a machine")
    parser.add_argument("--machine", required=True, help="Machine to discuss")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--ask", default=None, help="Ask one question and exit (non-interactive)"
    )
    parser.add_argument(
        "--history-hours",
        type=int,
        default=24,
        help="Hours of trend to ground history questions in (default: 24)",
    )
    parser.add_argument(
        "--provider", choices=["openai", "google", "ollama"], default=None
    )
    parser.add_argument("--model", default=None, help="Override the provider's model")
    parser.add_argument(
        "--show-facts", action="store_true", help="Print the grounded data on start"
    )
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_tables(args.data_dir or settings.raw_data_path)
    machine_id = coerce_machine_id(args.machine, dataset)

    predictor = Predictor()
    record = predictor.explain_machine(
        dataset, machine_id, history_hours=args.history_hours
    )

    if args.show_facts:
        print(format_machine_facts(record))
        print()

    llm_kwargs = {"model": args.model} if args.model else {}
    try:
        assistant = MaintenanceAssistant(provider=args.provider, **llm_kwargs)
    except LLMConnectionError as e:
        # The prediction stands even when no model is reachable.
        logger.error(str(e))
        print(
            f"\nmachine {record['machine_id']}: "
            f"probability {record['failure_probability']:.4f} "
            f"({record['risk_level']})"
            f"{'  ALERT' if record['will_fail'] else ''}"
        )
        print(
            "No LLM is available, so questions cannot be answered. "
            "Re-run with --show-facts to read the data directly."
        )
        sys.exit(2)

    assistant.start_session(record)

    header = (
        f"machine {record['machine_id']} — "
        f"probability {record['failure_probability']:.4f} ({record['risk_level']})"
    )
    print(header)
    print("-" * len(header))

    if args.ask:
        print(assistant.ask(args.ask))
        return

    print("Ask a question, or /facts, /reset, /quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue
        if question in ("/quit", "/exit", "/q"):
            return
        if question == "/facts":
            print(format_machine_facts(record))
            continue
        if question == "/reset":
            assistant.reset()
            print("(conversation cleared)")
            continue

        try:
            print()
            print(assistant.ask(question))
        except LLMConnectionError as e:
            # Don't kill the session — the provider may come back.
            logger.error(str(e))
            print("The model is unreachable. The prediction above still stands.")
        except PredMaintenanceError as e:
            print(f"({e})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
    except PredMaintenanceError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Assistant script failed: {e}", exc_info=True)
        sys.exit(1)
