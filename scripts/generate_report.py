#!/usr/bin/env python3
"""
scripts/generate_report.py — Turn Predictions Into Maintenance Reports
=======================================================================

WHY THIS FILE EXISTS:
    The end of the pipeline, and the point of the project: raw sensor tables
    in, a work order a technician can act on out.

HOW IT WORKS:
    Predictor scores the machine and extracts the evidence; ReportGenerator
    writes it up. `--dry-run` stops before the LLM call and prints the exact
    prompt, which is how you check grounding without a key, a network, or a
    bill — and how you debug a bad report, since a bad report is almost always
    a bad prompt.

USAGE:
    python scripts/generate_report.py --machine 51 --dry-run
    python scripts/generate_report.py --machine 51 --provider ollama
    python scripts/generate_report.py --fleet --data-dir data/sample
    python scripts/generate_report.py --machine 51 --ask "Which part fails first?"
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
from src.genai import ReportGenerator, format_machine_facts  # noqa: E402
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
    """Match the dtype of the machines table so lookups succeed."""
    ids = dataset["machines"]["machine_id"]
    if pd.api.types.is_numeric_dtype(ids):
        try:
            return type(ids.iloc[0])(raw)
        except (TypeError, ValueError):
            pass
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="Generate an AI maintenance report from sensor data"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Raw CSV directory")
    parser.add_argument("--machine", default=None, help="Machine to report on")
    parser.add_argument(
        "--fleet", action="store_true", help="Summarise the whole fleet instead"
    )
    parser.add_argument("--ask", default=None, help="Ask a question about the machine")
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "ollama"],
        default=None,
        help="LLM provider (default: whichever has credentials, else ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider's model name (e.g. an Ollama tag you have pulled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt instead of calling the LLM — no key or network needed",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Machines in the fleet summary (default 10)"
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write to file")
    args = parser.parse_args()

    if not args.machine and not args.fleet:
        parser.error("give --machine ID or --fleet")

    settings = get_settings()
    dataset = load_tables(args.data_dir or settings.raw_data_path)
    predictor = Predictor()

    # ---- Gather grounded facts (no LLM involved yet) -------------------
    if args.fleet:
        ranked = predictor.predict(dataset, latest_only=True)
        machine_ids = ranked.head(args.top)["machine_id"].tolist()
        records = [predictor.explain_machine(dataset, m) for m in machine_ids]
        prompt_preview = "\n\n---\n\n".join(format_machine_facts(r) for r in records)
    else:
        machine_id = coerce_machine_id(args.machine, dataset)
        records = [predictor.explain_machine(dataset, machine_id)]
        prompt_preview = format_machine_facts(records[0])

    if args.dry_run:
        print("=" * 72)
        print("PROMPT FACTS (this is everything the model will be given)")
        print("=" * 72)
        print(prompt_preview)
        print("=" * 72)
        print("Dry run — no LLM was called.")
        return

    # ---- Generate ------------------------------------------------------
    try:
        llm_kwargs = {"model": args.model} if args.model else {}
        generator = ReportGenerator(provider=args.provider, **llm_kwargs)
        if args.fleet:
            output = generator.generate_fleet_summary(records, limit=args.top)
        elif args.ask:
            output = generator.answer_question(records[0], args.ask)
        else:
            output = generator.generate_report(records[0])
    except LLMConnectionError as e:
        # The prediction succeeded; only the narrative failed. Say both, and
        # leave the operator with the number they can still act on.
        logger.error(str(e))
        print()
        print("The prediction is available; the written report is not.")
        for record in records[:5]:
            print(
                f"  machine {record['machine_id']}: "
                f"probability {record['failure_probability']:.4f} "
                f"({record['risk_level']}) "
                f"{'ALERT' if record['will_fail'] else ''}"
            )
        print()
        print("Re-run with --dry-run to see the grounded facts without an LLM.")
        sys.exit(2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        logger.info(f"Wrote report to {args.output}")

    print()
    print(output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except PredMaintenanceError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Report script failed: {e}", exc_info=True)
        sys.exit(1)
