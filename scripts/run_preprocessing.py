"""
scripts/run_preprocessing.py — Run the Full Preprocessing Pipeline
============================================================

WHY THIS FILE EXISTS:
    This script ties together data loading, validation, and
    preprocessing into a single executable pipeline. It's the
    "one command" that takes you from raw CSV files to
    LSTM-ready numpy arrays.

USAGE:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --data-dir data/sample
    python scripts/run_preprocessing.py --horizon 48 --seq-len 48
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run the complete preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Input data directory (default: data/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data (default: data/processed/)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="LSTM sequence length in timesteps (default: 24)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio, the most recent slice (default: 0.2)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help=(
            "Validation ratio, the slice just before the test period "
            "(default: 0.15). Model selection uses this so the test set is "
            "touched once. Pass 0 for a two-way split."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()

    # --- Step 1: Load Data ---
    logger.info("📥 Loading data...")
    ingestion = DataIngestion()
    data_dir = args.data_dir or settings.raw_data_path
    dataset = ingestion.load_dataset(data_dir=data_dir)

    if not dataset or "telemetry" not in dataset:
        logger.error(
            "❌ No telemetry data found! "
            "Run 'python scripts/generate_data.py' first."
        )
        sys.exit(1)

    # --- Step 2: Validate Data ---
    logger.info("\n🔍 Validating data...")
    validator = DataValidator()
    reports = validator.validate_dataset(dataset)

    # Check for critical failures
    critical_failures = [
        name for name, report in reports.items() if not report.is_valid
    ]
    if critical_failures:
        logger.warning(
            f"⚠ Validation failures in: {critical_failures}. "
            "Proceeding with caution."
        )

    # --- Step 3: Preprocess ---
    logger.info("\n⚙️ Running preprocessing pipeline...")
    preprocessor = DataPreprocessor(
        prediction_horizon=args.horizon,
        sequence_length=args.seq_len,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir else settings.processed_data_path
    )
    result = preprocessor.run_pipeline(dataset, save_dir=output_dir)

    # --- Summary ---
    logger.info("")
    logger.info("🎉 PIPELINE COMPLETE")
    logger.info(f"  Output: {output_dir}")
    logger.info("  Files saved:")
    logger.info(f"    X_train.npy:  {result['X_train'].shape}")
    logger.info(f"    y_train.npy:  {result['y_train'].shape}")
    logger.info(f"    X_test.npy:   {result['X_test'].shape}")
    logger.info(f"    y_test.npy:   {result['y_test'].shape}")
    logger.info("    scaler.joblib")
    logger.info(f"    feature_columns.txt ({len(result['feature_columns'])} features)")
    logger.info("")
    logger.info("Next step: Model Training (Day 4)")


if __name__ == "__main__":
    main()
