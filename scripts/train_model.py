#!/usr/bin/env python3
"""
scripts/train_model.py — Train the Predictive Maintenance LSTM Model
=====================================================================

WHY THIS FILE EXISTS:
    This script orchestrates the loading of preprocessed data,
    the initialization of the LSTM model, the execution of the
    training loop, and the final evaluation on the test set.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src can be found if running from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This line is why the abseil ordering in docs/Day4.md holds for this script:
# it drags TensorFlow in before `src.models` pulls sklearn (and therefore
# Arrow) in below. isort placed it here on its own — third-party sorts above
# first-party — and that placement is the correct one, so it needs no
# skip_file. Only `numpy` precedes it, which links no abseil of its own.
# If this import is ever removed, TensorFlow must still be first.
from tensorflow import keras  # noqa: E402

from config.settings import get_settings
from src.models import ModelEvaluator, ModelTrainer, PredictiveMaintenanceModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(data_dir: Path) -> dict:
    """
    Load preprocessed numpy arrays from disk as memmaps.

    Returns a dict with train/test always present and val present only if
    run_preprocessing.py produced a validation split.
    """
    logger.info(f"Loading data from {data_dir}...")
    try:
        data = {
            "X_train": np.load(data_dir / "X_train.npy", mmap_mode="r"),
            "y_train": np.load(data_dir / "y_train.npy", mmap_mode="r"),
            "X_test": np.load(data_dir / "X_test.npy", mmap_mode="r"),
            "y_test": np.load(data_dir / "y_test.npy", mmap_mode="r"),
        }
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Please run scripts/run_preprocessing.py first.")
        sys.exit(1)

    val_x, val_y = data_dir / "X_val.npy", data_dir / "y_val.npy"
    if val_x.exists() and val_y.exists():
        data["X_val"] = np.load(val_x, mmap_mode="r")
        data["y_val"] = np.load(val_y, mmap_mode="r")

    for name in ("train", "val", "test"):
        key = f"X_{name}"
        if key in data:
            logger.info(
                f"  {key:<8} {str(data[key].shape):<22} "
                f"positives: {int(np.sum(data[f'y_{name}'])):,}"
            )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM Predictive Maintenance Model"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Adam learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for weight initialisation, dropout masks, and batch "
            "shuffling (default: 42, matching scripts/generate_data.py). "
            "Pass a different one to measure run-to-run variance."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint and state file left by a previous run",
    )
    parser.add_argument(
        "--monitor",
        default="val_f1",
        choices=["val_f1", "val_auc", "val_loss", "val_precision", "val_recall"],
        help=(
            "Validation metric driving early stopping and checkpointing "
            "(default: val_f1). val_auc saturates under this class imbalance "
            "and selects on noise — see docs/Day5.md."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()

    # 1. Load Data
    data = load_data(settings.processed_data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Model selection (early stopping, checkpointing) monitors the VALIDATION
    # split so the test set stays untouched until the final evaluation below.
    # Falling back to the test set would make the reported metrics optimistic;
    # warn loudly rather than doing it silently.
    if "X_val" in data:
        X_val, y_val = data["X_val"], data["y_val"]
    else:
        logger.warning(
            "No X_val.npy found — falling back to monitoring the TEST set. "
            "Reported metrics will be optimistic because early stopping and "
            "checkpoint selection will have observed them. Re-run "
            "scripts/run_preprocessing.py to generate a validation split."
        )
        X_val, y_val = X_test, y_test

    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]

    # 2. Initialize Model
    #
    # Seeded BEFORE the model is built, because that is when the LSTM kernels
    # are drawn from glorot_uniform. Without this the script was not
    # reproducible: two runs on identical data produced different weights,
    # different dropout masks, and a different test F1 — while README.md,
    # CLAUDE.md and docs/RESULTS.md all quoted 0.8949 as a fact about this
    # repository. An unreproducible headline metric is an unfalsifiable one.
    # The model was retrained under this seed on Day 15; the figures those
    # files now carry are ones `--seed 42` re-derives.
    #
    # `set_random_seed` is one call for all three generators Keras draws from
    # (Python `random`, NumPy, TensorFlow); seeding them separately is the
    # same thing with three more places to forget one.
    #
    # NOT enabled: tf.config.experimental.enable_op_determinism(). It would
    # also pin down non-deterministic GPU kernel reductions, but it disables
    # the fused cuDNN LSTM path and costs several times the training time.
    # Seeding alone is what this CPU-trained model needs.
    keras.utils.set_random_seed(args.seed)
    logger.info(f"Seeded all RNGs with {args.seed}.")

    logger.info("Initializing model...")
    model_wrapper = PredictiveMaintenanceModel(
        sequence_length=seq_length, n_features=n_features
    )

    # 3. Train
    logger.info("Setting up trainer...")
    trainer = ModelTrainer(model=model_wrapper)
    trainer.compile(learning_rate=args.learning_rate)

    model_save_path = settings.model_artifacts_path / f"{settings.MODEL_NAME}.keras"
    settings.model_artifacts_path.mkdir(parents=True, exist_ok=True)

    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(model_save_path),
        monitor=args.monitor,
        resume=args.resume,
    )

    # Per-epoch curves, kept for the Day 5 evaluation plots.
    history_path = settings.model_artifacts_path / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")

    # 4. Evaluate — the ONLY place the test set is scored.
    logger.info("Starting final evaluation on best model checkpoint...")
    best_model = PredictiveMaintenanceModel.load(model_save_path)
    evaluator = ModelEvaluator(model=best_model)
    metrics = evaluator.evaluate(X_test, y_test, threshold=0.5)

    # Save metrics
    metrics_path = settings.model_artifacts_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}", exc_info=True)
        sys.exit(1)
