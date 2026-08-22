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

from config.settings import get_settings
from src.models import ModelEvaluator, ModelTrainer, PredictiveMaintenanceModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed numpy arrays from disk."""
    logger.info(f"Loading data from {data_dir}...")
    try:
        X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
        y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")
        X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
        y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")

        logger.info(f"Loaded X_train shape: {X_train.shape}")
        logger.info(
            f"Loaded y_train shape: {y_train.shape} (positives: {np.sum(y_train)})"
        )
        logger.info(f"Loaded X_test shape:  {X_test.shape}")

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Please run scripts/run_preprocessing.py first.")
        sys.exit(1)


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
    args = parser.parse_args()

    settings = get_settings()

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_data(settings.processed_data_path)

    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]

    # 2. Initialize Model
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
        X_val=X_test,  # Monitoring via test set
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_path=str(model_save_path),
    )

    # Per-epoch curves, kept for the Day 5 evaluation plots.
    history_path = settings.model_artifacts_path / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")

    # 4. Evaluate
    logger.info("Starting final evaluation on best model checkpoint...")
    best_model = PredictiveMaintenanceModel.load(model_save_path)
    evaluator = ModelEvaluator(model=best_model)
    metrics = evaluator.evaluate(X_test, y_test, threshold=0.5)

    # Save metrics
    metrics_path = settings.model_artifacts_path / "metrics.json"
    with open(metrics_path, "w") as f:
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
