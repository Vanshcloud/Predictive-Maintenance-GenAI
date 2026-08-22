"""
src/models/evaluator.py — Model Evaluation
===================================================

WHY THIS FILE EXISTS:
    Evaluates the model and computes proper metrics for imbalanced datasets.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.lstm_model import PredictiveMaintenanceModel
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates model performance using proper metrics."""

    def __init__(self, model: PredictiveMaintenanceModel):
        self.model = model.model

    def _predict_in_batches(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Run inference in manual batches via direct model calls.

        WHY: this mirrors the manual training loop in trainer.py. Both were
        written while Day 4's hangs were (incorrectly) blamed on Keras's
        background prefetch threads; the real cause was the TensorFlow/Arrow
        abseil collision documented in src/models/__init__.py.

        Calling the model directly is kept because it is simple, has no
        background threads, and lets the caller control batch size against a
        1 GB memmapped test set. model.predict() has not been re-benchmarked
        since the real fix landed.
        """
        n_samples = len(X)
        outputs = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = np.asarray(X[start:end], dtype=np.float32)
            outputs.append(self.model(batch, training=False).numpy())
        return np.concatenate(outputs, axis=0)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Probability threshold for classification (default 0.5)

        Returns:
            Dictionary of metrics
        """
        try:
            logger.info(f"Evaluating model on {len(X_test)} samples...")

            # Predict probabilities
            y_pred_prob = self._predict_in_batches(X_test).flatten()

            # Apply threshold for hard classes
            y_pred = (y_pred_prob >= threshold).astype(int)

            # Safeguard calculation if true labels only have one class
            try:
                auc = roc_auc_score(y_test, y_pred_prob)
            except ValueError:
                logger.warning("ROC AUC requires both classes in y_true, returning 0.0")
                auc = 0.0

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "auc": float(auc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": cm.tolist(),
            }

            logger.info("Evaluation Metrics:")
            logger.info(f"  AUC:       {auc:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise ModelTrainingError(f"Evaluation failed: {str(e)}")
