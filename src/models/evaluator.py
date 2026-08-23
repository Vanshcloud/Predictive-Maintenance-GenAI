"""
src/models/evaluator.py — Model Evaluation
===================================================

WHY THIS FILE EXISTS:
    Evaluates the model and computes proper metrics for imbalanced datasets.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Return failure probabilities for X as a flat array."""
        return self._predict_in_batches(X, batch_size=batch_size).flatten()

    def sweep_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_fn: float = 100.0,
        cost_fp: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluate every threshold on the precision-recall curve.

        WHY THIS EXISTS:
            0.5 is an arbitrary cut point inherited from balanced problems.
            At a ~1:864 positive rate it is almost never the right operating
            point, and the choice is a *deployment* decision the sigmoid
            output already supports without retraining.

        WHY COST-WEIGHTED:
            Precision and recall are not equally valuable here. A missed
            failure stops a production line; a false alarm costs one
            unnecessary inspection. `cost_fn / cost_fp` expresses that ratio,
            and the cost-optimal threshold follows from it. The default 100:1
            is a placeholder — replace it with the plant's real numbers.

        Args:
            y_true: Binary ground-truth labels.
            y_prob: Predicted probabilities.
            cost_fn: Cost of one missed failure (false negative).
            cost_fp: Cost of one false alarm (false positive).

        Returns:
            Dict with the full curve plus the best-F1 and lowest-cost points.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns one more point than thresholds.
        precision, recall = precision[:-1], recall[:-1]

        denom = precision + recall
        f1 = np.where(
            denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0
        )

        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos
        tp = recall * n_pos
        fn = n_pos - tp
        # precision = tp / (tp + fp)  ->  fp = tp * (1 - precision) / precision
        fp = np.where(
            precision > 0,
            tp * (1 - precision) / np.where(precision > 0, precision, 1),
            n_neg,
        )
        total_cost = fn * cost_fn + fp * cost_fp

        best_f1_i = int(np.argmax(f1))
        best_cost_i = int(np.argmin(total_cost))

        def point(i: int) -> Dict[str, float]:
            return {
                "threshold": float(thresholds[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "false_negatives": int(round(float(fn[i]))),
                "false_positives": int(round(float(fp[i]))),
                "total_cost": float(total_cost[i]),
            }

        result = {
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "cost_ratio": f"{cost_fn:g}:{cost_fp:g}",
            "best_f1": point(best_f1_i),
            "lowest_cost": point(best_cost_i),
            "curve": {
                "thresholds": thresholds.tolist(),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "total_cost": total_cost.tolist(),
            },
        }

        logger.info(
            f"Threshold sweep over {len(thresholds):,} candidates "
            f"(AP={result['average_precision']:.4f})"
        )
        logger.info(
            f"  best F1     : t={result['best_f1']['threshold']:.4f} "
            f"P={result['best_f1']['precision']:.4f} "
            f"R={result['best_f1']['recall']:.4f} "
            f"F1={result['best_f1']['f1']:.4f}"
        )
        # A cost optimum sitting in the extreme tail is fitted to noise, not
        # signal. With cost_fn >> cost_fp the objective is dominated by "reach
        # recall 1.0 at any cost", and on a few hundred positives the cheapest
        # way to do that is a threshold near zero — which does not transfer.
        # Observed on Day 5: t=0.0003 scored best on validation and cost 15
        # points of test F1.
        lc = result["lowest_cost"]
        result["lowest_cost_is_degenerate"] = bool(
            lc["threshold"] < 0.01 or lc["precision"] < 0.7
        )
        if result["lowest_cost_is_degenerate"]:
            logger.warning(
                f"  Cost-optimal threshold looks degenerate "
                f"(t={lc['threshold']:.5f}, precision={lc['precision']:.4f}). "
                f"It is likely fitted to the tail of a small positive sample "
                f"and will not transfer. Prefer the best-F1 point."
            )

        logger.info(
            f"  lowest cost : t={result['lowest_cost']['threshold']:.4f} "
            f"P={result['lowest_cost']['precision']:.4f} "
            f"R={result['lowest_cost']['recall']:.4f} "
            f"(FN={result['lowest_cost']['false_negatives']}, "
            f"FP={result['lowest_cost']['false_positives']})"
        )
        return result

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
