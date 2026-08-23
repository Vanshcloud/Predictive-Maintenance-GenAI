#!/usr/bin/env python3
"""
scripts/evaluate_model.py — Characterise a Trained Model
=========================================================

WHY THIS FILE EXISTS:
    Training answers "does it learn?". This script answers the questions a
    deployment actually needs: at what threshold should it alert, how many
    failures will it miss, how many false alarms will technicians chase, and
    is it still improving or just overfitting?

HOW IT WORKS:
    1. Score the VALIDATION split and sweep every threshold on its
       precision-recall curve.
    2. Pick an operating point there — best-F1 and lowest-cost.
    3. Score the TEST split ONCE, at the threshold chosen on validation.
    4. Plot training curves and the PR curve.

    The ordering is the whole point. Choosing a threshold on the test set
    and then reporting test metrics at that threshold is the same mistake as
    early-stopping on the test set: the number stops being an estimate of
    generalisation and becomes a description of the data it was tuned on.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src can be found if running from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402

# TensorFlow must be imported before pandas/scikit-learn pull in Arrow.
# See src/models/__init__.py — getting this order wrong deadlocks the process.
from src.models import ModelEvaluator, PredictiveMaintenanceModel  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_split(data_dir: Path, name: str):
    """Load one (X, y) split as memmaps, or (None, None) if absent."""
    x_path, y_path = data_dir / f"X_{name}.npy", data_dir / f"y_{name}.npy"
    if not (x_path.exists() and y_path.exists()):
        return None, None
    return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")


def plot_training_curves(history: dict, out_path: Path) -> None:
    """Plot per-epoch train vs validation curves (repays TD-3)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on a build machine
    import matplotlib.pyplot as plt

    panels = [
        ("loss", "Loss (class-weighted BCE)"),
        ("auc", "ROC-AUC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(epochs, history[key], marker="o", label=f"train {key}")
        val_key = f"val_{key}"
        if val_key in history:
            ax.plot(epochs, history[val_key], marker="s", label=f"val {key}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training curves — train vs validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved training curves to {out_path}")


def plot_pr_curve(sweep: dict, out_path: Path, chosen: float) -> None:
    """Plot the precision-recall curve and the chosen operating point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = sweep["curve"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(curve["recall"], curve["precision"], lw=2)
    ax1.scatter(
        [sweep["best_f1"]["recall"]],
        [sweep["best_f1"]["precision"]],
        color="crimson",
        zorder=5,
        label=f"best F1 (t={sweep['best_f1']['threshold']:.3f})",
    )
    ax1.scatter(
        [sweep["lowest_cost"]["recall"]],
        [sweep["lowest_cost"]["precision"]],
        color="darkgreen",
        marker="D",
        zorder=5,
        label=f"lowest cost (t={sweep['lowest_cost']['threshold']:.3f})",
    )
    base = sweep["n_positive"] / (sweep["n_positive"] + sweep["n_negative"])
    ax1.axhline(base, ls="--", color="grey", lw=1, label=f"base rate ({base:.4%})")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f"Precision-Recall (AP = {sweep['average_precision']:.4f})")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(curve["thresholds"], curve["total_cost"], lw=2)
    ax2.axvline(chosen, ls="--", color="darkgreen", label=f"chosen t={chosen:.3f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel(f"Total cost (FN:FP = {sweep['cost_ratio']})")
    ax2.set_title("Cost vs threshold")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved PR / cost curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model")
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=100.0,
        help="Cost of one missed failure (default: 100)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost of one false alarm (default: 1)",
    )
    parser.add_argument(
        "--select-by",
        choices=["f1", "cost"],
        default="f1",
        help=(
            "Which validation optimum to deploy (default: f1). 'cost' is "
            "available but defaults OFF: with only ~175 validation positives "
            "and a 100:1 cost ratio, the cost optimum collapses to a "
            "near-zero threshold that reaches recall 1.0 on validation and "
            "loses 15 points of F1 on test. See docs/Day5.md."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure output")
    args = parser.parse_args()

    settings = get_settings()
    data_dir = settings.processed_data_path
    artifacts = settings.model_artifacts_path
    model_path = artifacts / f"{settings.MODEL_NAME}.keras"

    if not model_path.exists():
        logger.error(f"No trained model at {model_path}. Run scripts/train_model.py.")
        sys.exit(1)

    model = PredictiveMaintenanceModel.load(model_path)
    evaluator = ModelEvaluator(model=model)

    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")
    if X_test is None:
        logger.error("No test split found. Run scripts/run_preprocessing.py.")
        sys.exit(1)

    report = {"model": str(model_path.name)}

    # ---- 1. Choose the operating point on VALIDATION -------------------
    if X_val is None:
        logger.warning(
            "No validation split found — falling back to selecting the "
            "threshold on the test set. The reported test metrics will be "
            "optimistic. Re-run scripts/run_preprocessing.py."
        )
        sweep_X, sweep_y, sweep_on = X_test, y_test, "test (FALLBACK)"
    else:
        sweep_X, sweep_y, sweep_on = X_val, y_val, "validation"

    logger.info(f"Scoring {len(sweep_X):,} {sweep_on} sequences...")
    y_prob_sel = evaluator.predict_proba(sweep_X)
    sweep = evaluator.sweep_thresholds(
        np.asarray(sweep_y), y_prob_sel, cost_fn=args.cost_fn, cost_fp=args.cost_fp
    )
    report["threshold_selection"] = {"selected_on": sweep_on, **sweep}
    report["threshold_selection"].pop("curve")  # keep the JSON readable

    chosen = (
        sweep["best_f1"]["threshold"]
        if args.select_by == "f1"
        else sweep["lowest_cost"]["threshold"]
    )
    logger.info(f"Chosen threshold ({args.select_by}, from {sweep_on}): {chosen:.4f}")

    # ---- 2. Score TEST once, at the chosen threshold -------------------
    logger.info("Final test evaluation (the test set is scored here and nowhere else)")
    report["test_at_chosen_threshold"] = evaluator.evaluate(
        X_test, y_test, threshold=chosen
    )
    report["test_at_chosen_threshold"]["threshold"] = chosen
    report["test_at_0.5"] = evaluator.evaluate(X_test, y_test, threshold=0.5)
    report["test_at_0.5"]["threshold"] = 0.5

    out = artifacts / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=4))
    logger.info(f"Saved evaluation report to {out}")

    # ---- 3. Figures ----------------------------------------------------
    if not args.no_plots:
        history_path = artifacts / "training_history.json"
        if history_path.exists():
            plot_training_curves(
                json.loads(history_path.read_text()), artifacts / "training_curves.png"
            )
        plot_pr_curve(sweep, artifacts / "pr_curve.png", chosen)

    # ---- 4. Summary ----------------------------------------------------
    a, b = report["test_at_chosen_threshold"], report["test_at_0.5"]
    logger.info("=" * 62)
    logger.info(f"{'metric':<12}{'t=0.5':>16}{f'  t={chosen:.4f}':>18}")
    for k in ("auc", "precision", "recall", "f1"):
        logger.info(f"{k:<12}{b[k]:>16.4f}{a[k]:>18.4f}")
    tn, fp = b["confusion_matrix"][0]
    fn, tp = b["confusion_matrix"][1]
    tn2, fp2 = a["confusion_matrix"][0]
    fn2, tp2 = a["confusion_matrix"][1]
    logger.info(f"{'missed':<12}{fn:>16}{fn2:>18}")
    logger.info(f"{'false alarms':<12}{fp:>16}{fp2:>18}")
    logger.info("=" * 62)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)
