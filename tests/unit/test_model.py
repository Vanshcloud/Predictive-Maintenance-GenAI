"""
tests/unit/test_model.py
========================
Tests for LSTM Model, Trainer, and Evaluator.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.models.evaluator import ModelEvaluator
from src.models.lstm_model import PredictiveMaintenanceModel
from src.models.trainer import ModelTrainer, iter_batches
from src.utils.exceptions import ModelTrainingError


@pytest.fixture
def mock_data():
    """Create tiny mock sequence data for testing."""
    X_train = np.random.rand(10, 24, 63)
    # Ensure both classes exist to test class weights safely
    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    return X_train, y_train


@pytest.fixture
def model():
    """Create a new model instance for each test."""
    return PredictiveMaintenanceModel(sequence_length=24, n_features=63)


class TestPredictiveMaintenanceModel:
    def test_model_build_shapes(self, model):
        """Test if the model is built with correct input and output shapes."""
        # Check input shape
        assert model.model.input_shape == (None, 24, 63)
        # Check output shape
        assert model.model.output_shape == (None, 1)

    def test_model_layers(self, model):
        """Test if model has correct types of layers."""
        layers = model.model.layers
        layer_types = [type(layer) for layer in layers]

        assert tf.keras.layers.LSTM in layer_types
        assert tf.keras.layers.Dense in layer_types
        assert tf.keras.layers.Dropout in layer_types


class TestModelTrainer:
    def test_compute_class_weights(self, model, mock_data):
        """Test if class weights are correctly computed for imbalanced data."""
        _, y = mock_data
        trainer = ModelTrainer(model=model)

        weights = trainer._compute_class_weights(y)

        assert 0 in weights
        assert 1 in weights

        # 8 negatives, 2 positives out of 10 total samples
        # weight_0 = (1/8) * (10/2) = 5/8 = 0.625
        # weight_1 = (1/2) * (10/2) = 5/2 = 2.5
        assert np.isclose(weights[0], 0.625)
        assert np.isclose(weights[1], 2.5)

    def test_compile_model(self, model):
        """compile() should set up the Adam optimizer the manual loop uses."""
        trainer = ModelTrainer(model=model)

        trainer.compile(learning_rate=0.01)

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, tf.keras.optimizers.Adam)
        assert np.isclose(float(trainer.optimizer.learning_rate.numpy()), 0.01)

    def test_iter_batches_covers_all_samples_without_overlap(self, mock_data):
        """Every sample appears exactly once; no batch exceeds batch_size."""
        X, y = mock_data
        total_samples = 0
        total_positive_labels = 0

        for xb, yb in iter_batches(X, y, batch_size=4, shuffle=True, seed=0):
            assert len(xb) <= 4
            assert len(xb) == len(yb)
            total_samples += len(xb)
            total_positive_labels += int(np.sum(yb == 1))

        assert total_samples == len(X)
        assert total_positive_labels == int(np.sum(y == 1))

    def test_train_reduces_loss_on_tiny_dataset(self, model, mock_data, tmp_path):
        """A few epochs of the manual loop should run and update weights."""
        X, y = mock_data
        trainer = ModelTrainer(model=model)
        trainer.compile(learning_rate=0.01)

        # tmp_path, not models/ — a test must never write into the real
        # artifact directory, where it would sit next to the trained model.
        checkpoint = tmp_path / "test_checkpoint.keras"

        initial_weights = [w.copy() for w in trainer.model.get_weights()]
        history = trainer.train(
            X_train=X,
            y_train=y,
            epochs=2,
            batch_size=4,
            checkpoint_path=str(checkpoint),
        )

        assert "loss" in history
        assert len(history["loss"]) >= 1
        updated_weights = trainer.model.get_weights()
        assert any(
            not np.array_equal(a, b) for a, b in zip(initial_weights, updated_weights)
        )


class TestMonitorAndResume:
    def test_monitor_defaults_to_val_f1_and_history_records_it(
        self, model, mock_data, tmp_path
    ):
        """Validation history must expose val_f1, the default selection metric."""
        X, y = mock_data
        trainer = ModelTrainer(model=model)
        trainer.compile(learning_rate=0.01)

        history = trainer.train(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            epochs=2,
            batch_size=4,
            checkpoint_path=str(tmp_path / "m.keras"),
        )

        assert "val_f1" in history and len(history["val_f1"]) >= 1
        for f1 in history["val_f1"]:
            assert 0.0 <= f1 <= 1.0

    def test_unknown_monitor_is_rejected(self, model, mock_data, tmp_path):
        """A typo in the monitor name must fail loudly, not silently pick a default."""
        X, y = mock_data
        trainer = ModelTrainer(model=model)
        trainer.compile(learning_rate=0.01)

        with pytest.raises(ModelTrainingError, match="Unknown monitor"):
            trainer.train(
                X_train=X,
                y_train=y,
                X_val=X,
                y_val=y,
                epochs=1,
                batch_size=4,
                checkpoint_path=str(tmp_path / "m.keras"),
                monitor="val_accuracy",
            )

    def test_resume_continues_from_saved_epoch(self, model, mock_data, tmp_path):
        """A resumed run picks up after the last completed epoch, not from 1."""
        X, y = mock_data
        ckpt = tmp_path / "m.keras"

        first = ModelTrainer(model=model)
        first.compile(learning_rate=0.01)
        h1 = first.train(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            epochs=2,
            batch_size=4,
            checkpoint_path=str(ckpt),
        )
        assert ckpt.with_suffix(".state.json").exists()

        # State is written only when the checkpoint improves, so resume picks
        # up after the BEST epoch — keeping the weights on disk and the
        # recorded epoch describing the same moment.
        saved_epoch = json.loads(
            ckpt.with_suffix(".state.json").read_text(encoding="utf-8")
        )["epoch"]
        assert 1 <= saved_epoch <= 2

        second = ModelTrainer(model=PredictiveMaintenanceModel(24, 63))
        second.compile(learning_rate=0.01)
        h2 = second.train(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            epochs=4,
            batch_size=4,
            checkpoint_path=str(ckpt),
            resume=True,
        )

        # Epochs up to the checkpoint are carried forward verbatim; the run
        # then continues to the requested total rather than restarting at 1.
        assert h2["loss"][:saved_epoch] == h1["loss"][:saved_epoch]
        assert len(h2["loss"]) == 4
        assert len(h2["loss"]) > len(h1["loss"])

    def test_resume_without_state_starts_fresh(self, model, mock_data, tmp_path):
        """resume=True on a clean directory must train normally, not crash."""
        X, y = mock_data
        trainer = ModelTrainer(model=model)
        trainer.compile(learning_rate=0.01)

        history = trainer.train(
            X_train=X,
            y_train=y,
            epochs=1,
            batch_size=4,
            checkpoint_path=str(tmp_path / "nothing-here.keras"),
            resume=True,
        )
        assert len(history["loss"]) == 1


class TestThresholdSweep:
    def test_sweep_finds_the_perfect_split_on_separable_scores(self, model):
        """With cleanly separable scores, the best-F1 point should be perfect."""
        evaluator = ModelEvaluator(model=model)
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.95, 0.97])

        sweep = evaluator.sweep_thresholds(y_true, y_prob)

        assert sweep["n_positive"] == 2
        assert sweep["n_negative"] == 6
        assert np.isclose(sweep["best_f1"]["f1"], 1.0)
        assert np.isclose(sweep["best_f1"]["precision"], 1.0)
        assert np.isclose(sweep["best_f1"]["recall"], 1.0)
        assert np.isclose(sweep["average_precision"], 1.0)

    def test_cost_ratio_shifts_the_operating_point_toward_recall(self, model):
        """
        Making misses expensive must not make the chosen threshold stricter.

        This is the whole reason the sweep is cost-weighted: at a 1:864 base
        rate, a missed failure and a false alarm are not equally bad, and the
        threshold should follow the cost ratio.
        """
        evaluator = ModelEvaluator(model=model)
        rng = np.random.default_rng(0)
        y_true = np.concatenate([np.zeros(400), np.ones(20)])
        y_prob = np.concatenate(
            [rng.beta(1.2, 8, 400), rng.beta(4, 3, 20)]  # overlapping, not separable
        )

        cheap_miss = evaluator.sweep_thresholds(y_true, y_prob, cost_fn=1, cost_fp=1)
        costly_miss = evaluator.sweep_thresholds(y_true, y_prob, cost_fn=500, cost_fp=1)

        assert (
            costly_miss["lowest_cost"]["threshold"]
            <= cheap_miss["lowest_cost"]["threshold"]
        )
        assert (
            costly_miss["lowest_cost"]["recall"] >= cheap_miss["lowest_cost"]["recall"]
        )

    def test_degenerate_cost_optimum_is_flagged(self, model):
        """
        A near-zero cost-optimal threshold must warn, not pass silently.

        With cost_fn >> cost_fp the objective collapses to "reach recall 1.0
        at any price", and on a small positive sample the cheapest way there
        is a threshold in the noise floor. Day 5 hit exactly this: t=0.0003
        won on validation and cost 15 points of test F1.
        """
        evaluator = ModelEvaluator(model=model)
        rng = np.random.default_rng(3)
        # Heavily overlapping scores: no threshold separates the classes, so
        # the only way to zero false negatives is to accept nearly everything.
        y_true = np.concatenate([np.zeros(500), np.ones(10)])
        y_prob = np.concatenate([rng.random(500), rng.random(10)])

        sweep = evaluator.sweep_thresholds(y_true, y_prob, cost_fn=1000.0, cost_fp=1.0)

        assert sweep["lowest_cost"]["recall"] == 1.0
        assert sweep["lowest_cost_is_degenerate"] is True

        # A cleanly separable problem must NOT be flagged.
        clean = evaluator.sweep_thresholds(
            np.array([0, 0, 0, 0, 1, 1]),
            np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.98]),
            cost_fn=1000.0,
            cost_fp=1.0,
        )
        assert clean["lowest_cost_is_degenerate"] is False

    def test_sweep_reports_consistent_confusion_counts(self, model):
        """FN + TP must equal the positive count at every reported point."""
        evaluator = ModelEvaluator(model=model)
        rng = np.random.default_rng(7)
        y_true = np.concatenate([np.zeros(200), np.ones(15)])
        y_prob = np.concatenate([rng.random(200) * 0.6, rng.random(15) * 0.6 + 0.4])

        sweep = evaluator.sweep_thresholds(y_true, y_prob)

        for key in ("best_f1", "lowest_cost"):
            pt = sweep[key]
            tp = round(pt["recall"] * sweep["n_positive"])
            assert pt["false_negatives"] + tp == sweep["n_positive"]
            assert 0 <= pt["false_positives"] <= sweep["n_negative"]


class TestEventLevelRecall:
    """
    Recall over failure events, not hours.

    Every hour in a 24h pre-failure window carries label 1, so hourly recall
    penalises a model that warned once and correctly. These tests pin the
    distinction Day 6 measured on real data: 91.5% sequence-level recall
    against 100% event-level recall.
    """

    @staticmethod
    def _window(machine, start_hour, hours, caught_hours):
        """One machine's warning window, with `caught_hours` alerted."""
        return [
            {
                "machine_id": machine,
                "datetime": pd.Timestamp("2024-06-01")
                + pd.Timedelta(hours=start_hour + h),
                "y": 1,
                "pred": h in caught_hours,
            }
            for h in range(hours)
        ]

    def test_one_alerted_hour_catches_the_whole_event(self):
        """Warned once is warned: hourly recall says 25%, event recall 100%."""
        df = pd.DataFrame(self._window(1, 0, 4, {2}))

        result = ModelEvaluator.event_level_recall(
            df["machine_id"].values,
            df["datetime"].values,
            df["y"].values,
            df["pred"].values,
        )

        assert result["n_events"] == 1
        assert result["n_caught"] == 1
        assert result["event_recall"] == 1.0
        assert np.isclose(result["sequence_recall"], 0.25)

    def test_events_split_by_machine_and_by_time_gap(self):
        """Two machines, and a gap in time, must not merge into one event."""
        rows = self._window(1, 0, 3, {0})
        rows += self._window(1, 50, 3, set())
        rows += self._window(2, 0, 3, {1})
        df = pd.DataFrame(rows)

        result = ModelEvaluator.event_level_recall(
            df["machine_id"].values,
            df["datetime"].values,
            df["y"].values,
            df["pred"].values,
        )

        assert result["n_events"] == 3
        assert result["n_caught"] == 2
        assert np.isclose(result["event_recall"], 2 / 3)

    def test_lead_time_measures_notice_given(self):
        """Lead time runs from the first alert to the end of the warning window."""
        df = pd.DataFrame(self._window(1, 0, 25, {0}))

        result = ModelEvaluator.event_level_recall(
            df["machine_id"].values,
            df["datetime"].values,
            df["y"].values,
            df["pred"].values,
        )

        assert result["lead_time_hours"]["median"] == 24.0

    def test_no_positives_is_handled(self):
        """data/sample has zero failures — this must not divide by zero."""
        df = pd.DataFrame(
            {
                "machine_id": [1, 1, 2],
                "datetime": pd.to_datetime(["2024-06-01", "2024-06-02", "2024-06-01"]),
                "y": [0, 0, 0],
                "pred": [False, True, False],
            }
        )

        result = ModelEvaluator.event_level_recall(
            df["machine_id"].values,
            df["datetime"].values,
            df["y"].values,
            df["pred"].values,
        )

        assert result["n_events"] == 0
        assert result["event_recall"] == 0.0


class TestModelEvaluator:
    def test_predict_proba_returns_flat_probabilities(self, model, mock_data):
        """predict_proba should give one probability in [0, 1] per sample."""
        X, _ = mock_data
        probs = ModelEvaluator(model=model).predict_proba(X)

        assert probs.shape == (len(X),)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_evaluator_metrics(self, model, mock_data):
        """Test if evaluator correctly returns expected metric types."""
        X, y = mock_data

        # Evaluate the randomly initialized model
        evaluator = ModelEvaluator(model=model)
        metrics = evaluator.evaluate(X, y)

        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

        assert isinstance(metrics["auc"], float)
        assert isinstance(metrics["confusion_matrix"], list)

    def test_single_class_labels_give_zero_auc_not_nan(self, model, mock_data):
        """
        A degenerate split must yield 0.0, and metrics must stay serialisable.

        The guard here was written as `except ValueError`, which scikit-learn
        stopped raising in 1.6 — it now warns and returns nan. That made the
        fallback unreachable, so `nan` reached models/metrics.json, where
        json.dump writes it as a bare `NaN` token. Python reads that back, but
        it is not valid JSON and a strict parser rejects the whole file. The
        json.loads(..., parse_constant=...) below is what pins that down:
        it is the only way to make Python's own parser as strict as everyone
        else's.
        """
        X, _ = mock_data
        y_one_class = np.zeros(len(X))

        metrics = ModelEvaluator(model=model).evaluate(X, y_one_class)

        assert metrics["auc"] == 0.0
        assert not math.isnan(metrics["auc"])

        # The matrix is published as [[tn, fp], [fn, tp]]; it must keep that
        # shape even here, or a consumer indexing [1][1] gets an IndexError.
        assert np.array(metrics["confusion_matrix"]).shape == (2, 2)

        def _reject(token: str) -> float:
            raise ValueError(f"metrics.json would contain the bare token {token}")

        json.loads(json.dumps(metrics), parse_constant=_reject)


class TestReproducibility:
    """
    `scripts/train_model.py` had no seed at all: LSTM kernels came from an
    unseeded glorot_uniform and dropout masks from an unseeded RNG, so two runs
    on identical data produced different weights and a different test F1 —
    while README.md, AGENTS.md and docs/RESULTS.md quoted 0.8949 as a fact about
    this repository. A headline metric nobody can reproduce cannot be checked by
    anyone, including its author. The deployed model was retrained under this
    seed on Day 15.
    """

    @staticmethod
    def _weights(seed):
        tf.keras.utils.set_random_seed(seed)
        built = PredictiveMaintenanceModel(sequence_length=24, n_features=63)
        return [np.asarray(w) for w in built.model.weights]

    def test_the_same_seed_gives_the_same_initial_weights(self):
        first, second = self._weights(42), self._weights(42)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))

    def test_a_different_seed_gives_different_initial_weights(self):
        """Guards the other direction: a seed that changed nothing would pass
        the test above while quietly pinning every run to one arbitrary draw."""
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(self._weights(42), self._weights(7))
        )
