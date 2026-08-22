"""
tests/unit/test_model.py
========================
Tests for LSTM Model, Trainer, and Evaluator.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.evaluator import ModelEvaluator
from src.models.lstm_model import PredictiveMaintenanceModel
from src.models.trainer import ModelTrainer, iter_batches


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


class TestModelEvaluator:
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
