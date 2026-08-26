"""
src/models/lstm_model.py — LSTM Model Architecture
===================================================

WHY THIS FILE EXISTS:
    We need to encapsulate the TensorFlow/Keras model definition.
    This module defines the neural network architecture used for
    time-series prediction of equipment failures.

HOW IT WORKS:
    Uses a Sequential Keras model with two LSTM layers and Dropout
    for regularization, ending with a Dense layer outputting a
    probability [0, 1].
"""

import os
from pathlib import Path

# Imported for its side effect as much as its namespace: this is the import that
# pulls TensorFlow in, and it must happen before sklearn/pandas load Arrow.
# See the note in src/models/__init__.py.
import tensorflow as tf  # noqa: F401
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model

from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictiveMaintenanceModel:
    """Wrapper class for the TensorFlow LSTM Model."""

    def __init__(self, sequence_length: int = 24, n_features: int = 63):
        """
        Initialize the model structure.

        Args:
            sequence_length: Number of timesteps per sequence (e.g., 24 hours).
            n_features: Number of features per timestep.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """
        Build and return the LSTM architecture.

        Returns:
            An UNCOMPILED, untrained tf.keras.Sequential model. Compilation is
            ModelTrainer.compile()'s job — the manual GradientTape loop needs
            an optimizer and loss it owns, not ones baked in here.
        """
        try:
            model = Sequential(
                [
                    Input(shape=(self.sequence_length, self.n_features)),
                    LSTM(128, return_sequences=True, name="lstm_1"),
                    Dropout(0.3, name="dropout_1"),
                    LSTM(64, return_sequences=False, name="lstm_2"),
                    Dropout(0.3, name="dropout_2"),
                    Dense(32, activation="relu", name="dense_1"),
                    Dense(1, activation="sigmoid", name="output"),
                ],
                name="predictive_maintenance_lstm",
            )

            logger.info(
                f"Built LSTM model: input_shape="
                f"({self.sequence_length}, {self.n_features})"
            )
            return model
        except Exception as e:
            logger.error(f"Failed to build LSTM model: {str(e)}")
            raise ModelTrainingError(f"Model building failed: {str(e)}")

    def save(self, filepath: str | Path) -> None:
        """Save the trained model to disk."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model to {filepath}: {str(e)}")
            raise ModelTrainingError(f"Model saving failed: {str(e)}")

    @classmethod
    def load(cls, filepath: str | Path) -> "PredictiveMaintenanceModel":
        """Load a trained model from disk."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            # Load model first to get input shapes
            loaded_keras_model = load_model(filepath)
            seq_len = loaded_keras_model.input_shape[1]
            n_feats = loaded_keras_model.input_shape[2]

            # Instantiate class and replace the model
            instance = cls(sequence_length=seq_len, n_features=n_feats)
            instance.model = loaded_keras_model

            logger.info(f"Model loaded successfully from {filepath}")
            return instance

        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise ModelTrainingError(f"Model loading failed: {str(e)}")
