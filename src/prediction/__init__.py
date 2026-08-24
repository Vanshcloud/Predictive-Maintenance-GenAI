"""
Inference Pipeline Package
==========================
Turns raw sensor tables into per-machine failure predictions.

`Predictor` reuses `DataPreprocessor`'s feature logic rather than
reimplementing it — see the module docstring for why that matters.
"""

from src.prediction.predictor import Predictor

__all__ = ["Predictor"]
