"""
tests/unit/test_smoke.py — Smoke Tests
============================================================

WHY THIS FILE EXISTS:
    Smoke tests verify that the basic setup is working. They answer:
    "Can we import all our dependencies? Is the config loading?"

    If these tests fail, nothing else will work. Run them FIRST
    after setting up the project.

    Think of smoke tests like turning the key in a car's ignition.
    You're not testing if the car can drive cross-country — you're
    testing if the engine starts at all.

NAMING CONVENTION:
    - Test files: test_<what_you_are_testing>.py
    - Test functions: test_<specific_behavior>()
    - Test classes: Test<ComponentName>

RUN: python -m pytest tests/unit/test_smoke.py -v
"""

import pytest


class TestDependencyImports:
    """Verify all critical dependencies are installed and importable."""

    def test_numpy_import(self):
        """NumPy is our numerical computing foundation."""
        import numpy as np

        assert hasattr(np, "array"), "NumPy should have 'array' function"

    def test_pandas_import(self):
        """Pandas is used for all tabular data manipulation."""
        import pandas as pd

        assert hasattr(pd, "DataFrame"), "Pandas should have 'DataFrame' class"

    def test_sklearn_import(self):
        """Scikit-learn provides preprocessing and evaluation utilities."""
        import sklearn

        assert hasattr(
            sklearn, "__version__"
        ), "Scikit-learn should be importable"

    def test_tensorflow_import(self):
        """TensorFlow is our deep learning framework for the LSTM model."""
        import tensorflow as tf

        assert hasattr(tf, "keras"), "TensorFlow should have 'keras' module"
        # Verify we can create a simple tensor (proves GPU/CPU backend works)
        tensor = tf.constant([1, 2, 3])
        assert tensor.shape == (3,), "Should create a tensor of shape (3,)"

    def test_langchain_import(self):
        """LangChain orchestrates our GenAI pipeline."""
        import langchain

        assert hasattr(
            langchain, "__version__"
        ), "LangChain should be importable"

    def test_fastapi_import(self):
        """FastAPI is our REST API framework."""
        import fastapi

        assert hasattr(fastapi, "FastAPI"), "FastAPI should have 'FastAPI' class"

    def test_pydantic_import(self):
        """Pydantic is used for data validation throughout the app."""
        import pydantic

        assert hasattr(
            pydantic, "BaseModel"
        ), "Pydantic should have 'BaseModel' class"

    def test_loguru_import(self):
        """Loguru is our logging framework."""
        import loguru

        assert hasattr(loguru, "logger"), "Loguru should have 'logger' object"


class TestConfiguration:
    """Verify the configuration system loads correctly."""

    def test_settings_load(self):
        """Settings should load from .env without errors."""
        from config.settings import get_settings

        settings = get_settings()
        assert settings is not None, "Settings should not be None"

    def test_settings_app_name(self):
        """APP_NAME should match our project name."""
        from config.settings import get_settings

        settings = get_settings()
        assert settings.APP_NAME == "predictive-maintenance-genai"

    def test_settings_defaults(self):
        """Default values should be set correctly."""
        from config.settings import get_settings

        settings = get_settings()
        assert settings.API_PORT == 8000
        assert settings.APP_ENV == "development"

    def test_settings_paths(self):
        """Path properties should return Path objects."""
        from config.settings import get_settings

        settings = get_settings()
        assert settings.model_artifacts_path.name == "models"
        assert settings.raw_data_path.name == "raw"

    def test_settings_singleton(self):
        """get_settings() should return the same cached instance."""
        from config.settings import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2, "Should return cached singleton"


class TestLogger:
    """Verify the logging system is configured correctly."""

    def test_logger_creation(self):
        """Should create a logger without errors."""
        from src.utils.logger import get_logger

        logger = get_logger("test")
        assert logger is not None

    def test_logger_can_log(self):
        """Logger should be able to write messages without crashing."""
        from src.utils.logger import get_logger

        logger = get_logger("test")
        # These should not raise any exceptions
        logger.debug("Debug message from smoke test")
        logger.info("Info message from smoke test")
        logger.warning("Warning message from smoke test")


class TestExceptions:
    """Verify custom exceptions work correctly."""

    def test_base_exception(self):
        """Base exception should carry message and details."""
        from src.utils.exceptions import PredMaintenanceError

        error = PredMaintenanceError(
            message="Test error",
            details={"key": "value"},
        )
        assert str(error) == "Test error | Details: {'key': 'value'}"
        assert error.details == {"key": "value"}

    def test_data_validation_error(self):
        """DataValidationError should be catchable as PredMaintenanceError."""
        from src.utils.exceptions import (
            DataValidationError,
            PredMaintenanceError,
        )

        with pytest.raises(PredMaintenanceError):
            raise DataValidationError(
                message="Empty dataframe",
                details={"rows": 0},
            )

    def test_api_error_has_status_code(self):
        """APIError should carry an HTTP status code."""
        from src.utils.exceptions import APIError

        error = APIError(message="Not found", status_code=404)
        assert error.status_code == 404

    def test_exception_hierarchy(self):
        """All custom exceptions should inherit from PredMaintenanceError."""
        from src.utils.exceptions import (
            DataIngestionError,
            LLMConnectionError,
            ModelNotFoundError,
            PredMaintenanceError,
        )

        assert issubclass(DataIngestionError, PredMaintenanceError)
        assert issubclass(ModelNotFoundError, PredMaintenanceError)
        assert issubclass(LLMConnectionError, PredMaintenanceError)
