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

        assert hasattr(sklearn, "__version__"), "Scikit-learn should be importable"

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

        assert hasattr(langchain, "__version__"), "LangChain should be importable"

    def test_fastapi_import(self):
        """FastAPI is our REST API framework."""
        import fastapi

        assert hasattr(fastapi, "FastAPI"), "FastAPI should have 'FastAPI' class"

    def test_pydantic_import(self):
        """Pydantic is used for data validation throughout the app."""
        import pydantic

        assert hasattr(pydantic, "BaseModel"), "Pydantic should have 'BaseModel' class"

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


class TestVersion:
    """
    The version is declared in three places and reported to operators by
    /health. A bump that lands in pyproject but not in settings makes the
    running service lie about which build it is.
    """

    def test_all_three_declarations_agree(self):
        import tomllib

        import src
        from config.settings import PROJECT_ROOT, get_settings

        with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
            packaged = tomllib.load(fh)["project"]["version"]

        assert get_settings().APP_VERSION == packaged, (
            f"APP_VERSION ({get_settings().APP_VERSION}) does not match "
            f"pyproject [project].version ({packaged}) — /health would report "
            "the wrong build."
        )
        assert src.__version__ == packaged


class TestServingContract:
    """
    The two configuration invariants that decide what a technician is told.

    These live here, in the smoke suite, and take no fixture — deliberately.
    The equivalent assertions in tests/unit/test_predictor.py hang off a
    `predictor` fixture that calls `pytest.skip` when no trained `.keras` is
    on disk, which is exactly the state CI runs in (models/ is gitignored).
    So the invariants were asserted only on machines that had already trained
    a model, and never on the gate that blocks a merge. Both facts below are
    readable from committed files alone, so both can be checked anywhere.
    """

    def test_risk_bands_ascend_and_high_is_the_alert_threshold(self):
        """
        Bands must ascend, and "high or above" must mean "alerting".

        `Predictor.risk_level()` tests the boundaries top-down, so a MEDIUM
        set above HIGH would not raise — it would silently make one band
        unreachable and the dashboard would stop showing a level the API can
        still emit. The HIGH == threshold equality is the stronger claim:
        the band boundary and the alert decision are one number, and if they
        drift the UI can show "medium" for a machine the API has flagged.
        """
        from config.settings import get_settings

        s = get_settings()
        assert s.RISK_BAND_MEDIUM < s.RISK_BAND_HIGH < s.RISK_BAND_CRITICAL, (
            "Risk bands must ascend low -> medium -> high -> critical; got "
            f"medium={s.RISK_BAND_MEDIUM}, high={s.RISK_BAND_HIGH}, "
            f"critical={s.RISK_BAND_CRITICAL}"
        )
        assert s.RISK_BAND_HIGH == s.PREDICTION_THRESHOLD, (
            f"RISK_BAND_HIGH ({s.RISK_BAND_HIGH}) must equal "
            f"PREDICTION_THRESHOLD ({s.PREDICTION_THRESHOLD}) so that the "
            '"high" band and the alert decision cannot diverge.'
        )

    def test_served_threshold_matches_the_committed_evaluation_report(self):
        """
        The threshold in settings must be the one evaluation actually chose.

        `scripts/evaluate_model.py` sweeps the validation PR curve and writes
        the chosen operating point to models/evaluation_report.json, but
        nothing copies it into `PREDICTION_THRESHOLD` — that edit is manual.
        A retrain that moves the optimum therefore leaves the API serving the
        PREVIOUS model's threshold, silently, with every test still green and
        every published metric describing an operating point nobody is using.
        Day 15 moved it from 0.6678 to 0.3415; this is what would have caught
        forgetting the second half of that change.

        The report is committed (the .keras is not), so this runs in CI.
        """
        import json

        from config.settings import PROJECT_ROOT, get_settings

        report_path = PROJECT_ROOT / "models" / "evaluation_report.json"
        if not report_path.exists():
            pytest.skip(f"no evaluation report at {report_path}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        chosen = report["test_at_chosen_threshold"]["threshold"]
        served = get_settings().PREDICTION_THRESHOLD

        # settings carries the rounded value a human can read and quote; the
        # report carries full float64 precision. Four places is the precision
        # README.md and docs/RESULTS.md publish, so it is the precision that
        # has to agree.
        assert round(served, 4) == round(chosen, 4), (
            f"PREDICTION_THRESHOLD is {served} but the committed evaluation "
            f"report chose {chosen}. Re-run scripts/evaluate_model.py and "
            "update config/settings.py (and RISK_BAND_HIGH with it)."
        )


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
        from src.utils.exceptions import DataValidationError, PredMaintenanceError

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
