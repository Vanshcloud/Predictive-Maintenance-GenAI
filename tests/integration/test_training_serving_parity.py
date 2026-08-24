"""
tests/integration/test_training_serving_parity.py
==================================================

WHY THIS FILE EXISTS:
    Risk R-6 — training/serving skew — is the highest-consequence, lowest-
    visibility failure mode in this project. If the inference path builds
    features even slightly differently from the training path, the model keeps
    returning well-formed probabilities that are quietly wrong. No exception,
    no warning, no shape error. Every unit test still passes.

    The only way to catch it is to run both paths over the same data and
    compare the numbers. That is what this does: score the raw CSV tables
    through `Predictor`, score the stored `X_test.npy` tensors through the
    model directly, and require the two to agree.

    These are marked `integration` and skip when the generated dataset is
    absent (it is gitignored, ~5 GB). Run them after any change to feature
    engineering, scaling, or windowing:

        python -m pytest tests/integration -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import get_settings
from src.prediction import Predictor
from src.utils.exceptions import ModelNotFoundError

pytestmark = pytest.mark.integration

RAW_TABLES = ("telemetry", "machines", "errors", "maintenance")


def _require(paths) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(
            "generated dataset/model not present (both are gitignored): "
            f"{missing[0]}. Run scripts/generate_data.py, run_preprocessing.py, "
            "and train_model.py."
        )


@pytest.fixture(scope="module")
def settings():
    return get_settings()


@pytest.fixture(scope="module")
def predictor(settings):
    _require([settings.model_path, settings.scaler_path, settings.feature_columns_path])
    try:
        return Predictor()
    except ModelNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def raw_dataset(settings):
    paths = [settings.raw_data_path / f"{name}.csv" for name in RAW_TABLES]
    _require(paths)
    return {name: pd.read_csv(p) for name, p in zip(RAW_TABLES, paths)}


@pytest.fixture(scope="module")
def scored(predictor, raw_dataset):
    """Full-fleet scoring through the inference path. Takes a few minutes."""
    return predictor.predict(raw_dataset)


def test_inference_matches_training_pipeline(predictor, settings, scored):
    """
    The two paths must produce the same probabilities.

    Aligns on the test period: preprocessing built test windows from the most
    recent slice, machine-major then time-ascending, so the same rows can be
    recovered from the predictor's full-period output.
    """
    x_test_path = settings.processed_data_path / "X_test.npy"
    _require([x_test_path])

    reference = predictor.predict_sequences(np.load(x_test_path, mmap_mode="r"))

    n_machines = scored["machine_id"].nunique()
    per_machine = len(reference) // n_machines
    assert per_machine * n_machines == len(reference), (
        "test tensor does not divide evenly across machines; the alignment "
        "assumption below no longer holds"
    )

    aligned = (
        scored.sort_values(["machine_id", "datetime"])
        .groupby("machine_id", group_keys=False)
        .tail(per_machine)
        .sort_values(["machine_id", "datetime"])["failure_probability"]
        .to_numpy()
    )
    assert len(aligned) == len(reference)

    # float32 reduction-order noise only — nothing that could move a decision.
    np.testing.assert_allclose(aligned, reference, atol=1e-6, rtol=0)


def test_alert_decisions_are_identical(predictor, settings, scored):
    """
    The number that actually matters: does either path raise a different alert?

    Probabilities agreeing to 1e-8 is reassuring; alert decisions agreeing is
    the operational guarantee. A single flipped decision is a work order that
    either did not happen or should not have.
    """
    x_test_path = settings.processed_data_path / "X_test.npy"
    _require([x_test_path])

    reference = predictor.predict_sequences(np.load(x_test_path, mmap_mode="r"))
    n_machines = scored["machine_id"].nunique()
    per_machine = len(reference) // n_machines

    aligned = (
        scored.sort_values(["machine_id", "datetime"])
        .groupby("machine_id", group_keys=False)
        .tail(per_machine)
        .sort_values(["machine_id", "datetime"])["failure_probability"]
        .to_numpy()
    )

    disagreements = int(
        ((aligned >= predictor.threshold) != (reference >= predictor.threshold)).sum()
    )
    assert disagreements == 0, (
        f"{disagreements} of {len(reference):,} alert decisions differ between "
        "the training and inference paths — this is training/serving skew"
    )


def test_every_machine_is_scored(predictor, raw_dataset, scored):
    """No machine may silently drop out of the fleet view."""
    expected = set(raw_dataset["machines"]["machine_id"])
    assert set(scored["machine_id"]) == expected


def test_latest_only_covers_the_whole_fleet(predictor, raw_dataset):
    """The dashboard's 'current status' must have a row per machine."""
    latest = predictor.predict(raw_dataset, latest_only=True)

    assert len(latest) == raw_dataset["machines"]["machine_id"].nunique()
    assert latest["machine_id"].is_unique
