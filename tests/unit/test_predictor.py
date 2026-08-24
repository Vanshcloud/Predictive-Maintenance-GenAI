"""
tests/unit/test_predictor.py
============================
Tests for the inference pipeline.

The contract tests matter more than the happy path here. A Predictor that
returns wrong numbers looks exactly like one that returns right numbers —
there is no exception, no warning, just quietly degraded predictions. So
most of these assert that mismatched artifacts are *rejected* rather than
tolerated.
"""

from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest

from src.prediction import Predictor
from src.utils.exceptions import ModelNotFoundError, PredictionError

# TensorFlow is imported by conftest before anything pulls in Arrow.


@pytest.fixture
def raw_dataset():
    """
    A small but complete set of raw tables, long enough to window.

    Feature engineering consumes the first 24 hours (rolling/lag windows) and
    a sequence needs 24 more, so 5 days per machine leaves comfortable margin.
    """
    start = datetime(2024, 3, 1)
    hours = 24 * 5
    rng = np.random.default_rng(11)

    rows = []
    for machine_id in (1, 2, 3):
        for h in range(hours):
            rows.append(
                {
                    "datetime": start + timedelta(hours=h),
                    "machine_id": machine_id,
                    "voltage": 170 + rng.normal(0, 15),
                    "rotation": 450 + rng.normal(0, 50),
                    "pressure": 100 + rng.normal(0, 12),
                    "vibration": 40 + rng.normal(0, 8),
                }
            )
    telemetry = pd.DataFrame(rows)

    machines = pd.DataFrame(
        {
            "machine_id": [1, 2, 3],
            "model": ["model1", "model2", "model1"],
            "age": [5, 12, 18],
        }
    )
    errors = pd.DataFrame(
        {
            "datetime": [start + timedelta(hours=30), start + timedelta(hours=60)],
            "machine_id": [1, 2],
            "error_id": ["error1", "error3"],
        }
    )
    maintenance = pd.DataFrame(
        {
            "datetime": [start + timedelta(hours=10), start + timedelta(hours=50)],
            "machine_id": [1, 3],
            "comp": ["comp1", "comp2"],
        }
    )
    return {
        "telemetry": telemetry,
        "machines": machines,
        "errors": errors,
        "maintenance": maintenance,
    }


@pytest.fixture(scope="module")
def predictor():
    """A Predictor backed by the real trained artifacts."""
    try:
        return Predictor()
    except ModelNotFoundError as e:
        pytest.skip(f"trained artifacts not available: {e}")


class TestPredictorContract:
    """The artifacts must agree with each other, or startup must fail."""

    def test_loads_with_real_artifacts(self, predictor):
        assert len(predictor.feature_columns) == 63
        assert predictor.sequence_length == 24
        assert 0.0 < predictor.threshold < 1.0

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(ModelNotFoundError, match="model not found"):
            Predictor(model_path=tmp_path / "nope.keras")

    def test_feature_count_mismatch_is_rejected(self, predictor, tmp_path):
        """
        A feature list that disagrees with the model must refuse to load.

        This is the training/serving skew guard: pairing a model with the
        feature contract from a different preprocessing run would produce
        correctly-shaped, silently-wrong predictions.
        """
        short_list = tmp_path / "feature_columns.txt"
        short_list.write_text("\n".join(predictor.feature_columns[:10]))

        with pytest.raises(PredictionError, match="Feature count mismatch"):
            Predictor(feature_columns_path=short_list)

    def test_scaler_mismatch_is_rejected(self, predictor, tmp_path):
        """A scaler fitted on a different feature count must be caught."""
        from sklearn.preprocessing import StandardScaler

        wrong = StandardScaler().fit(np.random.rand(50, 10))
        wrong_path = tmp_path / "scaler.joblib"
        joblib.dump(wrong, wrong_path)

        with pytest.raises(PredictionError, match="Scaler mismatch"):
            Predictor(scaler_path=wrong_path)

    def test_sequence_length_mismatch_is_rejected(self, predictor):
        with pytest.raises(PredictionError, match="Sequence length mismatch"):
            Predictor(sequence_length=48)


class TestRiskBanding:
    def test_bands_are_ordered_and_cover_the_range(self, predictor):
        s = predictor.settings
        assert predictor.risk_level(0.0) == "low"
        assert predictor.risk_level(s.RISK_BAND_MEDIUM) == "medium"
        assert predictor.risk_level(s.RISK_BAND_HIGH) == "high"
        assert predictor.risk_level(s.RISK_BAND_CRITICAL) == "critical"
        assert predictor.risk_level(1.0) == "critical"

    def test_high_band_starts_exactly_at_the_alert_threshold(self, predictor):
        """
        "High or above" must mean exactly "the model is alerting".

        If the band boundary and the decision threshold drift apart, a
        dashboard can show 'medium' for a machine the API has flagged.
        """
        assert predictor.settings.RISK_BAND_HIGH == predictor.threshold
        assert predictor.risk_level(predictor.threshold) == "high"
        assert predictor.risk_level(predictor.threshold - 1e-6) != "high"


class TestPredictSequences:
    def test_probabilities_are_in_range(self, predictor):
        X = np.random.rand(8, 24, 63).astype(np.float32)
        probs = predictor.predict_sequences(X)

        assert probs.shape == (8,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_wrong_shape_is_rejected(self, predictor):
        with pytest.raises(PredictionError, match="3D"):
            predictor.predict_sequences(np.random.rand(8, 24).astype(np.float32))

        with pytest.raises(PredictionError, match="Sequence shape mismatch"):
            predictor.predict_sequences(np.random.rand(4, 12, 63).astype(np.float32))

    def test_batching_does_not_change_results(self, predictor):
        """
        Batch size is a memory knob, not a correctness one.

        Asserted on ABSOLUTE tolerance, deliberately. These are probabilities
        in [0, 1] and many are ~1e-8, where relative tolerance measures
        nothing useful — a 1e-14 float32 reduction-order difference reads as
        a 1e-6 relative error purely because the denominator is tiny. What
        matters operationally is that no alert decision could flip.
        """
        X = np.random.rand(20, 24, 63).astype(np.float32)

        one_batch = predictor.predict_sequences(X, batch_size=64)
        many_batches = predictor.predict_sequences(X, batch_size=3)

        np.testing.assert_allclose(one_batch, many_batches, atol=1e-9, rtol=0)
        assert (
            (one_batch >= predictor.threshold) == (many_batches >= predictor.threshold)
        ).all(), "batch size must never change an alert decision"


class TestPredictEndToEnd:
    def test_predict_returns_one_row_per_machine_per_window(
        self, predictor, raw_dataset
    ):
        result = predictor.predict(raw_dataset)

        assert not result.empty
        for col in (
            "machine_id",
            "datetime",
            "failure_probability",
            "risk_level",
            "will_fail",
        ):
            assert col in result.columns

        assert set(result["machine_id"].unique()) == {1, 2, 3}
        assert result["failure_probability"].between(0.0, 1.0).all()

    def test_results_are_sorted_most_urgent_first(self, predictor, raw_dataset):
        """A fleet view is useless if the operator has to sort it themselves."""
        result = predictor.predict(raw_dataset)
        probs = result["failure_probability"].values
        assert np.all(np.diff(probs) <= 1e-12)

    def test_will_fail_agrees_with_the_threshold(self, predictor, raw_dataset):
        result = predictor.predict(raw_dataset)
        expected = result["failure_probability"] >= predictor.threshold
        assert (result["will_fail"] == expected).all()

    def test_latest_only_gives_one_row_per_machine(self, predictor, raw_dataset):
        result = predictor.predict(raw_dataset, latest_only=True)

        assert len(result) == result["machine_id"].nunique() == 3
        full = predictor.predict(raw_dataset)
        for machine_id, row in result.set_index("machine_id").iterrows():
            latest = full[full["machine_id"] == machine_id]["datetime"].max()
            assert row["datetime"] == latest

    def test_failures_table_is_ignored_if_supplied(self, predictor, raw_dataset):
        """
        At inference the future is what we predict, not something we read.

        A caller passing yesterday's failures table must not change today's
        prediction — otherwise the pipeline would leak labels into serving.
        """
        without = predictor.predict(raw_dataset, latest_only=True)

        with_failures = dict(raw_dataset)
        with_failures["failures"] = pd.DataFrame(
            {
                "datetime": [raw_dataset["telemetry"]["datetime"].max()],
                "machine_id": [1],
                "failure": ["comp1"],
            }
        )
        with_ = predictor.predict(with_failures, latest_only=True)

        np.testing.assert_allclose(
            without.sort_values("machine_id")["failure_probability"].values,
            with_.sort_values("machine_id")["failure_probability"].values,
            rtol=1e-6,
        )

    def test_insufficient_history_raises_a_useful_error(self, predictor, raw_dataset):
        """Too little history must say so, not return an empty frame."""
        short = dict(raw_dataset)
        short["telemetry"] = raw_dataset["telemetry"].groupby("machine_id").head(10)

        with pytest.raises(PredictionError, match="sequences"):
            predictor.predict(short)

    def test_predict_machine_returns_a_serialisable_record(
        self, predictor, raw_dataset
    ):
        record = predictor.predict_machine(raw_dataset, machine_id=2)

        assert record["machine_id"] == 2
        assert isinstance(record["failure_probability"], float)
        assert isinstance(record["will_fail"], bool)
        assert record["risk_level"] in {"low", "medium", "high", "critical"}
        assert record["threshold"] == predictor.threshold

    def test_predict_machine_record_is_json_serialisable(self, predictor, raw_dataset):
        """
        pandas hands back np.int64 / np.bool_, which json.dumps refuses.

        This record is what the Day 9 API returns and what the Day 7 report
        generator consumes, so the unwrapping has to happen here rather than
        being rediscovered by every caller.
        """
        import json

        record = predictor.predict_machine(raw_dataset, machine_id=1)
        round_tripped = json.loads(json.dumps(record))

        assert round_tripped["machine_id"] == 1
        assert isinstance(round_tripped["will_fail"], bool)
        assert isinstance(round_tripped["threshold"], float)

    def test_predict_machine_rejects_an_unknown_machine(self, predictor, raw_dataset):
        with pytest.raises(PredictionError, match="No prediction produced"):
            predictor.predict_machine(raw_dataset, machine_id=9999)
