"""
tests/unit/test_preprocessing.py — Preprocessing Pipeline Tests
============================================================

WHY THIS FILE EXISTS:
    Feature engineering is where most ML bugs hide:
    - Rolling stats computed across machine boundaries
    - Labels off by one timestep
    - Scaler fitted on test data (leakage!)
    - Wrong LSTM input shape

    These tests catch those bugs before they silently degrade
    your model's performance.

WHAT WE TEST:
    1. Feature engineering — rolling stats, lag features
    2. Label creation — correct windowing around failures
    3. Temporal split — train before test, no overlap
    4. Normalization — fit on train only
    5. Sequence windowing — correct LSTM shapes
    6. Full pipeline — end-to-end integration
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor

# ===========================================================================
# FIXTURES
# ===========================================================================


@pytest.fixture
def small_dataset():
    """
    Create a minimal but complete dataset for testing.

    10 machines, 5 days (120 hours) of data.
    2 failures to generate positive labels.
    """
    n_machines = 3
    n_hours = 120
    start = datetime(2024, 1, 1)

    # Telemetry
    rows = []
    for mid in range(1, n_machines + 1):
        for h in range(n_hours):
            rows.append(
                {
                    "datetime": start + timedelta(hours=h),
                    "machine_id": mid,
                    "voltage": 170 + np.random.normal(0, 5),
                    "rotation": 450 + np.random.normal(0, 10),
                    "pressure": 100 + np.random.normal(0, 3),
                    "vibration": 40 + np.random.normal(0, 2),
                }
            )
    telemetry = pd.DataFrame(rows)
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])

    # Machines
    machines = pd.DataFrame(
        {
            "machine_id": [1, 2, 3],
            "model": ["model1", "model2", "model1"],
            "age": [5, 10, 15],
        }
    )

    # Failures (machine 1 at hour 72, machine 2 at hour 96)
    failures = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    start + timedelta(hours=72),
                    start + timedelta(hours=96),
                ]
            ),
            "machine_id": [1, 2],
            "failure": ["comp1", "comp2"],
        }
    )

    # Errors
    errors = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    start + timedelta(hours=48),
                    start + timedelta(hours=70),
                    start + timedelta(hours=71),
                ]
            ),
            "machine_id": [1, 1, 1],
            "error_id": ["error1", "error2", "error1"],
        }
    )

    # Maintenance
    maintenance = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    start + timedelta(hours=24),
                    start + timedelta(hours=50),
                ]
            ),
            "machine_id": [1, 2],
            "comp": ["comp1", "comp2"],
        }
    )

    return {
        "telemetry": telemetry,
        "machines": machines,
        "failures": failures,
        "errors": errors,
        "maintenance": maintenance,
    }


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor with small parameters for testing."""
    return DataPreprocessor(
        prediction_horizon=24,
        sequence_length=12,  # Smaller for testing
        test_ratio=0.2,
    )


# ===========================================================================
# MERGE TESTS
# ===========================================================================


class TestMergeTables:
    """Test the table merging step."""

    def test_merge_produces_dataframe(self, preprocessor, small_dataset):
        """Merging should produce a valid DataFrame."""
        result = preprocessor.merge_tables(small_dataset)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_merge_includes_sensor_columns(self, preprocessor, small_dataset):
        """Merged result should contain all sensor columns."""
        result = preprocessor.merge_tables(small_dataset)

        for col in ["voltage", "rotation", "pressure", "vibration"]:
            assert col in result.columns

    def test_merge_includes_machine_age(self, preprocessor, small_dataset):
        """Merged result should include machine age."""
        result = preprocessor.merge_tables(small_dataset)

        assert "age" in result.columns
        assert result["age"].notna().all()

    def test_merge_includes_error_counts(self, preprocessor, small_dataset):
        """Merged result should include error count features."""
        result = preprocessor.merge_tables(small_dataset)

        assert "error_count" in result.columns or "errors_last_24h" in result.columns

    def test_merge_preserves_row_count(self, preprocessor, small_dataset):
        """Merging should not add or remove telemetry rows."""
        n_telemetry = len(small_dataset["telemetry"])
        result = preprocessor.merge_tables(small_dataset)

        assert len(result) == n_telemetry


# ===========================================================================
# FEATURE ENGINEERING TESTS
# ===========================================================================


class TestFeatureEngineering:
    """Test rolling stats and lag feature creation."""

    def test_rolling_features_created(self, preprocessor, small_dataset):
        """Should create rolling mean and std features."""
        merged = preprocessor.merge_tables(small_dataset)
        result = preprocessor.engineer_features(merged)

        # Check for rolling features
        assert "voltage_rolling_mean_3h" in result.columns
        assert "voltage_rolling_std_24h" in result.columns
        assert "vibration_rolling_mean_12h" in result.columns

    def test_lag_features_created(self, preprocessor, small_dataset):
        """Should create lag features."""
        merged = preprocessor.merge_tables(small_dataset)
        result = preprocessor.engineer_features(merged)

        assert "voltage_lag_1h" in result.columns
        assert "rotation_lag_6h" in result.columns
        assert "pressure_lag_24h" in result.columns

    def test_change_features_created(self, preprocessor, small_dataset):
        """Should create rate-of-change features."""
        merged = preprocessor.merge_tables(small_dataset)
        result = preprocessor.engineer_features(merged)

        assert "voltage_change_1h" in result.columns
        assert "vibration_change_24h" in result.columns

    def test_no_nans_after_engineering(self, preprocessor, small_dataset):
        """Feature engineering should not leave NaN values."""
        merged = preprocessor.merge_tables(small_dataset)
        result = preprocessor.engineer_features(merged)

        # Check numeric columns only
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        nan_count = result[numeric_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after engineering"

    def test_more_columns_after_engineering(self, preprocessor, small_dataset):
        """Should have significantly more columns after engineering."""
        merged = preprocessor.merge_tables(small_dataset)
        n_before = len(merged.columns)

        result = preprocessor.engineer_features(merged)
        n_after = len(result.columns)

        assert (
            n_after > n_before
        ), f"Expected more columns after engineering: {n_before} → {n_after}"


# ===========================================================================
# LABEL CREATION TESTS
# ===========================================================================


class TestLabelCreation:
    """Test binary failure label creation."""

    def test_labels_created(self, preprocessor, small_dataset):
        """Should create a 'label' column."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        result = preprocessor.create_labels(featured, small_dataset["failures"])

        assert "label" in result.columns

    def test_labels_are_binary(self, preprocessor, small_dataset):
        """Labels should be 0 or 1 only."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        result = preprocessor.create_labels(featured, small_dataset["failures"])

        unique_labels = set(result["label"].unique())
        assert unique_labels.issubset({0, 1})

    def test_positive_labels_exist(self, preprocessor, small_dataset):
        """Should have some positive labels (failures exist)."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        result = preprocessor.create_labels(featured, small_dataset["failures"])

        assert result["label"].sum() > 0, "No positive labels created"

    def test_labels_near_failure_time(self, preprocessor, small_dataset):
        """Positive labels should cluster around failure times."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        result = preprocessor.create_labels(featured, small_dataset["failures"])

        # Machine 1 failed at hour 72 with horizon=24
        # So hours 48-72 should have label=1 for machine 1
        machine1_labels = result[result["machine_id"] == 1]
        positive_times = machine1_labels[machine1_labels["label"] == 1]["datetime"]

        if len(positive_times) > 0:
            failure_time = pd.Timestamp("2024-01-04 00:00:00")  # hour 72
            max_label_time = positive_times.max()
            assert (
                max_label_time <= failure_time
            ), f"Labels should not extend past failure time: {max_label_time}"

    def test_empty_failures_all_zero(self, preprocessor, small_dataset):
        """With no failures, all labels should be 0."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        result = preprocessor.create_labels(featured, pd.DataFrame())

        assert result["label"].sum() == 0


# ===========================================================================
# TEMPORAL SPLIT TESTS
# ===========================================================================


class TestTemporalSplit:
    """Test temporal train/test split."""

    def test_split_produces_two_dfs(self, preprocessor, small_dataset):
        """Should return train and test DataFrames."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)

        assert len(train) > 0
        assert len(test) > 0

    def test_train_before_test(self, preprocessor, small_dataset):
        """ALL training data must be BEFORE all test data."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)

        assert (
            train["datetime"].max() <= test["datetime"].min()
        ), "Data leakage! Train data extends past test data start."

    def test_no_overlap(self, preprocessor, small_dataset):
        """Train and test sets should not share any rows."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)

        train_times = set(zip(train["machine_id"], train["datetime"]))
        test_times = set(zip(test["machine_id"], test["datetime"]))
        overlap = train_times & test_times

        assert len(overlap) == 0, f"Found {len(overlap)} overlapping rows"


# ===========================================================================
# NORMALIZATION TESTS
# ===========================================================================


class TestNormalization:
    """Test feature normalization."""

    def test_scaler_fitted(self, preprocessor, small_dataset):
        """Scaler should be fitted after normalization."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)
        preprocessor.normalize(train, test)

        assert preprocessor.scaler is not None

    def test_train_approximately_normalized(self, preprocessor, small_dataset):
        """Training features should have mean≈0, std≈1 after scaling."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)
        train_scaled, _, feature_cols = preprocessor.normalize(train, test)

        # Check a few feature columns
        for col in feature_cols[:3]:
            mean = train_scaled[col].mean()
            std = train_scaled[col].std()
            assert abs(mean) < 0.1, f"{col} mean should be ≈0, got {mean}"
            # std should be close to 1 (with some tolerance for small data)
            assert 0.5 < std < 1.5, f"{col} std should be ≈1, got {std}"


# ===========================================================================
# SEQUENCE WINDOWING TESTS
# ===========================================================================


class TestSequenceWindowing:
    """Test LSTM sliding window creation."""

    def test_output_shape_3d(self, preprocessor, small_dataset):
        """X should be 3D: (samples, timesteps, features)."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)
        train_scaled, _, feature_cols = preprocessor.normalize(train, test)

        X, y = preprocessor.create_sequences(train_scaled, feature_cols)

        assert X.ndim == 3, f"Expected 3D array, got {X.ndim}D"
        assert X.shape[1] == preprocessor.sequence_length
        assert X.shape[2] == len(feature_cols)

    def test_y_shape_1d(self, preprocessor, small_dataset):
        """y should be 1D: (samples,)."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)
        train_scaled, _, feature_cols = preprocessor.normalize(train, test)

        X, y = preprocessor.create_sequences(train_scaled, feature_cols)

        assert y.ndim == 1
        assert len(X) == len(y)

    def test_x_y_same_sample_count(self, preprocessor, small_dataset):
        """X and y should have the same number of samples."""
        merged = preprocessor.merge_tables(small_dataset)
        featured = preprocessor.engineer_features(merged)
        labeled = preprocessor.create_labels(featured, small_dataset["failures"])
        train, test = preprocessor.temporal_split(labeled)
        train_scaled, _, feature_cols = preprocessor.normalize(train, test)

        X, y = preprocessor.create_sequences(train_scaled, feature_cols)

        assert X.shape[0] == y.shape[0]


# ===========================================================================
# FULL PIPELINE TEST
# ===========================================================================


class TestFullPipeline:
    """Test the complete end-to-end pipeline."""

    def test_pipeline_runs_successfully(self, preprocessor, small_dataset):
        """Full pipeline should complete without errors."""
        result = preprocessor.run_pipeline(small_dataset)

        assert "X_train" in result
        assert "y_train" in result
        assert "X_test" in result
        assert "y_test" in result
        assert "feature_columns" in result
        assert "scaler" in result
        assert "metadata" in result

    def test_pipeline_output_shapes(self, preprocessor, small_dataset):
        """Pipeline outputs should have correct shapes."""
        result = preprocessor.run_pipeline(small_dataset)

        # X should be 3D
        assert result["X_train"].ndim == 3
        assert result["X_test"].ndim == 3

        # y should be 1D
        assert result["y_train"].ndim == 1
        assert result["y_test"].ndim == 1

        # Timesteps should match sequence_length
        assert result["X_train"].shape[1] == preprocessor.sequence_length
        assert result["X_test"].shape[1] == preprocessor.sequence_length

        # Features should match
        n_features = len(result["feature_columns"])
        assert result["X_train"].shape[2] == n_features
        assert result["X_test"].shape[2] == n_features

    def test_pipeline_metadata(self, preprocessor, small_dataset):
        """Metadata should contain key information."""
        result = preprocessor.run_pipeline(small_dataset)

        meta = result["metadata"]
        assert meta["prediction_horizon"] == 24
        assert meta["sequence_length"] == 12
        assert meta["n_features"] > 0
        assert meta["train_samples"] > 0
        assert meta["test_samples"] > 0

    def test_pipeline_no_nan_in_output(self, preprocessor, small_dataset):
        """Final output should have no NaN values."""
        result = preprocessor.run_pipeline(small_dataset)

        assert not np.isnan(result["X_train"]).any(), "NaN in X_train"
        assert not np.isnan(result["y_train"]).any(), "NaN in y_train"
        assert not np.isnan(result["X_test"]).any(), "NaN in X_test"
        assert not np.isnan(result["y_test"]).any(), "NaN in y_test"
