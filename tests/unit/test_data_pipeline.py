"""
tests/unit/test_data_pipeline.py — Data Pipeline Unit Tests
============================================================

WHY THIS FILE EXISTS:
    Tests ensure that our data pipeline works correctly and
    continues working as we make changes. Without tests:
    - A small refactor could silently break data loading
    - Schema changes go undetected
    - Validation logic might have edge case bugs

WHAT WE TEST:
    1. DataIngestion — loading files, error handling, dataset loading
    2. DataValidator — schema, null, duplicate, and range checks
    3. Integration — ingestion → validation pipeline

TESTING STRATEGY:
    - Use small, synthetic DataFrames (not real data files)
    - Each test is independent (no shared state)
    - Tests are fast (< 1 second each)
    - Test both happy paths and error cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator, ValidationCheck, ValidationReport
from src.utils.exceptions import DataIngestionError

# ===========================================================================
# FIXTURES — Reusable test data
# ===========================================================================
# WHY fixtures?
#   Instead of creating test data in every test function, fixtures
#   create it once and share it. pytest handles setup/teardown.
# ===========================================================================


@pytest.fixture
def sample_telemetry() -> pd.DataFrame:
    """Create a small, valid telemetry DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
            "machine_id": np.repeat([1, 2], n_rows // 2),
            "voltage": np.random.normal(170, 15, n_rows),
            "rotation": np.random.normal(450, 50, n_rows),
            "pressure": np.random.normal(100, 12, n_rows),
            "vibration": np.random.normal(40, 8, n_rows),
        }
    )


@pytest.fixture
def sample_machines() -> pd.DataFrame:
    """Create a small, valid machines DataFrame."""
    return pd.DataFrame(
        {
            "machine_id": [1, 2, 3],
            "model": ["model1", "model2", "model1"],
            "age": [5, 10, 15],
        }
    )


@pytest.fixture
def sample_failures() -> pd.DataFrame:
    """Create a small failures DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-03-15", "2024-06-20"]),
            "machine_id": [1, 2],
            "failure": ["comp1", "comp3"],
        }
    )


@pytest.fixture
def temp_csv_dir(sample_telemetry, sample_machines, sample_failures):
    """Create a temporary directory with CSV files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        sample_telemetry.to_csv(tmpdir / "telemetry.csv", index=False)
        sample_machines.to_csv(tmpdir / "machines.csv", index=False)
        sample_failures.to_csv(tmpdir / "failures.csv", index=False)

        # Create errors and maintenance as well
        errors_df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-02-01"]),
                "machine_id": [1],
                "error_id": ["error1"],
            }
        )
        errors_df.to_csv(tmpdir / "errors.csv", index=False)

        maint_df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-15"]),
                "machine_id": [1],
                "comp": ["comp2"],
            }
        )
        maint_df.to_csv(tmpdir / "maintenance.csv", index=False)

        yield tmpdir


# ===========================================================================
# DATA INGESTION TESTS
# ===========================================================================


class TestDataIngestion:
    """Test the DataIngestion class."""

    def test_load_csv_file(self, temp_csv_dir):
        """Should load a CSV file successfully."""
        ingestion = DataIngestion()
        df = ingestion.load_file(temp_csv_dir / "telemetry.csv")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "voltage" in df.columns

    def test_load_file_not_found(self):
        """Should raise DataIngestionError for missing files."""
        ingestion = DataIngestion()

        with pytest.raises(DataIngestionError) as exc_info:
            ingestion.load_file("/nonexistent/path/data.csv")

        assert "not found" in str(exc_info.value).lower()

    def test_load_unsupported_format(self, temp_csv_dir):
        """Should raise DataIngestionError for unsupported formats."""
        ingestion = DataIngestion()

        # Create a dummy file with unsupported extension
        dummy_file = temp_csv_dir / "data.xyz"
        dummy_file.write_text("dummy")

        with pytest.raises(DataIngestionError) as exc_info:
            ingestion.load_file(dummy_file)

        assert "unsupported" in str(exc_info.value).lower()

    def test_load_dataset_complete(self, temp_csv_dir):
        """Should load all 5 tables from a directory."""
        ingestion = DataIngestion()
        dataset = ingestion.load_dataset(data_dir=temp_csv_dir)

        assert "telemetry" in dataset
        assert "machines" in dataset
        assert "failures" in dataset
        assert "errors" in dataset
        assert "maintenance" in dataset

    def test_load_dataset_partial(self, temp_csv_dir):
        """Should handle missing tables gracefully."""
        # Remove one table
        (temp_csv_dir / "errors.csv").unlink()

        ingestion = DataIngestion()
        dataset = ingestion.load_dataset(data_dir=temp_csv_dir)

        # Should load available tables
        assert "telemetry" in dataset
        assert "machines" in dataset
        # Missing table should not be in the dict
        assert "errors" not in dataset

    def test_load_csv_convenience(self, temp_csv_dir):
        """Test the load_csv convenience method."""
        ingestion = DataIngestion()
        df = ingestion.load_csv(
            str(temp_csv_dir / "telemetry.csv"),
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ===========================================================================
# DATA VALIDATOR TESTS
# ===========================================================================


class TestValidationCheck:
    """Test the ValidationCheck dataclass."""

    def test_check_creation(self):
        """Should create a validation check with all fields."""
        check = ValidationCheck(
            name="test_check",
            passed=True,
            message="All good",
        )
        assert check.name == "test_check"
        assert check.passed is True
        assert check.severity == "error"  # default

    def test_check_with_details(self):
        """Should store arbitrary details."""
        check = ValidationCheck(
            name="range_check",
            passed=False,
            message="Out of range",
            details={"column": "voltage", "min": 50, "max": 300},
        )
        assert check.details["column"] == "voltage"


class TestValidationReport:
    """Test the ValidationReport dataclass."""

    def test_report_valid_when_all_pass(self):
        """Report should be valid when all error checks pass."""
        report = ValidationReport(
            table_name="test",
            total_rows=100,
            total_columns=5,
            checks=[
                ValidationCheck(name="check1", passed=True),
                ValidationCheck(name="check2", passed=True),
            ],
        )
        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_report_invalid_when_error_fails(self):
        """Report should be invalid when any error check fails."""
        report = ValidationReport(
            table_name="test",
            total_rows=100,
            total_columns=5,
            checks=[
                ValidationCheck(name="check1", passed=True),
                ValidationCheck(name="check2", passed=False, severity="error"),
            ],
        )
        assert report.is_valid is False
        assert len(report.errors) == 1

    def test_report_valid_with_warnings_only(self):
        """Report should be valid if only warnings fail (not errors)."""
        report = ValidationReport(
            table_name="test",
            total_rows=100,
            total_columns=5,
            checks=[
                ValidationCheck(name="check1", passed=True),
                ValidationCheck(name="check2", passed=False, severity="warning"),
            ],
        )
        assert report.is_valid is True
        assert len(report.warnings) == 1

    def test_report_summary(self):
        """Summary should contain table name and status."""
        report = ValidationReport(
            table_name="telemetry",
            total_rows=1000,
            total_columns=6,
            checks=[ValidationCheck(name="check1", passed=True)],
        )
        summary = report.summary()
        assert "telemetry" in summary
        assert "PASSED" in summary


class TestDataValidator:
    """Test the DataValidator class."""

    def test_validate_valid_telemetry(self, sample_telemetry):
        """Valid telemetry should pass all checks."""
        validator = DataValidator()
        report = validator.validate(sample_telemetry, "telemetry")

        assert report.is_valid is True
        assert report.total_rows == 100

    def test_validate_valid_machines(self, sample_machines):
        """Valid machines data should pass all checks."""
        validator = DataValidator()
        report = validator.validate(sample_machines, "machines")

        assert report.is_valid is True

    def test_validate_empty_dataframe(self):
        """Empty DataFrame should fail the not_empty check."""
        validator = DataValidator()
        empty_df = pd.DataFrame(columns=["datetime", "machine_id", "voltage"])
        report = validator.validate(empty_df, "telemetry")

        assert report.is_valid is False
        error_names = [e.name for e in report.errors]
        assert "not_empty" in error_names

    def test_validate_missing_columns(self, sample_telemetry):
        """DataFrame missing required columns should fail schema check."""
        validator = DataValidator()
        # Drop a required column
        bad_df = sample_telemetry.drop(columns=["voltage"])
        report = validator.validate(bad_df, "telemetry")

        assert report.is_valid is False
        error_names = [e.name for e in report.errors]
        assert "schema_check" in error_names

    def test_validate_high_null_rate(self, sample_telemetry):
        """Should flag columns with null percentage above threshold."""
        validator = DataValidator(max_null_pct=5.0)

        # Introduce 10% nulls in voltage column
        bad_df = sample_telemetry.copy()
        null_indices = np.random.choice(len(bad_df), size=10, replace=False)
        bad_df.loc[null_indices, "voltage"] = np.nan

        report = validator.validate(bad_df, "telemetry")

        # Should have a null check failure
        null_checks = [c for c in report.checks if c.name == "null_check"]
        assert len(null_checks) == 1
        assert null_checks[0].passed is False

    def test_validate_no_nulls(self, sample_telemetry):
        """Clean data should pass null check."""
        validator = DataValidator()
        report = validator.validate(sample_telemetry, "telemetry")

        null_checks = [c for c in report.checks if c.name == "null_check"]
        assert len(null_checks) == 1
        assert null_checks[0].passed is True

    def test_validate_range_check(self):
        """Should flag out-of-range sensor values."""
        validator = DataValidator()

        # Create telemetry with some out-of-range values
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "machine_id": [1] * 100,
                "voltage": [170.0] * 95 + [999.0] * 5,  # 5% out of range
                "rotation": [450.0] * 100,
                "pressure": [100.0] * 100,
                "vibration": [40.0] * 100,
            }
        )

        report = validator.validate(df, "telemetry")

        range_checks = [c for c in report.checks if c.name == "range_check_voltage"]
        assert len(range_checks) == 1
        assert range_checks[0].passed is False  # >1% out of range

    def test_validate_duplicates(self, sample_telemetry):
        """Should detect duplicate rows."""
        validator = DataValidator(max_duplicate_pct=1.0)

        # Add duplicates
        duplicated = pd.concat(
            [sample_telemetry, sample_telemetry.head(5)],
            ignore_index=True,
        )

        report = validator.validate(duplicated, "telemetry")

        dup_checks = [c for c in report.checks if c.name == "duplicate_check"]
        assert len(dup_checks) == 1
        # 5 duplicates out of 105 = 4.8% > 1% threshold
        assert dup_checks[0].passed is False

    def test_validate_dataset(self, temp_csv_dir):
        """Should validate all tables in the dataset."""
        ingestion = DataIngestion()
        dataset = ingestion.load_dataset(data_dir=temp_csv_dir)

        validator = DataValidator()
        reports = validator.validate_dataset(dataset)

        assert len(reports) == len(dataset)
        for table_name, report in reports.items():
            assert isinstance(report, ValidationReport)
            assert report.table_name == table_name

    def test_validate_unknown_table(self):
        """Should handle tables without a defined schema."""
        validator = DataValidator()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        report = validator.validate(df, "unknown_table")
        # Should pass (no schema to enforce)
        assert report.is_valid is True
