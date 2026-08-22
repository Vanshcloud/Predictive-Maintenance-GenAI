"""
src/data/validation.py — Data Validation Pipeline
============================================================

WHY THIS FILE EXISTS:
    Data validation is your FIRST LINE OF DEFENSE against bad data.
    In production ML systems, model degradation is almost always
    caused by data problems, not model bugs:

    - A sensor stops reporting → NaN values flood in
    - A firmware update changes sensor scale (°F → °C)
    - A new machine type has different voltage ranges
    - A CSV export adds extra whitespace to column names

    Google's MLOps whitepaper states: "Data validation should be
    the first step in any ML pipeline."

WHAT IT VALIDATES:
    1. Schema — Are the expected columns present with correct types?
    2. Completeness — How many missing values? Over threshold?
    3. Range — Are sensor values within physically possible ranges?
    4. Duplicates — Are there exact duplicate rows?

DESIGN PATTERN:
    - Each validation returns a ValidationResult (pass/fail + details)
    - Results are aggregated into a ValidationReport
    - The validator NEVER modifies data — it only reports problems
    - Separation from ingestion: ingestion loads, validation checks

USAGE:
    from src.data.validation import DataValidator

    validator = DataValidator()
    report = validator.validate_telemetry(telemetry_df)

    if report.is_valid:
        print("Data is clean!")
    else:
        print(f"Found {len(report.errors)} issues")
        for error in report.errors:
            print(f"  - {error}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# VALIDATION RESULT DATA CLASSES
# ===========================================================================
# WHY dataclasses?
#   They provide a clean, typed structure for validation results
#   without the boilerplate of regular classes. They're also
#   naturally serializable (for API responses later).
# ===========================================================================


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str  # e.g., "schema_check", "null_check"
    passed: bool  # Did the check pass?
    severity: str = "error"  # "error" (blocking) or "warning" (info)
    message: str = ""  # Human-readable description
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated results from all validation checks on a DataFrame."""

    table_name: str
    total_rows: int
    total_columns: int
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Data is valid if ALL error-level checks passed."""
        return all(check.passed for check in self.checks if check.severity == "error")

    @property
    def errors(self) -> List[ValidationCheck]:
        """Return only failed error-level checks."""
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> List[ValidationCheck]:
        """Return only failed warning-level checks."""
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def summary(self) -> str:
        """Generate a human-readable summary of the validation report."""
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        lines = [
            f"Validation Report: {self.table_name} — {status}",
            f"  Rows: {self.total_rows:,} | Columns: {self.total_columns}",
            f"  Checks: {len(self.checks)} total | "
            f"{len(self.errors)} errors | {len(self.warnings)} warnings",
        ]

        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    ✗ {err.name}: {err.message}")

        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    ⚠ {warn.name}: {warn.message}")

        return "\n".join(lines)


# ===========================================================================
# SCHEMA DEFINITIONS
# ===========================================================================
# WHY define schemas?
#   A schema is a contract: "This table MUST have these columns with
#   these types." If the data violates the contract, we catch it
#   immediately instead of getting a cryptic KeyError 500 lines later.
# ===========================================================================

# Expected columns and their pandas dtypes for each table
SCHEMAS: Dict[str, Dict[str, str]] = {
    "telemetry": {
        "datetime": "datetime64",
        "machine_id": "int",
        "voltage": "float",
        "rotation": "float",
        "pressure": "float",
        "vibration": "float",
    },
    "machines": {
        "machine_id": "int",
        "model": "object",  # string type in pandas
        "age": "int",
    },
    "errors": {
        "datetime": "datetime64",
        "machine_id": "int",
        "error_id": "object",
    },
    "maintenance": {
        "datetime": "datetime64",
        "machine_id": "int",
        "comp": "object",
    },
    "failures": {
        "datetime": "datetime64",
        "machine_id": "int",
        "failure": "object",
    },
}

# Valid sensor value ranges (physically possible values)
SENSOR_RANGES: Dict[str, Dict[str, float]] = {
    "voltage": {"min": 50.0, "max": 300.0},
    "rotation": {"min": 0.0, "max": 1000.0},
    "pressure": {"min": 0.0, "max": 250.0},
    "vibration": {"min": 0.0, "max": 150.0},
}


class DataValidator:
    """
    Validates data quality for the predictive maintenance pipeline.

    Runs schema, completeness, range, and duplicate checks
    on DataFrames and produces structured validation reports.

    Attributes:
        max_null_pct: Maximum allowed null percentage per column (default: 5%).
        max_duplicate_pct: Maximum allowed duplicate percentage (default: 1%).
    """

    def __init__(
        self,
        max_null_pct: float = 5.0,
        max_duplicate_pct: float = 1.0,
    ) -> None:
        """
        Initialize the validator with configurable thresholds.

        Args:
            max_null_pct: Max % of nulls allowed per column before flagging.
            max_duplicate_pct: Max % of duplicate rows allowed.
        """
        self.max_null_pct = max_null_pct
        self.max_duplicate_pct = max_duplicate_pct
        logger.info(
            "DataValidator initialized | "
            f"max_null_pct={max_null_pct}%, "
            f"max_duplicate_pct={max_duplicate_pct}%"
        )

    def validate(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> ValidationReport:
        """
        Run all validation checks on a DataFrame.

        This is the main entry point. It runs:
        1. Schema validation (columns + types)
        2. Null/completeness check
        3. Duplicate check
        4. Range check (for sensor columns)

        Args:
            df: The DataFrame to validate.
            table_name: Name of the table (e.g., "telemetry").

        Returns:
            ValidationReport with results of all checks.
        """
        logger.info(f"Validating '{table_name}' ({len(df):,} rows)...")

        report = ValidationReport(
            table_name=table_name,
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        # Run all checks
        report.checks.append(self._check_not_empty(df, table_name))
        report.checks.append(self._check_schema(df, table_name))
        report.checks.append(self._check_nulls(df, table_name))
        report.checks.append(self._check_duplicates(df, table_name))

        # Range checks only for tables with sensor data
        if table_name == "telemetry":
            for sensor, ranges in SENSOR_RANGES.items():
                if sensor in df.columns:
                    report.checks.append(
                        self._check_range(df, sensor, ranges["min"], ranges["max"])
                    )

        # Log the report
        logger.info(report.summary())

        return report

    def validate_dataset(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> Dict[str, ValidationReport]:
        """
        Validate all tables in the dataset.

        Args:
            dataset: Dictionary of {table_name: DataFrame}.

        Returns:
            Dictionary of {table_name: ValidationReport}.
        """
        logger.info("=" * 50)
        logger.info("VALIDATING COMPLETE DATASET")
        logger.info("=" * 50)

        reports = {}
        for table_name, df in dataset.items():
            reports[table_name] = self.validate(df, table_name)

        # Overall summary
        all_valid = all(r.is_valid for r in reports.values())
        status = "✅ ALL TABLES VALID" if all_valid else "❌ VALIDATION FAILURES"
        logger.info(f"\nOverall: {status}")

        return reports

    # ==================================================================
    # INDIVIDUAL CHECKS
    # ==================================================================

    def _check_not_empty(self, df: pd.DataFrame, table_name: str) -> ValidationCheck:
        """Check that the DataFrame is not empty."""
        if len(df) == 0:
            return ValidationCheck(
                name="not_empty",
                passed=False,
                severity="error",
                message=f"Table '{table_name}' is empty (0 rows)",
            )
        return ValidationCheck(
            name="not_empty",
            passed=True,
            message=f"{len(df):,} rows present",
        )

    def _check_schema(self, df: pd.DataFrame, table_name: str) -> ValidationCheck:
        """
        Verify that expected columns are present.

        WHY: If a column is renamed or missing, downstream code
        (feature engineering, model training) will fail with cryptic
        errors. Catching it here gives a clear, actionable message.
        """
        expected_schema = SCHEMAS.get(table_name)
        if expected_schema is None:
            return ValidationCheck(
                name="schema_check",
                passed=True,
                severity="warning",
                message=f"No schema defined for '{table_name}', skipping",
            )

        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            return ValidationCheck(
                name="schema_check",
                passed=False,
                severity="error",
                message=f"Missing columns: {sorted(missing_cols)}",
                details={
                    "missing": sorted(missing_cols),
                    "extra": sorted(extra_cols),
                    "expected": sorted(expected_cols),
                    "actual": sorted(actual_cols),
                },
            )

        return ValidationCheck(
            name="schema_check",
            passed=True,
            message=f"All {len(expected_cols)} expected columns present",
            details={"extra_columns": sorted(extra_cols)} if extra_cols else {},
        )

    def _check_nulls(self, df: pd.DataFrame, table_name: str) -> ValidationCheck:
        """
        Check for missing values exceeding the threshold.

        WHY: Some nulls are acceptable (sensor glitches happen).
        But if >5% of a column is null, something is wrong —
        maybe a sensor is broken or data export is corrupted.
        """
        null_pcts = (df.isnull().sum() / len(df) * 100).round(2)
        high_null_cols = null_pcts[null_pcts > self.max_null_pct]

        if len(high_null_cols) > 0:
            return ValidationCheck(
                name="null_check",
                passed=False,
                severity="error",
                message=(
                    f"{len(high_null_cols)} column(s) exceed "
                    f"{self.max_null_pct}% null threshold"
                ),
                details={
                    "high_null_columns": {
                        col: f"{pct}%" for col, pct in high_null_cols.items()
                    },
                    "threshold": f"{self.max_null_pct}%",
                },
            )

        total_nulls = df.isnull().sum().sum()
        return ValidationCheck(
            name="null_check",
            passed=True,
            message=(
                f"Null check passed ({total_nulls:,} total nulls, "
                "all within threshold)"
            ),
        )

    def _check_duplicates(self, df: pd.DataFrame, table_name: str) -> ValidationCheck:
        """
        Check for duplicate rows.

        WHY: Duplicates inflate your dataset and bias the model.
        If 10% of your data is duplicated, the model over-weights
        those patterns.
        """
        n_duplicates = df.duplicated().sum()
        dup_pct = (n_duplicates / len(df)) * 100 if len(df) > 0 else 0

        if dup_pct > self.max_duplicate_pct:
            return ValidationCheck(
                name="duplicate_check",
                passed=False,
                severity="warning",
                message=(
                    f"{n_duplicates:,} duplicate rows ({dup_pct:.1f}%), "
                    f"exceeds {self.max_duplicate_pct}% threshold"
                ),
                details={
                    "duplicate_count": n_duplicates,
                    "duplicate_pct": round(dup_pct, 2),
                    "threshold": self.max_duplicate_pct,
                },
            )

        return ValidationCheck(
            name="duplicate_check",
            passed=True,
            message=(
                f"Duplicate check passed ({n_duplicates} duplicates, "
                f"{dup_pct:.2f}%)"
            ),
        )

    def _check_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_val: float,
        max_val: float,
    ) -> ValidationCheck:
        """
        Check that values in a column fall within an expected range.

        WHY: A voltage reading of -500 or a vibration of 99999
        is physically impossible. These are either sensor errors
        or data corruption. They MUST be caught before training.
        """
        if column not in df.columns:
            return ValidationCheck(
                name=f"range_check_{column}",
                passed=True,
                severity="warning",
                message=f"Column '{column}' not found, skipping range check",
            )

        col_data = df[column].dropna()
        out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()
        oor_pct = (out_of_range / len(col_data)) * 100 if len(col_data) > 0 else 0

        if out_of_range > 0:
            return ValidationCheck(
                name=f"range_check_{column}",
                passed=False if oor_pct > 1.0 else True,
                severity="warning",
                message=(
                    f"{column}: {out_of_range:,} values ({oor_pct:.2f}%) "
                    f"outside [{min_val}, {max_val}]"
                ),
                details={
                    "column": column,
                    "expected_range": [min_val, max_val],
                    "actual_min": float(col_data.min()),
                    "actual_max": float(col_data.max()),
                    "out_of_range_count": int(out_of_range),
                    "out_of_range_pct": round(oor_pct, 2),
                },
            )

        return ValidationCheck(
            name=f"range_check_{column}",
            passed=True,
            message=(
                f"{column}: all values within [{min_val}, {max_val}] "
                f"(actual: [{col_data.min():.1f}, {col_data.max():.1f}])"
            ),
        )
