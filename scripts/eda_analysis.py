"""
scripts/eda_analysis.py — Exploratory Data Analysis
============================================================

WHY THIS FILE EXISTS:
    EDA is the MOST CRITICAL step before training any ML model.
    Skipping EDA is like performing surgery without an X-ray.

    This script produces a comprehensive analysis of the predictive
    maintenance dataset, answering:

    1. SHAPE — How large is each table?
    2. TYPES — What are the data types?
    3. MISSING — Where are the gaps in the data?
    4. DISTRIBUTIONS — What do sensor values look like?
    5. CORRELATIONS — Which sensors move together?
    6. TARGET — How imbalanced is the failure class?
    7. TIME PATTERNS — Are there temporal trends?
    8. MACHINE PATTERNS — Do some machines fail more?

USAGE:
    python scripts/eda_analysis.py
    python scripts/eda_analysis.py --data-dir data/sample  # Use sample data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_table_shapes(dataset: dict) -> None:
    """Print the shape of each table in the dataset."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("1. DATASET OVERVIEW")
    logger.info("=" * 60)

    for name, df in dataset.items():
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(
            f"  {name:15s}: {len(df):>10,} rows × {len(df.columns):>2} cols "
            f"| {memory_mb:.1f} MB"
        )

    total = sum(len(df) for df in dataset.values())
    logger.info(f"  {'TOTAL':15s}: {total:>10,} rows")


def analyze_data_types(dataset: dict) -> None:
    """Show data types for each table."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("2. DATA TYPES")
    logger.info("=" * 60)

    for name, df in dataset.items():
        logger.info(f"\n  {name}:")
        for col in df.columns:
            logger.info(f"    {col:20s} → {df[col].dtype}")


def analyze_missing_values(dataset: dict) -> None:
    """Analyze missing values across all tables."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("3. MISSING VALUES ANALYSIS")
    logger.info("=" * 60)

    for name, df in dataset.items():
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls == 0:
            logger.info(f"  {name}: ✅ No missing values")
        else:
            logger.info(f"  {name}: ⚠ {total_nulls:,} missing values")
            for col, count in null_counts.items():
                if count > 0:
                    pct = count / len(df) * 100
                    logger.info(f"    {col}: {count:,} ({pct:.1f}%)")


def analyze_sensor_distributions(telemetry: pd.DataFrame) -> None:
    """
    Analyze the distribution of each sensor column.

    WHY: Understanding distributions helps you:
    - Choose the right normalization method
    - Detect anomalies
    - Identify if sensors need different preprocessing
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("4. SENSOR VALUE DISTRIBUTIONS")
    logger.info("=" * 60)

    sensor_cols = ["voltage", "rotation", "pressure", "vibration"]

    for col in sensor_cols:
        if col not in telemetry.columns:
            continue

        stats = telemetry[col].describe()
        skew = telemetry[col].skew()
        kurtosis = telemetry[col].kurtosis()

        logger.info(f"\n  {col.upper()}:")
        logger.info(f"    Count:    {stats['count']:>12,.0f}")
        logger.info(f"    Mean:     {stats['mean']:>12.2f}")
        logger.info(f"    Std:      {stats['std']:>12.2f}")
        logger.info(f"    Min:      {stats['min']:>12.2f}")
        logger.info(f"    25%:      {stats['25%']:>12.2f}")
        logger.info(f"    50%:      {stats['50%']:>12.2f}")
        logger.info(f"    75%:      {stats['75%']:>12.2f}")
        logger.info(f"    Max:      {stats['max']:>12.2f}")
        logger.info(f"    Skewness: {skew:>12.4f}")
        logger.info(f"    Kurtosis: {kurtosis:>12.4f}")

        # Interpret skewness
        if abs(skew) < 0.5:
            logger.info("    → Distribution: approximately symmetric ✅")
        elif skew > 0:
            logger.info("    → Distribution: right-skewed (long right tail)")
        else:
            logger.info("    → Distribution: left-skewed (long left tail)")


def analyze_correlations(telemetry: pd.DataFrame) -> None:
    """
    Compute correlations between sensor features.

    WHY: If two sensors are highly correlated (>0.9), one is
    redundant. Removing it reduces model complexity without
    losing information (dimensionality reduction).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("5. SENSOR CORRELATIONS")
    logger.info("=" * 60)

    sensor_cols = ["voltage", "rotation", "pressure", "vibration"]
    available = [c for c in sensor_cols if c in telemetry.columns]

    if len(available) < 2:
        logger.info("  Not enough sensor columns for correlation analysis")
        return

    corr_matrix = telemetry[available].corr()

    logger.info("\n  Correlation Matrix:")
    logger.info(f"  {'':15s} " + " ".join(f"{c:>12s}" for c in available))

    for row in available:
        values = " ".join(f"{corr_matrix.loc[row, col]:>12.4f}" for col in available)
        logger.info(f"  {row:15s} {values}")

    # Flag high correlations
    logger.info("\n  Notable correlations:")
    flagged = False
    for i, col1 in enumerate(available):
        for col2 in available[i + 1 :]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > 0.7:
                logger.info(f"    ⚠ {col1} ↔ {col2}: {corr:.4f} (high)")
                flagged = True

    if not flagged:
        logger.info("    ✅ No highly correlated sensor pairs found")


def analyze_target_distribution(
    failures: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> None:
    """
    Analyze the failure/target distribution.

    WHY: This reveals CLASS IMBALANCE — the biggest challenge in
    predictive maintenance. If failures are 0.1% of the data,
    the model will learn to always predict "no failure."
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("6. TARGET DISTRIBUTION (Class Imbalance Analysis)")
    logger.info("=" * 60)

    n_failures = len(failures)
    n_telemetry = len(telemetry)
    failure_rate = (n_failures / n_telemetry) * 100 if n_telemetry > 0 else 0

    logger.info(f"  Total telemetry readings: {n_telemetry:>10,}")
    logger.info(f"  Total failure events:     {n_failures:>10,}")
    logger.info(f"  Failure rate:             {failure_rate:>10.3f}%")
    logger.info(f"  Imbalance ratio:          1:{n_telemetry // max(n_failures, 1)}")

    if failure_rate < 1.0:
        logger.info("")
        logger.info("  ⚠ SEVERE CLASS IMBALANCE DETECTED")
        logger.info("  Recommendations:")
        logger.info("    1. Use class weights in loss function")
        logger.info("    2. Consider SMOTE oversampling")
        logger.info("    3. Evaluate with F1-score and AUC-ROC, NOT accuracy")
        logger.info("    4. Use precision-recall curve (not just ROC)")

    # Failure mode breakdown
    if "failure" in failures.columns and not failures.empty:
        logger.info("\n  Failure mode breakdown:")
        mode_counts = failures["failure"].value_counts()
        for mode, count in mode_counts.items():
            pct = count / n_failures * 100
            logger.info(f"    {mode}: {count:>5} ({pct:.1f}%)")


def analyze_machine_patterns(
    machines: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Analyze which machines fail most and age-based patterns."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("7. MACHINE FAILURE PATTERNS")
    logger.info("=" * 60)

    if failures.empty:
        logger.info("  No failures to analyze")
        return

    # Failures per machine
    failures_per_machine = failures.groupby("machine_id").size()
    machines_with_failures = len(failures_per_machine)
    total_machines = len(machines)

    logger.info(f"  Machines with failures: {machines_with_failures}/{total_machines}")
    logger.info("  Failures per failed machine:")
    logger.info(f"    Mean:   {failures_per_machine.mean():.1f}")
    logger.info(f"    Median: {failures_per_machine.median():.1f}")
    logger.info(f"    Max:    {failures_per_machine.max()}")

    # Age vs failures
    if "age" in machines.columns:
        merged = machines.merge(
            failures_per_machine.rename("n_failures"),
            left_on="machine_id",
            right_index=True,
            how="left",
        )
        merged["n_failures"] = merged["n_failures"].fillna(0)

        logger.info("\n  Failures by age group:")
        age_groups = pd.cut(
            merged["age"],
            bins=[0, 5, 10, 15, 20],
            labels=["0-5 yrs", "6-10 yrs", "11-15 yrs", "16-20 yrs"],
        )
        age_failures = merged.groupby(age_groups, observed=True)["n_failures"].agg(
            ["mean", "sum", "count"]
        )
        for age_group, row in age_failures.iterrows():
            logger.info(
                f"    {age_group}: avg {row['mean']:.1f} failures "
                f"({int(row['sum'])} total from {int(row['count'])} machines)"
            )

    # Model vs failures
    if "model" in machines.columns:
        merged = machines.merge(
            failures_per_machine.rename("n_failures"),
            left_on="machine_id",
            right_index=True,
            how="left",
        )
        merged["n_failures"] = merged["n_failures"].fillna(0)

        logger.info("\n  Failures by machine model:")
        model_failures = merged.groupby("model")["n_failures"].agg(
            ["mean", "sum", "count"]
        )
        for model, row in model_failures.iterrows():
            logger.info(
                f"    {model}: avg {row['mean']:.1f} failures "
                f"({int(row['sum'])} total from {int(row['count'])} machines)"
            )


def analyze_temporal_patterns(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Analyze time-based patterns in the data."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("8. TEMPORAL PATTERNS")
    logger.info("=" * 60)

    if "datetime" in telemetry.columns:
        logger.info(
            "  Telemetry date range: "
            f"{telemetry['datetime'].min()} to "
            f"{telemetry['datetime'].max()}"
        )
        logger.info(
            "  Duration: "
            f"{(telemetry['datetime'].max() - telemetry['datetime'].min()).days} days"
        )

    if "datetime" in failures.columns and not failures.empty:
        logger.info(
            "  Failures date range:  "
            f"{failures['datetime'].min()} to "
            f"{failures['datetime'].max()}"
        )

        # Failures by month
        failures_by_month = failures.groupby(
            failures["datetime"].dt.to_period("M")
        ).size()
        logger.info("\n  Failures by month:")
        for month, count in failures_by_month.items():
            logger.info(f"    {month}: {count} failures")


def main():
    """Run the complete EDA analysis."""
    parser = argparse.ArgumentParser(description="Run EDA on the dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the CSV files (default: data/raw/)",
    )
    args = parser.parse_args()

    logger.info("🔍 EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    # --- Load data ---
    ingestion = DataIngestion()
    data_dir = args.data_dir or ingestion.data_dir
    dataset = ingestion.load_dataset(data_dir=data_dir)

    if not dataset:
        logger.error("No data loaded! Run 'python scripts/generate_data.py' first.")
        sys.exit(1)

    # --- Validate data ---
    validator = DataValidator()
    validator.validate_dataset(dataset)

    # --- Run all analyses ---
    analyze_table_shapes(dataset)
    analyze_data_types(dataset)
    analyze_missing_values(dataset)

    if "telemetry" in dataset:
        analyze_sensor_distributions(dataset["telemetry"])
        analyze_correlations(dataset["telemetry"])

    if "failures" in dataset and "telemetry" in dataset:
        analyze_target_distribution(dataset["failures"], dataset["telemetry"])

    if "machines" in dataset and "failures" in dataset:
        analyze_machine_patterns(dataset["machines"], dataset["failures"])

    if "telemetry" in dataset and "failures" in dataset:
        analyze_temporal_patterns(dataset["telemetry"], dataset["failures"])

    # --- Final summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("EDA COMPLETE")
    logger.info("=" * 60)
    logger.info("Key findings to carry forward:")
    logger.info("  1. Check class imbalance ratio above")
    logger.info("  2. Check sensor correlation matrix")
    logger.info("  3. Check age vs failure patterns")
    logger.info("  4. Review temporal patterns for seasonality")
    logger.info("")
    logger.info("Next step: Feature Engineering (Day 3)")


if __name__ == "__main__":
    main()
