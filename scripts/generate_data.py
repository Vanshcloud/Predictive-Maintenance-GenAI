"""
scripts/generate_data.py — Synthetic Predictive Maintenance Data Generator
============================================================

WHY THIS FILE EXISTS:
    We need realistic sensor data to train and test our ML model.
    Instead of depending on external downloads (which break, require
    accounts, or have licensing issues), we generate our own data
    that mirrors the Microsoft Azure Predictive Maintenance dataset.

HOW IT WORKS:
    Generates 5 interconnected tables:

    1. MACHINES — 100 machines with metadata (model, age)
    2. TELEMETRY — Hourly sensor readings per machine
       (voltage, rotation, pressure, vibration)
    3. ERRORS — Non-failure error events (5 error types)
    4. MAINTENANCE — Scheduled component replacements (4 components)
    5. FAILURES — Actual failure events (4 failure modes)

    The data is REALISTIC because:
    - Sensor values follow normal distributions with realistic ranges
    - Machines approaching failure show gradual degradation
    - Older machines fail more often
    - Error frequency increases before failures
    - Seasonal patterns exist in some sensors

DESIGN DECISIONS:
    - Uses NumPy's random generators with a fixed seed for reproducibility
    - Generates data for 1 year (8,760 hours per machine)
    - 100 machines → ~876,000 telemetry rows (same scale as Azure dataset)
    - Class imbalance is realistic (~0.1% failure rate)

USAGE:
    python scripts/generate_data.py
    python scripts/generate_data.py --machines 50 --days 180
    python scripts/generate_data.py --sample  # Small sample for testing
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add project root to path so we can import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# SENSOR CONFIGURATION
# ===========================================================================
# WHY: Define sensor characteristics in one place. If you add a new sensor
# type, you only change this dictionary — not the generation logic.
# Each sensor has a mean, std (standard deviation), and valid range.
# ===========================================================================

SENSOR_CONFIG = {
    "voltage": {
        "mean": 170.0,  # Volts — typical industrial motor voltage
        "std": 15.0,
        "min": 100.0,
        "max": 250.0,
        "degradation_shift": 25.0,  # How much it shifts near failure
    },
    "rotation": {
        "mean": 450.0,  # RPM — rotations per minute
        "std": 50.0,
        "min": 100.0,
        "max": 800.0,
        "degradation_shift": -80.0,  # RPM drops before failure
    },
    "pressure": {
        "mean": 100.0,  # PSI — pounds per square inch
        "std": 12.0,
        "min": 40.0,
        "max": 180.0,
        "degradation_shift": -20.0,  # Pressure drops (leaks)
    },
    "vibration": {
        "mean": 40.0,  # mm/s — vibration velocity
        "std": 8.0,
        "min": 10.0,
        "max": 100.0,
        "degradation_shift": 20.0,  # Vibration increases (wear)
    },
}

# Machine models with different reliability characteristics
MACHINE_MODELS = ["model1", "model2", "model3", "model4"]

# Error types that can occur (non-failure events)
ERROR_TYPES = ["error1", "error2", "error3", "error4", "error5"]

# Components that can be maintained/replaced
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]

# Failure modes (what finally breaks)
FAILURE_MODES = ["comp1", "comp2", "comp3", "comp4"]


def generate_machines(
    n_machines: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate machine metadata.

    WHY: Each machine has different characteristics that affect
    failure probability. Older machines and certain models are
    more failure-prone — just like real equipment.

    Args:
        n_machines: Number of machines to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with columns: machine_id, model, age
    """
    logger.info(f"Generating metadata for {n_machines} machines...")

    machines = pd.DataFrame(
        {
            "machine_id": range(1, n_machines + 1),
            "model": rng.choice(MACHINE_MODELS, size=n_machines),
            # Age in years: 0-20, with most machines being 5-10 years old
            "age": rng.integers(0, 21, size=n_machines),
        }
    )

    logger.info(
        f"  Age distribution: mean={machines['age'].mean():.1f}, "
        f"min={machines['age'].min()}, max={machines['age'].max()}"
    )
    return machines


def generate_telemetry(
    machines: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
    failure_windows: Dict[int, List[datetime]] = None,
) -> pd.DataFrame:
    """
    Generate hourly sensor telemetry readings for all machines.

    WHY: This is the core training data. Each row represents one
    hourly reading from one machine's sensors. The key insight is
    that sensors show GRADUAL DEGRADATION before failures — they
    don't just jump from normal to broken.

    HOW DEGRADATION WORKS:
    - For 24-48 hours before a failure, sensor values shift:
      - Voltage becomes erratic (±25V from normal)
      - Rotation drops (bearings wearing out)
      - Pressure drops (seals leaking)
      - Vibration increases (components loosening)
    - This gradual pattern is what the LSTM will learn to detect.

    Args:
        machines: Machine metadata DataFrame.
        start_date: When data collection begins.
        n_days: Number of days to generate.
        rng: NumPy random generator.
        failure_windows: Dict mapping machine_id to list of failure datetimes.

    Returns:
        DataFrame with columns: datetime, machine_id, voltage, rotation,
        pressure, vibration
    """
    n_hours = n_days * 24
    n_machines = len(machines)
    total_rows = n_machines * n_hours

    logger.info(
        f"Generating telemetry: {n_machines} machines × "
        f"{n_hours} hours = {total_rows:,} rows..."
    )

    # Pre-allocate arrays for performance
    # WHY: Appending to a list in a loop is O(n²) for DataFrames.
    # Pre-allocating and filling is O(n). For 876K rows, this matters.
    all_datetimes = []
    all_machine_ids = []
    all_voltage = []
    all_rotation = []
    all_pressure = []
    all_vibration = []

    timestamps = [start_date + timedelta(hours=h) for h in range(n_hours)]

    for _, machine in machines.iterrows():
        mid = machine["machine_id"]
        age_factor = machine["age"] / 20.0  # 0.0 to 1.0

        # Base sensor values with slight machine-specific offset
        # WHY: Each machine is slightly different (manufacturing variance)
        machine_offset = rng.normal(0, 2)

        for sensor_name, config in SENSOR_CONFIG.items():
            # Generate base readings
            base_values = rng.normal(
                config["mean"] + machine_offset,
                config["std"] * (1 + 0.1 * age_factor),  # Older = noisier
                size=n_hours,
            )

            # Add slight daily periodicity (temperature affects sensors)
            # WHY: Real sensors show daily patterns (factory heats up during
            # the day, cools at night). This makes the data more realistic.
            hours_of_day = np.array([t.hour for t in timestamps])
            daily_cycle = 3.0 * np.sin(2 * np.pi * hours_of_day / 24.0)
            base_values += daily_cycle

            # Apply degradation near failure windows
            if failure_windows and mid in failure_windows:
                for failure_time in failure_windows[mid]:
                    for h_idx, t in enumerate(timestamps):
                        hours_before_failure = (failure_time - t).total_seconds() / 3600
                        # Gradual degradation in the 48 hours before failure
                        if 0 < hours_before_failure <= 48:
                            # Degradation intensity: 0 at 48h → 1.0 at failure
                            intensity = 1.0 - (hours_before_failure / 48.0)
                            shift = config["degradation_shift"] * intensity
                            # Add noise to degradation (not perfectly smooth)
                            shift += rng.normal(0, abs(shift) * 0.2)
                            base_values[h_idx] += shift

            # Clip to valid range
            base_values = np.clip(base_values, config["min"], config["max"])

            if sensor_name == "voltage":
                all_voltage.extend(base_values)
            elif sensor_name == "rotation":
                all_rotation.extend(base_values)
            elif sensor_name == "pressure":
                all_pressure.extend(base_values)
            elif sensor_name == "vibration":
                all_vibration.extend(base_values)

        all_datetimes.extend(timestamps)
        all_machine_ids.extend([mid] * n_hours)

    telemetry = pd.DataFrame(
        {
            "datetime": all_datetimes,
            "machine_id": all_machine_ids,
            "voltage": np.round(all_voltage, 2),
            "rotation": np.round(all_rotation, 2),
            "pressure": np.round(all_pressure, 2),
            "vibration": np.round(all_vibration, 2),
        }
    )

    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    telemetry = telemetry.sort_values(["machine_id", "datetime"]).reset_index(drop=True)

    logger.info(f"  Generated {len(telemetry):,} telemetry rows")
    return telemetry


def generate_failures(
    machines: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[int, List[datetime]]]:
    """
    Generate failure events.

    WHY: Failures are RARE events (like the real dataset). We generate
    them with realistic frequency:
    - Each machine has a ~2-5% chance of failing per month
    - Older machines fail more often
    - Different failure modes have different frequencies

    Returns:
        Tuple of (failures DataFrame, failure_windows dict for telemetry)
    """
    logger.info("Generating failure events...")

    failure_records = []
    failure_windows: Dict[int, List[datetime]] = {}

    for _, machine in machines.iterrows():
        mid = machine["machine_id"]
        age = machine["age"]

        # Base monthly failure probability: 2% + age bonus
        # WHY: A 20-year-old machine fails ~3x more than a new one
        monthly_failure_prob = 0.02 + (age / 20.0) * 0.04

        n_months = n_days // 30
        for month in range(n_months):
            if rng.random() < monthly_failure_prob:
                # Random day and hour within the month
                day_offset = month * 30 + rng.integers(1, 29)
                hour_offset = rng.integers(0, 24)

                if day_offset >= n_days:
                    continue

                failure_time = start_date + timedelta(
                    days=int(day_offset), hours=int(hour_offset)
                )

                failure_mode = rng.choice(FAILURE_MODES)

                failure_records.append(
                    {
                        "datetime": failure_time,
                        "machine_id": mid,
                        "failure": failure_mode,
                    }
                )

                # Track for telemetry degradation
                if mid not in failure_windows:
                    failure_windows[mid] = []
                failure_windows[mid].append(failure_time)

    failures = pd.DataFrame(failure_records)
    if not failures.empty:
        failures["datetime"] = pd.to_datetime(failures["datetime"])
        failures = failures.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"  Generated {len(failures)} failures across "
        f"{len(failure_windows)} machines"
    )
    return failures, failure_windows


def generate_errors(
    machines: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
    failure_windows: Dict[int, List[datetime]] = None,
) -> pd.DataFrame:
    """
    Generate non-failure error events.

    WHY: Errors are early warning signs. A machine might log
    "error3" (overheating warning) several times before it actually
    fails. The ML model can learn that increased error frequency
    predicts failure.

    Error frequency INCREASES in the days before a failure.
    """
    logger.info("Generating error events...")

    error_records = []

    for _, machine in machines.iterrows():
        mid = machine["machine_id"]

        # Base: ~1 error per machine per week
        n_errors = rng.poisson(n_days / 7)

        for _ in range(n_errors):
            day_offset = rng.integers(0, n_days)
            hour_offset = rng.integers(0, 24)
            error_time = start_date + timedelta(
                days=int(day_offset), hours=int(hour_offset)
            )

            error_records.append(
                {
                    "datetime": error_time,
                    "machine_id": mid,
                    "error_id": rng.choice(ERROR_TYPES),
                }
            )

        # Extra errors near failure windows
        if failure_windows and mid in failure_windows:
            for failure_time in failure_windows[mid]:
                # 3-8 extra errors in the week before failure
                n_extra = rng.integers(3, 9)
                for _ in range(n_extra):
                    hours_before = rng.integers(1, 168)  # Up to 1 week
                    error_time = failure_time - timedelta(hours=int(hours_before))
                    if error_time >= start_date:
                        error_records.append(
                            {
                                "datetime": error_time,
                                "machine_id": mid,
                                "error_id": rng.choice(ERROR_TYPES),
                            }
                        )

    errors = pd.DataFrame(error_records)
    if not errors.empty:
        errors["datetime"] = pd.to_datetime(errors["datetime"])
        errors = errors.sort_values("datetime").reset_index(drop=True)

    logger.info(f"  Generated {len(errors)} error events")
    return errors


def generate_maintenance(
    machines: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate scheduled maintenance records.

    WHY: Maintenance (component replacements) "resets" the degradation
    clock. A machine that just had comp2 replaced is unlikely to fail
    from comp2 anytime soon. This is important context for the model.
    """
    logger.info("Generating maintenance records...")

    maint_records = []

    for _, machine in machines.iterrows():
        mid = machine["machine_id"]

        # Each component gets replaced every 60-120 days on average
        for comp in COMPONENTS:
            interval = rng.integers(60, 121)
            current_day = rng.integers(0, interval)

            while current_day < n_days:
                maint_time = start_date + timedelta(
                    days=int(current_day),
                    hours=int(rng.integers(8, 17)),  # During work hours
                )
                maint_records.append(
                    {
                        "datetime": maint_time,
                        "machine_id": mid,
                        "comp": comp,
                    }
                )
                # Next maintenance with some variance
                current_day += rng.integers(
                    int(interval * 0.8), int(interval * 1.2) + 1
                )

    maintenance = pd.DataFrame(maint_records)
    if not maintenance.empty:
        maintenance["datetime"] = pd.to_datetime(maintenance["datetime"])
        maintenance = maintenance.sort_values("datetime").reset_index(drop=True)

    logger.info(f"  Generated {len(maintenance)} maintenance records")
    return maintenance


def generate_dataset(
    n_machines: int = 100,
    n_days: int = 365,
    seed: int = 42,
    output_dir: Path = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate the complete 5-table predictive maintenance dataset.

    Args:
        n_machines: Number of machines to simulate.
        n_days: Number of days of data to generate.
        seed: Random seed for reproducibility.
        output_dir: Directory to save CSV files.

    Returns:
        Dictionary of DataFrames: {table_name: DataFrame}
    """
    logger.info("=" * 60)
    logger.info("GENERATING PREDICTIVE MAINTENANCE DATASET")
    logger.info(f"  Machines: {n_machines}")
    logger.info(f"  Days: {n_days}")
    logger.info(f"  Seed: {seed}")
    logger.info("=" * 60)

    rng = np.random.default_rng(seed)
    start_date = datetime(2024, 1, 1)

    # Generate in dependency order
    machines = generate_machines(n_machines, rng)
    failures, failure_windows = generate_failures(machines, start_date, n_days, rng)
    errors = generate_errors(machines, start_date, n_days, rng, failure_windows)
    maintenance = generate_maintenance(machines, start_date, n_days, rng)
    telemetry = generate_telemetry(machines, start_date, n_days, rng, failure_windows)

    dataset = {
        "machines": machines,
        "telemetry": telemetry,
        "errors": errors,
        "maintenance": maintenance,
        "failures": failures,
    }

    # Save to CSV
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dataset.items():
            filepath = output_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"  Saved {name}.csv ({len(df):,} rows, {filepath})")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    for name, df in dataset.items():
        logger.info(f"  {name:15s}: {len(df):>10,} rows × {len(df.columns)} cols")

    total_rows = sum(len(df) for df in dataset.values())
    logger.info(f"  {'TOTAL':15s}: {total_rows:>10,} rows")
    logger.info("=" * 60)

    return dataset


def main():
    """Entry point for the data generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic predictive maintenance data"
    )
    parser.add_argument(
        "--machines",
        type=int,
        default=100,
        help="Number of machines (default: 100)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of data (default: 365)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate a small sample dataset (10 machines, 30 days)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/raw/)",
    )

    args = parser.parse_args()

    settings = get_settings()

    if args.sample:
        args.machines = 10
        args.days = 30
        output_dir = PROJECT_ROOT / "data" / "sample"
        logger.info("Generating SAMPLE dataset (10 machines, 30 days)")
    else:
        output_dir = Path(args.output) if args.output else settings.raw_data_path

    generate_dataset(
        n_machines=args.machines,
        n_days=args.days,
        seed=args.seed,
        output_dir=output_dir,
    )

    logger.info(f"\n✅ Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
