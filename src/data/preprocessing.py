"""
src/data/preprocessing.py — Feature Engineering & Data Preprocessing
============================================================

WHY THIS FILE EXISTS:
    Raw sensor readings are just numbers. An LSTM can't learn from
    "voltage = 172.3" alone. It needs engineered features that
    capture PATTERNS:

    - "Voltage has dropped 15V in the last 12 hours" (rolling stats)
    - "Vibration variance doubled since yesterday" (lag features)
    - "This machine had 4 errors in the past 24 hours" (aggregates)
    - "Last maintenance was 95 days ago" (time-since-last)

    This module transforms the 5 raw tables into a single,
    feature-rich, normalized, time-windowed dataset ready for LSTM.

PIPELINE:
    Raw tables → Merge → Feature Engineering → Label Creation
    → Temporal Split → Normalization → LSTM Windowing → Save

DESIGN PATTERN:
    - Pipeline Pattern: Each transformation step is a separate method
    - Immutability: Original DataFrames are never modified
    - Configurability: All hyperparameters (window sizes, horizon)
      are configurable via __init__
    - Separation: Feature engineering is decoupled from the model

USAGE:
    from src.data.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(
        prediction_horizon=24,
        sequence_length=24,
    )
    result = preprocessor.run_pipeline(dataset)

    X_train, y_train = result["X_train"], result["y_train"]
    X_test, y_test = result["X_test"], result["y_test"]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Sensor columns in the telemetry data
SENSOR_COLUMNS = ["voltage", "rotation", "pressure", "vibration"]

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [3, 12, 24]

# Lag periods (in hours)
LAG_PERIODS = [1, 6, 24]


class DataPreprocessor:
    """
    Production-ready feature engineering and preprocessing pipeline.

    Transforms raw predictive maintenance data into LSTM-ready sequences:
    1. Merges all 5 tables into a single DataFrame
    2. Engineers rolling, lag, and time-since-last features
    3. Creates binary failure labels
    4. Splits temporally (no data leakage)
    5. Normalizes features (fit on train only)
    6. Creates sliding window sequences for LSTM

    Attributes:
        prediction_horizon: Hours ahead to predict failure (default: 24).
        sequence_length: Number of timesteps in each LSTM input (default: 24).
        test_ratio: Fraction of data for testing (default: 0.2).
        sensor_columns: List of sensor column names.
        scaler: StandardScaler instance (fitted on training data).
    """

    def __init__(
        self,
        prediction_horizon: int = 24,
        sequence_length: int = 24,
        test_ratio: float = 0.2,
        sensor_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the preprocessor with configurable hyperparameters.

        Args:
            prediction_horizon: How many hours ahead to predict failure.
                24 = "will this machine fail within the next 24 hours?"
            sequence_length: Number of timesteps per LSTM input sequence.
                24 = "use the last 24 hours of features as input."
            test_ratio: Fraction of data reserved for testing (temporal).
            sensor_columns: Override default sensor column names.
        """
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.test_ratio = test_ratio
        self.sensor_columns = sensor_columns or SENSOR_COLUMNS
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []

        logger.info(
            "DataPreprocessor initialized | "
            f"horizon={prediction_horizon}h, "
            f"seq_len={sequence_length}, "
            f"test_ratio={test_ratio}"
        )

    # ==================================================================
    # STEP 1: MERGE ALL TABLES
    # ==================================================================

    def merge_tables(
        self,
        dataset: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Merge all 5 tables into a single feature-rich DataFrame.

        WHY: The LSTM needs a single DataFrame where each row is one
        machine at one point in time, with ALL relevant information.

        Steps:
        1. Start with telemetry (the time backbone)
        2. Merge machine metadata (age, model)
        3. Aggregate errors into counts per time window
        4. Compute days since last maintenance per component
        5. Merge failure labels

        Args:
            dataset: Dict of {table_name: DataFrame} from DataIngestion.

        Returns:
            Merged DataFrame sorted by (machine_id, datetime).
        """
        logger.info("Step 1: Merging tables...")

        telemetry = dataset["telemetry"].copy()
        machines = dataset.get("machines")
        errors = dataset.get("errors")
        maintenance = dataset.get("maintenance")

        # Ensure datetime is parsed
        telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
        telemetry = telemetry.sort_values(["machine_id", "datetime"]).reset_index(
            drop=True
        )

        # --- Merge machine metadata ---
        if machines is not None and not machines.empty:
            telemetry = telemetry.merge(
                machines[["machine_id", "age", "model"]],
                on="machine_id",
                how="left",
            )
            # One-hot encode machine model
            if "model" in telemetry.columns:
                model_dummies = pd.get_dummies(
                    telemetry["model"], prefix="model", dtype=int
                )
                telemetry = pd.concat([telemetry, model_dummies], axis=1)
                telemetry = telemetry.drop(columns=["model"])
            logger.info("  Merged machines: added age + model columns")

        # --- Aggregate errors per machine per 24h window ---
        if errors is not None and not errors.empty:
            telemetry = self._merge_error_counts(telemetry, errors)
            logger.info("  Merged errors: added error count features")

        # --- Compute days since last maintenance ---
        if maintenance is not None and not maintenance.empty:
            telemetry = self._merge_maintenance_features(telemetry, maintenance)
            logger.info("  Merged maintenance: added time-since features")

        logger.info(
            f"  Merged result: {len(telemetry):,} rows × "
            f"{len(telemetry.columns)} cols"
        )
        return telemetry

    def _merge_error_counts(
        self,
        telemetry: pd.DataFrame,
        errors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Count errors per machine in rolling 24h windows.

        Creates features:
        - errors_last_24h: Total error count in past 24 hours
        - error1_last_24h, error2_last_24h, ...: Count per error type
        """
        errors = errors.copy()
        errors["datetime"] = pd.to_datetime(errors["datetime"])

        # Total errors per machine per 24h
        # Assign each error to the nearest hour for alignment
        errors["hour"] = errors["datetime"].dt.floor("h")

        # Count errors per machine per hour
        error_counts = (
            errors.groupby(["machine_id", "hour"])
            .size()
            .reset_index(name="error_count")
        )
        error_counts = error_counts.rename(columns={"hour": "datetime"})

        # Merge into telemetry
        telemetry = telemetry.merge(
            error_counts, on=["machine_id", "datetime"], how="left"
        )
        telemetry["error_count"] = telemetry["error_count"].fillna(0).astype(int)

        # Rolling sum of errors over 24 hours per machine
        telemetry = telemetry.sort_values(["machine_id", "datetime"])
        telemetry["errors_last_24h"] = telemetry.groupby("machine_id")[
            "error_count"
        ].transform(lambda x: x.rolling(24, min_periods=1).sum())

        return telemetry

    def _merge_maintenance_features(
        self,
        telemetry: pd.DataFrame,
        maintenance: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute hours since last maintenance for each component.

        Creates features:
        - hours_since_maint_comp1, hours_since_maint_comp2, ...

        WHY: A machine that was just maintained is less likely to fail
        from that component. A machine 90+ days past maintenance is overdue.
        """
        maintenance = maintenance.copy()
        maintenance["datetime"] = pd.to_datetime(maintenance["datetime"])

        components = maintenance["comp"].unique()

        for comp in components:
            comp_maint = maintenance[maintenance["comp"] == comp][
                ["machine_id", "datetime"]
            ].copy()
            comp_maint = comp_maint.rename(columns={"datetime": f"last_maint_{comp}"})

            # For each machine, get all maintenance times for this component
            # Then merge_asof to find the most recent one before each timestamp
            comp_maint = comp_maint.sort_values(f"last_maint_{comp}")
            telemetry = telemetry.sort_values(["machine_id", "datetime"])

            # Use merge_asof per machine group
            merged_parts = []
            for mid in telemetry["machine_id"].unique():
                tel_machine = telemetry[telemetry["machine_id"] == mid].copy()
                maint_machine = comp_maint[comp_maint["machine_id"] == mid].copy()

                if maint_machine.empty:
                    tel_machine[f"hours_since_maint_{comp}"] = 9999
                else:
                    tel_machine = pd.merge_asof(
                        tel_machine.sort_values("datetime"),
                        maint_machine.sort_values(f"last_maint_{comp}"),
                        left_on="datetime",
                        right_on=f"last_maint_{comp}",
                        by="machine_id",
                        direction="backward",
                    )
                    # Calculate hours since last maintenance
                    tel_machine[f"hours_since_maint_{comp}"] = (
                        (
                            tel_machine["datetime"] - tel_machine[f"last_maint_{comp}"]
                        ).dt.total_seconds()
                        / 3600
                    ).fillna(9999)
                    tel_machine = tel_machine.drop(
                        columns=[f"last_maint_{comp}"], errors="ignore"
                    )

                merged_parts.append(tel_machine)

            telemetry = pd.concat(merged_parts, ignore_index=True)
            telemetry = telemetry.sort_values(["machine_id", "datetime"]).reset_index(
                drop=True
            )

        return telemetry

    # ==================================================================
    # STEP 2: FEATURE ENGINEERING
    # ==================================================================

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features from the merged DataFrame.

        Features created:
        1. Rolling statistics (mean, std) over 3h, 12h, 24h windows
        2. Lag features (value at 1h, 6h, 24h ago)
        3. Rate of change (current - lag) / lag_hours

        Args:
            df: Merged DataFrame from merge_tables().

        Returns:
            DataFrame with additional engineered feature columns.
        """
        logger.info("Step 2: Engineering features...")
        df = df.copy()
        df = df.sort_values(["machine_id", "datetime"]).reset_index(drop=True)

        n_features_before = len(df.columns)

        # --- Rolling statistics ---
        for col in self.sensor_columns:
            if col not in df.columns:
                continue

            grouped = df.groupby("machine_id")[col]

            for window in ROLLING_WINDOWS:
                # Rolling mean — captures the trend direction
                df[f"{col}_rolling_mean_{window}h"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                # Rolling std — captures variability/stability
                df[f"{col}_rolling_std_{window}h"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

        # --- Lag features ---
        for col in self.sensor_columns:
            if col not in df.columns:
                continue

            grouped = df.groupby("machine_id")[col]

            for lag in LAG_PERIODS:
                # Value N hours ago
                df[f"{col}_lag_{lag}h"] = grouped.transform(lambda x: x.shift(lag))
                # Rate of change over N hours
                df[f"{col}_change_{lag}h"] = df[col] - df[f"{col}_lag_{lag}h"]

        # Fill NaN from rolling/lag operations
        # WHY: The first N rows per machine have no history for rolling.
        # Forward fill uses the first available value as the initial state.
        df = df.sort_values(["machine_id", "datetime"])
        feature_cols = [c for c in df.columns if c not in ["datetime", "machine_id"]]
        df[feature_cols] = df.groupby("machine_id")[feature_cols].transform(
            lambda x: x.ffill().bfill()
        )

        n_features_after = len(df.columns)
        logger.info(
            f"  Engineered {n_features_after - n_features_before} new features "
            f"({n_features_before} → {n_features_after} total columns)"
        )

        return df

    # ==================================================================
    # STEP 3: LABEL CREATION
    # ==================================================================

    def create_labels(
        self,
        df: pd.DataFrame,
        failures: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create binary failure labels for supervised learning.

        Label definition:
            label = 1 if this machine will fail within `prediction_horizon` hours
            label = 0 otherwise

        WHY this approach:
            We're solving a CLASSIFICATION problem: "Should we schedule
            maintenance for this machine NOW?" The prediction_horizon
            controls how far ahead we predict.

        Args:
            df: Feature-engineered DataFrame.
            failures: Failures DataFrame with (datetime, machine_id, failure).

        Returns:
            DataFrame with added 'label' column (0 or 1).
        """
        logger.info(
            "Step 3: Creating labels " f"(horizon={self.prediction_horizon}h)..."
        )
        df = df.copy()
        df["label"] = 0

        if failures is None or failures.empty:
            logger.warning("  No failures found — all labels will be 0")
            return df

        failures = failures.copy()
        failures["datetime"] = pd.to_datetime(failures["datetime"])

        # For each failure, mark the preceding `prediction_horizon` hours
        for _, failure in failures.iterrows():
            failure_time = failure["datetime"]
            machine_id = failure["machine_id"]

            # Time window: [failure_time - horizon, failure_time]
            window_start = failure_time - pd.Timedelta(hours=self.prediction_horizon)

            mask = (
                (df["machine_id"] == machine_id)
                & (df["datetime"] >= window_start)
                & (df["datetime"] <= failure_time)
            )
            df.loc[mask, "label"] = 1

        # Log label distribution
        n_positive = df["label"].sum()
        n_total = len(df)
        pct_positive = (n_positive / n_total) * 100 if n_total > 0 else 0

        logger.info("  Label distribution:")
        logger.info(f"    Positive (will fail): {n_positive:,} ({pct_positive:.2f}%)")
        logger.info(
            f"    Negative (normal):    {n_total - n_positive:,} "
            f"({100 - pct_positive:.2f}%)"
        )
        logger.info(f"    Imbalance ratio:      1:{n_total // max(n_positive, 1)}")

        return df

    # ==================================================================
    # STEP 4: TEMPORAL TRAIN/TEST SPLIT
    # ==================================================================

    def temporal_split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally — NO random shuffling.

        WHY temporal:
            In production, the model only has past data. If we randomly
            split, the model trains on March data to predict February —
            it literally sees the future. This is DATA LEAKAGE and makes
            metrics look great but production performance terrible.

        The split point is chosen so that:
            - Train = first (1 - test_ratio) of the data by time
            - Test = last test_ratio of the data by time

        Args:
            df: Labeled DataFrame sorted by datetime.

        Returns:
            Tuple of (train_df, test_df).
        """
        logger.info("Step 4: Temporal train/test split...")

        df = df.sort_values("datetime").reset_index(drop=True)

        # Find the split timestamp
        split_idx = int(len(df) * (1 - self.test_ratio))
        split_time = df.iloc[split_idx]["datetime"]

        train_df = df[df["datetime"] < split_time].copy()
        test_df = df[df["datetime"] >= split_time].copy()

        logger.info(f"  Split point: {split_time}")
        logger.info(
            f"  Train: {len(train_df):,} rows "
            f"({train_df['datetime'].min()} → {train_df['datetime'].max()})"
        )
        logger.info(
            f"  Test:  {len(test_df):,} rows "
            f"({test_df['datetime'].min()} → {test_df['datetime'].max()})"
        )
        logger.info(
            f"  Train labels: {train_df['label'].sum():,} positive "
            f"/ {len(train_df):,} total"
        )
        logger.info(
            f"  Test labels:  {test_df['label'].sum():,} positive "
            f"/ {len(test_df):,} total"
        )

        return train_df, test_df

    # ==================================================================
    # STEP 5: NORMALIZATION
    # ==================================================================

    def normalize(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Normalize features using StandardScaler.

        CRITICAL RULE: Fit scaler on TRAINING data only, then
        transform both train and test. Fitting on all data leaks
        test set statistics into training.

        Args:
            train_df: Training DataFrame.
            test_df: Testing DataFrame.
            exclude_columns: Columns to exclude from scaling
                (e.g., datetime, machine_id, label).

        Returns:
            Tuple of (scaled_train_df, scaled_test_df, feature_columns).
        """
        logger.info("Step 5: Normalizing features...")

        if exclude_columns is None:
            exclude_columns = ["datetime", "machine_id", "label"]

        # Identify feature columns (numeric, not excluded)
        feature_cols = [
            col
            for col in train_df.columns
            if col not in exclude_columns
            and pd.api.types.is_numeric_dtype(train_df[col])
        ]
        self.feature_columns = feature_cols

        logger.info(f"  Scaling {len(feature_cols)} numeric features")

        # Fit scaler on TRAINING data only
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[feature_cols])

        # Transform both sets
        train_scaled = train_df.copy()
        test_scaled = test_df.copy()

        train_scaled[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_scaled[feature_cols] = self.scaler.transform(test_df[feature_cols])

        # Verify no NaN introduced by scaling
        train_nans = train_scaled[feature_cols].isna().sum().sum()
        test_nans = test_scaled[feature_cols].isna().sum().sum()

        if train_nans > 0 or test_nans > 0:
            logger.warning(
                "  ⚠ NaN values after scaling: " f"train={train_nans}, test={test_nans}"
            )
            # Fill remaining NaN with 0 (scaled mean)
            train_scaled[feature_cols] = train_scaled[feature_cols].fillna(0)
            test_scaled[feature_cols] = test_scaled[feature_cols].fillna(0)

        logger.info("  Scaler fitted on training data (mean=0, std=1)")

        return train_scaled, test_scaled, feature_cols

    # ==================================================================
    # STEP 6: LSTM SEQUENCE WINDOWING
    # ==================================================================

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for the LSTM.

        Converts a flat DataFrame into 3D arrays:
            X shape: (n_samples, sequence_length, n_features)
            y shape: (n_samples,)

        Each sample is a window of `sequence_length` timesteps,
        and the label is from the LAST timestep in the window.

        WHY sliding windows:
            LSTMs need sequential input. A window of 24 hours gives
            the LSTM enough temporal context to detect degradation
            patterns that precede failure.

        IMPORTANT: Windows are created PER MACHINE. We never mix
        data from different machines in the same sequence.

        Args:
            df: Scaled DataFrame with features and labels.
            feature_cols: List of feature column names.

        Returns:
            Tuple of (X, y) as NumPy arrays.
        """
        logger.info("Step 6: Creating sequences " f"(window={self.sequence_length})...")

        all_X = []
        all_y = []

        for machine_id in sorted(df["machine_id"].unique()):
            machine_data = df[df["machine_id"] == machine_id].sort_values("datetime")

            features = machine_data[feature_cols].values
            labels = machine_data["label"].values

            # Create sliding windows
            for i in range(self.sequence_length, len(features)):
                # X = last `sequence_length` timesteps of features
                X_window = features[i - self.sequence_length : i]
                # y = label at the current timestep
                y_label = labels[i]

                all_X.append(X_window)
                all_y.append(y_label)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.float32)

        logger.info(f"  X shape: {X.shape} (samples, timesteps, features)")
        logger.info(f"  y shape: {y.shape}")
        logger.info(
            "  y distribution: "
            f"{int(y.sum())} positive / {len(y)} total "
            f"({y.mean() * 100:.2f}%)"
        )

        return X, y

    # ==================================================================
    # FULL PIPELINE
    # ==================================================================

    def run_pipeline(
        self,
        dataset: Dict[str, pd.DataFrame],
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.

        Steps:
        1. Merge all tables
        2. Engineer features (rolling, lag, time-since-last)
        3. Create binary failure labels
        4. Temporal train/test split
        5. Normalize features (fit on train only)
        6. Create LSTM sliding window sequences
        7. (Optional) Save processed data

        Args:
            dataset: Dict of {table_name: DataFrame} from DataIngestion.
            save_dir: Directory to save processed data (optional).

        Returns:
            Dict with keys: X_train, y_train, X_test, y_test,
            feature_columns, scaler, metadata.
        """
        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE STARTED")
        logger.info("=" * 60)

        # Step 1: Merge
        merged = self.merge_tables(dataset)

        # Step 2: Feature Engineering
        featured = self.engineer_features(merged)

        # Step 3: Labels
        failures = dataset.get("failures", pd.DataFrame())
        labeled = self.create_labels(featured, failures)

        # Step 4: Temporal Split
        train_df, test_df = self.temporal_split(labeled)

        # Step 5: Normalize
        train_scaled, test_scaled, feature_cols = self.normalize(train_df, test_df)

        # Step 6: Create Sequences
        X_train, y_train = self.create_sequences(train_scaled, feature_cols)
        X_test, y_test = self.create_sequences(test_scaled, feature_cols)

        # Save if requested
        if save_dir:
            self._save_artifacts(
                save_dir, X_train, y_train, X_test, y_test, feature_cols
            )

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  y_train: {y_train.shape} ({y_train.sum():.0f} positive)")
        logger.info(f"  X_test:  {X_test.shape}")
        logger.info(f"  y_test:  {y_test.shape} ({y_test.sum():.0f} positive)")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info("=" * 60)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
            "scaler": self.scaler,
            "metadata": {
                "prediction_horizon": self.prediction_horizon,
                "sequence_length": self.sequence_length,
                "n_features": len(feature_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_positive_rate": float(y_train.mean()),
                "test_positive_rate": float(y_test.mean()),
            },
        }

    def _save_artifacts(
        self,
        save_dir: Path,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_cols: List[str],
    ) -> None:
        """Save processed data and scaler to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / "X_train.npy", X_train)
        np.save(save_dir / "y_train.npy", y_train)
        np.save(save_dir / "X_test.npy", X_test)
        np.save(save_dir / "y_test.npy", y_test)

        if self.scaler:
            joblib.dump(self.scaler, save_dir / "scaler.joblib")

        # Save feature column names
        with open(save_dir / "feature_columns.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")

        logger.info(f"  Artifacts saved to: {save_dir}")
