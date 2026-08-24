"""
src/prediction/predictor.py — Inference Pipeline
=================================================

WHY THIS FILE EXISTS:
    Training produces a `.keras` file, a fitted scaler, and an ordered list
    of 63 feature names. None of that is useful on its own. Something has to
    take raw sensor readings — the five tables a plant actually has — and
    turn them into "machine 47 will probably fail in the next 24 hours,
    and here is why".

    This module is that boundary. Everything above it (GenAI, API, dashboard)
    talks in machines and probabilities; nothing above it imports TensorFlow
    or touches a tensor.

HOW IT WORKS:
    Raw tables -> DataPreprocessor.merge_tables -> engineer_features
               -> apply_scaler (TRAINING statistics) -> create_sequences
               -> model -> probability -> risk band

    It **reuses DataPreprocessor** rather than reimplementing the feature
    logic. That is the whole defence against training/serving skew (Risk
    R-6): if the feature code lives in two places it will drift, and the
    resulting bug is silent — the model keeps returning plausible numbers
    that are quietly wrong. One implementation, used by both paths.

    Two invariants are asserted at load time rather than trusted:

      1. The feature names in `feature_columns.txt` must match what the
         preprocessor produces, in order. Position matters — feature 17 of
         the tensor must be the same quantity the model was trained on.
      2. The model's input shape must match (sequence_length, n_features).

    Both raise on mismatch. A predictor that refuses to start is vastly
    preferable to one that serves confident nonsense.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.data.preprocessing import DataPreprocessor
from src.utils.exceptions import ModelNotFoundError, PredictionError
from src.utils.logger import get_logger

# TensorFlow must be imported before pandas/scikit-learn pull in Arrow.
# See src/models/__init__.py — the wrong order deadlocks the process.
from src.models import PredictiveMaintenanceModel  # isort: skip

logger = get_logger(__name__)


def _native(value: Any) -> Any:
    """Unwrap a numpy scalar to its Python equivalent; pass anything else through."""
    return value.item() if hasattr(value, "item") else value


class Predictor:
    """Scores raw sensor data with a trained model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        feature_columns_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        sequence_length: int = 24,
    ) -> None:
        """
        Load the model, scaler, and feature contract, and verify they agree.

        Args:
            model_path: Trained `.keras` file. Defaults to settings.
            scaler_path: Fitted StandardScaler. Defaults to settings.
            feature_columns_path: Ordered feature names. Defaults to settings.
            threshold: Alert threshold. Defaults to
                `settings.PREDICTION_THRESHOLD`, which was chosen on the
                validation split — not guessed, and not 0.5.
            sequence_length: Timesteps per window; must match training.

        Raises:
            ModelNotFoundError: if any artifact is missing.
            PredictionError: if the artifacts disagree with each other.
        """
        settings = get_settings()
        self.settings = settings
        self.threshold = (
            threshold if threshold is not None else settings.PREDICTION_THRESHOLD
        )
        self.sequence_length = sequence_length

        model_path = Path(model_path or settings.model_path)
        scaler_path = Path(scaler_path or settings.scaler_path)
        feature_columns_path = Path(
            feature_columns_path or settings.feature_columns_path
        )

        for name, path in (
            ("model", model_path),
            ("scaler", scaler_path),
            ("feature columns", feature_columns_path),
        ):
            if not path.exists():
                raise ModelNotFoundError(
                    f"Cannot start Predictor: {name} not found at {path}. "
                    "Run scripts/run_preprocessing.py and scripts/train_model.py."
                )

        self.model_wrapper = PredictiveMaintenanceModel.load(model_path)
        self.model = self.model_wrapper.model
        self.scaler = joblib.load(scaler_path)
        self.feature_columns: List[str] = [
            line.strip()
            for line in feature_columns_path.read_text().splitlines()
            if line.strip()
        ]

        self._verify_contract()

        logger.info(
            f"Predictor ready — {len(self.feature_columns)} features, "
            f"sequence_length={self.sequence_length}, threshold={self.threshold:.4f}"
        )

    def _verify_contract(self) -> None:
        """
        Check that model, scaler, and feature list describe the same thing.

        WHY: these three artifacts are produced by two different scripts and
        loaded independently. Nothing structurally prevents pairing a model
        with the scaler from a different preprocessing run — and the failure
        would be silent, because every array would still have a valid shape.
        """
        _, seq_len, n_features = self.model.input_shape

        if n_features != len(self.feature_columns):
            raise PredictionError(
                f"Feature count mismatch: model expects {n_features} features "
                f"but feature_columns.txt lists {len(self.feature_columns)}. "
                "The model and the processed data are from different runs."
            )

        scaler_n = getattr(self.scaler, "n_features_in_", None)
        if scaler_n is not None and scaler_n != len(self.feature_columns):
            raise PredictionError(
                f"Scaler mismatch: fitted on {scaler_n} features but "
                f"feature_columns.txt lists {len(self.feature_columns)}."
            )

        if seq_len != self.sequence_length:
            raise PredictionError(
                f"Sequence length mismatch: model expects {seq_len} timesteps "
                f"but Predictor was configured with {self.sequence_length}."
            )

    # ------------------------------------------------------------------
    # Feature reconciliation
    # ------------------------------------------------------------------

    # Some engineered features are DATA-DEPENDENT: their existence depends on
    # which categories happen to appear in the input. One-hot `model_*` columns
    # only exist for models present in the batch, and `hours_since_maint_*`
    # only for components that appear in the maintenance log. Score three
    # machines that are all model1 and the feature matrix is narrower than the
    # 63 columns the model was trained on.
    #
    # This is not an error — it is the normal shape of live data — but it must
    # be reconciled explicitly, with the fill value the training pipeline would
    # have produced for an absent category. Getting the fill wrong is worse
    # than getting it missing: 0 hours since maintenance means "just serviced",
    # the exact opposite of "never serviced".
    _FILL_RULES = (
        # prefix,                 fill,   meaning
        ("hours_since_maint_", 9999.0),  # sentinel: never maintained
        ("model_", 0.0),  # one-hot: not this model
        ("error", 0.0),  # no errors recorded
    )

    def _reconcile_features(self, featured: "pd.DataFrame") -> "pd.DataFrame":
        """
        Add back any training feature the input data could not produce.

        Raises:
            PredictionError: if a feature is missing that has no defensible
                default — a absent sensor column is a broken feed, not a
                sparse category, and must not be silently zero-filled.
        """
        missing = [c for c in self.feature_columns if c not in featured.columns]
        if not missing:
            return featured

        filled: Dict[str, float] = {}
        unexplained: List[str] = []
        for column in missing:
            for prefix, value in self._FILL_RULES:
                if column.startswith(prefix):
                    filled[column] = value
                    break
            else:
                unexplained.append(column)

        if unexplained:
            raise PredictionError(
                f"{len(unexplained)} required feature(s) missing with no safe "
                f"default: {unexplained[:5]}{'...' if len(unexplained) > 5 else ''}. "
                "These are derived from sensor readings, so their absence means "
                "the input tables are malformed rather than merely sparse."
            )

        logger.info(
            f"Reconciled {len(filled)} absent categorical feature(s) against the "
            f"training contract (e.g. {list(filled)[:3]}). This is expected when "
            "the scored batch does not contain every machine model or component."
        )
        return featured.assign(**filled)

    # ------------------------------------------------------------------
    # Risk banding
    # ------------------------------------------------------------------

    def risk_level(self, probability: float) -> str:
        """
        Map a probability to a human-facing band.

        WHY BANDS: "0.73" is not an instruction. A technician needs to know
        whether to finish their shift or stop the line. The HIGH boundary is
        the alert threshold itself, so "high or above" means exactly "the
        model is raising an alert" — the band and the decision cannot drift
        apart.
        """
        s = self.settings
        if probability >= s.RISK_BAND_CRITICAL:
            return "critical"
        if probability >= s.RISK_BAND_HIGH:
            return "high"
        if probability >= s.RISK_BAND_MEDIUM:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_sequences(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Score pre-built `(N, seq_len, n_features)` tensors.

        Calls the model directly rather than `model.predict()`, matching
        ModelEvaluator — no background threads in the inference path.
        """
        if X.ndim != 3:
            raise PredictionError(
                "Expected a 3D (samples, timesteps, features) array, "
                f"got shape {X.shape}"
            )
        expected = (self.sequence_length, len(self.feature_columns))
        if X.shape[1:] != expected:
            raise PredictionError(
                f"Sequence shape mismatch: expected {expected}, got {X.shape[1:]}"
            )

        outputs = []
        for start in range(0, len(X), batch_size):
            batch = np.asarray(X[start : start + batch_size], dtype=np.float32)
            outputs.append(self.model(batch, training=False).numpy())
        return np.concatenate(outputs, axis=0).flatten()

    def predict(
        self,
        dataset: Dict[str, pd.DataFrame],
        latest_only: bool = False,
    ) -> pd.DataFrame:
        """
        Score raw sensor tables end to end.

        Args:
            dataset: `{table_name: DataFrame}` as produced by DataIngestion —
                telemetry, machines, errors, maintenance. A `failures` table
                is not required and is ignored if present: at inference time
                the future is what we are predicting, not something we read.
            latest_only: return just the most recent prediction per machine,
                which is what a dashboard's "current fleet status" wants.

        Returns:
            DataFrame with machine_id, datetime, failure_probability,
            risk_level, and will_fail (probability >= threshold), sorted
            most-urgent first.

        Raises:
            PredictionError: if there is not enough history to build a window.
        """
        try:
            pre = DataPreprocessor(sequence_length=self.sequence_length)

            merged = pre.merge_tables(dataset)
            featured = pre.engineer_features(merged)

            featured = self._reconcile_features(featured)

            # Training-time statistics — never refit. Refitting here is the
            # training/serving skew bug that Risk R-6 is about.
            pre.scaler = self.scaler
            pre.feature_columns = self.feature_columns
            scaled = pre.apply_scaler(featured)

            X, _, index = pre.create_sequences(
                scaled,
                self.feature_columns,
                return_index=True,
                require_labels=False,
            )

            if len(X) == 0:
                raise PredictionError(
                    f"No complete sequences could be built. Each machine needs more "
                    f"than {self.sequence_length} hourly readings after feature "
                    "engineering, which itself consumes the first 24 hours for "
                    "rolling and lag windows. Supply at least "
                    f"{self.sequence_length + 24} hours of history per machine."
                )

            probabilities = self.predict_sequences(X)

            result = index.copy()
            result["failure_probability"] = probabilities
            result["risk_level"] = [self.risk_level(p) for p in probabilities]
            result["will_fail"] = probabilities >= self.threshold

            if latest_only:
                result = (
                    result.sort_values("datetime")
                    .groupby("machine_id", as_index=False)
                    .last()
                )

            result = result.sort_values(
                "failure_probability", ascending=False
            ).reset_index(drop=True)

            n_alerts = int(result["will_fail"].sum())
            logger.info(
                f"Scored {len(result):,} sequences across "
                f"{result['machine_id'].nunique()} machines — "
                f"{n_alerts} at or above threshold {self.threshold:.4f}"
            )
            return result

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}") from e

    # Sensors whose engineered features carry the degradation signal, with the
    # direction that indicates trouble and the physical story behind it. Used
    # to turn a probability into evidence a technician can check.
    # sensor: (concerning sign, unit, what that direction typically means)
    # +1 means high readings are the worry, -1 means low readings are.
    # Voltage is scored on volatility rather than level, so it has no sign.
    _SENSOR_SEMANTICS = {
        "vibration": (+1, "mm/s", "components loosening or bearings worn"),
        "rotation": (-1, "RPM", "bearing drag or load increase"),
        "pressure": (-1, "PSI", "a leak or a failing seal"),
        "voltage": (0, "V", "supply instability or winding fault"),
    }

    def _population_z(self, feature: str, value: float) -> float:
        """
        How unusual is `value` for `feature`, against the training population?

        The fitted StandardScaler already carries `mean_` and `scale_` for
        every one of the 63 features, so the reference distribution is the one
        the model was actually trained on rather than a number someone picked.
        Returns 0.0 when the feature is unknown, so an unrecognised name is
        treated as unremarkable rather than raising mid-report.
        """
        try:
            idx = self.feature_columns.index(feature)
        except ValueError:
            return 0.0
        mean = float(self.scaler.mean_[idx])
        scale = float(self.scaler.scale_[idx])
        return (value - mean) / scale if scale > 1e-9 else 0.0

    def explain_machine(
        self,
        dataset: Dict[str, pd.DataFrame],
        machine_id: Any,
        history_hours: int = 0,
    ) -> Dict[str, Any]:
        """
        Score one machine and return the evidence behind the score.

        WHY THIS EXISTS:
            A probability is not an explanation, and the GenAI layer must not
            be left to invent one. Handing an LLM only "0.87" invites it to
            confabulate a plausible cause; handing it "vibration 61.2 mm/s, up
            18.4 from 24h ago, 24h volatility 9.1" gives it something real to
            report. Every number here is read from the engineered features the
            model actually consumed — none is derived for presentation.

        Args:
            history_hours: Also include the last N hourly readings per sensor.
                A snapshot answers "is it failing?"; a conversation needs
                "has vibration been climbing all week?", which a single row
                cannot support. 0 (the default) keeps the record small for
                one-shot reports.

        Returns:
            The `predict_machine` record plus a `context` block: current
            sensor readings, 24h changes, volatility, error counts,
            maintenance recency, and the sensors ranked by how far they have
            deviated from their own recent baseline.

        Raises:
            PredictionError: if the machine has too little history.
        """
        pre = DataPreprocessor(sequence_length=self.sequence_length)
        merged = pre.merge_tables(dataset)
        featured = self._reconcile_features(pre.engineer_features(merged))

        rows = featured[featured["machine_id"] == machine_id]
        if rows.empty:
            raise PredictionError(f"Machine {machine_id!r} is not present in the data.")
        latest = rows.sort_values("datetime").iloc[-1]

        sensors = {}
        for sensor, (concerning_sign, unit, story) in self._SENSOR_SEMANTICS.items():
            if sensor not in latest:
                continue
            value = float(latest[sensor])
            baseline = float(latest.get(f"{sensor}_rolling_mean_24h", value))
            volatility = float(latest.get(f"{sensor}_rolling_std_24h", 0.0))
            change_24h = float(latest.get(f"{sensor}_change_24h", 0.0))

            # Deviation from the sensor's OWN recent baseline, in units of its
            # own recent volatility — so sensors on different scales (450 RPM
            # vs 40 mm/s) can be ranked against each other honestly.
            deviation = (value - baseline) / volatility if volatility > 1e-9 else 0.0

            # Is this sensor deviating in the direction that actually
            # matters? Pressure 0.7 sigma HIGH is not "a pressure drop", and
            # attaching the leak explanation to it invites a report that
            # contradicts its own numbers.
            if concerning_sign == 0:
                # Voltage fails by becoming erratic, not by drifting, so it is
                # judged on volatility. The reference is the TRAINING
                # population's volatility, taken from the fitted scaler —
                # never a constant invented here. An earlier hand-picked
                # threshold sat below the population mean and so flagged every
                # healthy machine.
                volatility_z = self._population_z(
                    f"{sensor}_rolling_std_24h", volatility
                )
                concerning = volatility_z >= 2.0
            else:
                concerning = (deviation * concerning_sign) >= 1.0

            # `direction` always describes the LEVEL, for every sensor — it is
            # read as "N sigma <direction> baseline", and "0.4 sigma erratic
            # baseline" is not a sentence.
            direction_word = "above" if deviation >= 0 else "below"

            sensors[sensor] = {
                "current": round(value, 2),
                "baseline_24h": round(baseline, 2),
                "change_24h": round(change_24h, 2),
                "volatility_24h": round(volatility, 2),
                "deviation_sigma": round(deviation, 2),
                "unit": unit,
                "direction": direction_word,
                "is_concerning": bool(concerning),
                # Only supplied when the reading actually points that way.
                "typical_cause": story if concerning else None,
            }

        ranked = sorted(
            sensors, key=lambda k: abs(sensors[k]["deviation_sigma"]), reverse=True
        )

        maintenance = {
            comp: int(latest[col])
            for comp in ("comp1", "comp2", "comp3", "comp4")
            if (col := f"hours_since_maint_{comp}") in latest
        }

        trend = []
        if history_hours > 0:
            recent = rows.sort_values("datetime").tail(history_hours)
            for _, r in recent.iterrows():
                trend.append(
                    {
                        "datetime": str(r["datetime"]),
                        **{
                            sensor: round(float(r[sensor]), 2)
                            for sensor in self._SENSOR_SEMANTICS
                            if sensor in r
                        },
                    }
                )

        record = self.predict_machine(dataset, machine_id)
        record["context"] = {
            "age_years": int(latest["age"]) if "age" in latest else None,
            "errors_last_24h": int(latest.get("errors_last_24h", 0)),
            "hours_since_maintenance": maintenance,
            "sensors": sensors,
            "most_deviant_sensors": ranked[:3],
            "recent_readings": trend,
        }
        return record

    def predict_machine(
        self, dataset: Dict[str, pd.DataFrame], machine_id: Any
    ) -> Dict[str, Any]:
        """
        Score one machine and return its latest prediction as a plain dict.

        This is the shape the API and the GenAI report generator consume.
        """
        result = self.predict(dataset, latest_only=True)
        row = result[result["machine_id"] == machine_id]
        if row.empty:
            raise PredictionError(
                f"No prediction produced for machine {machine_id!r} — it is either "
                "absent from the data or has too little history."
            )
        record = row.iloc[0]
        return {
            # .item() unwraps numpy scalars to native Python. pandas hands back
            # np.int64 / np.bool_, which json.dumps refuses — and this record is
            # exactly what the API and the report generator will serialise.
            "machine_id": _native(record["machine_id"]),
            "datetime": str(record["datetime"]),
            "failure_probability": float(record["failure_probability"]),
            "risk_level": str(record["risk_level"]),
            "will_fail": bool(record["will_fail"]),
            "threshold": float(self.threshold),
        }
