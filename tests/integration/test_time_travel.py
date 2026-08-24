"""
tests/integration/test_time_travel.py
======================================
Does rewinding the dashboard actually show the model working?

WHY THIS FILE EXISTS:
    The unit tests prove `as_of` is plumbed through — the timestamp reaches the
    service, rows after it are dropped. None of that proves the feature is
    *worth having*. It is worth having only if pointing it at a real
    pre-failure hour produces an alert, and pointing it a day earlier does not.

    That claim needs the trained model and the generated dataset, so it lives
    here rather than in the unit suite, and skips when they are absent.

HOW IT WORKS:
    `data/raw/failures.csv` is ground truth: 47 timestamped failures. For each
    one in the test period the store is rewound to six hours before it and the
    machine scored. Nothing is mocked; this is the same code path the API
    serves.

    The horizon test is the one that would catch a leak. A model rewound to 36
    hours before a failure must be *quiet* — it was trained to see 24 hours
    ahead. An alert there would mean either that the labels reach further than
    they claim or that filtering is not actually hiding the future.
"""

from pathlib import Path

import pandas as pd
import pytest

from config.settings import get_settings
from src.api.service import MachineDataStore
from src.prediction import Predictor
from src.utils.exceptions import ModelNotFoundError

pytestmark = pytest.mark.integration

RAW_TABLES = ("telemetry", "machines", "errors", "maintenance")

# Far enough inside the 24 h label horizon that an alert is expected, but not
# so close that the failure itself is in the window.
LEAD_HOURS = 6
# Comfortably outside it. The model has no reason to fire this early.
BEYOND_HORIZON_HOURS = 36


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def _require(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(
            "generated dataset/model not present (both are gitignored): "
            f"{missing[0]}. Run scripts/generate_data.py, run_preprocessing.py, "
            "and train_model.py."
        )


@pytest.fixture(scope="module")
def store(settings):
    _require([settings.raw_data_path / f"{n}.csv" for n in RAW_TABLES])
    return MachineDataStore.load(settings.raw_data_path)


@pytest.fixture(scope="module")
def predictor(settings):
    _require([settings.model_path, settings.scaler_path, settings.feature_columns_path])
    try:
        return Predictor()
    except ModelNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def failures(settings):
    path = settings.raw_data_path / "failures.csv"
    _require([path])
    return pd.read_csv(path, parse_dates=["datetime"])


def _score(predictor, store, machine_id, as_of):
    sliced = store.slice_for(machine_id, as_of=as_of)
    result = predictor.predict(sliced, latest_only=True)
    return result.iloc[0]


def _sample(failures, store, n=5):
    """The last few failures that have room for a 36 h rewind before them."""
    lo, _ = store.data_range
    usable = failures[
        failures["datetime"] > lo + pd.Timedelta(hours=BEYOND_HORIZON_HOURS + 200)
    ]
    return list(usable.tail(n).itertuples())


def test_rewinding_to_a_pre_failure_hour_raises_an_alert(predictor, store, failures):
    """
    The feature's whole justification.

    Assessed at the dataset's final hour the fleet is quiet, which reads as a
    model that does nothing. Rewound to hours before a real failure it should
    fire — and if it does not, the rewind control is decoration.
    """
    sample = _sample(failures, store)
    assert sample, "no failures far enough into the dataset to rewind before"

    alerted = []
    for f in sample:
        row = _score(
            predictor,
            store,
            int(f.machine_id),
            f.datetime - pd.Timedelta(hours=LEAD_HOURS),
        )
        alerted.append(bool(row["will_fail"]))

    assert all(alerted), (
        f"only {sum(alerted)}/{len(alerted)} machines alerted "
        f"{LEAD_HOURS} h before a known failure"
    )


def test_the_model_stays_quiet_beyond_its_horizon(predictor, store, failures):
    """
    Rewound past 24 h the alerts must stop.

    This is the leak check. The features include 24 h rolling windows, so if
    `as_of` filtering were incomplete — telemetry trimmed but errors left
    whole, say — a 36 h rewind could still see the failure's own aftermath and
    fire. Silence here is evidence the filtering is real.
    """
    sample = _sample(failures, store)
    early = [
        bool(
            _score(
                predictor,
                store,
                int(f.machine_id),
                f.datetime - pd.Timedelta(hours=BEYOND_HORIZON_HOURS),
            )["will_fail"]
        )
        for f in sample
    ]

    assert not any(early), (
        f"{sum(early)}/{len(early)} machines alerted {BEYOND_HORIZON_HOURS} h out, "
        "beyond the 24 h horizon the model was trained for — suspect leakage"
    )


def test_hiding_the_future_changes_the_answer(predictor, store, failures):
    """
    A rewound assessment must differ from the present-day one.

    If `as_of` were dropped somewhere between the route and the slice, every
    test above could still pass by scoring the latest hour each time. This
    asserts the two answers are genuinely different.
    """
    f = _sample(failures, store)[-1]
    machine_id = int(f.machine_id)

    rewound = _score(
        predictor, store, machine_id, f.datetime - pd.Timedelta(hours=LEAD_HOURS)
    )
    latest = _score(predictor, store, machine_id, None)

    assert rewound["datetime"] < latest["datetime"]
    assert rewound["failure_probability"] != latest["failure_probability"]


def test_a_rewound_slice_contains_nothing_after_the_cutoff(store, failures):
    """
    Belt and braces on the store itself, over the real 876,000-row dataset
    rather than the four-row frame the unit tests build.
    """
    f = _sample(failures, store)[-1]
    cutoff = f.datetime - pd.Timedelta(hours=LEAD_HOURS)
    sliced = store.slice_for(int(f.machine_id), as_of=cutoff)

    for name, frame in sliced.items():
        if "datetime" in frame.columns and len(frame):
            assert frame["datetime"].max() <= cutoff, f"{name} leaked past {cutoff}"
