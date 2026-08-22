# Day 3 Summary

| Field | Value |
|---|---|
| **Objective** | Turn five raw tables into leak-free, model-ready LSTM tensors. |
| **Expected outcome** | A `DataPreprocessor` performing merge → feature engineering → labeling → temporal split → scaling → windowing, producing `(N, 24, 63)` arrays plus a persisted scaler and feature list. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-21 |
| **Milestone** | M3 — Feature engineering |
| **Status** | ✅ Complete — commit `79c094a` |

> **Note on this document.** Written retroactively on 2026-08-23 when the
> `IMPLEMENTATION_PLAN.md` + `docs/DayX.md` system was introduced. Reconstructed from
> `docs/handoff.md`, git history, the preprocessing module, and the artifacts in
> `data/processed/`. Nothing has been invented to fill gaps.

---

# Starting State

| Field | Value |
|---|---|
| **Repository state** | Clean working tree |
| **Git commit** | `6ed62e9` — "feat(data): add data generation, ingestion pipeline, validation, and EDA" |
| **Existing files** | Day 1 foundation + Day 2 data layer |
| **Existing models** | None |
| **Existing checkpoints** | None |
| **Existing datasets** | `data/raw/` — 5 CSVs, 883,231 rows (gitignored); `data/sample/` — small committed fixture |
| **Known issues** | K-1: sample dataset has 0 failure events |
| **Pending work** | Feature engineering, labeling, splitting, scaling, windowing — then the entire model, GenAI, API, and dashboard stack |

The data existed but was unusable for modelling: five separate tables, no labels, raw
scales, and no sequence structure.

---

# Tasks Planned

### T1 — Merge the five tables

| Field | Detail |
|---|---|
| **Purpose** | An LSTM consumes one tensor. Five tables must become one time-indexed frame. |
| **Files affected** | `src/data/preprocessing.py` |
| **Dependencies** | Day 2 ingestion |
| **Priority** | P0 |
| **Expected outcome** | One frame keyed on `(datetime, machine_id)` carrying telemetry, machine metadata, error counts, and maintenance history. |

### T2 — Engineer rolling, lag, and change features

| Field | Detail |
|---|---|
| **Purpose** | With only 47 failure events, the model cannot discover trend features on its own. Degradation is gradual, so *slope and volatility* carry the signal — not instantaneous values. |
| **Files affected** | `src/data/preprocessing.py` |
| **Dependencies** | T1 |
| **Priority** | P0 |
| **Expected outcome** | 48 engineered features from 4 raw sensors, computed **per machine**. |

### T3 — Create 24-hour-horizon labels

| Field | Detail |
|---|---|
| **Purpose** | Define the supervised target: will this machine fail within 24 hours? |
| **Files affected** | `src/data/preprocessing.py` |
| **Dependencies** | T1 |
| **Priority** | P0 |
| **Expected outcome** | Binary `label` column, 1 for each hour in the 24 h preceding a failure. |

### T4 — Temporal train/test split

| Field | Detail |
|---|---|
| **Purpose** | Prevent data leakage. This is the project's most important correctness property. |
| **Files affected** | `src/data/preprocessing.py` |
| **Dependencies** | T3 |
| **Priority** | P0 |
| **Expected outcome** | Train Jan–Oct, test Oct–Dec, with `max(train.datetime) < min(test.datetime)` asserted by a test. |

### T5 — Normalization fitted on train only

| Field | Detail |
|---|---|
| **Purpose** | Fitting a scaler on the full dataset leaks test-set statistics into training. |
| **Files affected** | `src/data/preprocessing.py`, `data/processed/scaler.joblib` |
| **Dependencies** | T4 |
| **Priority** | P0 |
| **Expected outcome** | `StandardScaler.fit(train)`, applied to both splits, persisted for inference. |

### T6 — LSTM sequence windowing

| Field | Detail |
|---|---|
| **Purpose** | Convert a 2D frame into the 3D `(samples, timesteps, features)` tensor an LSTM requires. |
| **Files affected** | `src/data/preprocessing.py`, `data/processed/*.npy` |
| **Dependencies** | T5 |
| **Priority** | P0 |
| **Expected outcome** | `(N, 24, 63)` float32 arrays, with **no window spanning two machines**. |

### T7 — Pipeline runner script

| Field | Detail |
|---|---|
| **Purpose** | Preprocessing is expensive; it should run once and persist artifacts, not re-run inside training. |
| **Files affected** | `scripts/run_preprocessing.py` |
| **Dependencies** | T6 |
| **Priority** | P1 |
| **Expected outcome** | `python scripts/run_preprocessing.py` writes everything under `data/processed/`. |

### T8 — Tests for every invariant

| Field | Detail |
|---|---|
| **Purpose** | Leakage bugs are silent. They must be caught by assertion, not by noticing a suspiciously good metric. |
| **Files affected** | `tests/unit/test_preprocessing.py` |
| **Dependencies** | T1–T6 |
| **Priority** | P0 |
| **Expected outcome** | 27 tests, including explicit anti-leakage assertions. |

---

# Work Completed

All eight tasks completed. `src/data/preprocessing.py` became the project's largest module
at roughly **780 lines**.

## The pipeline, in its mandatory order

The ordering is not stylistic. Each step depends on the previous one having happened, and
**reversing steps 10–12 silently destroys the validity of every metric the project will
ever report.**

| # | Step | Detail |
|---|---|---|
| 1 | **Merge** | Left-join telemetry with machines, errors, maintenance, failures on `(datetime, machine_id)` |
| 2 | **One-hot encode** | `model` categorical → dummy columns |
| 3 | **Error aggregation** | Errors per machine-hour; rolling 24 h error count |
| 4 | **Maintenance features** | Hours since last replacement, per component |
| 5 | **Rolling statistics** | Per machine: mean and std over 3 h / 12 h / 24 h, for each of 4 sensors → 24 features |
| 6 | **Lag features** | Per machine: values at t−1 h, t−6 h, t−24 h → 12 features |
| 7 | **Change features** | `current − lag` for each lag → 12 features |
| 8 | **Forward/back fill** | Fill the NaNs that rolling and lag operations create at window edges |
| 9 | **Label creation** | `label = 1` for every hour within 24 h before a failure on that machine |
| 10 | **Temporal split** | Train Jan–Oct, test Oct–Dec |
| 11 | **Normalization** | `StandardScaler.fit(train)` → transform both |
| 12 | **Sequence windowing** | 24-step sliding windows, per machine |

## Output artifacts

| Artifact | Shape / size | Notes |
|---|---|---|
| `X_train.npy` | (698400, 24, 63) float32 | **4.2 GB** |
| `y_train.npy` | (698400,) | **957 positives** — 0.137%, ratio 1:730 |
| `X_test.npy` | (172800, 24, 63) float32 | **1.0 GB** |
| `y_test.npy` | (172800,) | **200 positives** — 0.116%, ratio 1:864 |
| `scaler.joblib` | 3.7 KB | Fitted on train only |
| `feature_columns.txt` | 63 lines | The ordered feature contract |

The 5.2 GB total is what forced Day 4's memmapping strategy — these arrays cannot be
resident alongside TensorFlow on a 16 GB machine.

## The 63 features

| Category | Count | Example |
|---|---|---|
| Raw sensors | 4 | `voltage`, `rotation`, `pressure`, `vibration` |
| Machine metadata | 1 + dummies | `age`, `model_*` |
| Rolling mean (3 windows × 4 sensors) | 12 | `voltage_rolling_mean_3h` |
| Rolling std (3 windows × 4 sensors) | 12 | `pressure_rolling_std_12h` |
| Lag (3 lags × 4 sensors) | 12 | `rotation_lag_6h` |
| Change (3 lags × 4 sensors) | 12 | `vibration_change_24h` |
| Error aggregates | ~5 | `error_count_24h` |
| Maintenance | ~4 | `hours_since_comp1_replacement` |

**Why hand-engineer these at all?** An LSTM can in principle learn rolling statistics from
raw sequences. With 957 positive training examples, it will not. Hand-engineering injects
the domain knowledge — *degradation is gradual, so slope and variance matter more than
level* — that label scarcity would otherwise prevent the network from discovering.

## Anti-leakage work

Three invariants, each enforced by design and asserted by a test:

### 1. Temporal split, never random

With lag features spanning 24 hours, a random split places hour `t` in train and hour
`t+1` in test. The model then "predicts" a failure whose evidence it has already
memorized from the other side of the boundary. Reported AUC becomes meaningless — and
*looks excellent*, which is what makes it dangerous.

Test asserts `max(train.datetime) < min(test.datetime)`.

### 2. Scaler fitted on train only

Fitting on the full dataset lets the test period's mean and variance influence the
training features. Subtle, but it is still the future informing the past.

Test asserts the scaler's statistics match train-only statistics.

### 3. Windows never span machines

Sequences are built per `machine_id` and concatenated afterward. A naive global sliding
window would splice machine 7's last hours onto machine 8's first — a sequence describing
a machine that does not exist.

Test asserts every window contains exactly one distinct `machine_id`.

## Tests (T8)

27 tests in `tests/unit/test_preprocessing.py`, covering: merge correctness and row counts;
rolling and lag values against hand-computed expectations; NaN handling at window edges;
label timing (a synthetic failure is planted and *exactly* the preceding 24 hours must be
labeled); temporal-split ordering; scaler statistics; sequence shape and label alignment;
and the full pipeline end to end producing correct shapes with no NaNs.

Total suite: **68 tests**.

---

# Code Changes

## `src/data/preprocessing.py` — created (~780 lines)

| Field | Detail |
|---|---|
| **Purpose** | The entire raw → LSTM-tensor transformation. |
| **Important changes** | `DataPreprocessor` with one method per pipeline stage, plus `run_pipeline()` orchestrating them in the mandatory order. |
| **Breaking changes** | None (new file). |
| **Configuration** | Horizon (24 h) and sequence length (24) are parameters, defaulted from settings and overridable via CLI. |
| **Imports added** | `pandas`, `numpy`, `sklearn.preprocessing.StandardScaler`, `joblib` |
| **Functions added** | Merge, error aggregation, maintenance features, rolling stats, lag features, change features, fill, label creation, temporal split, scaling, windowing, `run_pipeline` |
| **Classes changed** | `DataPreprocessor` (new) |

## `scripts/run_preprocessing.py` — created

| Field | Detail |
|---|---|
| **Purpose** | Run the pipeline once and persist artifacts. |
| **Important changes** | `--horizon` and `--seq-len` flags; writes all six artifacts to `data/processed/`. |
| **Imports added** | `src.data.preprocessing`, `config.settings` |

## `src/data/__init__.py` — modified

Now exports `DataPreprocessor` alongside `DataIngestion` and `DataValidator`.

## `tests/unit/test_preprocessing.py` — created

27 tests.

---

# Training Progress

No model training occurred on Day 3. The deliverable was the tensors the model would
consume. The label statistics established here (957 train / 200 test positives) set the
class weights used on Day 4: {0: 0.50, 1: 364.89}.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 27 new, **68 total**, all passing |
| **Integration tests** | None |
| **Manual testing** | Ran the full pipeline on 876K rows and verified output shapes, positive counts, and absence of NaNs |
| **Benchmark results** | Full pipeline run completes in minutes; the dominant cost is `groupby().rolling()` over 100 machines |
| **Performance metrics** | Not formally profiled |
| **Memory usage** | Peak pandas memory during feature engineering was the day's tightest constraint; the output arrays total 5.2 GB on disk |
| **CPU usage** | Single-threaded pandas; not measured |
| **GPU usage** | N/A |

---

# Bugs Encountered

No bugs were recorded on Day 3. The invariants were designed in from the start rather than
discovered by failure — which is the intended outcome of writing the anti-leakage tests
alongside the code rather than after it.

The day's real risk was **silent** rather than loud: a leakage bug produces no error, no
warning, and a *better-looking* metric. That is precisely why each invariant got an
explicit assertion instead of relying on review.

---

# Design Decisions

## D1 — Temporal split, never random

| Field | Detail |
|---|---|
| **Alternatives** | `train_test_split(shuffle=True)`; stratified split; group split by machine. |
| **Pros** | The only split that honestly simulates deployment: train on the past, predict the future. |
| **Cons** | Train and test come from different periods, so any distribution drift shows up as apparent model weakness. |
| **Reason for selection** | With 24-hour lag features, a random split leaks directly. The resulting metrics would be excellent and worthless. |
| **Impact** | Listed as a "must never change" invariant in `CLAUDE.md` and `IMPLEMENTATION_PLAN.md`. |

## D2 — Fit the scaler on training data only

| Field | Detail |
|---|---|
| **Alternatives** | Fit on the full dataset; fit per split; skip scaling. |
| **Pros** | No test statistics reach training; the persisted scaler is exactly what inference must use. |
| **Cons** | If the test period genuinely drifts, features will be off-center — but that is *information*, not a defect. |
| **Reason for selection** | Standard correct practice, and the drift signal is worth having. |
| **Impact** | `scaler.joblib` must ship with the model; refitting it at inference time would be a training/serving skew bug (Risk R-6). |

## D3 — `StandardScaler` over `MinMaxScaler`

| Field | Detail |
|---|---|
| **Alternatives** | MinMax; RobustScaler; no scaling. |
| **Pros** | Handles outliers better than MinMax; unbounded output suits LSTM gate activations; a single outlier cannot compress every other value. |
| **Cons** | Does not bound the range, so extreme inference-time values produce extreme inputs. |
| **Reason for selection** | Sensor spikes are exactly what the model must detect. MinMax would let one spike squash the rest of the distribution. |
| **Impact** | Features are roughly standard-normal, which is what LSTM initialization assumes. |

## D4 — 24-hour prediction horizon

| Field | Detail |
|---|---|
| **Alternatives** | 6 h; 48 h; 7 days. |
| **Pros** | Long enough to schedule a technician; short enough that the degradation signal is actually present. |
| **Cons** | Shorter horizons would be easier to predict but less operationally useful; longer ones the reverse. |
| **Reason for selection** | Matches the generator's 48-hour degradation ramp — 24 h sits inside the window where signal exists. |
| **Impact** | Defines the label and therefore the entire supervised problem. Parameterized as `--horizon` so alternatives can be tested. |

## D5 — 24-timestep sequence length

| Field | Detail |
|---|---|
| **Alternatives** | 12; 48; 168 (one week). |
| **Pros** | Captures a full daily cycle, so the model can distinguish the generator's daily periodicity from genuine degradation. |
| **Cons** | May miss slower multi-day trends. |
| **Reason for selection** | One day of history to predict one day ahead is symmetric and interpretable; 48 would double an already-4.2 GB tensor. |
| **Impact** | Fixes the input shape at `(24, 63)`. Parameterized as `--seq-len`. |

## D6 — Hand-engineered features rather than raw sequences alone

| Field | Detail |
|---|---|
| **Alternatives** | Feed the 4 raw sensors and let the LSTM learn everything; a deeper network; attention. |
| **Pros** | Injects domain knowledge the label scarcity cannot support learning; makes features interpretable, which the GenAI layer will need to *explain* a prediction. |
| **Cons** | 63 features instead of 4 — a 16× larger tensor, and some redundancy. |
| **Reason for selection** | 957 positive examples. Representation learning needs orders of magnitude more. |
| **Impact** | Drove the tensor size, which drove memmapping. Also gives Day 7's report generator named, meaningful quantities to cite. |

## D7 — Persist artifacts to disk instead of preprocessing inside training

| Field | Detail |
|---|---|
| **Alternatives** | Preprocess at the start of each training run; cache in memory. |
| **Pros** | Preprocessing runs once; training iterations are fast; the data and model layers stay genuinely decoupled; artifacts are inspectable. |
| **Cons** | 5.2 GB on disk; the artifacts can go stale relative to the code that produced them. |
| **Reason for selection** | Re-running an expensive pipeline on every experiment is a waste, and the decoupling enforces the project's layering rule. |
| **Impact** | Day 4's trainer reads `.npy` files and never imports pandas — which turned out to matter enormously, since pandas is what poisons TensorFlow (see `docs/Day4.md`). |

## D8 — Windows built per machine

| Field | Detail |
|---|---|
| **Alternatives** | Global sliding window over the sorted frame. |
| **Pros** | Every sequence describes one real machine's continuous history. |
| **Cons** | Slightly more code; loses the boundary windows between machines. |
| **Reason for selection** | A cross-machine window is a sequence describing a machine that does not exist. There is no defensible version of this shortcut. |
| **Impact** | A "must never change" invariant, asserted by test. |

---

# Remaining Tasks

None from Day 3 — all objectives met.

---

# Next Day Plan

**Day 4 — LSTM Model Architecture & Training**

1. Build `PredictiveMaintenanceModel` in `src/models/lstm_model.py` — the Keras
   architecture, with `save()` and `load()`.
2. Build `ModelTrainer` in `src/models/trainer.py` — compile with class-weighted binary
   crossentropy, AUC/precision/recall metrics, and EarlyStopping / ReduceLROnPlateau /
   ModelCheckpoint callbacks.
3. Build `ModelEvaluator` in `src/models/evaluator.py` — AUC, precision, recall, F1, and
   confusion matrix. **Never accuracy.**
4. Build `scripts/train_model.py` — load memmapped arrays, train, evaluate, save metrics.
5. Write model tests.
6. Train on the full 698,400-sequence dataset and record the metrics.

**Anticipated difficulty:** memory. `X_train` is 4.2 GB, so `np.load(..., mmap_mode='r')`
is mandatory and the batch pipeline must materialize only one batch at a time.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~25% (3 of 12 days) |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · rest 0% |
| **Technical debt** | None |
| **Known risks** | R-1 out-of-memory during training (4.2 GB tensor); R-7 imbalance collapse (1:730) |
| **Immediate priorities** | Build and train the LSTM |

---

# Files Created

```
src/data/preprocessing.py           (~780 lines)
scripts/run_preprocessing.py
tests/unit/test_preprocessing.py    (27 tests)
data/processed/X_train.npy          (gitignored, 4.2 GB)
data/processed/y_train.npy          (gitignored, 957 positives)
data/processed/X_test.npy           (gitignored, 1.0 GB)
data/processed/y_test.npy           (gitignored, 200 positives)
data/processed/scaler.joblib        (gitignored)
data/processed/feature_columns.txt  (gitignored, 63 names)
```

# Files Modified

```
src/data/__init__.py   — export DataPreprocessor
```

# Files Deleted

None.

# Models Generated

None.

# Checkpoints Generated

None.

# Reports Generated

Pipeline summary logged to console and `logs/app_2026-08-21.log`, reporting final shapes
and positive counts per split.

# Logs Generated

`logs/app_2026-08-21.log` (71 KB) — full pipeline execution.

# References

- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Keras: recurrent layers and input shape](https://keras.io/api/layers/recurrent_layers/lstm/)
- Pandas `groupby().rolling()` documentation — the mechanism behind the per-machine rolling features
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — on why time-series validation must respect chronology

---

# Final Summary

Day 3 built the project's most substantial module and its most important correctness
guarantees. `DataPreprocessor` takes five raw tables to `(698400, 24, 63)` training
tensors through twelve ordered steps, engineering 48 features from four sensors because
957 positive examples are far too few for a network to discover trend structure on its own.

The day's real work was not the feature count — it was the three anti-leakage invariants:
temporal split, train-only scaler fitting, and per-machine windowing. Each is enforced by
an explicit test, because leakage is a *silent* failure that makes results look better,
not worse. A random split here would have produced a beautiful, meaningless AUC, and
nothing in the code would have complained.

One decision made in passing turned out to matter far more than expected: persisting
artifacts to disk so that training reads `.npy` files and never imports pandas. Day 4 would
discover that pandas is precisely what breaks TensorFlow on this platform — and the data
layer's isolation is what made that bug tractable rather than pervasive.

Ending state: 68 passing tests, commit `79c094a`, and 5.2 GB of model-ready tensors.
