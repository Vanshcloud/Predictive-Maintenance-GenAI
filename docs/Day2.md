# Day 2 Summary

| Field | Value |
|---|---|
| **Objective** | Produce a realistic, reproducible dataset and the code to load and trust it. |
| **Expected outcome** | A synthetic 5-table dataset with learnable failure signal, a format-detecting ingestion class, a validation class producing structured reports, and an EDA analysis that quantifies the modelling problem. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-20 |
| **Milestone** | M2 — Data foundation |
| **Status** | ✅ Complete — commit `755e95c` |

> **Note on this document.** Written retroactively on 2026-08-23 when the
> `IMPLEMENTATION_PLAN.md` + `docs/DayX.md` system was introduced. Reconstructed from
> `docs/handoff.md`, git history, and the repository. Nothing has been invented to fill gaps.

---

# Starting State

| Field | Value |
|---|---|
| **Repository state** | Clean working tree |
| **Git commit** | `52824f6` — "docs: update README with actual GitHub repo URL and author info" |
| **Existing files** | 33 files, 1,511 lines — the Day 1 foundation |
| **Existing models** | None |
| **Existing checkpoints** | None |
| **Existing datasets** | **None** — `data/raw/`, `data/processed/`, `data/sample/` held only `.gitkeep` |
| **Known issues** | None |
| **Pending work** | Everything from Day 2 onward. The project had structure but no data, so nothing could be modelled yet. |

---

# Tasks Planned

### T1 — Synthetic data generator

| Field | Detail |
|---|---|
| **Purpose** | The project needs data with *known-correct* labels and *learnable* degradation patterns. Downloading a real dataset introduces licensing questions, a network dependency, and uncertainty about label quality. |
| **Files affected** | `scripts/generate_data.py`, `data/raw/*.csv`, `data/sample/*.csv` |
| **Dependencies** | Day 1 foundation |
| **Priority** | P0 — everything downstream is blocked on it |
| **Expected outcome** | 5 related tables, ~883K rows, seed-reproducible, with gradual pre-failure degradation. |

### T2 — `DataIngestion` class

| Field | Detail |
|---|---|
| **Purpose** | Loading data should be one call that works regardless of file format, and should log what it loaded so problems are visible immediately. |
| **Files affected** | `src/data/ingestion.py`, `src/data/__init__.py` |
| **Dependencies** | T1 |
| **Priority** | P0 |
| **Expected outcome** | CSV/Parquet/JSON auto-detection, metadata logging, `DataIngestionError` on failure. |

### T3 — `DataValidator` class

| Field | Detail |
|---|---|
| **Purpose** | Bad data that reaches the model produces a bad model *silently*. Validation makes data problems loud and early. |
| **Files affected** | `src/data/validation.py` |
| **Dependencies** | T2 |
| **Priority** | P0 |
| **Expected outcome** | Schema, null, duplicate, and physical-range checks producing a structured `ValidationReport`. |

### T4 — EDA analysis script

| Field | Detail |
|---|---|
| **Purpose** | Modelling decisions must follow from the data's actual properties. Above all: **how imbalanced is this problem?** |
| **Files affected** | `scripts/eda_analysis.py` |
| **Dependencies** | T1 |
| **Priority** | P1 |
| **Expected outcome** | An 8-dimension report: balance, distributions, correlations, age effects, failure modes, temporal patterns, nulls, duplicates. |

### T5 — Sample dataset for tests

| Field | Detail |
|---|---|
| **Purpose** | Tests cannot depend on a gitignored 883K-row dataset, and CI cannot regenerate it within a reasonable time budget. |
| **Files affected** | `scripts/generate_data.py` (`--sample` flag), `data/sample/*.csv` |
| **Dependencies** | T1 |
| **Priority** | P1 |
| **Expected outcome** | A small committed dataset (10 machines × 30 days). |

### T6 — Tests for the data layer

| Field | Detail |
|---|---|
| **Purpose** | Every module ships with its tests. |
| **Files affected** | `tests/unit/test_data_pipeline.py` |
| **Dependencies** | T2, T3 |
| **Priority** | P1 |
| **Expected outcome** | 22 tests covering ingestion and validation. |

---

# Work Completed

All six planned tasks were completed.

## Synthetic data generator (T1)

`scripts/generate_data.py` produces five related tables modeled on the Microsoft Azure
Predictive Maintenance dataset, with `seed=42`:

| Table | Rows | Columns |
|---|---|---|
| `telemetry.csv` | 876,000 | `datetime, machine_id, voltage, rotation, pressure, vibration` |
| `machines.csv` | 100 | `machine_id, model, age` |
| `errors.csv` | 5,386 | `datetime, machine_id, error_id` |
| `maintenance.csv` | 1,698 | `datetime, machine_id, comp` |
| `failures.csv` | 47 | `datetime, machine_id, failure` |

**Total: 883,231 rows** spanning 2024-01-01 → 2024-12-30 (364 days, hourly, 100 machines).

Sensor baselines and their pre-failure degradation signatures:

| Sensor | Unit | Mean | Std | Range | Degradation |
|---|---|---|---|---|---|
| voltage | V | 170 | 15 | [100, 250] | becomes erratic, ±25 V |
| rotation | RPM | 450 | 50 | [100, 800] | drops ~80 RPM (bearing wear) |
| pressure | PSI | 100 | 12 | [40, 180] | drops ~20 PSI (leaks) |
| vibration | mm/s | 40 | 8 | [10, 100] | rises ~20 mm/s (loosening) |

**Realism features, and why each one matters:**

| Feature | Why it was included |
|---|---|
| Daily periodicity in all sensors | Real factory sensors track ambient temperature cycles. Without it, any rolling-mean feature would be trivially clean and the model would look better than it should. |
| Older machines have noisier sensors | Makes `age` a genuinely informative feature rather than decoration. |
| **Gradual 48 h degradation ramp** | The single most important choice. A step change would make the problem trivial — a threshold rule would solve it and an LSTM would be pointless. A gradual ramp is what makes *sequence* modelling the right tool. |
| Error frequency rises the week before failure | Gives the error-count features real predictive content. |
| Per-machine sensor offsets | Models manufacturing variance, and forces per-machine feature engineering rather than global thresholds. |

`--machines` and `--days` flags allow custom sizes.

## `DataIngestion` (T2)

Loads a table with format detection by extension (CSV/Parquet/JSON), logs row count,
column count, and memory footprint on every load, and raises `DataIngestionError` with
context on failure. Accepts an injected `settings` object for testability.

## `DataValidator` (T3)

Four check families producing a structured `ValidationReport`:

| Check | What it catches |
|---|---|
| **Schema** | Missing columns, unexpected columns, wrong dtypes |
| **Null** | Any nulls, per column, with counts |
| **Duplicate** | Duplicate rows, and duplicate `(datetime, machine_id)` keys |
| **Range** | Sensor values outside physically plausible bounds |

**Design decision: the validator returns a report rather than raising.** A validation
failure is not always fatal — a caller ingesting a new data source may want to inspect
every problem at once rather than fix them one exception at a time. Callers decide
whether to proceed; the report gives them the facts to decide with.

## EDA analysis (T4)

`scripts/eda_analysis.py` reports across 8 dimensions. The findings:

### 1. Class imbalance — the defining property of this project

```
Total telemetry readings: 876,000
Total failure events:          47
Raw event rate:            0.005%
Imbalance ratio:          1:18,638
```

After Day 3's 24-hour labeling this becomes 1,175 positive *rows* (0.13%, ≈1:745), but
the conclusion was already fixed on Day 2:

**Accuracy is useless here.** A model predicting "no failure" for every row scores 99.87%.
Every metric from here on is AUC / precision / recall / F1, and class weighting is
mandatory. This one finding shaped Days 4 and 5 entirely.

### 2. Sensor distributions

All four sensors approximately symmetric, skewness < 0.5. No log transforms needed.

### 3. Sensor correlations

```
             voltage  rotation  pressure  vibration
voltage       1.0000    0.0086    0.0405     0.0628
rotation      0.0086    1.0000    0.0153     0.0163
pressure      0.0405    0.0153    1.0000     0.0754
vibration     0.0628    0.0163    0.0754     1.0000
```

Maximum pairwise |r| = 0.075. **All four sensors carry independent information** — none
can be dropped, and no dimensionality reduction is warranted.

### 4. Age vs failure

| Age bracket | Failures/year |
|---|---|
| 0–5 years | 0.2 |
| 16–20 years | 0.7 (**3.5× more**) |

`age` is a real predictor and belongs in the feature set.

### 5. Failure mode distribution

Roughly uniform: comp1 23%, comp2 28%, comp3 21%, comp4 28%. No single dominant mode,
which is what makes multi-class failure-mode prediction a viable *future* extension.

### 6. Temporal patterns

Failures spread uniformly across months — no seasonality to model or control for.

### 7. Missing values

Zero, in every table.

### 8. Duplicates

Zero, in every table.

## Sample dataset (T5)

`python scripts/generate_data.py --sample` writes a 10-machine × 30-day dataset to
`data/sample/`, which **is committed** so tests and CI have data without a 5 GB
regeneration step.

**Known limitation, discovered immediately:** 10 machines over 30 days produces **zero
failure events**. The failure rate is low enough that a small sample contains none. This
is logged as known issue **K-1** and has a real consequence — any test needing a positive
label must synthesize one, which is exactly what Day 4's model tests do.

## Tests (T6)

22 tests in `tests/unit/test_data_pipeline.py`: format detection for each supported type,
error paths for missing and malformed files, metadata logging, and each validator check
family against both clean and deliberately corrupted frames. Total suite: **41 tests**.

---

# Code Changes

## `scripts/generate_data.py` — created

| Field | Detail |
|---|---|
| **Purpose** | Reproducible synthetic dataset generation. |
| **Important changes** | Per-sensor config dicts (`SENSOR_CONFIG`) so a new sensor is a data change, not a code change; seeded RNG; `--sample`, `--machines`, `--days` flags. |
| **Breaking changes** | None (new file). |
| **Imports added** | `numpy`, `pandas`, `argparse`, `pathlib` |
| **Functions added** | Generators per table, plus degradation-injection helpers. |

## `src/data/ingestion.py` — created

| Field | Detail |
|---|---|
| **Purpose** | Uniform, logged, format-agnostic data loading. |
| **Important changes** | Extension-based dispatch; metadata logging; `DataIngestionError` wrapping. |
| **Configuration** | Accepts an injected `Settings` for testability. |
| **Classes changed** | `DataIngestion` (new) |

## `src/data/validation.py` — created

| Field | Detail |
|---|---|
| **Purpose** | Make data problems visible before they reach a model. |
| **Important changes** | Four check families; `ValidationReport` return type instead of raising. |
| **Classes changed** | `DataValidator`, `ValidationReport` (both new) |

## `src/data/__init__.py` — modified

Exports `DataIngestion` and `DataValidator`.

## `scripts/eda_analysis.py` — created

8-dimension analysis with a `--data-dir` flag so it can run against `data/sample/`.

## `tests/unit/test_data_pipeline.py` — created

22 tests.

---

# Training Progress

No model training occurred on Day 2. The data pipeline was the deliverable; the imbalance
finding is what shaped later training.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 22 new, **41 total**, all passing |
| **Integration tests** | None |
| **Manual testing** | Generated the full dataset twice with the same seed and confirmed identical output; ran EDA against both full and sample datasets |
| **Benchmark results** | Full generation completes in a couple of minutes |
| **Performance metrics** | Not formally measured |
| **Memory usage** | Telemetry frame ~40 MB in pandas — comfortable |
| **CPU usage** | Not measured |
| **GPU usage** | N/A |

---

# Bugs Encountered

No bugs. One **known limitation** was discovered:

## K-1 — Sample dataset contains zero failure events

| Field | Detail |
|---|---|
| **Description** | `data/sample/failures.csv` has 0 rows. |
| **Root cause** | Not a bug — arithmetic. At the real-world failure rate, 10 machines × 30 days has an expected failure count below 1. The sample is *correctly* generated; it is simply too small to contain a rare event. |
| **Files affected** | `data/sample/failures.csv` |
| **Solution** | Accepted rather than "fixed". Inflating the failure rate in the sample would make it unrepresentative, and enlarging it would defeat the purpose of a small committed fixture. |
| **Verification** | Documented as K-1 in `IMPLEMENTATION_PLAN.md`. |
| **Lessons learned** | A test fixture derived from a rare-event dataset will not contain the rare event. Tests that need a positive class must **synthesize** it explicitly. Day 4's model tests do exactly this — `y = [0,0,0,0,0,0,0,0,1,1]` — and are clearer for it, because the class-weight arithmetic is then asserted against known numbers rather than whatever the data happened to contain. |

---

# Design Decisions

## D1 — Synthetic data instead of downloading a real dataset

| Field | Detail |
|---|---|
| **Alternatives** | Kaggle Azure Predictive Maintenance download; NASA C-MAPSS turbofan; a real plant dataset. |
| **Pros** | No network or account dependency; exact reproducibility from a seed; no licensing ambiguity; **guaranteed-correct labels**; full control over difficulty. |
| **Cons** | Patterns are ones we invented — a model's *weights* would not transfer to real equipment, and the achievable metrics are somewhat artificial. |
| **Reason for selection** | The project's purpose is to demonstrate an end-to-end *pipeline*. Synthetic data makes every stage reproducible for any reader, and the degradation physics were designed to be non-trivial (gradual, noisy, machine-specific) so the ML problem is real. |
| **Impact** | Anyone can clone the repo and regenerate byte-identical data. The honest caveat — that metrics reflect synthetic difficulty — is recorded in the plan. |

## D2 — Five related tables instead of one flat file

| Field | Detail |
|---|---|
| **Alternatives** | A single pre-joined wide table. |
| **Pros** | Mirrors real industrial data, where telemetry, errors, maintenance logs, and failure records live in separate systems. Forces the pipeline to actually solve the merge problem. |
| **Cons** | Preprocessing is meaningfully harder. |
| **Reason for selection** | The merge is a real part of the work. Skipping it would make the project look easier than the problem is. |
| **Impact** | `DataPreprocessor` (Day 3) opens with a five-way merge. |

## D3 — Gradual degradation over a 48-hour ramp

| Field | Detail |
|---|---|
| **Alternatives** | Step change at failure; no pre-failure signal; a very short ramp. |
| **Pros** | Makes 24-hour-ahead prediction *possible but not trivial*; rewards trend and volatility features; justifies a sequence model. |
| **Cons** | Harder problem, lower achievable metrics. |
| **Reason for selection** | A step change would be solvable with a threshold and would make the LSTM decorative. The whole architecture rests on there being a *trend* to detect. |
| **Impact** | Directly justifies Day 3's rolling/lag/change features and Day 4's LSTM. |

## D4 — Validator returns a report instead of raising

| Field | Detail |
|---|---|
| **Alternatives** | Raise on the first failed check; return a bare boolean. |
| **Pros** | Callers see *every* problem at once; validation can be advisory in exploratory contexts and strict in production ones; the report is loggable and serializable. |
| **Cons** | Callers must remember to check it — a silently ignored report is worse than an exception. |
| **Reason for selection** | Fixing data problems one exception at a time is miserable. The caller, not the validator, knows whether a warning is fatal. |
| **Impact** | Day 9's API can validate a request and return all problems in one 422 response. |

## D5 — Seed 42, fixed

| Field | Detail |
|---|---|
| **Alternatives** | Random seed per run; configurable seed. |
| **Pros** | Byte-identical regeneration; the shapes quoted in documentation stay true; bug reports are reproducible. |
| **Cons** | One realization of the data-generating process; results could be mildly seed-specific. |
| **Reason for selection** | Reproducibility (NFR-1) outweighs seed diversity for a portfolio project. |
| **Impact** | Every documented number in this repository is verifiable by re-running the generator. |

## D6 — Commit the sample dataset, gitignore the full one

| Field | Detail |
|---|---|
| **Alternatives** | Commit everything; commit nothing; Git LFS. |
| **Pros** | Repository stays small; tests and CI have data with no generation step; the full dataset is one command away. |
| **Cons** | A fresh clone cannot train without running the generator — and the sample has no positives (K-1). |
| **Reason for selection** | 5 GB does not belong in git, and LFS adds setup friction for anyone cloning. |
| **Impact** | CI runs against `data/sample/`; the installation instructions include the regeneration step. |

---

# Remaining Tasks

None from Day 2 — all objectives met.

Carried forward as a *known limitation*, not a task: **K-1**, the sample dataset's absence
of failure events.

---

# Next Day Plan

**Day 3 — Feature Engineering & Preprocessing**

1. Build `DataPreprocessor` — merge all 5 tables on `(datetime, machine_id)`.
2. Engineer rolling statistics (mean/std over 3 h / 12 h / 24 h), lag features (1 h / 6 h /
   24 h), and change features (`current − lag`), **per machine**.
3. Aggregate error counts and compute hours-since-last-maintenance per component.
4. Create binary labels: 1 if the machine fails within the next 24 hours.
5. Perform a **temporal** train/test split — Jan–Oct train, Oct–Dec test. Never random:
   with lag features, a random split puts hour `t` in train and `t+1` in test, and the
   reported metrics become fiction.
6. Fit `StandardScaler` on **training data only**, then transform both splits.
7. Build 24-step sliding windows, **never crossing a `machine_id` boundary**.
8. Save `.npy` arrays, the fitted scaler, and the ordered feature list.
9. Write tests asserting each anti-leakage invariant.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~17% (2 of 12 days) |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` ~65% (ingestion + validation done, preprocessing pending) · rest 0% |
| **Technical debt** | None |
| **Known risks** | Class imbalance is now quantified and is the project's dominant modelling risk (R-7) |
| **Immediate priorities** | Feature engineering and leak-free LSTM tensors |

---

# Files Created

```
scripts/generate_data.py
scripts/eda_analysis.py
src/data/ingestion.py
src/data/validation.py
tests/unit/test_data_pipeline.py
data/raw/telemetry.csv        (gitignored, 876,000 rows)
data/raw/machines.csv         (gitignored, 100 rows)
data/raw/errors.csv           (gitignored, 5,386 rows)
data/raw/maintenance.csv      (gitignored, 1,698 rows)
data/raw/failures.csv         (gitignored, 47 rows)
data/sample/telemetry.csv     (committed, 7,200 rows)
data/sample/machines.csv      (committed, 10 rows)
data/sample/errors.csv        (committed, 53 rows)
data/sample/maintenance.csv   (committed, 18 rows)
data/sample/failures.csv      (committed, 0 rows)
```

# Files Modified

```
src/data/__init__.py     — export DataIngestion, DataValidator
README.md                — dataset section
```

# Files Deleted

None.

# Models Generated

None.

# Checkpoints Generated

None.

# Reports Generated

EDA console report across 8 dimensions (not persisted to a file; regenerate with
`python scripts/eda_analysis.py`).

# Logs Generated

`logs/app_2026-08-20.log` (53 KB) — generation, ingestion, validation, and EDA runs.

# Screenshots

None recorded.

# References

- [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) — the schema this generator models
- [scikit-learn: imbalanced classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Pandas documentation on `merge_asof` and time-based joins

---

# Final Summary

Day 2 turned an empty skeleton into a project with data. The generator produces 883,231
rows across five related tables from a fixed seed, with degradation physics designed
specifically to make the prediction problem non-trivial: a gradual 48-hour ramp rather
than a step change, per-machine offsets, daily periodicity, and age-correlated noise.

`DataIngestion` and `DataValidator` gave the project a trustworthy front door, with the
deliberate choice that validation *reports* rather than *raises* so callers can see every
problem at once.

The day's most consequential output was not code but a number. The EDA established a
1:18,638 raw imbalance — and with it, the decision that accuracy would never be reported
in this project. That single finding determined the loss weighting, the metric set, the
early-stopping criterion, and the evaluation strategy for every day that followed.

It also surfaced K-1: a sample dataset too small to contain any failure, a limitation
accepted rather than papered over, and one that shaped how the model tests were later
written.

Ending state: 41 passing tests, commit `755e95c`, and a dataset ready to be turned into
tensors.
