# Day 6 Summary

| Field | Value |
|---|---|
| **Objective** | Build the inference pipeline: turn raw sensor tables into per-machine failure predictions, without the caller touching a tensor. |
| **Expected outcome** | A `Predictor` that reuses the training feature logic, verifies its artifacts agree, emits machine-level predictions with risk bands, and is proven equivalent to the training path. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M6 — Prediction pipeline |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Repository state** | Clean working tree, everything pushed |
| **Git commit** | `b13d84b` — "docs: record Day 5 results and update the plan, README, and index" |
| **Existing models** | `lstm_predictive_maintenance.keras`, best weights from epoch 15, F1 0.8949 |
| **Existing datasets** | Three-way split: 567,000 train / 129,000 val / 172,800 test |
| **Known issues** | K-1 (sample dataset has no failures) |
| **Tests** | 90 passing, flake8 clean |
| **Technical debt** | TD-4 (`handoff.md` overlap), TD-7 (no integration tests) |

`src/prediction/` was an empty scaffold. The model existed but nothing could
*use* it: scoring required hand-loading tensors that had already been
preprocessed, which is not what a plant has.

Carried forward from Day 5: `create_sequences()` discarded `machine_id`, so a
scored sequence could not be traced to a machine. That blocked per-machine
error analysis and would have blocked Day 7 entirely — a report cannot say
which machine is at risk if the pipeline has forgotten.

---

# Tasks Planned

### T1 — Sequence index

| Field | Detail |
|---|---|
| **Purpose** | A prediction that cannot name its machine is not actionable. |
| **Files affected** | `src/data/preprocessing.py` |
| **Dependencies** | none |
| **Priority** | P0 — blocks everything else |
| **Expected outcome** | `create_sequences(..., return_index=True)` returns machine_id and timestamp per window. |

### T2 — `Predictor`

| Field | Detail |
|---|---|
| **Purpose** | The boundary between ML artifacts and application. Everything above it talks machines and probabilities; nothing above it imports TensorFlow. |
| **Files affected** | `src/prediction/predictor.py`, `src/prediction/__init__.py` |
| **Dependencies** | T1 |
| **Priority** | P0 |
| **Expected outcome** | Raw tables in, ranked predictions out, reusing `DataPreprocessor`. |

### T3 — Artifact contract verification

| Field | Detail |
|---|---|
| **Purpose** | Risk R-6. Model, scaler, and feature list are produced by two scripts and loaded independently; nothing structurally prevents pairing mismatched ones, and the failure is silent. |
| **Files affected** | `src/prediction/predictor.py` |
| **Dependencies** | T2 |
| **Priority** | P0 |
| **Expected outcome** | Refuse to start on any mismatch. |

### T4 — Risk banding

| Field | Detail |
|---|---|
| **Purpose** | "0.73" is not an instruction. |
| **Files affected** | `config/settings.py`, `src/prediction/predictor.py` |
| **Dependencies** | T2 |
| **Priority** | P1 |
| **Expected outcome** | low / medium / high / critical, with `high` starting exactly at the alert threshold. |

### T5 — CLI

| Field | Detail |
|---|---|
| **Purpose** | Reachable without writing Python; the thing Day 9's API will wrap. |
| **Files affected** | `scripts/predict.py` |
| **Dependencies** | T2 |
| **Priority** | P1 |
| **Expected outcome** | Fleet view, single machine as JSON, alerts-only filter. |

### T6 — Integration tests (starts TD-7)

| Field | Detail |
|---|---|
| **Purpose** | Prove training and serving agree numerically. Unit tests cannot catch skew — both paths would pass them independently. |
| **Files affected** | `tests/integration/`, `pyproject.toml`, `Makefile` |
| **Dependencies** | T2 |
| **Priority** | P0 |
| **Expected outcome** | Parity asserted over the full test set, excluded from the fast suite. |

### T7 — Per-machine error analysis

| Field | Detail |
|---|---|
| **Purpose** | Day 5 left this blocked on T1. Which machines produce the 17 misses and 26 false alarms? |
| **Files affected** | analysis; `src/models/evaluator.py` |
| **Dependencies** | T1, T2 |
| **Priority** | P1 |
| **Expected outcome** | An answer, and any metric worth keeping. |

---

# Work Completed

## T1 — Sequence index ✅

`create_sequences()` gained `return_index` and `require_labels`. With
`return_index=True` it returns a third value: a DataFrame carrying the
`machine_id` and `datetime` of each window's **final** timestep — the moment
the prediction is about. `require_labels=False` supports inference, where no
`label` column exists.

Both are opt-in, so the eight existing call sites were untouched.

## T2 — `Predictor` ✅

```
Raw tables -> merge_tables -> engineer_features -> apply_scaler
           -> create_sequences -> model -> probability -> risk band
```

It **reuses `DataPreprocessor`** rather than reimplementing feature logic. That
is the entire defence against training/serving skew: two implementations of 63
engineered features will drift, and when they do the model keeps returning
well-formed probabilities that are quietly wrong — no exception, no shape
error, every unit test still green.

API:

| Method | Returns |
|---|---|
| `predict(dataset, latest_only=False)` | DataFrame: machine_id, datetime, failure_probability, risk_level, will_fail — sorted most-urgent first |
| `predict_machine(dataset, machine_id)` | One JSON-serialisable record, the shape the API and report generator consume |
| `predict_sequences(X)` | Probabilities for pre-built tensors |

A `failures` table is accepted and **ignored**. At inference the future is what
we predict, not something we read — and a test asserts that passing one does
not change any probability, so the pipeline cannot leak labels into serving.

## T3 — Contract verification ✅

Checked at load, raising rather than warning:

| Check | Catches |
|---|---|
| model input features == `len(feature_columns)` | Model paired with a different preprocessing run |
| `scaler.n_features_in_` == `len(feature_columns)` | Scaler from a different run |
| model `sequence_length` == configured | Window length drift |

A predictor that refuses to start is vastly preferable to one serving confident
nonsense.

## T4 — Risk banding ✅

Thresholds moved into `config/settings.py`: `PREDICTION_THRESHOLD = 0.6678`
(Day 5's validation sweep, not a guess and not 0.5), plus band boundaries.

`RISK_BAND_HIGH` is **defined equal to** the alert threshold, so "high or above"
means exactly "the model is alerting". If those two drifted apart a dashboard
could show `medium` for a machine the API had flagged. A test asserts the
equality rather than trusting the comment.

## T5 — CLI ✅

```bash
python scripts/predict.py                          # current fleet status
python scripts/predict.py --machine 47             # one machine, JSON
python scripts/predict.py --alerts-only -o out.csv # only what needs action
```

## T6 — Integration tests ✅ (TD-7 started)

`tests/integration/test_training_serving_parity.py` — 4 tests, marked
`integration`, excluded from the default run (they score 876,000 rows and take
7½ minutes). `make test-integration` runs them; `make test-all` runs everything.

Skips cleanly when the gitignored dataset is absent, so CI on a fresh clone
does not fail on missing data.

## T7 — Error analysis ✅ — and it changed the headline metric

See **Training Progress** below. The short version: the model's real
operational performance is considerably better than the sequence-level metrics
suggested, and the reason is a measurement artifact rather than a modelling
improvement.

---

# Code Changes

## `src/prediction/predictor.py` — created (~300 lines)

| Field | Detail |
|---|---|
| **Purpose** | The inference boundary. |
| **Classes added** | `Predictor` |
| **Functions added** | `_native()` — unwraps numpy scalars for JSON |
| **Breaking changes** | None (new). |

## `src/data/preprocessing.py`

`create_sequences()` gained `return_index` and `require_labels`, both defaulting
to previous behaviour.

## `src/models/evaluator.py`

`event_level_recall()` — recall over failure *events* rather than hours. See
Design Decisions D3.

## `config/settings.py`

`PREDICTION_THRESHOLD`, `RISK_BAND_MEDIUM/HIGH/CRITICAL`, and `model_path` /
`scaler_path` / `feature_columns_path` properties so no caller composes an
artifact path by hand.

## `scripts/predict.py` — created

## `pyproject.toml`, `Makefile`

Default test run excludes `integration`; `test-integration` and `test-all`
targets added.

---

# Training Progress

No training occurred. Day 6 consumed the Day 5 model unchanged. What follows is
measurement.

## Training/serving parity — Risk R-6 closed empirically

Scoring the raw CSVs through `Predictor`, and the stored `X_test.npy` through
the model directly, over all 172,800 test sequences:

| Measure | Result |
|---|---|
| max absolute difference | **2.98e-08** (float32 reduction-order noise) |
| mean absolute difference | 1.26e-11 |
| correlation | **1.0000000000** |
| **alert-decision agreement** | **100.0000%** |
| alerts raised | 209 both paths |

R-6 was previously "mitigated by design". It is now measured, and the
measurement is a test that runs on demand.

## Error analysis — errors are extremely concentrated

Of 100 machines, **10 account for every one of the 43 errors**; the top 5
account for 35 of them (81%).

| machine | missed | false alarms | model | age |
|---|---|---|---|---|
| 51 | 9 | 2 | model4 | 17 |
| 96 | 7 | 2 | model2 | 1 |
| 99 | 0 | 8 | model3 | 7 |
| 6 | 0 | 4 | model4 | 16 |
| 39 | 0 | 3 | model2 | 7 |

Age does **not** explain it: mean age of error-producing machines is 9.3 against
9.8 for the fleet. Day 2's EDA found older machines fail more often; that does
not make them harder to predict. Worth recording, because "just look at the old
ones" is the obvious wrong conclusion to draw.

## The measurement artifact — sequence recall understates the model

The 17 "missed" sequences are not 17 missed failures. Their timestamps cluster:

```
machine 51  2024-10-30  12:00-20:00   9 rows
machine 96  2024-11-13  00:00-08:00   7 rows
machine 57  2024-11-27  14:00         1 row
```

Every hour in the 24 hours before a failure carries label 1. Missing 9 of those
hours while catching the other 15 scores as 9 failures "missed" — but the
technician was warned, once, which is all that was required.

Measured over failure **events** instead:

| Metric | Value |
|---|---|
| Failure events in the test period | **8** |
| Events with at least one hour flagged | **8** |
| **Event-level recall** | **100%** |
| Sequence-level recall | 91.5% |
| Lead time (median / min / max) | **24h / 15h / 24h** |

Every failure in the test period was warned about, with a median of the full
24-hour horizon of notice.

**This is not a better model — it is a better question.** The weights are
identical to Day 5's. Sequence-level recall was measuring something real but
operationally secondary, and optimising it would have pushed toward smoothing
predictions *inside* a window rather than catching more windows.

**Sample size caveat, stated plainly:** 8 events. "100% of 8" is not "never
misses" — it is "warned in 8 of 8 cases". The confidence interval on that is
wide, and one more quarter of data could easily show a miss.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 90 → **113 passing** (23 new) |
| **Integration tests** | **4 passing** (7m27s) — TD-7 started |
| **Quality gates** | flake8 **0**, Black and isort clean |

New unit tests: 18 for `Predictor` (contract rejection, risk banding, batching,
end-to-end, JSON serialisation), 4 for `event_level_recall`, 1 more for
threshold degeneracy.

The contract tests matter more than the happy path here. Most assert that
mismatched artifacts are *rejected* — a Predictor returning wrong numbers looks
exactly like one returning right numbers.

---

# Bugs Encountered

## B1 — Data-dependent features break inference on small batches

| Field | Detail |
|---|---|
| **Description** | `predict()` failed with "4 required features are absent: model_model3, model_model4, hours_since_maint_comp4, hours_since_maint_comp3". |
| **Root cause** | One-hot `model_*` columns exist only for models present in the input, and `hours_since_maint_*` only for components appearing in the maintenance log. Score three machines that are all model1/model2 and feature engineering produces 61 columns, not 63. Not a bug in feature engineering — the normal shape of live data. |
| **Files affected** | `src/prediction/predictor.py` |
| **Solution** | `_reconcile_features()` adds absent columns using the fill value the training pipeline would have produced: 0 for one-hot dummies, **9999 for `hours_since_maint_*`**. A feature with no defensible default (a sensor column) still raises — that is a broken feed, not a sparse category. |
| **Verification** | 18 predictor tests pass, including end-to-end on a 3-machine fixture with only two models. |
| **Lessons learned** | The fill value carries meaning. `hours_since_maint_comp3 = 0` means "just serviced" — the exact opposite of the 9999 sentinel for "never serviced". Zero-filling everything would have produced confident, wrong, plausible predictions. The contract check caught this because it compares against the training feature list rather than trusting the input. |

## B2 — `numpy.int64` is not JSON-serialisable

| Field | Detail |
|---|---|
| **Description** | `scripts/predict.py --machine 3` died with "Object of type int64 is not JSON serializable". |
| **Root cause** | pandas returns numpy scalars; `json.dumps` refuses them. `predict_machine()` cast probability and boolean but not the id. |
| **Files affected** | `src/prediction/predictor.py`, `tests/unit/test_predictor.py` |
| **Solution** | `_native()` helper unwrapping numpy scalars; JSON round-trip test. |
| **Lessons learned** | Found by the CLI, not by a unit test — the tests were asserting on Python objects, where the numpy types are invisible. It would have broken Day 9's API identically. A serialisation boundary needs a test that actually serialises. |

## B3 — A test with the wrong tolerance

| Field | Detail |
|---|---|
| **Description** | `test_batching_does_not_change_results` failed with a 1.86e-6 relative difference. |
| **Root cause** | Not a code bug. Absolute difference was **2.75e-14**; relative tolerance is meaningless on probabilities near 1e-8, where a float32 reduction-order difference reads as a large relative error purely because the denominator is tiny. |
| **Solution** | Assert absolute tolerance, plus the thing that actually matters: no batch size may flip an alert decision. |
| **Lessons learned** | Choose the tolerance that matches the quantity. Loosening `rtol` to make it pass would have hidden a genuine regression later; switching to `atol` measures the right thing. |

## B4 — Event detection assumed complete hourly coverage

| Field | Detail |
|---|---|
| **Description** | `test_events_split_by_machine_and_by_time_gap` found 2 events where 3 exist. |
| **Root cause** | `event_level_recall()` detected boundaries purely by label transitions, so two warning windows for one machine merged into one event when the negative rows between them were absent. True for the full test tensor, false for any filtered subset — exactly what a caller analysing one machine would pass. |
| **Files affected** | `src/models/evaluator.py` |
| **Solution** | Boundaries also split on a time gap exceeding `max_gap_hours` (default 1, the telemetry cadence). |
| **Verification** | Re-ran on real data after the fix: still 8 events, 100% recall. |
| **Lessons learned** | The test was contrived and the bug was real. An implementation that works only on the shape of data you happen to have is a latent bug, and the fix cost three lines. |

---

# Design Decisions

## D1 — Reuse `DataPreprocessor` instead of a dedicated inference path

| Field | Detail |
|---|---|
| **Alternatives** | A leaner inference-only feature builder; a serialised feature pipeline (sklearn `Pipeline`); duplicating the logic. |
| **Pros** | One implementation, so drift is structurally impossible. Bug fixes land in both paths at once. |
| **Cons** | Inference carries training-shaped code — `create_labels` exists on the object and is simply not called; feature engineering recomputes rolling windows over the whole input rather than incrementally. Slower than a purpose-built path. |
| **Reason for selection** | Training/serving skew is the highest-consequence silent failure available here. Speed is recoverable later; a silent correctness bug may never be noticed at all. |
| **Impact** | Verified: 100% alert-decision agreement over 172,800 sequences. |

## D2 — Reconcile absent categorical features rather than rejecting them

| Field | Detail |
|---|---|
| **Alternatives** | Reject any input missing a training feature; zero-fill everything; persist the training category set and one-hot against it. |
| **Pros** | Handles the normal case (a batch not containing every machine model) without failing, while still rejecting genuinely malformed input. |
| **Cons** | The fill rules are prefix-based and must track the feature engineering. A new data-dependent family added to `engineer_features` without a matching rule will raise — loudly, which is the right failure. |
| **Reason for selection** | Rejecting would make single-machine scoring impossible, which is the main API use case. Zero-filling everything would silently invert the meaning of the maintenance sentinel. |
| **Impact** | Single-machine and small-batch prediction work; malformed input still fails fast. |

## D3 — Add event-level recall as a first-class metric

| Field | Detail |
|---|---|
| **Alternatives** | Keep reporting sequence-level metrics only; report both; deduplicate labels so each failure contributes one row. |
| **Pros** | Matches how the system is actually used — one warning per failure is the goal, not a warning every hour. Exposes that the model missed **zero** failure events while sequence recall said 91.5%. |
| **Cons** | Easier to look good on: 8 events is a small denominator, and event recall hides *how many* hours were flagged, which matters for alert fatigue. It must be reported alongside precision, never instead of it. |
| **Reason for selection** | Sequence-level recall was misdirecting attention. Optimising it rewards smoothing predictions inside a window rather than catching more windows. |
| **Impact** | Both metrics are reported. The sample-size caveat travels with the number wherever it is quoted. |

## D4 — Risk bands in settings, with `high` pinned to the alert threshold

| Field | Detail |
|---|---|
| **Alternatives** | Hardcode bands in the Predictor; derive them from quantiles of the score distribution. |
| **Pros** | One place to change; the band boundary and the alert decision cannot drift apart. |
| **Cons** | Fixed cut points do not adapt if the score distribution shifts after retraining. |
| **Reason for selection** | A dashboard showing `medium` for a machine the API flagged as an alert is a trust-destroying inconsistency, and it is the kind of bug that survives for months. |
| **Impact** | Asserted by test rather than left as a convention. |

## D5 — Integration tests excluded from the default run

| Field | Detail |
|---|---|
| **Alternatives** | Run everything by default; keep parity checks as a manual script. |
| **Pros** | `make test` stays at ~18s, so it is actually run before every change. The slow, high-value check is one command away and named. |
| **Cons** | A default run no longer proves parity — someone can commit skew and see green. |
| **Reason for selection** | A 7½-minute default suite stops being run. Day 11's CI will run `test-all` on push, which is the right place for the slow gate. |
| **Impact** | 113 unit tests in 18s; 4 integration tests on demand. |

---

# Remaining Tasks

| Item | Priority | Dependencies | Effort |
|---|---|---|---|
| Report event-level recall in `evaluate_model.py` output, not just as a library function | P1 | none | 1 h |
| Investigate machine 99 (8 of 26 false alarms) — is it a sensor characteristic or a model blind spot? | P2 | none | 2 h |
| Contributing-feature attribution, so a report can cite evidence rather than correlation | P2 | Day 7 | 3 h |
| Incremental feature engineering for live scoring (currently recomputes the whole history) | P3 | none | 3 h |
| Hyperparameter comparison — carried from Day 5 | P3 | none | 3 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | none | 1 h |

---

# Next Day Plan

**Day 7 — LangChain Setup & Report Generation**

1. `src/genai/prompts.py` — templates with a maintenance-expert system persona.
2. `src/genai/chains.py` — `report_chain`, with the provider (OpenAI / Gemini /
   Ollama) selected from settings so switching is a config change.
3. Feed the chain a `predict_machine()` record plus recent sensor context.
   The record shape is already stable and JSON-serialisable, which is why B2
   mattered.
4. **Ground the report in numbers the pipeline actually has.** The failure mode
   to design against is an LLM inventing a plausible cause; the prompt should
   supply concrete feature values and instruct the model to cite them.
5. Graceful degradation — `LLMConnectionError` must leave the prediction intact
   and only the report unavailable.
6. Tests with a mocked LLM, asserting prompt construction rather than model
   output.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~50% |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · `src/models/` 100% · **`src/prediction/` 100%** · `src/genai/` 0% · `src/api/` 0% · `dashboard/` 0% |
| **Technical debt** | TD-4 (handoff overlap) · TD-7 (integration tests started, API/GenAI coverage still pending) |
| **Known risks** | ~~R-6 training/serving skew~~ ✅ **closed — measured at 100% agreement** · R-10 LLM provider failure (Day 7) |
| **Immediate priorities** | Day 7 GenAI report generation |
| **Quality gates** | 113 unit + 4 integration tests · flake8 0 · Black/isort clean |

---

# Files Created

```
src/prediction/predictor.py                      the inference boundary
scripts/predict.py                               CLI
tests/unit/test_predictor.py                     18 tests
tests/integration/test_training_serving_parity.py  4 tests
docs/Day6.md                                     this file
```

# Files Modified

```
src/prediction/__init__.py    export Predictor
src/data/preprocessing.py     create_sequences: return_index, require_labels
src/models/evaluator.py       event_level_recall
config/settings.py            threshold, risk bands, artifact paths
tests/unit/test_model.py      +5 tests
pyproject.toml                exclude integration from the default run
Makefile                      test-integration, test-all
```

# Models Generated

None — Day 6 consumed the Day 5 model unchanged.

# Reports Generated

Error analysis and event-level recall were computed ad hoc. The metric is now a
library function; wiring it into `evaluate_model.py`'s persisted output is
listed under Remaining Tasks.

# References

- [scikit-learn: model persistence and the training/serving contract](https://scikit-learn.org/stable/model_persistence.html)
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems* — training/serving skew and the cost of duplicated feature logic
- [pandas: groupby + diff for run detection](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.diff.html)

---

# Final Summary

Day 6 built the boundary between "we have a model" and "we can use it".
`Predictor` takes the five raw tables a plant actually has and returns ranked,
banded, machine-level predictions, reusing the training feature code rather
than reimplementing it — and that reuse was then *verified* rather than
assumed: 100% alert-decision agreement across 172,800 test sequences, closing
Risk R-6 with a measurement instead of a design argument.

Three of the day's four bugs were found by writing tests, and each was
instructive. Data-dependent one-hot columns break inference on any batch that
does not contain every machine model — and the fix had to respect that
`hours_since_maint = 0` means "just serviced" while the training pipeline uses
9999 for "never serviced", so zero-filling would have inverted a feature's
meaning silently. `numpy.int64` is not JSON-serialisable, which the CLI caught
and the unit tests could not, because they never serialised anything.

The most valuable finding was not a bug at all. Per-machine analysis showed the
17 "missed" sequences were three clusters, and measuring recall over failure
*events* rather than hours gives **100% (8 of 8), with a median 24 hours of
notice** — from the identical weights Day 5 trained. Sequence-level recall was
measuring something real but operationally secondary, and optimising it would
have pushed the model toward smoothing predictions inside a warning window
instead of catching more windows. The number comes with its denominator
attached: eight events is a small sample, and "100% of 8" is a claim about
eight cases, not a property of the model.

Ending state: 113 unit tests, 4 integration tests, R-6 closed, `src/prediction/`
complete, and a prediction record whose shape Day 7 can consume directly.
