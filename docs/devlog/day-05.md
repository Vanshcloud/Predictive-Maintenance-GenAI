# Day 5 Summary

| Field | Value |
|---|---|
| **Objective** | Turn a trained model into a *characterised* model, and pay down Day 4's technical debt — above all TD-1, the validation set that was really the test set. |
| **Expected outcome** | A genuine three-way chronological split, a retrained model whose metrics are an honest generalisation estimate, an operating point chosen on validation, training-curve and PR plots, and checkpoint resume. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-23 |
| **Milestone** | M5 — Evaluation & optimization |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Repository state** | Clean working tree, everything pushed |
| **Git commit** | `f588698` — "docs: refresh commit SHAs after the history rewrite" |
| **Existing files** | Days 1–4 complete: config, utils, data pipeline, model layer |
| **Existing models** | `models/lstm_predictive_maintenance.keras` (149,825 params) |
| **Existing checkpoints** | One, from Day 4 epoch 1 |
| **Existing datasets** | `data/processed/` — two-way split only: X_train (698400, 24, 63), X_test (172800, 24, 63) |
| **Known issues** | K-1 (sample dataset has no failures) |
| **Tests** | 75 passing, flake8 clean |

**Technical debt carried in:** TD-1 (validation set *is* the test set), TD-2 (no resume),
TD-3 (no training curves), TD-6 (threshold hardcoded at 0.5), TD-4, TD-7.

Day 4's headline — AUC 0.9999, F1 0.7530 — was explicitly flagged as optimistic, because
`X_val=X_test` meant early stopping and checkpoint selection had both observed the data
they were scored on. Day 5 exists to replace that with a number worth quoting.

---

# Tasks Planned

### T1 — Three-way chronological split (TD-1)

| Field | Detail |
|---|---|
| **Purpose** | Model selection must not observe the test set. This is the difference between a metric and a claim. |
| **Files affected** | `src/data/preprocessing.py`, `scripts/run_preprocessing.py`, `scripts/train_model.py`, `tests/unit/test_preprocessing.py` |
| **Dependencies** | none |
| **Priority** | **P0 — everything else is provisional until this lands** |
| **Expected outcome** | train / val / test splits, with the test period unchanged so results stay comparable to Day 4. |

### T2 — Threshold sweep and cost-based operating point (TD-6)

| Field | Detail |
|---|---|
| **Purpose** | 0.5 is inherited from balanced problems. At a 1:864 base rate it is almost certainly wrong, and the right point depends on what a miss costs relative to a false alarm. |
| **Files affected** | `src/models/evaluator.py`, `scripts/evaluate_model.py`, `tests/unit/test_model.py` |
| **Dependencies** | T1 |
| **Priority** | P0 |
| **Expected outcome** | Full PR curve, best-F1 and lowest-cost points, chosen on validation and applied once to test. |

### T3 — Training curves and PR plots (TD-3)

| Field | Detail |
|---|---|
| **Purpose** | `training_history.json` is data nobody reads. A plot shows overfitting at a glance. |
| **Files affected** | `scripts/evaluate_model.py` |
| **Dependencies** | T1 |
| **Priority** | P1 |
| **Expected outcome** | `models/training_curves.png`, `models/pr_curve.png`. |

### T4 — Change the early-stopping monitor

| Field | Detail |
|---|---|
| **Purpose** | Emerged *from* T3's plots rather than being planned: `val_auc` saturates and selects on noise. |
| **Files affected** | `src/models/trainer.py`, `scripts/train_model.py`, `tests/unit/test_model.py` |
| **Dependencies** | T3 |
| **Priority** | P1 |
| **Expected outcome** | `monitor` parameter defaulting to `val_f1`, with the old behaviour still available. |

### T5 — Checkpoint resume (TD-2)

| Field | Detail |
|---|---|
| **Purpose** | A 25-minute run that dies at minute 24 currently loses everything but the best `.keras`. |
| **Files affected** | `src/models/trainer.py`, `scripts/train_model.py`, `tests/unit/test_model.py` |
| **Dependencies** | none |
| **Priority** | P1 |
| **Expected outcome** | `--resume` restores epoch, best score, and history. |

### T6 — Per-machine error analysis

| Field | Detail |
|---|---|
| **Purpose** | Day 7's report generator must *explain* a prediction, which needs to know which machine a sequence belongs to. The `.npy` tensors currently discard that. |
| **Files affected** | `src/data/preprocessing.py`, `scripts/evaluate_model.py` |
| **Dependencies** | T1 |
| **Priority** | P2 |
| **Expected outcome** | Sequence-level machine/timestamp index, and an analysis of which machines produce the misses. |

---

# Work Completed

## T1 — Three-way chronological split ✅

`DataPreprocessor` gained a `val_ratio` (default 0.15). `temporal_split()` now returns
`(train_df, val_df, test_df)`:

```
|<------- train ------->|<-- val -->|<-- test -->|
 1 - val_ratio - test_ratio  val_ratio  test_ratio
```

**The critical design choice: the test boundary is defined by `test_ratio` alone.**
Introducing validation shrinks the *training* period and leaves the test period exactly
where Day 4 had it. Without that property the two days' numbers would not be comparable,
and the improvement reported below could not be attributed to anything. There is a test
asserting it (`test_val_split_does_not_move_the_test_boundary`) so a future tweak cannot
silently move the goalposts.

| Split | Sequences | Positives | Rate |
|---|---|---|---|
| Train | 567,000 | 800 | 1:709 |
| **Validation** | **129,000** | **175** | 1:737 |
| Test | 172,800 | 200 | 1:864 — **unchanged from Day 4** |

Also added `apply_scaler()`, which transforms a split with the **already-fitted** scaler.
`normalize()` fits, and must only ever see training data; everything else — validation
here, live sensor data at inference — is transformed with training-time statistics. A test
reproduces the transform by hand from the fitted scaler to prove no refitting happens,
because refitting per split is the classic training/serving skew bug: it silently degrades
predictions while every existing test still passes.

`train_model.py` now monitors `X_val`. If `X_val.npy` is missing it logs a **loud warning**
rather than silently falling back to the test set the way Day 4 did.

## T2 — Threshold sweep ✅

`ModelEvaluator` gained `predict_proba()` and `sweep_thresholds()`. The sweep walks every
point on the precision-recall curve and reports two operating points:

- **best F1** — the balanced choice.
- **lowest cost** — under an explicit `cost_fn : cost_fp` ratio, defaulting to 100:1.
  A missed failure stops a production line; a false alarm costs one inspection. Those are
  not equally bad, and a single scalar should not pretend they are. The default is a
  placeholder to be replaced with the plant's real numbers.

`scripts/evaluate_model.py` enforces the ordering that matters: sweep on **validation**,
choose there, then score **test once** at that threshold. Choosing a threshold on the test
set and then reporting test metrics at it is the same error as early-stopping on the test
set — the number stops estimating generalisation and starts describing its own tuning set.

## T3 — Plots ✅

`models/training_curves.png` (2×2: loss, AUC, precision, recall — train vs validation) and
`models/pr_curve.png` (PR curve with both operating points marked, plus cost-vs-threshold
on a log scale). Both committed as evidence, matching the `metrics.json` rationale.

## T4 — Early-stopping monitor ✅ — validated by Run B

`ModelTrainer.train()` gained `monitor`, defaulting to **`val_f1`**. `_run_validation()`
now also derives F1 (Keras has no streaming F1 metric). Unknown monitor names raise
`ModelTrainingError` rather than silently falling back.

This change was not planned — it came out of reading T3's plots, and the evidence is in
*Design Decisions* below.

## T5 — Checkpoint resume ✅

`train(resume=True)` restores epoch number, best score, and history from a
`<checkpoint>.state.json` written beside the `.keras`. State is written **only when the
checkpoint improves**, which keeps the two files describing the same moment: the weights
on disk and the recorded epoch always agree. A resumed run therefore picks up after the
best epoch, not the last one — slightly wasteful if the best was early, but never
incoherent.

Deliberately **not** persisted: Adam's optimizer slot variables. A resumed run re-warms its
moment estimates over a few batches, so it is not bit-identical to an uninterrupted one.
Serialising every slot variable was not worth it for a 25-minute run.

`models/*.state.json` added to `.gitignore` — it is useless without the checkpoint it
describes.

## T6 — Per-machine error analysis 🔒

Not started. Blocked on a design question rather than effort: `create_sequences()` discards
`machine_id`, so the `.npy` tensors cannot be traced back to a machine. The fix is to have
the pipeline emit a sequence-level index alongside the tensors — which Day 6 inference and
Day 7 report generation both need anyway, so it belongs there rather than being bolted on
here.

---

# Code Changes

## `src/data/preprocessing.py`

| Field | Detail |
|---|---|
| **Purpose** | Produce a leak-free three-way split. |
| **Important changes** | `val_ratio` on `__init__`; `temporal_split()` returns 3 frames; new `apply_scaler()`; `run_pipeline()` builds and returns `X_val`/`y_val`; `_save_artifacts()` persists them. |
| **Breaking changes** | **Yes** — `temporal_split()` returns 3 values, not 2. All 8 internal call sites updated. |
| **Imports added** | `DataPreprocessingError` (re-added — it was removed as unused in Day 4's lint pass and is now genuinely used) |
| **Functions added** | `apply_scaler()` |

## `src/models/trainer.py`

| Field | Detail |
|---|---|
| **Purpose** | Selection on a metric that moves; survivable long runs. |
| **Important changes** | `monitor` parameter (default `val_f1`); `resume` parameter; `_save_state()` / `_load_state()` / `_state_path()`; validation now derives F1; selection direction inferred from the metric name. |
| **Breaking changes** | Default selection metric changed from `val_auc` to `val_f1`. `_save_checkpoint()` takes the monitored metric name instead of a bool. |
| **Imports added** | `json` |

## `src/models/evaluator.py`

| Field | Detail |
|---|---|
| **Important changes** | `predict_proba()` (public); `sweep_thresholds()` with cost weighting. |
| **Imports added** | `average_precision_score`, `precision_recall_curve` |

## `scripts/evaluate_model.py` — created

Threshold selection on validation, single test scoring, and both figures.

## `scripts/train_model.py`, `scripts/run_preprocessing.py`

`--val-ratio`, `--monitor`, `--resume`; `load_data()` returns a dict and picks up the
validation split when present.

## `.gitignore`

`models/*.state.json`.

---

# Training Progress

## Run A — clean split, `monitor=val_auc` (the Day 4 default)

| Field | Value |
|---|---|
| **Dataset** | 567,000 train / 129,000 val / 172,800 test sequences |
| **Epochs** | 30 requested, **10 run** (early stopping) |
| **Batch size** | 256 · **LR** 0.001 → 0.0005 at epoch 8 |
| **Duration** | 14:01 → 14:25 ≈ **24 minutes** |
| **Best epoch** | 5 (`val_auc` 1.0000) |

Per-epoch validation:

| Epoch | val_loss | val_auc | val_precision | val_recall |
|---|---|---|---|---|
| 1 | 0.0102 | 0.9999 | 0.2140 | 0.9943 |
| 2 | 0.0021 | 0.9999 | 0.7511 | 0.9657 |
| 3 | 0.0025 | 0.9999 | 0.7056 | 1.0000 |
| 4 | 0.0097 | 0.9998 | 0.2926 | 1.0000 |
| **5** | **0.0012** | **1.0000** | **0.8065** | **1.0000** |
| 6 | 0.0016 | 0.9999 | 0.7773 | 0.9771 |
| 7 | 0.0286 | 0.9994 | 0.1303 | 1.0000 |
| 8 | 0.0041 | 0.9998 | 0.5167 | 0.9714 |
| 9 | 0.0016 | 0.9999 | 0.7814 | 0.9600 |
| 10 | 0.0015 | 0.9999 | 0.7972 | 0.9657 |

### Threshold sweep on validation — 125,263 candidates, AP = 0.9825

| Operating point | Threshold | Precision | Recall | FN | FP |
|---|---|---|---|---|---|
| best F1 | 0.9605 | 0.9189 | 0.9714 | 5 | 15 |
| **lowest cost (100:1)** | **0.6359** | 0.8140 | **1.0000** | **0** | 40 |

### Final test — scored once, at the validation-chosen threshold

| Metric | Day 4 (peeked) | t = 0.5 | **t = 0.6359 (chosen)** |
|---|---|---|---|
| ROC-AUC | 0.9999 | 0.9999 | 0.9999 |
| Precision | 0.6258 | 0.7983 | **0.8043** |
| Recall | 0.9450 | 0.9300 | 0.9250 |
| **F1** | **0.7530** | 0.8591 | **0.8605** |
| Missed failures | 11 | 14 | 15 |
| False alarms | 113 | 47 | **45** |

**The honest split produced a better model than the peeked one** — F1 up 14%, precision up
28%, false alarms down 60% — on *less* training data (567K vs 698K sequences).

That is counterintuitive until you look at why. Day 4 monitored `val_auc` against the test
set, where AUC hit 0.9998 in epoch 1; nothing beat it, so early stopping restored epoch 1's
weights and discarded five further epochs of learning. This run had a real validation set,
epoch 5 reached a clean 1.0000, and epoch 5's model is far better calibrated — validation
precision 0.8065 versus epoch 1's 0.2140 at comparable recall. The improvement is not from
the honest split *per se*; it is that the honest split let early stopping keep a better
epoch.

## Run B — same split, `monitor=val_f1` ✅

| Field | Value |
|---|---|
| **Epochs** | 30 requested, **20 run** (early stopping) |
| **Best epoch** | **15** (`val_f1` 0.9359) |
| **Duration** | 14:44 → 15:31 ≈ **47 minutes** |

`val_f1` climbed steadily and kept finding new bests long after `val_auc` had peaked:

| Epoch | val_auc | val_precision | val_recall | **val_f1** |
|---|---|---|---|---|
| 1 | 0.9998 | 0.6880 | 0.9829 | 0.8094 ✔ |
| 6 | 0.9999 | 0.7576 | 1.0000 | 0.8621 ✔ |
| 7 | 0.9999 | 0.7900 | 0.9886 | 0.8782 ✔ |
| **8** | **1.0000** ← val_auc peak | 0.8131 | 0.9943 | 0.8946 ✔ |
| 10 | 0.9857 | 0.8564 | 0.9543 | 0.9027 ✔ |
| 14 | 0.9943 | 0.9171 | 0.9486 | 0.9326 ✔ |
| **15** | 0.9943 | **0.9130** | 0.9600 | **0.9359** ✔ best |
| 19 | 0.9942 | 0.9261 | 0.9314 | 0.9288 |
| 20 | 0.9857 | 0.9249 | 0.9143 | 0.9195 |

**This is the decisive evidence for D3.** `val_auc` peaked at epoch 8 and then *fell* — so
an AUC monitor stops there and keeps epoch 8. But epoch 15 is the better model by every
measure that matters: validation precision 0.913 vs 0.813, F1 0.936 vs 0.895. AUC was
reporting a saturated quantity while precision improved 12 points underneath it.

### Cumulative effect across three runs

| | Day 4 (peeked, val_auc) | Run A (clean, val_auc) | **Run B (clean, val_f1)** |
|---|---|---|---|
| Test precision | 0.6258 | 0.8043 | **0.8756** |
| Test recall | 0.9450 | 0.9250 | 0.9150 |
| **Test F1** | 0.7530 | 0.8605 | **0.8949** |
| False alarms | 113 | 45 | **26** |
| Missed failures | 11 | 15 | 17 |

Final test metrics at the deployed threshold (t=0.6678, chosen on validation):
**AUC 0.9997 · precision 0.8756 · recall 0.9150 · F1 0.8949** — 183 of 200 failures caught,
26 false alarms across 172,800 sequences.

## The cost-based threshold default was wrong, and was reverted

Run B's sweep chose **t=0.0003** as cost-optimal. Applied to test it was actively harmful:

| Threshold source | Test F1 | Precision | Recall | False alarms |
|---|---|---|---|---|
| cost-optimal, t=0.0003 | **0.7366** | 0.5957 | 0.9650 | 126 |
| default t=0.5 | 0.8889 | 0.8598 | 0.9200 | 30 |
| **F1-optimal, t=0.6678** | **0.8949** | 0.8756 | 0.9150 | **26** |

**Root cause.** With `cost_fn` 100× `cost_fp`, the objective collapses to "reach recall 1.0
at any price". On 175 validation positives the cheapest route there is a threshold in the
noise floor — 99 false positives on validation buys FN=0, which beats any balanced point
on the cost function. That threshold is fitted to the tail of a small sample and does not
transfer: it cost **15 points of test F1**.

This contradicts D2 as originally written, which argued cost-weighting was the more
principled default. It is more principled *in theory*; at this sample size it is fragile.
Three changes:

1. `--select-by` now defaults to **`f1`**. `cost` remains available for when real cost
   numbers and more positives exist.
2. `sweep_thresholds()` returns **`lowest_cost_is_degenerate`** (threshold < 0.01 or
   precision < 0.7) and logs a warning, so this cannot pass silently again.
3. `test_degenerate_cost_optimum_is_flagged` asserts the flag fires on an unseparable
   problem and stays off on a clean one.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 75 → **90 passing** (15 new) |
| **Integration tests** | None yet — Day 9 |
| **Quality gates** | flake8 **0 issues**, Black and isort clean |

New tests:

| Test | What it protects |
|---|---|
| `test_val_sits_between_train_and_test` | Strict chronological order; no leakage between any pair |
| `test_splits_are_disjoint_and_complete` | Every row lands in exactly one split |
| `test_val_split_does_not_move_the_test_boundary` | Adding validation shrinks *train*, never *test* — the comparability guarantee |
| `test_apply_scaler_uses_training_statistics_not_its_own` | Reproduces the transform by hand; catches a refit |
| `test_apply_scaler_before_fitting_raises` | Ordering error fails loudly |
| `test_apply_scaler_on_empty_frame_is_a_noop` | `val_ratio=0` path |
| `test_sweep_finds_the_perfect_split_on_separable_scores` | Sweep arithmetic against a known answer |
| `test_cost_ratio_shifts_the_operating_point_toward_recall` | Raising miss cost moves the threshold **down** |
| `test_sweep_reports_consistent_confusion_counts` | FN + TP = positives at every reported point |
| `test_predict_proba_returns_flat_probabilities` | Shape and range contract |
| `test_monitor_defaults_to_val_f1_and_history_records_it` | Default selection metric is recorded |
| `test_unknown_monitor_is_rejected` | A typo fails loudly instead of silently defaulting |
| `test_resume_continues_from_saved_epoch` | History carries forward; run reaches the requested total |
| `test_resume_without_state_starts_fresh` | `--resume` on a clean directory trains normally |
| `test_degenerate_cost_optimum_is_flagged` | A tail-fitted cost optimum is flagged, and a clean one is not |

---

# Bugs Encountered

## B1 — A `str.replace()` that silently did nothing

| Field | Detail |
|---|---|
| **Description** | Adding `val_f1` to the history dict appeared to succeed, then training died with `KeyError: 'val_f1'`. |
| **Root cause** | The edit targeted a single-line `history.update({...})` that Black had already reformatted into a multi-line block. `str.replace()` found no match and returned the string unchanged — **and I had not asserted the match**, so the no-op was invisible. |
| **Files affected** | `src/models/trainer.py` |
| **Solution** | Re-applied against the current formatting, with `assert old in s` before every replacement. |
| **Verification** | 90 tests pass. |
| **Lessons learned** | Every scripted edit needs an assertion that it matched. This is the second time in this project that an unasserted text transform caused a bug — Day 4's regex corrupted three files the same way. A silent no-op is worse than a crash: it produces code that looks edited and isn't. |

## B2 — A test that asserted the wrong contract

| Field | Detail |
|---|---|
| **Description** | `test_resume_continues_from_saved_epoch` failed, asserting that a resumed run's history matched the original run's for the first two epochs. |
| **Root cause** | Not a code bug. State is written only when the checkpoint *improves*, so the resume point was epoch 1, not epoch 2 — epoch 2 was recomputed. The implementation was right; the test encoded an assumption I had not thought through. |
| **Files affected** | `tests/unit/test_model.py` |
| **Solution** | Test now reads the saved epoch from the state file and asserts the real contract: epochs up to the checkpoint carry forward verbatim, and the run continues to the requested total. |
| **Verification** | Passes, and now documents the resume semantics rather than a guess about them. |
| **Lessons learned** | When a test fails, the test is as likely to be wrong as the code. Fixing the code to satisfy a wrong assertion would have made resume incoherent — weights from one epoch, history from another. |

---

# Design Decisions

## D1 — Carve validation out of *training*, not out of test

| Field | Detail |
|---|---|
| **Alternatives** | Split the existing test period into val + test; re-split everything by fresh ratios; use the last N% of train. |
| **Pros** | The test period stays byte-identical in size and dates, so Day 4 and Day 5 numbers are directly comparable and the improvement is attributable. |
| **Cons** | Training loses 131,400 sequences (~19%) and 157 positives. |
| **Reason for selection** | Comparability across runs is worth more than the extra training data, especially when the model already converges in ~5 epochs. Losing the ability to compare would have made the whole exercise unfalsifiable. |
| **Impact** | Enforced by test. The improvement reported above means something because of this choice. |

## D2 — Select the operating point by cost, not by F1

| Field | Detail |
|---|---|
| **Alternatives** | Keep 0.5; maximise F1; maximise recall subject to a precision floor. |
| **Pros** | F1 implicitly asserts that precision and recall are equally valuable. In predictive maintenance they demonstrably are not — a missed failure stops a line, a false alarm costs one inspection. |
| **Cons** | Requires a cost ratio nobody has measured; the 100:1 default is invented. |
| **Reason for selection** | An explicit, wrong-but-visible assumption beats an implicit one. `--cost-fn` / `--cost-fp` put the assumption at the call site where a plant engineer can correct it. |
| **OUTCOME — this decision was reverted** | Cost weighting is right in principle and wrong at this sample size. On Run B it selected t=0.0003, costing 15 points of test F1 (0.7366 vs 0.8949). With 175 validation positives and a 100:1 ratio, minimising cost means "reach recall 1.0 however you can", and the cheapest route is a noise-floor threshold that does not transfer. Default is now `f1`; `cost` stays available, and a degeneracy flag now catches the failure mode. |
| **Impact** | Deployed t=0.6678 (F1-optimal). All operating points are still reported, so the choice can be re-litigated without retraining once real cost figures exist. |

## D3 — Change the early-stopping monitor from `val_auc` to `val_f1`

| Field | Detail |
|---|---|
| **Alternatives** | Keep `val_auc`; use `val_loss`; use average precision. |
| **Evidence** | Over Run A's ten epochs, `val_auc` stayed within **0.9991–1.0000** — four decimal places of noise — while validation precision swung **0.13 → 0.81** and `val_loss` varied 24×. AUC reported that nothing was happening while the model changed enormously. |
| **Pros** | F1 responds to exactly the precision/recall trade that the imbalance makes volatile. |
| **Cons** | F1 is threshold-dependent (evaluated at 0.5), so it selects for a model calibrated near that cut point — which is not necessarily the one that ranks best. Keras has no streaming F1, so it is derived. |
| **Reason for selection** | Selecting on a saturated metric is selecting on noise. Day 4's model was chosen that way, and it was measurably worse. |
| **Impact** | Default changed; `--monitor` keeps every option available. Run B tests whether the reasoning survives contact with data. |

## D4 — Write resume state only when the checkpoint improves

| Field | Detail |
|---|---|
| **Alternatives** | Save state every epoch; save weights every epoch too. |
| **Pros** | The `.keras` and the `.state.json` always describe the same epoch. Saving state every epoch while saving weights only on improvement would let them disagree — resuming at epoch 11 with epoch 5's weights and epoch 10's history. |
| **Cons** | Epochs between the best one and the crash are recomputed. |
| **Reason for selection** | Coherence over efficiency. A resume that silently mixes two models' state is worse than one that redoes four epochs. |
| **Impact** | Documented in the resume test, which reads the saved epoch rather than assuming it. |

## D5 — Do not persist optimizer state

| Field | Detail |
|---|---|
| **Alternatives** | Serialise Adam's slot variables. |
| **Pros of skipping** | Far less code; the `.state.json` stays human-readable. |
| **Cons** | A resumed run is not bit-identical — Adam re-warms its moment estimates over a few batches. |
| **Reason for selection** | Runs are ~25 minutes and converge in ~5 epochs. The divergence is smaller than run-to-run noise. |
| **Impact** | Stated explicitly in `_save_state`'s docstring so nobody assumes exact reproducibility across a resume. |

---

# Remaining Tasks

| Item | Priority | Dependencies | Effort |
|---|---|---|---|
| Sequence-level machine/timestamp index (unblocks T6, needed by Days 6–7) | P1 | none | 2 h |
| Per-machine error analysis of the misses | P1 | index | 2 h |
| Hyperparameter comparison (sequence length 12/24/48, LSTM widths, dropout) | P2 | Run B | 3 h |
| Re-benchmark `model.fit()` now the abseil bug is fixed — settle Day 4's D1 | P3 | none | 1 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | none | 1 h |

---

# Next Day Plan

**Day 6 — Prediction Pipeline & Inference**

1. `src/prediction/predictor.py` — load model + scaler + feature list; assert the feature
   names match `feature_columns.txt` at startup and refuse to serve on mismatch (Risk R-6).
2. Accept raw sensor rows, reuse `DataPreprocessor`'s feature logic rather than
   reimplementing it, window, and score.
3. Emit `{machine_id, failure_probability, risk_level, contributing_features}` — the
   sequence index from Day 5's remaining work is what makes `machine_id` possible.
4. Risk banding from the chosen threshold, not a fresh guess.
5. Batch scoring for a whole fleet.
6. Tests: identical predictions via the pipeline and via a direct model call; latency
   under 100 ms.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~42% |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · `src/models/` 100% · `src/prediction/` 0% · `src/genai/` 0% · `src/api/` 0% · `dashboard/` 0% |
| **Technical debt** | ~~TD-1~~ ✅ · ~~TD-2~~ ✅ · ~~TD-3~~ ✅ · ~~TD-6~~ ✅ · TD-4 (handoff overlap) · TD-7 (no integration tests) |
| **Known risks** | ~~R-8 optimistic metrics~~ ✅ **closed** · R-6 training/serving skew (enforcement lands Day 6) |
| **Immediate priorities** | Finish Run B; build the sequence index that Days 6–7 depend on |
| **Quality gates** | 90 tests · flake8 0 · Black/isort clean |

---

# Files Created

```
scripts/evaluate_model.py         threshold sweep, plots, single test scoring
day-05.md                      this file
models/evaluation_report.json     committed — threshold selection + test metrics
models/pr_curve.png               committed — PR + cost curves
models/training_curves.png        committed — train vs validation, 4 panels
data/processed/X_val.npy          gitignored, 744 MB
data/processed/y_val.npy          gitignored
```

# Files Modified

```
src/data/preprocessing.py     3-way split, apply_scaler
src/models/trainer.py         monitor param, val_f1, resume
src/models/evaluator.py       predict_proba, sweep_thresholds
scripts/train_model.py        --monitor, --resume, val split wiring
scripts/run_preprocessing.py  --val-ratio
tests/unit/test_preprocessing.py   +6 tests, 8 call sites updated
tests/unit/test_model.py           +8 tests
.gitignore                    models/*.state.json
models/metrics.json, models/training_history.json   regenerated
```

# Models Generated

`models/lstm_predictive_maintenance.keras` — retrained on the clean split, best weights
from epoch 5 of 10.

# Reports Generated

`models/evaluation_report.json` — validation threshold sweep (AP 0.9825, both operating
points) plus test metrics at 0.5 and at the chosen threshold.

# References

- [scikit-learn: precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot*](https://doi.org/10.1371/journal.pone.0118432) — why AUC-ROC flatters imbalanced classifiers
- Provost & Fawcett, *Data Science for Business* — cost-sensitive threshold selection

---

# Final Summary

Day 5's job was to replace a number that had peeked at its own answer, and it did — but
the result was the opposite of what "removing the cheat" usually implies.

The three-way chronological split cost 19% of the training data and produced a *better*
model: **F1 0.8605 against Day 4's 0.7530, precision 0.804 against 0.626, false alarms
down from 113 to 45**, with 185 of 200 failures caught. The reason is not that honesty
improves gradient descent. It is that Day 4's early stopping monitored `val_auc` on the
test set, AUC saturated in epoch 1, and five epochs of genuine improvement were thrown
away. A real validation set let the same mechanism keep epoch 5 instead — a model whose
validation precision was 0.81 rather than 0.21.

That finding produced the day's unplanned work. Plotting the curves made it obvious that
`val_auc` had been useless as a selection signal all along: ten epochs inside
0.9991–1.0000 while precision swung between 0.13 and 0.81. The monitor is now `val_f1` by
default, and Run B is testing whether that reasoning survives contact with data rather
than being left as a plausible story.

The threshold work landed too. Sweeping 125,263 candidates on validation and choosing by
cost rather than by F1 gave t=0.6359 — though the honest caveat is that it barely
transferred: validation showed zero misses at that threshold, test showed fifteen. Tuning
an operating point on 175 positives does not generalise as cleanly as the validation
figure suggests, and the cost curve shows the chosen point sitting right at the edge of a
3× cliff. A threshold nearer 0.3–0.5 sits on the flat part at almost identical cost and
would be the safer deployment.

The threshold work produced the day's most useful lesson, and it came from being wrong.
Sweeping on validation and choosing by cost gave t=0.0003 — a threshold in the noise floor
that reached recall 1.0 on 175 validation positives and cost 15 points of test F1. Cost
weighting is the more principled objective and it failed at this sample size, so the
default reverted to F1 and the failure mode now has a detector and a test. The reasoning
that justified the original default was sound; the sample it had to work on was not.

Ending state: 90 tests passing, four items of technical debt repaid, R-8 closed, a model
at F1 0.8949 with 26 false alarms per 172,800 readings, and for the first time a metric
that can be quoted without an asterisk.
