# Day 4 Summary

| Field | Value |
|---|---|
| **Objective** | Design, build, and train an LSTM that predicts equipment failure 24 hours in advance, with a full training pipeline, imbalance handling, proper evaluation, and tests. |
| **Expected outcome** | A production-grade Keras architecture, a training pipeline with early stopping / checkpointing / LR reduction, class-weighted loss for the ~1:730 imbalance, AUC/F1/precision/recall evaluation, a training script, a trained model on disk, and unit tests. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-22 → 2026-08-23 (spanned two sessions; the second was consumed largely by debugging) |
| **Milestone** | M4 — Model architecture & training |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Repository state** | **Dirty** — an interrupted session had left Day 4 work uncommitted |
| **Git commit** | `79c094a` — "feat(data): add feature engineering, preprocessing, and LSTM sequencing" (Day 3) |
| **Existing files** | Days 1–3 complete, plus uncommitted: `src/models/{lstm_model,trainer,evaluator}.py`, `scripts/train_model.py`, `tests/unit/test_model.py`, modified `src/utils/logger.py` and `src/models/__init__.py` |
| **Existing models** | **None** — `models/` contained only `.gitkeep` |
| **Existing checkpoints** | None |
| **Existing datasets** | `data/processed/` — 5.2 GB of tensors from Day 3, intact |
| **Known issues** | K-1 (sample dataset has no failures) |
| **Pending work** | Everything from the model layer onward |

## The state was inconsistent, and reconciling it was the first task

The repository, the documentation, and the tests disagreed with each other:

| Source | Claimed |
|---|---|
| `docs/handoff.md` | "Day 3 of 12 Complete", Day 4 "🔒 Next" |
| Project conventions | Day 4 in progress; `trainer.py` uses `create_tf_dataset()` with `tf.data...prefetch()`; debugging files `debug_fit*.py` exist at the repo root |
| `tests/unit/test_model.py` | Imports `iter_batches` from `trainer.py`, expects `trainer.optimizer` and a **dict** history — i.e. a *manual training loop* |
| `src/models/trainer.py` on disk | The `tf.data` + `model.fit()` version — **no `iter_batches`, no `.optimizer`** |
| `debug_fit*.py` | Did not exist |
| `logs/app_2026-08-23.log` | Two `fit()`-based runs at 02:16 and 02:18 that logged "Computed class weights" and then **stopped producing output** |

File mtimes showed `test_model.py` written at 02:19 and `trainer.py` at 02:25 — the trainer had been reverted *after* the tests describing its replacement were written. The manual-loop implementation was gone; `grep -rn iter_batches` found it only in the test file.

**Conclusion drawn from the evidence, not the docs:** the previous session had been fighting a training hang, had moved to a manual loop, and had lost that work. The tests were the surviving specification. That is what got rebuilt.

---

# Tasks Planned

### T1 — Reconcile repository state

| Field | Detail |
|---|---|
| **Purpose** | Nothing can proceed while the tests, the code, and the docs describe three different systems. |
| **Files affected** | investigation only |
| **Dependencies** | none |
| **Priority** | P0 |
| **Expected outcome** | A defensible account of what exists and what the target is. |

### T2 — LSTM architecture

| Field | Detail |
|---|---|
| **Purpose** | The model itself. |
| **Files affected** | `src/models/lstm_model.py` |
| **Dependencies** | Day 3 tensors |
| **Priority** | P0 |
| **Expected outcome** | Stacked LSTM + dropout + dense head, with `save()`/`load()`. |

### T3 — Training pipeline with class weights and callbacks

| Field | Detail |
|---|---|
| **Purpose** | Train the model without the 1:730 imbalance collapsing it to "always predict 0". |
| **Files affected** | `src/models/trainer.py` |
| **Dependencies** | T2 |
| **Priority** | P0 |
| **Expected outcome** | Class-weighted loss, early stopping, checkpointing, LR reduction, satisfying the API the tests specify. |

### T4 — Imbalance-aware evaluator

| Field | Detail |
|---|---|
| **Purpose** | Measure the model honestly. |
| **Files affected** | `src/models/evaluator.py` |
| **Dependencies** | T2 |
| **Priority** | P0 |
| **Expected outcome** | AUC, precision, recall, F1, confusion matrix. **Never accuracy.** |

### T5 — Training script

| Field | Detail |
|---|---|
| **Purpose** | One command from tensors to metrics. |
| **Files affected** | `scripts/train_model.py` |
| **Dependencies** | T2–T4 |
| **Priority** | P0 |
| **Expected outcome** | Memmapped load → train → reload best checkpoint → evaluate → write `metrics.json` + `training_history.json`. |

### T6 — Resolve the training hang

| Field | Detail |
|---|---|
| **Purpose** | Training did not run at all. Everything else was blocked on it. |
| **Files affected** | unknown at the outset |
| **Dependencies** | T3 |
| **Priority** | **P0 — the day's real work** |
| **Expected outcome** | Training completes on the full dataset. |

### T7 — Unit tests

| Field | Detail |
|---|---|
| **Purpose** | Architecture, class-weight math, batching, the training loop, and the evaluator contract. |
| **Files affected** | `tests/unit/test_model.py`, `tests/conftest.py` |
| **Dependencies** | T2–T4 |
| **Priority** | P0 |
| **Expected outcome** | 7 tests passing, and the *full* suite passing. |

### T8 — Train on the full dataset

| Field | Detail |
|---|---|
| **Purpose** | The day's headline deliverable. |
| **Files affected** | `models/*` |
| **Dependencies** | T1–T7 |
| **Priority** | P0 |
| **Expected outcome** | A trained model, metrics, and history on disk. |

### T9 — Documentation system

| Field | Detail |
|---|---|
| **Purpose** | The Day 4 state confusion was itself a documentation failure. It needed a structural fix, not a patch. |
| **Files affected** | `IMPLEMENTATION_PLAN.md`, `docs/Day1-4.md`, the conventions file, `docs/handoff.md` |
| **Dependencies** | none |
| **Priority** | P1 |
| **Expected outcome** | A single source of truth plus one document per day. |

---

# Work Completed

## LSTM architecture (T2)

`src/models/lstm_model.py` — `PredictiveMaintenanceModel`, **149,825 parameters**:

```
Input (batch, 24, 63)
  ├─ LSTM(128, return_sequences=True)   "lstm_1"
  ├─ Dropout(0.3)                       "dropout_1"
  ├─ LSTM(64)                           "lstm_2"
  ├─ Dropout(0.3)                       "dropout_2"
  ├─ Dense(32, relu)                    "dense_1"
  └─ Dense(1, sigmoid)                  "output"
Output (batch, 1) → P(failure within 24h)
```

`save()` writes Keras v3 `.keras`; `load()` reads the input shape back out of the file and
reconstructs the wrapper, so callers never have to remember the dimensions.

## Training pipeline (T3)

`src/models/trainer.py` was **rewritten from the `tf.data` + `fit()` version into the
manual loop the tests specify** (330 lines).

### `iter_batches()` — module-level batch generator

```python
indices = np.arange(len(X))
if shuffle:
    rng = np.random.default_rng(seed)   # seed = epoch number
    rng.shuffle(indices)
for start in range(0, len(X), batch_size):
    batch_idx = indices[start:start + batch_size]
    if shuffle:
        batch_idx = np.sort(batch_idx)   # keep memmap reads monotonic within a batch
    yield np.asarray(X[batch_idx], dtype=np.float32), \
          np.asarray(y[batch_idx], dtype=np.float32).reshape(-1)
```

Shuffling the epoch order but sorting *within* each batch is the compromise that matters
for a 4.2 GB memmap: the model still sees a fresh order each epoch, but the page cache
sees ascending offsets instead of 256 random seeks.

### Class weights

```
w_c = n_total / (2 · n_c)   →   {0: 0.50, 1: 364.89}
```

Applied as **per-sample weights**, because `class_weight=` only exists inside `fit()`:

```python
sample_weights = np.where(y_batch == 1, w_pos, w_neg).astype(np.float32)
loss = loss_fn(y_batch, preds, sample_weight=sample_weights)
```

### Graph-compiled train step

```python
@tf.function(input_signature=[
    tf.TensorSpec((None, seq_len, n_features), tf.float32),
    tf.TensorSpec((None,), tf.float32),
    tf.TensorSpec((None,), tf.float32),
])
def train_step(x_batch, y_batch, sample_weights): ...
```

The explicit `None` batch dimension prevents a retrace on each epoch's final short batch.
Compute stays graph-optimized; only data movement is synchronous Python.

### Callbacks, reimplemented inline

Keras callbacks only run inside `fit()`. Each was rebuilt with the same semantics:

| Behavior | Implementation |
|---|---|
| **EarlyStopping** | Tracks `val_auc` (max) or training `loss` (min); `patience=5`; breaks the loop; restores best weights from an in-memory NumPy copy |
| **ModelCheckpoint** | `model.save()` only on improvement — the file on disk is always the best model, not the last |
| **ReduceLROnPlateau** | `optimizer.learning_rate.assign(max(lr*0.5, 1e-6))` after 3 stagnant epochs on `val_loss` |
| **Progress logging** | Every 100 batches — the manual replacement for the Keras progress bar |

`train()` returns a **dict** history (`{"loss": [...], "val_auc": [...], ...}`), not a Keras
`History` object.

## Evaluator (T4)

`src/models/evaluator.py` — AUC, precision, recall, F1, confusion matrix. Inference runs
through `_predict_in_batches()`, calling `self.model(batch, training=False)` directly.
Guards the single-class edge case: on `data/sample/` (zero failures) `roc_auc_score`
raises `ValueError`, so it logs a warning and returns 0.0 rather than crashing.

**Accuracy is not computed anywhere**, so it cannot be reported by accident.

## Training script (T5)

`scripts/train_model.py` — `--epochs`, `--batch-size`, `--learning-rate`. Loads all four
arrays with `mmap_mode='r'`, trains, then **reloads the best checkpoint from disk** before
final evaluation. That reload is deliberate: it proves the artifact round-trips, so a
broken `save()` fails the training run rather than lurking until Day 6.

Modified this session to persist the returned history to `models/training_history.json`
(already specified in the Day 3 plan's "Model Saving Strategy").

## Deadlock resolution (T6) — the day's real work

Documented in full under **Bugs Encountered** below. Two hangs, one real root cause,
found by sampling a stuck process.

## Tests (T7)

7 tests in `tests/unit/test_model.py`, plus a new `tests/conftest.py`:

| Test | What it actually verifies |
|---|---|
| `test_model_build_shapes` | `(None, 24, 63)` in, `(None, 1)` out |
| `test_model_layers` | LSTM, Dense, and Dropout layers all present |
| `test_compute_class_weights` | The arithmetic, numerically: 8 neg / 2 pos / 10 total → `w0 = 0.625`, `w1 = 2.5` |
| `test_compile_model` | `trainer.optimizer` is an Adam with the requested learning rate |
| `test_iter_batches_covers_all_samples_without_overlap` | Every sample exactly once, positives conserved, no batch over the limit |
| `test_train_reduces_loss_on_tiny_dataset` | Captures weights, runs 2 epochs, asserts at least one tensor **changed** — catches a silently broken gradient path, which a shape assertion would not |
| `test_evaluator_metrics` | The metric dict contract and types |

**Suite: 75 tests, all passing, ~4 s warm.**

Fixed during this session: the training test wrote its checkpoint to `models/test_checkpoint.keras`,
polluting the real artifact directory. It now uses pytest's `tmp_path`.

## Full training run (T8)

See **Training Progress** below.

## Repository quality pass

`make quality` had **never actually passed**. Running it revealed 14 files out of Black
compliance and 51 flake8 issues accumulated across Days 2–4. Fixed: Black + isort applied
repo-wide, unused imports removed, redundant f-string prefixes dropped, one dead local
deleted, long lines wrapped, and `scripts/*.py: E402` added to `.flake8` (entry-point
scripts must extend `sys.path` before importing project packages, so their import
placement is deliberate).

**Result: 0 flake8 issues, Black and isort clean, 75/75 tests passing.**

## Documentation system (T9)

Created `IMPLEMENTATION_PLAN.md` (single source of truth) and `docs/Day1.md`–`Day4.md`.
Updated the conventions file (which described the deleted `tf.data` trainer and non-existent
`debug_fit*.py` files) and added a "superseded" banner to `docs/handoff.md`.

---

# Code Changes

## `src/models/lstm_model.py` — created

| Field | Detail |
|---|---|
| **Purpose** | The Keras architecture, isolated from training and data concerns. |
| **Important changes** | Sequential model; `save()`/`load()`; `load()` recovers input shape from the file. |
| **Breaking changes** | None. |
| **Imports added** | `tensorflow`, `tensorflow.keras.layers`, `tensorflow.keras.models` |
| **Classes changed** | `PredictiveMaintenanceModel` (new) |
| **Note** | `import tensorflow as tf` is kept with `# noqa: F401` and a comment — it is the import that pulls TF in, and its position matters (see `__init__.py` below). |

## `src/models/trainer.py` — rewritten

| Field | Detail |
|---|---|
| **Purpose** | Training orchestration. |
| **Important changes** | Replaced `create_tf_dataset()` + `model.fit()` with `iter_batches()` + a `GradientTape` loop; inline early stopping / checkpointing / LR reduction. |
| **Breaking changes** | **Yes.** `compile()` now sets `self.optimizer`; `train()` returns a `dict`, not a Keras `History`; `create_tf_dataset()` is gone. |
| **Functions added** | `iter_batches()`, `ModelTrainer._make_train_step()`, `ModelTrainer._run_validation()`, `ModelTrainer._save_checkpoint()` |
| **Functions removed** | `create_tf_dataset()` |
| **Configuration** | `patience=5`, `lr_patience=3`, `min_lr=1e-6`, `LOG_EVERY_N_BATCHES=100` |
| **Docstring** | Rewritten to record the *correct* root cause after the original diagnosis was disproven. |

## `src/models/evaluator.py` — created, docstring corrected

| Field | Detail |
|---|---|
| **Purpose** | Imbalance-aware evaluation. |
| **Important changes** | `_predict_in_batches()` instead of `model.predict()`; single-class AUC guard. |
| **Imports added** | `sklearn.metrics` |
| **Classes changed** | `ModelEvaluator` (new) |

## `src/models/__init__.py` — modified (**load-bearing**)

| Field | Detail |
|---|---|
| **Purpose** | Package exports — and, as it turns out, process-wide symbol resolution order. |
| **Important changes** | Imports reordered so `lstm_model` (TensorFlow) loads **before** `evaluator` (scikit-learn). Added `# isort: skip_file` so isort cannot silently reintroduce the bug, plus a 30-line docstring explaining why. |
| **Breaking changes** | None functionally — but reordering these imports back **will** hang the process. |

## `tests/conftest.py` — created

| Field | Detail |
|---|---|
| **Purpose** | Import TensorFlow before any test module loads pandas or scikit-learn. |
| **Important changes** | A single `import tensorflow as tf  # noqa: F401` under a docstring explaining the deadlock. |
| **Breaking changes** | None. Without it, the full suite hangs forever. |

## `scripts/train_model.py` — modified

Captures `train()`'s return value and writes `models/training_history.json`.

## `src/utils/logger.py` — modified

`enqueue=True` removed from the file sink (carried over from the previous session's
debugging). Also lost an unused `pathlib.Path` import in the lint pass.

## `tests/unit/test_model.py` — created, then modified

7 tests; the training test now writes its checkpoint to `tmp_path`.

## Repo-wide

`black` + `isort` applied to 14 files; unused imports and dead code removed from
`src/data/preprocessing.py`, `src/data/validation.py`, `src/utils/logger.py`,
`scripts/eda_analysis.py`; `.flake8` gained a `scripts/*.py: E402` exemption.

---

# Training Progress

| Field | Value |
|---|---|
| **Dataset** | `data/processed/` — X_train (698400, 24, 63), y_train 957 positives (1:730); X_test (172800, 24, 63), y_test 200 positives (1:864) |
| **Epochs** | 30 requested, **6 run** — early stopping fired |
| **Batch size** | 256 (2,729 batches/epoch) |
| **Learning rate** | 0.001, reduced to 0.0005 at epoch 5 by the plateau rule |
| **Optimizer** | Adam |
| **Loss** | Binary crossentropy with per-sample class weights {0: 0.50, 1: 364.89} |
| **Training duration** | 03:07:00 → 03:32:42 = **~25.7 minutes** (~4.3 min/epoch incl. validation; ~1.7 min/epoch train-only at 36 ms/batch) |
| **Checkpoint path** | `models/lstm_predictive_maintenance.keras` (1.8 MB) |
| **TensorBoard logs** | None — see TD-3 |
| **Saved models** | Best checkpoint (epoch 1), weights restored in-process |
| **Training curves** | `models/training_history.json` — 8 series × 6 epochs |

## Per-epoch results

| Epoch | loss | auc | precision | recall | val_loss | **val_auc** | val_precision | val_recall |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0319 | 0.9986 | 0.1232 | 0.9916 | 0.0032 | **0.9998** ← best | 0.6258 | 0.9450 |
| 2 | 0.0073 | 0.9997 | 0.3116 | 0.9990 | 0.0024 | 0.9873 | 0.7054 | 0.9100 |
| 3 | 0.0087 | 0.9997 | 0.3443 | 0.9969 | 0.0026 | 0.9948 | 0.6526 | 0.9300 |
| 4 | 0.0073 | 0.9997 | 0.4063 | 0.9990 | 0.0032 | 0.9898 | 0.6206 | 0.9650 |
| 5 | 0.0067 | 0.9997 | 0.4030 | 0.9979 | 0.0035 | 0.9873 | 0.5823 | 0.9550 |
| 6 | 0.0074 | 0.9993 | 0.5034 | 0.9979 | 0.0039 | 0.9948 | 0.5026 | 0.9700 |

Early stopping at epoch 6 (5 epochs without beating epoch 1). Best weights restored.

## Final test metrics (`models/metrics.json`)

| Metric | Value | Day 5 target | Met? |
|---|---|---|---|
| **ROC-AUC** | **0.9999** | ≥ 0.85 | ✅ |
| **Precision** | **0.6258** | ≥ 0.30 | ✅ |
| **Recall** | **0.9450** | ≥ 0.60 | ✅ |
| **F1** | **0.7530** | ≥ 0.40 | ✅ |

Confusion matrix at threshold 0.5:

```
                 predicted 0   predicted 1
actual 0            172,487           113     ← 113 false alarms
actual 1                 11           189     ← caught 189 of 200 failures
```

**Operationally:** 189 of 200 failures caught with 24 hours' warning, 11 missed, and 113
false alarms across 172,800 hourly readings — roughly one unnecessary inspection every
1,530 machine-hours.

## How to read these numbers — three caveats that matter

1. **The validation set *is* the test set (TD-1).** `X_val=X_test` was passed for
   monitoring, so early stopping and checkpoint selection both observed the test set. The
   reported metrics have peeked at what they measure. AUC 0.9999 is *not* a clean
   generalization estimate.
2. **The data is synthetic, with degradation the generator was designed to make
   detectable.** A 48-hour ramp with per-machine offsets is realistic in *shape*, but real
   equipment is messier. These metrics reflect the difficulty of this dataset, not of
   industrial predictive maintenance.
3. **The model converged in one epoch and then oscillated.** `val_precision` wandered
   0.63 → 0.71 → 0.65 → 0.62 → 0.58 → 0.50 while `val_recall` stayed 0.91–0.97. That is a
   model trading precision against recall around a fixed 0.5 threshold, not one that is
   still learning. It is the clearest possible signal that **Day 5's threshold sweep will
   matter more than any further training.**

## Failures during training

Two hangs, both fatal, both silent. See **Bugs Encountered**.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 7 new (`test_model.py`), **75 total**, all passing |
| **Integration tests** | None yet — planned Day 9 |
| **Manual testing** | Reloaded `models/lstm_predictive_maintenance.keras` from disk in a fresh process, confirmed `input_shape=(None, 24, 63)` and 149,825 parameters, and ran inference |
| **Benchmark: training throughput** | **36 ms/batch** steady-state (median of 30, after discarding the 1.38 s first batch that includes `tf.function` tracing) → 2,729 batches ≈ 1.7 min/epoch train-only |
| **Benchmark: inference latency** | **54.0 ms median, 55.4 ms p95** for a single sequence (100 calls, post-warm-up) — **NFR-3 (<100 ms) PASS** |
| **Memory usage** | ~200 MB RSS at model-build time; the 5.2 GB of tensors stayed memmapped and never resident |
| **CPU usage** | CPU-only (Apple Silicon, no CUDA); TF's Eigen thread pool saturated during training |
| **GPU usage** | N/A |
| **Quality gates** | flake8 **0 issues**; Black clean; isort clean |

Benchmark methodology followed the plan: idle machine (the throughput probe was explicitly
serialized behind the test suite), warm-up batch discarded, **median** rather than mean
reported.

---

# Bugs Encountered

## B1 — TensorFlow deadlock from an abseil symbol collision with Apache Arrow

**The day's defining bug.**

| Field | Detail |
|---|---|
| **Description** | Training hung indefinitely at **0% CPU** — no traceback, no timeout, no error, no log output past "Computed class weights". Indistinguishable from "slow". It struck `scripts/train_model.py` and, separately, the full test suite (which ran for 16 minutes before being killed). |
| **Files affected** | `src/models/__init__.py`, `tests/conftest.py`, `src/models/trainer.py` (docstring), `src/models/evaluator.py` (docstring) |

### Root cause

TensorFlow statically links its own copy of abseil. So does Apache Arrow
(`libarrow.2400.dylib`), which **pandas and scikit-learn both load**. On macOS, whichever
library loads first wins the `AbslInternalPerThreadSemWait` symbol **for the entire
process**.

If Arrow wins, TensorFlow's `absl::Mutex::Block()` waits on *Arrow's* per-thread
semaphore — different thread-local state, so the wake-up never arrives. The first
`tf.function` execution deadlocks forever.

Confirmed by sampling the stuck process:

```
tensorflow::...::ProcessFunctionLibraryRuntime::RunSync
  absl::Notification::WaitForNotification()      (libtensorflow_framework.2.dylib)
    absl::Mutex::LockSlowWithDeadline()          (libtensorflow_framework.2.dylib)
      absl::Mutex::Block()                       (libtensorflow_framework.2.dylib)
        AbslInternalPerThreadSemWait_lts_20250814  (libarrow.2400.dylib)   ← wrong abseil
          PthreadWaiter::Wait()                    (libarrow.2400.dylib)
            _pthread_cond_wait
```

Every Eigen worker thread was idle in `WaitForWork`. The main thread was blocked on a
semaphore belonging to a different library.

### Why it went undiagnosed for a whole session

`src/models/__init__.py` imported in alphabetical order:

```python
from src.models.evaluator import ModelEvaluator          # ← imports sklearn → loads libarrow
from src.models.lstm_model import PredictiveMaintenanceModel   # ← imports tensorflow, too late
from src.models.trainer import ModelTrainer
```

Every entry point that did `from src.models import ...` poisoned itself **before reaching
any training code**. The previous session had therefore blamed `model.fit()` — first
`keras.utils.Sequence`, then `PyDataset`, then `tf.data...prefetch()` — rewriting the
input pipeline three times. All three "failed" identically, because none of them was ever
reached. The evidence pointed at the last thing in the stack rather than the first thing
in the import list.

### Reproduction (deterministic, no pytest, no memmap, no `fit()`)

| Order | Result |
|---|---|
| TensorFlow only | ✅ trains |
| `pandas.rolling()` → `import tensorflow` → train step | ❌ deadlock, 0% CPU |
| `sklearn.StandardScaler` → `import tensorflow` → train step | ❌ deadlock, 0% CPU |
| **`import tensorflow`** → pandas + sklearn → train step | ✅ trains |

### Solution

1. **`src/models/__init__.py`** — TensorFlow-importing module first, sklearn-importing
   module last, with `# isort: skip_file` so the formatter cannot silently undo it, and a
   docstring carrying the stack trace.
2. **`tests/conftest.py`** — imports TensorFlow before any test module loads. conftest is
   imported before test collection, so this fixes the order for the whole session.
3. **Docstring corrections** in `trainer.py` and `evaluator.py`, which asserted the
   disproven prefetch-thread theory.
4. **The conventions file** updated with the rule for future entry points.

### Verification

- The throughput probe went from hanging at the first `train_step` to `FIRST BATCH DONE in
  1.38s`, then 36 ms/batch.
- The full suite went from a 16-minute hang to **75 passed in 4.04s**.
- Full training completed end to end.

### Lessons learned

- **A hang is a stack trace waiting to be read.** Twenty minutes with `sample <pid>` beat a
  session of rewriting the input pipeline. The very first diagnostic on a 0%-CPU process
  should be "what is the main thread actually blocked on", not "what did I change last".
- **Import order can be load-bearing**, and nothing in Python's tooling knows that. isort
  would have reverted this fix silently; hence `# isort: skip_file` plus a comment loud
  enough to survive a future cleanup.
- **A plausible diagnosis that fits the symptom is not a verified one.** "Background
  prefetch thread deadlocks against memmap" explained everything observed and was wrong.
  What distinguished it from the truth was a controlled experiment, which cost minutes.
- **The fix is invisible and the failure is silent.** That combination is why both the
  import and the conftest carry long docstrings naming the failure they prevent.

## B2 — Full test suite hung while individual test files passed

| Field | Detail |
|---|---|
| **Description** | `pytest tests/unit/test_model.py` → 7 passed in 9 s. `pytest tests/` → hung 16 minutes at 0% CPU inside `test_train_reduces_loss_on_tiny_dataset`. |
| **Root cause** | The same as B1, exposed by collection order: `test_preprocessing.py` (pandas/sklearn) is collected before `test_model.py`, so Arrow loaded first. Alphabetical ordering decided whether the suite terminated. |
| **Files affected** | `tests/conftest.py` (created) |
| **Solution** | `conftest.py` imports TensorFlow before any test module. |
| **Verification** | 75 passed in 4.04 s. |
| **Lessons learned** | Test isolation cuts both ways: passing in isolation and hanging in a suite is *evidence about process-global state*, not a flaky test. Also: this bug has no possible regression test, because the failure mode is non-termination — the suite's own ability to finish is the assertion. |

## B3 — `make quality` had never passed

| Field | Detail |
|---|---|
| **Description** | 14 files out of Black compliance, 51 flake8 issues, accumulated since Day 2. |
| **Root cause** | The gate was *configured* on Day 1 but never *run* in CI or pre-commit. Day 1's notes claimed "every commit since has been format-clean" — that claim was untested and false. |
| **Files affected** | 14 source files, `.flake8` |
| **Solution** | Black + isort repo-wide; unused imports and dead code removed; long lines wrapped; `scripts/*.py: E402` exemption added for `sys.path` bootstrapping. |
| **Verification** | flake8 0 issues; Black and isort clean; 75/75 tests still passing. |
| **Lessons learned** | Configuring a quality gate is not enforcing one. Day 11's CI is what will keep this from recurring. The retroactive claim in `docs/Day1.md` has been corrected rather than quietly dropped. |

## B4 — Self-inflicted: a regex lint fix corrupted three files

| Field | Detail |
|---|---|
| **Description** | While removing redundant f-string prefixes, a regex dropped the opening quote instead of the `f`, turning `f"text"` into `ftext"` and producing `SyntaxError` in `preprocessing.py`, `eda_analysis.py`, and `run_preprocessing.py`. A follow-up "repair" regex over-matched and made it worse (33 substitutions where ~10 were wanted). |
| **Root cause** | Regex applied to Python source without parsing it, and applied blind — no verification between the edit and the next edit. |
| **Files affected** | 3 files, all committed and therefore recoverable |
| **Solution** | `git checkout` to restore, then redo the fix line-anchored with `ast.parse()` validating **every** candidate edit before writing. |
| **Verification** | All files parse; 75/75 tests pass; flake8 clean. |
| **Lessons learned** | Never transform source with a regex without an `ast.parse()` gate on the result. And when a "fix" makes things worse, revert to a known-good state rather than layering a second heuristic on the first — the second regex was a worse mistake than the first. Being inside a clean git tree is what made this a five-minute detour instead of an hour. |

## B5 — Test wrote a checkpoint into the real artifact directory

| Field | Detail |
|---|---|
| **Description** | `test_train_reduces_loss_on_tiny_dataset` saved to `models/test_checkpoint.keras`, next to the actual trained model. |
| **Root cause** | A hardcoded path in the test. |
| **Files affected** | `tests/unit/test_model.py` |
| **Solution** | Use pytest's `tmp_path` fixture. |
| **Verification** | `models/` now contains only real artifacts after a test run. |
| **Lessons learned** | A test that writes outside its sandbox is a test that can be mistaken for a deliverable. |

---

# Design Decisions

## D1 — Keep the manual training loop after the real bug was found

| Field | Detail |
|---|---|
| **Alternatives** | Revert to `model.fit()` now that the actual cause is fixed; keep the manual loop; support both. |
| **Pros of reverting** | Less code; Keras callbacks for free; battle-tested; would likely enable TensorBoard. |
| **Cons of reverting** | The manual loop is written, tested, and demonstrably works at 36 ms/batch. Reverting means rewriting the trainer a *fourth* time and re-testing, for no measured gain. |
| **Reason for selection** | Kept the manual loop. It works; it makes class weighting, early stopping, LR reduction, and checkpointing explicit and inspectable rather than hidden behind framework callbacks; and it removes background-thread behavior from the training path. |
| **Impact** | Recorded honestly: `fit()` has **not** been re-benchmarked since the real fix, so the code and docs say "not retested", not "broken". If someone wants it back, the burden is a benchmark proving it completes. |

## D2 — Fix import order at the package level rather than globally

| Field | Detail |
|---|---|
| **Alternatives** | Import TF in `src/__init__.py` (fixes every entry point unconditionally); fix only at each entry point; set an environment variable. |
| **Pros of the global fix** | Airtight — no import order could ever poison the process. |
| **Cons** | Every data-only script and every data-layer test would pay TensorFlow's ~90 s cold import for nothing. `run_preprocessing.py` does not need TF. |
| **Reason for selection** | Guard in `src/models/__init__.py` (which covers every current TF consumer) plus `tests/conftest.py`, plus a documented rule for future entry points. |
| **Impact** | Covers all present code. A future `src/api/main.py` that imports `src.data` before `src.models` could still hit it — which is exactly why the rule is written into the conventions and the plan rather than left implicit. |

## D3 — Shuffle epoch order, sort indices within a batch

| Field | Detail |
|---|---|
| **Alternatives** | No shuffling; full random access per batch; block-level shuffling of contiguous slices. |
| **Pros** | Fresh order every epoch *and* monotonic disk reads within a batch. |
| **Cons** | Batch composition is still random across a 4.2 GB file, so reads are scattered between batches. |
| **Reason for selection** | Block shuffling would fix batch composition across epochs — bad with clustered positives. Unsorted random access wastes the page cache. This is the middle ground, and at 36 ms/batch the I/O is clearly not the bottleneck. |
| **Impact** | Reproducible per-epoch ordering (seed = epoch number). |

## D4 — Early stop on `val_auc`, reduce LR on `val_loss`

| Field | Detail |
|---|---|
| **Alternatives** | Both on `val_loss`; both on `val_auc`; monitor F1. |
| **Pros** | With a 365× positive weight, the weighted loss can drift while *ranking quality* — which is what AUC measures and what actually matters — keeps improving. |
| **Cons** | Two different monitored metrics is slightly more complex to reason about. |
| **Reason for selection** | Model *selection* should optimize the thing we care about (ranking); the *learning rate* schedule should respond to the thing being optimized (loss). |
| **Impact** | Vindicated: epoch 1 had the best `val_auc` (0.9998) but the worst training precision (0.12). Monitoring `val_loss` would have selected a different, worse-ranking epoch. |

## D5 — Reload the checkpoint from disk before final evaluation

| Field | Detail |
|---|---|
| **Alternatives** | Evaluate the in-memory model after restoring best weights. |
| **Pros** | Proves the artifact round-trips. A broken `save()` fails the training run rather than surfacing on Day 6 when inference is built. |
| **Cons** | A few seconds of load time; briefly holds two models in memory. |
| **Reason for selection** | The saved file — not the in-memory object — is what Day 6 will consume. Test what ships. |
| **Impact** | Caught nothing this time, which is the correct outcome for a guard. Independently re-verified afterward by loading in a fresh process. |

## D6 — Accept validating on the test set for Day 4, and log it

| Field | Detail |
|---|---|
| **Alternatives** | Build a proper three-way chronological split now; use a random validation slice; skip validation entirely. |
| **Pros of accepting** | Day 4's question was "does this train at all"; a third split was scope creep on a day already consumed by a deadlock. |
| **Cons** | Reported metrics are optimistic — model selection has peeked. |
| **Reason for selection** | Taken deliberately and **logged as TD-1 and Risk R-8**, so no reader mistakes 0.9999 for a clean generalization estimate. |
| **Impact** | Day 5's first task. An undocumented version of this shortcut would be a serious integrity problem; a documented one is a scheduling decision. |

## D7 — Build the documentation system now, not at the end

| Field | Detail |
|---|---|
| **Alternatives** | Keep patching `docs/handoff.md`; document everything on Day 12. |
| **Pros** | Day 4 opened with three sources describing three different systems and lost work — that *is* the cost of documentation drift, paid in full. |
| **Cons** | A substantial writing session mid-project. |
| **Reason for selection** | The failure had already happened. Fixing the process was cheaper than paying that cost again on Days 5–12. |
| **Impact** | `IMPLEMENTATION_PLAN.md` + `docs/DayX.md` + a mandatory session workflow; `docs/handoff.md` frozen with a "superseded" banner rather than deleted. |

---

# Remaining Tasks

| Item | Priority | Dependencies | Effort |
|---|---|---|---|
| **TD-1** — Replace `X_val=X_test` with a chronological validation slice from the training tail | **P0** | none | 2 h |
| **TD-6** — Threshold sweep + PR curve; choose a cost-based operating point | **P0** | TD-1 | 2 h |
| **TD-2** — Checkpoint resume (persist epoch/history/optimizer state, add `--resume`) | P1 | none | 2 h |
| **TD-3** — Plot `training_history.json`, or emit `tf.summary` from the loop | P1 | none | 1 h |
| Per-machine error analysis — which machines produce the 11 misses and 113 false alarms | P1 | TD-1 | 2 h |
| Hyperparameter comparison (sequence length, LSTM widths, dropout) | P2 | TD-1 | 3 h |
| Re-benchmark `model.fit()` now that the abseil bug is fixed — settle D1 with data | P3 | none | 1 h |
| **TD-4** — Fold or retire `docs/handoff.md` | P3 | none | 1 h |

---

# Next Day Plan

**Day 5 — Model Evaluation & Optimization**

1. **Fix TD-1 first.** Carve a chronological validation slice from the tail of the training
   period (e.g. train Jan–Aug, validate Sep–Oct, test Oct–Dec). Retrain. **The test set
   gets touched exactly once, at the end.** Every number in this document should be
   regarded as provisional until this is done.
2. **Threshold sweep.** Compute the full precision-recall curve. The 0.5 threshold is a
   placeholder; with 200 positives in 172,800 sequences the right operating point comes
   from the cost ratio between a missed failure and an unnecessary inspection. Report the
   curve, then pick a point and justify it.
3. **Plot the training curves** from `training_history.json` (TD-3) — loss, AUC, precision,
   recall, train vs validation.
4. **Error analysis.** Which machines account for the 11 misses? Are the 113 false alarms
   clustered on specific machines or specific sensor patterns? This is what Day 7's report
   generator will need in order to *explain* a prediction.
5. **Checkpoint resume** (TD-2).
6. **Hyperparameter comparison** — sequence length 12 vs 24 vs 48; LSTM widths; dropout.
   Compare on validation only.
7. Optionally, re-benchmark `model.fit()` to settle D1 empirically.

**Expected difficulty:** the model already scores near the ceiling on this synthetic data,
so Day 5's value is not in raising AUC — it is in producing an *honest* number and a
*defensible* operating point.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | **~35%** (4 of 12 days) |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · `src/models/` 100% · `src/prediction/` 0% · `src/genai/` 0% · `src/api/` 0% · `dashboard/` 0% · CI/Docker 0% |
| **Technical debt** | TD-1 (val=test) · TD-2 (no resume) · TD-3 (no TensorBoard) · TD-4 (`docs/handoff.md` overlap) · TD-6 (fixed 0.5 threshold) · TD-7 (no integration tests) |
| **Known risks** | R-8 optimistic metrics (**open**, scheduled) · R-6 training/serving skew (mitigated by design, enforcement lands Day 6) · R-2/R-3 deadlocks (**resolved**) |
| **Immediate priorities** | Fix TD-1, then sweep thresholds. Both must land before any metric from this document is quoted as final. |
| **Quality gates** | 75/75 tests · flake8 0 issues · Black clean · isort clean |

---

# Files Created

```
src/models/lstm_model.py          (LSTM architecture)
src/models/trainer.py             (manual training loop, 330 lines)
src/models/evaluator.py           (imbalance-aware metrics)
scripts/train_model.py            (training entry point)
tests/unit/test_model.py          (7 tests)
tests/conftest.py                 (TF-first import guard — CRITICAL)
IMPLEMENTATION_PLAN.md            (single source of truth)
docs/Day1.md  docs/Day2.md  docs/Day3.md  docs/Day4.md
models/metrics.json               (committed — evaluation evidence)
models/training_history.json      (committed — per-epoch curves)
models/lstm_predictive_maintenance.keras   (gitignored, 1.8 MB)
```

# Files Modified

```
src/models/__init__.py            (import order — load-bearing)
src/utils/logger.py               (enqueue=True removed; unused import)
Project conventions               (corrected stale trainer description; added abseil rule)
handoff.md                        (superseded banner)
.flake8                           (scripts/*.py: E402 exemption)
docs/Day1.md                      (corrected the false format-cleanliness claim)
+ 14 files reformatted by Black/isort across src/, scripts/, tests/, config/
```

# Files Deleted

None. (`debug_fit*.py` were referenced by the conventions file but did not exist — the reference
was removed.)

# Models Generated

`models/lstm_predictive_maintenance.keras` — 149,825 parameters, 1.8 MB, Keras v3 format.
Verified to reload in a fresh process and predict at 54 ms/sequence.

# Checkpoints Generated

One, written at epoch 1 and never superseded (no later epoch beat `val_auc` 0.9998).

# Reports Generated

`models/metrics.json` — AUC 0.9999, precision 0.6258, recall 0.9450, F1 0.7530, confusion
matrix `[[172487, 113], [11, 189]]`.

# Logs Generated

`logs/app_2026-08-23.log` — the full day, including both deadlocked runs (which end
abruptly after "Computed class weights") and the successful training run.

# Screenshots

None. Training curves are stored as data in `models/training_history.json`; plots are a
Day 5 deliverable (TD-3).

# References

- `sample(1)` — the macOS process sampler that produced the stack trace identifying B1
- [abseil: static linking and symbol collisions](https://abseil.io/docs/cpp/guides/base) — background on why two statically-linked copies conflict
- [Keras: writing a training loop from scratch](https://keras.io/guides/writing_a_training_loop_from_scratch/)
- [`tf.function` input signatures and retracing](https://www.tensorflow.org/guide/function)
- [scikit-learn: precision, recall and F-measure](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [NumPy `memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

---

# Final Summary

Day 4 delivered the brain of the platform — and spent most of its time proving that the
hardest bugs are the ones that produce no error at all.

The deliverables are complete: a 149,825-parameter stacked LSTM, a training pipeline with
class weighting for the 1:730 imbalance and inline early stopping / checkpointing / LR
reduction, an evaluator that reports AUC, precision, recall, F1, and a confusion matrix
while refusing to compute accuracy, a one-command training script, 7 unit tests inside a
75-test green suite, and a trained model on disk that reloads in a fresh process and
predicts in 54 ms. Final test metrics: **AUC 0.9999, precision 0.6258, recall 0.9450,
F1 0.7530** — 189 of 200 failures caught, 11 missed, 113 false alarms across 172,800
readings.

But the day's actual work was a deadlock. Training hung at 0% CPU with no traceback, and
the previous session had rewritten the input pipeline three times chasing a plausible,
confident, wrong theory about Keras prefetch threads. Sampling the stuck process took
twenty minutes and pointed somewhere else entirely: TensorFlow's `absl::Mutex::Block()`
waiting on a semaphore inside `libarrow.2400.dylib`. Two statically-linked copies of
abseil, and `src/models/__init__.py` imported the sklearn-dependent module before the
TensorFlow-dependent one — so alphabetical import order decided whether the process could
train. `fit()` was never the problem; it was never reached.

The fix is three lines and completely invisible, which is why it now carries thirty lines
of docstring, an `# isort: skip_file`, and a documented rule. A separate 51-issue lint
debt was cleared, a self-inflicted regex disaster was reverted rather than compounded, and
a test that had been writing into the real `models/` directory was sandboxed.

Two things should temper the headline numbers. The validation set is currently the test
set (TD-1), so model selection has peeked at what it reports. And the model converged in
a single epoch and then merely oscillated between precision and recall — which says the
remaining value is not in more training but in an honest split and a justified threshold.
Both are Day 5's job, and until they are done, every metric here is provisional.

Ending state: 75 passing tests, zero lint issues, a trained model, and — for the first
time — documentation that matches the code.
