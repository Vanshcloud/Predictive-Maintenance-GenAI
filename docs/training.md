# Training

How to reproduce the published model from a clean checkout, what each stage
costs, and how to tell whether it worked.

For *why* the model is built this way, see [`model.md`](model.md).

---

## Contents

- [The full run](#the-full-run)
- [Stage 1 — generate the dataset](#stage-1--generate-the-dataset)
- [Stage 2 — preprocess](#stage-2--preprocess)
- [Stage 3 — train](#stage-3--train)
- [Stage 4 — evaluate](#stage-4--evaluate)
- [Reproducibility](#reproducibility)
- [Resuming an interrupted run](#resuming-an-interrupted-run)
- [Tuning](#tuning)
- [What a good run looks like](#what-a-good-run-looks-like)

---

## The full run

```bash
source venv/bin/activate

python scripts/generate_data.py        # ~1 min    -> data/raw/       (41 MB)
python scripts/run_preprocessing.py    # ~3 min    -> data/processed/ (4.9 GB)
python scripts/train_model.py          # ~81 min   -> models/*.keras  (1.8 MB)
python scripts/evaluate_model.py       # ~2 min    -> models/evaluation_report.json
```

Timings are measured on Apple Silicon, CPU-only. **~6 GB of free disk is
required**, almost all of it the processed tensors.

None of this is needed to run the test suite — `data/sample/` is committed for
exactly that reason.

---

## Stage 1 — generate the dataset

```bash
python scripts/generate_data.py                 # full: 100 machines × 365 days
python scripts/generate_data.py --sample        # small fixture -> data/sample/
python scripts/generate_data.py --seed 7        # a different draw
```

Produces five related tables totalling **883,231 rows**: telemetry, machines,
errors, maintenance, failures. Byte-identical for a given seed.

> `data/sample/` deliberately contains **zero failure events**. It exists so the
> test suite runs on a clean checkout — it cannot be used to train or evaluate.

---

## Stage 2 — preprocess

```bash
python scripts/run_preprocessing.py
```

Merges the five tables, engineers 63 features, builds 24-hour-horizon labels,
splits chronologically, fits the scaler on training data only, and windows into
`(N, 24, 63)` tensors.

Writes to `data/processed/`:

| File | Contents |
|---|---|
| `X_train.npy` / `y_train.npy` | 567,000 sequences, 800 positive |
| `X_val.npy` / `y_val.npy` | 129,000 sequences, 175 positive |
| `X_test.npy` / `y_test.npy` | 172,800 sequences, 200 positive |
| `scaler.joblib` | The fitted `StandardScaler` — **training statistics** |
| `feature_columns.txt` | The ordered feature contract, 63 names |

The last two are as important as the tensors. `Predictor` verifies at load time
that the model, the scaler, and this feature list all describe the same thing,
and refuses to start if they disagree. A predictor that will not start beats one
that serves confident nonsense.

---

## Stage 3 — train

```bash
python scripts/train_model.py
python scripts/train_model.py --epochs 50 --monitor val_f1
python scripts/train_model.py --seed 7          # measure run-to-run variance
```

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | `30` | Early stopping usually fires first |
| `--batch-size` | `256` | |
| `--learning-rate` | `0.001` | Adam |
| `--seed` | `42` | Seeds weights, dropout, and shuffling |
| `--monitor` | `val_f1` | `val_f1` · `val_auc` · `val_loss` · `val_precision` · `val_recall` |
| `--resume` | off | Continue from the checkpoint and state file |

**Do not switch `--monitor` to `val_auc`.** Under this class imbalance AUC
saturates in the first epochs and then wanders in its fourth decimal place
while precision swings between 0.13 and 0.81. Selecting on it is selecting on
noise. This is measured, not assumed — see [`devlog/day-05.md`](devlog/day-05.md).

Outputs:

| File | Contents |
|---|---|
| `models/lstm_predictive_maintenance.keras` | Best weights, not final weights |
| `models/lstm_predictive_maintenance.state.json` | Resume state — epoch, best score, history |
| `models/training_history.json` | Per-epoch curves |
| `models/metrics.json` | Test metrics **at threshold 0.5** |

> `metrics.json` is scored at `0.5`, not the deployed threshold. The deployed
> operating point is chosen in stage 4 and lands in `evaluation_report.json`.
> The two files reporting slightly different F1 (0.9082 vs 0.9086) is expected.

---

## Stage 4 — evaluate

```bash
python scripts/evaluate_model.py
python scripts/evaluate_model.py --no-plots
```

1. Score the **validation** split and sweep every threshold on its PR curve.
2. Pick an operating point there.
3. Score the **test** split once, at that threshold.
4. Plot training curves and the PR curve.

The ordering is the whole point. Choosing a threshold on the test set and then
reporting test metrics at it is the same mistake as early-stopping on the test
set — the number stops estimating generalisation.

### After a retrain, update the threshold by hand

`evaluate_model.py` writes the chosen threshold to
`models/evaluation_report.json`. **Nothing copies it into
`config/settings.py`.** That edit is manual:

```python
PREDICTION_THRESHOLD: float = 0.3415
RISK_BAND_HIGH: float = 0.3415   # must equal the above
```

A test (`TestServingContract`) fails if they drift apart, so this cannot ship
half-done — but the test tells you *after* the fact. Do it deliberately.

---

## Reproducibility

`keras.utils.set_random_seed(seed)` runs **before the model is built**, because
that is when the LSTM kernels are drawn from `glorot_uniform`. One call covers
Python's `random`, NumPy, and TensorFlow.

Two runs at `--seed 42` on identical data produce the same weights and the same
test F1. Before this was added, `README.md` and `RESULTS.md` quoted an F1 that
nobody — including its author — could have verified.

**Not enabled:** `tf.config.experimental.enable_op_determinism()`. It would
also pin non-deterministic GPU kernel reductions, but it disables the fused
cuDNN LSTM path and costs several times the training time. Seeding alone is
what this CPU-trained model needs.

---

## Resuming an interrupted run

```bash
python scripts/train_model.py --resume
```

Restores epoch number, best score, and per-epoch history from
`*.state.json`. Weights come from the checkpoint itself, so a resumed run picks
up from the **best** model seen, not the last one. Silently starts fresh if no
state exists.

**Optimizer slot variables are not persisted.** Adam's moment estimates rebuild
from scratch, so a resumed run is not bit-identical to an uninterrupted one —
it re-warms over a few batches. A deliberate trade: serialising every slot
variable for marginal benefit on a run this short is not worth the complexity.

---

## Tuning

Everything below is configurable, and none of it has been systematically
searched. There is no hyperparameter sweep in this project, and claiming these
values are optimal would be false — they are reasonable defaults that work.

| Knob | Where | Default |
|---|---|---|
| Sequence length | `DataPreprocessor(sequence_length=…)` | 24 |
| Prediction horizon | `DataPreprocessor(prediction_horizon=…)` | 24 h |
| Rolling windows | `ROLLING_WINDOWS` in `preprocessing.py` | `[3, 12, 24]` |
| Lag periods | `LAG_PERIODS` | `[1, 6, 24]` |
| Split ratios | `test_ratio`, `val_ratio` | 0.20 / 0.15 |
| Layer sizes | `lstm_model.py` | 128 → 64 → 32 |
| Dropout | `lstm_model.py` | 0.3 |
| Patience | `ModelTrainer.train(patience=…)` | 5 |

Changing `sequence_length` or the feature set **invalidates the saved model** —
`Predictor` will refuse to start rather than serve mismatched artifacts. Retrain
and re-evaluate together.

---

## What a good run looks like

```
Epoch 23/30 — loss: 0.0089 | auc: 0.9998 | precision: 0.8043 | recall: 0.9250
   || val_loss: 0.0104 | val_auc: 0.9997 | val_precision: 0.9551
   |  val_recall: 0.9714 | val_f1: 0.9602
Epoch 23: val_f1 improved to 0.9602 — saved models/lstm_predictive_maintenance.keras
...
Early stopping at epoch 28 (no improvement for 5 epochs).
Restored best weights.
```

**Signs something is wrong:**

| Symptom | Likely cause |
|---|---|
| `val_precision` stuck at 0.0 | Class weights not applied, or no positives in validation |
| `val_auc` at 1.0000 from epoch 1 | Leakage — check the split is temporal |
| Training hangs at 0% CPU, no traceback | Import order. See [troubleshooting](troubleshooting.md#the-training-run-hangs-with-no-error) |
| F1 far from 0.9086 at `--seed 42` | The seed is not taking effect, or the data was regenerated with a different seed |

---

## See also

- [`model.md`](model.md) — why the architecture is what it is
- [`RESULTS.md`](RESULTS.md) — every metric with its caveat
- [`troubleshooting.md`](troubleshooting.md) — when a stage fails
- [`benchmarks.md`](benchmarks.md) — measured timings and resource use
