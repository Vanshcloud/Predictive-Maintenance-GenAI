# Results

Every number here is reproducible from a clean checkout, and every one is
traceable to a committed artifact. Where a figure needs a caveat, the caveat is
next to it rather than in a footnote.

**Reproduce:**

```bash
python scripts/generate_data.py        # seed=42, byte-identical every time
python scripts/run_preprocessing.py
python scripts/train_model.py --epochs 30
python scripts/evaluate_model.py
```

---

## The dataset

| Property | Value |
|---|---|
| Source | Synthetic, `scripts/generate_data.py`, seed 42 |
| Rows | 883,231 across 5 related tables |
| Period | 2024-01-01 → 2024-12-30 (364 days, hourly, 100 machines) |
| Sensors | voltage, rotation, pressure, vibration |
| Failure events | 47 |
| **Positive rate after 24 h labelling** | **0.13% — about 1:745** |

That last row is the defining property of the problem. A model predicting "no
failure" for every row scores **99.87% accuracy**, which is why accuracy is
never reported anywhere in this project and is not computed by
`ModelEvaluator` at all.

### Splits — chronological, never random

| Split | Sequences | Positives | Rate |
|---|---|---|---|
| Train | 567,000 | 800 | 1:709 |
| Validation | 129,000 | 175 | 1:737 |
| Test | 172,800 | 200 | 1:864 |

With 24-hour lag features, a random split puts hour *t* in train and *t+1* in
test — the model then "predicts" a failure whose evidence it has already
memorised, and the reported AUC becomes fiction. Enforced by
`test_train_before_test` and two sibling tests.

---

## The model

`PredictiveMaintenanceModel` — **149,825 parameters**.

```
Input (batch, 24, 63)
  LSTM(128, return_sequences=True) → Dropout(0.3)
  LSTM(64)                         → Dropout(0.3)
  Dense(32, relu) → Dense(1, sigmoid)
```

Trained with class-weighted binary crossentropy (**{0: 0.50, 1: 364.89}**),
Adam at 0.001, batch 256. Early stopping on `val_f1` fired at epoch 20 of 30;
best weights came from **epoch 15** (`val_f1` 0.9359).

---

## Test-set performance

Threshold **0.6678**, chosen by sweeping the precision-recall curve on the
**validation** split. The test set was scored once, at that threshold.

| Metric | Value |
|---|---|
| ROC-AUC | **0.9997** |
| Average precision (validation) | 0.9848 |
| Precision | **0.8756** |
| Recall | **0.9150** |
| **F1** | **0.8949** |
| Single-sequence inference | **54 ms** median |

```
                 predicted 0   predicted 1
actual 0            172,574            26     ← false alarms
actual 1                 17           183     ← caught 183 of 200 hourly labels
```

### Measured over failure *events*, which is what operations cares about

Every hour in the 24 hours before a failure carries a positive label, so the
table above counts **hours**, not failures. Missing 9 of 24 hours while
catching the other 15 scores as 9 "missed failures" — but the technician was
warned, once, which is all that was required.

| Metric | Value |
|---|---|
| Failure events in the test period | 8 |
| **Events warned about** | **8 (100%)** |
| Sequence-level recall | 91.5% |
| Lead time (median / min) | **24 h / 15 h** |

> **Eight events is a small denominator.** This says the model warned in 8 of 8
> cases, not that it never misses. Reported alongside precision, never instead
> of it — event recall says nothing about alert fatigue.

### Errors are concentrated

10 of 100 machines account for all 43 errors; the top 5 account for 81%. Age
does **not** explain it — mean age of error-producing machines is 9.3 against
9.8 fleet-wide. Day 2's finding that older machines fail more often does not
make them harder to predict.

---

## How the numbers moved, and why

| | Day 4 | Day 5 (Run A) | Day 5 (Run B) |
|---|---|---|---|
| Split | val = test | clean 3-way | clean 3-way |
| Early-stopping monitor | `val_auc` | `val_auc` | **`val_f1`** |
| Precision | 0.6258 | 0.8043 | **0.8756** |
| Recall | 0.9450 | 0.9250 | 0.9150 |
| **F1** | 0.7530 | 0.8605 | **0.8949** |
| False alarms | 113 | 45 | **26** |

**Removing the leak made the model better, on 19% less training data.** Not
because honesty improves gradient descent — because Day 4's early stopping
monitored `val_auc` against the test set, AUC saturated in epoch 1, and five
epochs of real improvement were discarded.

**`val_auc` was never a usable selection signal here.** Across Run B's twenty
epochs it stayed within **0.9991–1.0000** while validation precision swung
**0.13 → 0.81**. It peaked at epoch 8 and then *fell*, so an AUC monitor would
have kept epoch 8 — whose validation precision was 0.813 against epoch 15's
0.913.

### The threshold sweep, including where it went wrong

| Operating point | Threshold | Test F1 | False alarms |
|---|---|---|---|
| cost-optimal (100:1) | 0.0003 | 0.7366 | 126 |
| default | 0.5 | 0.8889 | 30 |
| **best-F1 (deployed)** | **0.6678** | **0.8949** | **26** |

Cost-weighted selection is the more principled objective and it **failed at
this sample size**: with 175 validation positives and a 100:1 ratio, minimising
cost collapses to "reach recall 1.0 at any price", and the cheapest route is a
noise-floor threshold that does not transfer. It cost 15 points of test F1. The
default is now best-F1, and `sweep_thresholds()` returns a
`lowest_cost_is_degenerate` flag so the failure mode cannot recur silently.

---

## Serving

| Measurement | Value |
|---|---|
| Prediction, sliced to one machine | **~160 ms** |
| Prediction, whole fleet dataset passed in | **> 120 s** |
| `GET /machines/{id}/predict` | **137 ms** median |
| `POST /predict` (60 supplied readings) | 108 ms |
| `GET /fleet` (100 machines) | 13.4 s cold → **1.6 ms** cached |
| `POST /report` (local LLM) | ~21 s |

That 800× gap is why `MachineDataStore.slice_for()` exists. `merge_tables` and
`engineer_features` run over whatever they are handed, so passing 876,000 rows
to score one machine does 99% of its work on rows it discards.

**NFR-4 (p95 `/predict` < 500 ms): met at 137 ms.**

---

## Training/serving parity

The inference path and the training path were compared over all 172,800 test
sequences:

| Measure | Result |
|---|---|
| Max absolute difference | 2.98e-08 |
| Correlation | 1.0000000000 |
| **Alert-decision agreement** | **100.0000%** |

Risk R-6 was "mitigated by design" until this was measured. It is now asserted
by `tests/integration/test_training_serving_parity.py`.

---

## Containers

| Image | Size |
|---|---|
| API (`requirements.txt`) | **2.87 GB** |
| Dashboard (`requirements-dashboard.txt`) | **803 MB** |
| Build context (after `.dockerignore`) | 2.9 MB, from a 7.3 GB repository |

`docker compose up` brings up both with health-gated ordering; a containerised
prediction returns a value byte-identical to the host run. An API container
with no model mounted reports `degraded` and refuses predictions with a 503
naming the fix.

---

## Tests

| Suite | Count |
|---|---|
| Unit | **229** |
| Integration | 13 (parity, live-model grounding, point-in-time assessment) |
| flake8 / Black / isort | clean |
| mypy | **0 errors**, blocking in CI |

On a clean checkout with no model and no generated data: **211 pass, 18 skip**
from the unit suite — tests needing a trained model skip rather than fail. The
13 integration tests skip in their entirety, for the same reason.

---

## The horizon is real, and observable

The API accepts an `as_of` timestamp and hides everything after it — telemetry,
errors, and maintenance alike — so a historical assessment sees only what was
known at the time. Rewinding to fixed offsets before the eleven failures in the
last quarter of the data:

| Rewound to | Alerts |
|---|---|
| 6 h before the failure | **5 of 5** sampled machines |
| 36 h before the failure | **0 of 5** |

The second row is the more informative one. A model that fired 36 hours out
would mean the 24-hour labels reach further than they claim, or that filtering
is not actually hiding the future. Silence there is the evidence that neither is
happening, and it is asserted by
`tests/integration/test_time_travel.py::test_the_model_stays_quiet_beyond_its_horizon`.

Concretely: machine 51 fails at 2024-10-31 12:00. Assessed at 06:00 that day the
model returns **p ≈ 1.0000**; assessed 24 hours earlier, **p ≈ 0.0000**.

---

## What these numbers do not establish

1. **The data is synthetic**, with a degradation pattern designed to be
   detectable — a 48-hour ramp, per-machine offsets, age-correlated noise. The
   *pipeline* transfers to real equipment; these *metrics* would not.
2. **Eight failure events** is a small test sample. The confidence interval on
   "100% of events caught" is wide.
3. **The deployed threshold barely transferred.** On validation it achieved
   zero misses; on test it missed 15. Tuning an operating point on 175
   positives does not generalise as cleanly as the validation figure suggests.
4. **The cost curve has a cliff.** t=0.6678 sits just before a ~3× jump in
   total cost. A threshold nearer 0.3–0.5 sits on the flat part at almost
   identical cost and would be the safer deployment.
5. **No auth.** The API is unauthenticated by design for v1.

---

## Where each figure comes from

| Artifact | Contains |
|---|---|
| `models/metrics.json` | Test metrics at threshold 0.5 |
| `models/evaluation_report.json` | Threshold sweep + test metrics at both points |
| `models/training_history.json` | Per-epoch loss, AUC, precision, recall, F1 |
| `models/training_curves.png` | Train vs validation, four panels |
| `models/pr_curve.png` | PR curve with both operating points, and cost vs threshold |
| [`docs/Day4.md`](Day4.md) – [`docs/Day11.md`](Day11.md) | The reasoning behind each number, including what went wrong |
