# The Model

What it predicts, how it was built, and — at the end and in detail — what it
does not establish.

Every number here is reproducible from a clean checkout. Full metrics with
their caveats are in [`RESULTS.md`](RESULTS.md).

---

## Contents

- [The problem](#the-problem)
- [Why an LSTM](#why-an-lstm)
- [Architecture](#architecture)
- [Preprocessing and feature engineering](#preprocessing-and-feature-engineering)
- [Scaling](#scaling)
- [The temporal split](#the-temporal-split)
- [Class imbalance](#class-imbalance)
- [Evaluation metrics](#evaluation-metrics)
- [Threshold selection](#threshold-selection)
- [Results](#results)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## The problem

**Given 24 hours of sensor history for one machine, will it fail within the
next 24 hours?**

Binary classification over a sliding window. Input is `(24, 63)` — twenty-four
hourly timesteps, sixty-three engineered features each. Output is a single
probability.

The labelling is the part worth stating precisely: every hour inside the 24
hours preceding a failure carries label `1`. So the model is not predicting
"the machine fails at this exact hour" — it is predicting "a failure is coming
within the horizon", which is the question a maintenance scheduler actually
has.

---

## Why an LSTM

The signal is **degradation over time**, not an instantaneous threshold. A
vibration reading of 41 mm/s means nothing alone; 41 mm/s after a week at 38,
climbing, with pressure falling, means something.

| Considered | Why not |
|---|---|
| Threshold rules | The failure signature is multivariate and gradual. A per-sensor threshold either fires constantly or too late; that is the status quo this project exists to improve on. |
| Gradient boosting on flattened windows | A strong baseline and genuinely competitive — but it discards ordering within the window unless you hand-engineer it back, and "vibration rose then fell" and "fell then rose" become the same row. |
| 1D CNN | Good at local shape, weaker at the long-range dependency between an early error burst and a later failure. |
| Transformer | Sequence length here is 24. Self-attention's advantage appears at far longer contexts, and it brings a much larger parameter count for a dataset this size. |
| **LSTM** | ✅ Purpose-built for ordered sequences with long-range dependencies, small enough to train on CPU, and it keeps the temporal structure the problem is made of. |

**Honest caveat:** gradient boosting was not benchmarked head-to-head on this
dataset. The LSTM was chosen on the reasoning above, and it performs well — but
"better than the alternatives here" is not a claim this project has earned. See
[Future improvements](#future-improvements).

---

## Architecture

**149,825 parameters · 1.76 MB on disk.**

```
Input                 (batch, 24, 63)
  LSTM(128, return_sequences=True)      →  Dropout(0.3)
  LSTM(64)                              →  Dropout(0.3)
  Dense(32, relu)
  Dense(1, sigmoid)                     →  P(failure within 24 h)
```

Two stacked LSTM layers: the first returns the full sequence so the second sees
a temporal representation rather than a single summary vector. Dropout at 0.3
between them — with 800 positive training examples, overfitting is the
principal risk.

Training uses a hand-written `GradientTape` loop rather than `model.fit()`.
That decision has a long story (see [`devlog/day-04.md`](devlog/day-04.md));
the short version is that it is kept because it works, is tested, and makes
class weighting, early stopping, and checkpointing explicit rather than hidden
inside framework callbacks.

---

## Preprocessing and feature engineering

Five raw tables → 63 features. Four base sensors (voltage, rotation, pressure,
vibration) plus 48 engineered features and machine metadata.

| Family | Windows | Count | What it captures |
|---|---|---|---|
| Rolling mean | 3 h, 12 h, 24 h | 12 | Trend direction |
| Rolling std | 3 h, 12 h, 24 h | 12 | Volatility — voltage fails by becoming erratic, not by drifting |
| Lag | 1 h, 6 h, 24 h | 12 | Where it was |
| Change | 1 h, 6 h, 24 h | 12 | Rate of change |
| Error counts | 24 h rolling | ~6 | Fault bursts precede failures |
| Hours since maintenance | per component | 4 | Overdue components fail more |
| Machine metadata | — | ~5 | Age, one-hot model |

Two details that matter more than they look:

**`hours_since_maint_*` uses `9999` as the sentinel for "never serviced",
not `0`.** Zero means *serviced this hour* — the exact opposite. Getting the
fill value wrong here is worse than leaving the feature out.

**Absent categories are reconciled explicitly at inference.** One-hot `model_*`
columns only exist for machine models present in the scored batch. Score three
machines that are all `model1` and the matrix is narrower than the 63 columns
the model was trained on. `Predictor._reconcile_features()` fills those with
the value the training pipeline would have produced — and **raises** for any
missing feature that has no defensible default, because an absent sensor column
is a broken feed, not a sparse category.

---

## Scaling

`StandardScaler`, **fitted on the training split only**, then applied to
validation, test, and every inference call.

`apply_scaler()` exists as a separate method from `normalize()` specifically to
make refitting hard to do by accident. Refitting per split leaks that split's
distribution into its own features and degrades predictions silently while
every test still passes — the classic training/serving skew bug.

The fitted scaler is also the reference distribution used to judge whether a
sensor reading is unusual, so the "1.91σ below baseline" in a report is
measured against the population the model was trained on rather than a number
someone picked.

---

## The temporal split

**Chronological. Never random.**

```
|<------------- train ------------->|<-- val -->|<-- test -->|
        567,000                        129,000      172,800
```

| Split | Sequences | Positives | Rate |
|---|---|---|---|
| Train | 567,000 | 800 | 1:709 |
| Validation | 129,000 | 175 | 1:737 |
| Test | 172,800 | 200 | 1:864 |

With 24-hour lag features, a random split puts hour *t* in training and *t+1*
in test. The model then "predicts" a failure whose evidence it has already
memorised, and the reported AUC becomes fiction.

The **validation split exists to absorb every choice**: early stopping,
checkpoint selection, and threshold tuning all *choose* something based on the
data they observe. If that data is the test set, the reported test score has
been optimised against and is no longer an estimate of generalisation. The test
set is scored exactly once, at the very end.

Split boundaries are positional, then converted to timestamps, so all rows
sharing a timestamp land in the same split.

---

## Class imbalance

Roughly **0.13% positive** — about 1 failure hour per 745.

Handled with class-weighted binary cross-entropy, `{0: 0.50, 1: 364.89}`,
computed as the inverse class ratio. Resampling was not used: SMOTE on a
time-series window synthesises sequences that never occurred, and undersampling
discards the majority-class variety the model needs in order to learn what
*normal* looks like.

---

## Evaluation metrics

**Accuracy is not reported anywhere, and `ModelEvaluator` does not compute it.**

At a 1:864 positive rate, a model that predicts "no failure" for every row
scores **99.88%**. Reporting that would be actively misleading, so the metric
is absent rather than merely de-emphasised.

| Metric | Why it is here |
|---|---|
| **Precision** | False alarms cost inspections and, eventually, credibility. A model nobody trusts is not deployed. |
| **Recall** | A missed failure stops a production line. |
| **F1** | The headline. Balances both, and unlike AUC it actually moves under this imbalance. |
| **ROC-AUC** | Reported, but see the caveat below. |
| **Average precision** | More informative than ROC-AUC when positives are rare. |
| **Event-level recall** | See below. |

**ROC-AUC saturates here.** Across one full run it stayed within
`0.9991 – 1.0000` while validation precision swung from `0.13` to `0.81`. It is
reported for completeness and was explicitly *not* used for model selection —
selecting on a saturated metric is selecting on noise.

### Event-level recall

Sequence-level recall counts *hours*. Every hour in the 24 before a failure is
labelled `1`, so missing 9 of 24 while catching the other 15 scores as 9
"misses" — when operationally the technician was warned once, which is all that
was needed.

Measured over **events** instead: **8 of 8 caught (100%)**, median lead time
**23.5 h**, worst case **16 h**.

Eight events is a small sample. This says the model warned in 8 of 8 cases, not
that it never misses — and it is reported *alongside* precision, never instead
of it, because event recall says nothing about alert fatigue.

---

## Threshold selection

`0.5` is inherited from balanced problems and is almost never right at a 1:864
positive rate. The threshold is a **deployment decision** the sigmoid output
already supports without retraining.

The precision–recall curve is swept on the **validation** split, and the test
set is then scored once at the chosen point.

| Operating point | Threshold | Test F1 | False alarms |
|---|---|---|---|
| Cost-optimal (100:1) | 0.00000015 | 0.8301 | 72 |
| Default | 0.5 | 0.9082 | 20 |
| **Best-F1 (deployed)** | **0.3415** | **0.9086** | **21** |

**The cost-weighted objective failed at this sample size, and that is
documented rather than hidden.** With 175 validation positives and a 100:1
cost ratio, minimising cost collapses to "reach recall 1.0 at any price", and
the cheapest route is a noise-floor threshold that does not transfer. It cost 8
points of test F1.

`sweep_thresholds()` now returns a `lowest_cost_is_degenerate` flag so the
failure mode cannot recur silently.

`RISK_BAND_HIGH` is pinned equal to `PREDICTION_THRESHOLD` by a test, so the
band boundary and the alert decision cannot drift apart. A second test pins the
served threshold to the committed evaluation report, so a retrain that moves
the optimum cannot silently leave the old value in service.

---

## Results

Threshold `0.3415`, test set scored once. Training is seeded, so
`python scripts/train_model.py --seed 42` re-derives these rather than landing
near them.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9999 |
| Precision | 0.8976 |
| Recall | 0.9200 |
| **F1** | **0.9086** |
| Events caught | **8 / 8** |
| Lead time (median / min / max) | 23.5 h / 16 h / 24 h |

```
                 predicted 0   predicted 1
actual 0            172,579            21     <- false alarms
actual 1                 16           184
```

---

## Limitations

**Read this section before drawing any conclusion from the numbers above.**

### The dataset is synthetic

`scripts/generate_data.py` produces it from a fixed seed, with a degradation
pattern **deliberately designed to be detectable**. These metrics reflect
*this dataset's* difficulty, not that of real industrial equipment.

**The pipeline transfers. The numbers would not.** Real telemetry has sensor
dropout, calibration drift, mislabelled failures, maintenance that is performed
but not recorded, and failure modes with no sensor signature at all. Expect
substantially lower performance on real data — that is not a defect in this
system, it is what the synthetic-to-real gap looks like.

### Eight events is a small sample

Event-level recall of 8/8 has wide error bars. One additional missed event
would make it 8/9 — 89%.

### The horizon is fixed at 24 hours

The model was trained to see 24 hours ahead and no further. A day before a
failure it is silent, and *should* be. It cannot be asked "will this fail next
week?" without retraining at a different `prediction_horizon`.

### It predicts *that*, not *which*

The dataset labels which component failed. The current model predicts only that
a failure is coming — component-level prediction is left on the roadmap.

### Cost weighting is unvalidated

The 100:1 false-negative-to-false-alarm ratio is a placeholder, not a plant's
real economics. The cost-optimal threshold it produces was rejected as
degenerate. Supply real numbers before relying on that mode.

### No drift detection

Nothing monitors whether live data still resembles the training distribution.
A model that silently goes stale is a worse failure than one that errors.

### Not validated in production

This has never scored a real machine. There is no deployment, no operator
feedback loop, and no measured business outcome.

---

## Future improvements

Ordered by expected value, not by ease.

1. **Validate against a real dataset.** By far the highest-value next step —
   NASA C-MAPSS or the Azure Predictive Maintenance set would test whether the
   pipeline survives contact with real telemetry.
2. **Benchmark against gradient boosting.** LightGBM on flattened windows is
   the honest baseline this project owes. If it wins, that is worth knowing.
3. **Per-component prediction.** The labels already exist.
4. **Calibration.** The probabilities are not calibrated; a reported 0.9 does
   not mean 90% of such cases fail. Reliability diagrams and isotonic
   regression would make the number mean what a reader assumes it means.
5. **Drift detection** on the input distribution.
6. **Uncertainty estimates** — MC dropout would give the report a way to say
   "unsure" instead of a confident number.
7. **Attention or SHAP over timesteps**, so the evidence block can say *which
   hours* drove the prediction rather than which sensors.

---

## See also

- [`RESULTS.md`](RESULTS.md) — every metric with its caveat
- [`training.md`](training.md) — how to reproduce this
- [`architecture.md`](architecture.md#training-pipeline) — the training-loop diagram
- [`devlog/day-05.md`](devlog/day-05.md) — where the threshold work happened
