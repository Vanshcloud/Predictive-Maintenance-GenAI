# Roadmap

What this project does not do yet, why, and roughly in what order it is worth
doing.

**Nothing here is scheduled or promised.** This is a portfolio and reference
project with a single maintainer; the list exists so that limitations are
stated rather than discovered. Items are ordered by value, not by ease.

Anything already true is in the [CHANGELOG](../CHANGELOG.md), not here.

---

## Status legend

| | |
|---|---|
| 🔴 | Blocking real deployment |
| 🟡 | Meaningful gap, not blocking |
| 🟢 | Enhancement |
| 💭 | Idea — no design work done |

---

## Before this could face an untrusted network

Three items, all 🔴. Together they are hours of work, not a rewrite — but until
they are done, this belongs on `localhost` or a trusted network. See
[`../SECURITY.md`](../SECURITY.md).

### 🔴 Authentication and authorisation

No endpoint has any. `POST /report` invokes a language model, so with a
provider key configured an anonymous caller can spend against the account in a
loop.

*Approach:* API-key middleware for machine-to-machine use is probably right —
this is not a user-facing product and full OAuth would be over-built. A
reverse proxy handles it today; see
[`deployment.md`](deployment.md#behind-a-reverse-proxy).

### 🔴 Rate limiting

None. `/report` most urgently, then the prediction endpoints.

### 🔴 Bounded request bodies

`PredictRequest.readings` enforces a minimum of 48 but **no maximum**, so a
large payload becomes proportional memory and CPU inside pandas.
`ReportRequest.question` has no length cap and is interpolated into the model
prompt.

*Approach:* `max_length` on both. One-line Pydantic constraints; the reason
they are not done is that changing them alters the API contract.

---

## Model and data

### 🔴 Validate against a real dataset

**The single highest-value item in this document.**

The dataset is synthetic, with a degradation pattern deliberately designed to
be detectable. The pipeline transfers; the numbers would not. Until this system
has scored real telemetry, every metric it reports describes a problem someone
constructed.

*Approach:* NASA C-MAPSS or the Azure Predictive Maintenance dataset. Expect
substantially worse numbers — that is the point of doing it.

### 🟡 Benchmark against gradient boosting

The LSTM was chosen on reasoning, not on a head-to-head comparison. LightGBM on
flattened windows is the baseline this project owes. If it wins, that is worth
knowing and worth publishing.

### 🟡 Probability calibration

The outputs are not calibrated. A reported 0.9 does not mean 90% of such cases
fail, which is exactly what a reader assumes. Reliability diagrams plus
isotonic regression would make the number mean what it appears to mean.

### 🟡 Per-component prediction

The dataset labels *which* component failed; the model predicts only that *a*
failure is coming. The labels already exist.

### 🟡 Input-drift detection

Nothing checks whether live data still resembles the training distribution. A
model that silently goes stale is a worse failure than one that errors.

### 🟢 Uncertainty estimates

MC dropout would let a report say "unsure" instead of producing a confident
number, which matters more for a maintenance decision than for a benchmark.

### 💭 Temporal attribution

Attention weights or SHAP over timesteps, so the evidence block can say *which
hours* drove the prediction rather than only which sensors.

---

## Engineering

### 🟡 `scripts/predict.py --machine N` scores the entire fleet

It passes the full dataset to `predict_machine()`, building 873,600 sequences
to return one row — about **3 minutes** for something the API does in 137 ms.
`MachineDataStore.slice_for()` already solves this; the CLI simply does not use
it.

Untouched so far because it changes the inference path and deserves a
deliberate change rather than an opportunistic one.

### 🟡 `explain_machine()` engineers features twice

Once directly, then again via `predict_machine()`. Measured at 158 ms against
120 ms — roughly **24% overhead** on every explain and every report.

### 🟢 Batch the fleet scoring

`/fleet` scores 100 machines in a Python loop. Batching into a single model
call should cut the cold path substantially. **Not measured.**

### 🟢 Metrics endpoint

No Prometheus endpoint, no request histograms, no model-level counters. There
is no way to answer "how often did we alert last week?" without parsing logs.

### 💭 Streaming ingestion

Today the API reads a CSV at startup. A real plant emits a stream. Kafka or
MQTT ingestion with a rolling window would make this an actual monitoring
system rather than a batch scorer with an HTTP interface.

### 💭 Model registry and versioned rollback

A retrain is a file swap and a restart. That works, and it does not give you
lineage, comparison, or a one-command rollback. MLflow is the obvious fit.

### 💭 Continuous training

Scheduled retrains gated on held-out performance, so a model only ships if it
beats its predecessor.

---

## Deployment

### 🟢 Kubernetes manifests

Compose is what exists. Deployment, Service, and HPA manifests with the
readiness probe already wired to `/health` would be a small, honest addition.

### 🟢 GPU inference

Everything here is CPU-only and has never been measured on a GPU. Likely a
large improvement to `/fleet`, little to single predictions — those are
dominated by pandas feature engineering, not the model.

### 💭 Cloud deployment guide

Nothing has ever been deployed to a cloud provider. A guide would be written
from documentation rather than experience, which is why there isn't one.

### 💭 Grafana dashboard

Depends on the metrics endpoint above.

---

## Documentation

### 🟢 Demo GIF

`docs/images/` holds the horizon chart and the three dashboard screenshots,
all generated by committed scripts. A short walkthrough GIF is still missing —
see [`images/README.md`](images/README.md) for what it should show.

### 🟢 Architecture decision records

The devlog captures decisions with their reasoning, but chronologically. A
small set of ADRs would make individual decisions findable without reading
fifteen entries.

---

## Explicitly not planned

Stating these saves someone proposing them.

| | Why not |
|---|---|
| **Multi-tenancy** | A reference implementation for a single fleet. Tenancy would touch every layer for no demonstrative gain. |
| **A web UI beyond Streamlit** | The dashboard exists to prove the API is complete enough to build on. A React front end would add a build toolchain and teach nothing new. |
| **PyPI publication** | Not a library. It is a service and a pipeline, and `pip install predictive-maintenance-genai` would imply an API contract that is not intended. |
| **Supporting Python 3.13+** | Blocked upstream: TensorFlow publishes no wheels. Will follow when it does. |
| **A larger model** | 149,825 parameters already reach F1 0.9086 on this data. A bigger model would fit the synthetic degradation pattern harder, not generalise better. |

---

## Contributing to any of this

Items marked 🟢 or 💭 are good first contributions — they are additive and do
not touch the invariants that fail silently. 🔴 and 🟡 items in the model or
data layers need a careful read of
[`development.md`](development.md#common-mistakes) first.

Open an issue before starting anything substantial.
