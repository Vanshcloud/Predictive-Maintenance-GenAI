# IMPLEMENTATION_PLAN.md

**Single source of truth for the Predictive Maintenance + GenAI Insight Generator project.**

This document is written so that any developer — or any AI model with no prior conversation
history — can read it plus the matching `docs/DayX.md` file and continue the project
immediately. It must be updated whenever meaningful progress is made.

| Field | Value |
|---|---|
| **Last updated** | 2026-08-25 (end of Day 15 — full-repository production review) |
| **Current milestone** | **Day 12 of 12 — complete.** All milestones delivered; Days 13–14 are post-project. |
| **Overall completion** | **100% of the 12-day plan** |
| **Repository** | https://github.com/Vanshcloud/Predictive-Maintenance-GenAI |
| **Branch** | `main` |
| **Latest commit at time of writing** | `79c094a` (Day 3); Day 4 work is staged in the working tree |
| **Companion documents** | `docs/Day1.md` … `docs/Day15.md`, `docs/RESULTS.md` (consolidated metrics), `AGENTS.md` (agent instructions) |


> **History note (2026-08-23).** Every commit in this repository was rewritten to
> normalise the author name to `Vanshcloud` and to strip AI co-author trailers. File
> contents were unchanged, but **all commit hashes changed**. SHAs quoted in this
> document have been updated to the current ones. If you have an older hash from a
> screenshot or terminal log, use `git log --oneline` to find its replacement by message.

---

# Project Overview

## Project name

**Predictive Maintenance + GenAI Insight Generator** (repository name: `Predictive-Maintenance-GenAI`).

## Problem statement

Industrial equipment fails without warning. Unplanned downtime is the single largest
controllable cost in manufacturing: a stopped line costs money every minute, emergency
repairs cost more than scheduled ones, and secondary damage from running a machine to
destruction is expensive. Two conventional strategies both waste money:

- **Reactive maintenance** — fix it after it breaks. Maximum downtime, maximum damage.
- **Scheduled maintenance** — service every N hours regardless of condition. Replaces
  healthy parts, and still misses failures that arrive early.

What operators actually need is a *condition-based* answer: "which machines will fail in
the next 24 hours, and what should I do about it?" The first half of that question is a
time-series machine-learning problem. The second half is a communication problem — a
probability of `0.87` means nothing to a maintenance technician on a factory floor at
3 a.m. It has to become an actionable, plain-English work order.

## Objectives

1. Predict equipment failure **24 hours in advance** from multivariate sensor telemetry,
   using a TensorFlow LSTM, with quality measured by AUC/F1/precision/recall.
2. Translate each prediction into a **plain-English maintenance report** using LangChain
   and an LLM — cause, urgency, recommended action.
3. Expose both capabilities through a **FastAPI REST API** with validated schemas and
   auto-generated OpenAPI docs.
4. Provide an **interactive Streamlit dashboard** for non-technical users.
5. Ship the whole thing **containerized with CI**, documented well enough that a stranger
   can run it from a clean checkout.

## Scope

**In scope**

- Synthetic but realistic sensor dataset (generated locally, seed-reproducible).
- Full data pipeline: ingestion → validation → feature engineering → LSTM tensors.
- Binary classification model (`will this machine fail within 24h?`).
- LLM-backed report generation and a maintenance Q&A assistant.
- REST API + dashboard + Docker + GitHub Actions CI.

**Out of scope**

- Real plant/SCADA integration or live sensor streams.
- Multi-class failure-mode prediction (which component fails) — the dataset supports it,
  but the model target is deliberately binary for v1.
- Remaining-useful-life (RUL) regression.
- Kubernetes, autoscaling, multi-tenant auth, or a managed database. State lives in
  files and in-process caches.
- Model retraining automation / MLOps feedback loop.

## Expected deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Reproducible synthetic dataset generator (5 tables, 883K rows) | ✅ Done (Day 2) |
| 2 | Data ingestion + validation layer | ✅ Done (Day 2) |
| 3 | Feature engineering + LSTM sequence pipeline | ✅ Done (Day 3) |
| 4 | Trained LSTM model + evaluation metrics | ✅ Done (Day 5, retrained seeded Day 15) — **AUC 0.9999, F1 0.9086** on a clean 3-way split |
| 5 | Inference/prediction pipeline | ✅ Done (Day 6) — `Predictor`, parity with training verified at 100% |
| 6 | LangChain report generator + Q&A assistant | ✅ Done (Days 7–8) — grounded reports and multi-turn Q&A |
| 7 | FastAPI REST API | ✅ Done (Day 9) — 9 endpoints, 137 ms predictions, LLM path isolated |
| 8 | Streamlit dashboard | ✅ Done (Day 10) — pure API client, three views |
| 9 | Docker + docker-compose + GitHub Actions CI | ✅ Done (Day 11) — API 2.87 GB, dashboard 803 MB, compose verified |
| 10 | Final docs, demo, README polish | ✅ Done (Day 12) — clean-checkout verified, `docs/RESULTS.md` written |

## Target users

| User | What they need from the system |
|---|---|
| **Maintenance technician** | A ranked list of at-risk machines and a readable work order. Does not know or care what an LSTM is. |
| **Plant/operations manager** | Downtime risk overview, prioritization, evidence for scheduling decisions. |
| **Reliability / data engineer** | The API, the model metrics, the feature definitions, the ability to retrain. |
| **Reviewer of this portfolio project** | Evidence of end-to-end engineering: clean layering, tests, honest metrics on imbalanced data, documented decisions. |

## Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Load raw sensor data from CSV/Parquet/JSON with automatic format detection. |
| FR-2 | Validate incoming data against a schema: required columns, dtypes, nulls, duplicates, physical ranges. Emit a structured `ValidationReport`. |
| FR-3 | Engineer rolling, lag, and change features per machine from 4 raw sensors; join error, maintenance, and machine-metadata features. |
| FR-4 | Label every hour with 1 if that machine fails within the next 24 hours, else 0. |
| FR-5 | Split train/test **temporally**, fit the scaler on train only, and window sequences **within a single `machine_id`**. |
| FR-6 | Train an LSTM binary classifier that handles a ~1:730 class imbalance. |
| FR-7 | Report AUC, precision, recall, F1, and a confusion matrix — never bare accuracy. |
| FR-8 | Persist model, scaler, feature list, metrics, and training history as versioned artifacts. |
| FR-9 | Serve single and batch predictions over REST. |
| FR-10 | Generate an LLM maintenance report from a prediction (cause, urgency, action). |
| FR-11 | Answer free-form maintenance questions about a machine's history. |
| FR-12 | Render a dashboard: fleet overview, machine detail, prediction timeline, reports. |

## Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Reproducibility** — same seed produces the same dataset and same split | Exact |
| NFR-2 | **No data leakage** — test-set information must never reach training | Enforced by design + tests |
| NFR-3 | **Inference latency** (single sequence, CPU) | < 100 ms |
| NFR-4 | **API p95 latency** for `/predict` | < 500 ms |
| NFR-5 | **Memory ceiling** — training must run on a 16 GB laptop | Achieved via `mmap_mode='r'` |
| NFR-6 | **Test coverage** of `src/` and `config/` | ≥ 70% |
| NFR-7 | **Type coverage** — type hints on all public function signatures | 100% |
| NFR-8 | **Observability** — every module logs via loguru, never `print()` | Enforced by review |
| NFR-9 | **Portability** — runs from a clean checkout with documented commands | `make` targets |
| NFR-10 | **Secret hygiene** — no API keys in git; everything via `.env` | `.gitignore` + `.env.example` |

## Assumptions

1. Sensor telemetry arrives at a **fixed hourly cadence** per machine, with no gaps.
   (The generator guarantees this; a real feed would need resampling.)
2. A 24-hour prediction horizon is operationally useful — long enough to schedule a
   technician, short enough that the signal is present in the data.
3. A 24-timestep (24-hour) input window captures the relevant degradation dynamics,
   because the synthetic degradation ramps over ~48h before failure.
4. Failure events in the training data are correctly and completely labeled.
5. Machines are independent — one machine's failure does not cause another's. No
   cross-machine features are engineered.
6. The synthetic dataset's degradation patterns are a fair enough proxy for real
   equipment that the *pipeline* transfers, even if the learned weights would not.
7. A single-node CPU deployment is sufficient; no GPU is assumed at inference time.

## Constraints

| Constraint | Consequence |
|---|---|
| **Python must be 3.12** | TensorFlow does not support 3.13+. System Python 3.14 is unusable; the venv is built from Homebrew `python@3.12`. |
| **Apple Silicon / ARM64 macOS dev machine** | No CUDA. Training is CPU-bound. TensorFlow's first import takes ~90 s. |
| **TensorFlow must be imported before pandas/scikit-learn touch BLAS** | Otherwise the first `tf.function` execution deadlocks (see Risk R-2). Enforced by `tests/conftest.py` and by import order in scripts. |
| **Keras `fit()` / `predict()` are unusable on this platform** | Their background prefetch threads deadlock against memmapped reads. Training and inference both use hand-written synchronous loops (see Risk R-3). |
| **Training arrays are 4.2 GB (train) + 1.0 GB (test)** | They must be memmapped, never fully loaded. |
| **Severe class imbalance (~1:730)** | Accuracy is meaningless; class weighting is mandatory; a "predict always 0" model scores 99.86% accuracy. |
| **Raw and processed data are gitignored** | Only `data/sample/` is committed, and it contains **zero** failure events. Any test needing a positive label must synthesize one. |
| **No paid infrastructure** | Deployment targets free tiers; LLM calls are pluggable and can fall back to local Ollama. |
| **12-day timeline, one milestone per day** | Scope per day must stay small enough to finish, test, document, and commit in one session. |

---

# Technology Stack

## Languages

| Language | Version | Where used |
|---|---|---|
| **Python** | 3.12.14 | Everything. Chosen because it is the ML/AI standard *and* because it is the newest version TensorFlow supports. |
| **Bash** | — | `scripts/setup.sh`, Makefile targets. |
| **Dockerfile / YAML** | — | Day 11: containerization and GitHub Actions. |

## Frameworks

| Framework | Version | Purpose | Why this one |
|---|---|---|---|
| **TensorFlow / Keras** | 2.21.0 | LSTM model definition, training, inference | Production deployment story (TF Serving, TF Lite), native LSTM layer, TensorBoard. Chosen over PyTorch for deployment tooling, not research ergonomics. |
| **LangChain** | 0.3.30 | LLM orchestration, prompt templating, chains | Industry standard for composing LLM calls; gives prompt management, output parsing, and memory without hand-rolling them. |
| **FastAPI** | 0.141.1 | REST API | Async-first, Pydantic-native, auto-generates OpenAPI/Swagger. Chosen over Flask (no async, no schema) and Django (far too heavy). |
| **Streamlit** | 1.61.1 | Dashboard | Builds a data dashboard in Python alone, ~10× faster than React for this purpose. Sufficient for an MVP. |
| **Pytest** | 8.4.2 | Test framework | Fixtures, parametrization, plugin ecosystem. |

## Libraries

| Library | Version | Purpose |
|---|---|---|
| **NumPy** | 1.26.4 | Arrays, memmapped `.npy` I/O, batch assembly. Pinned `<2.0` for TensorFlow compatibility. |
| **Pandas** | 2.3.3 | Table merges, `groupby().rolling()` feature engineering, time indexing. |
| **scikit-learn** | 1.9.0 | `StandardScaler`, and all evaluation metrics (`roc_auc_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`). |
| **Pydantic** | 2.13.4 | Typed settings, and later API request/response schemas. |
| **pydantic-settings** | 2.x | Reads `.env` into a typed `Settings` object (12-Factor config). |
| **python-dotenv** | 1.x | `.env` loading. |
| **Loguru** | 0.7.3 | Structured logging: colored console sink + rotating file sink. |
| **Joblib** | 1.3+ | Serializes the fitted `StandardScaler`. |
| **Matplotlib / Seaborn** | 3.8+ / 0.13+ | EDA plots, and later training-curve plots. |
| **httpx** | — | Test client for the FastAPI layer (Day 9). |
| **langchain-openai / langchain-community / openai** | — | LLM providers (Day 7–8). |

## Databases

**None.** This is a deliberate decision. All state is file-based:

| Data | Storage |
|---|---|
| Raw tables | CSV under `data/raw/` (gitignored) |
| Model-ready tensors | `.npy` under `data/processed/` (gitignored, memmapped) |
| Fitted scaler | `data/processed/scaler.joblib` |
| Feature name list | `data/processed/feature_columns.txt` |
| Trained model | `models/*.keras` (gitignored) |
| Metrics / history | `models/metrics.json`, `models/training_history.json` (committed — they are the evidence) |
| Logs | `logs/app_YYYY-MM-DD.log` (rotated at 5 MB, kept 3 days, zipped) |

If prediction history needs to persist across restarts (a Day 9+ concern), SQLite is the
intended addition — not Postgres, which would add operational weight this project does
not need.

## Cloud services

None are required to run the project. Optional/planned:

| Service | Use | Required? |
|---|---|---|
| **OpenAI API** | Default LLM for report generation (`gpt-4o-mini`) | Optional — one of three providers |
| **Google Gemini API** | Alternative LLM (`gemini-1.5-flash`) | Optional |
| **Ollama (local)** | Fully local LLM (`llama3`), no API key, no network | Optional — the zero-cost path |
| **GitHub Actions** | CI: lint → test → build | Day 11 |
| **Container registry / free-tier host** | Deployment target | Day 11, TBD |

## APIs

- **Internal (produced by this project):** `GET /health`, `POST /predict`, `POST /report`,
  `GET /machines`, `GET /machines/{id}/history`. Specified in Deployment Plan below.
- **External (consumed):** OpenAI Chat Completions, Google Generative AI, or a local
  Ollama HTTP endpoint — all behind LangChain's model abstraction so the provider is a
  config value, not a code change.

## Tools

| Tool | Purpose | Config location |
|---|---|---|
| **Black** (24.10.0) | Formatting, line length 88 | `pyproject.toml` |
| **isort** | Import ordering, Black-compatible profile | `pyproject.toml` |
| **Flake8** (7.x) | Linting | `.flake8` |
| **Mypy** (1.8+) | Static type checking | `pyproject.toml` |
| **pytest-cov** | Coverage reporting | `pyproject.toml` |
| **pre-commit** | Local quality gate before commit | `.pre-commit-config.yaml` |
| **Make** | Command shortcuts | `Makefile` |
| **Git** | Version control | — |

## Development environment

| Field | Value |
|---|---|
| **OS** | macOS (ARM64 / Apple Silicon), Darwin 25.5.0 |
| **Python** | 3.12.14 from `brew install python@3.12` |
| **System Python** | 3.14.0 — **not used**, incompatible with TensorFlow |
| **Virtualenv** | `venv/`, created via `/opt/homebrew/bin/python3.12 -m venv venv` |
| **Activation** | `source venv/bin/activate` |
| **IDE** | VS Code |
| **Package manager** | pip, with pinned `requirements.txt` |
| **Installed packages** | ~218 including transitive deps |
| **Accelerator** | None. CPU-only training. |

## Version requirements

```
python        == 3.12.x        # hard requirement: TF has no 3.13+ wheels
numpy         >= 1.24, < 2.0   # hard requirement: TF 2.21 is not NumPy-2 clean here
pandas        >= 2.0,  < 3.0
scikit-learn  >= 1.3,  < 2.0
tensorflow    >= 2.15, < 3.0
langchain     >= 0.2,  < 1.0
fastapi       >= 0.110, < 1.0
uvicorn[standard] >= 0.27, < 1.0
streamlit     >= 1.30, < 2.0
pydantic      >= 2.0,  < 3.0
pydantic-settings >= 2.0, < 3.0
loguru        >= 0.7,  < 1.0
joblib        >= 1.3,  < 2.0
```

## Installation instructions

```bash
# 1. Clone
git clone https://github.com/Vanshcloud/Predictive-Maintenance-GenAI.git
cd Predictive-Maintenance-GenAI

# 2. Create the 3.12 virtualenv (3.13+ will fail on the TensorFlow install)
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

# 3. Install
pip install --upgrade pip
pip install -r requirements.txt          # production
pip install -r requirements-dev.txt      # + test/lint/notebook tooling

# 4. Configure
cp .env.example .env                     # then fill in an LLM API key if you want reports

# 5. Regenerate the gitignored data (takes a couple of minutes)
python scripts/generate_data.py          # -> data/raw/  (883,231 rows)
python scripts/run_preprocessing.py      # -> data/processed/*.npy (~5.2 GB)

# 6. Verify
make test                                # 75 tests; first run pays ~90s for the TF import

# 7. Train
python scripts/train_model.py --epochs 30
```

`scripts/setup.sh` automates steps 2–4.

---

# System Architecture

## Layering rule

The system is a strictly ordered stack. **Each layer may only depend on layers to its
left.** No layer may reach into a later layer's internals, and no layer may be skipped.

```
config/  →  src/utils/  →  src/data/  →  src/models/  →  src/prediction/  →  src/genai/  →  src/api/  →  dashboard/
```

This is the single most important structural invariant in the project. It is what makes
each layer independently testable and independently deployable, and it is why
`src/data/` contains no TensorFlow import and `src/models/` contains no pandas import.

## Module descriptions and responsibilities

### `config/` — configuration layer

| Item | Detail |
|---|---|
| **File** | `config/settings.py` |
| **Responsibility** | Own every tunable value: paths, ports, model names, hyperparameter defaults, LLM provider/keys. |
| **Public API** | `get_settings()` — an `lru_cache`d factory returning a single `Settings` (pydantic-settings) instance. |
| **Depends on** | Nothing (except pydantic). |
| **Rule** | Never hardcode a path, port, or model name anywhere else. Add a field here instead. |
| **Key derived paths** | `settings.processed_data_path`, `settings.model_artifacts_path`, `settings.raw_data_path` |

### `src/utils/` — cross-cutting utilities

| File | Responsibility |
|---|---|
| `logger.py` | `get_logger(__name__)` → a configured loguru logger. Colored console sink + rotating file sink (`logs/app_{date}.log`, 5 MB rotation, 3-day retention, zip compression). **`enqueue=True` is deliberately off** — its background writer thread contributed to the TensorFlow deadlock family described in Risk R-2/R-3. |
| `exceptions.py` | The exception hierarchy. Everything inherits `PredMaintenanceError`, grouped by layer: `DataIngestionError`, `DataValidationError`, `DataPreprocessingError`, `ModelNotFoundError`, `ModelTrainingError`, `PredictionError`, `LLMConnectionError`, `ReportGenerationError`, `APIError`, `ResourceNotFoundError`. Callers can catch narrowly (`except DataValidationError`) or broadly (`except PredMaintenanceError`). |

### `src/data/` — data pipeline

| Class | File | Responsibility |
|---|---|---|
| `DataIngestion` | `ingestion.py` | Load a table from disk, detecting CSV/Parquet/JSON by extension. Logs row/column counts and memory footprint. Raises `DataIngestionError`. |
| `DataValidator` | `validation.py` | Run schema, null, duplicate, and physical-range checks. Produces a structured `ValidationReport` rather than raising, so callers decide whether to proceed. |
| `DataPreprocessor` | `preprocessing.py` | The largest module (~780 lines). Merges 5 tables → engineers 48 features → creates labels → temporal split → scale → window. See Dataset Documentation. |

### `src/models/` — machine learning

| Class / function | File | Responsibility |
|---|---|---|
| `PredictiveMaintenanceModel` | `lstm_model.py` | Owns the Keras architecture. Builds it, saves it, loads it. Knows nothing about training or data. |
| `ModelTrainer` | `trainer.py` | Compiles (optimizer + loss), computes class weights, and runs the **hand-written training loop**. Implements early stopping, LR reduction, and checkpointing inline because Keras callbacks are unavailable outside `fit()`. |
| `iter_batches()` | `trainer.py` | Module-level generator yielding `(X_batch, y_batch)` float32 arrays from memmapped inputs. Shuffles epoch order but sorts indices *within* a batch so memmap reads stay monotonic. |
| `ModelEvaluator` | `evaluator.py` | Computes AUC, precision, recall, F1, confusion matrix. Runs inference through `_predict_in_batches()` (direct `model(x)` calls), never `model.predict()`. |

### `src/prediction/` — inference (Day 6, scaffold only)

Planned `Predictor` class: load model + scaler + feature list, accept raw sensor rows,
reuse `DataPreprocessor`'s feature logic, emit `{machine_id, failure_probability,
risk_level, contributing_features}`. It is the boundary between "ML artifacts" and
"application" — nothing above this layer should ever import TensorFlow.

### `src/genai/` — LLM layer (Day 7–8, scaffold only)

| Planned file | Responsibility |
|---|---|
| `prompts.py` | Prompt templates (report generation, failure explanation, Q&A) with a maintenance-expert system persona. |
| `chains.py` | LangChain chains: `report_chain`, `qa_chain`. Provider selected from settings. |
| `assistant.py` | Conversational Q&A over a machine's prediction and maintenance history. |

### `src/api/` — REST layer (Day 9, scaffold only)

`main.py` (app, middleware, lifespan model loading), `schemas.py` (Pydantic
request/response models), `routes/health.py|predict.py|reports.py`.

### `dashboard/` — Streamlit UI (Day 10, scaffold only)

Overview / Machine Detail / Predictions / Reports pages, talking to the API over HTTP.

## Dependencies between components

```
config.settings ──────────────────────────────┐
      │                                       │ (everyone reads settings)
      ▼                                       │
src.utils.logger, src.utils.exceptions ───────┤ (everyone logs and raises)
      │                                       │
      ▼                                       │
src.data.ingestion ─→ src.data.validation ─→ src.data.preprocessing
                                                    │ produces .npy + scaler
                                                    ▼
                                    src.models.lstm_model
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
                src.models.trainer                      src.models.evaluator
                        │  produces .keras + metrics.json       │
                        └───────────────────┬───────────────────┘
                                            ▼
                                   src.prediction.predictor
                                            │
                                            ▼
                                      src.genai.chains
                                            │
                                            ▼
                                        src.api.main
                                            │
                                            ▼
                                       dashboard/app.py
```

## Data flow (end to end)

```
┌── DATA LAYER ────────────────────────────────────────────────────────────┐
│  5 raw CSVs (telemetry, machines, errors, maintenance, failures)         │
│      ↓ DataIngestion            format detection, metadata logging       │
│      ↓ DataValidator            schema / null / duplicate / range        │
│      ↓ DataPreprocessor                                                  │
│          ├─ merge 5 tables on (datetime, machine_id)                     │
│          ├─ rolling mean/std over 3h / 12h / 24h                         │
│          ├─ lag features at 1h / 6h / 24h, and change = now − lag        │
│          ├─ error counts + hours-since-last-maintenance                  │
│          ├─ label = 1 if machine fails within next 24h                   │
│          ├─ TEMPORAL split (train: Jan–Oct, test: Oct–Dec)               │
│          ├─ StandardScaler .fit(train) → .transform(train), .transform(test) │
│          └─ sliding 24-step windows, never crossing machine_id           │
│      → X_train (698400, 24, 63) · y_train (698400,)                      │
│        X_test  (172800, 24, 63) · y_test  (172800,)                      │
├── ML LAYER ──────────────────────────────────────────────────────────────┤
│      ↓ PredictiveMaintenanceModel      2×LSTM + dropout + dense          │
│      ↓ ModelTrainer                    manual loop, class-weighted BCE   │
│      ↓ ModelEvaluator                  AUC / P / R / F1 / confusion      │
│      → models/lstm_predictive_maintenance.keras + metrics.json           │
│      ↓ Predictor                                                          │
│      → {machine_id, failure_probability, risk_level}                     │
├── GENAI LAYER ───────────────────────────────────────────────────────────┤
│      ↓ prompt template + prediction context                              │
│      ↓ LangChain chain → LLM (OpenAI / Gemini / Ollama)                  │
│      → plain-English maintenance report                                  │
├── SERVING LAYER ─────────────────────────────────────────────────────────┤
│      ↓ FastAPI  /predict  /report  /machines  /health                    │
│      ↓ Streamlit dashboard (HTTP client of the API)                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component communication

| Boundary | Mechanism |
|---|---|
| data → models | Files on disk (`.npy` memmaps, `scaler.joblib`, `feature_columns.txt`). Deliberately decoupled: training does not re-run preprocessing. |
| models → prediction | Serialized `.keras` artifact loaded through `PredictiveMaintenanceModel.load()`. |
| prediction → genai | In-process Python objects (dicts of prediction fields). |
| genai → LLM provider | HTTPS via LangChain provider adapters (or local HTTP for Ollama). |
| api → everything | In-process imports; the model is loaded once at app startup, not per request. |
| dashboard → api | HTTP/JSON. The dashboard has **no** direct model access — it is a pure API client, so it can be deployed separately. |

---

# Folder Structure

```
predictive-maintenance-genai/
│
├── IMPLEMENTATION_PLAN.md        # ← THIS FILE. Single source of truth.
├── AGENTS.md                     # Instructions for AI agents working in this repo
├── LICENSE                       # MIT
├── README.md                     # Public-facing project README with badges
├── Makefile                      # make test / lint / format / typecheck / quality
├── pyproject.toml                # PEP 621 metadata + black/isort/mypy/pytest config
├── requirements.txt              # Pinned production dependencies
├── requirements-dev.txt          # Dev-only dependencies (extends production)
├── .env                          # Secrets — GITIGNORED
├── .env.example                  # Template listing every required env var
├── .flake8                       # Flake8 config (Black-compatible)
├── .gitignore                    # Python + data + model artifacts
│
├── config/                       # ══ CONFIGURATION LAYER ══
│   ├── __init__.py               # Re-exports get_settings
│   └── settings.py               # Pydantic Settings, cached factory, derived paths
│
├── src/                          # ══ ALL SOURCE CODE ══
│   ├── __init__.py
│   │
│   ├── utils/                    # ── SHARED UTILITIES (no deps on other src/ pkgs) ──
│   │   ├── __init__.py           # Re-exports get_logger
│   │   ├── logger.py             # Loguru setup: console + rotating file sink
│   │   └── exceptions.py         # PredMaintenanceError hierarchy, grouped by layer
│   │
│   ├── data/                     # ── DATA PIPELINE (Days 2-3) ──
│   │   ├── __init__.py           # Exports DataIngestion, DataValidator, DataPreprocessor
│   │   ├── ingestion.py          # Format-detecting loader + metadata logging
│   │   ├── validation.py         # Schema/null/duplicate/range checks → ValidationReport
│   │   └── preprocessing.py      # ~780 lines: the whole raw→tensor pipeline
│   │
│   ├── models/                   # ── ML MODELS (Day 4-5) ──
│   │   ├── __init__.py           # Exports the three classes below
│   │   ├── lstm_model.py         # PredictiveMaintenanceModel: build / save / load
│   │   ├── trainer.py            # ModelTrainer + iter_batches: manual training loop
│   │   └── evaluator.py          # ModelEvaluator: imbalance-aware metrics
│   │
│   ├── prediction/               # ── INFERENCE (Day 6) ── scaffold
│   │   └── __init__.py
│   │
│   ├── genai/                    # ── LANGCHAIN + LLM (Day 7-8) ── scaffold
│   │   └── __init__.py
│   │
│   └── api/                      # ── FASTAPI (Day 9) ── scaffold
│       ├── __init__.py
│       └── routes/
│           └── __init__.py
│
├── scripts/                      # ══ EXECUTABLE ENTRY POINTS ══
│   ├── setup.sh                  # One-command environment setup
│   ├── generate_data.py          # Synthetic generator (--sample for the small set)
│   ├── eda_analysis.py           # 8-dimension EDA report
│   ├── run_preprocessing.py      # raw CSVs → data/processed/*.npy + scaler
│   └── train_model.py            # load → train → evaluate → save metrics + history
│
├── data/                         # ══ DATA (mostly gitignored) ══
│   ├── raw/                      # GITIGNORED — regenerate with generate_data.py
│   │   ├── telemetry.csv         # 876,000 rows — hourly sensor readings
│   │   ├── machines.csv          # 100 rows — machine_id, model, age
│   │   ├── errors.csv            # 5,386 rows — non-failure error events
│   │   ├── maintenance.csv       # 1,698 rows — component replacements
│   │   └── failures.csv          # 47 rows — actual failure events
│   ├── processed/                # GITIGNORED — regenerate with run_preprocessing.py
│   │   ├── X_train.npy           # (698400, 24, 63) float32 — 4.2 GB, memmapped
│   │   ├── y_train.npy           # (698400,)  — 957 positives
│   │   ├── X_test.npy            # (172800, 24, 63) — 1.0 GB, memmapped
│   │   ├── y_test.npy            # (172800,)  — 200 positives
│   │   ├── scaler.joblib         # StandardScaler fitted on TRAIN ONLY
│   │   └── feature_columns.txt   # The 63 feature names, in order
│   └── sample/                   # COMMITTED — small fixture set for tests
│       ├── telemetry.csv         # 7,200 rows (10 machines × 30 days)
│       ├── machines.csv          # 10 rows
│       ├── errors.csv            # 53 rows
│       ├── maintenance.csv       # 18 rows
│       └── failures.csv          # 0 rows  ← intentionally has NO failures
│
├── models/                       # ══ MODEL ARTIFACTS ══
│   ├── .gitkeep
│   ├── lstm_predictive_maintenance.keras   # GITIGNORED — the trained model
│   ├── metrics.json              # COMMITTED — evaluation results
│   └── training_history.json     # COMMITTED — per-epoch loss/AUC/precision/recall
│
├── tests/                        # ══ TESTS ══
│   ├── __init__.py
│   ├── conftest.py               # CRITICAL: imports TensorFlow first (deadlock fix)
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_smoke.py         # 19 tests — imports, config, logging, exceptions
│   │   ├── test_data_pipeline.py # 22 tests — ingestion + validation
│   │   ├── test_preprocessing.py # 27 tests — features, labels, split, windows
│   │   └── test_model.py         #  7 tests — architecture, weights, batching, loop
│   └── integration/
│       └── __init__.py           # Day 9+: API → model → prediction → GenAI
│
├── docs/                         # ══ DOCUMENTATION ══
│   ├── README.md                 # Documentation index
│   ├── RESULTS.md                # Every metric, with its caveats
│   ├── architecture.md           # Architecture diagram + tech stack
│   ├── Day1.md                   # One file per implementation day
│   ├── Day2.md
│   ├── Day3.md
│   └── Day4.md
│
├── notebooks/                    # Jupyter scratch space (.gitkeep)
├── dashboard/                    # Streamlit app (Day 10) (.gitkeep)
├── docker/                       # Dockerfiles + compose (Day 11) (.gitkeep)
├── logs/                         # GITIGNORED — app_YYYY-MM-DD.log, rotated
└── venv/                         # GITIGNORED — Python 3.12 virtualenv
```

## Why the important files matter

| File | Why it exists / what breaks without it |
|---|---|
| `config/settings.py` | The only place values live. Deleting a field here breaks every layer, which is the point — there is exactly one place to look. |
| `src/utils/exceptions.py` | Lets the API map internal failures to HTTP status codes by *layer* without string-matching error messages. |
| `src/data/preprocessing.py` | Encodes the three anti-leakage invariants. If someone reorders split/scale/window, model metrics become fiction. |
| `src/models/trainer.py` | Contains the platform workaround that makes training possible at all. Its docstring explains why `fit()` is not used. |
| `tests/conftest.py` | Imports TensorFlow before anything else. Without it the test suite hangs forever, silently, at 0% CPU. |
| `data/processed/feature_columns.txt` | The contract between training and inference. Inference must build features in exactly this order. |
| `data/sample/` | The only committed data. It has zero failures on purpose, which is why model tests synthesize their own labels. |

---

# Dataset Documentation

## Data sources

| Field | Value |
|---|---|
| **Name** | Synthetic Predictive Maintenance Dataset |
| **Modeled after** | Microsoft Azure Predictive Maintenance Dataset |
| **Generated by** | `scripts/generate_data.py` (local, no download) |
| **Seed** | 42 — fully reproducible |
| **Licensing** | Self-generated; no restrictions |
| **Total rows** | 883,231 across 5 tables |
| **Date range** | 2024-01-01 → 2024-12-30 (364 days) |

Synthetic data was chosen over a Kaggle download deliberately: no external dependency, no
licensing question, exact reproducibility, and full control over the degradation physics
so the labels are guaranteed correct.

## Data format

Five CSV tables, joined on `(datetime, machine_id)`:

| Table | Rows | Columns |
|---|---|---|
| `telemetry.csv` | 876,000 | `datetime, machine_id, voltage, rotation, pressure, vibration` |
| `machines.csv` | 100 | `machine_id, model, age` |
| `errors.csv` | 5,386 | `datetime, machine_id, error_id` (error1–error5) |
| `maintenance.csv` | 1,698 | `datetime, machine_id, comp` (comp1–comp4) |
| `failures.csv` | 47 | `datetime, machine_id, failure` (comp1–comp4) |

### Sensor dictionary

| Sensor | Unit | Mean | Std | Range | Degradation signature before failure |
|---|---|---|---|---|---|
| `voltage` | Volts | 170 | 15 | [100, 250] | becomes erratic, ±25 V |
| `rotation` | RPM | 450 | 50 | [100, 800] | drops ~80 RPM as bearings wear |
| `pressure` | PSI | 100 | 12 | [40, 180] | drops ~20 PSI from leaks |
| `vibration` | mm/s | 40 | 8 | [10, 100] | rises ~20 mm/s as parts loosen |

### Realism features built into the generator

- Daily periodicity in all sensors (factory temperature cycles).
- Older machines have noisier sensors.
- Degradation ramps gradually over the 48 h before failure — never a step change.
- Error frequency rises in the week before a failure.
- Per-machine offsets modeling manufacturing variance.

## EDA findings (from `scripts/eda_analysis.py`)

| Dimension | Finding |
|---|---|
| **Class imbalance** | 47 failure events in 876,000 readings = 0.005% raw event rate; after 24 h labeling, 1,175 positive rows = 0.13%, ratio ≈ 1:745. **This is the defining property of the problem.** |
| **Distributions** | All 4 sensors approximately symmetric, skewness < 0.5. |
| **Correlations** | Max pairwise \|r\| = 0.075 — all four sensors carry independent information, so none can be dropped. |
| **Age effect** | 16–20-year-old machines fail ~3.5× more than 0–5-year-old ones (0.7 vs 0.2 failures/year). Age is a real feature, not noise. |
| **Failure modes** | Roughly uniform: comp1 23%, comp2 28%, comp3 21%, comp4 28%. |
| **Seasonality** | None detectable — failures spread uniformly across months. |
| **Missing values** | Zero in every table. |
| **Duplicates** | Zero in every table. |

## Preprocessing and cleaning

Executed by `DataPreprocessor.run_pipeline()`, in this exact order:

| # | Step | What it does |
|---|---|---|
| 1 | **Merge** | Left-join telemetry with machines, errors, maintenance, failures on `(datetime, machine_id)`. |
| 2 | **One-hot encode** | `model` categorical → dummy columns. |
| 3 | **Error aggregation** | Count errors per machine-hour; rolling 24 h error sum. |
| 4 | **Maintenance features** | Hours since last replacement, per component. |
| 5 | **Rolling statistics** | Per machine: mean and std over 3 h / 12 h / 24 h windows for each of the 4 sensors. |
| 6 | **Lag features** | Per machine: sensor values at t−1 h, t−6 h, t−24 h. |
| 7 | **Change features** | `current − lag` for each lag, i.e. rate of change. |
| 8 | **Fill** | Forward-fill then back-fill the NaNs that rolling/lag operations create at window edges. |
| 9 | **Label creation** | `label = 1` for every hour within 24 h *before* a failure event on that machine. |
| 10 | **Temporal split** | Train = Jan–Oct, Test = Oct–Dec. Never random. |
| 11 | **Normalization** | `StandardScaler.fit(train)`, then `.transform()` on train and test. |
| 12 | **Sequence windowing** | 24-step sliding windows, per machine. |

**Cleaning philosophy:** the generator produces no nulls or duplicates, so cleaning is
limited to (a) handling the NaNs that feature engineering itself introduces at window
boundaries, and (b) validating physical sensor ranges via `DataValidator` so that a real
data source substituted later would be caught rather than silently modeled.

## Feature engineering

**63 features total** = 4 raw sensors + 1 age + one-hot model dummies + 48 engineered +
error/maintenance aggregates. The exact ordered list lives in
`data/processed/feature_columns.txt` and is the contract inference must honor.

| Category | Count | Example |
|---|---|---|
| Raw sensors | 4 | `voltage`, `rotation`, `pressure`, `vibration` |
| Machine metadata | 1 + dummies | `age`, `model_model1`… |
| Rolling mean | 12 | `voltage_rolling_mean_3h`, `vibration_rolling_mean_24h` |
| Rolling std | 12 | `pressure_rolling_std_12h` |
| Lag | 12 | `rotation_lag_6h` |
| Change | 12 | `vibration_change_24h` |
| Error aggregates | ~5 | `error_count_24h` |
| Maintenance | ~4 | `hours_since_comp1_replacement` |

**Why these features:** an LSTM can in principle learn rolling statistics itself, but with
957 positive training examples it will not. Hand-engineering the trend/volatility features
injects the domain knowledge (degradation is *gradual*, so slope and variance matter more
than instantaneous value) that the label scarcity would otherwise prevent it from
discovering.

## Train / validation / test split

| Split | Range | Sequences | Positives | Rate |
|---|---|---|---|---|
| **Train** | Jan – Oct 2024 | 698,400 | 957 | 0.137% (1:730) |
| **Test** | Oct – Dec 2024 | 172,800 | 200 | 0.116% (1:864) |

**There is currently no separate validation split.** Day 4 monitors the *test* set during
training, which is a knowingly-taken shortcut: early stopping and checkpoint selection see
the test set, so reported test metrics are mildly optimistic. This is logged as technical
debt **TD-1** and is scheduled for correction on Day 5 by carving a chronological
validation slice out of the tail of the training period.

**Why temporal, never random:** with rolling and lag features, a random split puts hour
`t` in train and hour `t+1` in test. The model then "predicts" a failure it has already
seen from the other side. Reported AUC becomes meaningless. This is the project's
single most important correctness invariant.

## Normalization

- `StandardScaler` (zero mean, unit variance), **fitted on training rows only**, then
  applied to both train and test.
- Chosen over `MinMaxScaler` because it handles outliers better and does not bound the
  range — LSTM gates behave better with roughly standard-normal inputs.
- The fitted scaler is persisted to `data/processed/scaler.joblib` because **inference
  must use the training-time statistics**, not statistics recomputed from live data.

## Augmentation

**None.** Deliberate. The usual imbalance remedies were considered and rejected for this
stage:

| Technique | Verdict |
|---|---|
| SMOTE / synthetic minority oversampling | Rejected — interpolating between time-series *sequences* creates physically impossible sensor trajectories. |
| Random oversampling of positive windows | Rejected for v1; heavy overlap between adjacent windows already duplicates most of the positive signal. |
| Class weighting | **Adopted** — mathematically equivalent in effect, no fabricated data. |
| Noise injection | Deferred to Day 5 as a possible regularizer. |

## Sequence generation

- Window length **24 timesteps = 24 hours**; stride 1.
- The label for a window is the label of its **final** timestep.
- **Windows never span two `machine_id`s.** Sequences are built per machine and then
  concatenated. Violating this would splice machine 7's Tuesday onto machine 8's Monday.
- Output shape `(N, 24, 63)`, `float32`.

## Memory optimization

The training tensor is 698,400 × 24 × 63 × 4 bytes = **4.2 GB**; the test tensor is
**1.0 GB**. Loading both would exceed the working memory budget on a 16 GB laptop
alongside TensorFlow.

| Technique | Implementation |
|---|---|
| **Memory mapping** | `np.load(path, mmap_mode='r')` — arrays stay on disk; the OS pages in only the slices actually touched. |
| **Lazy batch materialization** | `iter_batches()` calls `np.asarray(X[idx], dtype=np.float32)` per batch — the only point where data becomes resident. |
| **Monotonic reads within a batch** | Shuffled indices are `np.sort`ed inside each batch, so the page cache sees ascending offsets instead of random seeks. |
| **float32 throughout** | Never float64; halves both disk and RAM. |
| **No full-array copies** | Class weights are computed from `y` only (5.6 MB), never from `X`. |

## Caching

- `get_settings()` is `lru_cache`d — settings are parsed from `.env` once per process.
- Preprocessing output is cached *as files*: `run_preprocessing.py` is run once, and
  training reads its artifacts. Training never re-derives features.
- OS page cache does the heavy lifting for repeated epochs over the memmaps.
- TensorFlow's `tf.function` traces the train step **once** per training run
  (`input_signature` with a `None` batch dimension prevents retracing on the final short
  batch of each epoch).

## Streaming

There is no streaming ingestion yet — the pipeline is batch/offline. The training loop is
nonetheless written as a *generator* over batches (`iter_batches`), so replacing the
memmap source with a live feed later is a change to one function, not to the loop.

## Batching

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 256 (default; 512 acceptable) | Balances gradient stability against per-step latency on CPU. Larger batches make the ~1:730 imbalance worse per batch — at 256, an average batch contains 0.35 positives, so class weighting is doing the real work. |
| Batches per epoch (train) | 2,729 | `ceil(698400 / 256)` |
| Shuffling | Epoch-level index shuffle, seeded by epoch number | Reproducible run-to-run, different order each epoch. |
| Final batch | Kept, not dropped | The `None` batch dimension in the `tf.function` signature handles it without retracing. |

## Prefetching

**Deliberately absent.** This is the most surprising design decision in the project and it
is forced by the platform: every asynchronous prefetch path in Keras
(`tf.data.Dataset.from_generator(...).prefetch()`, `keras.utils.Sequence`, `PyDataset`)
spawns a background worker thread that **deadlocks against the memmapped reads** on this
machine. Training hangs at 0% CPU with no traceback and no timeout.

The training loop is therefore fully synchronous: read batch → train step → read next
batch. The cost is real (I/O is not overlapped with compute) but a hanging trainer has
infinite cost. See Risk R-3 and `docs/Day4.md`.

## Versioning

| Artifact | Versioning strategy |
|---|---|
| Raw data | Reproducible from `generate_data.py` + `seed=42`; not stored in git. |
| Processed tensors | Regenerable from raw via `run_preprocessing.py`; not stored in git. |
| Feature list | `feature_columns.txt` is committed conceptually as the schema contract. |
| Scaler | Saved alongside processed data; must be shipped with the model. |
| Model | `models/lstm_predictive_maintenance.keras`, gitignored (too large). |
| **Metrics & history** | `models/metrics.json` and `models/training_history.json` **are committed** — they are small, and they are the auditable evidence of what a given commit's model actually achieved. |
| Code | Git; one commit per completed day, conventional-commit style. |

---

# Model Architecture

## The model: `PredictiveMaintenanceModel` (`src/models/lstm_model.py`)

A binary classifier over a 24-hour multivariate sensor window.

```
Input (batch, 24, 63)
   │
   ├─ LSTM(128, return_sequences=True)      "lstm_1"     97,792 params
   ├─ Dropout(0.3)                          "dropout_1"
   ├─ LSTM(64, return_sequences=False)      "lstm_2"     49,408 params
   ├─ Dropout(0.3)                          "dropout_2"
   ├─ Dense(32, activation="relu")          "dense_1"     2,080 params
   └─ Dense(1,  activation="sigmoid")       "output"         33 params
   │
Output (batch, 1) → P(failure within 24h)
```

### Why this architecture

| Choice | Reason |
|---|---|
| **LSTM over Transformer** | 957 positive training examples. A transformer's attention parameters would overfit instantly; LSTM's recurrent inductive bias ("recent history matters, in order") is exactly the right prior for gradual degradation. |
| **LSTM over 1D CNN** | CNNs capture local patterns well but the signal here is a slow 48-hour trend — a long-range dependency. |
| **LSTM over XGBoost/Random Forest** | Trees cannot model temporal ordering natively; they would need the sequence flattened, discarding the structure the whole design is built around. |
| **LSTM over ARIMA** | Univariate only; cannot fuse 4 correlated sensors plus metadata. |
| **Two stacked LSTM layers** | The first (returning sequences) learns per-timestep representations; the second compresses the sequence to a fixed vector. One layer underfits the interaction between sensors; three overfits at this data size. |
| **128 → 64 taper** | Standard funnel; halving keeps parameter count near 150K, roughly 150 parameters per positive example — already aggressive, so no wider. |
| **Dropout 0.3 (×2)** | The strongest available regularizer given the positive-class scarcity. Applied after each LSTM, not inside the recurrent connections (recurrent dropout is dramatically slower on CPU). |
| **Dense(32, relu) head** | Lets the classifier combine LSTM features nonlinearly before the decision. |
| **Sigmoid output** | Binary probability, and — critically — a *calibrated-ish score* the API can threshold at different operating points without retraining. |

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Sequence length | 24 | One full daily cycle |
| Features | 63 | Fixed by preprocessing |
| Optimizer | **Adam** | Adaptive per-parameter LR; robust without tuning |
| Learning rate | **0.001** | Adam default; reduced on plateau |
| Loss | **Binary crossentropy**, class-weighted | Standard for binary classification |
| Class weights | computed from label distribution: `w_c = n_total / (2 · n_c)` → **{0: 0.50, 1: 364.89}** | Makes the 957 positives count as much in aggregate as the 697,443 negatives |
| Batch size | **256** | See Batching above |
| Epochs | **30** (default; 50 supported) | With early stopping, rarely reached |
| Dropout | 0.3 | Regularization |
| Weight init | Keras defaults (Glorot/orthogonal) | No reason to deviate |

## Loss function detail

`tf.keras.losses.BinaryCrossentropy` with **per-sample weights**, not Keras's
`class_weight=` argument (which only exists inside `fit()`). The manual loop builds the
weight vector explicitly:

```python
sample_weights = np.where(y_batch == 1, w_pos, w_neg).astype(np.float32)
loss = loss_fn(y_batch, preds, sample_weight=sample_weights)
```

This is mathematically identical to what `fit(class_weight=...)` does internally, and it
keeps the imbalance handling visible at the call site instead of hidden in a framework.

## Evaluation metrics

| Metric | Why |
|---|---|
| **ROC-AUC** (primary) | Threshold-independent; the honest headline number under imbalance. |
| **Precision** | Of the machines we flagged, how many really fail? Low precision = technicians stop trusting the system. |
| **Recall** | Of the machines that failed, how many did we catch? Low recall = the system's whole purpose is unmet. |
| **F1** | Single number balancing the two, for model selection. |
| **Confusion matrix** | The raw counts. With 200 test positives, the difference between 40 and 80 true positives matters and is invisible in a rounded rate. |
| **Accuracy** | **Explicitly excluded.** A model that always predicts "no failure" scores 99.88%. Reporting it would be dishonest. |

## Callbacks — implemented manually

Keras callbacks only run inside `fit()`, which this project cannot use. Each is
reimplemented inline in `ModelTrainer.train()` with the same semantics:

| Behavior | Implementation | Configuration |
|---|---|---|
| **EarlyStopping** | Tracks the monitored metric; counts epochs without improvement; breaks the epoch loop; restores best weights at the end. | monitor `val_auc` (max) when validation data is given, else training `loss` (min); `patience=5`; `restore_best_weights` always on |
| **ModelCheckpoint** | `self.model.save(path)` called only on an improvement. | `save_best_only` semantics; path from `settings.model_artifacts_path` |
| **ReduceLROnPlateau** | Tracks `val_loss` (or training loss); after `lr_patience` stagnant epochs, `optimizer.learning_rate.assign(max(lr*0.5, min_lr))`. | `factor=0.5`, `lr_patience=3`, `min_lr=1e-6` |
| **Progress logging** | Replaces the Keras progress bar, which does not exist outside `fit()`. | Every 100 batches: running loss + running AUC |

## Checkpoint strategy

- Saved to `models/lstm_predictive_maintenance.keras` (Keras v3 single-file format).
- Written **only when the monitored metric improves**, so the file on disk is always the
  best model seen, not the last.
- The in-memory best weights are additionally kept as a NumPy copy and restored after the
  loop, so the returned trainer and the saved file agree.
- Final evaluation deliberately **reloads from disk** (`PredictiveMaintenanceModel.load()`)
  rather than evaluating the in-memory model — this proves the artifact round-trips, which
  is what Day 6 inference will actually depend on.

## Early stopping

`patience=5` on `val_auc`. AUC rather than `val_loss` because with class weighting the
weighted loss can drift while ranking quality (what actually matters) keeps improving.

## Scheduler

`ReduceLROnPlateau`-equivalent only — no cosine/step schedule. With early stopping
typically firing inside 30 epochs, a fixed schedule would never complete a cycle. Adam
already adapts per-parameter rates; the plateau halving handles the rest.

## Mixed precision

**Not used.** Mixed precision (`float16`) pays off on GPU tensor cores. This project
trains on Apple Silicon CPU where `float16` is emulated and would be *slower*, and the
numerical risk (gradient underflow with a 365× class weight) is real. Revisit only if
CUDA hardware enters the picture.

## Memory optimization (model side)

- ~149K parameters ≈ 0.6 MB of weights — negligible; **activations dominate**.
- Batch 256 × 24 timesteps × 128 units × 4 bytes ≈ 3.1 MB per LSTM-1 activation tensor,
  and gradients roughly double it. Comfortable.
- Best weights are stored as a NumPy copy (~0.6 MB), not a second Keras model.
- Evaluation streams the 1 GB test set in 512-sample batches and concatenates only the
  scalar predictions (172,800 floats = 0.7 MB), never the inputs.

---

# Training Pipeline

## Training workflow

Entry point: `python scripts/train_model.py [--epochs N] [--batch-size N] [--learning-rate F]`

```
1. get_settings()                              # paths, model name
2. load_data(settings.processed_data_path)     # np.load(..., mmap_mode='r') × 4
3. PredictiveMaintenanceModel(seq_len, n_feat)  # infer shapes from X_train
4. ModelTrainer(model).compile(lr)             # Adam + BinaryCrossentropy
5. trainer.train(X_train, y_train,
                 X_val=X_test, y_val=y_test,   # ← TD-1: monitors the test set today
                 epochs, batch_size,
                 checkpoint_path=models/<MODEL_NAME>.keras)
6. PredictiveMaintenanceModel.load(path)       # reload the BEST checkpoint from disk
7. ModelEvaluator(best).evaluate(X_test, y_test, threshold=0.5)
8. write models/metrics.json
9. write models/training_history.json
```

## Batch generation

`iter_batches(X, y, batch_size, shuffle, seed)` — a plain Python generator:

```python
indices = np.arange(len(X))
if shuffle:
    rng = np.random.default_rng(seed)   # seed = epoch number → reproducible, varies per epoch
    rng.shuffle(indices)
for start in range(0, len(X), batch_size):
    batch_idx = indices[start:start + batch_size]
    if shuffle:
        batch_idx = np.sort(batch_idx)   # keep memmap reads monotonic within the batch
    yield np.asarray(X[batch_idx], dtype=np.float32), np.asarray(y[batch_idx], dtype=np.float32).reshape(-1)
```

Guarantees, covered by `test_iter_batches_covers_all_samples_without_overlap`: every
sample appears **exactly once** per epoch, no batch exceeds `batch_size`, and the positive
count is preserved.

## Data loading

`np.load(path, mmap_mode='r')` for all four arrays. Nothing is resident until a batch is
sliced. `y` arrays (2.8 MB / 0.7 MB) are small enough that full residency is harmless and
class-weight computation touches them directly.

## tf.data pipeline or generator

**Generator — and this is a documented reversal.** The Day 4 implementation went through
three iterations:

| Attempt | Result |
|---|---|
| `NumpyDataGenerator(keras.utils.Sequence)` + `model.fit()` | Deadlock — `fit()` hung after the first log line, 0% CPU. |
| `tf.data.Dataset.from_generator(...).prefetch()` + `model.fit()` | Deadlock — same symptom. The prefetch thread is the common factor. |
| **Hand-written `GradientTape` loop over `iter_batches()`** | **Works.** No background threads anywhere in the data path. |

The train step itself is still graph-compiled for speed:

```python
@tf.function(input_signature=[
    tf.TensorSpec((None, seq_len, n_features), tf.float32),
    tf.TensorSpec((None,), tf.float32),
    tf.TensorSpec((None,), tf.float32),
])
def train_step(x_batch, y_batch, sample_weights):
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        loss = loss_fn(y_batch, preds, sample_weight=sample_weights)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds
```

So the *compute* keeps XLA-style graph optimization; only the *data movement* is
synchronous Python. The explicit `input_signature` with a `None` batch dimension is what
prevents a retrace on each epoch's final short batch.

## Memory mapping

Covered above under Dataset Documentation → Memory optimization. The essential rule:
**never call `np.array(X)` on a full memmapped tensor.** Slice, then convert.

## Checkpoint saving

On every improvement of the monitored metric:

```
models/lstm_predictive_maintenance.keras   ← best model so far, complete (arch + weights + optimizer state)
```

Keras v3 `.keras` format, single file, self-describing — `load_model()` recovers input
shape, which is how `PredictiveMaintenanceModel.load()` reconstructs the wrapper without
being told the dimensions.

## Resume strategy

Not yet implemented — logged as technical debt **TD-2**. Today an interrupted run loses
its epoch history, though the best-so-far `.keras` checkpoint survives on disk and can be
loaded manually. The intended fix (Day 5) is to persist `{epoch, best_score, history,
optimizer_state}` next to the checkpoint and add a `--resume` flag.

Current mitigation: `train_model.py` catches `KeyboardInterrupt` and exits cleanly rather
than corrupting a half-written checkpoint.

## Logging

| Channel | Content |
|---|---|
| Console (loguru, colored) | Everything at INFO |
| `logs/app_YYYY-MM-DD.log` | Same, rotated at 5 MB, retained 3 days, zip-compressed |
| Every 100 batches | `Epoch N \| batch B \| loss L \| auc A` — the manual replacement for Keras's progress bar |
| Every epoch | Train loss/AUC/precision/recall, and the same four for validation |
| On checkpoint | `Epoch N: val_auc improved to X — saved <path>` |
| On LR reduction | `Reducing learning rate: 1.00e-03 -> 5.00e-04` |
| On early stop | `Early stopping at epoch N (no improvement for 5 epochs)` |

`enqueue=True` is **off** on the file sink — its background writer thread is part of the
same thread-interaction family that causes the TF deadlocks.

## TensorBoard

Not wired up. `tf.keras.callbacks.TensorBoard` is a callback, and callbacks require
`fit()`. The equivalent data is captured in `models/training_history.json` (per-epoch
loss, AUC, precision, recall, plus their validation counterparts), which Day 5 will plot
with matplotlib. If TensorBoard becomes necessary, `tf.summary.create_file_writer()` can
be called directly from inside the manual epoch loop — no callback needed.

## Model saving

| Artifact | Path | Committed? |
|---|---|---|
| Best model | `models/lstm_predictive_maintenance.keras` | No (gitignored, too large) |
| Evaluation metrics | `models/metrics.json` | **Yes** |
| Per-epoch history | `models/training_history.json` | **Yes** |
| Fitted scaler | `data/processed/scaler.joblib` | No (regenerable) |
| Feature order | `data/processed/feature_columns.txt` | No (regenerable) |

---

# Evaluation Plan

## Metrics

Computed by `ModelEvaluator.evaluate(X_test, y_test, threshold=0.5)` and written to
`models/metrics.json`.

| Metric | Definition | Target (Day 5 goal) |
|---|---|---|
| **ROC-AUC** | Ranking quality across all thresholds | ≥ 0.85 |
| **Precision** | TP / (TP + FP) — trust in an alert | ≥ 0.30 |
| **Recall** | TP / (TP + FN) — coverage of real failures | ≥ 0.60 |
| **F1** | Harmonic mean of the two | ≥ 0.40 |
| **Confusion matrix** | Raw [[TN, FP], [FN, TP]] counts | reported, not thresholded |
| **Accuracy** | *not reported* | — |

The precision target is intentionally modest. With 200 positives among 172,800 test
sequences, precision 0.30 already means a ~260× lift over the 0.116% base rate. Chasing
precision 0.9 here would require sacrificing nearly all recall, which inverts the
business value: a missed failure costs a production line, a false alarm costs an
inspection.

## Threshold policy

`0.5` is the reported default, but it is **not** the right operating point for a 1:864
problem. Day 5 will sweep thresholds and report the precision-recall curve, then choose an
operating point by cost (roughly: cost of a missed failure ÷ cost of an unnecessary
inspection). The sigmoid output makes this a deployment-time decision, not a retraining
one.

## Confusion matrix handling

`sklearn.metrics.confusion_matrix` output is stored as a nested list in `metrics.json`.
`ModelEvaluator` guards the single-class edge case: if `y_true` contains only one class
(which happens on `data/sample/`, where there are zero failures), `roc_auc_score` raises
`ValueError` and the evaluator logs a warning and returns AUC 0.0 rather than crashing.

## Inference speed

| Measurement | Method | Target |
|---|---|---|
| Single-sequence latency | `model(x[None, ...])` timed over 100 calls, median | < 100 ms (NFR-3) |
| Batch throughput | sequences/second at batch 512 | measured during Day 5 |
| Cold start | time to `load_model()` + first inference | reported, informs API startup |

Inference runs through `ModelEvaluator._predict_in_batches()`, which calls
`self.model(batch, training=False)` directly rather than `model.predict()` — same
deadlock avoidance as training.

## Latency

For the API layer (Day 9): p50/p95/p99 for `POST /predict`, measured with the model
pre-loaded at app startup so per-request latency excludes model loading. Target p95
< 500 ms (NFR-4), of which inference should be < 100 ms and feature construction the rest.

## Memory usage

| Phase | Measurement | Expectation |
|---|---|---|
| Training RSS | peak resident set during a full epoch | ≪ 16 GB thanks to memmapping |
| Evaluation RSS | peak during the 172,800-sequence pass | bounded by 512-sequence batches |
| API RSS | steady state with model loaded | ~1 GB (TF runtime dominates) |

Measured with `psutil` / `/usr/bin/time -l` on macOS.

## Benchmark methodology

1. Quiesce the machine (no other heavy processes — the throughput probe on Day 4 was
   explicitly serialized behind the test suite for this reason).
2. Warm up: discard the first batch, which includes `tf.function` tracing (~50× the
   steady-state cost).
3. Report the **median** of ≥ 30 steady-state measurements, not the mean — a single page
   fault or scheduler hiccup skews the mean badly.
4. Record alongside every benchmark: batch size, sequence count, CPU model, TF version,
   and whether the array was memmapped or resident.
5. Re-run benchmarks whenever the model architecture or the batching strategy changes.

---

# Deployment Plan

## Backend

FastAPI application (`src/api/main.py`) served by Uvicorn.

- The model, scaler, and feature list load **once** in the app's lifespan startup hook and
  live in app state. No per-request loading — that would put a ~2 s penalty on every call.
- Request/response bodies validated by Pydantic schemas in `src/api/schemas.py`.
- The custom exception hierarchy maps to HTTP status codes by layer: `DataValidationError`
  → 422, `ModelNotFoundError` → 503, `LLMConnectionError` → 502, `ResourceNotFoundError`
  → 404, anything else → 500 with a generic body (never a stack trace).

## Frontend

Streamlit (`dashboard/app.py`), deployed as a separate container. It is a **pure HTTP
client of the API** — it never imports `src.models` or loads a `.keras` file. That
separation is what allows the dashboard and the API to scale, fail, and deploy
independently.

Pages: Overview (fleet status, alert feed) · Machine Detail (sensor charts, prediction
history, AI report) · Predictions (probability timeline, risk heatmap) · Reports.

## REST APIs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model-loaded status + version |
| `POST` | `/predict` | Sensor window → failure probability + risk level |
| `POST` | `/report` | Prediction → LLM-generated maintenance report |
| `GET` | `/machines` | List machines with current risk |
| `GET` | `/machines/{id}/history` | Prediction history for one machine |
| `GET` | `/docs` | Auto-generated Swagger UI |

## Authentication

Not implemented in v1 and **explicitly out of scope** — the demo runs on a trusted
network. The intended v2 design, documented so nobody has to re-derive it: an API-key
header checked by a FastAPI dependency, keys stored hashed, per-key rate limits. No
session cookies, no OAuth — this is a machine-to-machine API.

## Security

| Concern | Measure |
|---|---|
| Secrets | `.env` only, gitignored; `.env.example` documents the required keys with no values. Never logged. |
| Input validation | Pydantic schemas reject malformed payloads before they reach the model; sensor values are range-checked. |
| Error disclosure | Internal exceptions never serialize to the client; generic messages plus a correlation id in logs. |
| Dependency risk | Pinned version ranges; `pip-audit` in CI (Day 11). |
| Container | Non-root user, minimal base image, no build toolchain in the runtime stage. |
| LLM prompt injection | Machine data is inserted into prompts as *data*, not instructions; report output is treated as text for display, never executed. |
| CORS | Restricted to the dashboard origin, not `*`. |

## Docker

Multi-stage builds under `docker/`:

```
docker/
├── Dockerfile.api          # stage 1: build wheels · stage 2: slim runtime + uvicorn
├── Dockerfile.dashboard    # streamlit runtime
└── docker-compose.yml      # api + dashboard + volume mounts for models/ and data/processed/
```

Model artifacts are mounted as a **volume**, not baked into the image — a 150K-parameter
model is small, but rebuilding an image to ship a retrained model is the wrong workflow.

## CI/CD

GitHub Actions, triggered on push and PR to `main`:

```
checkout → setup-python 3.12 → pip install -r requirements-dev.txt (cached)
        → make lint            (flake8)
        → make format-check    (black --check, isort --check)
        → make typecheck       (mypy)
        → make test            (pytest, with data/sample/ fixtures only)
        → build Docker images
        → push to registry     (main branch only)
```

CI runs against `data/sample/` because `data/raw/` is gitignored and 883K rows would blow
the runner's time budget. This is exactly why the sample set is committed.

## Cloud deployment

Target: any container host with a free tier (Render / Railway / Fly.io). Two services from
one compose file. No Kubernetes — the operational complexity would exceed the application
complexity by an order of magnitude.

## Monitoring

| Signal | Mechanism |
|---|---|
| Liveness | `GET /health` polled by the platform |
| Application logs | loguru → stdout in containers (the platform aggregates), file sink locally |
| Prediction distribution | Log every prediction's probability; a sudden shift in the distribution is the cheapest available drift detector |
| Latency | Per-request duration logged; p95 tracked |
| LLM cost/failures | Log token counts and provider errors separately (`LLMConnectionError`) |

## Logging (production)

Same loguru configuration, but console-only in containers (stdout is the platform's log
pipeline). File rotation stays for local runs. Structured fields — machine id, request id,
model version — so logs are greppable rather than prose.

## Scaling strategy

| Bottleneck | Response |
|---|---|
| Inference throughput | Horizontal: the API is stateless once the model is loaded, so N replicas behind a load balancer. |
| Model load time / memory per replica | Accept ~1 GB RSS per replica; if that becomes expensive, move to TF Serving with a shared model. |
| LLM latency (seconds, not milliseconds) | Make report generation **asynchronous** — `POST /report` returns a job id; the dashboard polls. Never block a prediction on an LLM call. |
| Batch scoring the whole fleet | Offline job writing results to storage, not N synchronous API calls. |
| Retraining | Offline, on a schedule; deploy by swapping the mounted model volume. |

---

# Coding Standards

## Naming conventions

| Element | Convention | Example |
|---|---|---|
| Files/modules | `snake_case` | `lstm_model.py` |
| Classes | `PascalCase` | `DataPreprocessor` |
| Functions/methods | `snake_case` | `create_labels()` |
| Private helpers | leading underscore | `_compute_class_weights()` |
| Constants | `UPPER_SNAKE_CASE` | `LOG_EVERY_N_BATCHES` |
| Settings fields | `UPPER_SNAKE_CASE` | `MODEL_NAME` |
| Test files | `test_<module>.py` | `test_preprocessing.py` |
| Test functions | `test_<behavior>_<expectation>` | `test_iter_batches_covers_all_samples_without_overlap` |

## Formatting

- **Black**, line length 88. Non-negotiable, no per-file exceptions.
- **isort** with the Black-compatible profile; absolute imports only
  (`from src.data.ingestion import DataIngestion`), never relative.
- `make format` writes; `make quality` checks without writing.

## Documentation

Every module opens with a docstring in the project's fixed format:

```python
"""
src/models/trainer.py — Model Training Pipeline
===============================================

WHY THIS FILE EXISTS:
    The architectural/business reason this module is a separate thing.

HOW IT WORKS:
    The technical mechanism, including any non-obvious constraint.
"""
```

Functions and classes use Google-style docstrings with `Args:` / `Returns:` / `Raises:`.

**Comments explain *why*, never *what*.** `# increment counter` is noise; `# sorted so
memmap reads stay monotonic` is the reason the line exists. Every workaround must name the
failure it prevents — a future reader will otherwise "simplify" it back into a bug. This
rule is why `tests/conftest.py` is a 20-line docstring around a single import.

## Logging

- `logger = get_logger(__name__)` at module top. **Never `print()`.**
- INFO for lifecycle milestones, DEBUG for detail, WARNING for degraded-but-continuing,
  ERROR for failures with context.
- Log the *numbers* — shapes, counts, durations, metric values. A log line that says
  "training complete" without the metrics is not worth writing.

## Error handling

- Raise from the custom hierarchy in `src/utils/exceptions.py`, never bare `Exception`.
- Catch narrowly. Where a broad catch wraps and re-raises as a layer-specific error
  (as in `ModelTrainer.train()`), log the original before wrapping so context survives.
- Never swallow an exception silently. If a fallback is legitimate (single-class AUC),
  log a WARNING explaining the fallback.
- Validate at layer boundaries; trust within a layer.

## Testing strategy

Summarized here, detailed in the next section: every new module ships with its tests in
the same commit. A day is not "done" until `make test` is green.

## Commit message format

Conventional Commits:

```
<type>(<scope>): <imperative summary>

<optional body: why, not what>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`.
Scopes: `data`, `models`, `prediction`, `genai`, `api`, `dashboard`, `docker`, `ci`.

Examples from this repo:
```
feat(data): add feature engineering, preprocessing, and LSTM sequencing
feat(models): add LSTM architecture, manual training loop, and evaluator
docs: update README with actual GitHub repo URL and author info
```

One commit per completed milestone/day, with implementation, tests, and documentation
together — so any commit can be checked out and it works, tested and documented.

## Branch strategy

Currently **trunk-based on `main`**, appropriate for a single-developer project with a
green test suite gating each commit. If a second developer joins: `feat/<scope>-<slug>`
branches, PR into `main`, CI must pass, squash merge. `main` is always deployable.

---

# Testing Strategy

## Current state: 75 tests, all passing (~4 s warm, ~90 s cold)

| Suite | Tests | Covers |
|---|---|---|
| `tests/unit/test_smoke.py` | 19 | Imports, settings loading, derived paths, logger, exception hierarchy |
| `tests/unit/test_data_pipeline.py` | 22 | `DataIngestion` format detection and errors; `DataValidator` schema/null/duplicate/range checks |
| `tests/unit/test_preprocessing.py` | 27 | Merge correctness, rolling/lag features, NaN handling, label timing, temporal split ordering, scaler statistics, sequence shape and label alignment, full pipeline |
| `tests/unit/test_model.py` | 7 | Architecture shapes and layer types, class-weight math, optimizer configuration, batch-generator coverage, that the training loop actually updates weights, evaluator metric contract |
| `tests/conftest.py` | — | Session bootstrap: imports TensorFlow first (see Risk R-2) |

## Unit testing

Scope: one class or function, no disk beyond `data/sample/`, no network, no trained model.

Notable design points:

- **Model tests synthesize their own labels.** `data/sample/` has zero failures, so a
  fixture builds `y = [0,0,0,0,0,0,0,0,1,1]` explicitly. Testing class-weight math against
  a dataset with no positive class would test nothing.
- **Class-weight math is asserted numerically**, not just for presence:
  8 negatives / 2 positives / 10 total → `w0 = (1/8)(10/2) = 0.625`, `w1 = (1/2)(10/2) = 2.5`.
- **The training loop is tested for *effect*, not output shape** — capture the weights,
  run two epochs, assert at least one weight tensor changed. This catches a silently
  broken gradient path, which a shape assertion would not.
- **`iter_batches` is tested for the invariant that matters**: every sample exactly once,
  positives conserved, no batch over the limit.

## Integration testing

`tests/integration/` exists but is empty — planned for Day 9+:

- preprocessing → training → evaluation on the sample dataset end to end
- `Predictor` loading real artifacts and scoring a raw sensor window
- API routes via `httpx.AsyncClient` against the real app with a stub model
- GenAI chains with a mocked LLM, asserting prompt construction rather than LLM output

## End-to-end testing

Day 12: a scripted run of the full user journey — generate data → preprocess → train →
start API → dashboard renders a prediction and an AI report — executed from a clean
checkout, which doubles as validation of the installation instructions above.

## Performance testing

- **Training throughput**: median steady-state seconds/batch and projected epoch time,
  measured after `tf.function` warm-up on an otherwise idle machine.
- **Inference latency**: single-sequence median over 100 calls.
- **API load**: Day 9, with `locust` or `hey` — sustained RPS at p95 < 500 ms.

## Stress testing

- Full-dataset training run to confirm the memmap path does not exhaust memory over
  thousands of batches (the failure mode being gradual page-cache growth, not a spike).
- API with concurrent requests exceeding worker count, to confirm requests queue rather
  than the process falling over.
- Malformed and adversarial payloads (wrong shape, NaNs, out-of-range sensor values,
  strings where floats belong) — must return 422, never 500.
- LLM provider outage — must degrade to "prediction available, report unavailable",
  never fail the prediction.

## Regression testing

- The full suite runs before every commit; CI enforces it on every push (Day 11).
- **Every fixed bug gets a test.** The Day 4 deadlocks are the model case: the fix is a
  `conftest.py` whose absence causes an infinite hang, so the regression test is the test
  suite itself — if the import order regresses, the suite stops terminating.
- `models/metrics.json` is committed, giving a historical record; an unexplained metric
  drop between commits is a regression signal even without an automated check.

## Validation methodology

| Property | How it is validated |
|---|---|
| **No data leakage** | Tests assert `max(train.datetime) < min(test.datetime)` and that the scaler's mean/var match train-only statistics. |
| **No cross-machine sequences** | Test asserts every window's `machine_id` values are identical. |
| **Label correctness** | Tests place a synthetic failure and assert exactly the preceding 24 hours are labeled 1. |
| **Metric honesty** | Accuracy is not computed anywhere in `ModelEvaluator`, so it cannot be reported by accident. |
| **Artifact round-trip** | `train_model.py` reloads the checkpoint from disk before final evaluation, so a broken save fails the run rather than lurking until Day 6. |
| **Reproducibility** | Seeded generator and seeded epoch shuffling; the same command produces the same dataset and the same batch order. |

---

# Risk Register

## R-1 — Out of memory during training

| Field | Detail |
|---|---|
| **Root cause** | `X_train` is 4.2 GB and `X_test` 1.0 GB; a naive `np.load()` without `mmap_mode` loads both fully, and TensorFlow needs another ~1–2 GB. |
| **Impact** | Process killed by the OS, or thrashing that makes training 100× slower. **High.** |
| **Likelihood** | High if unguarded — the naive code is the obvious code. |
| **Mitigation** | `mmap_mode='r'` on every load; batch-level materialization in `iter_batches`; float32 everywhere; no full-array copies. |
| **Recovery** | Reduce batch size; if still failing, subsample machines to shrink the tensor and retrain. The `.keras` checkpoint from the last good epoch survives. |
| **Status** | Mitigated. |

## R-2 — TensorFlow deadlock from library import order

| Field | Detail |
|---|---|
| **Root cause** | If pandas or scikit-learn perform BLAS work **before** TensorFlow is imported, TF's first `tf.function` execution deadlocks. Reproduced deterministically outside pytest: `pandas.rolling()` → `import tensorflow` → train step **hangs**; `import tensorflow` → `pandas.rolling()` → train step **works**. Thread-pool initialization order between OpenBLAS and TF's Eigen pool. |
| **Impact** | Total, and *silent*: 0% CPU, no traceback, no timeout, no error. Indistinguishable from "slow". **Critical.** |
| **Likelihood** | Certain, on any code path that touches pandas before TF. |
| **Mitigation** | `tests/conftest.py` imports TensorFlow before any test module loads. Scripts import `src.models` (which imports TF) before doing pandas work. Both carry docstrings explaining why, so the import is not "cleaned up" as unused. |
| **Recovery** | Symptom recognition is the whole battle: a hung Python process at 0% CPU with TF loaded is this bug. Check import order before debugging anything else. |
| **Status** | Mitigated and regression-covered. |

## R-3 — Keras `fit()` / `predict()` deadlock on memmapped data

| Field | Detail |
|---|---|
| **Root cause** | `keras.utils.Sequence`, `PyDataset`, and `tf.data.Dataset.from_generator(...).prefetch()` all feed batches through a background worker thread, which deadlocks against memmapped reads on this platform (and, under pytest, against fd-level output capture). |
| **Impact** | Training and inference both hang indefinitely. **Critical.** |
| **Likelihood** | Certain — hit three times on Day 4 across three different input-pipeline styles. |
| **Mitigation** | Neither `fit()` nor `predict()` is used. `ModelTrainer.train()` is a synchronous `GradientTape` loop; `ModelEvaluator._predict_in_batches()` calls `model(x, training=False)` directly. Callbacks reimplemented inline. |
| **Recovery** | Documented in `src/models/trainer.py`'s docstring and `docs/Day4.md`. If a future TF version fixes it, the loop can be replaced — but only with a benchmark proving `fit()` completes. |
| **Cost accepted** | No I/O/compute overlap, and manual reimplementation of four callbacks. Worth it: a hanging trainer has infinite cost. |
| **Status** | Mitigated. |

## R-4 — Dependency conflicts

| Field | Detail |
|---|---|
| **Root cause** | TensorFlow constrains NumPy (`<2.0`) and Python (`≤3.12`). System Python is 3.14. LangChain moves fast and breaks its own APIs between minors. |
| **Impact** | Environment simply will not build, or breaks after an unrelated `pip install`. **Medium.** |
| **Likelihood** | Medium — already hit once on Day 1 (Python 3.14 vs TensorFlow). |
| **Mitigation** | Pinned version *ranges* in `requirements.txt`; the 3.12 requirement documented in three places; virtualenv isolation; CI installs from the pinned file on a clean runner. |
| **Recovery** | Delete `venv/`, recreate from `requirements.txt`. Nothing in the project depends on environment state. |
| **Status** | Mitigated. |

## R-5 — Corrupted or unloadable checkpoint

| Field | Detail |
|---|---|
| **Root cause** | Interruption mid-`save()`; a Keras version change altering the `.keras` format; disk exhaustion. |
| **Impact** | Hours of training lost, or a model that trains but cannot be served. **Medium-High.** |
| **Likelihood** | Low-Medium. |
| **Mitigation** | `train_model.py` reloads the checkpoint from disk and evaluates *that*, so a bad save fails the training run immediately rather than on Day 6. Best weights are also held in memory for the run's duration. `KeyboardInterrupt` is caught for a clean exit. |
| **Recovery** | Retrain — currently unavoidable, since there is no resume (**TD-2**). |
| **Status** | Partially mitigated; resume support is planned. |

## R-6 — Data inconsistency between training and inference

| Field | Detail |
|---|---|
| **Root cause** | Inference builds features in a different order, or applies a scaler refitted on live data, or windows across machine boundaries. Any of these silently degrades predictions while every test still passes. |
| **Impact** | Model appears healthy, predictions are garbage. **High** — and the hardest class of bug to notice. |
| **Likelihood** | Medium — it is the classic training/serving skew failure. |
| **Mitigation** | `feature_columns.txt` is the explicit ordered contract; the fitted scaler is persisted and must be loaded, never refitted; Day 6's `Predictor` will reuse `DataPreprocessor`'s feature code rather than reimplementing it. |
| **Recovery** | Assert on feature-name equality at inference startup and refuse to serve on mismatch. |
| **Status** | ✅ **Closed Day 6 — measured, not argued.** `Predictor` reuses `DataPreprocessor` rather than reimplementing features; scoring the raw CSVs and the stored tensors agrees to 2.98e-08 with **100% alert-decision agreement** across 172,800 test sequences. Asserted by `tests/integration/test_training_serving_parity.py`. |

## R-7 — Model learns nothing useful (imbalance collapse)

| Field | Detail |
|---|---|
| **Root cause** | 957 positives among 698,400 sequences. The trivial "always predict 0" solution achieves 99.86% accuracy and a very low loss. |
| **Impact** | A model that trains cleanly, reports a beautiful loss curve, and is worthless. **High.** |
| **Likelihood** | High without countermeasures. |
| **Mitigation** | Class weights of {0: 0.50, 1: 364.89}; AUC/precision/recall/F1 as the only reported metrics; accuracy deliberately not computed; early stopping monitors `val_auc`, not `val_loss`. |
| **Recovery** | If AUC ≈ 0.5, escalate through: focal loss → positive-window oversampling → longer sequences → wider prediction horizon. Day 5 owns this. |
| **Status** | Mitigated; awaiting empirical confirmation from the first full training run. |

## R-8 — Optimistic metrics from validating on the test set

| Field | Detail |
|---|---|
| **Root cause** | Day 4 passes `X_val=X_test` for monitoring, so early stopping and checkpoint selection observe the test set. |
| **Impact** | Reported test metrics are mildly optimistic — model selection has peeked. **Medium.** |
| **Likelihood** | Certain, currently. |
| **Mitigation** | Explicitly logged as **TD-1** here and in `docs/Day4.md`, so no reader mistakes the Day 4 numbers for a clean generalization estimate. |
| **Recovery** | Day 5: carve a chronological validation slice from the tail of the training period; touch the test set once, at the end. |
| **Status** | Open, scheduled. |

## R-9 — Performance bottleneck: synchronous data loading

| Field | Detail |
|---|---|
| **Root cause** | Removing prefetch (R-3) means disk reads are not overlapped with compute; each batch's I/O is dead time. |
| **Impact** | Longer epochs. **Low-Medium** — an inconvenience, not a correctness issue. |
| **Likelihood** | Certain, by construction. |
| **Mitigation** | Sorted indices within a batch keep reads monotonic; the OS page cache absorbs repeat epochs; `tf.function` keeps the compute side fast. |
| **Recovery** | If epochs prove intolerable: reduce sequence count by striding windows (e.g. every 2nd hour), or reduce feature count via importance analysis. Re-enabling prefetch is **not** an option until R-3 is proven fixed. |
| **Status** | Accepted. |

## R-10 — LLM provider failure or cost

| Field | Detail |
|---|---|
| **Root cause** | API outage, rate limiting, key expiry, or unbudgeted token spend. |
| **Impact** | Reports unavailable. Predictions unaffected. **Low-Medium.** |
| **Likelihood** | Medium over the project's life. |
| **Mitigation** | Three interchangeable providers (OpenAI / Gemini / **local Ollama**) selected by config; `LLMConnectionError` is a distinct exception type so the API can degrade gracefully; report generation will be async so it never blocks a prediction. |
| **Recovery** | Switch provider via `.env`; fall back to a deterministic template-generated report. |
| **Status** | ✅ **Closed Day 7.** Three interchangeable providers; `LLMConnectionError` is distinct from `ReportGenerationError`, and the CLI prints the prediction when the report fails. Exercised for real — every run before Ollama was configured took this path. |

## R-11 — Deployment failure

| Field | Detail |
|---|---|
| **Root cause** | Image missing the model volume; wrong Python version in the base image; missing env vars; free-tier memory limit below TF's ~1 GB footprint. |
| **Impact** | Service fails to start. **Medium.** |
| **Likelihood** | Medium — first deployment always finds something. |
| **Mitigation** | Multi-stage Dockerfiles pinned to `python:3.12-slim`; `.env.example` enumerates every variable; `/health` reports model-loaded status so a half-started service is visibly unhealthy; compose file mirrors the production topology locally. |
| **Recovery** | Roll back to the previous image tag; artifacts are volume-mounted so no rebuild is needed to restore a model. |
| **Status** | ✅ **Closed Day 11.** Both images build; `docker compose up` brings up a healthy stack with health-gated ordering; an API container with no model mounted reports `degraded` and refuses predictions with a 503 naming the fix. |

## R-12 — Security: secret leakage or prompt injection

| Field | Detail |
|---|---|
| **Root cause** | An API key committed to git or written to a log; or machine-derived text interpreted as instructions by the LLM. |
| **Impact** | Key compromise and unbudgeted spend; or manipulated report content. **Medium-High.** |
| **Likelihood** | Low with discipline, non-zero. |
| **Mitigation** | `.env` gitignored, `.env.example` valueless; settings never logged; sensor data is inserted into prompts as delimited data, not instructions; LLM output is displayed, never executed; `pip-audit` in CI. |
| **Recovery** | Rotate the key immediately; rewrite history if a key was committed; treat the old key as compromised regardless. |
| **Status** | Mitigated by convention; CI enforcement on Day 11. |

## R-13 — Documentation drifting behind the code

| Field | Detail |
|---|---|
| **Root cause** | Implementation moves faster than the write-up; a session ends without updating the plan. |
| **Impact** | The next session (human or AI) works from a false picture and re-derives or breaks decisions. This already happened once — `docs/handoff.md` described Day 3 as current while Day 4 code sat uncommitted in the working tree, and `CLAUDE.md` described a `tf.data` trainer that the tests had already replaced. **Medium-High**, and it compounds. |
| **Likelihood** | High without a process. |
| **Mitigation** | This document plus one `docs/DayX.md` per day; the mandatory session workflow (read plan → read latest day file → inspect repo → run tests → implement → test → **update both documents** → commit); documentation and implementation land in the same commit. |
| **Recovery** | Reconcile against the repository, which is the ultimate source of truth — file mtimes, git status, logs, and the test suite all carry evidence of true state. |
| **Status** | Addressed by this document's existence and the Day 4 reconciliation. |

---

# Milestones

## M1 — Foundation (Day 1) ✅

| Field | Detail |
|---|---|
| **Objective** | A professional, reproducible Python project skeleton that everything else can be built inside. |
| **Tasks** | Python 3.12 venv; folder structure; `Settings` via pydantic-settings; loguru logger; exception hierarchy; Makefile; `.gitignore`; README; smoke tests. |
| **Deliverables** | 33 files, 1,511 lines; 19 passing tests. |
| **Dependencies** | None. |
| **Effort** | 1 day. |
| **Success criteria** | `make test` green; `get_settings()` reads `.env`; every layer can log and raise. ✅ |

## M2 — Data foundation (Day 2) ✅

| Field | Detail |
|---|---|
| **Objective** | A realistic, reproducible dataset plus the code to load and trust it. |
| **Tasks** | Synthetic generator (5 tables, 883K rows, seed 42); `DataIngestion`; `DataValidator` + `ValidationReport`; 8-dimension EDA. |
| **Deliverables** | 5 source files, 5 CSVs, sample dataset, EDA report; 22 new tests (41 total). |
| **Dependencies** | M1. |
| **Effort** | 1 day. |
| **Success criteria** | Data regenerates identically from seed; validator catches injected schema/null/range faults; EDA quantifies the imbalance. ✅ |

## M3 — Feature engineering (Day 3) ✅

| Field | Detail |
|---|---|
| **Objective** | Turn 5 raw tables into leak-free LSTM tensors. |
| **Tasks** | Merge; 48 engineered features; 24 h labels; temporal split; train-only scaler; per-machine windowing. |
| **Deliverables** | `preprocessing.py` (~780 lines); `X_train (698400,24,63)`, `X_test (172800,24,63)`; scaler; feature list; 27 new tests (68 total). |
| **Dependencies** | M2. |
| **Effort** | 1 day. |
| **Success criteria** | All three leakage invariants asserted by tests; no NaNs in output; shapes correct. ✅ |

## M4 — Model architecture & training (Day 4) ✅

| Field | Detail |
|---|---|
| **Objective** | A working LSTM that trains to completion on the full dataset and saves an evaluable artifact. |
| **Tasks** | `PredictiveMaintenanceModel`; `ModelTrainer` with class weights and inline callbacks; `ModelEvaluator`; `train_model.py`; model tests; **resolve the training deadlocks**. |
| **Deliverables** | 4 source files + 1 test file + `conftest.py`; 7 new tests (75 total); `metrics.json`; `training_history.json`. |
| **Dependencies** | M3. |
| **Effort** | 1 day + deadlock debugging. |
| **Success criteria** | `make test` green (75/75) ✅; training runs to completion without hanging ✅; `metrics.json` written with AUC > 0.5 ✅ (**0.9999**). All met. |

## M5 — Evaluation & optimization (Day 5) ✅

| Field | Detail |
|---|---|
| **Objective** | Turn a trained model into a *characterized* model, and pay down Day 4's debt. |
| **Tasks** | Proper chronological validation split (**TD-1**); threshold sweep + PR curve; training-curve plots; per-machine error analysis; checkpoint resume (**TD-2**); hyperparameter comparison. |
| **Deliverables** | Evaluation report, plots, tuned model, updated `metrics.json`. |
| **Dependencies** | M4. |
| **Effort** | 1 day. |
| **Success criteria** | AUC ≥ 0.85 with a defensible operating point ✅ (0.9999 at t=0.3415, chosen on validation); validation set no longer the test set ✅. All met. |

## M6 — Prediction pipeline (Day 6) ✅

| Field | Detail |
|---|---|
| **Objective** | Score raw sensor data without touching the training code. |
| **Tasks** | `Predictor` (load model + scaler + feature list); raw-rows → features → window → probability; risk banding; feature-name assertion at startup; batch scoring. |
| **Deliverables** | `src/prediction/predictor.py` + tests. |
| **Dependencies** | M5. |
| **Effort** | 1 day. |
| **Success criteria** | Identical predictions via the pipeline and via direct model call ✅ (100% alert agreement over 172,800 sequences); < 100 ms single-sequence latency ✅ (54 ms). All met. |

## M7 — GenAI report generation (Day 7) ✅

| Field | Detail |
|---|---|
| **Objective** | A prediction becomes a maintenance work order in English. |
| **Tasks** | Prompt templates; `report_chain`; provider abstraction (OpenAI/Gemini/Ollama); output parsing; graceful degradation. |
| **Deliverables** | `src/genai/prompts.py`, `chains.py` + tests with a mocked LLM. |
| **Dependencies** | M6. |
| **Effort** | 1 day. |
| **Success criteria** | Report names the contributing sensors, an urgency level, and a recommended action ✅; provider swap is a config change only ✅. Verified against a live local model, which exposed three grounding bugs the mocked tests could not. All met. |

## M8 — GenAI assistant (Day 8) ✅

| Field | Detail |
|---|---|
| **Objective** | Conversational Q&A over machine history. |
| **Tasks** | `assistant.py`; conversation memory; context construction from predictions + maintenance history. |
| **Deliverables** | `src/genai/assistant.py` + tests. |
| **Dependencies** | M7. |
| **Effort** | 1 day. |
| **Success criteria** | Answers grounded in actual feature values, not invention ✅ — verified against a live model, which caught the assistant accepting a false premise. All met. |

## M9 — REST API (Day 9) ✅

| Field | Detail |
|---|---|
| **Objective** | Everything above, over HTTP. |
| **Tasks** | FastAPI app + lifespan model loading; Pydantic schemas; the five routes; exception→status mapping; CORS; integration tests. |
| **Deliverables** | `src/api/` complete + `tests/integration/`. |
| **Dependencies** | M8. |
| **Effort** | 1 day. |
| **Success criteria** | `/docs` renders ✅; p95 `/predict` < 500 ms ✅ (**137 ms median**); malformed input returns 422, never 500 ✅. All met, verified live against the real model and 100 machines. |

## M10 — Dashboard (Day 10) ✅

| Field | Detail |
|---|---|
| **Objective** | A non-technical user can see fleet risk and read a report. |
| **Tasks** | Four Streamlit pages; sensor charts; risk gauge; alert feed; one-click report. |
| **Deliverables** | `dashboard/app.py` + components. |
| **Dependencies** | M9. |
| **Effort** | 1 day. |
| **Success criteria** | Dashboard is a pure API client — zero direct model imports ✅, asserted by two tests that parse the imports. All met. |

## M11 — Deployment (Day 11) ✅

| Field | Detail |
|---|---|
| **Objective** | Reproducible deployment plus an automated quality gate. |
| **Tasks** | Two Dockerfiles; docker-compose; GitHub Actions (lint/format/typecheck/test/build); `pip-audit`; deploy to a free-tier host. |
| **Deliverables** | `docker/`, `.github/workflows/ci.yml`. |
| **Dependencies** | M10. |
| **Effort** | 1 day. |
| **Success criteria** | `docker compose up` yields a working API + dashboard ✅ — both containers healthy, health-gated ordering confirmed, containerised prediction byte-identical to the host run; CI green on push ✅ — all four jobs pass on `458ce03`, including amd64 image builds. All met. |

## M12 — Polish & demo (Day 12) ✅

| Field | Detail |
|---|---|
| **Objective** | Ship it. |
| **Tasks** | README with real metrics and screenshots; architecture diagrams; end-to-end run from clean checkout; demo recording; final doc sync. |
| **Deliverables** | Final docs, demo. |
| **Dependencies** | M11. |
| **Effort** | 1 day. |
| **Success criteria** | A stranger can clone, follow the README, and reach a working state ✅ — verified on a fresh clone: 193 passed / 18 skipped in 20.5 s, and a regenerated dataset with an MD5 identical to the original. |

---

# Roadmap

Each day has a matching document in `docs/`.

| Day | Date | Focus | Document | Status |
|---|---|---|---|---|
| **Day 1** | 2026-08-19 | Project setup & foundation | `docs/Day1.md` | ✅ Complete — commit `e952e24`, `e3408fd` |
| **Day 2** | 2026-08-20 | Dataset, EDA & data pipeline | `docs/Day2.md` | ✅ Complete — commit `6ed62e9` |
| **Day 3** | 2026-08-21 | Feature engineering & preprocessing | `docs/Day3.md` | ✅ Complete — commit `79c094a` |
| **Day 4** | 2026-08-22/23 | LSTM architecture & training | `docs/Day4.md` | ✅ Complete — 75/75 tests, trained model, AUC 0.9999 |
| **Day 5** | 2026-08-23 | Model evaluation & optimization | `docs/Day5.md` | ✅ Complete — 3-way split, F1 0.8949, 90 tests |
| **Day 6** | 2026-08-24 | Prediction pipeline & inference | `docs/Day6.md` | ✅ Complete — R-6 closed, 113 unit + 4 integration tests |
| **Day 7** | 2026-08-24 | LangChain setup & report generation | `docs/Day7.md` | ✅ Complete — grounded reports, R-10 closed, 141 tests |
| **Day 8** | 2026-08-24 | GenAI assistant & maintenance Q&A | `docs/Day8.md` | ✅ Complete — multi-turn sessions, live grounding tests, 161 tests |
| **Day 9** | 2026-08-24 | FastAPI REST API | `docs/Day9.md` | ✅ Complete — 9 endpoints, 185 unit tests |
| **Day 10** | 2026-08-24 | Streamlit dashboard | `docs/Day10.md` | ✅ Complete — 3 views, 211 unit tests |
| **Day 11** | 2026-08-24 | Docker, CI/CD & deployment | `docs/Day11.md` | ✅ Complete — images built (2.87 GB / 803 MB), compose verified, 4 CI jobs |
| **Day 12** | 2026-08-24 | Final polish, docs & demo | `docs/Day12.md` | ✅ Complete — clean-checkout verified, RESULTS.md, TD-4 closed |
| **Day 13** | 2026-08-25 | Point-in-time assessment (`as_of`) | `docs/Day13.md` | ✅ Complete — post-project; 229 unit + 13 integration tests |
| **Day 14** | 2026-08-25 | Quality-gate drift, accessibility & the horizon chart | `docs/Day14.md` | ✅ Complete — post-project; local/CI gates unified, 2 WCAG AA failures fixed, 233 unit tests, README demo asset |
| **Day 15** | 2026-08-25 | Full-repository production review + seeded retrain | `docs/Day15.md` | ✅ Complete — post-project; training made reproducible and retrained (**F1 0.9086**, t=0.3415), unbounded `/fleet` cache bounded, `plot_horizon.py` hardened, 238 unit tests |

Days 13 and 14 sit outside the original 12 milestones. Day 13 exists because the
finished product had a presentation defect the plan never anticipated: assessed at
the dataset's last hour the fleet is always quiet, so the demo showed a model doing
nothing. Day 14 exists because several checks were passing without proving what
they appeared to prove — the local and CI quality gates were checking different
programs, and the dashboard's two most-scanned risk colours failed WCAG AA. See
`docs/Day13.md` and `docs/Day14.md`.

```
Day 1  ✅ Foundation
Day 2  ✅ Data + EDA
Day 3  ✅ Features + LSTM tensors
Day 4  ✅ Model + training loop
Day 5  ✅ Evaluation + tuning
Day 6  ✅ Inference pipeline
Day 7  ✅ LLM reports
Day 8  ✅ LLM assistant
Day 9  ✅ REST API
Day 10 ✅ Dashboard
Day 11 ✅ Docker + CI
Day 12 ✅ Demo
```

---

# Current Project Status

**As of 2026-08-25.**

## Completed

| Item | Evidence |
|---|---|
| **Trained LSTM: AUC 0.9999 / F1 0.7530** | `models/metrics.json`, `models/lstm_predictive_maintenance.keras` |
| Project foundation, config, logging, exceptions | `config/`, `src/utils/`, 19 smoke tests |
| Synthetic dataset generator (883,231 rows, seed 42) | `scripts/generate_data.py`, `data/raw/` |
| Ingestion + validation layer | `src/data/ingestion.py`, `validation.py`, 22 tests |
| EDA across 8 dimensions | `scripts/eda_analysis.py` |
| Feature engineering → LSTM tensors | `src/data/preprocessing.py`, 27 tests, `data/processed/*.npy` |
| LSTM architecture | `src/models/lstm_model.py` |
| Manual training loop with class weights + inline callbacks | `src/models/trainer.py` |
| Imbalance-aware evaluator | `src/models/evaluator.py` |
| Training entry point | `scripts/train_model.py` |
| **Two platform deadlocks diagnosed and fixed** | `tests/conftest.py`, trainer/evaluator docstrings |
| **Full test suite green: 75/75** | `make test`, ~4 s warm |
| **This documentation system** | `IMPLEMENTATION_PLAN.md`, `docs/Day1-4.md` |

## Day 4 results (2026-08-23)

Trained on 698,400 sequences; early stopping fired at epoch 6 of 30 (best weights from
epoch 1); ~25.7 minutes wall clock at ~36 ms/batch.

| Metric | Value | Target | Met? |
|---|---|---|---|
| **ROC-AUC** | **0.9999** | ≥ 0.85 | ✅ |
| **Precision** | **0.6258** | ≥ 0.30 | ✅ |
| **Recall** | **0.9450** | ≥ 0.60 | ✅ |
| **F1** | **0.7530** | ≥ 0.40 | ✅ |

Confusion matrix at threshold 0.5 — **189 of 200 failures caught, 11 missed, 113 false
alarms** across 172,800 test sequences:

```
                 predicted 0   predicted 1
actual 0            172,487           113
actual 1                 11           189
```

Benchmarks: training 36 ms/batch (~1.7 min/epoch train-only); single-sequence inference
**54.0 ms median / 55.4 ms p95** — NFR-3 (<100 ms) **PASS**. The saved artifact was
reloaded in a fresh process and verified to predict.

> **Superseded by Day 5.** These Day 4 numbers came from a run where `X_val=X_test`, so
> early stopping and checkpoint selection observed the test set. Day 5 replaced the split
> and retrained; the current figures are below.

## Day 5 results (2026-08-23) — superseded by the Day 15 seeded retrain below

Retrained on a clean three-way split (567,000 train / 129,000 validation / 172,800 test),
monitoring `val_f1`. Early stopping at epoch 20 of 30; best weights from epoch 15.
Threshold chosen on **validation** (best-F1, t=0.6678), then test scored once.

| Metric | Day 4 (peeked) | Day 5 |
|---|---|---|
| ROC-AUC | 0.9999 | 0.9997 |
| Precision | 0.6258 | **0.8756** |
| Recall | 0.9450 | 0.9150 |
| **F1** | 0.7530 | **0.8949** |
| Missed failures | 11 | 17 |
| **False alarms** | 113 | **26** |

183 of 200 failures caught, 26 false alarms across 172,800 hourly readings.

### Day 15 retrain (2026-08-25) — current, and reproducible

The Day 5 model was trained without a seed, so nothing above could be
re-derived. `scripts/train_model.py --seed 42` now reproduces the deployed
model exactly. Early stopping at epoch 28 of 30; best weights from epoch 23
(`val_f1` 0.9602). Threshold chosen on **validation** (best-F1, t=0.3415),
then test scored once.

| Metric | Day 5 | Day 15 (deployed) |
|---|---|---|
| ROC-AUC | 0.9997 | **0.9999** |
| Precision | 0.8756 | **0.8976** |
| Recall | 0.9150 | **0.9200** |
| **F1** | 0.8949 | **0.9086** |
| Missed failures | 17 | **16** |
| **False alarms** | 26 | **21** |
| Reproducible | no | **yes** |

184 of 200 hourly labels caught, 21 false alarms across 172,800 readings, and
**8 of 8 failure events** warned about — median 23.5 h of lead time, worst case
16 h. The threshold move from 0.6678 to 0.3415 also pulled `RISK_BAND_MEDIUM`
down from 0.30 to 0.15; leaving it would have made "medium" a four-point sliver
nothing could land in.

**Why the honest split scored higher.** Day 4 monitored `val_auc`, which saturates under
this imbalance — it peaked in epoch 1 and early stopping kept those weights. With a real
validation set and `val_f1` as the monitor, training ran to epoch 20 and kept epoch 15,
whose validation precision was 0.913 against epoch 1's 0.214. AUC had been reporting a
flat quantity while precision improved underneath it.

**One reversal worth recording.** Selecting the operating point by cost (100:1 FN:FP) chose
t=0.0003 — a noise-floor threshold that reached recall 1.0 on 175 validation positives and
cost 15 points of test F1. The default is now best-F1; `sweep_thresholds()` returns a
`lowest_cost_is_degenerate` flag so the failure mode cannot recur silently.

## In progress

Nothing. All 12 milestones are delivered, plus the Day 13 `as_of` addition and
the Day 14 quality-gate and accessibility fixes.

## Pending

Nothing inside the planned scope. Anything further is in **Future improvements**
below.

## Blocked

Nothing is blocked. Both Day 4 blockers (the import-order deadlock and the `fit()`
deadlock) were resolved on Day 4.

## Technical debt

| ID | Item | Why it exists | Repayment |
|---|---|---|---|
| ~~TD-1~~ | ~~Validation set *is* the test set~~ | — | ✅ **Repaid Day 5** — 3-way chronological split; test period unchanged |
| ~~TD-2~~ | ~~No checkpoint resume~~ | — | ✅ **Repaid Day 5** — `--resume` + `<checkpoint>.state.json` |
| ~~TD-3~~ | ~~No training curves~~ | — | ✅ **Repaid Day 5** — `training_curves.png`, `pr_curve.png` |
| ~~TD-4~~ | ~~`docs/handoff.md` overlaps this plan~~ | — | ✅ **Repaid Day 12** — removed; superseded by this plan and by `docs/Day1-3.md`, and preserved in git history |
| ~~TD-8~~ | ~~mypy reports 159 errors~~ | — | ✅ **Repaid Day 12** — 0 errors across 29 files; the CI check is now **blocking**. 128 of the 159 came from one line: `get_logger()` annotated with `loguru.logger`, an instance rather than a class |
| ~~TD-5~~ | ~~`CLAUDE.md` referenced a `tf.data` trainer and root-level `debug_fit*.py` files~~ | Drift during Day 4's debugging | ✅ **Repaid Day 4** — `CLAUDE.md` corrected |
| ~~TD-6~~ | ~~Threshold fixed at 0.5~~ | — | ✅ **Repaid Day 5** — validation sweep; deployed t=0.3415 since the Day 15 retrain |
| ~~TD-7~~ | ~~Integration tests: 9~~ | — | ✅ **Repaid Day 13** — 13 integration tests; `test_time_travel.py` adds four covering point-in-time assessment against the real model, including a leakage check |

## Known issues

| ID | Severity | Issue | Workaround |
|---|---|---|---|
| **K-1** | Low | `data/sample/` contains **zero** failure events (30 days × 10 machines is too small) | Tests synthesize positive labels; use the full dataset for anything model-quality-related |
| **K-2** | Info | TensorFlow's first import takes ~90 s on ARM64 macOS | Expected; subsequent imports are fast |
| **K-3** | Info | Raw and processed data are gitignored (5.2 GB) | Regenerate with `generate_data.py` + `run_preprocessing.py` |
| **K-4** | Medium | Training is CPU-only and epochs are long | Accepted for this project; documented in benchmarks |
| **K-5** | Info | VS Code may show thousands of pending changes from `~/.antigravity/` | Not project files; ignore |
| **K-6** | Info | The dataset is fixed to 2024-01-01 → 2024-12-30 (`scripts/generate_data.py`, seed 42) | Deliberate — reproducibility outranks a plausible-looking date. Use the dashboard's **Rewind** control to assess any hour in that range |

## Future improvements

Beyond the 12-day scope, in rough priority order:

1. **Multi-class failure-mode prediction** — the dataset already labels comp1–comp4.
2. **Remaining-useful-life regression** alongside binary classification.
3. **Explainability** — SHAP or attention weights so a report can cite evidence, not just correlation.
4. **Model monitoring & drift detection** in production.
5. **Automated retraining** on a schedule with champion/challenger comparison.
6. **Real data connector** (SCADA/OPC-UA/MQTT ingestion) replacing the synthetic feed.
7. **Attention or Transformer variant**, revisited once positive-class volume supports it.
8. **Model quantization / TF Lite** for edge deployment.

## Completion percentage

**100%**

```
[██████████████████████████████] 100%
 Foundation ✅  Data ✅  Features ✅  Model ✅  Eval ✅  Inference ✅  GenAI ✅  API ✅  UI ✅  Deploy ✅
```

| Module | Completion |
|---|---|
| `config/` | 100% |
| `src/utils/` | 100% |
| `src/data/` | 100% |
| `src/models/` | 100% — trained, evaluated, artifacts on disk |
| `src/prediction/` | 100% |
| `src/genai/` | 100% |
| `src/api/` | 100% |
| `dashboard/` | 100% |
| `docker/` + CI | 100% — images built, compose verified. **CI status belongs on the [live badge](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/actions/workflows/ci.yml), not in this table** — a flat "CI green" here read as true through three consecutive red runs (#6–#8). |
| Documentation | 95% |
| Tests | 241 unit + 13 integration passing; flake8/Black/isort/mypy clean across five paths, locally and in CI |

---

# Session Workflow (mandatory)

Every implementation session, in order:

1. Read **`IMPLEMENTATION_PLAN.md`** (this file) — especially *Current Project Status*.
2. Read the **latest `docs/DayX.md`** — especially *Remaining Tasks* and *Next Day Plan*.
3. **Inspect the repository** — `git status`, `ls` the relevant directories, check file
   mtimes. Do not trust the docs over the filesystem; when they disagree, the filesystem
   wins and the docs get corrected.
4. **Check git status** for uncommitted work from an interrupted session.
5. **Verify previous work still exists** — artifacts, data files, model checkpoints.
6. **Run the tests before changing anything** (`make test`) to establish a baseline.
7. **Continue from the latest unfinished task.**
8. **Work autonomously to the next meaningful milestone.**
9. **Run the tests after every major change.**
10. **Fix failures before proceeding** — never build on a red suite.
11. **Update `IMPLEMENTATION_PLAN.md`** — at minimum *Current Project Status*.
12. **Update `docs/DayX.md`** with everything performed.
13. **Never let documentation fall behind implementation.** Docs and code ship in the
    same commit.

At session end, report: completed work · files changed · tests executed · bugs fixed ·
completion percentage · remaining tasks · recommended next steps.
