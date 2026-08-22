# Architecture Overview

> **Scope of this document.** This is the *design-level* view: layers, responsibilities,
> and how data moves between them. It describes the finished system, so layers not yet
> built (prediction, GenAI, API, dashboard) appear here as designed rather than as
> implemented. For what actually exists today, current status, and the reasoning behind
> each decision, see [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ CSV/JSON │  │ Sensor APIs  │  │ Database (future)         │  │
│  └────┬─────┘  └──────┬───────┘  └─────────┬─────────────────┘  │
│       └───────────────┬┘                    │                   │
│                       ▼                     │                   │
│              ┌────────────────┐             │                   │
│              │  Data Loader   │◄────────────┘                   │
│              └────────┬───────┘                                 │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA VALIDATION LAYER                         │
│              ┌────────────────┐                                 │
│              │  Schema Check  │  - Column types                 │
│              │  Quality Check │  - Missing values               │
│              │  Range Check   │  - Outlier detection            │
│              └────────┬───────┘                                 │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                       │
│              ┌────────────────┐                                 │
│              │  Preprocessing │  - Normalization                │
│              │  Feature Eng.  │  - Rolling statistics           │
│              │  Sequencing    │  - Time window creation         │
│              └────────┬───────┘                                 │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODEL LAYER                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 TensorFlow LSTM Model                     │   │
│  │  Input (sequence) → LSTM → Dense → Sigmoid → P(failure)  │   │
│  └──────────────┬───────────────────────────────────────────┘   │
│                 │                                               │
│  ┌──────────────▼───────────────────────────────────────────┐   │
│  │              Model Evaluator                              │   │
│  │  AUC-ROC, Precision, Recall, F1, Confusion Matrix        │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION PIPELINE                            │
│              ┌────────────────┐                                 │
│              │   Predictor    │  - Load saved model             │
│              │                │  - Process new sensor data      │
│              │                │  - Return failure probability   │
│              └────────┬───────┘                                 │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENAI LAYER (LangChain)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Prompt     │  │   Chains     │  │   Maintenance         │  │
│  │   Templates  │  │  (Summary,   │  │   Assistant           │  │
│  │              │  │   Diagnose)  │  │   (Q&A)               │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ /health  │  │  /predict    │  │  /reports                 │  │
│  │          │  │  /predict/   │  │  /reports/summary         │  │
│  │          │  │   batch      │  │  /reports/diagnostic      │  │
│  └──────────┘  └──────────────┘  └───────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DASHBOARD (Streamlit)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Live Sensor Data │ Failure Probability │ AI Summary     │   │
│  │  Equipment Health │ Maintenance History │ Trend Charts   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Processing | Pandas, NumPy | Data manipulation & numerical computing |
| ML Framework | TensorFlow/Keras | LSTM model for time-series prediction |
| ML Utilities | Scikit-learn | Preprocessing, metrics, train/test split |
| GenAI | LangChain + LLM | Report generation, Q&A assistant |
| API | FastAPI + Uvicorn | REST API with auto-generated docs |
| Dashboard | Streamlit | Interactive visualization |
| Configuration | Pydantic Settings | Type-safe config management |
| Logging | Loguru | Structured, rotated logging |
| Testing | Pytest | Unit + integration testing |
| Code Quality | Black, Flake8, MyPy | Formatting, linting, type checking |
| Containerization | Docker | Reproducible deployments |
| CI/CD | GitHub Actions | Automated testing + deployment |

## Design Principles

1. **Separation of Concerns** — Each module handles one responsibility
2. **Dependency Injection** — Components receive dependencies, don't create them
3. **Configuration as Code** — All config in environment variables
4. **Fail Fast** — Validate data early, surface errors immediately
5. **Observability** — Structured logging at every layer

## Layer Dependency Rule

The packages form a strict chain. **Each layer may import only from layers to its left.**
No layer may reach into a later layer's internals, and none may be skipped.

```
config/ -> src/utils/ -> src/data/ -> src/models/ -> src/prediction/ -> src/genai/ -> src/api/ -> dashboard/
```

This is why `src/data/` contains no TensorFlow import and `src/models/` contains no
pandas import, and it is what allows each layer to be tested and deployed independently.

## Non-Negotiable Invariants

These are correctness properties, not preferences. Each is enforced by an explicit test,
because every one of them fails *silently* — producing better-looking numbers, not errors.

| Invariant | Why |
|---|---|
| Train/test split is **temporal**, never random | With 24-hour lag features, a random split puts hour `t` in train and `t+1` in test. Reported metrics become fiction. |
| `StandardScaler` is fit on **training data only** | Fitting on the full dataset leaks test-period statistics into training. |
| Sequence windows never span two `machine_id`s | A cross-machine window describes a machine that does not exist. |
| Model quality is judged on **AUC / precision / recall / F1**, never accuracy | At a 1:864 positive rate, "always predict no failure" scores 99.88%. |
| TensorFlow is imported **before** pandas / scikit-learn | They load Apache Arrow, which bundles a conflicting copy of abseil; the wrong order deadlocks the process at 0% CPU with no traceback. |
