# Architecture Overview

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
│  │  Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Mat │   │
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
