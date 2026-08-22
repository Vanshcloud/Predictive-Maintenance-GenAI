# Predictive Maintenance + GenAI Insight Generator
## AI Project Handoff Document

**Last Updated:** 2026-08-22 (content below is frozen at end of Day 3)
**Current Milestone:** see `IMPLEMENTATION_PLAN.md`
**Repository:** [github.com/Vanshcloud/vigilant-lamp](https://github.com/Vanshcloud/vigilant-lamp)
**Author:** Vansh Tomar ([@Vanshcloud](https://github.com/Vanshcloud))

---

> ## ⚠️ SUPERSEDED — read `IMPLEMENTATION_PLAN.md` first
>
> As of **2026-08-23 (Day 4)** this project's documentation moved to:
>
> | Document | Role |
> |---|---|
> | **`IMPLEMENTATION_PLAN.md`** | **Single source of truth** — scope, stack, architecture, dataset, model, training, evaluation, deployment, standards, testing, risk register, milestones, roadmap, and current status |
> | **`docs/Day1.md` … `docs/DayN.md`** | One file per implementation day: what was planned, what was built, what broke, what was decided, and what comes next |
> | `handoff.md` (this file) | **Historical narrative log.** Accurate for Days 1–3; **not updated after 2026-08-22**. Kept because its Day 1–3 detail is the source the retroactive `docs/Day1-3.md` files were reconstructed from |
>
> Where this file and `IMPLEMENTATION_PLAN.md` disagree, **the plan wins**. Known
> divergences already: this file lists Day 4 as "Next" (it is complete), describes the
> project as ~25% done (it is ~35%), and its §11 "Planned Callbacks" describes Keras
> callbacks that the final implementation replaces with an inline manual loop.

---

# 1. Project Overview

| Field | Value |
|---|---|
| **Project Name** | Predictive Maintenance + GenAI Insight Generator |
| **Repository** | `Vanshcloud/vigilant-lamp` |
| **Local Path** | `/Users/vanshtomar/Desktop/VT22/Vigilant lamp/predictive-maintenance-genai/` |
| **Language** | Python 3.12 |
| **Framework** | TensorFlow (ML) + LangChain (GenAI) + FastAPI (API) + Streamlit (Dashboard) |

## Purpose
Build an end-to-end Predictive Maintenance platform that uses sensor telemetry data to predict equipment failures before they happen, then uses Generative AI to explain predictions in plain English.

## Problem Statement
Industrial equipment failure costs the global manufacturing industry $50 billion/year. Reactive maintenance is too late (equipment already broken). Preventive maintenance is wasteful (replacing parts on a schedule regardless of condition). Predictive maintenance uses data to fix equipment just before it fails — the optimal strategy.

## Goals
1. **ML Goal:** Train an LSTM neural network on sensor telemetry to predict failures 24 hours in advance.
2. **GenAI Goal:** Use LangChain + LLM to convert ML predictions into human-readable maintenance reports and a Q&A assistant.
3. **API Goal:** Expose predictions and reports via a FastAPI REST API.
4. **Dashboard Goal:** Build a Streamlit dashboard for maintenance managers.
5. **Production Goal:** Docker containerization, CI/CD, production-ready logging and error handling.

## Expected Outcome
A complete, deployable system where a user can:
1. Upload or stream sensor data
2. Get real-time failure predictions with confidence scores
3. Receive AI-generated maintenance reports explaining *why* a failure is predicted and *what to do*
4. Ask natural language questions ("Why is Machine #3 at risk?")
5. View all predictions and trends on an interactive dashboard

## Resume Objective
A production-quality project demonstrating expertise in: TensorFlow, time-series ML, LangChain, GenAI, FastAPI, Streamlit, data engineering, testing, Docker, and CI/CD — suitable for ML Engineer, AI Engineer, and Software Engineer interviews.

## Production Objective
A system that could be deployed at a manufacturing facility to reduce unplanned downtime by 30-50% through proactive maintenance scheduling.

---

# 2. Project Vision

## The Finished Product

The final product is a **full-stack AI platform** with 4 integrated components:

### 1. ML Prediction Engine (TensorFlow LSTM)
- Takes 24 hours of sensor data (voltage, rotation, pressure, vibration)
- Outputs: failure probability (0-100%), failure mode (comp1-comp4), urgency level
- Handles class imbalance (failures are <0.2% of data)

### 2. GenAI Report Generator (LangChain + LLM)
- Takes ML predictions and generates plain-English maintenance reports
- Example output: "Machine #47 has a 87% probability of component 3 failure within 24 hours. Recommended action: Schedule immediate bearing replacement. Evidence: vibration has increased 340% over the past 12 hours while pressure has dropped 18%."
- Q&A assistant: maintenance managers can ask questions in natural language

### 3. REST API (FastAPI)
- `POST /predict` — Submit sensor data, get failure predictions
- `POST /report` — Generate AI maintenance report
- `GET /health` — System health check
- `GET /machines/{id}/history` — Machine prediction history
- OpenAPI documentation auto-generated

### 4. Interactive Dashboard (Streamlit)
- Real-time prediction display for all machines
- Sensor trend charts (line charts, heatmaps)
- Alert feed (sorted by urgency)
- Machine detail view with historical predictions
- AI report generation with one click

### What Makes This Different
- **Not just model.fit():** Full data pipeline (ingestion → validation → feature engineering → model → prediction → GenAI → API → dashboard)
- **Production patterns:** Configuration management, logging, exception hierarchy, data validation, temporal train/test split
- **GenAI integration:** ML predictions are useless to maintenance managers without human-readable explanations
- **Tested:** 68+ unit tests, comprehensive error handling

---

# 3. Current Progress

## Current Milestone
**Day 3 of 12 Complete** — Data engineering backbone is fully built.

## Completed Work

| Day | Focus | Status | Commit |
|---|---|---|---|
| Day 1 | Project Setup & Foundation | ✅ Complete | `475e722` |
| Day 2 | Dataset, EDA & Data Pipeline | ✅ Complete | `755e95c` |
| Day 3 | Feature Engineering & Preprocessing | ✅ Complete | `020ad63` |

### Day 1 Deliverables
- Python 3.12 environment with all dependencies
- Professional folder structure (33 files)
- Configuration management (`pydantic-settings`)
- Logging system (`loguru`)
- Custom exception hierarchy
- Makefile, .gitignore, README
- 19 smoke tests passing

### Day 2 Deliverables
- Synthetic data generator (5 tables, 883K rows)
- DataIngestion class (auto-format detection, metadata logging)
- DataValidator class (schema, null, duplicate, range checks)
- EDA analysis script (8 dimensions)
- 22 data pipeline tests passing

### Day 3 Deliverables
- DataPreprocessor class (full feature engineering pipeline)
- 48 engineered features from 4 raw sensors
- Rolling statistics (3h/12h/24h), lag features (1h/6h/24h)
- Error count aggregation, maintenance time-since-last features
- Binary failure labels (24h prediction horizon)
- Temporal train/test split (no data leakage)
- StandardScaler normalization (fit on train only)
- LSTM sliding window sequences: X_train (698400, 24, 63)
- 27 preprocessing tests passing

## Pending Work

| Day | Focus | Status |
|---|---|---|
| Day 4 | LSTM Model Architecture & Training | 🔒 Next |
| Day 5 | Model Evaluation & Optimization | 🔒 Upcoming |
| Day 6 | Prediction Pipeline & Inference | 🔒 Upcoming |
| Day 7 | LangChain Setup & Report Generation | 🔒 Upcoming |
| Day 8 | GenAI Assistant & Maintenance Q&A | 🔒 Upcoming |
| Day 9 | FastAPI REST API | 🔒 Upcoming |
| Day 10 | Streamlit Dashboard | 🔒 Upcoming |
| Day 11 | Docker, CI/CD & Deployment | 🔒 Upcoming |
| Day 12 | Final Polish, Docs & Demo | 🔒 Upcoming |

## Progress: ~25% Complete

```
[████████░░░░░░░░░░░░░░░░░░░░░░] 25%
Data Pipeline ✅ → Model Training 🔜 → GenAI → API → Dashboard → Deploy
```

## Git Status

| Field | Value |
|---|---|
| **Branch** | `main` |
| **Latest Commit** | `020ad63` — "feat(data): add feature engineering, preprocessing, and LSTM sequencing" |
| **Remote** | `origin` → `https://github.com/Vanshcloud/vigilant-lamp.git` |
| **Status** | Clean — `nothing to commit, working tree clean` |
| **Total Commits** | 4 |

---

# 4. Folder Structure

```
predictive-maintenance-genai/
│
├── .env                          # Environment variables (GITIGNORED — contains secrets)
├── .env.example                  # Template showing required env vars
├── .flake8                       # Flake8 linting config (compatible with Black)
├── .gitignore                    # Comprehensive Python gitignore
├── Makefile                      # Developer shortcuts (make test, make lint, etc.)
├── README.md                     # Professional README with badges and architecture
├── pyproject.toml                # PEP 621 project metadata + tool configs
├── requirements.txt              # Pinned production dependencies (7 layers)
├── requirements-dev.txt          # Dev-only dependencies (extends production)
│
├── config/                       # ── CONFIGURATION LAYER ──
│   ├── __init__.py               # Re-exports get_settings
│   └── settings.py               # Pydantic-based settings (reads .env, typed, cached)
│
├── src/                          # ── ALL SOURCE CODE ──
│   ├── __init__.py               # Root package
│   │
│   ├── data/                     # ── DATA PIPELINE ──
│   │   ├── __init__.py           # Exports: DataIngestion, DataValidator, DataPreprocessor
│   │   ├── ingestion.py          # Load data from CSV/Parquet/JSON with metadata logging
│   │   ├── validation.py         # Schema, null, duplicate, range validation
│   │   └── preprocessing.py      # Feature engineering, labels, split, scale, LSTM windowing
│   │
│   ├── models/                   # ── ML MODELS (Day 4-5) ──
│   │   └── __init__.py           # Empty scaffold
│   │
│   ├── prediction/               # ── INFERENCE PIPELINE (Day 6) ──
│   │   └── __init__.py           # Empty scaffold
│   │
│   ├── genai/                    # ── LANGCHAIN + LLM (Day 7-8) ──
│   │   └── __init__.py           # Empty scaffold
│   │
│   ├── api/                      # ── FASTAPI REST API (Day 9) ──
│   │   ├── __init__.py
│   │   └── routes/
│   │       └── __init__.py       # Empty scaffold
│   │
│   └── utils/                    # ── SHARED UTILITIES ──
│       ├── __init__.py           # Re-exports get_logger
│       ├── logger.py             # Loguru: colored console + rotated file output
│       └── exceptions.py         # Custom exception hierarchy (Data/Model/GenAI/API)
│
├── scripts/                      # ── UTILITY SCRIPTS ──
│   ├── setup.sh                  # One-command project setup
│   ├── generate_data.py          # Synthetic data generator (5 tables, configurable)
│   ├── eda_analysis.py           # 8-dimension EDA analysis
│   └── run_preprocessing.py      # End-to-end preprocessing pipeline runner
│
├── data/                         # ── DATA STORAGE (mostly gitignored) ──
│   ├── raw/                      # Raw CSV files (876K telemetry + 4 support tables)
│   │   ├── telemetry.csv         # 876,000 rows: hourly sensor readings
│   │   ├── machines.csv          # 100 rows: machine metadata
│   │   ├── errors.csv            # 5,386 rows: non-failure error events
│   │   ├── maintenance.csv       # 1,698 rows: scheduled maintenance records
│   │   └── failures.csv          # 47 rows: actual failure events
│   ├── processed/                # Preprocessed numpy arrays + scaler
│   │   ├── X_train.npy           # (698400, 24, 63) — training sequences
│   │   ├── y_train.npy           # (698400,) — training labels
│   │   ├── X_test.npy            # (172800, 24, 63) — test sequences
│   │   ├── y_test.npy            # (172800,) — test labels
│   │   ├── scaler.joblib         # Fitted StandardScaler
│   │   └── feature_columns.txt   # 63 feature names
│   └── sample/                   # Small sample dataset for tests (committed to git)
│       ├── telemetry.csv         # 7,200 rows (10 machines × 30 days)
│       ├── machines.csv          # 10 rows
│       ├── errors.csv            # 53 rows
│       ├── maintenance.csv       # 18 rows
│       └── failures.csv          # 0 rows
│
├── models/                       # ── SAVED MODEL ARTIFACTS (Day 4+) ──
│   └── .gitkeep
│
├── notebooks/                    # ── JUPYTER NOTEBOOKS ──
│   └── .gitkeep
│
├── tests/                        # ── ALL TESTS ──
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_smoke.py         # 19 tests: imports, config, logging, exceptions
│   │   ├── test_data_pipeline.py # 22 tests: ingestion + validation
│   │   └── test_preprocessing.py # 27 tests: feature eng + labels + split + sequences
│   └── integration/
│       └── __init__.py
│
├── docs/                         # ── DOCUMENTATION ──
│   └── architecture.md           # System architecture diagram + tech stack
│
├── dashboard/                    # ── STREAMLIT DASHBOARD (Day 10) ──
│   └── .gitkeep
│
└── docker/                       # ── CONTAINERIZATION (Day 11) ──
    └── .gitkeep
```

---

# 5. Technologies Used

| Technology | Version | Purpose | Why Chosen |
|---|---|---|---|
| **Python** | 3.12.14 | Primary language | Industry standard for ML/AI. 3.12 specifically because TensorFlow doesn't support 3.13+ |
| **TensorFlow** | 2.21.0 | LSTM model for time-series prediction | Production-grade, supports LSTM, TensorBoard, TF Serving for deployment |
| **LangChain** | 0.3.30 | LLM orchestration for report generation | Industry standard for chaining LLM calls, prompt management, and RAG |
| **LangChain-OpenAI** | 0.1.x | OpenAI model integration | Direct GPT-4o integration via LangChain |
| **FastAPI** | 0.141.1 | REST API framework | Async-first, auto-OpenAPI docs, Pydantic integration, best Python API framework |
| **Streamlit** | 1.61.1 | Interactive dashboard | Fastest way to build data dashboards in Python, no frontend needed |
| **Pandas** | 2.3.3 | Data manipulation and analysis | The DataFrame standard for tabular data in Python |
| **NumPy** | 1.26.4 | Numerical computing | Required by TensorFlow, Pandas, scikit-learn |
| **Scikit-learn** | 1.9.0 | StandardScaler, metrics, utilities | Best preprocessing and evaluation toolkit |
| **Loguru** | 0.7.3 | Structured logging | Zero-config, colored output, file rotation — replaces Python's verbose `logging` |
| **Pydantic** | 2.13.4 | Data validation and settings | Type-safe configuration, API request/response models |
| **Pydantic-Settings** | 2.x | Environment variable management | Reads `.env`, validates types, follows 12-Factor App |
| **Pytest** | 8.4.2 | Testing framework | Most popular Python testing framework, rich plugin ecosystem |
| **Black** | 24.10.0 | Code formatting | Opinionated formatter — no debates about style |
| **Flake8** | 7.x | Linting | Catches code quality issues Black doesn't handle |
| **Mypy** | 1.8+ | Static type checking | Catches type errors before runtime |
| **Joblib** | 1.3+ | Model/scaler serialization | Fast, efficient serialization for sklearn objects |
| **Matplotlib/Seaborn** | 3.8+/0.13+ | Visualization (EDA) | Standard Python plotting libraries |
| **Docker** | TBD (Day 11) | Containerization | Reproducible deployment across environments |
| **GitHub Actions** | TBD (Day 11) | CI/CD | Automated testing and deployment on every push |

---

# 6. Development Environment

| Field | Value |
|---|---|
| **Operating System** | macOS (ARM64/Apple Silicon) |
| **Python Version** | 3.12.14 (installed via `brew install python@3.12`) |
| **System Python** | 3.14.0 (NOT used — incompatible with TensorFlow) |
| **Virtual Environment** | `venv/` (created with `/opt/homebrew/bin/python3.12 -m venv venv`) |
| **Activation** | `source venv/bin/activate` |
| **IDE** | VS Code with Antigravity IDE |
| **Package Manager** | pip (with requirements.txt pinning) |
| **Git Remote** | `origin` → `https://github.com/Vanshcloud/vigilant-lamp.git` |
| **Git Branch** | `main` |
| **Total Installed Packages** | ~218 (including transitive dependencies) |

### Important Commands

```bash
# Activate environment
source venv/bin/activate

# Run all tests
make test
# or: python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_preprocessing.py -v

# Format code
make format
# or: black . && isort .

# Lint code
make lint
# or: flake8 .

# Generate dataset
python scripts/generate_data.py
python scripts/generate_data.py --sample  # Small sample

# Run EDA
python scripts/eda_analysis.py

# Run preprocessing pipeline
python scripts/run_preprocessing.py

# Start API (future)
make run-api
# or: uvicorn src.api.main:app --reload --port 8000

# Start dashboard (future)
make run-dashboard
# or: streamlit run dashboard/app.py
```

---

# 7. Installed Dependencies

## Production Dependencies (`requirements.txt`)

| Layer | Package | Version Range | Purpose |
|---|---|---|---|
| **Core** | numpy | ≥1.24, <2.0 | Numerical computing, array operations |
| **Core** | pandas | ≥2.0, <3.0 | DataFrame operations, data manipulation |
| **Core** | scikit-learn | ≥1.3, <2.0 | StandardScaler, train_test_split, metrics |
| **ML** | tensorflow | ≥2.15, <3.0 | LSTM model building, training, inference |
| **GenAI** | langchain | ≥0.2, <1.0 | LLM chain orchestration |
| **GenAI** | langchain-community | ≥0.2, <1.0 | Community integrations |
| **GenAI** | langchain-openai | ≥0.1, <1.0 | OpenAI model wrappers |
| **GenAI** | openai | ≥1.0, <2.0 | OpenAI Python SDK |
| **API** | fastapi | ≥0.110, <1.0 | REST API framework |
| **API** | uvicorn[standard] | ≥0.27, <1.0 | ASGI server for FastAPI |
| **Dashboard** | streamlit | ≥1.30, <2.0 | Interactive web dashboard |
| **Config** | pydantic | ≥2.0, <3.0 | Data validation, API schemas |
| **Config** | pydantic-settings | ≥2.0, <3.0 | Environment variable management |
| **Config** | python-dotenv | ≥1.0, <2.0 | .env file loading |
| **Config** | loguru | ≥0.7, <1.0 | Structured logging |
| **Data** | joblib | ≥1.3, <2.0 | Scaler/model serialization |

## Dev Dependencies (`requirements-dev.txt`)

| Package | Purpose |
|---|---|
| pytest, pytest-cov, pytest-asyncio | Testing framework + coverage + async tests |
| black | Opinionated code formatter |
| flake8 | Linting / code quality |
| isort | Import sorting |
| mypy | Static type checking |
| jupyter, ipykernel | Notebook support |
| matplotlib, seaborn | Visualization (EDA) |
| httpx | API testing client |
| pre-commit | Git hooks for quality gates |

---

# 8. Dataset Information

## Overview

| Field | Value |
|---|---|
| **Name** | Synthetic Predictive Maintenance Dataset |
| **Modeled After** | Microsoft Azure Predictive Maintenance Dataset |
| **Source** | Generated locally via `scripts/generate_data.py` |
| **License** | Self-generated (no licensing restrictions) |
| **Real-world Reference** | [Kaggle: Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) |
| **Seed** | 42 (reproducible) |
| **Total Rows** | 883,231 |

## 5 Tables

| Table | Rows | Columns | Description |
|---|---|---|---|
| **telemetry** | 876,000 | 6 | Hourly sensor readings (datetime, machine_id, voltage, rotation, pressure, vibration) |
| **machines** | 100 | 3 | Machine metadata (machine_id, model, age) |
| **errors** | 5,386 | 3 | Non-failure error events (datetime, machine_id, error_id: error1-error5) |
| **maintenance** | 1,698 | 3 | Scheduled maintenance records (datetime, machine_id, comp: comp1-comp4) |
| **failures** | 47 | 3 | Actual failure events (datetime, machine_id, failure: comp1-comp4) |

## Target Variable
- **Name:** `label` (created during preprocessing)
- **Type:** Binary classification (0 = normal, 1 = will fail within 24h)
- **Positive rate:** 0.13% (1,175 out of 876,000)
- **Imbalance ratio:** 1:745

## Sensor Data Dictionary

| Sensor | Unit | Mean | Std | Range | Degradation Pattern |
|---|---|---|---|---|---|
| voltage | Volts | 170 | 15 | [100, 250] | Becomes erratic (±25V) before failure |
| rotation | RPM | 450 | 50 | [100, 800] | Drops (-80 RPM) as bearings wear |
| pressure | PSI | 100 | 12 | [40, 180] | Drops (-20 PSI) due to leaks |
| vibration | mm/s | 40 | 8 | [10, 100] | Increases (+20 mm/s) as components loosen |

## Data Realism Features
- Sensors show daily periodicity (factory temperature cycles)
- Older machines have noisier sensors
- Gradual degradation in 48h before failure (not sudden jumps)
- Error frequency increases in the week before failure
- Machine-specific offsets (manufacturing variance)

## Known Issues
- Sample dataset (`data/sample/`) has 0 failures (too few machines/days)
- Raw data files are gitignored (must regenerate with `python scripts/generate_data.py`)

---

# 9. Exploratory Data Analysis

## Key Findings

### 1. Class Imbalance (CRITICAL)
```
Total telemetry readings: 876,000
Total failure events:          47
Failure rate:              0.005%
Imbalance ratio:          1:18,638
```
**Impact:** Must use class weights, SMOTE, and F1/AUC-ROC (NOT accuracy).

### 2. Sensor Distributions
All 4 sensors are approximately symmetric (skewness < 0.5):
- Voltage: mean=170, std=15
- Rotation: mean=450, std=50
- Pressure: mean=100, std=12
- Vibration: mean=40, std=8

### 3. Sensor Correlations
```
             voltage  rotation  pressure  vibration
voltage       1.0000    0.0086    0.0405     0.0628
rotation      0.0086    1.0000    0.0153     0.0163
pressure      0.0405    0.0153    1.0000     0.0754
vibration     0.0628    0.0163    0.0754     1.0000
```
✅ **No highly correlated pairs** — all 4 sensors carry independent information.

### 4. Age vs Failure
Older machines fail significantly more:
- 0-5 years: avg 0.2 failures/year
- 16-20 years: avg 0.7 failures/year (3.5× more)

### 5. Failure Mode Distribution
Roughly uniform across 4 components: comp1 (23%), comp2 (28%), comp3 (21%), comp4 (28%).

### 6. Temporal Patterns
- Failures distributed roughly uniformly across months (no strong seasonality)
- Date range: 2024-01-01 to 2024-12-30 (364 days)

### 7. Missing Values
✅ No missing values in any table.

### 8. Duplicates
✅ No duplicate rows in any table.

---

# 10. Feature Engineering

## Pipeline: Raw → LSTM-Ready

```
876,000 rows × 6 cols (raw telemetry)
        ↓ Merge machines, errors, maintenance
876,000 rows × 17 cols
        ↓ Rolling stats + lag features + change rates
876,000 rows × 65 cols (+48 engineered)
        ↓ Binary label creation (24h horizon)
876,000 rows × 66 cols
        ↓ Temporal split (Oct 19, 2024)
Train: 700,800 | Test: 175,200
        ↓ StandardScaler (fit on train only)
        ↓ LSTM windowing (24-step sliding windows)
X_train: (698,400, 24, 63) | X_test: (172,800, 24, 63)
```

## Feature Categories (63 total)

| Category | Count | Features |
|---|---|---|
| Raw sensors | 4 | voltage, rotation, pressure, vibration |
| Rolling mean (3h/12h/24h) | 12 | {sensor}_rolling_mean_{3,12,24}h |
| Rolling std (3h/12h/24h) | 12 | {sensor}_rolling_std_{3,12,24}h |
| Lag values (1h/6h/24h) | 12 | {sensor}_lag_{1,6,24}h |
| Rate of change (1h/6h/24h) | 12 | {sensor}_change_{1,6,24}h |
| Machine metadata | 5 | age, model_model1, model_model2, model_model3, model_model4 |
| Error counts | 2 | error_count, errors_last_24h |
| Maintenance | 4 | hours_since_maint_comp{1,2,3,4} |

## Preprocessing Steps (in order)

| Step | What | Why |
|---|---|---|
| 1. Merge | Join telemetry + machines + errors + maintenance | Create single feature table |
| 2. One-hot encode | Machine model → dummy variables | Convert categorical to numeric |
| 3. Error aggregation | Count errors per machine per hour, rolling 24h sum | Quantify error frequency |
| 4. Maintenance features | Hours since last component replacement | Encode maintenance history |
| 5. Rolling statistics | Mean and std over 3h/12h/24h windows per machine | Capture trends and variability |
| 6. Lag features | Values at 1h/6h/24h ago | Explicit recent history |
| 7. Change features | Current - lag value | Rate of change |
| 8. Forward/back fill | Fill NaN from rolling/lag operations | Handle window edges |
| 9. Label creation | Mark 24h before each failure as label=1 | Supervised learning target |
| 10. Temporal split | Train: Jan-Oct, Test: Oct-Dec | Prevent data leakage |
| 11. StandardScaler | Normalize to mean=0, std=1 | Equal feature weighting |
| 12. Sliding windows | 24-timestep windows per machine | LSTM input format |

## Critical Design Decisions
- **Temporal split, not random:** Prevents data leakage (model can't see future)
- **Scaler fit on train only:** Test statistics never leak into training
- **Windows per machine:** Never mix data from different machines in same sequence
- **24h prediction horizon:** Enough lead time for maintenance scheduling
- **24-step sequence length:** 24 hours of history captures daily patterns

---

# 11. Machine Learning Pipeline

## Current Status: Ready for Model Building (Day 4)

### Selected Algorithm
**LSTM (Long Short-Term Memory)** — a recurrent neural network architecture specialized for sequential/time-series data.

### Why LSTM?
1. **Temporal dependencies:** Failures develop over hours/days — LSTM remembers long sequences
2. **Multivariate:** Handles multiple sensor inputs simultaneously
3. **Proven:** Used by GE Digital, Siemens, and NASA for predictive maintenance
4. **TensorFlow support:** Native Keras LSTM layer with GPU acceleration

### Alternatives Considered
| Algorithm | Why Not Primary |
|---|---|
| Random Forest / XGBoost | Can't model temporal sequences natively |
| 1D CNN | Good for local patterns, less effective for long-range dependencies |
| Transformer | Overkill for our dataset size, harder to interpret |
| ARIMA | Univariate only, can't handle multivariate sensor data |

### Planned Architecture (Day 4)
```
Input: (batch, 24, 63)
   ↓
LSTM(128, return_sequences=True)
   ↓
Dropout(0.3)
   ↓
LSTM(64)
   ↓
Dropout(0.3)
   ↓
Dense(32, relu)
   ↓
Dense(1, sigmoid)
   ↓
Output: failure probability [0, 1]
```

### Planned Hyperparameters
| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Adaptive learning rate, works well out-of-box |
| Loss | Binary crossentropy | Binary classification standard |
| Learning rate | 0.001 (with ReduceLROnPlateau) | Adaptive reduction on plateau |
| Batch size | 256-512 | Balance between speed and gradient stability |
| Epochs | 50 (with EarlyStopping) | Stop when validation loss plateaus |
| Class weights | Computed from training label distribution | Handle 1:745 imbalance |

### Planned Metrics
- **Primary:** AUC-ROC (handles class imbalance)
- **Secondary:** F1-score, Precision, Recall
- **NOT:** Accuracy (useless with 99.87% negative class)

### Planned Callbacks
1. **EarlyStopping:** Stop if val_loss doesn't improve for 5 epochs
2. **ModelCheckpoint:** Save best model based on val_auc
3. **ReduceLROnPlateau:** Halve learning rate if val_loss stalls for 3 epochs

### Model Saving Strategy
- Save Keras model: `models/lstm_predictive_maintenance.keras`
- Save scaler: `data/processed/scaler.joblib`
- Save feature columns: `data/processed/feature_columns.txt`
- Save training history: `models/training_history.json`

---

# 12. GenAI Module

## Current Status: Scaffolded (Day 7-8)

### Planned LangChain Architecture
```
Prediction Results → Prompt Template → LLM Chain → Structured Report
                                          ↑
                                    System Prompt
                                    (maintenance expert persona)
```

### Planned Components

| Component | Purpose |
|---|---|
| `src/genai/prompts.py` | Prompt templates for report generation and Q&A |
| `src/genai/chains.py` | LangChain chains: report_chain, qa_chain |
| `src/genai/assistant.py` | Maintenance Q&A assistant (conversational) |

### LLM Choice
Flexible — supports:
1. **OpenAI GPT-4o-mini** (default, via `OPENAI_API_KEY`)
2. **Google Gemini 1.5 Flash** (via `GOOGLE_API_KEY`)
3. **Ollama/Llama 3** (local, no API key needed)

### Planned Features
- **Report Generation:** Convert ML predictions into plain-English maintenance reports
- **Failure Explanation:** "Machine #47 is at risk because vibration increased 340% in 12h"
- **Maintenance Scheduling:** Suggest priority and timeline based on urgency
- **Q&A Assistant:** "What caused the last failure on Machine #12?"

---

# 13. Backend

## Current Status: Scaffolded (Day 9)

### Planned FastAPI Structure
```
src/api/
├── main.py           # FastAPI app instance, middleware, startup/shutdown
├── schemas.py        # Pydantic request/response models
└── routes/
    ├── health.py     # GET /health — system health check
    ├── predict.py    # POST /predict — submit sensor data, get predictions
    └── reports.py    # POST /report — generate AI maintenance report
```

### Planned Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | System health, model status |
| POST | `/predict` | Submit sensor data → failure prediction |
| POST | `/report` | Generate AI maintenance report |
| GET | `/machines` | List all machines |
| GET | `/machines/{id}/history` | Machine prediction history |

### Planned Features
- Pydantic request/response validation
- Structured error responses (using custom exceptions)
- CORS middleware
- Auto-generated OpenAPI/Swagger docs at `/docs`

---

# 14. Dashboard

## Current Status: Scaffolded (Day 10)

### Planned Pages

| Page | Purpose |
|---|---|
| **Overview** | All machines status, alerts, summary stats |
| **Machine Detail** | Sensor charts, prediction history, AI report |
| **Predictions** | Failure probability timeline, risk heatmap |
| **Reports** | Generate and view AI maintenance reports |

### Planned Components
- Real-time prediction cards with color-coded risk levels
- Sensor trend line charts (plotly/altair)
- Failure probability gauge
- Alert feed sorted by urgency
- One-click AI report generation
- Filterable machine table

---

# 15. Deployment

## Current Status: Not Started (Day 11)

### Planned Stack
- **Docker:** Multi-stage build for API + Dashboard
- **Docker Compose:** Orchestrate API + Dashboard + volume mounts
- **GitHub Actions:** CI pipeline (lint → test → build → push)
- **Environment Variables:** All secrets via `.env` / Docker env

### Planned CI/CD Pipeline
```
Push to main → GitHub Actions → Lint → Test → Build Docker → Push to Registry
```

---

# 16. System Architecture

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│                                                              │
│  Sensor Data (CSV/API)                                       │
│       ↓                                                      │
│  DataIngestion (src/data/ingestion.py)                       │
│       ↓                                                      │
│  DataValidator (src/data/validation.py)                      │
│       ↓                                                      │
│  DataPreprocessor (src/data/preprocessing.py)                │
│       │                                                      │
│       ├── Merge 5 tables                                     │
│       ├── Rolling statistics (3h/12h/24h)                    │
│       ├── Lag features (1h/6h/24h)                           │
│       ├── Error/maintenance aggregates                       │
│       ├── Label creation (24h horizon)                       │
│       ├── Temporal train/test split                          │
│       ├── StandardScaler normalization                       │
│       └── LSTM sliding window sequences                      │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                    ML LAYER (Day 4-6)                         │
│                                                              │
│  LSTM Model (src/models/lstm_model.py)                       │
│       ↓                                                      │
│  Trainer (src/models/trainer.py)                             │
│       ↓                                                      │
│  Evaluator (src/models/evaluator.py)                         │
│       ↓                                                      │
│  Predictor (src/prediction/predictor.py)                     │
│       ↓                                                      │
│  Prediction: {machine_id, failure_prob, failure_mode}        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                    GENAI LAYER (Day 7-8)                      │
│                                                              │
│  Prediction → Prompt Template → LLM (GPT-4o/Gemini/Llama)   │
│       ↓                                                      │
│  Maintenance Report (plain English)                          │
│  Q&A Assistant (conversational)                              │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                    PRESENTATION LAYER (Day 9-10)             │
│                                                              │
│  FastAPI REST API (src/api/)                                 │
│       ↓                                                      │
│  Streamlit Dashboard (dashboard/)                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Responsibility | Depends On |
|---|---|---|
| `config/settings.py` | Centralized configuration | `.env` |
| `src/utils/logger.py` | Logging | None |
| `src/utils/exceptions.py` | Error hierarchy | None |
| `src/data/ingestion.py` | Load raw data | config, logger, exceptions |
| `src/data/validation.py` | Validate data quality | logger |
| `src/data/preprocessing.py` | Feature engineering + LSTM prep | config, logger, exceptions, sklearn |
| `src/models/` | LSTM model definition + training | tensorflow, preprocessing |
| `src/prediction/` | Inference pipeline | models, preprocessing |
| `src/genai/` | LLM report generation | langchain, prediction |
| `src/api/` | REST API endpoints | fastapi, prediction, genai |
| `dashboard/` | Interactive UI | streamlit, api |

---

# 17. Configuration Files

| File | Purpose | Details |
|---|---|---|
| **`.env`** | Actual environment variables (GITIGNORED) | API keys, model selection, ports |
| **`.env.example`** | Template for `.env` | Shows required variables with placeholder values |
| **`config/settings.py`** | Pydantic settings class | Reads `.env`, validates types, cached singleton via `get_settings()` |
| **`pyproject.toml`** | Project metadata + tool configs | PEP 621 metadata, Black (line-length=88), pytest (pythonpath, asyncio), isort, mypy |
| **`.flake8`** | Flake8 linting config | max-line-length=88 (matches Black), ignore E203/W503 |
| **`requirements.txt`** | Production dependencies | 7 layers, version-pinned with ranges |
| **`requirements-dev.txt`** | Dev dependencies | Extends `-r requirements.txt` |
| **`.gitignore`** | Git ignore patterns | .env, data/raw/, data/processed/, models/*.h5, venv/, __pycache__, logs/ |
| **`Makefile`** | Developer shortcuts | `make test`, `make lint`, `make format`, `make run-api`, `make run-dashboard` |

---

# 18. Logging

## Strategy
All logging uses **Loguru** via `src/utils/logger.py`. Every module gets its own named logger via `get_logger(__name__)`.

## Configuration

| Setting | Value | Purpose |
|---|---|---|
| **Console output** | Colored, with timestamps | Developer experience |
| **File output** | `logs/app.log` | Persistent log storage |
| **Rotation** | 10 MB per file | Prevent disk exhaustion |
| **Retention** | 7 days | Auto-cleanup old logs |
| **Format** | `{time:HH:mm:ss} | {level} | {name}:{function}:{line} | {message}` | Traceable |
| **Levels** | DEBUG in dev, INFO in production | Configurable via `LOG_LEVEL` in `.env` |

## Usage Pattern
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Loading data...")
logger.warning(f"⚠ High null rate: {pct}%")
logger.error(f"Failed to load: {filepath}")
```

---

# 19. Testing

## Current Status: 68 Tests, All Passing

| Test File | Tests | Time | Coverage |
|---|---|---|---|
| `test_smoke.py` | 19 | 91s | Imports, config, logging, exceptions |
| `test_data_pipeline.py` | 22 | 0.7s | Ingestion + validation |
| `test_preprocessing.py` | 27 | 5.0s | Feature engineering + labels + split + sequences |
| **Total** | **68** | **~97s** | — |

## Test Categories

| Category | What's Tested |
|---|---|
| **Dependency imports** | numpy, pandas, sklearn, tensorflow, langchain, fastapi, pydantic, loguru |
| **Configuration** | Settings loading, defaults, paths, singleton caching |
| **Logger** | Logger creation, message writing |
| **Exceptions** | Hierarchy, details, status codes |
| **Data ingestion** | CSV loading, error handling, format detection, dataset loading |
| **Data validation** | Schema, nulls, duplicates, range checks, dataset validation |
| **Merge tables** | Column preservation, row count, metadata join |
| **Feature engineering** | Rolling stats, lag features, NaN handling |
| **Label creation** | Binary labels, temporal correctness, edge cases |
| **Temporal split** | Train < test, no overlap, no leakage |
| **Normalization** | Scaler fitting, zero-mean verification |
| **Sequence windowing** | 3D shape, label alignment, sample counts |
| **Full pipeline** | End-to-end, output shapes, metadata, no NaN |

## Pending Tests (future days)
- Model architecture tests
- Training pipeline tests
- Prediction pipeline tests
- GenAI chain tests
- API endpoint tests (using httpx TestClient)
- Integration tests (API → model → prediction → GenAI)

## Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/unit/test_preprocessing.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Makefile shortcut
make test
```

---

# 20. Coding Standards

## Python Style
- **Formatter:** Black (line length 88)
- **Linting:** Flake8 (compatible with Black)
- **Import sorting:** isort (Black-compatible profile)
- **Type checking:** mypy (strict mode ready)

## Principles
| Principle | How Applied |
|---|---|
| **Single Responsibility** | Each module has one job: ingestion loads, validation checks, preprocessing transforms |
| **Open/Closed** | New sensor types can be added to `SENSOR_CONFIG` without modifying generation logic |
| **Dependency Injection** | `DataIngestion(settings=mock_settings)` for testability |
| **Don't Repeat Yourself** | Sensor configs defined once in dicts, not repeated across functions |
| **Separation of Concerns** | Data layer, model layer, GenAI layer, API layer fully isolated |

## Conventions
| Convention | Rule |
|---|---|
| **File naming** | snake_case (e.g., `data_pipeline.py`) |
| **Class naming** | PascalCase (e.g., `DataPreprocessor`) |
| **Function naming** | snake_case (e.g., `create_labels`) |
| **Constants** | UPPER_SNAKE_CASE (e.g., `SENSOR_COLUMNS`) |
| **Type hints** | Required on all function signatures |
| **Docstrings** | Google-style, with Args/Returns/Raises |
| **Comments** | "WHY" comments explaining design decisions, not "WHAT" |
| **Logging** | Use `logger.info/warning/error`, never `print()` |
| **Exceptions** | Use custom exceptions from `src/utils/exceptions.py` |
| **Configuration** | Always use `get_settings()`, never hardcode values |
| **Imports** | Absolute imports (`from src.data.ingestion import...`) |

## Documentation Style
Every file has a module-level docstring with:
- **WHY THIS FILE EXISTS** — business/architectural reason
- **HOW IT WORKS** — technical overview
- **DESIGN PATTERN** — which pattern and why
- **USAGE** — example code

---

# 21. Important Design Decisions

| Decision | Rationale |
|---|---|
| **TensorFlow over PyTorch** | TF has better production deployment (TF Serving, TF Lite), more common in industry for production ML |
| **LangChain over raw API calls** | Provides chain composition, prompt templating, output parsing, memory management — production patterns |
| **FastAPI over Flask/Django** | Async-first, auto-OpenAPI docs, Pydantic integration, modern Python |
| **Streamlit over React** | 10x faster to build data dashboards, Python-only (no JS), sufficient for MVP |
| **Synthetic data over real download** | No Kaggle dependency, reproducible (seed=42), realistic patterns, no licensing issues |
| **Pydantic-settings over os.environ** | Type validation, .env support, defaults, clean API |
| **Loguru over stdlib logging** | Zero-config, colored output, rotation, better DX |
| **Temporal split over random** | Prevents data leakage in time-series — critical for honest evaluation |
| **StandardScaler over MinMaxScaler** | Better handling of outliers, doesn't bound range, preferred for LSTM |
| **24h prediction horizon** | Enough lead time for maintenance, specific enough for model precision |
| **24-step sequence length** | Captures full daily cycle (24 hours = 1 day pattern) |
| **Custom exception hierarchy** | Precise error handling by architectural layer |
| **Modular folder structure** | Each layer independently deployable and testable |
| **Python 3.12 over 3.14** | TensorFlow only supports ≤3.12 (discovered during Day 1 setup) |

---

# 22. Known Bugs

| ID | Severity | Description | Workaround | Status |
|---|---|---|---|---|
| 1 | Low | Sample dataset has 0 failures (30 days too short) | Use full dataset for model testing | Known limitation |
| 2 | Info | VS Code shows 10K pending changes from `~/.antigravity/` | These are IDE files, not project files — ignore or close that repo in VS Code | Not a bug |
| 3 | Info | TensorFlow first-import takes ~90s | Normal behavior on ARM64 Mac, subsequent imports are fast | Expected |

No blocking bugs exist.

---

# 23. TODO List

## High Priority (Day 4 — Next)
- [ ] Build LSTM model architecture (`src/models/lstm_model.py`)
- [ ] Implement training pipeline (`src/models/trainer.py`)
- [ ] Train model on preprocessed data
- [ ] Implement model evaluation (`src/models/evaluator.py`)
- [ ] Save trained model and training history

## Medium Priority (Days 5-8)
- [ ] Model optimization (hyperparameter tuning, class weights)
- [ ] Build prediction/inference pipeline (`src/prediction/predictor.py`)
- [ ] Implement LangChain report generation
- [ ] Build maintenance Q&A assistant
- [ ] Create prompt templates for different report types

## Low Priority (Days 9-12)
- [ ] Build FastAPI REST API with all endpoints
- [ ] Build Streamlit dashboard
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] Final documentation and demo recording

## Completed ✅
- [x] Project setup and environment (Day 1)
- [x] Synthetic data generation (Day 2)
- [x] Data ingestion pipeline (Day 2)
- [x] Data validation pipeline (Day 2)
- [x] EDA analysis (Day 2)
- [x] Feature engineering (Day 3)
- [x] Label creation (Day 3)
- [x] Temporal train/test split (Day 3)
- [x] Feature normalization (Day 3)
- [x] LSTM sequence windowing (Day 3)
- [x] 68 unit tests (Days 1-3)

---

# 24. Development Timeline

## Day 1 — 2026-08-19 (Project Setup)
| Field | Value |
|---|---|
| **Work Completed** | Environment setup, folder structure, config, logging, exceptions, smoke tests |
| **Problems** | System Python 3.14 incompatible with TensorFlow |
| **Solution** | Installed Python 3.12 via Homebrew |
| **Files Created** | 33 files, 1,511 lines |
| **Tests** | 19/19 passing |
| **Commits** | `475e722` (init), `52824f6` (README update) |
| **Lesson** | Always check framework compatibility before choosing Python version |

## Day 2 — 2026-08-20 (Dataset & Data Pipeline)
| Field | Value |
|---|---|
| **Work Completed** | Data generator, ingestion, validation, EDA |
| **Problems** | None |
| **Files Created** | 5 source + 5 CSVs |
| **Tests** | 22/22 new, 41/41 total |
| **Commit** | `755e95c` |
| **Lesson** | EDA before modeling reveals class imbalance (0.005% failure rate) |

## Day 3 — 2026-08-21 (Feature Engineering)
| Field | Value |
|---|---|
| **Work Completed** | Full preprocessing pipeline, 48 engineered features, LSTM sequences |
| **Problems** | None |
| **Files Created** | 3 source files |
| **Tests** | 27/27 new, 68/68 total |
| **Commit** | `020ad63` |
| **Lesson** | Temporal split is critical — random split = data leakage in time-series |

---

# 25. Session Log

## Session 1 (2026-08-19)
- Discussed project scope and architecture
- Decided on 12-day development plan
- Set up Python 3.12 environment (TF compatibility issue)
- Created all foundational infrastructure
- All smoke tests passing
- Pushed to GitHub

## Session 2 (2026-08-20)
- Built synthetic data generator (5 tables, 883K rows)
- Built DataIngestion with auto-format detection
- Built DataValidator with 4 check types
- Ran comprehensive EDA (discovered 1:18,638 imbalance)
- Created separate daily plan files (user requested)
- Pushed to GitHub

## Session 3 (2026-08-21)
- Built DataPreprocessor (most complex module so far)
- Implemented rolling stats, lag features, merge, labels
- Created temporal split and LSTM windowing
- Ran full pipeline: 876K rows → (698K, 24, 63) training sequences
- All 27 preprocessing tests passing
- Pushed to GitHub

---

# 26. Commands

## Development
```bash
source venv/bin/activate          # Activate virtual environment
make test                          # Run all tests
make lint                          # Run flake8 linting
make format                        # Run Black + isort formatting
python -m pytest tests/ -v         # Verbose test output
python -m pytest tests/ --cov=src  # Test with coverage report
```

## Data Pipeline
```bash
python scripts/generate_data.py              # Generate full dataset (100 machines, 365 days)
python scripts/generate_data.py --sample     # Generate sample dataset (10 machines, 30 days)
python scripts/generate_data.py --machines 50 --days 180  # Custom size
python scripts/eda_analysis.py               # Run EDA analysis
python scripts/eda_analysis.py --data-dir data/sample     # EDA on sample
python scripts/run_preprocessing.py          # Run full preprocessing pipeline
python scripts/run_preprocessing.py --horizon 48 --seq-len 48  # Custom params
```

## Git
```bash
git status                         # Check working tree status
git log --oneline -10              # Recent commits
git add . && git commit -m "..."   # Stage and commit
git push                           # Push to GitHub
```

## Future Commands (Days 4+)
```bash
# Model Training (Day 4)
python scripts/train_model.py

# API (Day 9)
make run-api
uvicorn src.api.main:app --reload --port 8000

# Dashboard (Day 10)
make run-dashboard
streamlit run dashboard/app.py

# Docker (Day 11)
docker build -t pred-maintenance .
docker-compose up
```

---

# 27. Important Files

| File | Purpose | Key Details |
|---|---|---|
| `config/settings.py` | All configuration | Reads `.env`, typed, cached singleton |
| `src/utils/logger.py` | Logging system | Loguru: colored console + rotated files |
| `src/utils/exceptions.py` | Error hierarchy | 10 exception classes across 4 layers |
| `src/data/ingestion.py` | Load data | Auto-format detection (CSV/Parquet/JSON/Excel) |
| `src/data/validation.py` | Validate data | Schema, nulls, duplicates, ranges; produces ValidationReport |
| `src/data/preprocessing.py` | Feature engineering | Rolling stats, lag, labels, split, scale, LSTM windowing |
| `scripts/generate_data.py` | Create dataset | 5 tables, configurable, seed=42, realistic degradation |
| `scripts/eda_analysis.py` | Analyze data | 8-dimension analysis, class imbalance detection |
| `scripts/run_preprocessing.py` | Run pipeline | End-to-end: load → validate → preprocess → save |
| `tests/unit/test_smoke.py` | Smoke tests | 19 tests: all imports + config + logging |
| `tests/unit/test_data_pipeline.py` | Pipeline tests | 22 tests: ingestion + validation |
| `tests/unit/test_preprocessing.py` | Preprocessing tests | 27 tests: features + labels + split + windowing |

---

# 28. Current Problems

## Current Blockers
None. The project is in a clean state, ready for Day 4.

## Current Debugging Tasks
None.

## Current Research Questions
1. Optimal LSTM architecture for this dataset size (698K samples, 63 features)
2. Best class weight strategy for 1:745 imbalance
3. Whether to use single LSTM or bidirectional LSTM

## Unknowns
1. Which LLM provider the user will choose (OpenAI/Gemini/Ollama) — affects Day 7-8
2. Target deployment platform (local Docker vs cloud) — affects Day 11

---

# 29. Next Immediate Tasks

The next AI model should do the following, in order:

1. **Read this handoff document** to understand the complete project context.
2. **Verify the environment** is working: `source venv/bin/activate && python -m pytest tests/ -v` (expect 68 tests passing).
3. **Create Day 4 implementation plan** (`implementation_plan_day4.md`) covering LSTM model architecture and training.
4. **Build `src/models/lstm_model.py`** — Define the LSTM architecture using TensorFlow/Keras.
5. **Build `src/models/trainer.py`** — Training pipeline with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau), class weights, and logging.
6. **Build `src/models/evaluator.py`** — Evaluation with AUC-ROC, F1, precision, recall, confusion matrix.
7. **Create `scripts/train_model.py`** — Runner script that loads processed data and trains the model.
8. **Write tests** in `tests/unit/test_model.py`.
9. **Train the model** on the preprocessed data in `data/processed/`.
10. **Save the model** to `models/`.
11. **Commit and push** with a conventional commit message.
12. **Update this handoff document** with Day 4 results.

---

# 30. Future Improvements

| Enhancement | Complexity | Value | When |
|---|---|---|---|
| **MLflow** experiment tracking | Medium | Track model versions, hyperparams, metrics | Day 5+ |
| **SHAP / Explainable AI** | Medium | Feature importance visualization | Day 6+ |
| **Redis caching** | Low | Cache predictions for frequently queried machines | Day 9+ |
| **Kafka streaming** | High | Real-time sensor data ingestion | Post-MVP |
| **Kubernetes** | High | Container orchestration for scaling | Post-MVP |
| **Prometheus + Grafana** | Medium | System monitoring, model performance dashboards | Post-MVP |
| **Authentication (JWT)** | Medium | Secure API endpoints | Day 9 |
| **Rate limiting** | Low | Prevent API abuse | Day 9 |
| **Feature Store (Feast)** | High | Reusable, versioned features | Post-MVP |
| **Model A/B testing** | Medium | Compare model versions in production | Post-MVP |
| **Alerting (PagerDuty/Slack)** | Low | Send alerts when failure predicted | Post-MVP |
| **Multi-site support** | Medium | Handle multiple factory locations | Post-MVP |
| **Edge deployment (TF Lite)** | High | Run model on IoT devices at machine level | Post-MVP |

---

# 31. Resume Perspective

| Feature | Why It Impresses Recruiters |
|---|---|
| **Full ML pipeline** | "Most candidates just call model.fit(). This person built ingestion → validation → feature engineering → training → inference." |
| **Feature engineering** | "They engineered 48 features from 4 raw sensors — rolling stats, lag features, aggregates. That's senior-level data science." |
| **Temporal split** | "They know about data leakage in time-series. Most candidates randomly split and get inflated metrics." |
| **GenAI integration** | "ML + GenAI in one project shows they understand the modern AI stack." |
| **Production patterns** | "Pydantic settings, loguru logging, custom exceptions, data validation — this person writes production code." |
| **Testing** | "68 unit tests covering every pipeline step. This is rare in ML projects." |
| **Docker + CI/CD** | "They can deploy, not just prototype. That's a full-stack ML engineer." |
| **Clean architecture** | "Modular design, SOLID principles, type hints, docstrings — this is maintainable code." |

### Resume Bullet Point
> *Designed and implemented an end-to-end Predictive Maintenance platform using TensorFlow LSTM and LangChain, predicting equipment failures from 876K sensor telemetry readings with a 24-hour horizon. Built a complete data pipeline (ingestion → validation → 48 engineered features → temporal split), REST API (FastAPI), and AI-powered maintenance report generator. Achieved production-ready quality with 68+ unit tests, Docker deployment, and CI/CD.*

---

# 32. Interview Preparation

## Core Topics to Prepare

| Topic | Key Questions |
|---|---|
| **Time-Series ML** | LSTM vs GRU vs Transformer? Rolling statistics? Temporal split vs random? |
| **Feature Engineering** | Why rolling windows? What lag features? How to handle NaN? |
| **Class Imbalance** | SMOTE? Class weights? F1 vs accuracy? Precision-recall tradeoff? |
| **Data Leakage** | What is it? How temporal split prevents it? Scaler fit on train only? |
| **TensorFlow** | Keras Sequential API? Callbacks? Custom training loops? Model saving? |
| **LangChain** | Chains? Prompts? Output parsers? Memory? RAG? |
| **FastAPI** | Async? Pydantic schemas? Middleware? Dependency injection? |
| **System Design** | Design a predictive maintenance system for 10K sensors? |
| **Docker** | Multi-stage builds? Docker Compose? Volume mounts? Networking? |
| **CI/CD** | GitHub Actions? Testing pipeline? Deployment strategy? |
| **Production ML** | Model monitoring? Data drift? A/B testing? Feature stores? |
| **Python** | Virtual environments? Type hints? Decorators? Context managers? |

---

# 33. Project Memory

## Overall Vision
An end-to-end Predictive Maintenance + GenAI platform that is production-ready, resume-worthy, and demonstrates mastery of ML, GenAI, API development, and software engineering.

## Long-term Goals
1. Complete all 12 days of development
2. Deploy to a cloud platform
3. Record a demo video
4. Use as primary portfolio project for ML/AI engineering interviews

## Completed Milestones
- ✅ Day 1: Foundation (environment, config, logging, exceptions)
- ✅ Day 2: Data (generation, ingestion, validation, EDA)
- ✅ Day 3: Features (engineering, labels, split, normalization, LSTM windowing)

## Pending Milestones
- 🔜 Day 4: LSTM model architecture and training
- 🔒 Days 5-12: Evaluation, prediction, GenAI, API, dashboard, deployment

## Lessons Learned
1. TensorFlow requires Python ≤3.12 (not 3.13+)
2. Class imbalance in predictive maintenance is extreme (1:18,638)
3. Temporal split is critical for time-series — random split = data leakage
4. Feature engineering has more impact than model architecture choice
5. Every module needs its own logger and exception handling

## Architectural Constraints
- **Python ≤3.12** — TensorFlow compatibility
- **Modular structure** — each layer independently testable and deployable
- **Config via .env** — 12-Factor App methodology
- **No print()** — all logging via loguru
- **Temporal split only** — no random split for time-series
- **Scaler fit on train** — never fit on test data

## Important Assumptions
- Equipment has 4 sensor types (voltage, rotation, pressure, vibration)
- Hourly sensor readings (not real-time streaming)
- 24-hour prediction horizon is sufficient
- Binary classification (fail/not fail), not multi-class
- LLM API key will be provided for GenAI module

## Coding Philosophy
- **Teach first, implement second** — every file has WHY/HOW docstrings
- **Test early** — write tests on the same day as implementation
- **Production patterns from Day 1** — not "we'll add that later"
- **Explain decisions** — comments explain WHY, not WHAT

## Things That Must Never Change
1. The modular folder structure (`src/data/`, `src/models/`, `src/genai/`, `src/api/`)
2. The config pattern (`get_settings()` from `config/settings.py`)
3. The logging pattern (`get_logger(__name__)` from `src/utils/logger.py`)
4. The exception hierarchy (all inherit from `PredMaintenanceError`)
5. Temporal train/test split (NEVER use random split)
6. Scaler fit on training data only

## Future Roadmap
```
Day 4  → LSTM Model + Training
Day 5  → Model Evaluation + Optimization
Day 6  → Prediction/Inference Pipeline
Day 7  → LangChain + Report Generation
Day 8  → GenAI Assistant + Q&A
Day 9  → FastAPI REST API
Day 10 → Streamlit Dashboard
Day 11 → Docker + CI/CD
Day 12 → Final Polish + Demo
```

---

# 34. Rules For Future AI Models

> **READ THESE RULES BEFORE MAKING ANY CHANGES.**

1. **Continue from the current state.** Do NOT restart the project. All foundation work is complete and tested.

2. **Never redesign the project** without a strong technical reason and user approval. The architecture was designed intentionally.

3. **Never remove completed functionality.** All existing code is tested and working. Build on top of it.

4. **Preserve the project architecture.** The modular structure (`src/data/`, `src/models/`, `src/genai/`, `src/api/`, `src/utils/`) must remain.

5. **Follow SOLID principles.** Single Responsibility, Open/Closed, Dependency Injection.

6. **Follow production-level coding standards.** Type hints, docstrings (WHY/HOW), loguru logging, custom exceptions.

7. **Explain concepts before implementation.** The user is learning. Teach first, code second.

8. **Always document design decisions.** Use "WHY" comments, not "WHAT" comments.

9. **Always explain why files are created or modified.** The user should understand the reasoning.

10. **Keep code modular, scalable, and maintainable.** No monolithic files, no god classes.

11. **Prefer industry best practices over shortcuts.** This is a resume project — quality matters.

12. **Update this handoff document** at the end of every development session so it always reflects the latest project state.

13. **Use the established patterns:**
    - Config: `from config.settings import get_settings`
    - Logging: `from src.utils.logger import get_logger; logger = get_logger(__name__)`
    - Exceptions: `from src.utils.exceptions import DataIngestionError`
    - Tests: pytest with fixtures, one test file per module

14. **Never use random train/test split** for time-series data. Always use temporal split.

15. **Never fit scaler/encoder on test data.** Always fit on training data, transform both.

16. **Create separate implementation plans** for each day (e.g., `implementation_plan_day4.md`).

17. **Run all tests** after making changes. Current baseline: **68 tests, all passing.**

18. **Commit with conventional commit messages.** Format: `feat(scope): description`

19. **Push to GitHub** at the end of each session.

20. **Never hardcode values.** Use `config/settings.py` for all configurable parameters.

---

*End of Handoff Document*
*Generated: 2026-08-22 | Milestone: Day 3/12 | Tests: 68/68 Passing*
