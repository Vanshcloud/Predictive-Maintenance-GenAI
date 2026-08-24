# 🔧 Predictive Maintenance + GenAI Insight Generator

> An end-to-end Predictive Maintenance platform that uses **TensorFlow** to predict equipment failures from sensor telemetry data and **LangChain** with LLMs to generate human-readable maintenance reports and diagnostic insights.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?logo=chainlink&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/tests-170%20passing-brightgreen)
![Model F1](https://img.shields.io/badge/model%20F1-0.8949-success)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)
- [Development Progress](#development-progress)
- [License](#license)

---

## Overview

Industrial equipment failures cost the global manufacturing industry **$50 billion per year** in unplanned downtime. This project implements a **Predictive Maintenance** system that:

1. **Predicts failures** — Uses an LSTM (Long Short-Term Memory) neural network trained on sensor telemetry data to predict when equipment will fail.
2. **Explains failures** — Uses LangChain + LLM to convert ML predictions into plain-English maintenance reports.
3. **Enables proactive maintenance** — Provides a REST API and interactive dashboard for maintenance teams.

### Why This Project?

| Traditional Approach | This Project |
|---|---|
| Fix equipment **after** it breaks | Predict failures **before** they happen |
| ML model outputs a number (0.87) | GenAI explains: *"Bearing temperature rising 3°C/hr, recommend immediate inspection"* |
| Requires ML expertise to interpret | Maintenance managers can read plain-English reports |
| Isolated scripts | Production-ready API + Dashboard |

---

## Architecture

```
Sensor Data → Data Pipeline → Feature Engineering → LSTM Model (TensorFlow)
                                                         ↓
                                                    Predictions
                                                         ↓
                                                LangChain + LLM
                                                         ↓
                                              Maintenance Reports
                                                         ↓
                                              FastAPI + Streamlit
```

See [docs/architecture.md](docs/architecture.md) for the detailed architecture diagram.

---

## Features

- 🤖 **LSTM Predictive Model** — TensorFlow-based time-series model for failure prediction
- 🧠 **GenAI Insights** — LangChain-powered natural language maintenance reports
- 🔌 **REST API** — FastAPI with auto-generated OpenAPI documentation
- 📊 **Interactive Dashboard** — Real-time equipment health monitoring (Streamlit)
- 🐳 **Dockerized** — One-command deployment with Docker
- ✅ **Tested** — Unit + integration tests with pytest
- 📝 **Well-Documented** — Comprehensive code documentation and architecture docs

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | TensorFlow / Keras |
| GenAI | LangChain + OpenAI / Ollama |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Configuration | Pydantic Settings |
| Logging | Loguru |
| Testing | Pytest |
| Code Quality | Black, Flake8, MyPy |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## Getting Started

### Prerequisites

- **Python 3.12** — required. TensorFlow publishes no wheels for 3.13+, so a newer
  system Python (e.g. 3.14) **will fail** at install time. On macOS:
  `brew install python@3.12`
- Git
- ~6 GB free disk space (the generated dataset and its processed tensors)
- (Optional) An OpenAI / Google API key for the GenAI features — a local Ollama model
  works with no key at all
- (Optional) Docker for containerized deployment

### Quick Setup

```bash
# 1. Clone
git clone https://github.com/Vanshcloud/vigilant-lamp.git
cd vigilant-lamp

# 2a. Automated setup (finds a compatible Python for you)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2b. …or manually. Note python3.12 explicitly — plain `python3` may be
#     a version TensorFlow does not support.
python3.12 -m venv venv          # macOS: /opt/homebrew/bin/python3.12
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env             # then add an LLM API key if you want reports
```

### Generate the data

The raw and processed datasets are **not** in this repository — together they are
about 6 GB. They are fully reproducible from a fixed seed:

```bash
source venv/bin/activate
python scripts/generate_data.py        # -> data/raw/  (883,231 rows, seed=42)
python scripts/run_preprocessing.py    # -> data/processed/*.npy (~5.2 GB)
```

`data/sample/` is committed and is what the test suite runs against, so `make test`
works immediately after install — but note it deliberately contains **zero** failure
events, so it cannot be used to train or evaluate a model.

### Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Run smoke tests
make test

# See all available commands
make help
```

---

## Usage

```bash
source venv/bin/activate

# Data
python scripts/generate_data.py               # full dataset (100 machines x 365 days)
python scripts/generate_data.py --sample      # small fixture -> data/sample/
python scripts/eda_analysis.py                # 8-dimension exploratory report
python scripts/run_preprocessing.py           # raw CSVs -> LSTM tensors + scaler

# Model
python scripts/train_model.py                 # train, evaluate, write metrics
python scripts/train_model.py --epochs 50 --monitor val_f1 --resume
python scripts/evaluate_model.py              # threshold sweep + curves

# Predict
python scripts/predict.py                     # current fleet status, most urgent first
python scripts/predict.py --machine 47        # one machine, as JSON
python scripts/predict.py --alerts-only -o alerts.csv

# AI maintenance reports
python scripts/generate_report.py --machine 51 --dry-run    # prompt only — no API key needed
python scripts/generate_report.py --machine 51              # full report
python scripts/generate_report.py --fleet

# Ask follow-up questions about one machine
python scripts/ask.py --machine 51
python scripts/ask.py --machine 51 --ask "Has vibration been rising?"

# Quality
make test                                     # 75 tests
make quality                                  # lint + format-check + typecheck
```

---

## Results

Trained on a three-way chronological split — 567,000 train / 129,000 validation /
172,800 test sequences of shape `(24, 63)`. Early stopping at epoch 20 of 30, best
weights from epoch 15. Wall clock ~47 minutes, CPU-only (Apple Silicon).

The alert threshold (0.6678) was chosen by sweeping the precision-recall curve on the
**validation** split; the test set is scored once, at that threshold.

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9997** |
| **Precision** | **0.8756** |
| **Recall** | 0.9150 |
| **F1** | **0.8949** |
| Single-sequence inference | 54 ms median |

Confusion matrix over 172,800 held-out sequences:

```
                 predicted 0   predicted 1
actual 0            172,574            26     <- false alarms
actual 1                 17           183     <- caught 183 of 200 hourly labels
```

### Failures caught, measured over events

Every hour in the 24 hours before a failure carries a positive label, so the
hourly figures above count *hours*, not failures. Measured over failure events —
where catching any hour means the technician was warned:

| Metric | Value |
|---|---|
| Failure events in the test period | 8 |
| **Events warned about** | **8 (100%)** |
| Lead time (median / min) | **24h / 15h** |

Eight events is a small sample: this says the model warned in 8 of 8 cases, not
that it never misses. Reported alongside precision, never instead of it —
event recall says nothing about alert fatigue.

**Accuracy is deliberately not reported.** With a 1:864 positive rate a model that
predicts "no failure" every time scores 99.88%, so accuracy would be actively
misleading. Quality is judged on AUC, precision, recall, and F1 only.

### From prediction to work order

Reports are grounded in the engineered features the model actually consumed —
never in the model's imagination. Each sensor line carries an explicit verdict,
and causal explanations are attached *only* when a reading deviates in the
direction that matters:

```
pressure: 65.89 PSI (24h baseline 93.47 PSI, 1.91 sigma below; -32.06 PSI over 24h)
    -> ABNORMAL; typically indicates a leak or a failing seal
rotation: 400.04 RPM (24h baseline 418.31 RPM, 0.33 sigma below)
    -> within normal variation; no action indicated by this sensor
```

Run it with no API key at all: `--dry-run` prints the grounded facts, and a
local Ollama model generates the full report keyless.

> **One caveat, stated plainly.** The dataset is synthetic, with a degradation pattern
> deliberately designed to be detectable. These metrics reflect *this dataset's*
> difficulty, not that of real industrial equipment. The pipeline transfers; the numbers
> would not. Full accounting in [`docs/Day5.md`](docs/Day5.md).

---

## Project Structure

```
vigilant-lamp/
├── IMPLEMENTATION_PLAN.md   # Single source of truth: scope, architecture, risks, status
├── CLAUDE.md                # Working notes for AI agents in this repo
├── LICENSE                  # MIT
├── Makefile                 # make test / lint / format / typecheck / quality
├── pyproject.toml           # PEP 621 metadata + Black/isort/pytest/mypy/coverage config
│
├── config/                  # Centralized configuration (pydantic-settings)
├── src/
│   ├── utils/               # Logging, exception hierarchy
│   ├── data/                # Ingestion, validation, preprocessing + feature engineering
│   ├── models/              # LSTM architecture, training loop, evaluator
│   ├── prediction/          # Predictor: raw tables -> ranked predictions
│   ├── genai/               # LangChain prompts + report chains
│   └── api/                 # FastAPI REST API + routes     (Day 9)
├── dashboard/               # Streamlit dashboard           (Day 10)
├── docker/                  # Dockerfiles + compose         (Day 11)
│
├── scripts/                 # Entry points: generate_data, eda, preprocessing,
│                            #   train_model, evaluate_model, predict
├── tests/
│   ├── conftest.py          # Session bootstrap (import order matters — see the file)
│   ├── unit/                # 113 tests, ~18s
│   └── integration/         # 4 tests — training/serving parity; `make test-integration`
├── docs/
│   ├── README.md            # Documentation index
│   ├── architecture.md      # System architecture diagram
│   ├── Day1.md … Day4.md    # One report per implementation day
│   └── handoff.md           # Frozen Day 1-3 narrative log (superseded)
│
├── data/
│   ├── raw/                 # gitignored — regenerate with generate_data.py
│   ├── processed/           # gitignored — regenerate with run_preprocessing.py
│   └── sample/              # committed test fixture (no failure events — by design)
├── models/                  # .keras gitignored; metrics.json + history committed
├── notebooks/               # Jupyter scratch space
└── logs/                    # gitignored, rotated daily
```

The `src/` packages form a strict dependency chain — each may only import from those to
its left, which is what keeps the layers independently testable:

```
config/ -> src/utils/ -> src/data/ -> src/models/ -> src/prediction/ -> src/genai/ -> src/api/ -> dashboard/
```

---

## Documentation

| Document | What it is |
|---|---|
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | **Start here.** Single source of truth — objectives, requirements, stack, architecture, dataset, model, training, evaluation, deployment, coding standards, testing strategy, risk register, milestones, and current status |
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/architecture.md`](docs/architecture.md) | System architecture diagram and layer responsibilities |
| [`docs/Day1.md`](docs/Day1.md) … [`docs/Day4.md`](docs/Day4.md) | One report per implementation day: plan, work done, code changes, training results, bugs, design decisions, next steps |
| [`docs/handoff.md`](docs/handoff.md) | Historical narrative log, frozen at end of Day 3 |
| [`CLAUDE.md`](CLAUDE.md) | Repo conventions and non-negotiable invariants, for AI agents |

---

## Testing

```bash
make test          # 75 tests
make test-cov      # with coverage report
make lint          # flake8
make format        # black + isort (writes)
make quality       # lint + format-check + typecheck (no writes)
```

**Current status: 161 unit + 9 integration tests passing, 0 flake8 issues, Black and isort clean.**

Tests run against the committed `data/sample/` fixture, so they need no generated data.
The first run pays roughly 90 seconds for TensorFlow's initial import on ARM64 macOS;
subsequent runs finish in about 4 seconds.

> `tests/conftest.py` imports TensorFlow before anything else, and that is
> **load-bearing** — not a stray import. TensorFlow and Apache Arrow (pulled in by
> pandas/scikit-learn) each bundle their own copy of abseil, and whichever loads first
> claims the shared symbol for the whole process. Get the order wrong and the suite
> deadlocks at 0% CPU with no traceback. The file explains it in full.

---

## Development Progress

**~67% complete (8 of 12 days).**

- [x] **Day 1** — Project setup, folder structure, configuration, logging, testing infrastructure
- [x] **Day 2** — Synthetic dataset (883K rows), exploratory data analysis, ingestion + validation
- [x] **Day 3** — Feature engineering (63 features), labeling, temporal split, LSTM sequencing
- [x] **Day 4** — LSTM architecture, training pipeline, evaluation
- [x] **Day 5** — 3-way split, threshold sweep, training curves, resume — **F1 0.8949**
- [x] **Day 6** — Prediction pipeline — training/serving parity verified, **8/8 failure events caught**
- [x] **Day 7** — LangChain reports — grounded in real sensor evidence, runs keyless on local Ollama
- [x] **Day 8** — Conversational assistant — multi-turn Q&A that declines what the data cannot answer
- [ ] **Day 9** — FastAPI REST API
- [ ] **Day 10** — Streamlit dashboard
- [ ] **Day 11** — Docker, CI/CD, deployment
- [ ] **Day 12** — Final polish, documentation, demo

---

## License

Released under the MIT License — see [LICENSE](LICENSE).

---

## Author

**Vansh Tomar** — [GitHub](https://github.com/Vanshcloud)

---

*Built as a production-ready portfolio project demonstrating Machine Learning, Generative AI, and Software Engineering best practices.*
