# 🔧 Predictive Maintenance + GenAI Insight Generator

> An end-to-end Predictive Maintenance platform that uses **TensorFlow** to predict equipment failures from sensor telemetry data and **LangChain** with LLMs to generate human-readable maintenance reports and diagnostic insights.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?logo=chainlink&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
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

- Python 3.10+
- Git
- (Optional) OpenAI API key for GenAI features
- (Optional) Docker for containerized deployment

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Vanshcloud/vigilant-lamp.git
cd predictive-maintenance-genai

# Run the automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or set up manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env with your API keys
```

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

## Project Structure

```
predictive-maintenance-genai/
├── config/             # Centralized configuration (pydantic-settings)
├── src/
│   ├── data/           # Data ingestion, validation, preprocessing
│   ├── models/         # LSTM model definition, training, evaluation
│   ├── prediction/     # Inference pipeline
│   ├── genai/          # LangChain chains, prompts, assistant
│   ├── api/            # FastAPI REST API + routes
│   └── utils/          # Logging, exceptions, shared utilities
├── dashboard/          # Streamlit interactive dashboard
├── data/               # Raw + processed data (gitignored)
├── models/             # Saved model artifacts (gitignored)
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit + integration tests
├── docs/               # Architecture docs + diagrams
├── docker/             # Docker configuration
└── scripts/            # Utility scripts
```

---

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linter
make lint

# Format code
make format
```

---

## Development Progress

- [x] **Day 1** — Project setup, folder structure, configuration, logging, testing infrastructure
- [ ] **Day 2** — Dataset acquisition, exploratory data analysis, data pipeline
- [ ] **Day 3** — Feature engineering, data preprocessing
- [ ] **Day 4** — LSTM model architecture, training pipeline
- [ ] **Day 5** — Model evaluation, metrics, optimization
- [ ] **Day 6** — Prediction pipeline, inference engine
- [ ] **Day 7** — LangChain setup, prompt engineering, report generation
- [ ] **Day 8** — GenAI assistant, maintenance Q&A
- [ ] **Day 9** — FastAPI REST API
- [ ] **Day 10** — Streamlit dashboard
- [ ] **Day 11** — Docker, CI/CD, deployment
- [ ] **Day 12** — Final polish, documentation, demo

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Vansh Tomar** — [GitHub](https://github.com/Vanshcloud)

---

*Built as a production-ready portfolio project demonstrating Machine Learning, Generative AI, and Software Engineering best practices.*
