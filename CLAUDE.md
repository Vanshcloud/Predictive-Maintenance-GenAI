# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An end-to-end Predictive Maintenance platform: a TensorFlow LSTM predicts equipment failures 24h in advance from sensor telemetry, and LangChain + an LLM turns those predictions into plain-English maintenance reports, exposed via FastAPI and a Streamlit dashboard. It's being built incrementally as a 12-day/12-milestone portfolio project — read **`IMPLEMENTATION_PLAN.md`** first — it is the single source of truth for scope, architecture, risks, milestones, and current status — then the latest `docs/DayX.md` for what the last session actually did. Both must be updated before a session ends; docs and code ship in the same commit.

Progress: data pipeline and feature engineering are complete (Days 1-3); model training (Day 4) is complete — the LSTM trains end-to-end on the full dataset and writes `models/metrics.json` + `models/training_history.json`. Prediction pipeline, GenAI, API, and dashboard layers are scaffolded but not yet implemented (`src/prediction/`, `src/genai/`, `src/api/`, `dashboard/`).

## Commands

```bash
source venv/bin/activate           # Python 3.12 venv — required, TensorFlow does not support 3.13+

make test                          # python -m pytest tests/ -v --tb=short
python -m pytest tests/unit/test_preprocessing.py -v   # single test file
python -m pytest tests/ --cov=src --cov=config --cov-report=term-missing  # make test-cov

make lint                          # flake8 src/ config/ tests/
make format                        # black src/ config/ tests/ && isort src/ config/ tests/
make typecheck                     # mypy src/ config/
make quality                       # lint + format-check + typecheck

python scripts/generate_data.py              # regenerate data/raw/ (883K rows, seed=42; gitignored)
python scripts/generate_data.py --sample     # small dataset -> data/sample/ (committed, has 0 failures)
python scripts/eda_analysis.py               # 8-dimension EDA report
python scripts/run_preprocessing.py          # raw CSVs -> data/processed/*.npy + scaler.joblib
python scripts/train_model.py                # train LSTM on data/processed/, evaluate, save metrics.json
```

Note: `make test` runs the full suite including TensorFlow import, which can take ~90s on first run (ARM64 Mac). Data files under `data/raw/` and `data/processed/` are gitignored — regenerate via the scripts above if missing; `data/sample/` is the only committed dataset and is intentionally too small to contain any failure events.

## Architecture

Layered pipeline, each layer only depending on layers before it — never skip a layer or reach into a later one's internals:

```
config/  ->  src/utils/  ->  src/data/  ->  src/models/  ->  src/prediction/  ->  src/genai/  ->  src/api/  ->  dashboard/
```

- **`config/settings.py`** — single `Settings` (pydantic-settings) class reading `.env`, accessed everywhere via the cached `get_settings()` factory. Never hardcode a path/port/model name — add it here instead.
- **`src/utils/logger.py`** / **`exceptions.py`** — every module gets `get_logger(__name__)` (loguru; no bare `print()`). All custom exceptions inherit from `PredMaintenanceError`, grouped by layer (Data*, Model*, LLM*/Report*, API*) so callers can catch precisely or broadly.
- **`src/data/`** — `DataIngestion` (format-detecting loader) -> `DataValidator` (schema/null/duplicate/range checks, produces a `ValidationReport`) -> `DataPreprocessor` (`preprocessing.py`, the largest module at ~780 lines): merges the 5 raw tables, engineers 48 rolling/lag/change features per sensor, builds 24h-horizon binary labels, performs a **temporal** (never random) train/test split, fits `StandardScaler` on train only, and slides 24-step windows into `(N, 24, 63)` LSTM tensors. This ordering (split before scaling before windowing) is deliberate — reversing it leaks test-set information into training.
- **`src/models/`** — `PredictiveMaintenanceModel` (`lstm_model.py`, defines/saves/loads the Keras architecture) is consumed by both `ModelTrainer` (`trainer.py`) and `ModelEvaluator` (`evaluator.py`: AUC/precision/recall/F1/confusion matrix — never plain accuracy, since positives are ~0.13% of rows).
  - **`src/models/__init__.py`'s import order is load-bearing — do not alphabetise it, and do not remove its `# isort: skip_file`.** TensorFlow and Apache Arrow (loaded by pandas/sklearn) each statically link their own abseil; whichever loads first wins the `AbslInternalPerThreadSemWait` symbol process-wide. If Arrow wins, TF's first `tf.function` execution deadlocks at 0% CPU with no traceback, no timeout, and no error. `evaluator` imports sklearn and `lstm_model` imports TF, so TF must be imported first. `tests/conftest.py` does the same for the test suite. Any new entry point that uses TensorFlow must import `src.models` (or tensorflow) before `src.data`. See `docs/Day4.md`.
  - `ModelTrainer.train()` is a hand-written `GradientTape` loop over `iter_batches()` rather than `model.fit()`, with early stopping, LR reduction, and checkpointing implemented inline (Keras callbacks only run inside `fit()`). `ModelEvaluator` likewise calls `model(x, training=False)` rather than `model.predict()`. Both were written while the deadlock was misattributed to Keras's background prefetch threads; the real cause was the abseil collision above. They are kept because they work, are tested, and keep class weighting and callbacks explicit — but `fit()` has not been re-benchmarked since the real fix, so don't claim it is broken.
- **`src/prediction/`, `src/genai/`, `src/api/`, `dashboard/`** — currently empty scaffolds (`__init__.py` only); build in that order since each depends on the previous.

### Non-negotiable invariants (each enforced by a test — see `docs/RESULTS.md`)

- Temporal train/test split only — a random split silently leaks the future into training for time-series data.
- `StandardScaler` (and any future encoder) is fit on training data only, then applied to test.
- Class imbalance (~1:745 positive rate) means model quality is judged on AUC/F1/precision/recall, never accuracy.
- LSTM sequence windows never mix rows from different `machine_id`s.
- New code follows the existing module docstring convention: WHY THIS FILE EXISTS / HOW IT WORKS (see any file in `src/` for the pattern), and inline comments explain *why*, not *what*.
