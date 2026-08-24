# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An end-to-end Predictive Maintenance platform: a TensorFlow LSTM predicts equipment failures 24h in advance from sensor telemetry, and LangChain + an LLM turns those predictions into plain-English maintenance reports, exposed via FastAPI and a Streamlit dashboard. It was built incrementally as a 12-day/12-milestone portfolio project — read **`IMPLEMENTATION_PLAN.md`** first — it is the single source of truth for scope, architecture, risks, milestones, and current status — then the latest `docs/DayX.md` for what the last session actually did. Both must be updated before a session ends; docs and code ship in the same commit.

Progress: **all 12 milestones are delivered**, plus a post-project Day 13 (`docs/Day13.md`). Every layer is implemented and tested end to end: data pipeline, LSTM (test F1 0.8949, 8/8 failure events caught), prediction pipeline, LangChain reports and assistant, FastAPI, Streamlit dashboard, and two Docker images wired by compose. 229 unit + 13 integration tests; flake8/Black/isort/mypy all clean and blocking in CI. Work from here is enhancement, not construction.

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
python scripts/evaluate_model.py             # threshold sweep + test metrics -> models/evaluation_report.json

python scripts/predict.py --machine 51       # score one machine from the CLI
python scripts/generate_report.py --machine 51   # LLM report for one machine
python scripts/ask.py                        # interactive assistant

make run-api                                 # uvicorn on :8000  (docs at /docs)
make run-dashboard                           # streamlit on :8501
make docker-up                               # both in containers, health-gated

make test-integration                        # needs the trained model + generated data
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
- **`src/prediction/`** — `Predictor` loads the model, scaler, and feature-column contract, then reproduces the training feature pipeline at inference time. `_reconcile_features()` fills categorical columns absent from a scored batch using per-family defaults (9999 for `hours_since_maint_*`, 0 for `model_*`) — a plain zero-fill for maintenance would read as "serviced this hour". Parity with training is asserted over all 172,800 test sequences.
- **`src/genai/`** — `prompts.py` (system prompts + `format_machine_facts()`), `chains.py` (LCEL report chain), `assistant.py` (multi-turn Q&A). Every figure an LLM quotes comes from the prediction record it is handed; it is given nothing else. An LLM outage degrades to a 502 that still carries the prediction — the model's answer never depends on the language model.
- **`src/api/`** — `service.py` owns the loaded model and dataset as process-wide state; routes stay thin. **`MachineDataStore.slice_for()` is mandatory, not an optimisation**: handing the predictor the full fleet to score one machine takes over two minutes against ~160 ms sliced. Exceptions map to status codes through the `PredMaintenanceError` hierarchy.
- **`dashboard/`** — a pure API client. It holds no model and does no scoring, and must never recompute risk bands from a probability; `risk_level` is the API's to assign (asserted by `test_risk_colours_are_keyed_off_the_api_level_only`).

### Point-in-time assessment (`as_of`)

Every prediction endpoint takes an optional `as_of` timestamp; `None` means "the latest reading", which is the pre-existing behaviour. When set, `slice_for()` drops everything after it — **telemetry, errors, and maintenance alike**. Filtering telemetry alone would leak, because `errors_last_24h` and `hours_since_maintenance` are model features. The cutoff is inclusive: the chosen hour has already happened.

`/fleet`'s cache is keyed by `as_of`. It was a single slot before, and adding the parameter without re-keying would serve a cached present-day answer to a request about a past date. See `docs/Day13.md` and `tests/integration/test_time_travel.py`.

### Non-negotiable invariants (each enforced by a test — see `docs/RESULTS.md`)

- Temporal train/test split only — a random split silently leaks the future into training for time-series data.
- `StandardScaler` (and any future encoder) is fit on training data only, then applied to test.
- Class imbalance (~1:745 positive rate) means model quality is judged on AUC/F1/precision/recall, never accuracy.
- LSTM sequence windows never mix rows from different `machine_id`s.
- `as_of` filtering covers every table, not just telemetry — see above.
- New code follows the existing module docstring convention: WHY THIS FILE EXISTS / HOW IT WORKS (see any file in `src/` for the pattern), and inline comments explain *why*, not *what*.
