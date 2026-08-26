# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Per-milestone engineering reports — including the bugs, the dead ends, and the
reasoning behind each design decision — live in [`docs/`](docs/README.md).
Every metric quoted below is stated with its caveats in
[`docs/RESULTS.md`](docs/RESULTS.md).

---

## [Unreleased]

### Added

- Contributor-facing documentation: `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, and this changelog. Development conventions, the layering
  rule, and the correctness invariants are now documented in `CONTRIBUTING.md`
  and [`docs/architecture.md`](docs/architecture.md).
- Optional dependency extras for the LLM providers, which are imported lazily and
  were previously undeclared: `pip install -e ".[ollama]"`, `".[google]"`, or
  `".[llm]"`.
- Two configuration tests that run without a trained model, so they execute in
  CI: one asserting the risk bands ascend and that `RISK_BAND_HIGH` equals
  `PREDICTION_THRESHOLD`, one asserting the served threshold matches the
  committed evaluation report. The equivalent assertions previously lived in a
  test that skipped whenever no model artifact was present — which is every CI
  run.

### Fixed

- **Concurrent `/fleet` requests stampeded the cache.** Route handlers are
  synchronous, so FastAPI runs them in a threadpool; checking the cache and
  filling it were unsynchronised, so every request arriving for an uncached
  `as_of` while another was computing missed too and recomputed the same
  answer. Measured: four concurrent requests for one cold timestamp produced
  four full fleet scorings and made every caller wait 57.8 s for work that
  takes 13.4 s once. Now serialised behind a compute lock, with cache reads on
  a separate short-lived lock so a hit still returns in milliseconds while a
  cold scoring is in flight.
- **`ModelEvaluator.evaluate()` wrote invalid JSON on single-class input.** The
  guard was written as `except ValueError`, which scikit-learn stopped raising in
  1.6 — it now warns and returns `nan`. The fallback had become unreachable, so
  `nan` reached `models/metrics.json`, where `json.dump` writes it as a bare
  `NaN` token that no strict JSON parser accepts. Now tests the precondition
  directly.
- **The confusion matrix could change shape.** `confusion_matrix()` inferred its
  labels from the data, returning a 1×1 matrix when only one class was present,
  despite being published and read as `[[tn, fp], [fn, tp]]`. Pinned with
  `labels=[0, 1]`, which is identical for every two-class input.
- **`make setup` failed on every platform except Apple Silicon**, having
  hardcoded a Homebrew ARM interpreter path. It now delegates to
  `scripts/setup.sh`, which already resolved `python3.12 → 3.11 → 3.10` from
  `PATH`. That script is also now independent of the working directory it is
  invoked from.
- Corrected published figures that had drifted from the artifacts they describe:
  the README summary quoted the pre-retrain model's false-alarm count and lead
  time, `docs/RESULTS.md` reported an early-stopping epoch that contradicted
  `models/training_history.json`, and test counts were stale in six places.
- Corrected two docstrings that described behaviour the code does not have —
  most importantly `create_sequences()`, whose description of the label pairing
  implied a leak the implementation does not contain.

---

## [0.1.0] — 2026-08-25

First complete end-to-end system: data generation through to a containerised API
and dashboard.

### Added

**Data pipeline**

- Synthetic dataset generator producing 883,231 rows across five related tables
  (telemetry, machines, errors, maintenance, failures) from a fixed seed.
- `DataIngestion` with format detection, `DataValidator` producing a structured
  `ValidationReport`, and an eight-dimension exploratory analysis script.
- `DataPreprocessor`: merges the five tables, engineers 63 rolling, lag, and
  time-since-last features, builds 24-hour-horizon binary labels, performs a
  chronological train/validation/test split, fits `StandardScaler` on the
  training split only, and slides 24-step windows into `(N, 24, 63)` tensors.

**Model**

- LSTM architecture (128 → 64 → 32 → 1, 149,825 parameters) with a hand-written
  `GradientTape` training loop implementing class weighting, early stopping,
  learning-rate reduction, and checkpoint resume explicitly.
- `ModelEvaluator` reporting AUC, precision, recall, F1, and a confusion matrix —
  and deliberately not accuracy, which is meaningless at a 1:864 positive rate.
- Threshold sweep over the precision-recall curve with cost weighting, including
  a `lowest_cost_is_degenerate` flag after a cost-optimal threshold selected on
  validation was found not to transfer.
- Recall measured over failure **events** as well as hourly sequences, with
  lead-time statistics.

**Inference**

- `Predictor`, which reuses the training feature code rather than
  reimplementing it, and verifies at load time that model, scaler, and feature
  contract describe the same thing.
- Feature reconciliation for categorical columns absent from a scored batch,
  using per-family defaults rather than a blanket zero fill.

**GenAI layer**

- Grounded report generation over LangChain, supporting OpenAI, Google Gemini,
  and a local Ollama model. Every figure a model quotes is supplied from the
  prediction record.
- Multi-turn maintenance assistant that declines questions the data cannot
  answer.

**API and dashboard**

- FastAPI service with nine endpoints, an exception hierarchy mapped onto status
  codes, and report generation isolated so a slow language model cannot delay a
  prediction.
- Point-in-time assessment: every prediction endpoint accepts an `as_of`
  timestamp and hides all later data — telemetry, errors, and maintenance alike.
- Streamlit dashboard as a pure HTTP client that holds no model and performs no
  scoring.

**Infrastructure**

- Two Docker images (API 2.87 GB, dashboard 803 MB) with multi-stage builds,
  non-root users, and health checks that report readiness rather than liveness.
- `docker compose` topology with health-gated startup ordering.
- GitHub Actions CI: lint, format, and type checks with pinned tooling; unit and
  integration tests; a dependency audit; image builds; and a smoke test asserting
  the API starts degraded rather than crashing when no model is mounted.

### Changed

- Training is seeded end to end, so published metrics are reproducible rather
  than merely reported. The model was retrained under the seed —
  **test F1 0.9086** at threshold 0.3415, 8 of 8 failure events caught with a
  median 23.5 hours of warning.
- Early stopping and checkpoint selection monitor `val_f1` rather than `val_auc`,
  which saturates under this class imbalance and selects on noise.
- Model selection moved onto a dedicated validation split so the test set is
  scored exactly once, at a threshold chosen without observing it.
- Dashboard risk colours meet WCAG AA contrast at their rendered size, asserted
  numerically by tests rather than judged by eye.

### Fixed

- An abseil symbol collision between TensorFlow and Apache Arrow that deadlocked
  the process at 0% CPU with no traceback. Import order is now load-bearing and
  documented at each site that depends on it.
- The API image shipped Streamlit, which nothing in `src/` imports; dependencies
  are now split per service.
- `/fleet`'s cache is keyed by `as_of` and bounded by an LRU, having previously
  been a single unbounded slot that would serve a present-day answer to a
  request about a past date.
- All 159 outstanding type errors resolved, making `mypy` blocking in CI.

[Unreleased]: https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/releases/tag/v0.1.0
