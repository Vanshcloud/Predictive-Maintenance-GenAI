# Contributing

Thanks for taking an interest in this project. This document covers how to get a
working development environment, what the quality gates are, and — most
importantly — the handful of correctness rules in this codebase that are easy to
break by accident.

If you only read one section, make it
[Invariants that must not break](#invariants-that-must-not-break).

---

## Table of contents

- [Development setup](#development-setup)
- [Everyday commands](#everyday-commands)
- [Quality gates](#quality-gates)
- [Project layout and the layering rule](#project-layout-and-the-layering-rule)
- [Invariants that must not break](#invariants-that-must-not-break)
- [Two non-obvious gotchas](#two-non-obvious-gotchas)
- [Code style](#code-style)
- [Testing](#testing)
- [Commits and pull requests](#commits-and-pull-requests)
- [Reporting bugs](#reporting-bugs)

---

## Development setup

**Python 3.12 is required.** TensorFlow publishes no wheels for 3.13+, so a
newer system interpreter will fail at install time.

```bash
git clone https://github.com/Vanshcloud/Predictive-Maintenance-GenAI.git
cd Predictive-Maintenance-GenAI

make setup                # creates venv/, installs requirements-dev.txt, copies .env
source venv/bin/activate
```

`make setup` looks for `python3.12`, then `3.11`, then `3.10` on your `PATH` and
stops with a readable message if none is present. To do it by hand:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
```

### Optional LLM providers

The report-generation layer supports OpenAI, Google Gemini, and a local Ollama
model. Provider packages are imported lazily, so install only the one you want:

```bash
pip install -e ".[ollama]"    # local model, no API key required
pip install -e ".[google]"    # Gemini
pip install -e ".[llm]"       # both
```

The test suite does not require any of them.

### Data and model artifacts

`data/raw/`, `data/processed/`, and the trained `.keras` file are gitignored —
together they are roughly 6 GB. They are fully reproducible:

```bash
python scripts/generate_data.py        # -> data/raw/  (883,231 rows, seed 42)
python scripts/run_preprocessing.py    # -> data/processed/*.npy + scaler.joblib
python scripts/train_model.py          # -> models/*.keras, metrics.json
```

You do **not** need any of this to run the test suite. `data/sample/` is
committed for exactly that reason. Note that it deliberately contains zero
failure events, so it cannot be used to train or evaluate a model.

---

## Everyday commands

```bash
make help                 # list every target

make test                 # unit tests (integration excluded)
make test-integration     # integration tests (slow; needs generated data + model)
make test-all             # both
make test-cov             # unit tests with a coverage report

make lint                 # flake8
make format               # black + isort (writes)
make format-check         # black + isort (checks only)
make typecheck            # mypy, in both configurations — see below
make quality              # lint + format-check + typecheck

make run-api              # uvicorn on :8000  (interactive docs at /docs)
make run-dashboard        # streamlit on :8501
make docker-up            # both services in containers, health-gated
```

The paths the linters check are declared once, as `PY_PATHS` in the `Makefile`.
CI calls `make lint` and `make format-check` rather than repeating the list —
when the two drifted, `make quality` passed locally on a commit that went red in
CI.

---

## Quality gates

`make quality` must pass before a pull request is merged. CI runs the same
checks and they are blocking.

**`make typecheck` runs mypy twice, deliberately.** The two runs check different
programs:

- `mypy src/ config/` sees your installed virtualenv, so pandas and LangChain
  have real type information.
- `mypy --no-site-packages src/ config/` is what CI does, because CI's lint job
  installs only the linters. Without the libraries, `ignore_missing_imports`
  turns every third-party symbol into `Any`, and `warn_return_any` starts firing
  at each seam.

Both must be clean. Four `no-any-return` errors once reached the main branch
because only the first was being run locally.

---

## Project layout and the layering rule

```
config/  ->  src/utils/  ->  src/data/  ->  src/models/  ->  src/prediction/
         ->  src/genai/  ->  src/api/  ->  dashboard/
```

**Each package may import only from packages to its left.** Never skip a layer,
and never reach into a later layer's internals.

This is not a style preference — it is load-bearing. `src/data/` imports no
TensorFlow and `src/models/` imports no pandas, which is what lets each layer be
tested independently. The dashboard imports nothing from `src/` at all, and that
boundary is why its container image is 803 MB against the API's 2.87 GB. The
dashboard `Dockerfile` copies only `dashboard/`, so a violation breaks the build
— which is the intended failure.

See [`docs/architecture.md`](docs/architecture.md) for what each layer is
responsible for and why.

---

## Invariants that must not break

These are correctness properties, not preferences. Every one of them fails
**silently** — producing better-looking numbers rather than an error — which is
why each is pinned by an explicit test. If a change of yours makes one of these
tests fail, the test is almost certainly right.

| Invariant | Why it matters |
|---|---|
| The train/validation/test split is **temporal**, never random | With 24-hour lag features, a random split puts hour `t` in training and `t+1` in test. The model "predicts" a failure whose evidence it has memorised, and the reported metrics become fiction. |
| `StandardScaler` is fit on **training data only**, then applied | Fitting on the full dataset leaks test-period statistics into training. `apply_scaler()` exists separately from `normalize()` precisely to make refitting hard to do by accident. |
| Sequence windows never span two `machine_id`s | A cross-machine window describes a machine that does not exist. |
| Quality is judged on **AUC / precision / recall / F1**, never accuracy | At a 1:864 positive rate, "always predict no failure" scores 99.88%. `ModelEvaluator` does not compute accuracy at all. |
| The alert threshold is chosen on the **validation** split; test is scored once | Choosing a threshold on test and then reporting test metrics at it is the same mistake as early-stopping on test. |
| `RISK_BAND_HIGH` equals `PREDICTION_THRESHOLD` | "High or above" must mean exactly "the model is alerting". If the band boundary and the decision drift apart, the dashboard can show *medium* for a machine the API has flagged. |
| The served threshold matches the committed evaluation report | `scripts/evaluate_model.py` chooses an operating point; copying it into `config/settings.py` is a manual step. A retrain that moves the optimum otherwise leaves the API serving the previous model's threshold, silently. |
| Point-in-time (`as_of`) filtering covers **every** table | Filtering telemetry alone leaks, because `errors_last_24h` and `hours_since_maintenance` are model features. |
| Training is seeded before the model is built | A quoted F1 is only meaningful if a second run reproduces it. `keras.utils.set_random_seed` must run before the LSTM kernels are drawn. |
| `/fleet`'s cache is keyed by `as_of` and stays bounded | A single-slot cache serves a present-day answer to a request about a past date. An unbounded one grows without limit, because `as_of` is a caller-supplied query parameter. |
| The dashboard never recomputes a risk band | `risk_level` is the API's to assign. Two sources of truth here look correct from either side while destroying trust in both. |

---

## Two non-obvious gotchas

### 1. TensorFlow must be imported before pandas or scikit-learn

`src/models/__init__.py` has a deliberate, non-alphabetical import order and an
`# isort: skip_file` directive. **Do not "tidy" either one.**

TensorFlow statically links its own copy of abseil, and so does Apache Arrow,
which pandas and scikit-learn both load. Whichever library loads first claims
the `AbslInternalPerThreadSemWait` symbol for the whole process. If Arrow wins,
TensorFlow's first graph execution deadlocks at 0% CPU — no traceback, no
timeout, no error message, indistinguishable from "slow".

Any new entry point that uses TensorFlow must import `src.models` (or
`tensorflow` itself) before importing `src.data`. `tests/conftest.py` does this
for the test suite, which is why it imports TensorFlow and nothing else.

The full diagnosis, including the sampled stack trace, is in
[`docs/devlog/day-04.md`](docs/devlog/day-04.md).

### 2. The API must slice before it scores

`Predictor.explain_machine()` runs feature engineering over whatever dataset it
is handed. Handed the whole fleet in order to score one machine, that takes over
two minutes; sliced to one machine and a recent window first, the same call is
~160 ms.

`MachineDataStore.slice_for()` is therefore mandatory, not an optimisation.
Without it the endpoint cannot exist.

---

## Code style

- **Black** (88 columns) and **isort** (black profile) are authoritative. Run
  `make format` rather than arguing with them.
- **Type hints** on public functions. `mypy` is blocking; `disallow_untyped_defs`
  is deliberately off, so private helpers and test fixtures need not be annotated.
- **No `print()`.** Every module gets `logger = get_logger(__name__)`.
- **Exceptions inherit from `PredMaintenanceError`**, grouped by layer (`Data*`,
  `Model*`, `LLM*`/`Report*`, `API*`) so callers can catch precisely or broadly.
  The API maps that hierarchy onto status codes, with no string matching on
  error messages.
- **Never hardcode a path, port, or model name.** Add it to `config/settings.py`
  and read it through `get_settings()`.

### Module docstrings

Modules in `src/` follow a `WHY THIS FILE EXISTS` / `HOW IT WORKS` convention —
see any existing file for the pattern. Inline comments should explain *why*, not
*what*; the code already says what.

This convention is the reason a reader can tell the difference between a
deliberate oddity and a mistake. When you write something that looks wrong but
is right, say why, next to it.

---

## Testing

```bash
make test              # what CI runs on every push
```

Tests run against the committed `data/sample/` fixture, so they need no
generated data. Tests that require a trained model skip themselves rather than
failing, which is how CI stays green without a 5 GB checkout.

The first run on ARM64 macOS pays roughly 90 seconds for TensorFlow's initial
import; subsequent runs finish in about 25 seconds.

Integration tests are excluded from the default run — they score the full
876,000-row dataset and some of them talk to a live language model. Run them
explicitly after touching feature engineering, scaling, or the prediction path:

```bash
make test-integration
```

**New behaviour needs a test.** The bar is not coverage percentage, it is
whether the test would fail if the behaviour regressed. Prefer a test that pins
an invariant over one that pins an implementation detail.

---

## Commits and pull requests

- Branch from `main`. Do not commit to `main` directly.
- Conventional-commit prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `chore:`, `build:`, with an optional scope — `fix(api): ...`.
- The subject line says what changed; **the body says why.** A diff shows the
  what for free. If a change is subtle or reverses an earlier decision, the body
  is where that belongs.
- **Documentation ships with the code that changed it** — same commit, not a
  follow-up.
- Run `make quality` and `make test` before opening a pull request. CI runs the
  same checks and will block on them.
- Keep commits logically grouped. One concern per commit is easier to review and
  far easier to revert.

---

## Reporting bugs

Open an issue with: what you expected, what happened, the versions involved
(`python --version`, and the relevant package from `pip freeze`), and the
smallest set of steps that reproduces it. If it involves a prediction, include
the machine id and the `as_of` timestamp — the point-in-time behaviour makes
those essential to reproducing anything.

For anything with a security dimension, follow
[`SECURITY.md`](SECURITY.md) instead of opening a public issue.
