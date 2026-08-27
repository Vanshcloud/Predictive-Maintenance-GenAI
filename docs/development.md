# Development Guide

The deeper guide. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) is the entry
point — setup, commands, and the invariants. This covers how to work in the
codebase once you are past that: debugging, the mistakes that are easy to make
here, and the reasoning the design rests on.

---

## Contents

- [Your first contribution](#your-first-contribution)
- [Architecture philosophy](#architecture-philosophy)
- [Local development loop](#local-development-loop)
- [Debugging](#debugging)
- [Testing](#testing)
- [Common mistakes](#common-mistakes)
- [Code review expectations](#code-review-expectations)

---

## Your first contribution

Good places to start, roughly in order of difficulty:

| Where | Why it's approachable |
|---|---|
| **Documentation** | The largest surface, and the easiest place to find something genuinely wrong. Numbers drift. |
| **`dashboard/`** | Pure HTTP client. No model, no TensorFlow, fast test loop. |
| **`scripts/`** | Self-contained entry points; `scripts/predict.py` has a known performance bug on the [roadmap](roadmap.md). |
| **`src/api/routes/`** | Thin handlers with a clear contract and good test coverage. |
| **`src/data/preprocessing.py`** | The heart of the project — and where a careless change silently corrupts every metric. Read [Common mistakes](#common-mistakes) first. |

### The loop

```bash
git checkout -b fix/short-description

make test          # establish a green baseline BEFORE changing anything
# ... edit ...
make quality       # lint, format, types
make test

git commit         # subject says what; body says why
```

Opening a pull request runs the same checks. If `make quality` passes locally
and CI disagrees, that is a bug worth reporting — the two are deliberately
wired to the same `PY_PATHS` in the `Makefile` precisely so they cannot drift.

---

## Architecture philosophy

Four ideas explain most of the decisions in this repository. Knowing them makes
the code predictable; not knowing them makes parts of it look arbitrary.

### 1. Layers are a dependency chain, not folders

```
config/ → src/utils/ → src/data/ → src/models/ → src/prediction/ → src/genai/ → src/api/ → dashboard/
```

Each package imports only from those to its left. This is enforced, not
suggested: `src/data/` imports no TensorFlow, `src/models/` imports no pandas,
and the dashboard imports nothing from `src/` at all — which is why its
container image is 803 MB against the API's 2.87 GB.

If a change needs to reach backwards, the abstraction is in the wrong place.

### 2. Silent failures get tests; loud ones can wait

The invariants in `CONTRIBUTING.md` share one property: **breaking them
produces better-looking numbers rather than an error.** A random train/test
split does not crash — it reports a higher AUC. Refitting the scaler at
inference does not crash — it returns plausible predictions that are quietly
wrong.

So test coverage is deliberately uneven. Test the things that fail silently.
A crash reports itself.

### 3. The comment explains why, never what

`# increment counter` is noise. `# sorted so memmap reads stay monotonic` is
the reason the line exists.

Every workaround must name the failure it prevents, because otherwise a future
reader will "simplify" it back into a bug. That rule is why `tests/conftest.py`
is a 20-line docstring wrapped around a single import.

### 4. Degrade, don't disappear

The prediction is safety-critical; the narrative is a convenience over it. So:

- No model? The API starts **degraded** and still serves `/health` — because an
  API that refuses to start cannot tell an operator why.
- LLM down? `502` **with the prediction attached**.
- One machine unscoreable? It is skipped; the fleet view still renders.

The system's answer never depends on its least reliable component.

---

## Local development loop

### Running the stack

```bash
make run-api            # :8000, --reload
make run-dashboard      # :8501, needs the API
make docker-up-d        # both, containerised, health-gated
```

The dashboard is a pure client, so you can point it anywhere:

```bash
API_BASE_URL=http://staging.internal:8000 streamlit run dashboard/app.py
```

### Working without a trained model

Most work does not need one. The unit suite runs against the committed
`data/sample/` fixture, and tests requiring a model skip themselves.

The API starts degraded without one, which is a legitimate state to develop
against — `/health`, `/machines`, and error paths all work.

### Fast feedback

```bash
python -m pytest tests/unit/test_preprocessing.py -v
python -m pytest tests/ -k "threshold or risk_band"
python -m pytest tests/ -x --ff              # stop on first failure, failures first
python -m pytest tests/ -q -p no:randomly    # if ordering is suspected
```

> The first run in a session pays ~90 s for TensorFlow's import on ARM64 macOS.
> Subsequent runs are ~26 s. That is not your change being slow.

---

## Debugging

### Getting inside a prediction

`explain_machine()` returns everything the model consumed. Use it before
reaching for a debugger:

```python
from src.models import PredictiveMaintenanceModel   # TensorFlow FIRST
from src.api.service import MachineDataStore
from src.prediction import Predictor
from config.settings import get_settings

store = MachineDataStore.load(get_settings().raw_data_path)
predictor = Predictor()

record = predictor.explain_machine(store.slice_for(51), 51)
print(record["failure_probability"])
for name in record["context"]["most_deviant_sensors"]:
    print(name, record["context"]["sensors"][name])
```

> Note the import order. `src.models` (TensorFlow) **before** anything that
> pulls in pandas. See [Common mistakes](#common-mistakes).

### Turning up the logs

```bash
LOG_LEVEL=DEBUG make run-api
```

`DEBUG` includes cache hits and per-batch training progress. File logs are in
`logs/app_YYYY-MM-DD.log`, at `DEBUG` regardless of console level.

### Tracing a 500

Unexpected exceptions return an opaque body with a `correlation_id`. The detail
is in the logs:

```bash
grep '\[a1b2c3d4\]' logs/app_*.log
```

### Inspecting the feature pipeline

```python
from src.data.preprocessing import DataPreprocessor
from src.data.ingestion import DataIngestion

pre = DataPreprocessor()
merged = pre.merge_tables(DataIngestion().load_dataset())
featured = pre.engineer_features(merged)

print(featured.shape)
print([c for c in featured.columns if "vibration" in c])
print(featured[featured.machine_id == 51].tail(3).T)
```

### Debugging the dashboard

Streamlit reruns the whole script on every interaction, which makes `print`
debugging confusing. The pure logic lives in `dashboard/risk.py` specifically
so it can be tested without a running app — put logic there and test it
directly.

### `pdb` in a test

```bash
python -m pytest tests/unit/test_predictor.py -x --pdb
```

---

## Testing

### Layout

```
tests/
├── conftest.py          # imports TensorFlow first — load-bearing, do not "clean up"
├── unit/                # 246 tests, ~26 s, run against data/sample/
└── integration/         # 13 tests, ~9 min, need generated data + a model
```

Integration tests are excluded by default (`-m 'not integration'` in
`pyproject.toml`). Run them after touching feature engineering, scaling, or the
prediction path:

```bash
make test-integration
```

### What to test

The bar is **not** coverage percentage. It is: *would this test fail if the
behaviour regressed?*

Prefer a test that pins an invariant over one that pins an implementation
detail. `test_risk_colours_are_keyed_off_the_api_level_only` parses the
dashboard's source to prove it never derives a risk band — that survives a
refactor, where asserting on a specific function call would not.

### Writing a regression test

When fixing a bug, **verify the test fails against the old code**:

```bash
git stash push src/path/to/fix.py
python -m pytest tests/unit/test_thing.py::test_new -q     # must FAIL
git stash pop
python -m pytest tests/unit/test_thing.py::test_new -q     # must PASS
```

A regression test that passes both ways is not testing the fix.

### Tests that need a model

```python
try:
    predictor = Predictor()
except ModelNotFoundError as e:
    pytest.skip(f"trained artifacts not available: {e}")
```

Skip, never fail — CI has no model, and a permanently red build teaches people
to ignore red builds.

**But:** if what you are asserting is readable from committed files, do not put
it behind that fixture. Two invariants once lived in a model-gated test and so
were never checked by CI at all. They now live in `tests/unit/test_smoke.py`.

---

## Common mistakes

Ranked by how much damage they do before anyone notices.

### 1. Importing pandas before TensorFlow

**Symptom:** the process hangs at 0% CPU. No traceback, no timeout, no error.
Indistinguishable from "slow".

**Cause:** TensorFlow and Apache Arrow (loaded by pandas/scikit-learn) each
statically link their own copy of abseil. Whichever loads first claims
`AbslInternalPerThreadSemWait` process-wide. If Arrow wins, TensorFlow's first
graph execution waits on a semaphore that never signals.

**Rule:** any entry point using TensorFlow must import `src.models` (or
`tensorflow`) **before** `src.data`. `src/models/__init__.py` carries an
`# isort: skip_file` for this reason — do not alphabetise it, and do not remove
the directive.

Full diagnosis, with the sampled stack: [`devlog/day-04.md`](devlog/day-04.md).

### 2. Refitting the scaler at inference

**Symptom:** none. Predictions look plausible and every test passes.

Use `apply_scaler()`, never `normalize()`, outside training. `normalize()`
*fits*, and it must only ever see the training split.

### 3. Zero-filling a missing categorical

`hours_since_maint_comp1 = 0` means **serviced this hour** — the opposite of
"never serviced", which is `9999`. `Predictor._FILL_RULES` encodes the correct
default per family, and raises for anything with no defensible one.

### 4. Recomputing a risk band in the dashboard

`risk_level` is the API's to assign. A second source of truth looks correct
from either side while destroying trust in both. A test greps the dashboard
source for threshold literals.

### 5. Changing the threshold in one place

`PREDICTION_THRESHOLD` and `RISK_BAND_HIGH` must stay equal, and both must
match the committed `evaluation_report.json`. Two tests enforce this.

### 6. Adding a feature without retraining

Changing the feature set or `sequence_length` invalidates the saved model.
`Predictor` verifies the model, scaler, and feature contract agree, and refuses
to start otherwise — which is the correct failure, but it means feature changes
and retraining ship together.

### 7. Quoting a metric without its conditions

Every number in this repository states the threshold, the split, and the seed
that produced it. A bare "F1 0.91" is not a fact about anything.

---

## Code review expectations

What gets asked in review here:

- **Does the subject say what and the body say why?** The diff shows the what.
- **Would the new test fail without the change?**
- **Does this touch an invariant?** If yes, the reasoning belongs in the commit
  body, not just the code.
- **Is a workaround explained?** An unexplained workaround gets deleted by
  someone in six months.
- **Do docs ship in the same commit?** Not a follow-up.
- **Is a quoted number reproducible?** If a metric changed, say how it was
  measured.

---

## See also

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — setup, commands, invariants
- [`architecture.md`](architecture.md) — layer diagram and module responsibilities
- [`troubleshooting.md`](troubleshooting.md) — when something is broken
- [`devlog/`](devlog/README.md) — why decisions were made, including the wrong turns
