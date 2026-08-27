# Troubleshooting

Symptoms, causes, and fixes — ordered by how often they come up.

If none of this helps, [open an issue](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/issues/new/choose)
with the output of:

```bash
python --version && pip --version
git rev-parse --short HEAD
make test 2>&1 | tail -30
```

---

## Contents

- [Installation](#installation)
- [Training](#training)
- [The API](#the-api)
- [The dashboard](#the-dashboard)
- [Reports and LLM providers](#reports-and-llm-providers)
- [Docker](#docker)
- [Tests and quality gates](#tests-and-quality-gates)

---

## Installation

### `Could not find a version that satisfies the requirement tensorflow`

**Your Python is too new.** TensorFlow publishes no wheels for 3.13+.

```bash
python --version        # must be 3.10, 3.11, or 3.12
```

```bash
brew install python@3.12                 # macOS
sudo apt install python3.12 python3.12-venv   # Debian/Ubuntu

rm -rf venv
make setup              # finds 3.12 → 3.11 → 3.10 on PATH
```

`make setup` used to hardcode a Homebrew path and fail everywhere else. If you
are on an older checkout and it fails on Linux, that is why — pull, or run
`bash scripts/setup.sh` directly.

### `make setup` fails with "Python 3.10–3.12 required"

None of those interpreters is on your `PATH`. Install one, or point the script
at a specific binary:

```bash
/usr/local/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### `ModuleNotFoundError: No module named 'src'`

Run from the **repository root**, with the venv active:

```bash
cd /path/to/Predictive-Maintenance-GenAI
source venv/bin/activate
```

Scripts insert the root into `sys.path` themselves; a bare `python` REPL does
not. For interactive work, `pip install -e .`.

### Illegal instruction / crash on `import tensorflow`

Usually a CPU without AVX support, or an x86 wheel on ARM. Verify the
architecture matches:

```bash
python -c "import platform; print(platform.machine())"
pip debug --verbose | grep -i tag | head
```

---

## Training

### The training run hangs with no error

**The single most confusing failure in this project.** The process sits at 0%
CPU indefinitely. No traceback, no timeout, no message.

**Cause:** import order. TensorFlow and Apache Arrow (pulled in by
pandas/scikit-learn) each statically link their own copy of abseil, and
whichever loads first claims `AbslInternalPerThreadSemWait` process-wide. If
Arrow wins, TensorFlow's first graph execution waits on a semaphore that never
signals it.

**Fix:** import TensorFlow — or `src.models`, which imports it — **before**
anything that pulls in pandas.

```python
from src.models import PredictiveMaintenanceModel   # ✅ first
from src.data.preprocessing import DataPreprocessor
```

```python
from src.data.preprocessing import DataPreprocessor  # ❌ deadlocks later
from src.models import PredictiveMaintenanceModel
```

Do not alphabetise `src/models/__init__.py` and do not remove its
`# isort: skip_file`. Confirm a hang by sampling the process:

```bash
sample $(pgrep -f train_model) 5     # macOS
py-spy dump --pid $(pgrep -f train_model)
```

`AbslInternalPerThreadSemWait ... libarrow` in the stack confirms it. Full
writeup: [`devlog/day-04.md`](devlog/day-04.md).

### `FileNotFoundError: X_train.npy`

Preprocessing has not run:

```bash
python scripts/generate_data.py
python scripts/run_preprocessing.py
```

### `val_precision` stays at 0.0000

Either class weights are not being applied, or the validation split contains no
positives. Check the split summary in the log:

```
Val: 129,000 rows (...) | 175 positive
```

Zero positives means the split boundary landed badly — with `data/sample/` this
is expected, because it deliberately contains **no failure events** and cannot
be used to train.

### `val_auc` is 1.0000 from epoch 1

Suspect leakage. Verify the split is temporal and the scaler was fitted on
training data only:

```bash
python -m pytest tests/unit/test_preprocessing.py -k "split or scaler" -v
```

### Out of memory during preprocessing

The processed tensors are ~4.9 GB and the merge holds a large frame. Free ~8 GB
of RAM and ~6 GB of disk, or reduce the dataset:

```bash
python scripts/generate_data.py --machines 25
```

### The retrained model scores differently than the docs

Check the seed is taking effect — the log should say
`Seeded all RNGs with 42`. If the data was regenerated with a different
`--seed`, the dataset itself differs and no training seed will reproduce the
published figures.

---

## The API

### `503 — The model or dataset is not loaded`

Expected when artifacts are missing. Confirm:

```bash
curl -s localhost:8000/health | jq
```

```json
{"status": "degraded", "model_loaded": false, "dataset_loaded": false}
```

Then produce them — `generate_data.py`, `run_preprocessing.py`,
`train_model.py`. This is a designed state, not a crash: the API stays up so
`/health` can tell you what is wrong.

### `404 — Machine N is not in the dataset`

```bash
curl -s localhost:8000/machines | jq '.[].machine_id' | head
```

### `422 — No complete sequences could be built`

Not enough history. Feature engineering consumes the first 24 hours for rolling
and lag windows, and the LSTM needs 24 more — **at least 48 hours per machine**.

With `as_of`, check you have not rewound to before the data starts:

```bash
curl -s localhost:8000/health | jq '{data_start, data_end}'
```

### A prediction is always 0.0 or always 1.0

If every machine scores identically, the scaler and model are probably from
different runs. `Predictor` catches shape mismatches but cannot detect a scaler
fitted on different data. Re-run preprocessing and training together.

### `/fleet` is slow the first time

Expected: ~13.5 s cold for 100 machines, ~3 ms cached (5-minute TTL). Bypass
with `?refresh=true`. Concurrent requests for the same uncached timestamp are
serialised — the first computes, the rest wait and receive its result.

### CORS errors in a browser

The API allows only the dashboard origin, not `*`, deliberately. To add one,
edit the `CORSMiddleware` block in `src/api/main.py` or change
`DASHBOARD_PORT`.

---

## The dashboard

### "Cannot reach the API"

```bash
curl localhost:8000/health          # is it up?
make run-api                        # if not
```

In Docker the URL is `http://api:8000`, not `localhost` — inside the compose
network `localhost` is the dashboard itself. Set it in the sidebar or via
`API_BASE_URL`.

### Blank page behind a reverse proxy

Streamlit is a WebSocket application. Without upgrade headers it renders
nothing:

```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade    $http_upgrade;
proxy_set_header Connection "upgrade";
```

Under a path prefix, add `--server.baseUrlPath=dashboard`.

### `use_container_width` deprecation warnings

Your Streamlit is older than the pinned floor:

```bash
pip install -U "streamlit>=1.49.0,<2.0.0"
```

### Rewind shows "0 alerting" for every hour

Often correct — most hours are quiet. The dataset's final hour in particular
has no machine inside a pre-failure window. Try a documented case:
**2024-10-31 hour 6** (machine 51) or **2024-11-13 hour 12** (machine 96).

Rewind to 2024-10-30 and machine 51 goes quiet: that is the 24-hour horizon,
not a bug.

---

## Reports and LLM providers

### `502 — The language model is unavailable`

By design, and **the prediction is in the error detail**. Diagnose the provider:

```bash
curl localhost:11434/api/tags                    # is Ollama running?
ollama pull llama3                               # is the model pulled?
```

### `OPENAI_API_KEY is not set`

Either set it in `.env`, or use the keyless path:

```bash
pip install -e ".[ollama]"
ollama serve && ollama pull llama3
```

Note that `.env.example` placeholders (`your-openai-api-key-here`) are
deliberately treated as **unset** — otherwise the app believes it has
credentials and returns a 401 that reads like a broken key rather than an
absent one.

### `Google provider requires pip install langchain-google-genai`

```bash
pip install -e ".[google]"
```

### `LangChainDeprecationWarning: ChatOllama was deprecated`

`langchain-ollama` is not installed, so the deprecated community class is being
used as a fallback:

```bash
pip install -e ".[ollama]"
```

### `504 — the model did not respond within 120s`

A local model on CPU can exceed the ceiling. Use a smaller one:

```bash
curl -X POST localhost:8000/report \
  -d '{"machine_id": 51, "provider": "ollama", "model": "llama3.2:1b"}'
```

### A report contradicts its own numbers

That is a bug worth reporting — the whole design intent is that every figure
comes from the prediction record. Include the `/explain` output and the report
text. See [`devlog/day-07.md`](devlog/day-07.md) for three such bugs found by a
live model.

---

## Docker

### The API container is unhealthy

```bash
docker compose -f docker/docker-compose.yml logs api
docker compose -f docker/docker-compose.yml ps
```

Usually the model volume is empty. Health uses `/health` rather than a TCP
probe, so "listening but no model" correctly reports unhealthy.
`start_period` is 90 s — a container marked unhealthy before that has genuinely
failed.

### The dashboard starts before the API is ready

It should not — `depends_on: condition: service_healthy` gates it. If you are
running containers by hand rather than via compose, you have bypassed the gate.

### The build is enormous or slow

Check `.dockerignore` is being honoured. Build context should be ~2.9 MB:

```bash
docker build -f docker/Dockerfile.api . 2>&1 | head -3
```

A context in the gigabytes means `data/`, `venv/`, or `.git` is being sent.

### Permission denied writing to `/app/logs`

The container runs as `appuser`. On Linux the mounted `logs/` directory must be
writable by that uid:

```bash
mkdir -p logs && chmod 777 logs      # development only
```

### `host.docker.internal` does not resolve on Linux

That name is Docker Desktop's. On Linux, add:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

---

## Tests and quality gates

### The suite hangs

Almost always the abseil import order — see
[the training section](#the-training-run-hangs-with-no-error).
`tests/conftest.py` imports TensorFlow first for exactly this reason. If you
added a `conftest.py` that imports pandas earlier, that is the cause.

### The first run takes 90 seconds

TensorFlow's initial import on ARM64 macOS. Subsequent runs are ~26 s.

### `make quality` passes locally but CI fails

`make typecheck` runs mypy **twice** on purpose, because the two runs check
different programs — CI installs only the linters, so third-party symbols
become `Any` and `warn_return_any` starts firing. Run both:

```bash
mypy src/ config/
mypy --no-site-packages src/ config/
```

If flake8 or Black disagree between local and CI, your tooling has drifted:

```bash
pip install -r requirements-lint.txt
```

### 18 tests skip

Expected without a trained model — they skip rather than fail so CI stays
green on a clean checkout. Full run: 246 unit, of which 18 need artifacts.

### `test_served_threshold_matches_the_committed_evaluation_report` fails

You retrained and the chosen threshold moved, but `config/settings.py` was not
updated. Set both `PREDICTION_THRESHOLD` and `RISK_BAND_HIGH` to the value in
`models/evaluation_report.json`. This test exists precisely so a retrain cannot
half-land.

---

## See also

- [`development.md`](development.md#common-mistakes) — mistakes ranked by damage
- [`deployment.md`](deployment.md) — health checks and proxy configuration
- [`devlog/`](devlog/README.md) — how these problems were originally diagnosed
