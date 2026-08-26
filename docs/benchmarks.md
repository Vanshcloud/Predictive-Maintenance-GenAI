# Benchmarks

Measured numbers, with the method that produced each. Anything not measured is
marked **not measured** rather than estimated.

> **These are single-machine figures from one hardware configuration.** They
> establish orders of magnitude and relative cost, not a performance guarantee.
> Nothing here has been measured under sustained production load.

---

## Test environment

| | |
|---|---|
| Hardware | Apple Silicon, CPU-only |
| OS | macOS (Darwin 25.5) |
| Python | 3.12.14 |
| TensorFlow | 2.x, CPU build — **no GPU used anywhere in this document** |
| Dataset | Synthetic, seed 42 — 100 machines × 365 days, 883,231 rows |

**Not measured:** GPU inference, Linux, Windows, ARM servers, containers under
resource limits, or any cloud instance type. Numbers will differ.

---

## Inference latency

End-to-end HTTP, measured against a running API with the model and full dataset
loaded. Client and server on the same machine, so **network time is excluded**.

| Endpoint | Median | p95 | Range | n |
|---|---|---|---|---|
| `GET /machines/{id}/predict` | **137 ms** | 139 ms | 134 – 148 ms | 30 |
| `GET /machines/{id}/explain` | **175 ms** | — | — | 1 |
| `GET /fleet` (cold, 100 machines) | **13.5 s** | — | — | 1 |
| `GET /fleet` (cached) | **2.8 ms** | — | — | 1 |

<details>
<summary>Method</summary>

```python
import time, statistics, requests
lat = []
for i in range(30):
    t = time.perf_counter()
    requests.get(f"http://localhost:8000/machines/{(i % 50) + 1}/predict", timeout=60)
    lat.append((time.perf_counter() - t) * 1000)
print(statistics.median(lat))
```

Different machine ids each iteration, so this measures the real path rather
than a warm cache. The service was already warm — first-call latency after
startup is higher because TensorFlow traces the graph.
</details>

**NFR-4 (p95 `/predict` < 500 ms): met, at 139 ms.**

### Why `/fleet` costs 13.5 s

100 machines × ~135 ms each. It is cached for 5 minutes and bounded to 16
entries keyed by `as_of`.

Concurrent requests for the same uncached key are serialised behind a compute
lock — measured before that lock existed, four simultaneous requests produced
four independent scorings and made every caller wait **57.8 s** for work that
takes 13.4 s once. Cache *hits* never block: a warm hit returns in **5 ms**
while a cold scoring is in flight.

### The slicing constraint

| Approach | Time to score one machine |
|---|---|
| `slice_for()` — one machine, last 200 h | **~160 ms** |
| Full dataset handed to the predictor | **> 120 s** |

Roughly **800×**. `merge_tables` and `engineer_features` run over whatever they
are given, so passing 876,000 rows to score one machine does 99% of its work on
rows it discards. This is why `MachineDataStore.slice_for()` is mandatory
rather than an optimisation.

> `scripts/predict.py --machine N` does **not** use it, and takes ~3 minutes to
> return one row. Known, on the [roadmap](roadmap.md).

### Report generation

| Path | Time |
|---|---|
| `POST /report` — local Ollama | **~21 s** |
| `POST /report` — hosted provider | **not measured** |

Isolated in its own router precisely because of this. A 21-second call must
never be able to delay a 137 ms prediction.

---

## Training

| Stage | Time | Output size |
|---|---|---|
| `generate_data.py` | ~1 min | 41 MB |
| `run_preprocessing.py` | ~3 min | 4.9 GB |
| `train_model.py` | **~81 min** | 1.8 MB |
| `evaluate_model.py` | ~2 min | — |

Training ran 28 of 30 epochs before early stopping, with best weights from
epoch 23 — roughly **2.9 min/epoch** over 567,000 training sequences at batch
size 256 (~2,215 batches/epoch, ~36 ms/batch).

**Not measured:** GPU training time, multi-GPU, or mixed precision.

---

## Memory

Resident set size of the `uvicorn` process, model and full dataset loaded.

| State | RSS |
|---|---|
| Idle, fully loaded | **560 MB** |
| After 20 single predictions | 597 MB |
| After a full fleet scoring | **604 MB** |

<details>
<summary>Method</summary>

`ps -o rss= -p <uvicorn pid>`, sampled after each phase. RSS includes the
876,000-row pandas dataset held in memory by `MachineDataStore`, which
dominates — the model itself is 1.76 MB.
</details>

Growth across a fleet scoring is ~44 MB and settles; the fleet cache is bounded
at 16 entries. **Not measured:** RSS over days of continuous operation, so
slow-leak behaviour is unverified.

**Planning figure: budget ~1 GB per worker.** `uvicorn --workers N` gives each
worker its own copy of both the model and the dataset.

---

## Artifact sizes

| Artifact | Size |
|---|---|
| Trained model (`.keras`) | **1.76 MB** |
| Model parameters | **149,825** |
| Fitted scaler | ~9 KB |
| Raw dataset | 41 MB |
| Processed tensors | **4.9 GB** |
| Committed sample fixture | 356 KB |
| API container image | **2.87 GB** |
| Dashboard container image | **803 MB** |
| Docker build context (after `.dockerignore`) | 2.9 MB, from a 7.3 GB tree |

The image gap is the payoff of the dashboard importing nothing from `src/`.
Installing `requirements.txt` there would pull TensorFlow, scikit-learn, and
LangChain into an image whose entire job is calling `requests` and drawing
charts.

The processed tensors are 4.9 GB because 869,000 sequences × 24 timesteps × 63
features × 4 bytes is inherently large. They are memory-mapped during training
rather than loaded.

---

## Test suite

| Suite | Count | Time |
|---|---|---|
| Unit | 245 | **~26 s** |
| Integration | 13 | ~9 min |
| Coverage | 86% | — |

First run on ARM64 macOS pays roughly **90 s** for TensorFlow's initial import;
subsequent runs are ~26 s. The integration suite is slow because it scores the
full 172,800-sequence test tensor and some tests call a live language model.

---

## CPU utilisation

**Not measured** beyond the observation that fleet scoring is CPU-bound.

The evidence for that claim is indirect but solid: four concurrent fleet
scorings took 57.8 s each versus 13.4 s for one, which is what contention for
the same cores looks like rather than parallel speed-up. No profiler run, no
per-core breakdown, no flame graph.

Anyone planning capacity should measure on their own hardware.

---

## What would change these numbers

| Change | Expected effect | Measured? |
|---|---|---|
| GPU inference | Large improvement on `/fleet`, little on single predictions (dominated by pandas feature engineering, not the model) | ❌ |
| Batching fleet scoring into one model call | Should cut `/fleet` substantially | ❌ |
| `slice_for()` in `scripts/predict.py` | ~3 min → ~160 ms | ❌ (arithmetic from the two measured figures) |
| Deduplicating `explain_machine`'s double feature pass | ~24% off `/explain` | ✅ measured at 158 ms vs 120 ms |
| Smaller `DEFAULT_WINDOW_HOURS` | Less per-request work; risks losing feature history | ❌ |

---

## Reproducing

```bash
make docker-up-d                      # or: make run-api

python -c "
import time, statistics, requests
lat = []
for i in range(30):
    t = time.perf_counter()
    requests.get(f'http://localhost:8000/machines/{(i % 50) + 1}/predict', timeout=60)
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()
print(f'median {statistics.median(lat):.0f} ms  p95 {lat[int(.95 * len(lat)) - 1]:.0f} ms')
"

docker images predictive-maintenance-api --format '{{.Size}}'
```

---

## See also

- [`architecture.md`](architecture.md#request-lifecycle) — why the slow path is isolated
- [`deployment.md`](deployment.md) — worker sizing and health checks
- [`RESULTS.md`](RESULTS.md) — model quality metrics, as opposed to speed
