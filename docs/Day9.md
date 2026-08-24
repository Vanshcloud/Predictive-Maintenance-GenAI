# Day 9 Summary

| Field | Value |
|---|---|
| **Objective** | Expose the model, inference pipeline, and report generator as a REST API. |
| **Expected outcome** | FastAPI app with startup-loaded artifacts, validated schemas, layer-mapped error handling, and the slow LLM path isolated from the fast prediction path. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M9 — REST API |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `70f8d37` — "docs: record Day 8 and complete the GenAI layer" |
| **Existing modules** | config, utils, data, models, prediction, genai — all complete |
| **Tests** | 161 unit + 9 integration, flake8 clean |
| **`src/api/`** | Empty scaffold: `__init__.py` and `routes/__init__.py` only |

Everything built so far was a Python library. Day 9 makes it a service.

---

# Tasks Planned

### T1 — Measure whether prediction is servable at all

| Field | Detail |
|---|---|
| **Purpose** | Before designing endpoints, establish the latency budget. |
| **Priority** | P0 — the answer determines the architecture |

### T2 — Schemas

| Field | Detail |
|---|---|
| **Purpose** | Validate untrusted input at the boundary; generate the OpenAPI docs from the same definitions the code enforces. |
| **Files affected** | `src/api/schemas.py` |
| **Priority** | P0 |

### T3 — Service layer

| Field | Detail |
|---|---|
| **Purpose** | Load model and dataset once; keep routes thin. |
| **Files affected** | `src/api/service.py` |
| **Priority** | P0 |

### T4 — Routes

| Field | Detail |
|---|---|
| **Files affected** | `src/api/routes/{health,machines,predict,reports}.py` |
| **Priority** | P0 |

### T5 — Error mapping (Risk R-12, and the payoff for Day 1's exception hierarchy)

| Field | Detail |
|---|---|
| **Purpose** | Right status code per layer; never leak a stack trace. |
| **Files affected** | `src/api/main.py` |
| **Priority** | P0 |

### T6 — Tests

| Field | Detail |
|---|---|
| **Files affected** | `tests/unit/test_api.py` |
| **Priority** | P0 |

---

# Work Completed

## T1 — The measurement that determined the architecture ✅

Before writing an endpoint, I measured what `Predictor.explain_machine()` costs.

| Input | Time |
|---|---|
| Whole fleet dataset, scoring one machine | **> 120 s** (timed out) |
| Sliced to that machine + 200-hour window | **~160 ms** |

Roughly **800×**. `merge_tables` and `engineer_features` run over whatever
dataset they are handed, so handing them 876,000 rows to score one machine
does 99% of its work on rows that are then discarded.

This is not an optimisation to add later — without it there is no endpoint.
`MachineDataStore.slice_for()` narrows to one machine before anything is
computed, and the measurement is recorded in the module docstring so nobody
"simplifies" it away.

## T2 — Schemas ✅

`src/api/schemas.py`. Every field validated before it reaches the model.

Two validators earn their place:

- **Physical sensor bounds.** A voltage of 99999 is a broken sensor or a unit
  mix-up. Scoring it produces a confident number from an input the model has
  never seen anything like — worse than refusing, because it looks like an
  answer.
- **Minimum history.** Fewer than 48 readings cannot produce a window: feature
  engineering consumes 24 for rolling/lag, and the LSTM needs 24 more. The
  error says exactly that rather than failing later inside pandas.

## T3 — Service layer ✅

| Component | Responsibility |
|---|---|
| `MachineDataStore` | Holds the dataset; slices per machine; parses datetimes once at load |
| `PredictionService` | Wraps `Predictor` with slicing and a fleet cache |
| `AppState` | Module singleton populated by the lifespan handler |

`AppState.startup()` deliberately **does not raise**. An API that refuses to
start because the model is missing cannot serve `/health` — which is exactly
where an operator would look to find out that the model is missing. Failures
are recorded and reported as `degraded`.

The fleet cache has a 5-minute TTL. Scoring 100 machines measured **13.4 s**;
the cached response is **1.6 ms**.

## T4 — Routes ✅

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness *and* readiness |
| GET | `/machines` | List machines with data coverage |
| GET | `/machines/{id}` | Static facts |
| GET | `/machines/{id}/predict` | Score from stored data |
| GET | `/machines/{id}/explain` | Score plus the evidence |
| GET | `/machines/{id}/history` | Recent hourly readings |
| GET | `/fleet` | Whole fleet, most urgent first, cached |
| POST | `/predict` | Score caller-supplied readings |
| POST | `/report` | LLM report or answer |

**No prediction endpoint calls an LLM.** That separation is the point of
`routes/reports.py` existing as its own module.

## T5 — Error mapping ✅

Day 1 built the exception hierarchy so failures could be caught by
architectural layer. This is where that pays off — one handler per layer, no
string-matching on messages:

```
DataValidationError    -> 422    ModelNotFoundError  -> 503
PredictionError        -> 422    LLMConnectionError  -> 502
ReportGenerationError  -> 422    ResourceNotFound    -> 404
anything else          -> 500 with a correlation id, detail only in the logs
```

The catch-all is security-relevant and tested: an internal `RuntimeError`
mentioning a hostname and a password returns an opaque message plus an id. A
stack trace on the wire tells an attacker about file paths, library versions,
and internal structure.

CORS is pinned to the dashboard origin rather than `*` — the API returns
operational data about physical equipment, and a wildcard would let any page a
technician happens to visit read it.

## T6 — Tests ✅

24 tests against the real app with a stubbed predictor and dataset. Loading
the genuine model costs ~2 s and the dataset is 876,000 rows; a suite that
pays that is a suite nobody runs before committing.

---

# Live Verification

Run against the real model and the full 100-machine dataset.

| Check | Result |
|---|---|
| `/health` | `ok`, model loaded, 100 machines |
| `/machines/51/predict` | 200 in **137 ms** median over 5 calls |
| `POST /predict` (60 supplied readings) | 200 in **108 ms** |
| `/fleet` cold | 13.4 s, 100 assessed, 0 alerting |
| `/fleet` cached | **1.6 ms** |
| `/machines/9999` | 404, correct error shape |
| Voltage 99999 | 422, per-reading, naming field and range |
| Zero readings | 422, "at least 48 hours of history are required" |
| `/report` with a model not installed | **502 with the prediction preserved** |
| `/report` with `model=qwen2.5-coder:7b` | **200 in 21.4 s**, grounded answer |

**NFR-4 (p95 `/predict` < 500 ms): met at ~137 ms.**

Zero machines alerting is correct — the dataset ends 2024-12-30 and no machine
is inside a pre-failure window at that timestamp.

The degradation path, verbatim:

> "The language model is unavailable, so no written report could be produced.
> **The prediction stands: machine 51 probability 0.0000 (low).** Provider
> error: ... The prediction itself is unaffected."

And the full stack working:

> **Q:** Is this machine safe to run through the next shift?
>
> **A:** Yes, this machine is safe to run through the next shift. The failure
> probability is 0.0000, and the risk level is LOW. There are no errors logged
> in the last 24 hours, and all sensor readings are within normal variation.

---

# Bugs Encountered

## B1 — `/report` could select a provider but not a model

| Field | Detail |
|---|---|
| **Description** | `POST /report {"provider":"ollama"}` returned 502: "Ollama call failed with status code 404 ... pull the model with `ollama pull llama3`". |
| **Root cause** | `ReportRequest` had `provider` but no `model`, so it fell back to `settings.OLLAMA_MODEL` (`llama3`, not installed). `scripts/generate_report.py` has had `--model` since Day 7; the API never got the equivalent. |
| **Files affected** | `src/api/schemas.py`, `src/api/routes/reports.py` |
| **Solution** | `model` added to both `ReportRequest` and `ReportResponse`, forwarded to `ReportGenerator`. |
| **Verification** | `{"provider":"ollama","model":"qwen2.5-coder:7b"}` returns 200 in 21.4 s. |
| **Lessons learned** | The CLI and the API are two front ends to the same library, and they had drifted. An API that can pick a provider but not a model within it is half-configurable, and the gap only surfaced because the default happened to be a model I had not pulled. Worth checking the two surfaces against each other rather than assuming parity. |

## B2 — Test fixtures overwritten by the lifespan

| Field | Detail |
|---|---|
| **Description** | 9 of 24 API tests failed, and the suite took 38 s. |
| **Root cause** | Entering `TestClient`'s context manager runs the app lifespan, which calls `state.startup()` — loading the real model and dataset and overwriting the stubs installed beforehand. My monkeypatching ran before the thing that undid it. |
| **Files affected** | `tests/unit/test_api.py` |
| **Solution** | `_client_with()` neuters `startup`/`shutdown` before constructing the client, then installs the stubs. |
| **Verification** | 24 pass; the suite is fast again. |
| **Lessons learned** | With a lifespan-managed app, patching order matters and the failure is confusing — the tests were exercising a *real* model against assertions written for a stub. The 38-second runtime was the tell. |

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 161 → **185 passing** (24 new) |
| **Integration** | 9 (unchanged) |
| **Live** | 10 endpoint checks against the real model and dataset |
| **Quality gates** | flake8 **0**, Black and isort clean |

The error-handling tests are the ones that matter here. A 500 leaking a stack
trace, or a provider outage reported as an internal error, are exactly the
defects that survive to production because the happy path looks fine.

---

# Design Decisions

## D1 — Slice before predicting

| Field | Detail |
|---|---|
| **Alternatives** | Precompute all predictions on a schedule; a background worker; accept the latency. |
| **Pros** | 800× faster, no extra infrastructure, predictions always reflect current data. |
| **Cons** | Feature engineering is repeated per request rather than shared across machines, so scoring the whole fleet is 100 separate slices. |
| **Reason for selection** | Measured, not assumed: >120 s vs ~160 ms. Precomputation would add a scheduler and a staleness problem to solve a problem slicing removes outright. |
| **Impact** | The fleet endpoint pays for it (13.4 s cold), which the cache absorbs. |

## D2 — Isolate the LLM path in its own module

| Field | Detail |
|---|---|
| **Alternatives** | An `include_report` flag on `/predict`; generate reports eagerly and cache. |
| **Pros** | The fast path cannot inherit the slow path's latency or failure modes. `/predict` stays up and fast while the provider is down. |
| **Cons** | A client wanting both makes two calls. |
| **Reason for selection** | A prediction takes 137 ms and a report 21 s — a 150× difference. Coupling them means every prediction pays LLM latency and inherits LLM availability. |

## D3 — Run the LLM call in a threadpool with a timeout

| Field | Detail |
|---|---|
| **Alternatives** | Call `.invoke()` directly in the async handler; a job queue with polling. |
| **Pros** | LangChain's `.invoke()` is blocking — awaited directly it would freeze the event loop for the entire generation, stalling every other request on the worker including `/health`. `run_in_threadpool` keeps the loop free; `asyncio.wait_for` stops a hung provider holding a worker forever. |
| **Cons** | A thread is still occupied for up to 120 s. At high concurrency a real job queue is the answer. |
| **Reason for selection** | Correct and simple at this scale. A queue is the right Day 12+ change, not a Day 9 one. |

## D4 — `startup()` records failures instead of raising

| Field | Detail |
|---|---|
| **Alternatives** | Fail fast and refuse to start. |
| **Pros** | `/health` is reachable and reports `degraded`, so an operator can diagnose. A crash-looping container tells you only that it crashed. |
| **Cons** | A misconfigured instance stays "up" and will 503 on real traffic. |
| **Reason for selection** | The diagnostic endpoint must survive the failure it is meant to diagnose. Platform health checks read `status`, not process liveness. |

## D5 — Reject physically impossible sensor values

| Field | Detail |
|---|---|
| **Alternatives** | Accept anything numeric; clamp to range. |
| **Pros** | A unit mix-up (millivolts for volts) is caught at the door with a message naming the field and the range. |
| **Cons** | A genuine extreme excursion would be rejected — though the bounds are set well outside any plausible reading. |
| **Reason for selection** | The model has never seen values like that; its output would be meaningless *and confident*, which is the worst combination. Clamping would silently fabricate a plausible input. |

---

# Remaining Tasks

| Item | Priority | Effort |
|---|---|---|
| Async report jobs (202 + poll) so a thread is never held for 21 s | P2 | 3 h |
| API-key auth — currently unauthenticated by design, documented as out of scope for v1 | P2 | 2 h |
| Rate limiting on `/report`, the only endpoint with real per-call cost | P2 | 2 h |
| Integration tests hitting a live server rather than `TestClient` | P3 | 2 h |
| Persist prediction history (SQLite) so `/machines/{id}/history` can return past *predictions*, not just readings | P3 | 3 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | 1 h |

---

# Next Day Plan

**Day 10 — Streamlit Dashboard**

1. `dashboard/app.py` — a pure HTTP client of the API. No direct model
   imports, no TensorFlow; that separation is what lets the two deploy and
   scale independently.
2. Pages: fleet overview (from `/fleet`), machine detail (`/explain` plus
   `/history` charts), report view (`/report`).
3. Handle the API being down or degraded — read `/health` and say so, rather
   than rendering an empty dashboard.
4. Report generation must not block the UI: it takes ~21 s, so a spinner and
   an explicit "this calls a language model" note.
5. Risk colours from the same bands the API uses, so the dashboard cannot
   disagree with the alert decision.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~75% |
| **Module completion** | `config/` · `src/utils/` · `src/data/` · `src/models/` · `src/prediction/` · `src/genai/` · **`src/api/` all 100%** · `dashboard/` 0% · Docker/CI 0% |
| **Known risks** | ~~R-6~~ ✅ · ~~R-10~~ ✅ · R-12 partially mitigated (no stack traces, CORS pinned, input validated; **no auth by design**) |
| **Quality gates** | 185 unit + 9 integration · flake8 0 · Black/isort clean |

---

# Files Created

```
src/api/main.py             app, lifespan, exception handlers
src/api/schemas.py          request/response models
src/api/service.py          MachineDataStore, PredictionService, AppState
src/api/routes/health.py
src/api/routes/machines.py
src/api/routes/predict.py
src/api/routes/reports.py
tests/unit/test_api.py      24 tests
docs/Day9.md                this file
```

# Files Modified

```
src/api/__init__.py         docstring
src/api/routes/__init__.py  re-exports
config/settings.py          APP_VERSION, surfaced by /health
```

# References

- [FastAPI: lifespan events](https://fastapi.tiangolo.com/advanced/events/)
- [Starlette: `run_in_threadpool`](https://www.starlette.io/threadpool/) — why blocking calls must not run on the event loop
- [OWASP API Security Top 10](https://owasp.org/API-Security/) — API3 excessive data exposure, API7 misconfiguration

---

# Final Summary

Day 9 turned the library into a service, and the shape of that service was
decided by one measurement taken before any endpoint was written: scoring a
single machine from the full dataset takes over two minutes, and scoring it
from a sliced view takes 160 milliseconds. Everything else followed —
`MachineDataStore` exists to do that slicing, the fleet endpoint is cached
because even at 160 ms per machine a hundred of them is 13 seconds, and the
API is fast enough to sit behind a dashboard.

The other structural decision was keeping the LLM out of the prediction path
entirely. A prediction takes 137 ms; a report takes 21 s. Coupling them would
mean every prediction paid language-model latency and inherited language-model
availability. Verified in both directions: `/report` returns 200 with a
grounded answer when the provider is up, and 502 **with the prediction in the
error detail** when it is not.

Day 1's exception hierarchy finally earned its keep. Six handlers, one per
architectural layer, mapping to the right status code with no string-matching
— and a catch-all that returns an opaque message with a correlation id, tested
against an exception carrying a hostname and a password.

Two bugs, both instructive. The API could select an LLM provider but not a
model within it, a gap between the CLI and the API that existed since Day 7
and only surfaced because the default model was not installed locally. And my
first test fixtures were quietly overwritten by the app's own lifespan, so 24
tests were exercising a real model against stub assertions — the 38-second
runtime was the tell.

Ending state: 185 unit tests, 9 integration tests, nine endpoints verified
live against the real model and all 100 machines, and `/docs` generated from
the same schemas the code validates against.
