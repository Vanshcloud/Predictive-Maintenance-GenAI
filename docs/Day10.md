# Day 10 Summary

| Field | Value |
|---|---|
| **Objective** | A dashboard a maintenance supervisor can use — fleet status, machine detail, AI reports — as a pure client of the API. |
| **Expected outcome** | Streamlit app importing nothing from `src/`, risk colours driven by the API's own risk levels, and every API failure mode rendered as instructions rather than a traceback. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M10 — Dashboard |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `3f259e7` — "docs: record Day 9 and update the plan" |
| **Existing modules** | Everything through `src/api/` complete; 9 endpoints verified live |
| **Tests** | 185 unit + 9 integration, flake8 clean |
| **`dashboard/`** | `.gitkeep` only |

The API works and has OpenAPI docs. That is a product for a developer, not
for the person this project is nominally built for.

---

# Tasks Planned

### T1 — API client

| Field | Detail |
|---|---|
| **Purpose** | All the dashboard's real logic: talking HTTP and interpreting failures. |
| **Files affected** | `dashboard/api_client.py` |
| **Priority** | P0 |

### T2 — Streamlit app

| Field | Detail |
|---|---|
| **Purpose** | Fleet overview, machine detail, AI report. |
| **Files affected** | `dashboard/app.py` |
| **Priority** | P0 |

### T3 — Failure rendering

| Field | Detail |
|---|---|
| **Purpose** | The API being down is the most likely thing a user will hit. |
| **Priority** | P0 |

### T4 — Tests

| Field | Detail |
|---|---|
| **Purpose** | Client logic, and that the pages actually render. |
| **Files affected** | `tests/unit/test_dashboard_client.py`, `tests/unit/test_dashboard_app.py` |
| **Priority** | P0 |

---

# Work Completed

## T1 — API client ✅

`dashboard/api_client.py`. One method per endpoint, plain dicts out.

**Three exception types, not one.** The three ways the API can fail need three
different things on screen:

| Exception | Cause | What the user is told |
|---|---|---|
| `APIUnavailable` | nothing listening, or timeout | "Cannot reach the API. `make run-api`" |
| `APIDegraded` | 200, but `status != ok` | "Model or dataset missing. `python scripts/train_model.py`" |
| `APIError` | 4xx/5xx | The server's own message |

Collapsing them into one error produces a dashboard that says "something went
wrong" to a user who could have been told exactly what to restart.

Two details that matter:

- **The report timeout (150 s) is longer than the API's own ceiling (120 s).**
  A client that gave up sooner would abandon work about to succeed and report
  a failure that did not happen. There is a test asserting the inequality.
- **FastAPI's validation errors use a different shape from ours**, so
  `_describe()` flattens the per-field list rather than putting a nested JSON
  blob on screen.

## T2 — Streamlit app ✅

Three views: fleet overview (metrics, ranked table, risk distribution),
machine detail (evidence table, sensor charts), AI report.

**Risk colours are keyed off the `risk_level` string the API assigned** — never
recomputed from the probability. If the dashboard applied its own thresholds it
could show "medium" for a machine the API is alerting on, and that
inconsistency is trust-destroying, invisible in review, and survives for
months because both halves look individually correct. A test greps the source
for invented threshold constants.

The evidence table renders the API's `is_concerning` verdict rather than
deciding for itself, which keeps Day 7's grounding fix intact all the way to
the screen.

**Report generation is labelled as slow**: "Calling the language model — this
usually takes 20–30 seconds…". A bare spinner on a 21-second call reads as a
hang.

## T3 — Failure rendering ✅

The API being down is the single most likely thing a user hits, and the
dashboard's response to it is most of its perceived quality. Each failure
renders an explanation and the command that fixes it.

The 502 case is special: the API's contract is that an LLM outage preserves
the prediction, so the dashboard says *"The prediction is unaffected — only the
written report could not be produced"* rather than reporting a general error.

## T4 — Tests ✅

**26 new tests**, in two files with different jobs:

- `test_dashboard_client.py` (14) — the logic: failure classification, error
  flattening, request construction, timeouts.
- `test_dashboard_app.py` (12) — that the pages **render**. A Streamlit app
  that throws still serves HTTP 200; the exception surfaces inside the
  session, so "the server started" proves nothing. `AppTest` runs the script
  in-process with a stubbed client and exposes `app.exception`.

---

# Live Verification

Both services started together against the real model and 100 machines:

| Check | Result |
|---|---|
| `GET /health` | `ok`, model loaded, 100 machines |
| `GET http://localhost:8501/` | **HTTP 200** in 2 ms |
| Streamlit startup log | clean — no tracebacks |

The HTTP 200 is necessary but not sufficient, which is exactly why
`test_dashboard_app.py` exists — see B2 below.

---

# Bugs Encountered

## B1 — Substring matching flagged my own docstring

| Field | Detail |
|---|---|
| **Description** | `test_the_app_imports_nothing_from_src` failed on `assert "tensorflow" not in source.lower()`. |
| **Root cause** | The app's docstring says "no TensorFlow" — explaining that it is *not* used. A substring check cannot tell prose from an import. |
| **Solution** | `_imported_modules()` parses the file with `ast` and returns actual top-level import names, checked against a forbidden set (`src`, `config`, `tensorflow`, `keras`, `sklearn`, `joblib`). |
| **Lessons learned** | A test asserting something about code structure should parse the code. The grep version would also have missed `importlib.import_module("tensorflow")` while failing on a comment — wrong in both directions. |

## B2 — `@st.cache_resource` leaked stubs between tests

| Field | Detail |
|---|---|
| **Description** | 4 of 12 render tests failed with empty rendered output — the failure-path assertions found nothing on screen. |
| **Root cause** | `get_client()` is decorated with `@st.cache_resource`, keyed on the API URL. That is correct in production — one client per URL, reused across reruns — but in-process it means the stub installed by the *first* test is handed to every later one. The failure-path tests were silently exercising the happy path. |
| **Files affected** | `tests/unit/test_dashboard_app.py` |
| **Solution** | `run_app()` clears `st.cache_resource` and `st.cache_data` before each run. |
| **Lessons learned** | The tests were **passing for the wrong reason** in the other direction too — the ones that succeeded may have been reading a stub from a previous test. Caching that is correct at runtime is frequently wrong in tests, and the symptom (empty assertions rather than an error) points nowhere near the cause. |

## B3 — `api_client` not importable from the test module

| Field | Detail |
|---|---|
| **Description** | `ModuleNotFoundError: No module named 'api_client'` when patching. |
| **Root cause** | The dashboard is standalone by design and is not a package under `src/`, so its modules are only importable once its directory is on `sys.path`. The app does that for itself at runtime; the test module has to do it to patch. |
| **Solution** | `sys.path.insert(0, str(DASHBOARD))` in the test module, with a comment explaining why it is not an oversight. |

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 185 → **211 passing** (26 new) |
| **Integration** | 9 (unchanged) |
| **Live** | API + dashboard started together, both responding |
| **Quality gates** | flake8 **0**, Black and isort clean |

---

# Design Decisions

## D1 — The dashboard imports nothing from `src/`

| Field | Detail |
|---|---|
| **Alternatives** | Import `Predictor` directly and skip the API; import `src.utils.exceptions` for consistency. |
| **Pros** | The UI and API can be built, deployed, scaled, and restarted independently. Containerising the dashboard does not drag in TensorFlow and a model file to render charts — which is Day 11's problem, made much easier today. |
| **Cons** | The exception types are duplicated in miniature, and the dashboard cannot work offline. |
| **Reason for selection** | The duplication is three tiny classes; the coupling would be a whole ML stack. Two tests assert the boundary by parsing imports, so it cannot erode quietly. |
| **Impact** | The dashboard image will be small and CPU-only. |

## D2 — Risk colours from the API's level, never recomputed

| Field | Detail |
|---|---|
| **Alternatives** | Colour by probability with local thresholds; fetch the band boundaries and apply them client-side. |
| **Pros** | The dashboard *cannot* disagree with the alert decision. |
| **Cons** | The UI cannot offer its own finer-grained banding without an API change. |
| **Reason for selection** | Day 6 pinned `RISK_BAND_HIGH` to the alert threshold for exactly this reason, and recomputing here would reintroduce the drift one layer up. |
| **Impact** | Asserted by a test that greps for invented threshold constants. |

## D3 — Three exception types instead of one

| Field | Detail |
|---|---|
| **Alternatives** | One `APIError` with a status code; let `requests` exceptions propagate. |
| **Pros** | Each failure gets the instruction that resolves it. |
| **Cons** | Three classes for what is arguably one concept. |
| **Reason for selection** | The distinction is not about the error, it is about the remedy — start the API, train the model, or read the server's message. Those are different actions. |

## D4 — Test that pages render, not what they look like

| Field | Detail |
|---|---|
| **Alternatives** | Snapshot the rendered HTML; Selenium; skip UI tests entirely. |
| **Pros** | `AppTest` catches the failure that matters — the page raising — without pinning layout, which changes constantly and would make every cosmetic edit a test failure. |
| **Cons** | A page can render "successfully" while looking wrong. |
| **Reason for selection** | The realistic defect is an exception in a rarely-visited branch, e.g. the degraded-API path nobody exercises by hand. B2 proved the point: four failure-path tests were quietly not testing anything. |

---

# Remaining Tasks

| Item | Priority | Effort |
|---|---|---|
| Auto-refresh the fleet view on a timer | P3 | 1 h |
| Historical *prediction* charts, not just sensor readings (needs API persistence) | P3 | 3 h |
| Multi-machine comparison view | P3 | 2 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | 1 h |

---

# Next Day Plan

**Day 11 — Docker, CI/CD & Deployment**

1. `docker/Dockerfile.api` — multi-stage, `python:3.12-slim`, non-root user.
   Model artifacts mounted as a **volume**, not baked in: rebuilding an image
   to ship a retrained model is the wrong workflow.
2. `docker/Dockerfile.dashboard` — small, because D1 kept TensorFlow out of it.
3. `docker/docker-compose.yml` — both services plus volume mounts, mirroring
   the production topology locally.
4. `.github/workflows/ci.yml` — lint → format-check → typecheck → test → build.
   CI runs against `data/sample/`, which is why that fixture is committed.
5. Run `make test-all` in CI, including the integration tests that are
   excluded from the fast local suite.
6. `pip-audit` for dependency vulnerabilities.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~83% |
| **Module completion** | `config/` · `src/utils/` · `src/data/` · `src/models/` · `src/prediction/` · `src/genai/` · `src/api/` · **`dashboard/` all 100%** · Docker/CI 0% |
| **Known risks** | ~~R-6~~ ✅ · ~~R-10~~ ✅ · R-11 deployment (Day 11) · R-12 partially mitigated (no auth, by design for v1) |
| **Quality gates** | 211 unit + 9 integration · flake8 0 · Black/isort clean |

---

# Files Created

```
dashboard/api_client.py             HTTP client, three failure types
dashboard/app.py                    three-view Streamlit dashboard
tests/unit/test_dashboard_client.py 14 tests
tests/unit/test_dashboard_app.py    12 render tests
docs/Day10.md                       this file
```

# References

- [Streamlit `AppTest`](https://docs.streamlit.io/develop/api-reference/app-testing) — running an app in-process for assertions
- [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching) — and why it needs clearing in tests
- [Altair](https://altair-viz.github.io/) — faceted sensor charts

---

# Final Summary

Day 10 gave the project a face. Three views, all fed by HTTP: fleet status
ranked by urgency, per-machine sensor evidence with the API's own verdicts, and
AI reports generated on demand.

The structural decision was to import nothing from `src/`. The dashboard holds
no model, does no scoring, and knows nothing about TensorFlow — every number on
screen arrived over the wire. That keeps the UI deployable and scalable
separately from the API, and it makes tomorrow's dashboard image small. Two
tests parse the imports to assert the boundary, so it cannot erode quietly.

The second decision was to let the API own what "risk" means. Colours key off
the `risk_level` string it assigns rather than being recomputed from the
probability, because a dashboard showing "medium" for a machine the API is
alerting on is a trust failure that looks correct from either side alone.

Two of the three bugs were in my tests rather than the code, and both were the
same species: a test that passes without testing anything. A substring check
flagged the word "tensorflow" in a docstring that said TensorFlow was *not*
used, and `@st.cache_resource` — correct in production — handed the first
test's stub to every later one, so four failure-path tests were quietly
exercising the happy path. The second is the more instructive: the symptom was
an empty assertion, which points nowhere near the cause.

Ending state: 211 unit tests, 9 integration tests, `dashboard/` complete, and
an API and dashboard verified running side by side against the real model and
all 100 machines.
