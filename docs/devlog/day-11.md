# Day 11 Summary

| Field | Value |
|---|---|
| **Objective** | Package both services as containers and put the quality gate in CI. |
| **Expected outcome** | Two Dockerfiles, a compose topology, and a GitHub Actions pipeline that lints, tests, audits, and builds. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M11 — Deployment |
| **Status** | ✅ Complete — images built and verified (see Verification below) |

---

## Verification (added after Docker was installed)

The images were originally written and only *statically* validated — Docker was
not available on the machine at the time, and this section said so. Docker was
installed later the same day and everything below was then run for real.

| Check | Result |
|---|---|
| Dashboard image builds | ✅ **803 MB** |
| API image builds | ✅ **2.87 GB** (3.23 GB before the fix in B2) |
| `docker compose up` | ✅ both containers healthy |
| Startup ordering | ✅ `api Started → Waiting → Healthy` **then** `dashboard Starting` |
| API `/health` in container | ✅ `ok`, model loaded, 100 machines |
| Prediction in container | ✅ 200 in 341 ms, value **byte-identical** to the host run |
| Dashboard in container | ✅ HTTP 200, `/_stcore/health` returns `ok` |
| API with **no model mounted** | ✅ `degraded`, and `/predict` returns **503** with the command to fix it |

That last row is the behaviour `AppState.startup()` was deliberately made
non-raising for on Day 9, and this is the first time it was exercised in a
container rather than argued for.

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `f265f41` — "docs: record Day 10 and complete the application layers" |
| **Modules** | Every layer complete through `dashboard/` |
| **Tests** | 211 unit + 9 integration, flake8 clean |
| **`docker/`, `.github/`** | `.gitkeep` only / absent |

---

# Work Completed

## T1 — Verified the suite survives a fresh clone ✅

Before writing CI, I checked the assumption it depends on: that the tests pass
without the gitignored 5 GB of data and the trained model. Hid both and re-ran:

```
193 passed, 18 skipped, 9 deselected
```

The 18 skips are `test_predictor.py`, whose module fixture skips on
`ModelNotFoundError`. Integration tests skip on missing data. **This is exactly
the state CI sees**, and knowing it in advance is why the workflow does not
need to fabricate a model.

## T2 — Requirements split ✅

`requirements-dashboard.txt` — four packages: `streamlit`, `altair`, `pandas`,
`requests`.

Verified by parsing the dashboard's imports with `ast`: those four are exactly
what it uses. Installing `requirements.txt` into the dashboard image would pull
TensorFlow, scikit-learn, and LangChain — roughly 1.5 GB of wheels — to render
charts and call `requests`.

This is what turns Day 10's import boundary from a stylistic preference into a
deployment saving.

## T3 — Dockerfiles ✅

Both multi-stage, both `python:3.12-slim`, both non-root.

| Decision | Why |
|---|---|
| Multi-stage | `build-essential` lives only in the builder and is discarded. A compiler in a production image is dead weight *and* a useful tool for anyone who gets a shell. |
| Python 3.12 pinned | TensorFlow has no 3.13+ wheels — the same constraint that forced a venv rebuild on Day 1. Pinning stops a base-image bump breaking the install silently. |
| Requirements copied before source | Editing a source file must not invalidate the very slow TensorFlow install layer. |
| Non-root `appuser` | A container that does not need to write to its own filesystem should not be able to. |
| Artifacts **not** baked in | A retrained model should be a volume swap and a restart, not an image rebuild. The processed tensors are ~5 GB and have no business in a layer. |
| Healthcheck hits `/health`, not a TCP port | The process can be listening while the model failed to load; `/health` reports `degraded` in exactly that case and a TCP probe would call it healthy. |

The dashboard image copies **only `dashboard/`**. If a future change makes the
UI import from `src/`, that build breaks — which is the intended failure.

## T4 — `.dockerignore` ✅

The single highest-impact file of the day. Everything not excluded is uploaded
to the daemon as build context *before any instruction runs*.

| | Size |
|---|---|
| Repository | **7.3 GB** |
| Build context after `.dockerignore` | **2.9 MB** |

Without it, every build would copy `venv/` (2.3 GB), `data/` (5 GB) and `.git`
to produce images that need none of them. `data/sample/` is deliberately *not*
excluded — it is small, committed, and what the tests run against.

## T5 — Compose topology ✅

Two services, separate images, talking over HTTP. Running them together locally
is what proves the dashboard really is a client rather than a co-process.

Two details that matter:

- **`depends_on: condition: service_healthy`** — waits for `/health` to report
  ok, not merely for the process to start. The model takes ~2 s to load and the
  dataset longer; without this the dashboard's first render races startup.
- **`API_BASE_URL=http://api:8000`** — service name, not localhost. See B1.

## T6 — CI pipeline ✅

Four jobs:

| Job | Does | Blocking? |
|---|---|---|
| `quality` | flake8, black, isort, mypy | flake8/black/isort yes; mypy advisory |
| `test` | unit + integration + coverage | yes |
| `security` | `pip-audit` | advisory |
| `docker` | build both images, report sizes, smoke-test the API | yes, after quality+test |

The `quality` job installs only the linters — it does not need TensorFlow, and
skipping it turns a multi-minute install into seconds.

The docker job's smoke test is the interesting one: it runs the API image with
**no model or dataset mounted** and asserts `/health` returns `degraded`. That
is precisely the behaviour `AppState.startup()` was written for on Day 9 — not
raising, so the diagnostic endpoint survives the failure it diagnoses — and
this is the only place it gets exercised for real.

It also reports both image sizes. If the gap between them closes, something has
started importing the ML stack into the UI.

---

# Bugs Encountered

## B1 — The dashboard hardcoded `localhost:8000`

| Field | Detail |
|---|---|
| **Description** | Writing the compose file surfaced it: inside a container, `localhost:8000` is the *dashboard itself*, not the API. The packaged app could never have reached anything. |
| **Root cause** | Day 10 built the dashboard against a locally-running API and hardcoded the default. Correct for development, broken for every deployment. |
| **Files affected** | `dashboard/app.py`, `dashboard/api_client.py` |
| **Solution** | Both read `API_BASE_URL` from the environment, defaulting to localhost. Compose sets `http://api:8000`. |
| **Lessons learned** | Containerisation is a good audit of hidden environment assumptions. This one was invisible while everything ran on one host, and no test would have caught it — the tests inject a base URL explicitly. |

## B2 — The API image shipped Streamlit

| Field | Detail |
|---|---|
| **Description** | The API image installed Streamlit 1.62.0 — a web UI framework nothing in `src/` imports. Confirmed by running `python -c "import streamlit"` inside the built image. |
| **Root cause** | `requirements.txt` bundled all seven layers including the dashboard's. Day 11 split the *dashboard's* dependencies out into their own file and left the API carrying everything — the same problem, unfixed in the other direction. |
| **Files affected** | `requirements.txt`, `requirements-dev.txt` |
| **Solution** | `requirements.txt` is now API-only; `requirements-dev.txt` pulls in both stacks so local development is unchanged. |
| **Measured** | **3.23 GB → 2.87 GB, a 360 MB (11%) saving.** Verified Streamlit is absent and TensorFlow, FastAPI and LangChain still import. |
| **Lessons learned** | Static validation could not have found this. Both images were *correct* — they just were not *minimal*, and nothing about a Dockerfile reveals that its requirements file is a superset of what it needs. It took an actual build reporting an actual number. The general form: "I solved this problem once" is not the same as "I solved this problem in every direction it occurs." |

## B3 — mypy reports 159 errors

| Field | Detail |
|---|---|
| **Description** | `mypy src/ config/` finds 159 errors across 14 files. |
| **Root cause** | Mostly loguru: `logger` is dynamically constructed, so `logger.info` reads as `logger? has no attribute "info"`. The rest is genuinely missing annotations. |
| **Solution** | mypy runs in CI but is **advisory** (`|| true`). Failing the build on it would block real work; hiding it would let the debt disappear. |
| **Lessons learned** | The Day 1 claim that mypy was set up "strict-mode ready" was optimistic — it was configured, never run against a full codebase. Same pattern as `make quality` never being run until Day 4. Recorded as TD-8 rather than quietly dropped. |

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | **211 passing** (unchanged — today was packaging) |
| **Fresh-clone simulation** | 193 pass, 18 skip, with model and data hidden |
| **Compose YAML** | parses; services, ports, volumes, healthcheck all correct |
| **CI workflow YAML** | parses; 4 jobs, dependencies correct |
| **Build context** | 7.3 GB → 2.9 MB |
| **Image contents** | API copies `config/ src/ scripts/`; dashboard copies only `dashboard/` |
| **Docker build** | ✅ both images built; compose stack verified end to end |

---

# Design Decisions

## D1 — Mount artifacts as volumes, never bake them in

| Field | Detail |
|---|---|
| **Alternatives** | `COPY models/ /app/models/`; download from object storage at startup. |
| **Pros** | Retraining is a volume swap and a restart. The image stays small and stays valid across model versions. |
| **Cons** | The image is not self-contained — running it requires the operator to supply artifacts, and a misconfigured mount means a degraded API rather than a loud failure. |
| **Reason for selection** | Rebuilding an image to ship a retrained model is the wrong workflow, and the processed tensors are 5 GB. The degraded-startup path exists to make a missing mount visible via `/health`. |

## D2 — Separate requirements for the dashboard

| Field | Detail |
|---|---|
| **Alternatives** | One requirements file for both images. |
| **Pros** | The dashboard image excludes TensorFlow, scikit-learn and LangChain entirely. |
| **Cons** | Two files to keep in sync; a dashboard dependency added to only one will fail at build time rather than import time. |
| **Reason for selection** | It is the payoff for Day 10's boundary. A single file would make that boundary purely decorative. |

## D3 — mypy and pip-audit advisory, lint and tests blocking

| Field | Detail |
|---|---|
| **Alternatives** | Everything blocking; everything advisory. |
| **Pros** | The gates that are currently green stay green; the ones that are not stay visible without blocking a docs commit. |
| **Cons** | An advisory check is one nobody reads. The 159 mypy errors could sit there indefinitely. |
| **Reason for selection** | A red build that is always red teaches people to ignore red builds. Recorded as TD-8 with a plan (annotate incrementally, tighten per-module) rather than left as a permanent `|| true`. |

## D4 — Smoke-test the *degraded* path in CI

| Field | Detail |
|---|---|
| **Alternatives** | Mount a model in CI and test the happy path; skip the container test. |
| **Pros** | Tests a real behaviour with no fixtures: with nothing mounted, the API must start and report `degraded`. CI has no trained model, so this is the only container test it *can* run — and it happens to cover the failure mode that matters operationally. |
| **Cons** | The happy path is never exercised in a container. |
| **Reason for selection** | Day 9 deliberately made `startup()` non-raising so `/health` survives a missing model. Until now nothing verified that in a container. |

---

# Remaining Tasks

| Item | Priority | Effort |
|---|---|---|
| **TD-8** — reduce mypy's 159 errors; tighten per-module and make it blocking | P2 | 4 h |
| Push images to a registry and deploy to a free-tier host | P2 | 3 h |
| API-key auth before any public deployment (currently unauthenticated by design) | P2 | 2 h |
| Pin exact versions (`pip-compile`) for reproducible image builds | P3 | 1 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | 1 h |

---

# Next Day Plan

**Day 12 — Final Polish, Docs & Demo**

1. **Confirm CI is green** and record the real image sizes — the outstanding
   verification from today.
2. End-to-end run from a clean checkout, following the README exactly. That
   doubles as validation of the installation instructions.
3. README pass: architecture diagram, results, screenshots.
4. Retire or fold `docs/handoff.md` (TD-4), the last piece of documentation
   debt.
5. A `docs/RESULTS.md` consolidating the metrics that are currently spread
   across day files.
6. Final review of the technical-debt register: repay what is cheap, and state
   plainly what is being left.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~93% |
| **Module completion** | Every layer 100%; Docker images built and verified; CI unrun until first push |
| **Technical debt** | TD-4 (handoff overlap) · **TD-8 (159 mypy errors, advisory)** |
| **Known risks** | ~~R-6~~ ✅ · ~~R-10~~ ✅ · ~~**R-11 deployment**~~ ✅ **closed — both images build and run** · R-12 (no auth, by design) |
| **Quality gates** | 211 unit + 9 integration · flake8 0 · Black/isort clean · mypy advisory |

---

# Files Created

```
docker/Dockerfile.api             multi-stage, non-root, healthcheck on /health
docker/Dockerfile.dashboard       four packages, no ML stack
docker/docker-compose.yml         two services, volumes, health-gated startup
.dockerignore                     7.3 GB -> 2.9 MB build context
requirements-dashboard.txt        streamlit, altair, pandas, requests
.github/workflows/ci.yml          quality, test, security, docker
day-11.md                     this file
```

# Files Modified

```
dashboard/app.py          reads API_BASE_URL from the environment
dashboard/api_client.py   same default, so both entry points agree
Makefile                  docker-build / docker-up / docker-down
```

# References

- [Docker: multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [Compose: `depends_on` with `service_healthy`](https://docs.docker.com/compose/how-tos/startup-order/)
- [GitHub Actions: `docker/build-push-action`](https://github.com/docker/build-push-action)
- [pip-audit](https://pypi.org/project/pip-audit/)

---

# Final Summary

Day 11 packaged the project. Two images rather than one, because Day 10's
decision that the dashboard imports nothing from `src/` finally pays off: the
UI installs four packages where the API installs the whole ML stack, and the
CI job reports both sizes so that gap stays visible.

The highest-impact file was the smallest. Without `.dockerignore`, every build
would ship 7.3 GB of build context — `venv/`, 5 GB of generated data, `.git` —
to produce images that need none of it. With it, 2.9 MB.

CI runs the gate that Day 4 discovered had never been run at all: `make
quality` was configured on Day 1 and first executed on Day 4, by which point 14
files had drifted and flake8 reported 51 issues. It now runs on every push,
with lint and tests blocking, and mypy and pip-audit advisory — because a build
that is permanently red teaches people to ignore red builds. mypy's 159 errors
are recorded as TD-8 rather than hidden.

Writing the compose file found the one real bug: the dashboard hardcoded
`localhost:8000`, which inside a container points at the dashboard itself. No
test would have caught it — they inject a base URL explicitly — and it was
invisible while everything ran on one host. Containerisation is a good audit of
environment assumptions.

This document originally carried a caveat at the top: Docker was not installed,
so nothing had been built. Docker was installed later the same day, and the
caveat has been replaced with measurements — both images build, the compose
stack comes up healthy in the right order, a containerised prediction returns a
value byte-identical to the host run, and an API container with no model
mounted reports `degraded` and refuses predictions with a 503 that names the
fix.

Building them found one thing static validation could not. The API image was
shipping Streamlit: `requirements.txt` bundled every layer, so the service
installed a web UI framework nothing in `src/` imports. I had split the
*dashboard's* dependencies out on Day 11 and left the API carrying everything —
the same problem, unfixed in the other direction. Removing it cost 360 MB, and
nothing short of a real build would have surfaced it, because both images were
correct and only one of them was minimal.
