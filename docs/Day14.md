# Day 14 — Gate Drift & Accessibility

**Date:** 2026-08-25
**Focus:** Make the local quality gate and CI check the same program; fix two WCAG AA failures in the dashboard.
**Status:** ✅ Complete

---

## Summary

Two audits, one theme: a check that passes is not the same as a check that
means something.

The first half is about **drift between the gate a developer runs and the gate
CI runs**. Four `no-any-return` errors had already reached `main` green-on-a-
laptop, because CI installs only the linters and therefore type-checks a
different program than a full venv does. Chasing that turned up two more copies
of the same bug: CI installed lint tooling **unpinned** (running Black 26.5.1
and mypy 2.3.1 against a tree developed against 24.10.0 and 1.20.2, both outside
the project's own declared bounds), and `make lint` checked three directories
while CI checked five. Every one of these has the same shape — two declarations
of "what gets checked" that were free to diverge, and did.

The second half is the **dashboard's accessibility**, and it produced the more
uncomfortable finding. The risk badge renders white text on the risk colour.
`high` measured **3.19:1** and `medium` **2.94:1** against a 4.5:1 requirement.
Those are the two levels a supervisor scans the fleet table for, and they were
the two that failed. It survived twelve days of review because it is perfectly
legible on a good monitor in a dim room — which is not where a maintenance
supervisor works.

Nothing here is a new feature. Day 14 is entirely about the difference between
looking correct and being correct.

---

## Starting state

- 13 days complete, commit `d8a02ac`, all four CI jobs green.
- 229 unit + 13 integration tests; flake8/Black/isort/mypy clean and blocking.
- Technical debt register: empty.

---

## Tasks planned

1. Pin the lint tooling CI installs, and declare the versions in one place.
2. Make `make typecheck` reproduce CI's type-check conditions.
3. Remove the path drift between the Makefile and the CI workflow.
4. Audit the dashboard against WCAG 2.1 AA and fix what fails.
5. Clear the deprecation warning the API emits on every 422.
6. Update the documentation.

---

## Work completed

### 1. The lint tooling CI installed was unpinned

`ci.yml` ran `pip install black isort flake8 mypy` — no versions. On any given
morning that resolves to whatever released overnight. It had drifted to Black
26.5.1 and mypy 2.3.1 while `requirements-dev.txt` declared `black<25.0.0` and
`mypy<2.0.0`; CI was enforcing a style the project had not agreed to, and a
formatter release could turn the build red with no code change and no way for a
developer to reproduce it locally.

The versions now live in **`requirements-lint.txt`**, which `requirements-dev.txt`
includes with `-r` and CI installs directly. One declaration, two consumers. The
pip cache key moved to the same file so a tooling bump actually busts the cache.

### 2. `make typecheck` and CI's mypy checked different programs

CI's quality job installs only the linters, so mypy cannot see pandas or
langchain. Combined with `ignore_missing_imports = true`, every symbol from them
becomes `Any`, and `warn_return_any` then fires at each third-party seam. That
is a legitimate check — it forces those seams to be annotated — but it is *not*
the check a developer runs against a populated venv, which is how four
`no-any-return` errors reached `main` looking clean.

CI now states this explicitly with `--no-site-packages`, and `make typecheck`
runs **both** modes, so the stricter one is reproducible before pushing. The
four annotations that were missing (`src/genai/chains.py`, `src/genai/assistant.py`)
are in this change.

### 3. `make lint` checked three paths; CI checked five

| | Before | After |
|---|---|---|
| `make lint` / `format-check` | `src/ config/ tests/` | `$(PY_PATHS)` |
| CI quality job | `src/ config/ tests/ scripts/ dashboard/` | `make lint` / `make format-check` |

`make quality` could pass on a commit that CI rejected, on a file the local run
never opened. The paths are now declared once as `PY_PATHS` in the Makefile, and
the CI job **shells out to the make targets** rather than keeping a second copy
of the list — the drift is not fixed, it is made impossible.

Widening the local gate immediately caught a real isort violation in
`src/api/routes/reports.py` that had never been linted locally, which is the
proof the gap was real and not theoretical.

`.PHONY` was also missing five targets (`test-cov`, `format-check`, `typecheck`,
`quality`, `smoke`) — any file with one of those names would silently disable
it — and `make test`'s help text claimed "Run all tests" while `addopts`
deselects integration.

### 4. Two WCAG AA contrast failures in the risk badge

Badge text is 0.8em bold ≈ 12.8px. That is **normal** text under WCAG 2.1, so
the 3:1 large-text allowance does not apply and 4.5:1 is required.

| Level | Was | Ratio | Now | Ratio |
|---|---|---|---|---|
| critical | `#b3202c` | 6.65:1 ✅ | unchanged | 6.65:1 |
| **high** | `#d97706` | **3.19:1 ❌** | `#c2410c` | **5.18:1 ✅** |
| **medium** | `#ca8a04` | **2.94:1 ❌** | `#a16207` | **4.92:1 ✅** |
| low | `#15803d` | 5.02:1 ✅ | unchanged | 5.02:1 |
| fallback | `#6b7280` | 4.83:1 ✅ | unchanged | 4.83:1 |

The red → orange → yellow → green severity ramp is preserved, and all four also
clear the 3:1 that WCAG 1.4.11 asks of the bar chart's fills against the white
page.

Contrast is a number, so it is now checked like one. `TestBadgeContrast` computes
the WCAG relative-luminance ratio for every entry in `RISK_COLOURS` and fails
below 4.5:1 — this exact regression shipped once precisely because it was
eyeballed rather than measured.

### 5. The badge interpolated an unescaped value into raw HTML

`risk_badge()` renders through `unsafe_allow_html=True`. `risk_level` is a
pydantic `Literal` — but only on *our* API. `api_client._request()` returns
`response.json()` with no validation, and the API URL is a **user-editable
sidebar field**, so the string is not ours to trust. One stdlib `html.escape()`.
The colour lookup was already safe: an unrecognised level falls back to grey
rather than reaching the `style` attribute.

### 6. Heading levels skipped h2 on every page

`st.title` renders `h1` and `st.subheader` renders `h3`, so every page went
h1 → h3 with no h2 — a structural failure that axe flags and that makes
heading-based screen-reader navigation misleading. Four `st.subheader` calls
became `st.header`, and the two `### Machine {id}` banners became `##`. These
are the pages' genuine top-level sections, so `h2` is also the semantically
correct level, not merely the one that silences the warning.

**This is a visible change** — those headings render larger. It is the only
change in Day 14 with a visual consequence.

### 7. A deprecation warning on every 422 response

Starlette renamed `HTTP_422_UNPROCESSABLE_ENTITY` to `..._UNPROCESSABLE_CONTENT`
(RFC 9110 dropped "Entity") and deprecated the old spelling, which warns **on
access** — so every 422 the API returned emitted a `DeprecationWarning`.

Reaching for the new name instead would break the lower end of the
`fastapi>=0.110` range in `requirements.txt`, which predates the rename. The
first attempt, `getattr(status, NEW, status.OLD)`, did not work: Python evaluates
the default argument eagerly, so it still touched the deprecated name and merely
moved the warning to import time. The status code is a fixed integer that no
rename can change, so `src/api/schemas.py` binds `HTTP_422 = 422` and the four
call sites use it. Test-suite warnings went 2 → 1.

---

## Deliberately not done

| Item | Why |
|---|---|
| `use_container_width` → `width="stretch"` | Doc-deprecated in Streamlit 1.61 but emits no runtime warning, and the replacement needs ≥1.49 while `requirements-dashboard.txt` declares `>=1.30.0`. Same tension as the 422 constant, without the runtime cost. Do it when that floor next rises. |
| Remaining test warning | `fastapi/testclient.py` importing `httpx`. Not our code; fixed only by upgrading FastAPI. |
| `disallow_untyped_defs` | Unchanged. Still ~100 annotations across test helpers for little safety gain. |

---

## Files changed

| File | Change |
|---|---|
| `requirements-lint.txt` | **New** — pinned lint tooling, single source of truth |
| `requirements-dev.txt` | Includes it with `-r` instead of re-declaring the pins |
| `.github/workflows/ci.yml` | Installs from the pinned file; quality job calls `make lint` / `make format-check`; mypy is explicit with `--no-site-packages`; cache key follows the lint file |
| `Makefile` | `PY_PATHS`; widened lint/format/format-check; `typecheck` runs both mypy modes; `.PHONY` completed; `test` help corrected |
| `src/genai/chains.py`, `src/genai/assistant.py` | Four missing return annotations (`no-any-return` under CI conditions) |
| `src/api/schemas.py` | `HTTP_422` constant |
| `src/api/main.py`, `src/api/routes/reports.py` | Four call sites use it |
| `dashboard/app.py` | Accessible risk palette; `html.escape` in `risk_badge`; heading levels |
| `tests/unit/test_dashboard_app.py` | `TestBadgeContrast` — 3 tests |
| `.gitignore` | `uv.lock` — a stray 155-byte artifact resolving no dependencies; nothing here uses uv |

---

## Project health

| Check | State |
|---|---|
| Unit tests | **232 passing** (229 + 3) |
| Integration tests | 13 passing |
| flake8 / Black / isort | Clean across **five** paths, locally and in CI |
| mypy | 0 errors across 29 files, in **both** modes |
| Test-suite warnings | 1 (third-party, unfixable here) |
| Technical debt | Empty |

---

## Final summary

Every defect found today had been in the repository for days, behind a green
build. None was found by a failing test; all were found by asking what the green
actually proved.

The pattern repeats at both ends of the day. CI's mypy and the developer's mypy
printed the same "Success" for different programs. `make quality` and the CI
quality job printed the same "clean" for different file sets. A risk badge that
reads perfectly on the machine it was designed on is unreadable in the workshop
it was designed for. In each case the check was real, the result was true, and
the conclusion drawn from it was wrong.

The fixes are correspondingly small — one variable, one include file, one flag,
two hex values. **What changed is not the code's behaviour but what its passing
checks are entitled to claim.**
