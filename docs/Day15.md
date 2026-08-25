# Day 15 — Production Review

**Date:** 2026-08-25
**Focus:** A full-repository staff-engineer audit before publishing; fix everything it found.
**Status:** ✅ Complete

---

## Summary

Day 14 ended with every gate green, so this session went looking for the things
a gate cannot see. It found two real defects, and they sit at opposite ends of
the stack.

The first is that **training was never reproducible.** There is no
`tf.random.set_seed` anywhere in the repository and never has been. LSTM kernels
came from an unseeded `glorot_uniform`, dropout masks from an unseeded RNG, and
batch shuffling was the only thing that was seeded at all. Two runs of
`scripts/train_model.py` on byte-identical data produce different weights and a
different test F1 — while `README.md`, `CLAUDE.md` and `docs/RESULTS.md` all
quote **0.8949** as a fact about this repository. Nobody could have checked
that number, including me. Fourteen days of careful work on temporal splits and
train-only scaling, guarding against leakage that would inflate the metric, and
the metric itself was not reproducible.

The second is an **unbounded cache on a public endpoint.** `/fleet` is cached by
`as_of`, which is a caller-supplied query parameter, and nothing ever evicted:
the 5-minute TTL was consulted only on a *hit*, so an expired entry stayed
resident forever and every distinct timestamp added another ~100 prediction
records permanently. This was not a theoretical attack — the dashboard's rewind
control is a date picker plus an hour slider, so ordinary use walks thousands of
distinct keys in one session and the API process grows without bound.

Both had the same thing in common as Day 14's findings: they were invisible
because nothing was measuring them. A metric quoted in three files is not the
same as a metric anyone has reproduced, and a cache with a TTL is not the same
as a cache that evicts.

---

## Starting state

- 14 days complete, commit `c5b2974`. CI green (run #13), working tree clean.
- 233 unit + 13 integration tests, `make quality` clean in both mypy modes.

---

## 1. Training was not reproducible

`scripts/train_model.py` now takes `--seed` (default 42, matching
`scripts/generate_data.py`'s existing convention) and calls
`keras.utils.set_random_seed(seed)` **before the model is built** — which is
when the kernels are drawn. One call covers all three generators Keras reaches
for (Python `random`, NumPy, TensorFlow); seeding them individually is the same
thing with three more places to forget one.

`tf.config.experimental.enable_op_determinism()` was deliberately **not**
enabled. It would additionally pin down non-deterministic GPU reductions, but it
disables the fused cuDNN LSTM path at several times the training cost, and this
model trains on CPU.

Two tests in `tests/unit/test_model.py` assert both directions — same seed gives
identical initial weights, and a *different* seed gives different ones. Only the
first would pass a change that pinned every run to one arbitrary draw.

Note on the import: `from tensorflow import keras` sits in the third-party block
above `from src.models import ...`, which is where isort put it. That placement
is correct rather than merely tolerated — it drags TensorFlow in before
`src.models` pulls in sklearn and therefore Arrow, which is the abseil ordering
`docs/Day4.md` exists to protect. Only `numpy` precedes it, and it links no
abseil of its own.

## 2. The fleet cache grew without bound

`PredictionService._fleet_cache` is now an `OrderedDict` capped at 16 entries,
evicting least-recently-used on insert, with `_fleet_cached_at` evicted
alongside it — otherwise the leak simply moves from one dict to the other. A
cache *hit* also refreshes recency, so the view an operator keeps returning to
is not evicted on age-of-insertion alone; re-inserting it would re-pay the ~16 s
fleet scoring cost.

Sixteen covers the rewinding a person actually does while keeping the ceiling
fixed and small. Three tests in `tests/unit/test_api.py` cover the bound, the
eviction order, and the LRU refresh.

## 3. `scripts/plot_horizon.py` failed badly on every unhappy path

Day 14 shipped this script in a hurry and it only ever ran against a warm,
fully-loaded API. Four separate ways it broke, all now fixed:

- **It read `/health` but ignored `status`.** `/health` answers 200 whether or
  not the model loaded — "degraded" is a field, not a status code, and that is
  deliberate, because an operator needs the check to *answer* in order to learn
  the model is missing. On a fresh clone `models/` and `data/` are gitignored and
  merely mounted, so degraded is the normal first-run state — exactly the reader
  the README's "See it work" section is written for. The friendly "cannot reach
  the API" branch was never taken and the first scoring call died on an
  unhandled 503 traceback.
- **`score()` had no error handling at all.** An unknown `--machine` (404) or a
  window that starts before the data does (503) aborted mid-loop with a raw
  traceback, after already printing twenty-odd hourly rows and writing no chart.
  The window is now validated up front against the `data_start`/`data_end` the
  script already fetches, and a mid-run HTTP error reports which hour failed and
  why.
- **`as_of` was interpolated into the query string unencoded.** A timezone-aware
  `--failure` emits a literal `+`, which decodes to a space server-side, so
  every point 422s. `urllib.parse.quote` fixes it.
- **The crossing annotation was positioned at a hardcoded `crossing - 21`** while
  the x-limits follow `--hours`. At `--hours 24` the label landed at x=−36,
  outside the axes, and rendered off-figure. It is now offset as a fraction of
  the axis and clamped to the left edge. The committed 48h chart is unchanged.

## 4. Documentation that had gone stale

- `README.md` carried **three contradictory test counts** in one file: the badge
  said 233, the status line said 229, and the commands block said 75.
- `CLAUDE.md` said 232.
- `docs/Day14.md`'s measurement table skipped from −15h to −13h. The −14h row is
  now present and **measured** (0.9996) rather than interpolated — the first
  value written into it was a guess of 0.9999, which was wrong.
- `README.md` claimed the chart could be reproduced "in one command with the
  stack running". It could not: `make docker-up` has no `-d`, so it holds the
  terminal and the next line never runs; the stack additionally needs
  `generate_data.py` → `run_preprocessing.py` → `train_model.py` first, since
  `data/` and `models/` are gitignored; and the script needs matplotlib, which
  only `requirements-dev.txt` installs. The section now lists the real sequence,
  and `make docker-up-d` was added for the detached case.

All counts are now 238 unit + 13 integration, which is what `pytest` collects.

## 5. Text I/O assumed a UTF-8 locale

Twelve `open()` / `read_text()` / `write_text()` calls relied on Python's
default encoding, which is the *locale's* — cp1252 on a stock Windows, UTF-8
everywhere this project has ever run. Two of them are real crashes rather than
theoretical ones:

- `tests/unit/test_dashboard_app.py` and `tests/unit/test_dashboard_client.py`
  read `dashboard/*.py` as source text, and those files are full of em-dashes.
- `scripts/generate_report.py` writes **LLM output** to a file, which routinely
  contains characters cp1252 cannot encode (`—`, `°`, `µ`).

All twelve now pass `encoding="utf-8"` explicitly. No behaviour changes on
macOS or Linux; the suite is simply runnable on Windows now.

---

## What was reviewed and found sound

Worth recording, because a review that only lists faults implies the rest was
not looked at.

- **ML pipeline.** Temporal split before scaling before windowing; `StandardScaler`
  fit on train only and applied to val and test; validation split drives early
  stopping and checkpoint selection, with the test set scored exactly once and a
  loud warning if it is ever substituted. No leakage found.
- **Security.** No secrets committed; `.env` ignored with `!.env.example` kept;
  CORS restricted to the dashboard origin rather than `*`; API containers run as
  a non-root user from a multi-stage build with no compiler in the runtime layer;
  artifacts mounted read-only. No bare `except:` anywhere; every broad
  `except Exception` is at a genuine boundary.
- **`.gitignore`.** Correct, and unusually well-reasoned — `metrics.json` and
  `training_history.json` are deliberately *not* ignored because they are the
  auditable evidence of what a commit's model achieved.
- **Docker.** Healthcheck hits `/health` and asserts `status == "ok"` rather than
  probing the TCP port, so a container with a failed model load is correctly
  reported unhealthy; the dashboard waits on `service_healthy`, not on start.
- **CI.** Pinned lint tooling, `PY_PATHS` as the single declaration of what gets
  checked, mypy run in both site-package modes, and a smoke test that asserts the
  API starts *degraded* with no model mounted.

---

## Deliberately not changed

- **`fit()` vs the hand-written `GradientTape` loop.** Still not re-benchmarked
  since the abseil fix, and still working, tested, and explicit about class
  weighting. Out of scope for a review.
- **The CI `test` job runs the suite twice** (once bare, once under coverage),
  costing roughly 30 seconds. Merging them would put test failures under a step
  named "Coverage". Left alone.
- **No rate limiting on the API.** The bounded cache removes the unbounded-memory
  consequence; a request-rate limit is a deployment concern, not an application
  one, and there is no deployment.

---

## 6. The retrain

Seeding is only half of it. The model in `models/` was trained *before* the seed
existed, so the committed `metrics.json` was still an unreproducible artifact
and the 0.8949 in the README still could not be checked by anyone. Retrained
with `--seed 42` and re-evaluated; the whole chain moved with it.

Early stopping at epoch 28 of 30, best weights from epoch 23 (`val_f1` 0.9602).
81 minutes, CPU-only. Then `scripts/evaluate_model.py` re-swept the threshold on
validation.

| | Day 5 (unseeded) | Day 15 (`--seed 42`) |
|---|---|---|
| ROC-AUC | 0.9997 | **0.9999** |
| Precision | 0.8756 | **0.8976** |
| Recall | 0.9150 | **0.9200** |
| **Test F1** | 0.8949 | **0.9086** |
| Missed | 17 | **16** |
| False alarms | 26 | **21** |
| Deployed threshold | 0.6678 | **0.3415** |
| Reproducible | no | **yes** |

It is better across the board, but that is luck, not tuning — the only change
was the seed, which happened to draw a start that trained 28 epochs to a best at
23 instead of stopping at 20. **The improvement is not the point.** The point is
the last row: every figure this project has published for fourteen days
described a model nobody could rebuild, including me.

Three things had to move with the threshold:

- **`PREDICTION_THRESHOLD` 0.6678 → 0.3415**, and `RISK_BAND_HIGH` with it —
  they are asserted equal by `tests/unit/test_predictor.py`, because "high or
  above" must mean exactly "the model is alerting".
- **`RISK_BAND_MEDIUM` 0.30 → 0.15.** Not cosmetic: left at 0.30 it would have
  made "medium" a four-point sliver between 0.30 and 0.3415, a band almost
  nothing can land in, which tells a technician nothing. 0.15 holds medium at
  ~44% of the alert threshold — the proportion 0.30 held against 0.6678.
- **`docs/images/horizon.png` regenerated.** Machine 51 now crosses at **−16h**
  rather than −15h, and the climb is sharper: 0.0005 at −17h to 0.9943 at −16h.

**`8/8 failure events caught` was re-verified, not carried over.** It had no test
behind it anywhere in the tree — it came from a one-off Day 6 script run. Scoring
all eight held-out failure events hour by hour through the API confirms the
retrained model warns on every one, with a median of 23.5 h of lead time and a
worst case of 16 h (machine 51, the machine in the chart).

`docs/Day5.md` and this file's §4 table are left as they were measured. They are
session records, not current-state claims; `docs/RESULTS.md` and
`IMPLEMENTATION_PLAN.md` carry the current numbers and now say which model they
describe.

---

## 7. Housekeeping

`CLAUDE.md` → **`AGENTS.md`**. The repository is a portfolio piece and the file
sat at the top of its root listing named after one vendor's tool; `AGENTS.md` is
the cross-tool convention and says what the file is rather than who reads it.
Six current-state references updated (README twice, the plan twice,
`docs/README.md`, and two source comments). References inside Day reports are
left alone — they record the name the file had at the time.

Unrelated but discovered alongside it: GitHub's Contributors sidebar was still
showing two people after Day 14's history rewrite. The commit data was already
correct — `/contributors?anon=1` and all 42 commits on `main` return one author
— but the sidebar renders from a separately-computed statistics cache that had
not been regenerated. Requesting `/stats/contributors` returned `202` twice
(GitHub computing) and then `200` with a single entry. Worth knowing: after a
history rewrite the API tells the truth immediately and the sidebar does not.

**README's progress section was wrong.** It claimed "Complete — 12 of 12 days"
above a list of thirteen, and the Day 15 line added earlier in this session had
been inserted between Day 6 and Day 7 rather than at the end. Now split into
"The build — 12 of 12 days" and "After the build" (Days 13–15), and two document
index ranges that still stopped at Day 12 and Day 13 extended to Day 15.

---

## Verification

```
make quality      # flake8, black, isort across 5 paths; mypy in both modes — clean
pytest tests/unit # 238 passed
```

Both error paths and both render paths of `plot_horizon.py` were exercised
against the running stack: API down, window before the data starts, unknown
machine, `--hours 24`, and the default 48h.
