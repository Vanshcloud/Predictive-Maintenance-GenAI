# Day 12 Summary

| Field | Value |
|---|---|
| **Objective** | Verify the project works from a clean checkout, consolidate the results, and close out the documentation debt. |
| **Expected outcome** | A stranger can clone, follow the README, and reach a working state; every metric in one place with its caveats; the debt register honest about what remains. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M12 — Final polish & demo |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `ac65aca` — "fix(docker): stop shipping Streamlit in the API image" |
| **Modules** | Every layer complete; images built and verified |
| **Tests** | 211 unit + 9 integration, flake8 clean |
| **Debt** | TD-4 (`handoff.md` overlap), TD-8 (159 mypy errors) |

---

# Work Completed

## T1 — Clean-checkout verification ✅

The claim a portfolio project lives or dies on: *can a stranger run this?*
Tested rather than assumed — cloned to a fresh directory and followed the
README's setup section **verbatim**, no shortcuts from knowing the codebase.

| Step | Result |
|---|---|
| `git clone` | ✅ **3.9 MB**, 103 tracked files |
| `python3.12 -m venv venv` | ✅ |
| `pip install -r requirements-dev.txt` | ✅ |
| `cp .env.example .env` | ✅ |
| `python -m pytest tests/` | ✅ **193 passed, 18 skipped in 20.5 s** |
| `python scripts/generate_data.py` | ✅ 883,231 rows across 5 tables |

The 18 skips are the tests needing a trained model, which a fresh clone does
not have. They skip rather than fail — that behaviour was built for CI and it
is what makes the first-run experience clean.

### Reproducibility, proven rather than claimed

NFR-1 says the dataset is byte-identical from a fixed seed. Generated it in the
clean clone and compared to the original:

```
clean clone : 37587935cd98e11095a444549e2b774d
this repo   : 37587935cd98e11095a444549e2b774d
```

Same MD5 on a 43 MB file. Every number in `docs/RESULTS.md` is therefore
reproducible by anyone who runs the four documented commands.

## T2 — `docs/RESULTS.md` ✅

The metrics were spread across eight day files, each correct in context and
none of them a place to *look up* a number. `RESULTS.md` consolidates them:
dataset, splits, architecture, test performance, event-level recall, serving
latency, parity, container sizes, tests.

Two sections matter more than the tables. **"How the numbers moved, and why"**
shows F1 going 0.7530 → 0.8605 → 0.8949 across the split fix and the monitor
change, so a reader sees that the improvement came from methodology rather than
tuning. **"What these numbers do not establish"** lists the five things a
sceptical reader should hold against them — synthetic data, eight test events,
a threshold that barely transferred, the cost cliff, and no auth.

## T3 — TD-4 closed: `docs/handoff.md` removed ✅

1,377 lines, superseded section by section by `IMPLEMENTATION_PLAN.md`, with
its Day 1–3 detail already living in the reconstructed `Day1.md`–`Day3.md`.

It had carried a "superseded — read the plan instead" banner since Day 4, which
is its own kind of clutter: 60 KB whose only message is *read something else*.
Deleted, with git preserving it (`git show 9ceb349:docs/handoff.md`) and
`docs/README.md` recording where it went and why. Live references in
`CLAUDE.md`, the plan, and both READMEs were updated; the historical mentions
inside `Day1.md`–`Day3.md` ("reconstructed from handoff.md") were left alone,
because they describe how those documents were written and remain true.

## T4 — README pass ✅

Two changes, both about what a reader meets first:

- **An "at a glance" box** above the fold: what it does, what it achieved
  (8/8 events, 26 false alarms, 137 ms), and that it runs with no API key.
  Previously a reader had to get through the setup instructions before finding
  a single result.
- **A real architecture diagram** replacing the generic arrow sketch — showing
  the actual layering, the verified parity claim, and why `/report` is isolated
  from `/predict`.

## T5 — Final debt review ✅

See the register below. The honest summary: one item repaid today, one left
open with a plan, and several deliberate v1 omissions that are documented as
decisions rather than oversights.

---

# CI — verified green

This section originally read "what could not be verified": the repository was
private, so the GitHub API returned `Not Found`. It was made public afterwards
and the pipeline was then observed.

**Run `458ce03` — all four jobs passed.**

| Job | Result |
|---|---|
| Dependency audit | ✅ success |
| Lint, format, types | ✅ success |
| Tests | ✅ success |
| Build images | ✅ success |

`Lint, format, types` passing confirms flake8/Black/isort agree on CI's Linux
runner rather than only on macOS. `Build images` passing means both Dockerfiles
build on **amd64** — they had only ever been built locally on arm64 — and the
smoke test genuinely executed, confirming the API container starts `degraded`
with no model mounted instead of crashing.

## One bug found before CI found it

Auditing the workflow while the first run was queued turned up a real defect:
the `docker` job's smoke test called `python -c`, but that job is the only one
without an `actions/setup-python` step — it builds images and needs no Python
toolchain. `ubuntu-latest` provides `python3`, not bare `python`.

The step would have failed with "command not found" **while presenting as a
smoke-test failure**, which is the worse outcome: the assertion it appears to
make would never have run, and the error would have pointed at the container
rather than the script.

Fixed in `458ce03`, along with two adjacent weaknesses — the readiness loop
fell through silently after 90 s and the JSON parse then failed on an empty
body, burying the cause; and cleanup moved to an `if: always()` step so a
failed assertion no longer leaves a container holding port 8000.

The three earlier runs show as `cancelled`. That is `concurrency:
cancel-in-progress: true` working correctly — several pushes in quick
succession, each superseding the last.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | **211 passing** (this repo) |
| **Clean checkout** | **193 passed, 18 skipped** — README followed verbatim |
| **Integration** | 9 |
| **Reproducibility** | dataset MD5 identical across two independent clones |
| **Quality gates** | flake8 0, Black clean, isort clean |

---

# Design Decisions

## D1 — Delete `handoff.md` rather than keep it archived

| Field | Detail |
|---|---|
| **Alternatives** | Keep the superseded banner; move it to an `archive/` folder; trim it to a short narrative log. |
| **Pros** | `docs/` now contains only documents worth opening. Nothing is lost — git has every version. |
| **Cons** | A reader following a `Day1.md` reference to "reconstructed from handoff.md" will not find the file in the working tree. |
| **Reason for selection** | It had been superseded for eight days and the banner did not make it less confusing, only longer. `docs/README.md` names the retrieval command, so the cost is one line of friction against 60 KB of clutter. |

## D2 — Put the results above the setup instructions

| Field | Detail |
|---|---|
| **Alternatives** | Leave results in a section two-thirds down; link only to `RESULTS.md`. |
| **Pros** | The first question a reader has is "did it work?", and they can now answer it in ten seconds. |
| **Cons** | Headline figures without their caveats can mislead, which is why the box links straight to `RESULTS.md` and the caveat stays in the Results section. |
| **Reason for selection** | A reviewer who has to hunt for the outcome usually stops hunting. |

## D3 — Report the CI gap rather than assert success

| Field | Detail |
|---|---|
| **Alternatives** | Say "CI configured" and leave the impression it passes; install `gh` and authenticate. |
| **Reason for selection** | Every other claim in this project is backed by something I ran. Asserting a green build I have not seen would be the one exception, and it would be the easiest thing to check and be caught on. |

---

# Technical Debt — final register

| ID | Status |
|---|---|
| TD-1 validation set was the test set | ✅ repaid Day 5 |
| TD-2 no checkpoint resume | ✅ repaid Day 5 |
| TD-3 no training curves | ✅ repaid Day 5 |
| TD-4 `handoff.md` overlap | ✅ **repaid Day 12** |
| TD-5 stale `CLAUDE.md` | ✅ repaid Day 4 |
| TD-6 threshold hardcoded at 0.5 | ✅ repaid Day 5 |
| TD-7 no integration tests | ✅ 9 tests (parity + live grounding); API coverage via `TestClient` |
| **TD-8 mypy: 159 errors, advisory in CI** | ⏳ **open** — mostly loguru's dynamic `logger`; plan is to annotate incrementally, tighten per-module, then make it blocking |

## Deliberate v1 omissions — decisions, not oversights

| Omission | Why, and what it would take |
|---|---|
| **No authentication** | The demo runs on a trusted network. v2 design is written down in `IMPLEMENTATION_PLAN.md`: an API-key header checked by a FastAPI dependency, keys stored hashed, per-key rate limits. ~2 h. |
| **No rate limiting** | `/report` is the only endpoint with real per-call cost. Needed before any public deployment. |
| **Report generation holds a worker** | Up to 120 s in a threadpool. Correct at this scale; a job queue (202 + poll) is the right change at higher concurrency. |
| **No prediction persistence** | `/machines/{id}/history` returns sensor readings, not past predictions. SQLite would do it. |
| **Not deployed** | Images build and run locally; pushing to a registry and a free-tier host is ~3 h. |
| **Model quality is bounded by synthetic data** | The pipeline transfers; the weights would not. Real data is the only fix. |

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | **100% of the 12-day plan** |
| **Modules** | config · utils · data · models · prediction · genai · api · dashboard · docker/CI — all complete |
| **Risks** | ~~R-6 training/serving skew~~ ✅ · ~~R-10 LLM failure~~ ✅ · ~~R-11 deployment~~ ✅ · R-12 partially mitigated (no auth, by design) |
| **Debt** | TD-8 open with a plan |
| **Quality gates** | 211 unit + 9 integration · flake8 0 · Black/isort clean · mypy advisory |
| **CI** | ✅ green — all four jobs pass on `458ce03` |

---

# Files Created

```
docs/RESULTS.md    every metric in one place, with its caveats
docs/Day12.md      this file
```

# Files Modified

```
README.md                 at-a-glance box, real architecture diagram, docs table
IMPLEMENTATION_PLAN.md    TD-4 closed, companion docs, folder tree
CLAUDE.md                 handoff references removed
docs/README.md            handoff row replaced with a provenance note
```

# Files Deleted

```
docs/handoff.md    1,377 lines, superseded; git show 9ceb349:docs/handoff.md
```

---

# Final Summary

Day 12 was about proving the claims rather than adding to them.

The one that mattered most: a stranger can actually run this. Cloned to a fresh
directory, followed the README's setup section verbatim, and got **193 passing
tests in 20 seconds** from a 3.9 MB checkout — then generated the dataset and
found its MD5 identical to the original. Reproducibility had been a stated
non-functional requirement since Day 1; now it is a measurement.

`docs/RESULTS.md` collects the metrics that had been spread across eight day
files, and gives equal space to what they do not establish — synthetic data,
eight test events, a threshold that transferred worse than validation
suggested, and a cost curve whose optimum sits at the edge of a cliff. The
history section is the part I would point a reviewer at: F1 moved 0.7530 →
0.8605 → 0.8949, and every step came from fixing methodology rather than tuning
the model.

TD-4 closed by deleting `docs/handoff.md`. It had worn a "superseded" banner
for eight days, which made it longer without making it less confusing.

The one claim this document could not originally back was CI: the repository was
private, so the pipeline had been written and parsed but never observed. It was
made public afterwards, and **all four jobs pass on `458ce03`** — including the
image builds on amd64, which had only ever been built locally on arm64.

Auditing the workflow while that run sat queued found a defect CI would have
reported misleadingly: the smoke test called `python -c` in the one job without
a Python toolchain, so it would have failed with "command not found" while
looking like the container was broken. Fixed before it ran.

**Ending state: 12 milestones, 211 unit tests, 9 integration tests, an LSTM
catching 8 of 8 failure events with 24 hours of warning, grounded LLM reports
that decline what the data cannot answer, a 137 ms API, a dashboard holding no
model, and two containers that come up healthy in the right order.**
