# Day 13 — Point-in-Time Assessment

**Date:** 2026-08-25
**Focus:** `as_of` — assess the fleet as it looked at any hour in the dataset.
**Status:** ✅ Complete

---

## Summary

The project was finished on Day 12 and it had a defect that twelve days of
planning never anticipated, because it is not a defect in any component. Every
layer was correct; the product still demonstrated nothing.

Starting the stack and opening the dashboard shows **0 machines alerting**. That
is the truthful answer — the API scores the most recent hour it has, and at
2024-12-30 23:00 no machine on the fleet is inside a pre-failure window. It is
also indistinguishable, to anyone looking at it for the first time, from a model
that does not work.

The fix is a timestamp. The API now accepts `as_of` on every prediction
endpoint, hides everything after it, and the dashboard grew a **Rewind** control
in the sidebar. Dial to 2024-10-31 hour 6 and machine 51 goes critical, six
hours before it actually failed.

The part worth keeping is the second-order effect: rewind another day and the
alert disappears. That is the 24-hour horizon becoming visible, and it turned
into the strongest leakage test in the repository.

---

## Starting state

- 12 milestones complete, commit `458ce03`, all four CI jobs green.
- 211 unit tests, 9 integration tests, mypy clean and blocking.
- `docker compose up` bringing both containers healthy.
- Technical debt register: one open item, TD-7 (integration coverage).

---

## Tasks planned

1. Add `as_of` to `MachineDataStore.slice_for()` and the service methods.
2. Expose it as a query parameter on `/machines/{id}/predict`, `/explain`,
   `/history`, and `/fleet`, and as a field on `POST /report`.
3. Publish the dataset's date range from `/health` so a client can bound a
   picker.
4. Add the Rewind control to the dashboard.
5. Test it — including that it is not merely accepted but actually applied.
6. Update the documentation.

---

## Work completed

### The question that started it

The user asked why the dataset is dated 2024. The answer is that
`scripts/generate_data.py` hardcodes `datetime(2024, 1, 1)` so that seed 42
produces a byte-identical dataset on every machine, forever — a property
`day-12.md` verified by MD5 on a fresh clone.

The obvious response was to regenerate with recent dates. I did not, and the
reasoning is the substance of the day:

> Regenerating gives the same roughly 12% chance that any machine happens to be
> alerting at the final hour. You would spend forty minutes retraining and most
> likely still see "0 alerting" — with a dataset that no longer reproduces.

The date was never the problem. **Only ever assessing the last row** was the
problem, and that is fixed by letting the caller choose the row.

### `as_of` in the data store

```python
telemetry = telemetry[telemetry["machine_id"] == machine_id]
if as_of is not None:
    telemetry = telemetry[telemetry["datetime"] <= as_of]
telemetry = telemetry.sort_values("datetime").tail(window_hours)
```

Errors and maintenance are filtered by the same cutoff. That is not tidiness:
`errors_last_24h` and `hours_since_maintenance` are model features, so filtering
telemetry alone would let a historical prediction be made with knowledge of
breakdowns that had not happened yet. Leakage wearing the clothes of a feature.

The cutoff is inclusive. The chosen hour has already occurred, so its reading is
evidence, not the future.

### The fleet cache had to be re-keyed

`GET /fleet` scores 100 machines in 13.4 seconds cold and 1.6 ms cached, which
is what makes the endpoint usable at all. The cache was a single slot. Adding
`as_of` without touching it would have served a cached present-day answer to a
request asking about October — a correct-looking wrong answer, the worst kind.

```python
self._fleet_cache: Dict[Any, List[Dict[str, Any]]] = {}
key = None if as_of is None else pd.Timestamp(as_of)
```

Covered by `test_the_fleet_cache_is_keyed_by_as_of`, which alternates between
two timestamps and asserts the service saw both.

### Reports too

`POST /report` gained an `as_of` field. A report is written entirely from the
prediction record it is handed, so a report about a past moment must be grounded
in a prediction from that moment — otherwise the dashboard shows an October
assessment above a paragraph describing December.

`run_in_threadpool` takes no keyword arguments, so the call is bound with
`functools.partial` before being handed over.

### The dashboard

A **Rewind** toggle in the sidebar, off by default, revealing a date picker and
an hour slider bounded by the range `/health` publishes. The bounds come from
the API rather than being assumed by the UI: outside the range every assessment
is empty, and a picker offering hours that cannot be answered is worse than no
picker.

The control is deliberately hidden when `/health` publishes no range — a
degraded API has no data to bound it with.

---

## Testing

| Suite | Before | After |
|---|---|---|
| Unit | 211 | **229** |
| Integration | 9 | **13** |

The unit tests cover plumbing: the timestamp reaches the service, an
unparseable one is a 422, the cache is keyed, the picker is bounded, the toggle
is off by default.

The integration tests cover whether the feature is *worth having*, which is a
different question and needs the real model:

| Test | Asserts |
|---|---|
| `test_rewinding_to_a_pre_failure_hour_raises_an_alert` | 5 of 5 sampled machines alert 6 h before a known failure |
| `test_the_model_stays_quiet_beyond_its_horizon` | 0 of 5 alert at 36 h — the leakage check |
| `test_hiding_the_future_changes_the_answer` | A rewound assessment differs from the present-day one |
| `test_a_rewound_slice_contains_nothing_after_the_cutoff` | No table leaks past the cutoff, over the real 876,000 rows |

The second one is the reason this file exists. The features include 24-hour
rolling windows, so incomplete filtering — telemetry trimmed but errors left
whole — could let a 36-hour rewind see a failure's own aftermath and fire.
Silence is the evidence that it does not.

Measured directly:

| Machine 51 assessed at | Probability |
|---|---|
| 2024-10-31 06:00 (6 h before failure) | **1.0000** |
| 2024-10-30 12:00 (24 h earlier) | **0.0000** |

---

## Bugs encountered

**The sidebar hint named the wrong dates.** I wrote "try 2024-10-30 (machine
51)" from memory before checking `failures.csv`. Machine 51 fails at 2024-10-31
12:00, so 10-30 is 36 hours out — outside the horizon, and the machine reads as
quiet. A hint that demonstrates nothing is worse than no hint, since the reader
concludes the feature is broken.

Corrected to 2024-10-31 hour 6, and the wrong date was kept deliberately as the
second half of the hint: *rewind to 2024-10-30 and machine 51 goes quiet — that
is the 24-hour horizon, not a bug.* The mistake made the better demo.

**Five API tests began failing with `KeyError: 'status'`.** The test stubs did
not accept the new keyword, so every call raised `TypeError` and returned a 500
whose body had none of the expected keys. The symptom pointed at the response
schema; the cause was the stub's signature. Stubs standing in for a real
interface have to track it, so they now do — and they record `(method,
machine_id, as_of)` so a test can assert the timestamp actually arrived rather
than merely being accepted.

**The stub's `explain` delegated to its own `predict`,** which appended a second
call record and made `calls[-1]` the wrong one. Fixed by passing `as_of` down —
which is what the real service does anyway — and asserting on the first entry.

**`../IMPLEMENTATION_PLAN.md` still said Day 4 was in progress** and Days 5–12
pending, four sections below a completion bar reading 100%. Exactly the drift
R-13 was written to catch, sitting in the file that documents R-13. Corrected.

---

## Design decisions

**Rewind is off by default.** The honest default is the latest reading. A
dashboard that silently opens on a hand-picked interesting moment is a
demonstration, not a tool.

**The range comes from the API.** The dashboard holds no model and, by the same
logic, should hold no assumptions about the data. `/health` gained `data_start`
and `data_end`.

**`None` means now.** Every `as_of` parameter is optional and defaults to
`None`, so all existing callers and the previous API contract are unchanged.
This was additive throughout — no endpoint changed shape, no response field
moved.

**Not regenerating the dataset.** Discussed above. Reproducibility was a
non-functional requirement from Day 1 and had been verified by MD5 on a clean
clone; spending it to make a date look current would have been a poor trade,
and it would not have fixed the actual problem.

---

## Files changed

| File | Change |
|---|---|
| `src/api/service.py` | `as_of` on `slice_for`/`predict_machine`/`explain_machine`/`fleet`; `data_range`; cache re-keyed |
| `src/api/routes/predict.py` | `as_of` query parameter on four endpoints |
| `src/api/routes/health.py` | Publishes the data range |
| `src/api/routes/reports.py` | `as_of` bound via `partial` |
| `src/api/schemas.py` | `data_start`/`data_end` on health; `as_of` on `ReportRequest` |
| `dashboard/app.py` | Rewind control; `as_of` threaded to every call |
| `dashboard/api_client.py` | `as_of` on `predict`/`explain`/`history`/`fleet`/`report` |
| `tests/unit/test_api.py` | `TestTimeTravel`, 12 tests; stubs updated |
| `tests/unit/test_dashboard_app.py` | `TestRewind`, 5 tests |
| `tests/unit/test_dashboard_client.py` | 2 tests |
| `tests/integration/test_time_travel.py` | **New** — 4 tests |
| `../IMPLEMENTATION_PLAN.md` | Roadmap entry, stale status sections, TD-7 closed, K-6 added |
| `docs/RESULTS.md` | Horizon evidence; counts corrected |

---

## Project health

| Check | State |
|---|---|
| Unit tests | 229 passing |
| Integration tests | 13 passing |
| flake8 / Black / isort | Clean |
| mypy | 0 errors across 29 files |
| Technical debt | **Empty** — TD-7 closed |

---

## Final summary

A finished product can be correct in every part and still fail to show what it
does. The fix was small — an optional timestamp, defaulting to the existing
behaviour, threaded through four layers — and it did not require retraining,
regenerating, or changing a single existing contract.

What it bought was more than a demo. The horizon is now something the repository
asserts rather than claims: 5 of 5 machines alerting six hours before failure,
0 of 5 at thirty-six. **Twelve days of work built a model that predicts failures
24 hours ahead. Day 13 made that sentence checkable.**
