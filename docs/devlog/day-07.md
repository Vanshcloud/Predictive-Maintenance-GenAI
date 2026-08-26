# Day 7 Summary

| Field | Value |
|---|---|
| **Objective** | Turn predictions into maintenance reports a technician can act on, using LangChain and an LLM — without the LLM inventing anything. |
| **Expected outcome** | Grounded evidence extraction, prompt templates that make confabulation hard, a provider-agnostic chain, graceful degradation when the LLM is unavailable, and tests that need no API key. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M7 — GenAI report generation |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `307fdd2` — "docs: record Day 6 and close Risk R-6" |
| **Existing modules** | config, utils, data, models, prediction — all complete |
| **Existing models** | `lstm_predictive_maintenance.keras`, F1 0.8949, event recall 8/8 |
| **Tests** | 113 unit + 4 integration, flake8 clean |
| **Technical debt** | TD-4 (`handoff.md` overlap), TD-7 (integration coverage partial) |

`src/genai/` was an empty scaffold. `Predictor.predict_machine()` returned a
JSON-serialisable record — deliberately, from Day 6 — which is exactly what a
chain needs as input.

**No LLM credentials were available**, so everything had to work without one:
tests against a fake model, and a keyless local path for real runs.

---

# Tasks Planned

### T1 — Evidence extraction

| Field | Detail |
|---|---|
| **Purpose** | A probability is not an explanation. Handing an LLM only "0.87" invites it to invent a cause. |
| **Files affected** | `src/prediction/predictor.py` |
| **Priority** | P0 — everything else is downstream |
| **Expected outcome** | Real feature values, per sensor, with baselines and deviations. |

### T2 — Prompt templates

| Field | Detail |
|---|---|
| **Purpose** | Where the anti-hallucination work actually lives. |
| **Files affected** | `src/genai/prompts.py` |
| **Priority** | P0 |
| **Expected outcome** | System persona, report and Q&A templates, fact formatter. |

### T3 — Provider-agnostic chains

| Field | Detail |
|---|---|
| **Purpose** | Tying a portfolio project to one vendor's API key means it stops working for anyone without one. |
| **Files affected** | `src/genai/chains.py` |
| **Priority** | P0 |
| **Expected outcome** | OpenAI / Google / Ollama behind one interface, selected from settings. |

### T4 — Graceful degradation (Risk R-10)

| Field | Detail |
|---|---|
| **Purpose** | The prediction decides whether a technician is dispatched; the report is a convenience over it. An LLM outage must not take the prediction down. |
| **Files affected** | `src/genai/chains.py`, `scripts/generate_report.py` |
| **Priority** | P0 |
| **Expected outcome** | `LLMConnectionError` vs `ReportGenerationError`, and a CLI that prints the prediction anyway. |

### T5 — CLI

| Field | Detail |
|---|---|
| **Purpose** | End-to-end usability, and a way to inspect prompts with no key. |
| **Files affected** | `scripts/generate_report.py` |
| **Priority** | P1 |

### T6 — Tests without an API key

| Field | Detail |
|---|---|
| **Purpose** | Untestable code is untested code. |
| **Files affected** | `tests/unit/test_genai.py` |
| **Priority** | P0 |

---

# Work Completed

## T1 — Evidence extraction ✅

`Predictor.explain_machine()` returns the `predict_machine()` record plus a
`context` block built from engineered features the model actually consumed:

- current reading, 24h baseline, 24h change, 24h volatility, per sensor
- **deviation in units of that sensor's own volatility**, so a 450 RPM sensor
  and a 40 mm/s sensor can be ranked against each other honestly
- errors in last 24h, hours since maintenance per component, age
- sensors ranked by absolute deviation

Nothing is derived for presentation. Every number traces to a feature.

## T2 — Prompts ✅

Three defences, all in `prompts.py`:

1. **Every usable fact is supplied pre-formatted with units and baselines.**
   Formatting rather than dumping JSON is deliberate: a raw blob makes the
   model parse structure *and* write prose, and numbers get transposed in the
   parsing step.
2. **The system prompt names the only four sensors that exist** and forbids
   stating any measurement not in the DATA block. Without this, models reach
   for temperature and lubrication, neither of which this equipment measures.
3. **Sensor data is delimited and labelled as data**, never instructions
   (Risk R-12, prompt injection).

The 9999 maintenance sentinel renders as "no record" — passing it through as a
number invites "maintained 9999 hours ago", which is both wrong and absurd.

## T3 — Provider abstraction ✅

`get_llm()` builds OpenAI, Google, or Ollama from settings, importing each
lazily so a missing optional extra produces a clear message rather than an
ImportError at module load. Default selection prefers whatever has credentials,
falling back to keyless local Ollama.

Temperature defaults to 0.2: this is a work order, not creative writing, and
variation between runs on identical data would undermine trust in it.

## T4 — Graceful degradation ✅ (Risk R-10 closed)

`_wrap()` classifies provider failures:

| Class | Triggers | Meaning |
|---|---|---|
| `LLMConnectionError` | connection, timeout, rate limit, 429/503, auth, api key, 404/model-not-found | Retryable; the provider is unavailable |
| `ReportGenerationError` | anything else | The input was wrong |

The CLI catches the former and prints the prediction anyway:

```
The prediction is available; the written report is not.
  machine 3: probability 0.0000 (low)
```

## T5 — CLI ✅

```bash
python scripts/generate_report.py --machine 51 --dry-run     # prompt only, no key
python scripts/generate_report.py --machine 51 --provider ollama --model qwen2.5-coder:7b
python scripts/generate_report.py --fleet
python scripts/generate_report.py --machine 51 --ask "Which part fails first?"
```

`--dry-run` prints the exact grounded facts without calling an LLM. It is how
you check grounding with no key, network, or cost — and how you debug a bad
report, since a bad report is nearly always a bad prompt.

## T6 — Tests ✅

28 tests against `FakeListChatModel`: no key, no network, no cost. They assert
what we control — prompt content, grounding, error classification — and
deliberately **not** what the model writes. Asserting on LLM output would test
OpenAI's weights, which are not ours, not deterministic, and not what breaks.

---

# Live-Model Verification — and what it found

The mocked tests all passed. Then a real model was pointed at the pipeline
(`qwen2.5-coder:7b`, local Ollama, no key), and found three bugs in a row that
**26 passing mock-based tests had not**.

## Iteration 1 — the causal hint read as an observation

Report said: *"Pressure drop suggests a leak or failing seal."*
Actual reading: pressure **0.66 sigma ABOVE** baseline.

The evidence block attached `concerning_when: "drops"` and
`typical_cause: "a leak or a failing seal"` to *every* pressure reading,
unconditionally. The model read the conditional as an observation — which is a
completely reasonable thing to do with that input.

The same report also opened with "risk LOW, no threshold breached" and then
recommended inspecting bearings and seals **within 24 hours**.

**Fix:** the judgement moved into the evidence layer. `is_concerning` is
computed from the actual deviation direction, and `typical_cause` is `None`
unless the reading points that way. Each sensor line now carries an explicit
verdict, and the template states the recommended action must match the risk
level.

## Iteration 2 — my fix invented a fault of its own

After the fix, a healthy machine had **voltage flagged ABNORMAL**.

Voltage fails by becoming erratic rather than by drifting, so it is judged on
volatility. I had written the threshold by hand:

```python
concerning = volatility > 1.5 * abs(baseline) * 0.05     # = 12.6 for voltage
```

The training population's mean 24h voltage volatility is **15.71**. My
threshold sat *below the mean*, so every healthy machine tripped it.

This is the more interesting failure. Fixing a hallucination bug with a
hand-picked constant did not remove the invention — it moved it. Instead of the
model inventing a fault, my heuristic invented one and the model faithfully
reported it. The output was *more* dangerous, because it now looked grounded.

**Fix:** `_population_z()` uses the fitted `StandardScaler`, which already
carries `mean_` and `scale_` for all 63 features. "Is this volatility unusual?"
is now answered against the distribution the model was trained on. The
reference was sitting inside an artifact the Predictor already had loaded.

## Iteration 3 — a wording glitch

`"0.43 sigma steady baseline"`. `direction` was overloaded to carry both level
and volatility semantics. It now always describes the level.

## Final verification

Healthy machine (`data/sample`, machine 3) — all four sensors correctly
unremarkable:

```
  vibration: 46.2 mm/s (0.88 sigma above baseline) -> within normal variation
  pressure: 106.82 PSI (0.66 sigma above baseline) -> within normal variation
  voltage: 160.58 V (0.43 sigma below baseline)    -> within normal variation
  rotation: 407.66 RPM (0.37 sigma below baseline) -> within normal variation
```

Genuinely degrading machine — machine 51, hours before its real 2024-10-30
failure:

```
  pressure: 65.89 PSI (1.91 sigma below baseline; -32.06 PSI over 24h)
      -> ABNORMAL; typically indicates a leak or a failing seal
  vibration: 62.27 mm/s (1.77 sigma above baseline; +24.04 mm/s over 24h)
      -> ABNORMAL; typically indicates components loosening or bearings worn
  voltage:  1.04 sigma below baseline  -> within normal variation
  rotation: 0.33 sigma below baseline  -> within normal variation
```

Those two flagged sensors are exactly the degradation signature the generator
injects before a failure (pressure drops ~20 PSI from leaks, vibration rises
~20 mm/s from loosening). The other two are correctly left alone.

**A detail worth keeping.** That machine-51 hour scored **0.3474 — below the
alert threshold**, and is one of the 17 hours Day 6 identified as "missed". The
model under-called it, but the evidence layer surfaces the degradation
unambiguously: two sensors nearly 2 sigma out, moving in the directions that
matter. A technician reading that report would investigate regardless of the
headline number. That is an argument for shipping evidence *alongside*
probability rather than instead of it — the narrative layer is not merely
decoration over the model, it is a second, independent view of the same data.

---

# Bugs Encountered

## B1 — Unconditional causal hints presented as observations

| Field | Detail |
|---|---|
| **Root cause** | `typical_cause` attached to every sensor regardless of whether it was deviating in the concerning direction. |
| **Solution** | `is_concerning` computed from deviation direction; hints withheld otherwise; explicit per-sensor verdict lines. |
| **Verification** | `test_causal_hint_is_withheld_when_the_reading_is_not_concerning`, plus live re-run. |
| **Lessons learned** | The prompt was *complete* and still *wrong*. Every fact was present; the framing made a conditional look like a finding. Grounding is not only about supplying data — it is about supplying it in a form that cannot be misread. |

## B2 — A hand-picked threshold flagged every healthy machine

| Field | Detail |
|---|---|
| **Root cause** | `1.5 * baseline * 0.05` = 12.6 for voltage, below the training population's mean volatility of 15.71. |
| **Solution** | `_population_z()` against the fitted scaler's `mean_` / `scale_`. |
| **Verification** | Healthy machine now clean; degraded machine still flags the right two sensors. |
| **Lessons learned** | I fixed a hallucination with an invented number, which relocated the invention from the model into my own code — and made it *harder* to spot, because the output then looked grounded. When a threshold is needed, look for a fitted statistic before reaching for a constant. This project had one loaded in memory the whole time. |

## B3 — Placeholder API key read as a real one

| Field | Detail |
|---|---|
| **Description** | `--machine 3` returned a 401 "Incorrect API key provided: your-ope***here". |
| **Root cause** | `.env.example` ships `OPENAI_API_KEY=your-openai-api-key-here`, and the README says `cp .env.example .env`. A placeholder is a non-empty string, so the app believed it had credentials. |
| **Solution** | A pydantic validator nulls placeholder-looking values, so the keyless Ollama path is selected and the error message is the true one. |
| **Lessons learned** | Affects **every** new user following the documented setup. Found only because I ran the documented setup rather than my own. |

## B4 — Deprecated `ChatOllama` import

| Field | Detail |
|---|---|
| **Root cause** | `langchain_community.chat_models.ChatOllama` is deprecated and slated for removal in 1.0. |
| **Solution** | Prefer `langchain_ollama`, fall back to community so base requirements still work. |

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 113 → **141 passing** (28 new) |
| **Integration tests** | 4 (unchanged) |
| **Live model** | `qwen2.5-coder:7b` via local Ollama — 3 report generations |
| **Quality gates** | flake8 **0**, Black and isort clean |

Two of the new tests are regressions that quote the failing output in their
docstrings, so a future prompt edit reintroducing either failure fails loudly.

---

# Design Decisions

## D1 — Put the "is this concerning?" judgement in the evidence layer, not the prompt

| Field | Detail |
|---|---|
| **Alternatives** | Give the model raw numbers and let it judge; state the rule in the prompt and trust it. |
| **Pros** | Deterministic, testable, and identical across providers. A weaker model gets the same verdicts as a stronger one. |
| **Cons** | The thresholds (1 sigma on level, 2 sigma on population volatility) are ours, and a genuinely subtle pattern the model might have noticed is now pre-filtered away. |
| **Reason for selection** | Judgement that must be consistent should not be delegated to a sampling process. B1 showed a capable model misreading a conditional it was handed. |
| **Impact** | Reports are as reliable as the weakest supported model, not the strongest. |

## D2 — Test prompts, not outputs

| Field | Detail |
|---|---|
| **Alternatives** | Snapshot-test generated reports; LLM-as-judge scoring. |
| **Pros** | Deterministic, instant, free, no key. |
| **Cons** | **Demonstrably insufficient.** All 26 tests passed while the prompt actively misled the model. |
| **Reason for selection** | Still the right default for CI. But the gap is real and now documented: mock tests verify facts *reach* the prompt, not that the prompt *reads correctly*. A periodic live-model check is required, and is listed under Remaining Tasks. |

## D3 — Ollama as the keyless default

| Field | Detail |
|---|---|
| **Pros** | The project runs end to end with no account, key, or spend — which is what makes it reviewable by a stranger. |
| **Cons** | Requires a local install and a pulled model; quality is below GPT-4o-mini. |
| **Reason for selection** | A portfolio project that demands an API key to demonstrate its headline feature does not get demonstrated. |

## D4 — Fail the report, never the prediction

| Field | Detail |
|---|---|
| **Reason for selection** | The prediction is safety-relevant; the narrative is convenience. Degrading to "prediction available, report unavailable" preserves the part that matters. |
| **Impact** | Risk R-10 closed. Exercised for real — every live run before Ollama was configured took this path. |

---

# Remaining Tasks

| Item | Priority | Effort |
|---|---|---|
| A live-model smoke check in `make test-integration` (marked `slow`, skipped without Ollama) — D2's documented gap | **P1** | 2 h |
| Report caching — regenerating an identical report costs a call every time | P2 | 2 h |
| Async report generation so the API never blocks a prediction on an LLM (Day 9) | P2 | 2 h |
| Tune the 1-sigma / 2-sigma verdict thresholds against labelled events rather than by eye | P2 | 3 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | 1 h |

---

# Next Day Plan

**Day 8 — GenAI Assistant & Maintenance Q&A**

1. `src/genai/assistant.py` — conversational Q&A with history, over one machine
   or the fleet.
2. Extend context beyond the latest reading: recent trend, prior failures,
   maintenance history — the assistant needs more than a snapshot.
3. Reuse `answer_question()`, adding memory and multi-turn grounding.
4. Guard against the same failure class: the assistant must decline questions
   the data cannot answer rather than improvising.
5. Tests with a fake model, including a conversation where the second turn
   depends on the first.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~58% |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · `src/models/` 100% · `src/prediction/` 100% · **`src/genai/` 70%** (reports done, assistant is Day 8) · `src/api/` 0% · `dashboard/` 0% |
| **Known risks** | ~~R-6~~ ✅ · ~~R-10 LLM provider failure~~ ✅ **closed** · R-12 prompt injection (mitigated: data delimited and labelled) |
| **Quality gates** | 141 unit + 4 integration · flake8 0 · Black/isort clean |

---

# Files Created

```
src/genai/prompts.py              personas, templates, fact formatter
src/genai/chains.py               provider factory, ReportGenerator
scripts/generate_report.py        CLI with --dry-run
tests/unit/test_genai.py          28 tests
day-07.md                      this file
```

# Files Modified

```
src/genai/__init__.py         exports
src/prediction/predictor.py   explain_machine, _population_z, conditional hints
config/settings.py            placeholder API keys read as unset
```

# References

- [LangChain Expression Language](https://python.langchain.com/docs/concepts/lcel/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM01 prompt injection, LLM09 overreliance
- [Ollama](https://ollama.com/) — the keyless local path

---

# Final Summary

Day 7 closed the loop the project was built for: raw sensor tables in, a work
order out. `src/genai/` turns a grounded prediction into prose through a
provider-agnostic chain that runs on OpenAI, Google, or a local keyless model,
and fails in the one direction that is safe — the report can be unavailable,
the prediction cannot.

The day's real content was three iterations of a single lesson. A capable model
was handed a complete, accurate set of facts and still wrote *"Pressure drop
suggests a leak"* about a pressure reading that was **above** its baseline,
because the causal hint sat next to the number unconditionally. Fixing that
with a hand-written threshold then produced a worse bug: my constant of 12.6
sat below the training population's mean voltage volatility of 15.71, so every
healthy machine was flagged — the invention had moved out of the model and into
my own code, where it looked grounded. The fix was to stop inventing and use
the fitted `StandardScaler`, which had carried the right reference distribution
in memory the entire time.

None of that was visible to 26 passing tests. They verified that facts reached
the prompt, which was true and insufficient — the prompt was complete and still
misleading. One run against a real model found what the mocks structurally
could not, and the gap is now written down with a task attached rather than
quietly closed.

The final check is the one that matters: on a machine hours from a real
failure, the evidence layer flags pressure 1.91 sigma low and vibration 1.77
sigma high — precisely the degradation signature — and leaves the other two
sensors alone. That hour scored 0.3474, below the alert threshold, and is one
of Day 6's "missed" hours. The number under-called it; the evidence did not.

Ending state: 141 unit tests, 4 integration tests, R-10 closed, and a report a
technician could act on.
