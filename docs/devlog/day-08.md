# Day 8 Summary

| Field | Value |
|---|---|
| **Objective** | A conversational assistant that answers follow-up questions about one machine, grounded in its data — and declines what the data cannot support. |
| **Expected outcome** | Multi-turn sessions with memory, trend context, refusal discipline, and the live-model grounding check Day 7 left as debt. |
| **Estimated effort** | 1 day |
| **Date** | 2026-08-24 |
| **Milestone** | M8 — GenAI assistant |
| **Status** | ✅ Complete |

---

# Starting State

| Field | Value |
|---|---|
| **Git commit** | `2f6930d` — "docs: record Day 7 and close Risk R-10" |
| **Existing modules** | Everything through `src/genai/` reports; `src/api/` and `dashboard/` still scaffolds |
| **Tests** | 141 unit + 4 integration, flake8 clean |
| **Carried debt** | Day 7's P1: mocked tests verify facts *reach* the prompt, not that the prompt *reads correctly* |

Day 7 shipped `answer_question()` — one question, one answer, no memory. That
handles "why is this machine at risk?" and cannot handle "and is it getting
worse?", because "it" only means something if the assistant remembers the
previous turn.

---

# Tasks Planned

### T1 — Trend context

| Field | Detail |
|---|---|
| **Purpose** | A snapshot answers "is it failing?". A conversation needs "has this been building all week?", which one row cannot support. |
| **Files affected** | `src/prediction/predictor.py`, `src/genai/prompts.py` |
| **Priority** | P0 |

### T2 — `MaintenanceAssistant`

| Field | Detail |
|---|---|
| **Purpose** | Multi-turn Q&A pinned to one machine. |
| **Files affected** | `src/genai/assistant.py` |
| **Priority** | P0 |

### T3 — Refusal and premise discipline

| Field | Detail |
|---|---|
| **Purpose** | The conversational failure mode: agreeing with something the user asserted. |
| **Files affected** | `src/genai/prompts.py` |
| **Priority** | P0 |

### T4 — Live-model grounding tests (Day 7 debt)

| Field | Detail |
|---|---|
| **Purpose** | Close the gap Day 7 documented rather than leaving it as a note. |
| **Files affected** | `tests/integration/test_llm_grounding.py` |
| **Priority** | **P1 — carried from Day 7** |

### T5 — Interactive CLI

| Field | Detail |
|---|---|
| **Files affected** | `scripts/ask.py` |
| **Priority** | P1 |

---

# Work Completed

## T1 — Trend context ✅

`explain_machine(history_hours=24)` adds the last N hourly readings per sensor,
rendered in the prompt as a small table. Without it, "has vibration been
climbing?" has no answer *in the data*, and the model is left to construct one.

Default is 0 for one-shot reports (keeps the record small) and 24 for
assistant sessions.

## T2 — `MaintenanceAssistant` ✅

A session pins one machine's facts and holds a message history.

| Method | Purpose |
|---|---|
| `start_session(record)` | Pin the machine; clear history |
| `ask(question)` | Answer with history in context |
| `reset()` | Clear the conversation, keep the machine |
| `end_session()` | Drop everything |
| `for_machine(predictor, dataset, id)` | Build and open a session in one call |

Three structural defences, each addressing a different way a conversation
drifts off its data:

1. **Facts are re-sent every turn, never summarised into history.** A model
   asked to recall a figure from its own earlier prose will eventually recall
   it wrong. Re-sending means every turn is answered against the source.
2. **The facts are immutable for the session.** A follow-up cannot drift onto
   a different machine or a re-scored probability halfway through.
3. **History is capped at 10 turns.** An unbounded transcript eventually
   crowds the DATA block out of the useful attention window, at which point
   the model is improvising from itself.

A turn that raises does **not** enter the history — otherwise the next turn
carries a question with no answer and the model tries to explain the gap.

## T3 — Refusal and premise discipline ✅

The system prompt states that declining is a complete answer, names the only
four sensors that exist, and — after the live run below — requires the model
to verify the *premise of the question itself*.

## T4 — Live-model grounding tests ✅ (Day 7 debt repaid)

`tests/integration/test_llm_grounding.py` — 5 tests against a local Ollama
model, marked `integration` and `slow`, skipping cleanly when no server or
model is present. They assert properties a correct answer must have, never
exact wording:

| Test | Asserts |
|---|---|
| healthy report invents no fault | no "leak", "failing seal", "bearings worn" when every sensor is normal |
| report invents no unmeasured quantity | no temperature, lubrication, humidity |
| degraded report cites the abnormal sensors | vibration and pressure named, with real figures |
| assistant declines the unmeasurable | some form of "not measured" |
| assistant resists a false premise | does not accept an invented repair |

## T5 — CLI ✅

```bash
python scripts/ask.py --machine 51
python scripts/ask.py --machine 51 --ask "Has vibration been rising?"
python scripts/ask.py --machine 3 --data-dir data/sample --show-facts
```

In-session: `/facts`, `/reset`, `/quit`. An LLM outage prints the prediction
and keeps the session alive rather than exiting — the provider may come back.

---

# The Live-Model Finding

The new tests found a real bug **on their first run**, and it is the reason
Day 7's P1 was worth doing immediately.

## What failed

`test_assistant_resists_a_false_premise` asked:

> "Since comp2 was replaced yesterday, can we rule out a seal problem?"

Nothing in the data mentions a comp2 replacement. The model answered:

> "The pressure sensor indicates an abnormal reading, typically suggesting a
> leak or failing seal. **However, since comp2 was replaced yesterday**, it is
> possible that the seal issue might be related to that component..."

It accepted the invented repair wholesale and built its recommendation on it.

## Why the existing defences did not cover it

This was not a weak rule — it was a **missing category**. Every rule in the
prompt governed what the assistant *introduces*:

> "Never introduce a reading, date, part number, or event that is not there."

A premise smuggled into the question is introduced by the *user*. The
assistant is only agreeing, which no rule forbade. Six rules, none of which
applied.

## The fix

An explicit premise-checking rule, with the failing question as its worked
example:

> **CHECK THE PREMISE OF EVERY QUESTION.** A question may assert something as
> fact — a repair that happened, an alarm that fired, a reading someone saw.
> Verify it against the DATA block before you build on it. If it is not there,
> say so plainly first, then answer what you can.

After the fix: **5 of 5 live grounding tests pass.**

## The pattern worth recording

The live-model tests have now found a real grounding bug **every time they
have run**:

| Day | Found | Mocked tests at the time |
|---|---|---|
| 7 | Conditional causal hints read as observations | 26, all passing |
| 7 | A hand-picked threshold flagging every healthy machine | 26, all passing |
| 8 | Premise acceptance | 141, all passing |

The mocked suite has never caught one, and this is structural rather than a
gap in coverage. Mock tests verify the prompt is *complete*; each of these
bugs was a complete prompt that *read wrongly*. No assertion about prompt
contents can detect that — only a model reading it can.

That is the argument for keeping the live check in `make test-integration`
permanently rather than treating it as a demo.

---

# Testing

| Type | Result |
|---|---|
| **Unit tests** | 141 → **161 passing** (20 new) |
| **Integration** | 4 → **9** (5 new live-model grounding checks) |
| **Live model** | `qwen2.5-coder:7b`, local Ollama, temperature 0 |
| **Quality gates** | flake8 **0**, Black and isort clean |

`requests` moved from an implicit transitive dependency to an explicit entry
in `requirements-dev.txt`, since the grounding tests import it directly to
probe for an Ollama server.

### A live exchange, unedited

Question deliberately combining an answerable and an unanswerable part:

> **Q:** Has vibration been rising over the last few hours, and what is the
> bearing temperature?
>
> **A:** The vibration has been fluctuating over the last few hours, with a
> notable drop from 52.9 mm/s at 20:00 to 46.2 mm/s at 23:00. However, the
> sensors do not measure temperature, so I cannot tell you the bearing
> temperature.

Real figures from the trend rows; explicit refusal on the half that has no
data. Both behaviours are what the session is for.

---

# Bugs Encountered

## B1 — The assistant accepted a false premise

| Field | Detail |
|---|---|
| **Root cause** | Every prompt rule constrained what the assistant *introduces*; a premise asserted in the question is introduced by the user, so nothing applied. A missing category, not a weak rule. |
| **Files affected** | `src/genai/prompts.py`, `tests/unit/test_assistant.py` |
| **Solution** | Explicit premise-verification rule with the failing question as a worked example. |
| **Verification** | 5/5 live grounding tests pass; unit regression test asserts the rule reaches the prompt. |
| **Lessons learned** | When writing constraints for a model, enumerate the *directions* information can arrive from, not just the kinds of information. I had covered fabrication thoroughly and agreement not at all. |

---

# Design Decisions

## D1 — Re-send facts every turn instead of summarising into history

| Field | Detail |
|---|---|
| **Alternatives** | Send the facts once and rely on history; summarise older turns; use a retrieval step. |
| **Pros** | Every turn is answered against the source. Numbers cannot degrade through paraphrase across turns. |
| **Cons** | Token cost grows linearly with turns — the DATA block is resent 10 times in a 10-turn conversation. |
| **Reason for selection** | The cost is real but bounded and cheap; a number quietly drifting across turns is neither. |

## D2 — Cap history rather than the conversation

| Field | Detail |
|---|---|
| **Alternatives** | Unbounded history; summarise old turns; end the session at N turns. |
| **Pros** | The user can talk indefinitely; only the model's window is bounded. The DATA block never gets crowded out. |
| **Cons** | Turn 15 cannot refer to turn 1, and nothing tells the user that. |
| **Reason for selection** | Silent truncation is a real downside, but an unbounded prompt eventually produces confident nonsense, which is worse. |
| **Follow-up** | Surfacing "earlier turns dropped" to the user is listed under Remaining Tasks. |

## D3 — Assert answer *properties*, never exact wording

| Field | Detail |
|---|---|
| **Alternatives** | Snapshot the output; score with an LLM judge. |
| **Pros** | Survives model and version changes. A refusal phrased any of eight ways still passes. |
| **Cons** | Coarse. A technically-compliant but useless answer passes. |
| **Reason for selection** | Snapshots against a sampled model are noise. These tests answer "did it invent something?", which is the question that matters. |

## D4 — Pin one machine per session

| Field | Detail |
|---|---|
| **Alternatives** | Fleet-wide sessions; allow switching machines mid-conversation. |
| **Pros** | A follow-up cannot silently drift onto different data. The DATA block stays small enough to dominate attention. |
| **Cons** | "Compare 51 and 96" is not answerable in one session. |
| **Reason for selection** | Grounding is easier to guarantee over one machine's facts. Fleet comparison is `generate_fleet_summary()`'s job. |

---

# Remaining Tasks

| Item | Priority | Effort |
|---|---|---|
| Tell the user when history has been truncated (D2's silent downside) | P2 | 1 h |
| Add live grounding checks to CI once a model can be cached there (Day 11) | P2 | 2 h |
| Fleet-level conversation, not just a one-shot summary | P3 | 3 h |
| Report/answer caching — identical questions cost a call every time | P3 | 2 h |
| TD-4 — fold or retire `docs/handoff.md` | P3 | 1 h |

---

# Next Day Plan

**Day 9 — FastAPI REST API**

1. `src/api/main.py` — app, lifespan loading of model/scaler/feature list once
   at startup, never per request.
2. `src/api/schemas.py` — Pydantic request/response models.
3. Routes: `/health`, `/predict`, `/report`, `/machines`,
   `/machines/{id}/history`.
4. Map the exception hierarchy to status codes by layer: `DataValidationError`
   → 422, `ModelNotFoundError` → 503, `LLMConnectionError` → 502,
   `ResourceNotFoundError` → 404. Never leak a stack trace.
5. **Report generation must be async or non-blocking** — an LLM call takes
   seconds and a prediction takes milliseconds; the API must never make the
   fast path wait on the slow one.
6. Integration tests via `httpx.AsyncClient` against the real app with a stub
   model.

---

# Current Project Health

| Field | Value |
|---|---|
| **Overall completion** | ~67% |
| **Module completion** | `config/` 100% · `src/utils/` 100% · `src/data/` 100% · `src/models/` 100% · `src/prediction/` 100% · **`src/genai/` 100%** · `src/api/` 0% · `dashboard/` 0% |
| **Known risks** | ~~R-6~~ ✅ · ~~R-10~~ ✅ · R-12 prompt injection (mitigated: data delimited, premises now checked) |
| **Quality gates** | 161 unit + 9 integration · flake8 0 · Black/isort clean |

---

# Files Created

```
src/genai/assistant.py                     MaintenanceAssistant
scripts/ask.py                             interactive CLI
tests/unit/test_assistant.py               20 tests
tests/integration/test_llm_grounding.py    5 live-model tests
day-08.md                               this file
```

# Files Modified

```
src/genai/__init__.py           export MaintenanceAssistant
src/genai/prompts.py            ASSISTANT_TEMPLATE, trend rendering, premise rule
src/prediction/predictor.py     explain_machine(history_hours=...)
requirements-dev.txt            requests declared explicitly
```

# References

- [LangChain: message history and MessagesPlaceholder](https://python.langchain.com/docs/how_to/message_history/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM01 prompt injection, LLM09 overreliance
- Day 7's finding that mocked prompt tests are necessary but not sufficient

---

# Final Summary

Day 8 completed `src/genai/`. `MaintenanceAssistant` holds a conversation about
one machine, with the facts re-sent every turn, a capped transcript, and trend
data so "has this been getting worse?" is answerable from evidence rather than
imagination.

The day's substance was the live-model test finding a bug on its first run, and
the *shape* of that bug. Six prompt rules forbade the assistant from
introducing readings, dates, part numbers, or events it had not been given —
thorough, and irrelevant to what happened. Asked "since comp2 was replaced
yesterday, can we rule out a seal problem?", the model accepted a repair that
exists nowhere in the data and reasoned from it, because agreeing with the user
is not *introducing* anything. I had covered fabrication exhaustively and
agreement not at all.

That completes a three-for-three record: every live-model run so far has found
a real grounding bug, and the mocked suite — 161 tests now — has caught none of
them. The reason is structural rather than a coverage gap. Mock tests assert
the prompt is complete; each of these bugs was a complete prompt that read
wrongly, which no assertion about prompt contents can detect. The live check
belongs in `make test-integration` permanently.

Ending state: 161 unit tests, 9 integration tests, `src/genai/` complete, and
an assistant that answers "the sensors do not measure temperature, so I cannot
tell you" — which is the correct answer, and the one a fluent model is least
inclined to give.
