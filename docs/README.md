# Documentation Index

Everything written about this project, and what each document is for.

---

## Start here

| Document | Purpose | Updated |
|---|---|---|
| [**`../IMPLEMENTATION_PLAN.md`**](../IMPLEMENTATION_PLAN.md) | **Single source of truth.** Objectives, scope, requirements, technology stack, system architecture, dataset documentation, model architecture, training pipeline, evaluation plan, deployment plan, coding standards, testing strategy, risk register, milestones, roadmap, and current project status. | Every session |
| [`../README.md`](../README.md) | Public-facing overview: what the project is, how to install and run it, and current results. | Every milestone |
| [`../AGENTS.md`](../AGENTS.md) | Repository conventions and non-negotiable invariants, written for AI agents working in this codebase. | When conventions change |

---

## Daily implementation reports

One file per implementation day. Each follows the same structure: objective, starting
state, tasks planned, work completed, code changes, training progress, testing, bugs
encountered, design decisions, remaining tasks, next-day plan, and project health.

| Day | Focus | Outcome |
|---|---|---|
| [Day 1](Day1.md) | Project setup & foundation | 33 files, typed config, logging, exception hierarchy, 19 tests |
| [Day 2](Day2.md) | Dataset, EDA & data pipeline | 883,231-row synthetic dataset, ingestion + validation, 41 tests |
| [Day 3](Day3.md) | Feature engineering & preprocessing | 63 features, leak-free temporal split, `(698400, 24, 63)` tensors, 68 tests |
| [Day 4](Day4.md) | LSTM architecture & training | Trained model — 75 tests, and an abseil symbol-collision deadlock diagnosed and fixed |
| [Day 5](Day5.md) | Model evaluation & optimization | Clean 3-way split — **F1 0.8949, 26 false alarms** — 90 tests; TD-1/2/3/6 repaid |
| [Day 6](Day6.md) | Prediction pipeline & inference | `Predictor` — training/serving parity **100%**, event-level recall **8/8** — 113 unit + 4 integration tests |
| [Day 7](Day7.md) | LangChain setup & report generation | Grounded reports — three hallucination bugs found by a live model, R-10 closed — 141 tests |
| [Day 8](Day8.md) | GenAI assistant & maintenance Q&A | Multi-turn sessions; live-model tests caught the assistant accepting a false premise — 161 unit + 9 integration tests |
| [Day 9](Day9.md) | FastAPI REST API | 9 endpoints, **137 ms** predictions, LLM path isolated — 185 unit + 9 integration tests |
| [Day 10](Day10.md) | Streamlit dashboard | Pure API client — three views, risk colours owned by the API — 211 unit + 9 integration tests |
| [Day 11](Day11.md) | Docker, CI/CD & deployment | 2 images built and verified (**2.87 GB** / **803 MB**), compose stack healthy, build context 7.3 GB → 2.9 MB |
| [Day 12](Day12.md) | Final polish, docs & demo | Clean-checkout verified (193 pass / 18 skip), dataset MD5 reproducible, TD-4 closed |
| [Day 13](Day13.md) | Point-in-time assessment (`as_of`) | Rewind the fleet to any hour; **5/5 alert 6 h before failure, 0/5 at 36 h** — 229 unit + 13 integration tests |
| [Day 14](Day14.md) | Quality-gate drift & accessibility | Local and CI gates unified on one path list; two WCAG AA contrast failures fixed; the horizon chart |
| [Day 15](Day15.md) | Production review | Training seeded and retrained — **F1 0.9086, t=0.3415**; `/fleet` cache bounded; Windows-safe file I/O — 238 unit + 13 integration tests |

> These are **daily** reports, not weekly. The project is structured as 12 one-day
> milestones, followed by post-project enhancement sessions (Days 13–15); see the
> Roadmap section of `IMPLEMENTATION_PLAN.md`.

---

## Reference

| Document | Purpose |
|---|---|
| [**`RESULTS.md`**](RESULTS.md) | **Every metric in one place**, each with the caveat it needs, plus what the numbers do *not* establish. |
| [`architecture.md`](architecture.md) | Design-level system architecture: layer diagram, technology stack, design principles, the layer dependency rule, and the non-negotiable invariants. Describes the finished system, including layers not yet built. |
| ~~`handoff.md`~~ | **Removed on Day 12.** The original 1,377-line handoff document, superseded section-by-section by `IMPLEMENTATION_PLAN.md`. Its Day 1–3 detail lives on in `Day1.md`–`Day3.md`, which were reconstructed from it. Nothing is lost — retrieve it with `git show 9ceb349:docs/handoff.md`. It was kept for eight days with a "superseded" banner, which is its own kind of clutter: a 60 KB file whose only message is "read something else". |

---

## Conventions

- **The repository is the source of truth.** Where a document and the code disagree, the
  code is right and the document is a bug. Fix the document.
- **Documentation ships with the code that changed it** — same commit, never a follow-up.
- **Each day updates exactly two files**: `IMPLEMENTATION_PLAN.md` (at minimum its
  *Current Project Status* section) and that day's `DayN.md`.
- **Record what failed, not just what worked.** Every `DayN.md` has a *Bugs Encountered*
  section with root cause, fix, verification, and lessons learned — the Day 4 deadlock
  writeup exists so nobody re-derives that diagnosis from scratch.
- **State caveats where the numbers are.** Metrics are quoted with the conditions that
  produced them; see the TD-1 note wherever Day 4's AUC appears.
