# Documentation Index

Everything written about this project, and what each document is for.

---

## Start here

| Document | Purpose | Updated |
|---|---|---|
| [**`../IMPLEMENTATION_PLAN.md`**](../IMPLEMENTATION_PLAN.md) | **Single source of truth.** Objectives, scope, requirements, technology stack, system architecture, dataset documentation, model architecture, training pipeline, evaluation plan, deployment plan, coding standards, testing strategy, risk register, milestones, roadmap, and current project status. | Every session |
| [`../README.md`](../README.md) | Public-facing overview: what the project is, how to install and run it, and current results. | Every milestone |
| [`../CLAUDE.md`](../CLAUDE.md) | Repository conventions and non-negotiable invariants, written for AI agents working in this codebase. | When conventions change |

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
| [Day 4](Day4.md) | LSTM architecture & training | Trained model — AUC 0.9999, F1 0.7530 — 75 tests, and an abseil symbol-collision deadlock diagnosed and fixed |
| Day 5 | Model evaluation & optimization | 🔒 Next |
| Days 6–12 | Inference → GenAI → API → dashboard → deployment → demo | 🔒 Planned |

> These are **daily** reports, not weekly. The project is structured as 12 one-day
> milestones; see the Roadmap section of `IMPLEMENTATION_PLAN.md`.

---

## Reference

| Document | Purpose |
|---|---|
| [`architecture.md`](architecture.md) | Design-level system architecture: layer diagram, technology stack, design principles, the layer dependency rule, and the non-negotiable invariants. Describes the finished system, including layers not yet built. |
| [`handoff.md`](handoff.md) | **Historical.** The original long-form narrative handoff document, accurate through Day 3 and **frozen on 2026-08-22**. Superseded by `IMPLEMENTATION_PLAN.md`; kept because it is the source from which the retroactive `Day1.md`–`Day3.md` reports were reconstructed. Where it disagrees with the plan, the plan wins. |

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
