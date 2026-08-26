# Documentation

Everything written about this project, and what each document is for.

---

## Start here

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Project overview: what it is, how to install and run it, and current results. |
| [**`IMPLEMENTATION_PLAN.md`**](IMPLEMENTATION_PLAN.md) | **The full engineering specification.** Objectives, scope, requirements, technology stack, system architecture, dataset documentation, model architecture, training and evaluation plan, deployment plan, coding standards, testing strategy, risk register, and milestones. |
| [`architecture.md`](architecture.md) | Design-level view: layer diagram, module responsibilities, the layer dependency rule, and the correctness invariants. |
| [**`RESULTS.md`**](RESULTS.md) | **Every metric in one place**, each with the caveat it needs — plus what the numbers do *not* establish. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Development setup, quality gates, code style, and the invariants a change must not break. |
| [`../SECURITY.md`](../SECURITY.md) | Threat model, what is and is not hardened, and how to report a vulnerability. |

---

## Engineering devlog

[`devlog/`](devlog/README.md) holds one entry per milestone, written as the work
was done — what was attempted, what broke, and why each decision went the way it
did. They are historical records and are not updated after the fact.

Start with [`devlog/day-04.md`](devlog/day-04.md), which documents a training
hang that produced no traceback and took two wrong diagnoses before the real
cause — two libraries statically linking incompatible copies of the same
threading primitive — was found.

---

## Reference

| Document | Purpose |
|---|---|
| [`images/`](images/) | Figures used by the README, regenerable with `scripts/plot_horizon.py`. |
| ~~`handoff.md`~~ | **Removed on Day 12.** A 1,377-line planning document, superseded section by section by `IMPLEMENTATION_PLAN.md`; its Day 1–3 detail was folded into `devlog/day-01.md`–`day-03.md`. Retrievable with `git show 9ceb349:docs/handoff.md`. It had been kept for eight days behind a "superseded" banner, which is its own kind of clutter: a 60 KB file whose only message is "read something else". |

---

## Conventions

- **The repository is the source of truth.** Where a document and the code
  disagree, the code is right and the document is a bug. Fix the document.
- **Documentation ships with the code that changed it** — same commit, never a
  follow-up.
- **Record what failed, not just what worked.** Every devlog entry has a *Bugs
  Encountered* section with root cause, fix, verification, and what it taught.
- **State caveats where the numbers are.** Metrics are quoted with the conditions
  that produced them.
