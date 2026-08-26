# Documentation

Everything written about this project, and what each document is for.

---

## Start here

| Document | For |
|---|---|
| [`../README.md`](../README.md) | What this is, how to run it, and what it achieves |
| [`architecture.md`](architecture.md) | **How it fits together** — layer diagram, module responsibilities, and the correctness invariants |
| [`api.md`](api.md) | Every endpoint: schemas, error codes, `curl` and Python examples |
| [`model.md`](model.md) | Why an LSTM, how the threshold was chosen, and **what the numbers do not establish** |

## Running it

| Document | For |
|---|---|
| [`training.md`](training.md) | Reproducing the model from a clean checkout |
| [`deployment.md`](deployment.md) | Containers, configuration, health checks, reverse proxy, logging |
| [`dashboard.md`](dashboard.md) | The Streamlit UI and the `Rewind` control |
| [`troubleshooting.md`](troubleshooting.md) | **Check here first.** Symptoms, causes, fixes |

## Working on it

| Document | For |
|---|---|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Setup, quality gates, and the invariants a change must not break |
| [`development.md`](development.md) | Debugging, testing, common mistakes, architecture philosophy |
| [`releasing.md`](releasing.md) | Versioning policy and the release checklist |
| [`../SECURITY.md`](../SECURITY.md) | Threat model, and how to report a vulnerability |

## Evidence

| Document | For |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **Every metric in one place**, each with the caveat it needs |
| [`benchmarks.md`](benchmarks.md) | Measured latency, memory, and artifact sizes — with the method for each |
| [`roadmap.md`](roadmap.md) | What it does not do yet, and why |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Release history |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | The full engineering specification: scope, requirements, risk register, milestones |

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

| | |
|---|---|
| [`images/`](images/README.md) | Figures, and how to regenerate them |
| [`../examples/`](../examples/README.md) | Runnable API examples |
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
