# Engineering devlog

A build journal kept while this project was developed: one entry per milestone,
written at the end of it.

Each entry records what was attempted, what was built, **what went wrong**, and
which decisions were made and why. They are historical records — they describe
the state of the repository on the day they were written and are not updated
afterwards. For current behaviour see [`../architecture.md`](../architecture.md),
for current numbers see [`../RESULTS.md`](../RESULTS.md), and for overall scope
and status see [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

They are kept because the failures are the useful part. A repository that only
records what worked teaches nothing about why the code looks the way it does —
and several things in this codebase look wrong until you know what they prevent.

**If you read one entry, read [`day-04.md`](day-04.md)**: a training run that
hung at 0% CPU with no traceback, misdiagnosed twice, and eventually traced to
two libraries statically linking incompatible copies of the same threading
primitive. The import-order comments scattered through `src/` exist because of
it.

---

## The build

| # | Focus | Outcome |
|---|---|---|
| [01](day-01.md) | Project setup and foundation | Typed configuration, logging, exception hierarchy, testing infrastructure |
| [02](day-02.md) | Dataset, EDA, and data pipeline | 883,231-row synthetic dataset; ingestion and validation |
| [03](day-03.md) | Feature engineering and preprocessing | 63 features, leak-free temporal split, `(698400, 24, 63)` tensors |
| [04](day-04.md) | LSTM architecture and training | Trained model — and an abseil symbol-collision deadlock diagnosed and fixed |
| [05](day-05.md) | Model evaluation and optimisation | Clean three-way split; threshold swept on validation |
| [06](day-06.md) | Prediction pipeline and inference | Training/serving parity verified at **100%** over 172,800 sequences |
| [07](day-07.md) | Report generation | Grounded reports; three hallucination bugs found by a live model |
| [08](day-08.md) | Conversational assistant | Multi-turn Q&A that declines what the data cannot answer |
| [09](day-09.md) | REST API | Nine endpoints, **137 ms** predictions, LLM path isolated |
| [10](day-10.md) | Dashboard | Pure API client; risk bands owned by the API |
| [11](day-11.md) | Containers and CI | Two images built and verified; build context 7.3 GB → 2.9 MB |
| [12](day-12.md) | Verification and consolidation | Clean-checkout verified; dataset reproducible from seed |

## After the build

Enhancement rather than construction.

| # | Focus | Outcome |
|---|---|---|
| [13](day-13.md) | Point-in-time assessment | Rewind the fleet to any hour; **5/5 alert at 6 h, 0/5 at 36 h** |
| [14](day-14.md) | Quality gates and accessibility | Local and CI gates unified; two WCAG AA contrast failures fixed |
| [15](day-15.md) | Production review | Training seeded and retrained — **F1 0.9086**; `/fleet` cache bounded |

---

## Conventions

- **The repository is the source of truth.** Where an entry and the code
  disagree, the code is right and the entry is a historical record of something
  that has since changed.
- **Record what failed, not just what worked.** Every entry has a *Bugs
  Encountered* section with root cause, fix, verification, and what it taught.
- **State caveats where the numbers are.** Metrics are quoted with the
  conditions that produced them.
