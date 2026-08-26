# Support

Setting expectations honestly, so nobody waits on a response that was never
coming.

---

## What this project is

A **reference implementation and portfolio project**, maintained by one person
in their own time. It is complete and tested, but it is not a product and there
is no support contract behind it.

That shapes what you can reasonably expect:

| | |
|---|---|
| **Bug reports** | Read and triaged. Fixes when I can — no timeline promised |
| **Questions** | Answered when I have time; the docs are usually faster |
| **Feature requests** | Welcome as ideas. Most will land on the [roadmap](docs/roadmap.md) rather than get built |
| **Pull requests** | Genuinely welcome, and reviewed properly. See [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Security reports** | **Prioritised.** See [SECURITY.md](SECURITY.md) |
| **Production support** | None. Do not depend on this for anything that matters without reading [the limitations](docs/model.md#limitations) first |

---

## Before opening an issue

Most reported problems are already answered, with the fix:

1. **[`docs/troubleshooting.md`](docs/troubleshooting.md)** — symptoms and
   causes, ordered by how often they come up. Start here.
2. **[The FAQ](README.md#faq)** — the questions that come up most.
3. **[`docs/`](docs/README.md)** — API reference, model card, deployment,
   development guide.
4. **[Existing issues](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/issues?q=is%3Aissue)**
   — including closed ones.

Three things account for most reports, and all three are already documented:

| Symptom | Answer |
|---|---|
| `Could not find a version that satisfies the requirement tensorflow` | Python 3.13+. Use 3.10–3.12 |
| Training hangs at 0% CPU with no traceback | Import order — [full diagnosis](docs/troubleshooting.md#the-training-run-hangs-with-no-error) |
| `503` from every endpoint | No trained model. `curl localhost:8000/health` confirms it |

---

## Opening a good issue

Use a [template](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/issues/new/choose)
— bug report, feature request, or documentation.

Include:

```bash
python --version
git rev-parse --short HEAD
make test 2>&1 | tail -30
```

If it involves a prediction, **include the machine id and the `as_of`
timestamp**. The point-in-time behaviour makes those essential — without them a
prediction cannot be reproduced.

---

## What is out of scope

Said plainly, so nobody spends effort on a report that will be closed:

- **Help with your own dataset or model.** The pipeline is documented and the
  code is readable, but adapting it is your work.
- **Deployment consulting** for a specific cloud or cluster.
- **Requests to make it production-ready for you.** The gaps are listed in
  [`SECURITY.md`](SECURITY.md) and [`docs/roadmap.md`](docs/roadmap.md) — they
  are known, and closing them is a deliberate decision with a cost.
- **Anything depending on real-world accuracy claims.** The dataset is
  synthetic. See [limitations](docs/model.md#limitations).

---

## Security

**Do not open a public issue for a security problem.** Use
[private reporting](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/security/advisories/new)
or the email in [`SECURITY.md`](SECURITY.md), which also carries the threat
model — several things people might report are documented scope decisions
rather than defects.
