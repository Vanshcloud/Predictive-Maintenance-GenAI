# Releasing

How a version gets cut. Written for the maintainer, but public because a
contributor should be able to see what happens to their merged change.

---

## Versioning

[Semantic Versioning](https://semver.org). The public surface is **the REST
API, the CLI scripts, and the configuration schema** — not the Python
internals, which are not distributed as a library.

| Bump | When | Examples |
|---|---|---|
| **MAJOR** | A breaking change to that surface | Removing or renaming an endpoint · changing a response field's type · dropping a `.env` variable · changing the feature contract so old artifacts stop loading |
| **MINOR** | Backwards-compatible capability | A new endpoint · a new optional query parameter · a new script · a new optional setting |
| **PATCH** | Backwards-compatible fix | Bug fixes · docs · dependency bumps · performance work that changes no contract |

Two project-specific rules:

- **A retrained model is at least a MINOR bump.** Predictions change even when
  no code does, and a caller who pinned a version is entitled to stable
  behaviour.
- **A threshold change is at least a MINOR bump**, for the same reason — it
  changes which machines alert.

Pre-releases use `1.1.0-rc.1`.

---

## The version lives in three places

They must agree. `tests/unit/test_smoke.py::TestVersion` fails if they do not,
and the release workflow re-checks it against the tag.

| File | Field |
|---|---|
| `pyproject.toml` | `[project].version` |
| `config/settings.py` | `APP_VERSION` — reported by `/health` |
| `src/__init__.py` | `__version__` |

---

## Checklist

### 1. Confirm the tree is releasable

```bash
git checkout main && git pull
make quality
make test
make test-integration        # needs generated data + a trained model
```

- [ ] `make quality` clean — flake8, Black, isort, mypy in **both** modes
- [ ] Unit suite green
- [ ] Integration suite green, or the skips are understood
- [ ] `make docker-build` succeeds and the stack comes up healthy

### 2. Confirm the documentation is true

The most common release defect here is a number that moved and a document that
did not.

- [ ] Metrics in `README.md`, `docs/RESULTS.md`, and `docs/model.md` match
      `models/evaluation_report.json`
- [ ] Test counts in `README.md` and `docs/RESULTS.md` match `make test`
- [ ] `PREDICTION_THRESHOLD` matches the committed evaluation report
      (a test enforces this)
- [ ] `docs/roadmap.md` no longer lists anything that shipped
- [ ] No broken relative links — the `Docs` workflow checks this on every push

### 3. Bump the version

```bash
VERSION=1.1.0
sed -i '' "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
sed -i '' "s/^__version__ = .*/__version__ = \"$VERSION\"/" src/__init__.py
# config/settings.py: APP_VERSION — edit by hand, it carries a comment
python -m pytest tests/unit/test_smoke.py::TestVersion -q
```

### 4. Update the CHANGELOG

Move `[Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section and leave
`[Unreleased]` empty above it. Update the link references at the bottom.

**The release workflow extracts this section verbatim as the release notes**,
and fails if no section matching the tag exists. One source of truth, so the
notes cannot drift from the file people actually read.

- [ ] Entries are grouped: `Added` · `Changed` · `Deprecated` · `Removed` · `Fixed` · `Security`
- [ ] Each entry says what changed **and why it matters**, not just what moved
- [ ] Breaking changes are called out with a migration note

### 5. Commit, tag, push

```bash
git add -A
git commit -m "chore(release): v$VERSION"
git tag -a "v$VERSION" -m "v$VERSION"
git push origin main
git push origin "v$VERSION"
```

The tag push triggers `.github/workflows/release.yml`:

```
verify  → tag matches pyproject, CHANGELOG has the section
gate    → lint, format, types, full test suite
build   → sdist + wheel, twine check
images  → both container images build
publish → GitHub Release, notes from the CHANGELOG, distributions attached
```

The gate runs **before** the release is created. A tag that fails its own
quality bar should not become a published release; a release you have to delete
is worse than one that never appeared.

### 6. After

- [ ] The release appears with the right notes and attached artifacts
- [ ] `curl localhost:8000/health` reports the new `version`
- [ ] Add a new empty `[Unreleased]` section if you did not already

---

## If a release goes wrong

**Before anyone has pulled it** — delete and redo:

```bash
git tag -d v1.1.0
git push --delete origin v1.1.0
# delete the GitHub Release in the UI, fix, re-tag
```

**After** — do not retag. Ship `1.1.1`. A moving tag is worse than a bad
release, because it breaks the one thing a version number is for.

---

## Not published to PyPI

Deliberately. This is a service and a pipeline, not a library, and
`pip install predictive-maintenance-genai` would imply an API contract that is
not intended. The sdist and wheel are built anyway — so the packaging metadata
is proven to work — and attached to the release for anyone who wants them.

---

## Release notes template

The workflow generates notes from the CHANGELOG. When writing that section by
hand, this is the shape:

```markdown
### Added
- **Short title.** What it does, and why someone would want it.

### Changed
- **Short title.** What changed and what a caller must do differently.

### Fixed
- **Short title.** The symptom, the cause, and how to tell if you were affected.

### Security
- **Short title.** Severity, affected versions, and how to mitigate without upgrading.

---
**Breaking changes**
- `GET /old` is removed. Use `GET /new`, which returns the same shape plus `field`.

**Model**
- Retrained. Test F1 0.9086 → 0.91xx at threshold 0.34xx.
  Predictions will differ. Update `PREDICTION_THRESHOLD` and `RISK_BAND_HIGH`.

**Upgrading**
    docker compose -f docker/docker-compose.yml pull
    docker compose -f docker/docker-compose.yml up -d
```

---

## See also

- [`../CHANGELOG.md`](../CHANGELOG.md)
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — commit conventions
- [`deployment.md`](deployment.md#updating-the-model) — swapping a model without a rebuild
