## What this changes

<!-- The behaviour, not the diff. One or two sentences. -->

## Why

<!-- The diff shows what changed; this is where the reason goes. If this
     reverses an earlier decision, say which one and what changed. -->

## How it was verified

<!-- What you actually ran, not what should pass. -->

```
make quality
make test
```

- [ ] `make quality` passes (flake8, Black, isort, mypy in both modes)
- [ ] `make test` passes
- [ ] `make test-integration` run, or not applicable
- [ ] New behaviour has a test that fails without this change

## Correctness

`CONTRIBUTING.md` lists invariants that fail **silently** — producing
better-looking numbers rather than an error.

- [ ] The temporal split, train-only scaler fitting, and per-machine windowing are untouched, or the change is justified above
- [ ] `as_of` filtering still covers every table, not just telemetry
- [ ] The alert threshold and `RISK_BAND_HIGH` still agree
- [ ] The dashboard still derives no risk band of its own

## Documentation

- [ ] Docs updated in this same commit, or no docs change needed
- [ ] Any metric quoted is stated with the conditions that produced it
