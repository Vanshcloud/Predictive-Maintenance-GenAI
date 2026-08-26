# Examples

Runnable, and verified against a live API rather than written from the docs.

All of them need the API running:

```bash
make docker-up-d          # or: make run-api
```

---

| File | What it does |
|---|---|
| [`api_client_demo.py`](api_client_demo.py) | Six-step walkthrough of the whole API, including what each failure mode looks like |
| [`curl_examples.sh`](curl_examples.sh) | Every endpoint from the shell, plus the error shapes |
| [`predict_request.json`](predict_request.json) | A valid `POST /predict` body, generated from the committed fixture |
| [`.env.example`](.env.example) | An annotated configuration, with the three decisions that matter |

---

## `api_client_demo.py`

```bash
python examples/api_client_demo.py
python examples/api_client_demo.py --machine 96 --as-of 2024-11-13T12:00:00
python examples/api_client_demo.py --no-report      # skip the ~21 s LLM call
python examples/api_client_demo.py --as-of none     # latest reading instead
```

Walks readiness → inventory → fleet → point-in-time → evidence → report. It
reuses `dashboard/api_client.py` rather than adding a fourth HTTP wrapper, so
it also demonstrates the three failure types a client has to tell apart.

**The interesting step is 4.** At the dataset's latest hour nothing is
alerting, which makes the model look inert. Rewound to `2024-10-31T06:00:00`,
two machines are critical — six hours before machine 51 actually failed.

---

## `curl_examples.sh`

```bash
bash examples/curl_examples.sh
API=http://staging:8000 MACHINE=96 bash examples/curl_examples.sh
```

Uses `jq` when it is installed and falls back cleanly when it is not. The last
section deliberately triggers a `404` and a `422` so the error envelope is
visible.

---

## `predict_request.json`

A valid body for `POST /predict` — 48 consecutive hourly readings for machine 1,
taken from the committed `data/sample/` fixture.

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @examples/predict_request.json
```

It was **generated from the fixture and validated against the real
`PredictRequest` schema**, not hand-written. Two constraints make hand-writing
one annoying:

- **At least 48 readings.** Feature engineering consumes the first 24 hours for
  rolling and lag windows; the LSTM needs 24 more to form one sequence.
- **Physical bounds per sensor** — voltage 0–500 V, rotation 0–1500 RPM,
  pressure 0–400 PSI, vibration 0–300 mm/s. Anything outside is rejected with a
  `422` rather than scored.

To regenerate it for a different machine:

```python
import json, pandas as pd

tel = pd.read_csv("data/sample/telemetry.csv")
tel["datetime"] = pd.to_datetime(tel["datetime"])
rows = tel[tel.machine_id == 1].sort_values("datetime").tail(48)

payload = {
    "machine_id": 1,
    "readings": [
        {
            "datetime": r["datetime"].isoformat(),
            "voltage": round(float(r["voltage"]), 3),
            "rotation": round(float(r["rotation"]), 3),
            "pressure": round(float(r["pressure"]), 3),
            "vibration": round(float(r["vibration"]), 3),
        }
        for _, r in rows.iterrows()
    ],
}
json.dump(payload, open("examples/predict_request.json", "w"), indent=2)
```

> Sample-fixture readings score near zero. `data/sample/` deliberately contains
> **no failure events**, so a low probability here is correct, not a bug. For an
> interesting score, use the full generated dataset and the `as_of` timestamps
> above.

---

## What is deliberately not here

**No notebook.** `notebooks/` is scratch space and is not part of the tested
surface. Anything worth keeping belongs in `scripts/` where CI runs it.

**No training example.** Training is four commands, documented in
[`../docs/training.md`](../docs/training.md); wrapping them in a script that
could drift from the real ones would be worse than the commands themselves.

---

## See also

- [`../docs/api.md`](../docs/api.md) — the full endpoint reference
- [`../docs/dashboard.md`](../docs/dashboard.md) — the UI these calls back
- [`../docs/troubleshooting.md`](../docs/troubleshooting.md) — when one of these fails
