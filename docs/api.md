# API Reference

Nine endpoints. Interactive documentation is generated from the same Pydantic
schemas the code validates against, so it cannot drift from the behaviour:

- **Swagger UI** — <http://localhost:8000/docs>
- **ReDoc** — <http://localhost:8000/redoc>
- **OpenAPI JSON** — <http://localhost:8000/openapi.json>

```bash
make run-api        # uvicorn on :8000
```

> **No authentication.** Every endpoint is open. This service is designed for
> `localhost` or a trusted network — see [`../SECURITY.md`](../SECURITY.md)
> before exposing it.

---

## Contents

- [Conventions](#conventions)
- [Health](#health)
- [Machines](#machines)
- [Predictions](#predictions)
- [Reports](#reports)
- [Error codes](#error-codes)
- [Python client](#python-client)

---

## Conventions

### Point-in-time assessment (`as_of`)

Every prediction endpoint accepts an optional `as_of` timestamp. Omitted, it
means "the latest reading". Supplied, **everything after it is hidden —
telemetry, errors, and maintenance alike**, so a historical assessment cannot
see data that did not exist yet.

Filtering telemetry alone would leak, because `errors_last_24h` and
`hours_since_maintenance` are model features. The cutoff is inclusive: the
chosen hour has already happened.

```bash
# What did we know about machine 51 at 06:00 on the day it failed?
curl "localhost:8000/machines/51/predict?as_of=2024-10-31T06:00:00"
```

### Risk bands

`risk_level` is assigned by the API and is never recomputed by clients.
`high` begins exactly at the alert threshold, so "high or above" means
precisely "the model is alerting".

| Band | Range | Meaning |
|---|---|---|
| `low` | `< 0.15` | Nothing indicated |
| `medium` | `0.15 – 0.3415` | Worth watching |
| `high` | `≥ 0.3415` | **Alerting** — `will_fail` is `true` |
| `critical` | `≥ 0.90` | Alerting, near-certain |

---

## Health

### `GET /health`

Liveness **and** readiness. `status` is `ok` only when predictions can actually
be served; a process that is running but cannot predict is not healthy, and
reporting otherwise defeats the point of the check.

```bash
curl localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "lstm_predictive_maintenance",
  "dataset_loaded": true,
  "machines_known": 100,
  "threshold": 0.3415,
  "version": "1.0.0",
  "data_start": "2024-01-01T00:00:00",
  "data_end": "2024-12-30T23:00:00"
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | `"ok"` \| `"degraded"` | `degraded` when the model or dataset failed to load |
| `machines_known` | int | 0 when degraded |
| `threshold` | float | The alert threshold currently in force |
| `data_start` / `data_end` | datetime \| null | The window `as_of` may address |

**Always returns 200.** An API that refuses to start because the model is
missing cannot serve `/health`, which is exactly what an operator needs in
order to discover that the model is missing.

---

## Machines

### `GET /machines`

Every machine this instance knows about. Requires no scoring.

```bash
curl localhost:8000/machines
```

```json
[
  {
    "machine_id": 1,
    "model": "model3",
    "age": 18,
    "readings_available": 8760,
    "first_reading": "2024-01-01T00:00:00",
    "last_reading": "2024-12-30T23:00:00"
  }
]
```

### `GET /machines/{machine_id}`

Static facts about one machine. Same shape as an element above.

**Errors:** `404` if the machine is not in the dataset · `503` if no dataset is loaded.

---

## Predictions

### `GET /machines/{machine_id}/predict`

Score one machine from the dataset this instance has loaded. **~137 ms.**

| Parameter | In | Type | Default | Notes |
|---|---|---|---|---|
| `machine_id` | path | int | — | Must exist in the dataset |
| `as_of` | query | datetime | latest | Hide everything after this moment |

```bash
curl "localhost:8000/machines/51/predict?as_of=2024-10-31T06:00:00"
```

```json
{
  "machine_id": 51,
  "datetime": "2024-10-31 06:00:00",
  "failure_probability": 0.9999940395355225,
  "risk_level": "critical",
  "will_fail": true,
  "threshold": 0.3415
}
```

### `GET /machines/{machine_id}/explain`

The prediction **plus the evidence behind it** — the payload the report
generator and the dashboard both consume. Every number is read from the
engineered features the model actually consumed; none is derived for
presentation.

| Parameter | In | Type | Default | Constraints |
|---|---|---|---|---|
| `history_hours` | query | int | `24` | `0 – 168` |
| `as_of` | query | datetime | latest | — |

```bash
curl "localhost:8000/machines/51/explain?as_of=2024-10-31T06:00:00"
```

```json
{
  "machine_id": 51,
  "datetime": "2024-10-31 06:00:00",
  "failure_probability": 0.9999940395355225,
  "risk_level": "critical",
  "will_fail": true,
  "threshold": 0.3415,
  "age_years": 12,
  "errors_last_24h": 3,
  "hours_since_maintenance": {"comp1": 412, "comp2": 9999},
  "most_deviant_sensors": ["pressure", "vibration", "rotation"],
  "sensors": {
    "pressure": {
      "current": 65.89,
      "baseline_24h": 93.47,
      "change_24h": -32.06,
      "volatility_24h": 14.4,
      "deviation_sigma": -1.91,
      "unit": "PSI",
      "direction": "below",
      "is_concerning": true,
      "typical_cause": "a leak or a failing seal"
    }
  }
}
```

`typical_cause` is supplied **only** when a reading deviates in the direction
that actually matters. Pressure 0.7σ *high* is not a pressure drop, and
attaching the leak explanation to it would produce a report that contradicts
its own numbers. `hours_since_maintenance` uses `9999` as a sentinel for "no
record" — `0` would mean "serviced this hour", the exact opposite.

### `GET /machines/{machine_id}/history`

Recent hourly sensor readings. No scoring.

| Parameter | In | Type | Default | Constraints |
|---|---|---|---|---|
| `hours` | query | int | `48` | `1 – 720` |
| `as_of` | query | datetime | latest | — |

```json
[{"datetime": "2024-10-31T05:00:00", "voltage": 172.31, "rotation": 400.04, "pressure": 65.89, "vibration": 41.22}]
```

### `GET /fleet`

Score every machine, most urgent first. **~13.4 s cold, ~2 ms cached** (5-minute
TTL, keyed by `as_of`, bounded to 16 entries).

Concurrent requests for the same uncached `as_of` are serialised — the first
computes, the rest wait and receive the cached result. Cache *hits* never block.

| Parameter | In | Type | Default | Notes |
|---|---|---|---|---|
| `alerts_only` | query | bool | `false` | Return only machines at or above threshold |
| `refresh` | query | bool | `false` | Bypass the cache and recompute |
| `as_of` | query | datetime | latest | Assess the whole fleet at this moment |

```bash
curl "localhost:8000/fleet?alerts_only=true&as_of=2024-10-31T06:00:00"
```

```json
{
  "machines_assessed": 100,
  "machines_alerting": 2,
  "threshold": 0.3415,
  "generated_at": "2026-08-26T17:41:03.221Z",
  "predictions": [{"machine_id": 51, "failure_probability": 0.99999, "risk_level": "critical", "will_fail": true, "datetime": "2024-10-31 06:00:00", "threshold": 0.3415}]
}
```

### `POST /predict`

Score readings supplied in the request rather than from stored data. **This is
the endpoint a real plant uses** — its own sensors are the source of truth, not
a CSV on the API host.

**Request**

| Field | Type | Required | Constraints |
|---|---|---|---|
| `machine_id` | int | ✅ | — |
| `readings` | array | ✅ | **≥ 48** consecutive hourly readings, oldest first |
| `model` | string | ➖ | e.g. `model3`; improves accuracy if known |
| `age` | int | ➖ | `0 – 100` |

Each reading requires `datetime`, `voltage`, `rotation`, `pressure`,
`vibration`, range-checked against physical bounds:

| Sensor | Unit | Valid range |
|---|---|---|
| `voltage` | V | `0 – 500` |
| `rotation` | RPM | `0 – 1500` |
| `pressure` | PSI | `0 – 400` |
| `vibration` | mm/s | `0 – 300` |

A value outside these is a broken sensor or a unit mix-up. Either way it is
rejected at the door rather than scored, because the model has never seen
values like that and its output would be meaningless rather than merely wrong.

**Why 48 readings and not 24?** Feature engineering consumes the first 24 hours
for rolling and lag windows, and the LSTM needs 24 more to form one sequence.

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @examples/predict_request.json
```

Response is identical to `GET /machines/{id}/predict`.

---

## Reports

### `POST /report`

Generate a plain-English maintenance report, or answer a question about one
machine. **~21 s** against a local model.

| Field | Type | Required | Notes |
|---|---|---|---|
| `machine_id` | int | ✅ | — |
| `provider` | `"openai"` \| `"google"` \| `"ollama"` | ➖ | Defaults to whichever has credentials, preferring OpenAI → Google → keyless Ollama |
| `model` | string | ➖ | Override the provider's model, e.g. an Ollama tag you have pulled |
| `question` | string | ➖ | Ask something specific instead of a full report |
| `as_of` | datetime | ➖ | A report about a past moment must not cite facts from after it |

```bash
curl -X POST localhost:8000/report \
  -H 'Content-Type: application/json' \
  -d '{"machine_id": 51, "as_of": "2024-10-31T06:00:00"}'
```

**The prediction survives an LLM failure.** A provider outage returns `502` and
a timeout returns `504` — both **with the prediction in the error detail**. The
model's answer never depends on the language model.

Every figure the report quotes comes from the `/explain` record it is handed.
It is given nothing else.

---

## Error codes

Errors map to status codes through the exception hierarchy in
`src/utils/exceptions.py` — one handler per architectural layer, with no string
matching on error messages.

| Status | `error_type` | Cause |
|---|---|---|
| `404` | `ResourceNotFoundError` | No such machine |
| `422` | `DataValidationError` | The request body is wrong |
| `422` | `PredictionError` | The input could not be scored — usually too little history |
| `422` | `ReportGenerationError` | The report input was wrong |
| `502` | `LLMConnectionError` | Provider unreachable — **retryable**, prediction attached |
| `503` | `ModelNotFoundError` | This instance cannot serve; check `/health` |
| `504` | — | The language model exceeded 120 s; prediction attached |
| `500` | `InternalServerError` | Unanticipated. Opaque body + correlation id |

```json
{
  "detail": "Machine 999 is not in the dataset. Known machines: [1, 2, 3, 4, 5]...",
  "error_type": "ResourceNotFoundError",
  "correlation_id": null
}
```

A `500` returns an opaque message and a `correlation_id`; the detail goes to the
logs, where an operator can find it, and not to a client who may be hostile.
Grep the logs for that id.

---

## Python client

The dashboard ships a small synchronous client that already distinguishes the
failure modes a UI must tell apart. Reuse it rather than writing another:

```python
from dashboard.api_client import APIClient, APIUnavailable, APIDegraded, APIError

client = APIClient("http://localhost:8000")

client.require_ready()                       # raises APIDegraded if it cannot predict
fleet = client.fleet(alerts_only=True)
record = client.explain(51, as_of="2024-10-31T06:00:00")

for name in record["most_deviant_sensors"]:
    s = record["sensors"][name]
    verdict = s["typical_cause"] if s["is_concerning"] else "normal"
    print(f"{name}: {s['current']} {s['unit']} ({s['deviation_sigma']}σ) -> {verdict}")
```

| Exception | Meaning | What to tell a user |
|---|---|---|
| `APIUnavailable` | Nothing answered | "Is the API running?" |
| `APIDegraded` | Up, but cannot predict | "Model not loaded" |
| `APIError` | Answered 4xx/5xx | Show the server's message; `.status_code` is set |

A runnable end-to-end example is in
[`../examples/api_client_demo.py`](../examples/api_client_demo.py).

Or with plain `requests`:

```python
import requests

r = requests.get(
    "http://localhost:8000/machines/51/predict",
    params={"as_of": "2024-10-31T06:00:00"},
    timeout=30,
)
r.raise_for_status()
print(r.json()["failure_probability"])
```

---

## See also

- [`architecture.md`](architecture.md#request-lifecycle) — why the LLM path is isolated
- [`model.md`](model.md) — what the probability means and how the threshold was chosen
- [`deployment.md`](deployment.md) — running this behind a reverse proxy
- [`../SECURITY.md`](../SECURITY.md) — what is not hardened
