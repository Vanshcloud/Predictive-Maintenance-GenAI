# Dashboard

A Streamlit UI for maintenance supervisors. Three views, one control that makes
the model's behaviour visible.

```bash
make run-api            # required — the dashboard holds no model
make run-dashboard      # http://localhost:8501
```

---

## Contents

- [What it is](#what-it-is)
- [Views](#views)
- [Rewind](#rewind)
- [Configuration](#configuration)
- [When something is wrong](#when-something-is-wrong)
- [Accessibility](#accessibility)
- [Extending it](#extending-it)

---

## What it is

**A pure HTTP client.** It imports nothing from `src/` — no TensorFlow, no
model file, no preprocessing. Everything on screen arrived over the wire.

That is not stylistic. It is what lets the UI and the API be built, deployed,
scaled, and restarted independently, and it is why the dashboard's container
image is **803 MB** against the API's **2.87 GB**. Its `Dockerfile` copies only
`dashboard/`, so a future import from `src/` breaks the build — the intended
failure.

```
dashboard/
├── app.py          # the Streamlit script — views and layout
├── api_client.py   # one method per endpoint, three distinct failure types
└── risk.py         # pure presentation: the colour palette and the badge
```

`risk.py` exists separately because `app.py` is a *script*: importing it
executes it, which reaches the network on the first line of the sidebar. The
pure logic lives where it can be tested without a running app.

---

## Views

### Fleet overview

Every machine, most urgent first.

- Four metrics: machines assessed, alerting, the active threshold, and
  critical + high combined.
- A sortable table with a risk badge per machine.
- A risk-distribution bar chart.
- **Alerts only** filters to machines at or above threshold.
- **Refresh** bypasses the API's 5-minute fleet cache.

When nothing is alerting it says so explicitly, rather than showing an empty
table that reads as broken.

### Machine detail

One machine, with the evidence behind its score.

- Probability, risk badge, and alerting state.
- **Sensor evidence** — current reading, 24 h baseline, change, deviation in
  sigma, and a verdict per sensor.
- Hours since last maintenance per component (`no record` rather than a
  misleading `9999`).
- Faceted line charts for all four sensors over a selectable window (24–336 h).

The verdict column comes from the API. A sensor is marked concerning only when
it deviates in the direction that matters — pressure 0.7σ *high* is not a
pressure drop, and labelling it one would produce a screen that contradicts its
own numbers.

### AI report

A plain-English maintenance report generated from the prediction and its
evidence.

- Optional free-text question instead of a full report.
- Provider selector (default / OpenAI / Google / Ollama) and a model override.
- The spinner says *"Calling the language model — this usually takes 20–30
  seconds"*, because a bare spinner on a 21-second call reads as a hang.

**If the language model fails, the prediction is still shown.** A `502` renders
as "the language model is unavailable — the prediction is unaffected", not as
an error page.

---

## Rewind

The control that makes the model's behaviour visible. Toggle **Rewind** in the
sidebar, pick a date and hour, and the entire dashboard reassesses the fleet as
of that moment — with everything after it hidden.

Without it the dashboard can only ever assess the dataset's final hour, which
on this data is a quiet one. The demo then shows "0 alerting" and looks broken
rather than calm.

**Try these**, both verifiable against `data/raw/failures.csv`:

| Setting | What you see |
|---|---|
| 2024-10-31, hour 6 | Machine 51 at ~0.99999 — critical, six hours before it failed |
| 2024-11-13, hour 12 | Machine 96 alerting |
| 2024-10-30, any hour | Machine 51 **silent** |

That last row is the point. The model was trained to see 24 hours ahead and no
further, so a day before the failure it says nothing — and should. That is the
horizon, not a bug.

The API hides everything after the chosen timestamp — **telemetry, errors, and
maintenance alike**. Filtering telemetry alone would leak, because
`errors_last_24h` and `hours_since_maintenance` are model features.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | In Docker this is `http://api:8000` — inside the compose network `localhost` is the dashboard itself |
| `DASHBOARD_PORT` | `8501` | Also the origin the API's CORS policy allows |

The sidebar's **API URL** field overrides it at runtime, which is useful for
pointing a local dashboard at a staging API.

> In a shared deployment that field is a server-side request forgery vector —
> the Streamlit *server* performs the fetch. Fine for single-user local use;
> see [`../SECURITY.md`](../SECURITY.md).

Timeouts: 30 s for predictions, **150 s** for reports. The report timeout is
deliberately longer than the API's own 120 s ceiling — a client that gives up
first abandons work that would have succeeded.

---

## When something is wrong

Three failure modes, three different messages, because they need different
actions:

| State | What it says | What to do |
|---|---|---|
| `APIUnavailable` | "Cannot reach the API" | `make run-api`, or fix the URL |
| `APIDegraded` | "Running but cannot serve predictions" | Model or dataset missing on the API host |
| `APIError` | The server's own message | Depends |

Collapsing these into "something went wrong" would produce a dashboard that
tells a user nothing when it could tell them exactly what to restart.

More in [`troubleshooting.md`](troubleshooting.md#the-dashboard).

---

## Accessibility

Risk colours are **keyed off the `risk_level` string the API assigns** and are
never recomputed from a probability. A second source of truth here would let
the dashboard show "medium" for a machine the API is alerting on — an
inconsistency that looks correct from either side.

Badge text is white at 0.8 em bold, which is *normal* text under WCAG 2.1 and
therefore needs 4.5:1, not the 3:1 large-text allowance:

| Level | Colour | Contrast |
|---|---|---|
| `critical` | `#b3202c` | 6.65:1 |
| `high` | `#c2410c` | 5.18:1 |
| `medium` | `#a16207` | 4.92:1 |
| `low` | `#15803d` | 5.02:1 |
| unknown | `#6b7280` | 4.83:1 |

The amber and yellow this started with measured 3.19:1 and 2.94:1 — fine on a
good monitor in a dim room, unreadable in a lit workshop, and they were the two
levels a supervisor scans the fleet table for. **The ratios are asserted by
tests**, because contrast is a number and shipped broken once by being
eyeballed instead.

An unrecognised level renders a readable grey badge rather than an unstyled
sliver, and the text is HTML-escaped before interpolation.

---

## Extending it

Keep the boundary intact:

- ✅ Add a view that calls a new API endpoint.
- ✅ Add a chart over data the API already returns.
- ❌ Import from `src/` — it breaks the container build, by design.
- ❌ Derive a risk band from a probability — a test greps for threshold
  literals in the source.

Pure logic goes in `risk.py` (or a sibling), where it can be tested without
Streamlit. `tests/unit/test_dashboard_app.py` uses Streamlit's `AppTest` for
the script itself and plain unit tests for everything else.

```bash
python -m pytest tests/unit/test_dashboard_app.py tests/unit/test_dashboard_client.py -v
```

---

## See also

- [`api.md`](api.md) — the endpoints this consumes
- [`deployment.md`](deployment.md#behind-a-reverse-proxy) — WebSocket headers
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — the layering rule
