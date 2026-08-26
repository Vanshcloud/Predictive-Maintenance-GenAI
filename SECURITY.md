# Security Policy

## Supported versions

This project is a reference implementation rather than a released product.
Security fixes are applied to `main`; there are no maintained release branches.

| Version | Supported |
|---|---|
| `main` | ✅ |
| Tagged releases | ⚠️ Best effort |

---

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it through
[GitHub's private vulnerability reporting](https://github.com/Vanshcloud/Predictive-Maintenance-GenAI/security/advisories/new),
or by email to **vanshwar@gmail.com** with `SECURITY` in the subject line.

Please include:

- what the issue is and what an attacker gains from it,
- the smallest reproduction you have,
- the affected commit or version,
- any deployment assumptions your report depends on (see the threat model below —
  a finding that assumes an internet-facing deployment should say so).

You can expect an acknowledgement within **5 working days** and an assessment
within **15**. If a fix is warranted I will credit you in the release notes
unless you would rather I didn't.

---

## Threat model, stated plainly

This matters more than a disclosure address, because most of what someone might
report is a known and deliberate scope decision rather than a defect.

**This application is designed to run on `localhost` or a trusted internal
network.** It is not hardened for direct exposure to the public internet, and
the following are known, intentional gaps rather than oversights:

| Gap | Consequence if exposed |
|---|---|
| **No authentication or authorisation** on any endpoint | Anyone who can reach the port can read operational data about every machine. |
| **No rate limiting** | `POST /report` invokes a language model. With an OpenAI or Gemini key configured, an anonymous caller can spend against your account in a loop. |
| **Unbounded request bodies** | `PredictRequest.readings` enforces a minimum of 48 but no maximum, so a large payload becomes proportional memory and CPU in pandas. `ReportRequest.question` has no length cap and is interpolated into the model prompt. |
| **No CSRF protection on the dashboard** | Streamlit is not deployed here with a multi-user posture in mind. |
| **The dashboard's API URL is user-editable** | The Streamlit *server* performs the fetch, so in a shared deployment that field is a server-side request forgery vector. |

If you intend to deploy this beyond a trusted network, put it behind an
authenticating reverse proxy and add rate limiting to `/report` at minimum.

### What *is* handled

These are deliberate and tested, so a report claiming otherwise is likely a
misunderstanding:

- **No secrets in version control.** `.env`, model weights, and generated data
  are gitignored; only `.env.example` is committed. Placeholder values in the
  example file are normalised to "not configured" so a copied template cannot be
  mistaken for a real credential.
- **API keys are wrapped in `SecretStr`**, keeping them out of `repr()` output
  and tracebacks.
- **CORS is restricted to the dashboard origin**, not `*`.
- **Errors do not leak internals.** An unexpected exception returns an opaque
  message plus a correlation id; the detail goes to the logs. No stack trace
  crosses the HTTP boundary.
- **Input is validated at the boundary.** Sensor readings are range-checked
  against physical bounds in `src/api/schemas.py`, so out-of-range values are
  rejected with a 422 rather than scored.
- **Containers run as a non-root user**, with the compiler toolchain confined to
  the builder stage and artifacts mounted read-only.
- **Untrusted strings are escaped before rendering.** The dashboard escapes
  values received over HTTP before interpolating them into markup, because the
  API schema constrains this project's server and not whatever host a user
  points the sidebar at.

---

## Dependency vulnerabilities

CI runs [`pip-audit`](https://pypi.org/project/pip-audit/) on every push.

At the time of writing it reports findings in the LangChain packages whose fixes
are only available in their 1.x line, which is outside the version range this
project pins. Each was assessed for reachability and none of the affected code
paths is used here — this project defines prompts in code via
`ChatPromptTemplate.from_messages`, loads no prompt or chain configuration from
disk, uses no HTML text splitter, and never sends an image to a model.

If you find a dependency advisory that *is* reachable from this codebase, that
is a genuine report and I would like to hear about it.

---

## Model-specific considerations

Three things worth knowing about the machine-learning surface:

### Model artifacts are executable code — treat them as such

`Predictor` loads two files at startup:

| File | Loader | Risk |
|---|---|---|
| `scaler.joblib` | `joblib.load()` | **Pickle-backed. Loading a malicious file executes arbitrary code.** |
| `*.keras` | `keras.load_model()` | Can execute code via custom or `Lambda` layers |

**Only load artifacts you produced yourself, or that came from a source you
trust as much as you trust your own code.** This is not specific to this
project — it is true of every pickle-based ML artifact — but it is worth
stating, because a `.joblib` file looks like data and is not.

The processed tensors are safer: `np.load(..., mmap_mode="r")` leaves
`allow_pickle` at its default of `False`, so a `.npy` file cannot execute code.

In the container topology, `models/` and `data/` are mounted **read-only**, so
a compromised API process cannot rewrite the artifacts it will load on restart.


- **Predictions are advisory.** This system is a decision aid for maintenance
  scheduling. It should not be wired to anything that actuates equipment
  without a human in the loop.
- **Generated reports are grounded, not trusted.** Every figure a language model
  quotes is supplied to it from the prediction record; it is given nothing else,
  and the model's narrative never influences the prediction. An LLM failure
  degrades the system to "prediction available, narrative unavailable" and never
  to a wrong number. Report text is rendered without raw HTML enabled.
