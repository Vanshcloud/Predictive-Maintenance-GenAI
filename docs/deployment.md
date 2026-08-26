# Deployment

Running this as a service. Covers containers, configuration, health checks,
logging, and what has to change before it faces anything untrusted.

> **Read [`../SECURITY.md`](../SECURITY.md) first if you intend to expose this.**
> There is no authentication and no rate limiting. It is built for `localhost`
> or a trusted network, and `POST /report` spends money against an upstream
> provider when a key is configured.

---

## Contents

- [Docker Compose](#docker-compose-recommended)
- [The two images](#the-two-images)
- [Environment variables](#environment-variables)
- [Health checks](#health-checks)
- [Running without containers](#running-without-containers)
- [Behind a reverse proxy](#behind-a-reverse-proxy)
- [Logging](#logging)
- [Updating the model](#updating-the-model)
- [Production checklist](#production-checklist)

---

## Docker Compose (recommended)

```bash
# Artifacts are mounted, not baked in — build them first.
python scripts/generate_data.py
python scripts/run_preprocessing.py
python scripts/train_model.py

make docker-build
make docker-up          # foreground
make docker-up-d        # detached
make docker-down
```

| Service | Port | Image |
|---|---|---|
| `api` | 8000 | `predictive-maintenance-api` |
| `dashboard` | 8501 | `predictive-maintenance-dashboard` |

The dashboard waits on `condition: service_healthy`, so it starts only once the
API reports it can actually serve predictions — not merely once the process is
listening. Without that gate the dashboard's first render races startup.

**Artifacts are volumes, deliberately:**

```yaml
volumes:
  - ../models:/app/models:ro     # retraining is a volume swap, not a rebuild
  - ../data:/app/data:ro         # read-only — the API scores, it does not train
  - ../logs:/app/logs
```

Baking a 5 GB dataset into an image layer would make every retrain an image
rebuild.

---

## The two images

| Image | Size | Installs |
|---|---|---|
| API | **2.87 GB** | `requirements.txt` — TensorFlow, scikit-learn, LangChain |
| Dashboard | **803 MB** | `requirements-dashboard.txt` — four packages |

The gap is the point. The dashboard imports nothing from `src/`, so its image
needs no ML stack at all. Its `Dockerfile` copies only `dashboard/`, which means
a future import from `src/` breaks the build — the intended failure.

Both images:

- are **multi-stage** — the compiler toolchain lives in the builder and is
  discarded, because a compiler in a production image is dead weight and a
  useful tool for anyone who gets a shell;
- run as a **non-root** user (`appuser`);
- pin **Python 3.12** exactly, since TensorFlow publishes no wheels for 3.13+.

Build them individually:

```bash
docker build -f docker/Dockerfile.api       -t predictive-maintenance-api .
docker build -f docker/Dockerfile.dashboard -t predictive-maintenance-dashboard .
```

---

## Environment variables

Copy `.env.example` to `.env`. Real environment variables override the file,
which is what makes container and CI configuration work.

### Application

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `development` · `staging` · `production` |
| `DEBUG` | `true` | Set `false` in production |
| `LOG_LEVEL` | `DEBUG` | Use `INFO` in production |
| `APP_VERSION` | `1.0.0` | Reported by `/health` so an operator can identify the build |

### Model and data

| Variable | Default |
|---|---|
| `MODEL_DIR` | `models` |
| `MODEL_NAME` | `lstm_predictive_maintenance` |
| `RAW_DATA_DIR` | `data/raw` |
| `PROCESSED_DATA_DIR` | `data/processed` |

### Serving

| Variable | Default | Notes |
|---|---|---|
| `PREDICTION_THRESHOLD` | `0.3415` | Chosen on validation. **Not** 0.5 |
| `RISK_BAND_MEDIUM` | `0.15` | |
| `RISK_BAND_HIGH` | `0.3415` | **Must equal `PREDICTION_THRESHOLD`** — a test enforces this |
| `RISK_BAND_CRITICAL` | `0.90` | |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | |
| `DASHBOARD_PORT` | `8501` | Also the CORS allow-list origin |

### LLM providers (all optional)

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Placeholder values are treated as unset |
| `OPENAI_MODEL` | `gpt-4o-mini` | |
| `GOOGLE_API_KEY` | — | Needs `pip install -e ".[google]"` |
| `GOOGLE_MODEL` | `gemini-1.5-flash` | |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Use `http://host.docker.internal:11434` from a container |
| `OLLAMA_MODEL` | `llama3` | Keyless path |

> The settings validator nulls `.env.example` placeholders like
> `your-openai-api-key-here`. Without that, the app believes it has credentials,
> calls the provider, and returns a 401 that reads like a broken key rather than
> an absent one.

**No key is required.** With none set, prediction endpoints work fully and
`/report` degrades to a `502` that still carries the prediction.

---

## Health checks

`/health` reports **readiness**, not liveness. `status` is `ok` only when
predictions can actually be served.

```bash
curl -fsS localhost:8000/health | jq -r .status     # ok | degraded
```

The container health check uses it rather than a TCP probe, because the process
can be listening while the model failed to load — a TCP check would call that
healthy.

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "..."]   # exits non-zero unless status == "ok"
  interval: 30s
  timeout: 10s
  start_period: 90s      # model load + 876k-row dataset parse
  retries: 3
```

`start_period` is 90 s deliberately. The model takes ~2 s and the dataset
longer; a shorter grace period restart-loops a healthy service.

**With no model mounted the API starts `degraded` rather than crashing** — an
API that refuses to start cannot serve `/health`, which is exactly what an
operator needs in order to find out the model is missing. CI asserts this.

For orchestrators, map it as:

| Probe | Endpoint | Meaning |
|---|---|---|
| Liveness | `GET /` | The process is alive; restart if this fails |
| Readiness | `GET /health` → `status == "ok"` | Route traffic only when true |

---

## Running without containers

```bash
source venv/bin/activate

# API — reload is for development only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Dashboard, pointed at the API
API_BASE_URL=http://localhost:8000 \
  streamlit run dashboard/app.py --server.port 8501
```

**Workers.** `uvicorn --workers N` gives each worker its own model and dataset
copy — roughly 1 GB of RSS each, and N independent fleet caches. Prediction is
CPU-bound TensorFlow inference, so past a small N the workers contend for the
same cores rather than adding throughput. Start with 1 and measure before
increasing.

A `systemd` unit:

```ini
[Unit]
Description=Predictive Maintenance API
After=network.target

[Service]
Type=exec
User=appuser
WorkingDirectory=/opt/predictive-maintenance
Environment="APP_ENV=production" "LOG_LEVEL=INFO"
ExecStart=/opt/predictive-maintenance/venv/bin/uvicorn src.api.main:app \
          --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Binding to `127.0.0.1` and terminating TLS at a proxy is the shape you want —
see below.

---

## Behind a reverse proxy

This is where the missing authentication gets added. The example below is
nginx; any proxy works.

```nginx
upstream pm_api       { server 127.0.0.1:8000; }
upstream pm_dashboard { server 127.0.0.1:8501; }

# Rate limits. /report invokes a language model, so it gets a much tighter
# bucket than the prediction endpoints.
limit_req_zone $binary_remote_addr zone=api:10m    rate=30r/s;
limit_req_zone $binary_remote_addr zone=report:10m rate=6r/m;

server {
    listen 443 ssl http2;
    server_name maintenance.example.com;

    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # The service has no auth of its own. Something must supply it.
    auth_basic           "Maintenance";
    auth_basic_user_file /etc/nginx/.htpasswd;

    client_max_body_size 2m;   # POST /predict has no upper bound of its own

    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://pm_api;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }

    location /report {
        limit_req zone=report burst=2;
        proxy_pass http://pm_api;
        proxy_set_header Host $host;
        # The API caps itself at 120 s; the proxy must wait longer than that
        # or it gives up on work that would have succeeded.
        proxy_read_timeout 150s;
    }

    location /dashboard/ {
        proxy_pass http://pm_dashboard/;
        proxy_http_version 1.1;
        # Streamlit is a WebSocket application; without these it will not render.
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       $host;
        proxy_read_timeout 86400s;
    }
}
```

Three things that bite if you skip them:

1. **`client_max_body_size`** — `POST /predict` accepts an unbounded number of
   readings. The proxy is currently the only thing bounding it.
2. **WebSocket upgrade headers** for Streamlit, or the dashboard renders blank.
3. **`proxy_read_timeout` above 120 s on `/report`**, or the proxy times out
   before the API's own ceiling and you lose work that was about to finish.

If the dashboard is served under a path prefix, set
`--server.baseUrlPath=dashboard`.

---

## Logging

Loguru, configured once in `src/utils/logger.py`.

| Sink | Format | Level | Rotation |
|---|---|---|---|
| stderr | Coloured, human-readable | `LOG_LEVEL` | — |
| `logs/app_YYYY-MM-DD.log` | Structured | `DEBUG` | 5 MB, 3 days, zipped |

In containers, `logs/` is a mounted volume. To ship to a log collector instead,
read stderr — the console sink is what container runtimes capture.

Conventions the code follows, worth preserving:

- `INFO` for lifecycle milestones, `DEBUG` for detail, `WARNING` for
  degraded-but-continuing, `ERROR` for failures with context.
- **Log the numbers.** Shapes, counts, durations, metric values. A line that
  says "training complete" without the metrics is not worth writing.
- Unexpected exceptions log a `correlation_id` that is also returned to the
  client. Grep for it.

```bash
docker compose -f docker/docker-compose.yml logs -f api
grep '\[a1b2c3d4\]' logs/app_*.log        # by correlation id
```

---

## Updating the model

Because artifacts are mounted read-only, a retrain is a swap and a restart:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py

# If the chosen threshold moved, update config/settings.py — both
# PREDICTION_THRESHOLD and RISK_BAND_HIGH. A test enforces that they agree.

docker compose -f docker/docker-compose.yml restart api
curl -fsS localhost:8000/health | jq '{status, version, threshold}'
```

No image rebuild. Roll back by restoring the previous `.keras` and restarting.

---

## Production checklist

Everything unticked is genuinely not done — this list is not aspirational.

**Before exposing beyond a trusted network**

- [ ] Authentication in front of the API (the proxy example above)
- [ ] Rate limiting, especially on `/report`
- [ ] `client_max_body_size` — `POST /predict` has no ceiling of its own
- [ ] `APP_ENV=production`, `DEBUG=false`, `LOG_LEVEL=INFO`
- [ ] TLS terminated at the proxy; API bound to `127.0.0.1`
- [ ] Secrets injected as environment variables, never baked into an image

**Operational**

- [ ] `/health` wired to your orchestrator's readiness probe
- [ ] Log shipping configured; `logs/` is not durable storage
- [ ] Model artifacts backed up outside the repository
- [ ] Alerting on `status: degraded`

**Known gaps — see [`roadmap.md`](roadmap.md)**

- [ ] No metrics endpoint (no Prometheus, no request histograms)
- [ ] No distributed tracing
- [ ] No input-drift detection
- [ ] No model registry or versioned rollback beyond swapping a file

---

## See also

- [`api.md`](api.md) — the endpoint reference
- [`troubleshooting.md`](troubleshooting.md) — when a container will not start
- [`benchmarks.md`](benchmarks.md) — measured latency, memory, and image sizes
- [`../SECURITY.md`](../SECURITY.md) — the threat model in full
