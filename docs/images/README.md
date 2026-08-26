# Images

What is here, and how to regenerate it.

---

## Committed

| File | What it is | Regenerate with |
|---|---|---|
| `banner.svg` | README header. Hand-authored SVG; the trace is illustrative, not plotted data. | — |
| `horizon.png` | **Real output.** Machine 51's failure probability at every hour of the two days before it failed, each point scored only on evidence available at that hour. | `python scripts/plot_horizon.py` |

`horizon.png` is committed *and* so is the script that draws it. An image
nobody can reproduce is the same problem as a status badge that cannot go red.

```bash
make docker-up-d                      # or: make run-api
pip install -r requirements-dev.txt   # the script needs matplotlib
python scripts/plot_horizon.py                                        # machine 51
python scripts/plot_horizon.py --machine 96 --failure 2024-11-14T00:00:00
```

---

## Not committed — placeholders

**No dashboard screenshots or demo GIF exist yet.** They are referenced from
the README as placeholders rather than being faked. Producing them requires a
running stack and a person to drive it.

If you would like to contribute them, this is genuinely useful and is a good
first contribution.

### What is needed

| File | Shows | Suggested state |
|---|---|---|
| `dashboard-fleet.png` | Fleet overview | Rewind on, **2024-10-31 hour 6** — two machines alerting, so the risk distribution is not one flat bar |
| `dashboard-machine.png` | Machine detail | Machine **51** at the same timestamp — the sensor evidence table is the interesting part |
| `dashboard-report.png` | AI report | A generated report for machine 51, showing that every figure it cites is one from the evidence table |
| `training-curves.png` | Training progress | Copy from `models/training_curves.png` after `scripts/evaluate_model.py` |
| `pr-curve.png` | Threshold selection | Copy from `models/pr_curve.png` |
| `demo.gif` | 15–20 s walkthrough | Fleet → rewind to 2024-10-31 h6 → machine 51 → generate report |

### How to produce them

```bash
# 1. A trained model and generated data are required.
python scripts/generate_data.py
python scripts/run_preprocessing.py
python scripts/train_model.py
python scripts/evaluate_model.py

# 2. Bring up the stack.
make docker-up-d

# 3. Open http://localhost:8501, turn on Rewind, set 2024-10-31 hour 6.
```

Screenshots: full browser window at **1440×900**, light theme, no browser
chrome or personal bookmarks visible.

GIF: any screen recorder, then

```bash
ffmpeg -i recording.mov -vf "fps=12,scale=1000:-1:flags=lanczos" -loop 0 demo.gif
```

Keep it under **5 MB** — GitHub will not render a larger one inline.

### Two things to check before submitting

1. **No real credentials on screen.** The sidebar shows the API URL; the report
   view shows the provider. Neither should reveal a key, but check the whole
   frame.
2. **Use the documented timestamps.** `2024-10-31 06:00` and machine 51 are
   referenced throughout the docs, so a screenshot showing them lets a reader
   reproduce exactly what they are looking at.

Two of these — `training_curves.png` and `pr_curve.png` — are produced
automatically by `scripts/evaluate_model.py` into `models/`. They are
gitignored there; copy them here to publish them.
