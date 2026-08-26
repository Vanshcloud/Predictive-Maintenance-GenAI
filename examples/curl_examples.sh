#!/usr/bin/env bash
# =============================================================================
# examples/curl_examples.sh — every endpoint, from the shell
# =============================================================================
# Usage:  make docker-up-d  &&  bash examples/curl_examples.sh
#         API=http://staging:8000 bash examples/curl_examples.sh
#
# Requires: curl. `jq` is used when present and skipped when not.
# =============================================================================
set -euo pipefail

API="${API:-http://localhost:8000}"
MACHINE="${MACHINE:-51}"
# Machine 51 fails at 2024-10-31 12:00 in the seeded dataset.
AS_OF="${AS_OF:-2024-10-31T06:00:00}"

if command -v jq >/dev/null 2>&1; then pretty() { jq; }; else pretty() { cat; echo; }; fi
step() { printf '\n\033[36m── %s\033[0m\n' "$1"; }

step "Readiness — status is 'ok' only if predictions can actually be served"
curl -fsS "$API/health" | pretty

step "Every machine (no scoring)"
curl -fsS "$API/machines" | pretty | head -20

step "One machine, latest reading"
curl -fsS "$API/machines/$MACHINE/predict" | pretty

step "The same machine as of $AS_OF — everything after it is hidden"
curl -fsS --get "$API/machines/$MACHINE/predict" --data-urlencode "as_of=$AS_OF" | pretty

step "The evidence behind that score"
curl -fsS --get "$API/machines/$MACHINE/explain" --data-urlencode "as_of=$AS_OF" | pretty

step "Recent sensor history"
curl -fsS --get "$API/machines/$MACHINE/history" \
     --data-urlencode "hours=6" --data-urlencode "as_of=$AS_OF" | pretty

step "Fleet, alerting machines only (~13.5 s cold, ~3 ms cached)"
curl -fsS --get "$API/fleet" \
     --data-urlencode "alerts_only=true" --data-urlencode "as_of=$AS_OF" | pretty

step "Score caller-supplied readings — what a real plant would use"
curl -fsS -X POST "$API/predict" \
     -H 'Content-Type: application/json' \
     -d @"$(dirname "$0")/predict_request.json" | pretty

step "A written report (~21 s — calls a language model)"
# Degrades to 502 WITH the prediction attached if no provider is reachable,
# so -f is deliberately omitted here.
curl -sS -X POST "$API/report" \
     -H 'Content-Type: application/json' \
     -d "{\"machine_id\": $MACHINE, \"as_of\": \"$AS_OF\"}" | pretty

step "Error shapes"
echo "404 — unknown machine:"
curl -sS "$API/machines/99999/predict" | pretty
echo "422 — a reading outside physical bounds:"
curl -sS -X POST "$API/predict" -H 'Content-Type: application/json' \
     -d '{"machine_id": 1, "readings": [{"datetime": "2024-01-01T00:00:00", "voltage": 99999, "rotation": 400, "pressure": 100, "vibration": 40}]}' | pretty

printf '\n\033[32mdone\033[0m\n'
