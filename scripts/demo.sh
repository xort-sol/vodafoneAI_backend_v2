#!/usr/bin/env bash
# Run the live-call API and the dialler together; Ctrl-C stops both. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
allow=${DIAL_ALLOWLIST:-}; allow=${allow//[[:space:]]/}  # the dialler strips too: " *" means any number
if [ -z "$allow" ] || [ "$allow" = "*" ]; then
  echo "Set DIAL_ALLOWLIST to the numbers the demo may ring, e.g. DIAL_ALLOWLIST=+353871234567 (not empty, not *)." >&2
  exit 1
fi
trap 'kill 0' EXIT
uv run uvicorn api.main:app --port 8000 &
uv run python -m dialler.main --every 5 &
wait  # bash 3.2 on macOS has no wait -n
