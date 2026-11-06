#!/usr/bin/env bash
# Run the live-call API and the dialler together; Ctrl-C stops both, and so does either one dying. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
allow=${DIAL_ALLOWLIST:-}; allow=${allow//[[:space:]]/}  # the dialler strips too: " *" means any number
if [ -z "$allow" ] || [ "$allow" = "*" ]; then
  echo "Set DIAL_ALLOWLIST to the numbers the demo may ring, e.g. DIAL_ALLOWLIST=+353871234567 (not empty, not *)." >&2
  exit 1
fi
trap 'trap "" TERM; kill 0' EXIT  # the script ignores its own TERM, so it keeps its exit code
uv run uvicorn api.main:app --port 8000 &
api=$!
uv run python -m dialler.main --every 5 &
dialler=$!
while kill -0 "$api" 2>/dev/null && kill -0 "$dialler" 2>/dev/null; do  # bash 3.2 on macOS has no wait -n
  sleep 1
done
echo "The API or the dialler stopped, so the demo is stopping." >&2
exit 1
