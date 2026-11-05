import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

from shared import db

logger = logging.getLogger("sales-agent")

# Google Sheet columns, in the sheet header's order
FIELDS = ["timestamp", "name", "phone", "company", "outcome", "interested_services", "callback_time", "notes", "room"]


def record_result(row: dict) -> None:
    """Save one call outcome to the database and, if configured, the Google Sheet. Blocking: call from a thread."""
    try:
        db.save_call(row)
    except Exception as e:
        # Logged with the full row so the lead can be recovered from the LiveKit Cloud logs
        logger.error("could not save call to database: %s; row=%s", e, json.dumps(row, default=str))
    if url := os.getenv("RESULTS_WEBHOOK_URL"):
        sheet_row = {f: row.get(f) or "" for f in FIELDS}
        _send_to_sheet(url, sheet_row | {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _send_to_sheet(url: str, row: dict) -> None:
    body = json.dumps({"token": os.getenv("RESULTS_WEBHOOK_TOKEN", ""), **row}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            reply = response.read().decode()
        if reply.strip() != "ok":
            raise RuntimeError(f"sheet replied {reply[:200]!r}")
    except Exception as e:
        # Logged with the full row so the lead can be recovered from the LiveKit Cloud logs
        logger.error("could not send result to Google Sheet: %s; row=%s", e, json.dumps(row))
