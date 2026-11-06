import json
import logging

from shared import db

logger = logging.getLogger("sales-agent")


def record_result(row: dict) -> None:
    """Save one call outcome to the database. Blocking: call from a thread."""
    try:
        db.save_call(row)
    except Exception as e:
        # Logged with the full row so the lead can be recovered from the LiveKit Cloud logs
        logger.error("could not save call to database: %s; row=%s", e, json.dumps(row, default=str))
