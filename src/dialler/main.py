"""Dialler worker: every minute, dial due leads of running campaigns.

Usage (from the repo root): DIAL_ALLOWLIST=+3538... uv run python -m dialler.main [--concurrency 2] [--every 60]
DIAL_ALLOWLIST is required: comma-separated numbers it may ring, or "*" for any number.
"""

import argparse
import asyncio
import json
import logging
import os
import time as clock
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import api

from shared import db

logger = logging.getLogger("dialler")
IRELAND = ZoneInfo("Europe/Dublin")
DEFAULT_HOURS = (time(9), time(20))  # when a campaign has no calling hours set


def in_calling_hours(local: time, start: time | None, end: time | None) -> bool:
    start, end = start or DEFAULT_HOURS[0], end or DEFAULT_HOURS[1]
    return start <= local < end


def free_slots(daily_cap: int | None, called_today: int, dialling: int, concurrency: int) -> int:
    """How many more calls to start now: under the concurrency limit, and calls in progress count toward the cap."""
    slots = concurrency - dialling
    if daily_cap is not None:
        slots = min(slots, daily_cap - called_today - dialling)
    return max(slots, 0)


def parse_allowlist(value: str | None) -> set[str] | None:
    """The numbers the worker may ring; None means any. Exits when unset, so a demo never rings a real customer by accident."""
    if (value or "").strip() == "*":
        return None
    numbers = {p.strip() for p in (value or "").split(",") if p.strip()}
    if not numbers:
        raise SystemExit('DIAL_ALLOWLIST is not set: give the numbers to ring (DIAL_ALLOWLIST=+3538...) or "*" for any number')
    return numbers


async def tick(dispatch, now: datetime, concurrency: int, allowlist: set[str] | None) -> None:
    local = now.astimezone(IRELAND)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    for c in db.running_campaigns():
        db.finish_campaign_if_empty(c["id"])
        if not in_calling_hours(local.time(), c["calling_hours_start"], c["calling_hours_end"]):
            continue
        called_today, dialling = db.campaign_counts(c["id"], since=midnight, now=now)
        n = free_slots(c["daily_cap"], called_today, dialling, concurrency)
        claimed = []
        for _ in range(n):  # one at a time: a claimed lead marks its number busy, so one number never rings twice at once
            claimed += db.claim_leads(c["id"], 1, allowlist, now)
        for lead in claimed:
            room = f"call-{lead['phone'].lstrip('+')}-{int(clock.time())}"
            try:
                await dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(agent_name="sales-agent", room=room, metadata=json.dumps(lead))
                )
                logger.info("dispatched lead %s (%s) -> %s", lead["lead_id"], lead["phone"], room)
            except Exception:
                # the lead stays "dialling" and is retried once stale
                logger.exception("could not dispatch lead %s", lead["lead_id"])


async def main(concurrency: int, every: float) -> None:
    allowlist = parse_allowlist(os.getenv("DIAL_ALLOWLIST"))
    logger.info("dialling %s", "any number" if allowlist is None else f"only {sorted(allowlist)}")
    async with api.LiveKitAPI() as lkapi:
        while True:
            try:
                await tick(lkapi.agent_dispatch, datetime.now(timezone.utc), concurrency, allowlist)
            except Exception:
                logger.exception("tick failed")  # e.g. database briefly down: try again next minute
            await asyncio.sleep(every)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--every", type=float, default=60.0)
    args = p.parse_args()
    asyncio.run(main(args.concurrency, args.every))
