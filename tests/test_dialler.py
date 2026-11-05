import json
from datetime import datetime, time, timedelta, timezone

import psycopg
import pytest

from dialler.main import free_slots, in_calling_hours, parse_allowlist, tick
from shared import db

NOW = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)  # 11:00 in Dublin (GMT in winter)


def test_calling_hours_window():
    assert in_calling_hours(time(10), time(9), time(17))
    assert not in_calling_hours(time(8, 59), time(9), time(17))
    assert not in_calling_hours(time(17), time(9), time(17))  # end is exclusive


def test_calling_hours_default_when_campaign_has_none():
    assert in_calling_hours(time(12), None, None)
    assert not in_calling_hours(time(22), None, None)  # never at night, even if a manager left hours blank


@pytest.mark.parametrize(
    "cap, called, dialling, concurrency, expected",
    [(10, 0, 0, 2, 2), (10, 9, 0, 2, 1), (10, 8, 1, 2, 1), (10, 10, 0, 2, 0), (None, 50, 1, 2, 1), (3, 5, 0, 2, 0)],
)
def test_free_slots(cap, called, dialling, concurrency, expected):
    assert free_slots(cap, called, dialling, concurrency) == expected


def test_allowlist_is_required_and_star_means_everyone():
    assert parse_allowlist(" +3531, +3532 ") == {"+3531", "+3532"}
    assert parse_allowlist("*") is None


@pytest.mark.parametrize("value", [None, "", " , "])
def test_worker_refuses_to_start_without_an_allowlist(value):
    # it used to start and silently dial nobody, which looked like a campaign that never started
    with pytest.raises(SystemExit, match="DIAL_ALLOWLIST"):
        parse_allowlist(value)


@pytest.fixture
def campaign(test_db):
    """A running campaign with an empty call history; gives a helper that adds leads to it."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("truncate leads, campaigns, calls, suppression restart identity cascade")
        c.execute("insert into campaigns (name, status, daily_cap) values ('c', 'running', 10)")

    def add(phone, **cols):
        cols = {"campaign_id": 1, "phone": phone, "consent": True, **cols}
        with psycopg.connect(test_db, autocommit=True) as c:
            sql = f"insert into leads ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))}) returning id"
            return c.execute(sql, list(cols.values())).fetchone()[0]

    return add


def statuses(test_db):
    with psycopg.connect(test_db) as c:
        return dict(c.execute("select phone, status from leads").fetchall())


def test_claim_picks_only_due_allowed_unsuppressed_leads(test_db, campaign):
    campaign("+1")
    campaign("+2", status="retry", next_attempt_at=NOW + timedelta(hours=1))  # not due yet
    campaign("+3", status="callback", next_attempt_at=NOW - timedelta(minutes=1))
    campaign("+4", status="closed")
    campaign("+5")
    campaign("+6")
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("insert into suppression (phone, source) values ('+5', 'manual')")
    claimed = db.claim_leads(1, 10, allowlist={"+1", "+2", "+3", "+4", "+5"}, now=NOW)  # +6 not allowed
    assert sorted(l["phone"] for l in claimed) == ["+1", "+3"]
    assert all(l["attempts"] == 1 for l in claimed)  # the attempt is counted at claim time
    assert statuses(test_db) == {"+1": "dialling", "+2": "retry", "+3": "dialling", "+4": "closed", "+5": "new", "+6": "new"}


def test_claim_respects_n(test_db, campaign):
    for p in ("+1", "+2", "+3"):
        campaign(p)
    assert len(db.claim_leads(1, 2, allowlist=None, now=NOW)) == 2


def test_stale_dialling_is_retried_then_exhausted(test_db, campaign):
    # the agent never saved a call (bad trunk, crash): the lead must not stay "dialling" forever
    stale = NOW - timedelta(hours=1)
    campaign("+1", status="dialling", dialled_at=stale, attempts=1)
    campaign("+2", status="dialling", dialled_at=stale, attempts=db.MAX_ATTEMPTS)
    campaign("+3", status="dialling", dialled_at=NOW, attempts=1)  # a call in progress
    assert [l["phone"] for l in db.claim_leads(1, 10, allowlist=None, now=NOW)] == ["+1"]
    assert statuses(test_db) == {"+1": "dialling", "+2": "exhausted", "+3": "dialling"}


class FakeDispatch:
    def __init__(self):
        self.requests = []

    async def create_dispatch(self, request):
        self.requests.append(request)


@pytest.mark.asyncio
async def test_tick_dispatches_claimed_leads_with_their_ids(test_db, campaign):
    lead_id = campaign("+1", name="Aoife", company="Acme")
    dispatch = FakeDispatch()
    await tick(dispatch, NOW, concurrency=2, allowlist=None)
    (request,) = dispatch.requests
    lead = json.loads(request.metadata)
    assert request.agent_name == "sales-agent"
    assert (lead["lead_id"], lead["campaign_id"], lead["phone"], lead["name"]) == (lead_id, 1, "+1", "Aoife")


@pytest.mark.asyncio
async def test_tick_stops_at_the_daily_cap(test_db, campaign):
    campaign("+1")
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("update campaigns set daily_cap = 1")
        c.execute("insert into calls (job_id, campaign_id, outcome, created_at) values ('j', 1, 'hung_up', %s)", [NOW])
    dispatch = FakeDispatch()
    await tick(dispatch, NOW, concurrency=2, allowlist=None)
    assert dispatch.requests == []


@pytest.mark.asyncio
async def test_tick_skips_campaigns_outside_calling_hours(test_db, campaign):
    campaign("+1")
    dispatch = FakeDispatch()
    await tick(dispatch, NOW.replace(hour=22), concurrency=2, allowlist=None)
    assert dispatch.requests == []


@pytest.mark.asyncio
async def test_tick_marks_a_campaign_done_when_no_leads_are_left(test_db, campaign):
    campaign("+1", status="closed")
    await tick(FakeDispatch(), NOW, concurrency=2, allowlist=None)
    with psycopg.connect(test_db) as c:
        assert c.execute("select status from campaigns").fetchone()[0] == "done"


def test_claim_never_rings_a_number_already_in_a_call(test_db, campaign):
    campaign("+1", status="dialling", dialled_at=NOW, attempts=1)
    campaign("+1")
    campaign("+2")
    assert [l["phone"] for l in db.claim_leads(1, 10, allowlist=None, now=NOW)] == ["+2"]


@pytest.mark.asyncio
async def test_tick_rings_each_number_once_at_a_time(test_db, campaign):
    for _ in range(3):
        campaign("+1")
    dispatch = FakeDispatch()
    await tick(dispatch, NOW, concurrency=2, allowlist=None)
    assert len(dispatch.requests) == 1
