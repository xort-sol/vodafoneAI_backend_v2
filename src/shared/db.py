"""Supabase Postgres access for the agent and the dialler. Blocking: call from a thread in async code."""

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

CALL_FIELDS = ["job_id", "room", "phone", "name", "company", "lead_id", "campaign_id", "outcome",
               "interested_services", "callback_time", "callback_at", "notes"]
MAX_ATTEMPTS = 3
RETRY_AFTER = timedelta(hours=4)
UNREACHED = {"no_answer", "voicemail", "no_response", "unreachable"}
CLOSED = {"not_interested", "hung_up"}
IRELAND = ZoneInfo("Europe/Dublin")
CALLABLE = "consent and not archived and sales_status is null"  # leads the dialler may still ring


def _connect() -> psycopg.Connection:
    # DATABASE_URL is the Supabase Session pooler string: the direct connection is IPv6-only
    return psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10)


def next_lead_state(outcome: str, attempts: int, callback_at: datetime | None, now: datetime,
                    max_attempts: int = MAX_ATTEMPTS, retry_after: timedelta = RETRY_AFTER) -> tuple[str, datetime | None]:
    """The lead's (status, next_attempt_at) after a call; attempts includes this call."""
    if outcome in UNREACHED:
        return ("retry", now + retry_after) if attempts < max_attempts else ("exhausted", None)
    if outcome == "callback" and callback_at:
        return "callback", callback_at
    if outcome in CLOSED:
        return "closed", None
    if outcome == "do_not_call":
        return "do_not_call", None
    return "interested", None  # interested, or a callback with no usable time: a salesperson follows up


def _parse_time(value: str | None, now: datetime) -> datetime | None:
    """A callback time the dialler can use: a time without an offset is Irish time, and one already past is no time at all."""
    try:
        t = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    if t and t.tzinfo is None:
        t = t.replace(tzinfo=IRELAND)
    return t if t and t >= now else None


def save_call(row: dict) -> None:
    """One row per LiveKit job; saving the same job again is ignored. A do-not-call also goes on the suppression list.
    A dialler call also moves its lead to the next status, in the same transaction."""
    now = datetime.now(timezone.utc)
    row = {**row, "callback_at": _parse_time(row.get("callback_at"), now)}
    columns = ", ".join(CALL_FIELDS)
    with _connect() as conn:
        inserted = conn.execute(
            f"insert into calls ({columns}) values ({', '.join(['%s'] * len(CALL_FIELDS))}) on conflict (job_id) do nothing",
            [row.get(f) or None for f in CALL_FIELDS],
        ).rowcount
        if inserted and row.get("lead_id"):
            # the dialler counted this attempt when it claimed the lead
            attempts, max_attempts, retry_hours = conn.execute(
                """select l.attempts, coalesce(c.max_attempts, 3), coalesce(c.retry_hours, 4)
                   from leads l left join campaigns c on c.id = l.campaign_id where l.id = %s""",
                [row["lead_id"]],
            ).fetchone()
            status, next_at = next_lead_state(row["outcome"], attempts, row["callback_at"], now,
                                              max_attempts, timedelta(hours=retry_hours))
            conn.execute(
                """update leads set status = %s, next_attempt_at = %s, last_call_at = %s, next_follow_up = %s
                   where id = %s""",
                [status, next_at, now, row["callback_at"], row["lead_id"]],
            )
        if row.get("outcome") == "do_not_call" and row.get("phone"):
            conn.execute(
                "insert into suppression (phone, reason, source) values (%s, 'asked on a call', 'call') on conflict do nothing",
                [row["phone"]],
            )


def save_transcript(job_id: str, lines: list[tuple[str, str, int]]) -> None:
    with _connect() as conn:
        call = conn.execute("select id from calls where job_id = %s", [job_id]).fetchone()
        if call is None:
            return
        with conn.cursor() as cur:
            cur.executemany(
                "insert into transcripts (call_id, position, role, text, t_seconds) values (%s, %s, %s, %s, %s) on conflict do nothing",
                [(call[0], i, role, text, t) for i, (role, text, t) in enumerate(lines)],
            )



STALE_AFTER = timedelta(minutes=30)  # a claimed lead with no saved call by then: the call never happened
LEAD_FIELDS = ["name", "phone", "company", "staff", "current_plan", "contract_end", "notes"]


def running_campaigns() -> list[dict]:
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "select id, calling_hours_start, calling_hours_end, daily_cap, days, concurrency, max_attempts from campaigns where status = 'running' order by id"
        ).fetchall()


def campaign_counts(campaign_id: int, since: datetime, now: datetime) -> tuple[int, int]:
    """(calls since `since`, calls in progress) for one campaign."""
    with _connect() as conn:
        return conn.execute(
            """select (select count(*) from calls where campaign_id = %(c)s and created_at >= %(since)s),
                      (select count(*) from leads where campaign_id = %(c)s and status = 'dialling' and dialled_at >= %(stale)s)""",
            {"c": campaign_id, "since": since, "stale": now - STALE_AFTER},
        ).fetchone()


def claim_leads(campaign_id: int, n: int, allowlist: set[str] | None, now: datetime,
                max_attempts: int = MAX_ATTEMPTS) -> list[dict]:
    """Marks up to n due leads as dialling, counting the attempt, and returns them. allowlist None means any number."""
    params = {"c": campaign_id, "n": n, "now": now, "stale": now - STALE_AFTER, "max": max_attempts,
              "any": allowlist is None, "allow": sorted(allowlist or [])}
    with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """update leads set status = 'do_not_call', next_attempt_at = null
               where campaign_id = %(c)s and status in ('new', 'retry', 'callback') and phone in (select phone from suppression)""",
            params,
        )
        cur.execute(
            """update leads set status = 'exhausted', next_attempt_at = null
               where campaign_id = %(c)s and status = 'dialling' and dialled_at < %(stale)s and attempts >= %(max)s""",
            params,
        )
        return cur.execute(
            """with due as (
                 select id from leads
                 where campaign_id = %(c)s and """ + CALLABLE + """
                   and (status in ('new', 'retry', 'callback') or (status = 'dialling' and dialled_at < %(stale)s))
                   and (next_attempt_at is null or next_attempt_at <= %(now)s)
                   and phone is not null and phone not in (select phone from suppression)
                   and phone not in (select phone from leads  -- that number is in a call right now
                                     where status = 'dialling' and dialled_at >= %(stale)s)
                   and (%(any)s or phone = any(%(allow)s))
                 order by next_attempt_at nulls first, id
                 limit %(n)s
                 for update skip locked)
               update leads l set status = 'dialling', dialled_at = %(now)s, attempts = l.attempts + 1
               from due where l.id = due.id
               returning l.id as lead_id, l.campaign_id, l.attempts, """ + ", ".join(f"l.{f}" for f in LEAD_FIELDS),
            params,
        ).fetchall()


def finish_campaign_if_empty(campaign_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            """update campaigns set status = 'done' where id = %(c)s and status = 'running' and not exists
                 (select 1 from leads where campaign_id = %(c)s and """ + CALLABLE + """
                    and status in ('new', 'retry', 'callback', 'dialling'))""",
            {"c": campaign_id},
        )


def save_analysis(job_id: str, analysis: dict | None, model: str, hot: bool) -> None:
    """The analysis (if any) and, for a hot lead, one inbox task due at the customer's callback time. Saving again is a no-op."""
    with _connect() as conn:
        call = conn.execute("select id, lead_id, callback_at from calls where job_id = %s", [job_id]).fetchone()
        if call is None:
            return
        call_id, lead_id, callback_at = call
        if analysis:
            conn.execute(
                """insert into analyses (call_id, score, reason, objections, sentiment, next_step, brief, model, questions, unanswered, qualification)
                   values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (call_id) do nothing""",
                [call_id, analysis["score"], analysis["reason"], analysis["objections"], analysis["sentiment"],
                 analysis["next_step"], analysis["brief"], model,
                 analysis["questions"], analysis["unanswered"], Json(analysis["qualification"])],
            )
            conn.execute(
                "update leads set score = %s, sentiment = %s, ai_summary = %s where id = %s",
                [analysis["score"], analysis["sentiment"], analysis["brief"], lead_id],
            )
        if hot:
            conn.execute(
                """insert into tasks (lead_id, call_id, kind, due_at) values (%s, %s, 'follow_up', %s)
                   on conflict (call_id) where kind = 'follow_up' do nothing""",
                [lead_id, call_id, callback_at],
            )


def save_call_stats(job_id: str, duration_s: int, avg_latency_ms: int | None) -> None:
    with _connect() as conn:
        conn.execute(
            "update calls set duration_s = %s, avg_latency_ms = %s where job_id = %s", [duration_s, avg_latency_ms, job_id]
        )


def role_of(user_id: str) -> str | None:
    """The web app role an admin gave this user, or None (no profile, or no role yet)."""
    with _connect() as conn:
        row = conn.execute("select role from profiles where id = %s", [user_id]).fetchone()
    return row[0] if row else None
