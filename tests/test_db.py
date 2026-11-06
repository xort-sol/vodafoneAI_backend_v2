import logging
import os

from datetime import datetime, timedelta, timezone

import psycopg
import pytest

from shared import db
from shared.results import record_result


def test_tests_use_the_test_database():
    # .env holds the production DATABASE_URL; load_dotenv must never win inside tests
    assert os.environ["DATABASE_URL"] == os.getenv("TEST_DATABASE_URL", "postgresql:///vodafoneai_test")


def test_calls_are_keyed_by_job_not_room(call_rows):
    db.save_call({"job_id": "j1", "room": "console-room", "outcome": "hung_up"})
    db.save_call({"job_id": "j2", "room": "console-room", "outcome": "interested", "callback_time": "Fri 10am"})
    db.save_call({"job_id": "j2", "room": "console-room", "outcome": "hung_up"})  # same job again: ignored
    assert [r["outcome"] for r in call_rows()] == ["hung_up", "interested"]


def test_do_not_call_goes_on_the_suppression_list(call_rows, test_db):
    db.save_call({"job_id": "j1", "phone": "+353870000000", "outcome": "do_not_call"})
    with psycopg.connect(test_db) as conn:
        assert conn.execute("select phone, source from suppression").fetchall() == [("+353870000000", "call")]


def test_transcript_is_stored_in_order(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "hung_up"})
    db.save_transcript("j1", [("agent", "[warmly] Hi, I'm an AI assistant.", 3), ("customer", "Hello?", 7)])
    with psycopg.connect(test_db) as conn:
        lines = conn.execute("select role, text, t_seconds from transcripts order by position").fetchall()
    assert lines == [("agent", "[warmly] Hi, I'm an AI assistant.", 3), ("customer", "Hello?", 7)]


def test_transcript_without_a_call_is_skipped(call_rows):
    db.save_transcript("no-such-job", [("agent", "Hi", 0)])  # must not raise


def test_lead_is_logged_when_the_database_is_down(call_rows, monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    with caplog.at_level(logging.ERROR):
        record_result({"job_id": "j1", "phone": "+1", "outcome": "interested", "callback_time": "Fri 10am"})
    assert "could not save call to database" in caplog.text
    assert "Fri 10am" in caplog.text  # full row in the log, so the lead can be recovered


NOW = datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "outcome, attempts, callback_at, expected",
    [
        ("no_answer", 1, None, ("retry", NOW + db.RETRY_AFTER)),
        ("voicemail", 2, None, ("retry", NOW + db.RETRY_AFTER)),
        ("no_answer", db.MAX_ATTEMPTS, None, ("exhausted", None)),
        ("callback", 1, NOW + timedelta(days=1), ("callback", NOW + timedelta(days=1))),
        ("callback", 1, None, ("interested", None)),  # no usable time: a salesperson follows up
        ("interested", 1, None, ("interested", None)),
        ("not_interested", 1, None, ("closed", None)),
        ("hung_up", 1, None, ("closed", None)),
        ("do_not_call", 1, None, ("do_not_call", None)),
    ],
)
def test_next_lead_state(outcome, attempts, callback_at, expected):
    assert db.next_lead_state(outcome, attempts, callback_at, NOW) == expected


def add_lead(test_db, phone="+353870000001") -> int:
    with psycopg.connect(test_db) as c:
        c.execute("truncate leads restart identity cascade")
        # as the dialler leaves it: claimed, with this attempt already counted
        sql = "insert into leads (phone, status, attempts, consent) values (%s, 'dialling', 1, true) returning id"
        return c.execute(sql, [phone]).fetchone()[0]


def lead(test_db, lead_id) -> tuple:
    with psycopg.connect(test_db) as c:
        return c.execute("select status, attempts, next_attempt_at from leads where id = %s", [lead_id]).fetchone()


def test_saving_a_call_moves_its_lead_on(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "no_answer"})
    status, attempts, next_at = lead(test_db, lead_id)
    assert (status, attempts) == ("retry", 1) and next_at is not None


def test_saving_a_call_records_last_call_and_follow_up(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "callback", "callback_at": "2037-11-05T10:00:00+00:00"})
    with psycopg.connect(test_db) as c:
        last_call, follow_up = c.execute("select last_call_at, next_follow_up from leads where id = %s", [lead_id]).fetchone()
    assert last_call is not None
    assert follow_up == datetime(2037, 11, 5, 10, tzinfo=timezone.utc)


def test_a_call_with_no_callback_time_keeps_the_follow_up_a_rep_set(call_rows, test_db):
    lead_id = add_lead(test_db)
    rep_date = datetime(2037, 11, 5, 10, tzinfo=timezone.utc)
    with psycopg.connect(test_db) as c:
        c.execute("update leads set next_follow_up = %s where id = %s", [rep_date, lead_id])
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "no_answer"})
    with psycopg.connect(test_db) as c:
        assert c.execute("select next_follow_up from leads where id = %s", [lead_id]).fetchone()[0] == rep_date


def test_a_callback_redials_at_the_customers_time(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "callback", "callback_time": "Tue 3pm",
                  "callback_at": "2037-11-03T15:00:00+00:00"})
    assert lead(test_db, lead_id) == ("callback", 1, datetime(2037, 11, 3, 15, tzinfo=timezone.utc))
    assert call_rows()[0]["callback_at"] == datetime(2037, 11, 3, 15, tzinfo=timezone.utc)


def test_an_unreadable_callback_time_still_saves_the_call(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "callback", "callback_at": "next week sometime"})
    assert lead(test_db, lead_id)[0] == "interested"
    assert call_rows()[0]["callback_at"] is None


def test_saving_a_call_leaves_attempts_to_the_dialler(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "no_answer"})
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "no_answer"})
    assert lead(test_db, lead_id)[1] == 1


def test_callback_time_without_offset_is_read_as_irish_time(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "callback", "callback_at": "2037-07-01T10:00:00"})
    assert call_rows()[0]["callback_at"] == datetime(2037, 7, 1, 9, tzinfo=timezone.utc)  # Irish summer time is UTC+1


def test_callback_time_in_the_past_counts_as_no_usable_time(call_rows, test_db):
    lead_id = add_lead(test_db)
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "callback", "callback_at": "2020-01-01T10:00:00+00:00"})
    assert lead(test_db, lead_id) == ("interested", 1, None)


def test_the_campaigns_retry_settings_decide_the_next_call(call_rows, test_db):
    lead_id = add_lead(test_db)
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("insert into campaigns (name, max_attempts, retry_hours) values ('c', 1, 1)")
        c.execute("update leads set campaign_id = 1")
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "no_answer"})
    assert lead(test_db, lead_id) == ("exhausted", 1, None)
