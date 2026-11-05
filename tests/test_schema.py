"""Row Level Security on the Supabase schema: what each role may read and change."""

import uuid

import psycopg
import pytest

ROLES = ["admin", "sales_manager", "salesperson", "analyst"]


def new_user(conn, role: str | None = "salesperson") -> str:
    user_id = str(uuid.uuid4())
    conn.execute("insert into auth.users (id, email) values (%s, %s)", [user_id, f"{user_id[:8]}@example.com"])
    if role:
        conn.execute("update profiles set role = %s where id = %s", [role, user_id])
    return user_id


def add_call(conn) -> None:
    call_id = conn.execute(
        "insert into calls (job_id, outcome) values (%s, 'hung_up') returning id", [str(uuid.uuid4())]
    ).fetchone()[0]
    conn.execute("insert into transcripts (call_id, position, role, text) values (%s, 0, 'agent', 'Hi')", [call_id])


def count(conn, table: str) -> int:
    return conn.execute(f"select count(*) from {table}").fetchone()[0]


def test_new_user_has_no_role_and_sees_no_calls(conn, as_user):
    # Supabase allows sign-ups by default: a stranger who signs up must not see customer calls
    add_call(conn)
    user_id = new_user(conn, role=None)
    assert conn.execute("select role from profiles where id = %s", [user_id]).fetchone()[0] is None
    as_user(conn, user_id)
    assert count(conn, "calls") == 0


def test_signed_out_sees_no_calls(conn, as_user):
    add_call(conn)
    as_user(conn, None)
    assert count(conn, "calls") == 0


def test_user_without_profile_sees_no_calls(conn, as_user):
    add_call(conn)
    user_id = new_user(conn)
    conn.execute("delete from profiles where id = %s", [user_id])
    as_user(conn, user_id)
    assert count(conn, "calls") == 0


def test_any_role_sees_calls_and_transcripts(conn, as_user):
    add_call(conn)
    users = {role: new_user(conn, role) for role in ROLES}
    for role, user_id in users.items():
        as_user(conn, user_id)
        assert (count(conn, "calls"), count(conn, "transcripts")) == (1, 1), role
        conn.execute("reset role")


def test_only_managers_and_admins_see_suppression(conn, as_user):
    conn.execute("insert into suppression (phone, source) values ('+353870000000', 'call')")
    users = {role: new_user(conn, role) for role in ROLES}
    for role, user_id in users.items():
        as_user(conn, user_id)
        assert count(conn, "suppression") == (1 if role in ("admin", "sales_manager") else 0), role
        conn.execute("reset role")


def test_only_admins_see_audit_log(conn, as_user):
    conn.execute("insert into audit_log (action) values ('test')")
    users = {role: new_user(conn, role) for role in ROLES}
    for role, user_id in users.items():
        as_user(conn, user_id)
        assert count(conn, "audit_log") == (1 if role == "admin" else 0), role
        conn.execute("reset role")


def test_only_admins_change_roles(conn, as_user):
    salesperson, admin = new_user(conn, "salesperson"), new_user(conn, "admin")
    as_user(conn, salesperson)
    assert conn.execute("update profiles set role = 'admin' where id = %s", [salesperson]).rowcount == 0
    conn.execute("reset role")
    as_user(conn, admin)
    assert conn.execute("update profiles set role = 'sales_manager' where id = %s", [salesperson]).rowcount == 1


def allowed(conn, sql: str, params=()) -> bool:
    """Run one write as the current user; False when RLS blocks it or changes no rows."""
    try:
        with conn.transaction():  # savepoint, so a blocked write doesn't abort the test's transaction
            return conn.execute(sql, params).rowcount > 0
    except psycopg.errors.InsufficientPrivilege:
        return False


def test_only_managers_and_admins_create_campaigns_and_import_leads(conn, as_user):
    users = {role: new_user(conn, role) for role in ROLES}
    for role, user_id in users.items():
        as_user(conn, user_id)
        manager = role in ("admin", "sales_manager")
        assert allowed(conn, "insert into campaigns (name) values ('c')") == manager, role
        assert allowed(conn, "insert into leads (phone) values (%s)", [f"+3538{role}"]) == manager, role
        conn.execute("reset role")


def test_only_managers_and_admins_change_campaigns(conn, as_user):
    users = {role: new_user(conn, role) for role in ROLES}
    conn.execute("insert into campaigns (name) values ('c')")
    for role, user_id in users.items():
        as_user(conn, user_id)
        manager = role in ("admin", "sales_manager")
        assert allowed(conn, "update campaigns set status = 'approved'") == manager, role
        conn.execute("reset role")


def test_team_writes_audit_log_only_as_themselves(conn, as_user):
    me, other = new_user(conn, "salesperson"), new_user(conn, "admin")
    as_user(conn, me)
    assert allowed(conn, "insert into audit_log (actor, action) values (%s, 'x')", [me])
    assert not allowed(conn, "insert into audit_log (actor, action) values (%s, 'x')", [other])


def test_lead_status_is_one_of_the_dialler_states(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("insert into leads (phone, status) values ('+1', 'bogus')")


def add_task(conn, assigned_to=None) -> int:
    return conn.execute(
        "insert into tasks (kind, assigned_to) values ('follow_up', %s) returning id", [assigned_to]
    ).fetchone()[0]


def test_salespeople_claim_unclaimed_tasks(conn, as_user):
    task = add_task(conn)
    me = new_user(conn, "salesperson")
    as_user(conn, me)
    assert allowed(conn, "update tasks set assigned_to = %s, claimed_at = now() where id = %s", [me, task])


def test_nobody_but_managers_takes_a_task_someone_else_claimed(conn, as_user):
    owner, other, manager = new_user(conn, "salesperson"), new_user(conn, "salesperson"), new_user(conn, "sales_manager")
    task = add_task(conn, assigned_to=owner)
    as_user(conn, other)
    assert not allowed(conn, "update tasks set assigned_to = %s where id = %s", [other, task])
    assert not allowed(conn, "update tasks set done = true where id = %s", [task])
    conn.execute("reset role")
    as_user(conn, manager)
    assert allowed(conn, "update tasks set assigned_to = %s where id = %s", [manager, task])


def test_a_salesperson_cannot_claim_a_task_for_someone_else(conn, as_user):
    me, other = new_user(conn, "salesperson"), new_user(conn, "salesperson")
    task = add_task(conn)
    as_user(conn, me)
    assert not allowed(conn, "update tasks set assigned_to = %s where id = %s", [other, task])


def test_owner_marks_their_task_done(conn, as_user):
    me = new_user(conn, "salesperson")
    task = add_task(conn, assigned_to=me)
    as_user(conn, me)
    assert allowed(conn, "update tasks set done = true where id = %s", [task])


def test_analysts_change_no_tasks(conn, as_user):
    task = add_task(conn)
    analyst = new_user(conn, "analyst")
    as_user(conn, analyst)
    assert not allowed(conn, "update tasks set assigned_to = %s where id = %s", [analyst, task])


def test_dashboard_counts_the_funnel_per_day(conn):
    for outcome in ["interested", "no_answer", "voicemail", "not_interested", "do_not_call"]:
        conn.execute(
            "insert into calls (job_id, outcome, duration_s, created_at) values (%s, %s, 60, '2026-11-03 10:00+00')",
            [outcome, outcome],
        )
    row = conn.execute(
        "select day, dialled, reached, interested, opt_outs, minutes from dashboard_daily"
    ).fetchone()
    assert row[1:] == (5, 3, 1, 1, 5)


def test_a_number_can_be_several_leads(conn):
    # test campaigns ring one phone many times
    conn.execute("insert into leads (phone) values ('+353857180577'), ('+353857180577')")
    assert conn.execute("select count(*) from leads where phone = '+353857180577'").fetchone()[0] == 2


@pytest.fixture
def users(conn):
    return {role: new_user(conn, role) for role in ROLES}


def test_salesperson_can_set_sales_status_but_not_dialler_state(conn, as_user, users):
    lead_id = conn.execute("insert into leads (phone) values ('+353870000001') returning id").fetchone()[0]
    as_user(conn, users["salesperson"])
    assert conn.execute("update leads set sales_status = 'Converted' where id = %s", [lead_id]).rowcount == 1
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("update leads set attempts = 0 where id = %s", [lead_id])


def test_analyst_cannot_change_leads(conn, as_user, users):
    lead_id = conn.execute("insert into leads (phone) values ('+353870000002') returning id").fetchone()[0]
    as_user(conn, users["analyst"])
    assert conn.execute("update leads set sales_status = 'Hot' where id = %s", [lead_id]).rowcount == 0


def test_every_role_can_read_team_names(conn, as_user, users):
    as_user(conn, users["analyst"])
    assert conn.execute("select count(*) from profiles").fetchone()[0] >= 4


def test_calling_hours_must_end_after_they_start(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("insert into campaigns (name, calling_hours_start, calling_hours_end) values ('x', '22:00', '06:00')")


def test_sellers_add_suppression_and_managers_remove_it(conn, as_user, users):
    as_user(conn, users["salesperson"])
    assert allowed(conn, "insert into suppression (phone, source) values ('+353870000003', 'manual')", [])
    assert conn.execute("delete from suppression where phone = '+353870000003'").rowcount == 0
