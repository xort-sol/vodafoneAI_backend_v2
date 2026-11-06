import json
import os
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql:///vodafoneai_test")
MIGRATIONS = Path(__file__).resolve().parents[1] / "supabase" / "migrations"


@pytest.fixture(autouse=True)
def safe_env(monkeypatch):
    # importing agent loads .env, with the production database
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)


@pytest.fixture(scope="session")
def test_db() -> str:
    """A fresh local copy of the Supabase schema. First time: brew services start postgresql@18 && createdb vodafoneai_test"""
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade; drop schema if exists auth cascade; create schema public")
        conn.execute((Path(__file__).parent / "sql" / "supabase_stub.sql").read_text())
        for migration in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(migration.read_text())
    return TEST_DATABASE_URL


@pytest.fixture
def conn(test_db):
    """A connection whose changes are rolled back after the test."""
    with psycopg.connect(test_db) as c:
        # inside the transaction, so other tests' committed calls are back after the rollback
        c.execute("truncate calls, transcripts, suppression, audit_log cascade")
        yield c
        c.rollback()


@pytest.fixture
def as_user():
    """Act as a signed-in user (or signed out, with None) for the rest of the transaction, the way Supabase's API does."""

    def switch(conn, user_id: str | None) -> None:
        conn.execute("set local role " + ("authenticated" if user_id else "anon"))
        conn.execute("select set_config('request.jwt.claims', %s, true)", [json.dumps({"sub": user_id or ""})])

    return switch


@pytest.fixture
def call_rows(test_db):
    """Empties the call tables, then gives a reader for the calls saved during the test."""
    with psycopg.connect(test_db, autocommit=True) as c:
        c.execute("truncate calls, transcripts, suppression restart identity cascade")

    def read() -> list[dict]:
        with psycopg.connect(test_db, row_factory=dict_row) as c:
            return c.execute("select * from calls order by id").fetchall()

    return read
