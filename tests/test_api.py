import asyncio

import httpx
import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from livekit import api as lk

from api import main
from shared import db

AUTH = {"Authorization": "Bearer good"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 32)
    return TestClient(main.app)


@pytest.fixture
def signed_in(monkeypatch):
    async def user_id(token):
        return "u1" if token == "good" else None

    monkeypatch.setattr(main, "user_id_for", user_id)
    return lambda role: monkeypatch.setattr(db, "role_of", lambda uid: role)


@pytest.fixture
def rooms(monkeypatch):
    def set_rooms(items):
        async def fake():
            return [lk.Room(name=n, num_participants=p, creation_time=1793710403) for n, p in items]

        monkeypatch.setattr(main, "list_rooms", fake)

    return set_rooms


@pytest.fixture
def rooms_fail(monkeypatch):
    async def boom():
        raise RuntimeError("livekit down")

    monkeypatch.setattr(main, "list_rooms", boom)


def jwt_video_grant(token):
    return jwt.decode(token, options={"verify_signature": False})["video"]


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_live_calls_needs_a_token(client):
    assert client.get("/live-calls").status_code == 401


def test_invalid_token_is_401(client, signed_in):
    assert client.get("/live-calls", headers={"Authorization": "Bearer bad"}).status_code == 401


def test_analyst_cannot_see_live_calls(client, signed_in):
    signed_in("analyst")
    assert client.get("/live-calls", headers=AUTH).status_code == 403


def test_user_without_a_role_is_refused(client, signed_in):
    signed_in(None)
    assert client.get("/live-calls", headers=AUTH).status_code == 403


def test_only_customer_call_rooms_are_listed(client, signed_in, rooms):
    signed_in("salesperson")
    rooms([("call-353871234567-1793710403", 2), ("web_test_x", 2), ("call-353870000000-1", 0)])
    body = client.get("/live-calls", headers=AUTH).json()
    assert body == [{"room": "call-353871234567-1793710403", "phone": "+353871234567", "started_at": 1793710403}]


def test_token_is_hidden_and_listen_only(client, signed_in, rooms):
    signed_in("sales_manager")
    rooms([("call-353871234567-1793710403", 2)])
    body = client.post("/live-token", json={"room": "call-353871234567-1793710403"}, headers=AUTH).json()
    assert body["serverUrl"] == "wss://example.livekit.cloud"
    grant = jwt_video_grant(body["participantToken"])
    assert grant["hidden"] is True and grant["canPublish"] is False and grant["canSubscribe"] is True
    assert grant["canPublishData"] is False and grant["room"] == "call-353871234567-1793710403"


def test_token_for_a_room_that_is_not_live_is_404(client, signed_in, rooms):
    signed_in("sales_manager")
    rooms([])
    assert client.post("/live-token", json={"room": "call-1-1"}, headers=AUTH).status_code == 404


def test_livekit_outage_returns_502(client, signed_in, rooms_fail):
    signed_in("salesperson")
    assert client.get("/live-calls", headers=AUTH).status_code == 502


def test_role_of_reads_the_profile(test_db):
    import uuid

    import psycopg

    uid = str(uuid.uuid4())
    with psycopg.connect(test_db, autocommit=True) as conn:
        conn.execute("insert into auth.users (id, email) values (%s, 'r@example.com')", [uid])
        assert db.role_of(uid) is None  # signed up, no role yet
        conn.execute("update public.profiles set role = 'salesperson' where id = %s", [uid])
        assert db.role_of(uid) == "salesperson"
        conn.execute("delete from auth.users where id = %s", [uid])
    assert db.role_of(uid) is None


def test_live_token_needs_a_token(client):
    assert client.post("/live-token", json={"room": "call-1-1"}).status_code == 401


def test_analyst_cannot_get_a_live_token(client, signed_in, rooms):
    signed_in("analyst")
    rooms([("call-353871234567-1793710403", 2)])
    assert client.post("/live-token", json={"room": "call-353871234567-1793710403"}, headers=AUTH).status_code == 403


def test_token_for_a_non_call_room_is_404(client, signed_in, rooms):
    signed_in("sales_manager")
    rooms([("web_test_x", 2)])
    assert client.post("/live-token", json={"room": "web_test_x"}, headers=AUTH).status_code == 404


def test_call_room_without_digits_lists_a_null_phone_and_can_be_listened_to(client, signed_in, rooms):
    signed_in("salesperson")
    rooms([("call-abc-1", 2)])
    assert client.get("/live-calls", headers=AUTH).json() == [{"room": "call-abc-1", "phone": None, "started_at": 1793710403}]
    assert client.post("/live-token", json={"room": "call-abc-1"}, headers=AUTH).status_code == 200


def supabase(monkeypatch, handler):
    """Route user_id_for's HTTP calls to handler instead of the network."""
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "pub-key")
    real = httpx.AsyncClient
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_supabase_check_sends_the_token_and_the_publishable_key(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "u1"})

    supabase(monkeypatch, handler)
    assert asyncio.run(main.user_id_for("tok")) == "u1"
    assert seen["apikey"] == "pub-key" and seen["authorization"] == "Bearer tok"
    assert seen["url"] == "https://x.supabase.co/auth/v1/user"


@pytest.mark.parametrize("status", [401, 403])
def test_supabase_rejecting_the_token_is_401(monkeypatch, status):
    supabase(monkeypatch, lambda r: httpx.Response(status))
    assert asyncio.run(main.user_id_for("tok")) is None


@pytest.mark.parametrize("status", [429, 500, 503])
def test_supabase_failing_is_502_not_401(monkeypatch, status):
    supabase(monkeypatch, lambda r: httpx.Response(status))
    with pytest.raises(HTTPException) as e:
        asyncio.run(main.user_id_for("tok"))
    assert e.value.status_code == 502


def test_supabase_unreachable_is_502(monkeypatch):
    def handler(request):
        raise httpx.ConnectTimeout("slow")

    supabase(monkeypatch, handler)
    with pytest.raises(HTTPException) as e:
        asyncio.run(main.user_id_for("tok"))
    assert e.value.status_code == 502
