"""Live-call API: the few things the static web app cannot do because they need the LiveKit secret.

Run: uv run uvicorn api.main:app --port 8000
Callers send a Supabase access token; only admin, sales_manager and salesperson may listen in.
"""

import asyncio
import os
import re
import uuid
from datetime import timedelta

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel

from shared import db

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
CALL_ROOM = re.compile(r"^call-")
PHONE = re.compile(r"^call-(\d+)-\d+$")  # the dialler's naming, see dialler.main
MAY_LISTEN = {"admin", "sales_manager", "salesperson"}


async def user_id_for(token: str) -> str | None:
    """The Supabase user this access token belongs to; None if Supabase rejects the token, 502 if it cannot answer."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{os.environ['SUPABASE_URL']}/auth/v1/user",
                headers={"Authorization": f"Bearer {token}", "apikey": os.environ["SUPABASE_PUBLISHABLE_KEY"]},
            )
    except httpx.HTTPError as e:
        raise HTTPException(502, "Could not reach Supabase") from e
    if r.status_code in (401, 403):
        return None
    if r.status_code != 200:  # a Supabase outage must not look like a bad token: the web app would sign the user out
        raise HTTPException(502, "Could not check your sign-in")
    return r.json().get("id")


async def listener(authorization: str | None = Header(None)) -> str:
    token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
    uid = await user_id_for(token) if token else None
    if not uid:
        raise HTTPException(401, "Sign in again")
    if await asyncio.to_thread(db.role_of, uid) not in MAY_LISTEN:
        raise HTTPException(403, "Your role cannot listen to calls")
    return uid


async def list_rooms() -> list[api.Room]:
    async with api.LiveKitAPI() as lk:
        return list((await lk.room.list_rooms(api.ListRoomsRequest())).rooms)


async def live_rooms() -> list[api.Room]:
    try:
        rooms = await list_rooms()
    except Exception as e:
        raise HTTPException(502, "Could not reach LiveKit") from e
    return [r for r in rooms if CALL_ROOM.match(r.name) and r.num_participants > 0]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/live-calls", dependencies=[Depends(listener)])
async def live_calls():
    return [
        {"room": r.name, "phone": "+" + m.group(1) if (m := PHONE.match(r.name)) else None, "started_at": r.creation_time}
        for r in await live_rooms()
    ]


class TokenRequest(BaseModel):
    room: str


@app.post("/live-token", dependencies=[Depends(listener)])
async def live_token(body: TokenRequest):
    if body.room not in {r.name for r in await live_rooms()}:
        raise HTTPException(404, "That call is not live")
    grants = api.VideoGrants(room_join=True, room=body.room, hidden=True, can_subscribe=True, can_publish=False,
                             can_publish_data=False)
    token = (api.AccessToken().with_identity(f"observer_{uuid.uuid4().hex}").with_grants(grants)
             .with_ttl(timedelta(hours=1)).to_jwt())
    return {"serverUrl": os.environ["LIVEKIT_URL"], "participantToken": token}
