import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterable, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import agents, api
from livekit.agents import AMD, Agent, AgentServer, AgentSession, JobProcess, RunContext, TurnHandlingOptions, function_tool, inference, llm
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import elevenlabs, silero

from analysis import analyse
from shared import db
from shared.results import record_result

load_dotenv()
logger = logging.getLogger("sales-agent")

BUSINESS_NAME = os.getenv("BUSINESS_NAME", "our company")
SERVICES = Path(__file__).with_name("services.md").read_text()
OBJECTIONS = Path(__file__).with_name("objections.md").read_text()
IRELAND = ZoneInfo("Europe/Dublin")
# Used when no lead is dispatched: console and web-app testers, whose name we don't know
# No notes: the agent reads notes as facts about the customer and said them out loud
TEST_LEAD = {"name": "", "phone": "", "company": "", "notes": ""}

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

Outcome = Literal["interested", "callback", "not_interested", "do_not_call"]


def who_line(lead: dict) -> str:
    if not lead.get("name"):
        return "You don't know their name, so don't use one."
    return f"You're calling {lead['name']} from {lead.get('company') or 'their company'}."


def lead_facts(lead: dict) -> str:
    facts = [
        f"{lead['staff']} staff" if lead.get("staff") else "",
        f"current plan: {lead['current_plan']}" if lead.get("current_plan") else "",
        f"contract ends: {lead['contract_end']}" if lead.get("contract_end") else "",
        lead.get("notes", ""),
    ]
    return "What we know: " + ("; ".join(f for f in facts if f) or "nothing yet") + "."


async def first_question_only(chunks: AsyncIterable) -> AsyncIterable:
    """Ends the reply at its first "?": callers answer one question, and no tool call follows a question."""
    async with contextlib.aclosing(chunks):  # stops the LLM stream instead of letting it generate unheard text
        async for chunk in chunks:
            text = chunk.delta.content if isinstance(chunk, llm.ChatChunk) and chunk.delta else None
            if text and "?" in text:
                chunk.delta.content, chunk.delta.tool_calls = text[: text.index("?") + 1], []
                yield chunk
                return
            yield chunk


def transcript_lines(items) -> list[tuple[str, str]]:
    """What was said, in order: (agent or customer, text). Delivery tags stay; the web app hides them."""
    speakers = {"assistant": "agent", "user": "customer"}
    return [(speakers[i.role], i.text_content) for i in items if i.type == "message" and i.role in speakers and i.text_content]


def now_line(now: datetime) -> str:
    return f"Right now it's {now:%A} {now.day} {now:%B %Y}, {now:%-I:%M %p} in Ireland."


class SalesAgent(Agent):
    def __init__(self, lead: dict) -> None:
        self.lead = lead
        self.recorded = False
        self.outcome = ""
        self.started = time.monotonic()  # whole session, ringing included: what the voice pipeline bills
        self.reached = False  # True once a person or voicemail is on the line
        super().__init__(
            instructions=f"""You are Sam, a warm, relaxed sales rep for {BUSINESS_NAME}, talking to a customer on the phone.
You are an AI assistant and must say so in your first sentence.

{now_line(datetime.now(IRELAND))}
{who_line(lead)} {lead_facts(lead)}
Use what we know to pick the right service and make the call relevant (a contract ending soon is a good moment to compare),
but don't read it back to them.

How you talk:
- Like a real person on a phone call: contractions, plain words, then let them talk.
- Keep every turn under twenty words: one idea, one question. Callers can't scroll back, so long sentences get lost.
- Exactly one question per turn, always at the end. Use small acknowledgements like "sure", "got it", "makes sense", but don't repeat their words back.
- If they ask you something, answer it first, honestly: if the answer isn't in the services below, say a team member will confirm.
- If they interrupt you, stop and respond to what they just said. Never restart your pitch from the top.
- If they say "sorry?", "what?" or "pardon?", they didn't catch you: say your last point again in fewer, simpler words,
  with no new details, and ask the same question again.
- Say numbers and prices the way people speak them ("forty euro a month"). No symbols, lists, abbreviations or markdown.

How you sound (your voice reads these marks; the customer never sees them):
- Start every turn with one delivery tag in square brackets that fits the moment: [warmly], [curious], [empathetic],
  [reassuring], [thoughtful], [excited] or [cheerfully]. You may add a second tag before a later sentence when the mood shifts.
- Let the mood build across the call. Start relaxed and friendly, get warmer as they open up, sound genuinely pleased when they say yes.
  Match their feelings: if they're frustrated, sound sympathetic, not cheerful. Save [excited] for real good news.
- Slow down for what matters: put "..." before the price or a key point, and keep that sentence short.
- Stress the one word that carries the point in CAPITALS, at most once per turn: "that's REALLY frustrating", "it starts at... forty euro".
- Tags and capitals don't count toward your twenty words.
  Example: "[empathetic] Oh, that's REALLY frustrating. [curious] What happens when it drops?"

How the call goes:
- Open: greet them (by name if you know it), say you're an AI assistant from {BUSINESS_NAME}, and ask if they have a quick minute.
  Only a clear yes ("sure", "yes", "go ahead") counts as permission. A bare "hi" or "hello" is not: briefly ask again if now's okay.
  If it's a bad time, ask when is better, record_outcome "callback", and wrap up.
- Discover: ask an open question about how things are going in the area our services cover, then one follow-up that builds on their answer.
  If they hint at a problem ("not really", "it's okay I guess", "could be better"), ask what's going on before you pitch anything.
- Pitch: pick the one service that fits what they told you. Give the main benefit and the starting price in one short sentence,
  then ask if they'd like to hear more. Share more details only if they ask. Don't open with "you mentioned".
  Never ask them to choose between plans or speeds; a team member helps pick on the follow-up call.
- Objections: acknowledge it in a few words, then give the matching answer from the objection answers below, in your own short words.
  Don't repeat the pitch or the price.
  Then ask if they'd like a team member to follow up or would rather leave it, and respect their answer. Never push twice.
- Close: after the pitch and at most one round of details, stop selling: no more features, plans or discovery questions.
  If they sound interested or give a soft reply like "okay", "sounds good" or "maybe", ask just "When would suit a team member
  to give you a call about it?", with no extra "would you like" question in front. If they say they're not interested, respect that.
  Once they give a time, record_outcome "interested" with that callback_time.
  Write callback_time as the time they said plus the exact day and date, worked out from today's date,
  like "tomorrow at 10am (Saturday 31 October), by phone". Never guess a time they didn't give.
  Also pass callback_at: the same moment in ISO 8601 with the Irish UTC offset, like "2026-10-31T10:00:00+00:00".
  In your goodbye, repeat the time they gave you so they know what happens next.
  If they're not, thank them warmly and record_outcome "not_interested". If they ask not to be called again, apologise and record_outcome "do_not_call".
- Only record an outcome once the customer has clearly answered. Never call record_outcome or end_call in a turn where you ask a question;
  wait for their reply first. After record_outcome, say a short friendly goodbye with no question and call end_call.

Rules: only use facts and prices from the services below. If you don't know something, say a team member will follow up with the details.
Call record_outcome exactly once, before ending the call.

Services:
{SERVICES}

Objection answers:
{OBJECTIONS}

Before every reply, check:
0. Have they clearly said yes to talking now? If not (for example they only said "hi"), just ask if now's a good time,
   like "[warmly] Is now an okay time for a quick chat?". Don't say what the call is about yet.
1. One question only, at the very end. No add-on second question like "Has it been slow or costly?".
2. Under twenty words.
3. Did they hint at a problem? Then ask about it; don't pitch yet.
4. Asking a question? Then don't call record_outcome or end_call this turn.
4b. Already pitched and they said "okay" or similar? Then close: ask when a team member could call them.
5. Does it start with a delivery tag that fits how they feel right now?""",
            tools=[EndCallTool(extra_description="Call record_outcome before this.")],
        )

    def llm_node(self, chat_ctx, tools, model_settings):
        # gpt-4.1-mini still adds a second question now and then, despite the prompt
        return first_question_only(Agent.default.llm_node(self, chat_ctx, tools, model_settings))

    @function_tool()
    async def record_outcome(
        self,
        context: RunContext,
        outcome: Outcome,
        interested_services: str = "",
        callback_time: str = "",
        callback_at: str = "",
        notes: str = "",
    ) -> str:
        """Save the result of this call for the sales team.

        Args:
            outcome: interested, callback, not_interested, or do_not_call.
            interested_services: Services the customer showed interest in, comma separated.
            callback_time: When and how the customer wants to be contacted, if they said.
            callback_at: The callback_time moment in ISO 8601 with the Irish UTC offset, if they gave a time.
            notes: One-sentence summary of the conversation and any objections.
        """
        if outcome in ("interested", "callback") and not callback_time:
            # The sales team can't act on a lead without a time; make the agent ask instead of saving
            return "Not saved. First ask when and how a team member should follow up, then call record_outcome again with callback_time."
        await self.save(
            outcome, interested_services=interested_services, callback_time=callback_time, callback_at=callback_at,
            notes=notes,
        )
        return "Saved. Now say goodbye and end the call."

    async def save(self, outcome: str, **fields: str) -> None:
        if self.recorded:
            return
        self.recorded = True
        self.outcome = outcome
        # The Google Sheet request can take a second; keep it off the audio loop
        await asyncio.to_thread(record_result, {**self.lead, "outcome": outcome, **fields})
        logger.info("recorded outcome %s for %s", outcome, self.lead.get("phone"))

    async def finish(self, lines: list[tuple[str, str]], latency_ms: int | None = None) -> None:
        """At hang-up: save a hung_up outcome if none was recorded, then the transcript, timings and analysis."""
        await self.record_hang_up()
        if not self.recorded:
            return
        job_id = self.lead["job_id"]
        try:
            await asyncio.to_thread(db.save_transcript, job_id, lines)
            await asyncio.to_thread(db.save_call_stats, job_id, round(time.monotonic() - self.started), latency_ms)
        except Exception as e:
            logger.error("could not save transcript or timings for job %s: %s", job_id, e)
        try:
            await analyse.analyse_call(job_id, lines, self.outcome)
        except Exception as e:  # e.g. database down: the call row and transcript are already saved
            logger.error("could not save analysis for job %s: %s", job_id, e)

    async def record_hang_up(self) -> None:
        # Not reached means a setup error (bad trunk, outage): leave no row so the dialler retries once the lead goes stale
        if self.reached:
            await self.save("hung_up")


class SilenceTracker:
    """First silence: check in. Second silence in a row: end the call."""

    def __init__(self) -> None:
        self.silences = 0

    def on_away(self) -> Literal["ask", "end"]:
        self.silences += 1
        return "ask" if self.silences == 1 else "end"

    def on_speaking(self) -> None:
        self.silences = 0


def should_open_call(category: str, transcript: str) -> bool:
    # If the callee spoke ("Hello?"), the session already replies to that turn
    return category == "uncertain" and not transcript


def build_session(vad) -> AgentSession:
    return AgentSession(
        vad=vad,  # adaptive interruption needs a VAD
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(
            voice_id=os.getenv("ELEVEN_VOICE_ID") or elevenlabs.tts.DEFAULT_VOICE_ID,
            # expressive and as fast as flash in our test (0.17s to first audio); reads [warmly]-style tags.
            # Only stability applies: pacing comes from the text ("...", short sentences, CAPS for stress)
            model="eleven_v4_turbo",
            voice_settings=elevenlabs.VoiceSettings(stability=0.4, similarity_boost=0.8),
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # dynamic learns each caller's pause length; 1.5s cap stops short answers ("Yeah.") waiting the old 2.5s
            endpointing={"mode": "dynamic", "max_delay": 1.5},
            # adaptive: ignores "uh-huh"/"okay"/coughs, stops for real barge-ins, resumes after false stops
            interruption={"mode": "adaptive", "resume_false_interruption": True, "false_interruption_timeout": 1.5},
            preemptive_generation={"enabled": True, "preemptive_tts": True},
        ),
        user_away_timeout=8.0,
    )


def build_amd(session: AgentSession, identity: str) -> AMD:
    # AMD's default STT (cartesia/ink-whisper) isn't served in LiveKit's EU region; reuse the session's model
    return AMD(session, participant_identity=identity, stt=session.stt)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# LiveKit's 10 s default can cut off the hang-up work: Google Sheet (15 s timeout) and analysis (30 s)
server = AgentServer(shutdown_process_timeout=60)
server.setup_fnc = prewarm


@server.rtc_session(agent_name="sales-agent")
async def sales_call(ctx: agents.JobContext):
    lead = json.loads(ctx.job.metadata) if ctx.job.metadata else TEST_LEAD
    # room links the row to the session in the LiveKit dashboard; job_id keys it (console room names repeat)
    lead = {**lead, "room": ctx.room.name, "job_id": ctx.job.id}
    trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
    phone_mode = bool(trunk_id and lead.get("phone"))
    agent = SalesAgent(lead)
    agent.reached = not phone_mode  # a console or web tester is on the line from the start

    async def on_shutdown():
        await agent.finish(transcript_lines(session.history.items), analyse.avg_latency_ms(session.history.items))

    ctx.add_shutdown_callback(on_shutdown)

    session = build_session(vad=ctx.proc.userdata["vad"])
    silence = SilenceTracker()

    async def hang_up_on_silence():
        await agent.save("no_response")
        handle = session.generate_reply(
            instructions="They still haven't answered. Say a brief, friendly goodbye.", allow_interruptions=False
        )
        await handle.wait_for_playout()
        await ctx.delete_room()  # hangs up the phone line

    @session.on("user_state_changed")
    def on_user_state(ev):
        if phone_mode and not agent.reached:
            return  # still ringing
        if ev.new_state == "speaking":
            silence.on_speaking()
        elif ev.new_state == "away":
            if silence.on_away() == "ask":
                session.generate_reply(instructions="The customer has gone quiet. Gently check if they're still there, in a few words.")
            else:
                asyncio.create_task(hang_up_on_silence())

    await session.start(room=ctx.room, agent=agent)

    if not phone_mode:
        # Test mode (Agent Console): no phone line, just start the call
        await session.generate_reply(instructions="Open the call.")
        return

    identity = f"sip-{lead['phone']}"
    async with build_amd(session, identity) as detector:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=trunk_id,
                    sip_call_to=lead["phone"],
                    room_name=ctx.room.name,
                    participant_identity=identity,
                    wait_until_answered=True,
                )
            )
        except api.SipCallError as e:
            logger.info("call not answered: %s", e)
            await agent.save("no_answer", notes=str(e))
            ctx.shutdown("no answer")
            return
        await ctx.wait_for_participant(identity=identity)
        agent.reached = True
        result = await detector.execute()

    # machine-ivr is navigated automatically by AMD; human/uncertain continue as a normal conversation
    if result.category == "machine-vm":
        handle = session.generate_reply(
            instructions=f"You reached voicemail. Leave a brief message: you're an AI assistant from {BUSINESS_NAME}, "
            "calling about services that may help them, and a team member will try again. Then stop."
        )
        await handle.wait_for_playout()
        await agent.save("voicemail", notes=result.transcript)
        ctx.shutdown("voicemail")
    elif result.category == "machine-unavailable":
        await agent.save("unreachable", notes=result.transcript)
        ctx.shutdown("mailbox unavailable")
    elif should_open_call(result.category, result.transcript):
        await session.generate_reply(instructions="The customer answered but is silent. Open the call.")


if __name__ == "__main__":
    agents.cli.run_app(server)
