import asyncio
import contextlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import AsyncIterable, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from livekit import agents, api
from livekit.agents import (AMD, Agent, AgentServer, AgentSession, JobProcess, RunContext, ToolError, TurnHandlingOptions,
                            function_tool, get_job_context, inference, llm)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import WarmTransferTask
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

GOODBYE = ("Say one short, friendly goodbye with no question. If a callback time was agreed, repeat it so they know what happens next. "
           "For a visit request, say the team will confirm the visit time. For an email request, say a team member will email them. "
           "Never say anything is booked, confirmed or sent.")

Outcome = Literal["interested", "callback", "not_interested", "do_not_call", "field_visit", "email_request"]
# calls.follow_up's reason for each outcome a person (or the dialler) follows up
FOLLOW_UP_REASON = {"interested": "callback", "callback": "callback", "field_visit": "field_visit", "email_request": "email"}
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")  # basic shape only: a team member checks it before writing


def specialist_phone(value: str) -> str:
    """The number live transfers dial, if it's valid E.164; "" turns transfer off."""
    return value if re.fullmatch(r"\+[1-9]\d{7,14}", value) else ""


# Only ever from env, never from the model
SALES_SPECIALIST_PHONE = specialist_phone(os.getenv("SALES_SPECIALIST_PHONE", ""))


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


def in_sales_hours(callback_at: str) -> str:
    """"Ring me at 3" means the afternoon, but the model writes 03:00 whatever the prompt says, so a time before 8am moves on 12 hours."""
    try:
        moment = datetime.fromisoformat(callback_at)
    except ValueError:
        return callback_at
    return (moment + timedelta(hours=12)).isoformat() if moment.hour < 8 else callback_at


def opening_line(lead: dict) -> str:
    first_name = (lead.get("name") or "").split()[:1]
    greeting = f"Hi {first_name[0]}" if first_name else "Hi"
    return f"{greeting}, I'm an AI assistant calling on behalf of {BUSINESS_NAME}. We're offering local businesses a free connectivity review. Have you got a few minutes?"


def has_disclosed(chat_ctx) -> bool:
    """True once the customer heard the disclosure; a line cut off before it doesn't count."""
    return any(i.type == "message" and i.role == "assistant" and "AI assistant" in (i.text_content or "") for i in chat_ctx.items)


def transcript_lines(items, started_at: float) -> list[tuple[str, str, int]]:
    """What was said, in order: (agent or customer, text, seconds from started_at). """
    speakers = {"assistant": "agent", "user": "customer"}
    return [
        (speakers[i.role], i.text_content, round(i.created_at - started_at))
        for i in items
        if i.type == "message" and i.role in speakers and i.text_content
    ]


def now_line(now: datetime) -> str:
    return f"Right now it's {now:%A} {now.day} {now:%B %Y}, {now:%-I:%M %p} in Ireland. UTC offset: {now.isoformat()[-6:]}"


class SalesAgent(Agent):
    def __init__(self, lead: dict, phone_mode: bool = False) -> None:
        self.lead = lead
        self.phone_mode = phone_mode
        self.transferring = False  # set from dialling the specialist onwards: the AI must never speak over them
        self.transfer_failed = False
        can_transfer = bool(SALES_SPECIALIST_PHONE and phone_mode)
        self.recorded = False
        self.started_at = time.time()  # wall clock, to date each transcript line
        self.outcome = ""
        self.started = time.monotonic()  # whole session, ringing included: what the voice pipeline bills
        self.reached = False  # True once a person or voicemail is on the line
        self.on_voicemail = False  # voicemail has its own message, not the opening
        super().__init__(
            instructions=f"""You are Sam, a warm, relaxed sales rep for {BUSINESS_NAME}, talking to a customer on the phone.
You are an AI assistant and must say so in your first sentence.

{now_line(datetime.now(IRELAND))}
{who_line(lead)} {lead_facts(lead)}
Use what we know to skip questions it already answers (a contract ending soon is a good moment to compare),
but don't read it back to them.

How you talk:
- Like a real person on a phone call: contractions, plain words, then let them talk.
- Keep every turn under twenty words: one idea, one question. Callers can't scroll back, so long sentences get lost.
  The one exception is the summary, which can run to about thirty-five words.
- Exactly one question per turn, always at the end. Use small acknowledgements like "sure", "got it", "makes sense", but don't repeat their words back.
- If they ask you something, answer it first, honestly: if the answer isn't in the services below, say a team member will confirm.
- If they interrupt you, stop and respond to what they just said. Never restart from the top.
- If they say "sorry?", "what?" or "pardon?", they didn't catch you: say your last point again in fewer, simpler words,
  with no new details, and ask the same question again.
- Say numbers and prices the way people speak them ("forty euro a month"). No symbols, lists, abbreviations or markdown.

How you sound:
- Never write tags or stage directions in square brackets: everything you write is spoken.
- Let the mood build across the call. Start relaxed and friendly, get warmer as they open up, sound genuinely pleased when they say yes.
  Match their feelings: if they're frustrated, sound sympathetic, not cheerful.
- Slow down for what matters: put "..." before the price or a key point, and keep that sentence short.
- Stress the one word that carries the point in CAPITALS, at most once per turn: "that's REALLY frustrating", "it starts at... forty euro".
- Capitals don't count toward your twenty words.
  Example: "Oh, that's REALLY frustrating. What happens when it drops?"

How the call goes:
- Open: your first line is spoken for you: it greets them, says you're an AI assistant calling on behalf of {BUSINESS_NAME},
  offers a free connectivity review and asks if they've got a few minutes. Don't repeat it.
  Only a clear yes ("sure", "yes", "go ahead") counts as permission. A bare "hi" or "hello" is not: briefly ask again if they've got a few minutes.
  If they ask what it's about, say it's a quick look at their broadband and mobiles to see if anything could work better, then ask again.
  If it's a bad time, ask when is better, record_outcome "callback", and wrap up.
- Discover: you're here to understand their business, not to sell. Never start by selling or naming a plan.
  Ask one natural question per turn that builds on their last answer. Never read out a list of questions.
  Roughly in this order, skipping anything we already know:
  their business (what they do, how many staff, how many sites, who decides on phones and internet),
  then broadband (provider, what they pay, when the contract ends, any drop-outs or slow times),
  then mobiles (how many, which network, what they pay, when the contract ends, coverage),
  then what frustrates them most and what it costs the business.
  Bring up Microsoft 365, business phone lines or landlines only when relevant: five or more staff, they use the cloud or Teams, or they mention it.
  If an answer is vague and it matters, like what they pay, when the contract ends or who decides, ask one short follow-up.
  Offer ranges only if they don't know the figure: for broadband "around fifty, seventy-five, or a hundred plus a month?",
  for mobiles "around a hundred, two fifty, or five hundred plus a month?".
  Aim for three to five minutes; take longer only if they're engaged and the opportunity is bigger.
  If they ask a price, give it exactly as written in the services below, then carry on with your questions.
- Summarise: once discovery is done, before suggesting a next step, one turn that sums up the key facts, ending "Have I got that right?".
  Not when it's a bad time, they want an email straight away, or they're not interested: go straight to that.
  If they correct you, thank them and use the corrected facts.
- Next step: pick the one that fits what you learned, suggest it, and go ahead only if they agree.
  - They want to talk to someone now: {"offer to put them through to a sales specialist. Once they clearly agree to go now, call transfer_to_specialist." if can_transfer else "offer a callback with a team member."}
  - Callback with a team member: ask "Would tomorrow or the next day suit?", then "Morning or afternoon?", then confirm the exact day and time.
    record_outcome "interested".
  - Field visit: for several sites, connections or services, a more complex setup, or if they'd rather meet, offer a visit from a team member.
    Ask when suits, the address, and who they'll meet. record_outcome "field_visit" with visit_address, contact_name, and attendees if they said.
  - Renewal: if they're in contract and open to comparing before it renews, ask when it ends, then agree a time before then.
    record_outcome "callback" with reason "renewal" and contract_end. If "What we know" gives the contract end, use that date rather than "unknown"; pass "unknown" only if nobody knows.
  - Email: if they'd rather get details by email, ask which services interest them, then their business email, and read it back.
    record_outcome "email_request" with email and products.
  A "yes" or "sure" is not a time: ask when, and never write "now" or invent one.
  Team members work Monday to Friday: if they pick a Saturday or Sunday, ask if Monday at that time suits.
  Write callback_time as the time they said plus the exact day and date, worked out from today's date,
  like "tomorrow at 10am (Saturday 31 October), by phone". Never guess a time they didn't give.
  Also pass callback_at: the same moment in ISO 8601 using the UTC offset given above.
  If they're not interested, thank them warmly and record_outcome "not_interested". If they ask not to be called again, apologise and record_outcome "do_not_call".
- Objections: acknowledge it in a few words, then give the matching answer from the objection answers below, in your own short words.
  Never push twice.
- Only record an outcome once the customer has clearly answered. Never call record_outcome or end_call in a turn where you ask a question;
  wait for their reply first. After record_outcome, call end_call straight away and say nothing yourself: ending the call says the goodbye.

Rules: only use facts and prices from the services below, and give prices only when asked. If you don't know something, say a team member will follow up with the details.
Never promise savings, speeds or coverage. Never criticise another provider.
Call record_outcome exactly once, before ending the call.

Services:
{SERVICES}

Objection answers:
{OBJECTIONS}

Before every reply, check:
0. Have they clearly said yes to talking now? If not (for example they only said "hi"), just ask again,
   saying exactly "Have you got a few minutes?". Add nothing else: no company, services or reason for the call.
1. One question only, at the very end. No add-on second question like "Has it been slow or costly?".
2. Under twenty words, unless it's the summary.
3. Did they hint at a problem? Then ask about it. No plans or prices unless they asked.
4. Asking a question? Then don't call record_outcome or end_call this turn.
5. Discovery done and about to suggest a next step? Have you summarised and heard "yes, that's right" first?""",
            # the only goodbye: the model saying its own as well gave two, and no instructions gave none
            tools=[EndCallTool(extra_description="Call record_outcome before this. Never use this to transfer.", end_instructions=GOODBYE)],
        )

    def llm_node(self, chat_ctx, tools, model_settings):
        if self.transferring:
            return ""  # on hold, or the specialist has the customer: say nothing
        # The law needs the disclosure first, so the opening comes from code: the model skipped it when the customer spoke first
        if not self.on_voicemail and not has_disclosed(chat_ctx):
            return opening_line(self.lead)
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
        reason: Literal["callback", "renewal"] = "callback",
        contract_end: str = "",
        visit_address: str = "",
        contact_name: str = "",
        attendees: str = "",
        email: str = "",
        products: str = "",
    ) -> str:
        """Save the result of this call for the sales team.

        Args:
            outcome: interested (a team member calls them), callback (we ring back), not_interested, do_not_call,
                field_visit (they asked for a visit), or email_request (they asked for details by email).
            interested_services: Services the customer showed interest in, comma separated.
            callback_time: When and how the customer wants to be contacted, or when suits for the visit, if they said.
            callback_at: The callback_time moment in ISO 8601 with the Irish UTC offset, if they gave a time.
            notes: One-sentence summary of the conversation and any objections.
            reason: renewal for a callback before their contract renews; otherwise callback.
            contract_end: For a renewal: when their contract ends as YYYY-MM-DD, or "unknown" if they don't know.
            visit_address: For a field visit: the address to visit.
            contact_name: For a field visit: who to meet.
            attendees: For a field visit: anyone else who'll be there, if they said.
            email: For an email request: their business email address.
            products: For an email request: the services they want details on.
        """
        callback_at = in_sales_hours(callback_at)
        when = db.parse_time(callback_at, datetime.now(IRELAND)) if callback_time else None
        # Each refusal tells the agent what to ask, instead of saving a follow-up the team can't act on
        if outcome == "email_request":
            if not EMAIL.fullmatch(email):
                return "Not saved. Ask for their business email address and read it back, then call record_outcome again with email."
            if not products:
                return "Not saved. Ask which services they'd like details on, then call record_outcome again with products."
        elif outcome in ("interested", "callback", "field_visit") and not when:
            who = "suits for the visit" if outcome == "field_visit" else "a team member should call"
            return f"Not saved. Ask when {who}, then call record_outcome again with callback_time and a future callback_at."
        if outcome == "field_visit" and not (visit_address and contact_name):
            return ("Not saved. Ask for the address to visit and the name of the person to meet, "
                    "then call record_outcome again with visit_address and contact_name.")
        if reason == "renewal" and outcome in ("interested", "callback"):
            try:
                ends = None if contract_end == "unknown" else date.fromisoformat(contract_end)
            except ValueError:
                return ('Not saved. Ask when their contract ends, then call record_outcome again with contract_end as YYYY-MM-DD, '
                        'or "unknown" if they still don\'t know.')
            if ends and when.date() >= ends:
                return f"Not saved. The callback must be before their contract ends on {contract_end}: ask for a time before then."
        follow_up = None
        if outcome in FOLLOW_UP_REASON:
            details = {"reason": reason if FOLLOW_UP_REASON[outcome] == "callback" else FOLLOW_UP_REASON[outcome],
                       "contract_end": contract_end, "visit_address": visit_address, "contact_name": contact_name,
                       "attendees": attendees, "email": email, "products": products}
            follow_up = {k: v for k, v in details.items() if v}
        if self.transfer_failed:
            follow_up = {**(follow_up or {}), "transfer": "failed"}
        await self.save(
            outcome, interested_services=interested_services, callback_time=callback_time, callback_at=callback_at,
            notes=notes, follow_up=follow_up,
        )
        return "Saved. Now call end_call."

    @function_tool()
    async def transfer_to_specialist(self, context: RunContext, brief: str) -> str:
        """Put the customer through to a human sales specialist now. Only call this after the customer has clearly agreed to be transferred now.

        Args:
            brief: For the specialist: the business, current provider, what they pay, their main pain point and what they want.
        """
        if not (SALES_SPECIALIST_PHONE and self.phone_mode):
            return "Transfer isn't available. Offer a callback with a team member instead."
        self.transferring = True
        try:
            await WarmTransferTask(
                SALES_SPECIALIST_PHONE, sip_trunk_id=os.getenv("SIP_OUTBOUND_TRUNK_ID"), chat_ctx=self.chat_ctx,
                ringing_timeout=25, extra_instructions=brief,
            )
        except ToolError as e:  # no answer, declined, voicemail, hang-up or failed dial
            logger.info("transfer failed: %s", e)
            self.transferring = False
            self.transfer_failed = True
            return "The specialist isn't available. Apologise and offer a callback."
        await self.save("transferred", notes=brief, follow_up={"transfer": "done"})
        # LiveKit turns our audio back on after the hand-over; leaving the job ends the AI but not the call
        get_job_context().shutdown(reason="transferred")
        return ""

    async def save(self, outcome: str, **fields) -> None:
        if self.recorded:
            return
        self.recorded = True
        self.outcome = outcome
        # A database write can take a second; keep it off the audio loop
        await asyncio.to_thread(record_result, {**self.lead, "outcome": outcome, **fields})
        logger.info("recorded outcome %s for %s", outcome, self.lead.get("phone"))

    async def finish(self, lines: list[tuple[str, str, int]], latency_ms: int | None = None) -> None:
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
            # expressive and as fast as flash in our test (0.17s to first audio).
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
    # a silent pick-up would otherwise hear ten seconds of nothing before the agent speaks
    return AMD(session, participant_identity=identity, stt=session.stt, detection_options={"no_speech_threshold": 4.0})


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# LiveKit's 10 s default can cut off the hang-up work: the analysis alone can take 30 s
server = AgentServer(shutdown_process_timeout=60)
server.setup_fnc = prewarm


@server.rtc_session(agent_name="sales-agent")
async def sales_call(ctx: agents.JobContext):
    lead = json.loads(ctx.job.metadata) if ctx.job.metadata else TEST_LEAD
    # room links the row to the session in the LiveKit dashboard; job_id keys it (console room names repeat)
    lead = {**lead, "room": ctx.room.name, "job_id": ctx.job.id}
    trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
    phone_mode = bool(trunk_id and lead.get("phone"))
    agent = SalesAgent(lead, phone_mode)
    agent.reached = not phone_mode  # a console or web tester is on the line from the start

    async def on_shutdown():
        await agent.finish(transcript_lines(session.history.items, agent.started_at), analyse.avg_latency_ms(session.history.items))

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
        if (phone_mode and not agent.reached) or agent.transferring:
            return  # still ringing, or on hold for the specialist: silence must not hang up
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
        agent.on_voicemail = True
        handle = session.generate_reply(
            instructions=f"You reached voicemail. Leave a brief message: you're an AI assistant calling on behalf of {BUSINESS_NAME}, "
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
