import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
import pytest
from livekit.agents import AgentSession, inference
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, FunctionCall

from agent import main as agent_module
from analysis import analyse
from agent.main import TEST_LEAD, SalesAgent, SilenceTracker, build_session

# Saved calls need a job id; on LiveKit it is the job's id
JOB_LEAD = {**TEST_LEAD, "job_id": "test-job"}


def test_agent_starts_as_a_module():
    # the Dockerfile CMD runs exactly this; a broken package layout only shows up after deploy otherwise
    out = subprocess.run([sys.executable, "-m", "agent.main", "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


@pytest.mark.asyncio
async def test_outcome_saved_once(call_rows):
    agent = SalesAgent(JOB_LEAD)
    await agent.save("interested")
    await agent.save("hung_up")  # shutdown fallback must not overwrite the real outcome
    assert [r["outcome"] for r in call_rows()] == ["interested"]


def test_test_call_prompt_has_nothing_the_agent_could_read_out_as_customer_info():
    # a 29 Oct console call: the agent asked "how's your web or console testing going?"
    assert "console" not in SalesAgent(JOB_LEAD).instructions.lower()


def test_prompt_has_the_date_and_time_in_ireland():
    now = datetime(2026, 10, 30, 14, 5, tzinfo=agent_module.IRELAND)
    assert agent_module.now_line(now) == "Right now it's Friday 30 October 2026, 2:05 PM in Ireland. UTC offset: +00:00"
    assert datetime.now(agent_module.IRELAND).strftime("%A %-d %B %Y") in SalesAgent(JOB_LEAD).instructions


def test_lead_context_reaches_the_prompt():
    lead = {"name": "Jane", "company": "Acme", "staff": "8", "current_plan": "Eir fibre 300",
            "contract_end": "March 2027", "notes": "asked about internet"}
    facts = agent_module.lead_facts(lead)
    for value in ["8 staff", "Eir fibre 300", "March 2027", "asked about internet"]:
        assert value in facts
    assert agent_module.lead_facts(TEST_LEAD) == "What we know: nothing yet."


def test_prompt_uses_the_objection_answers_and_no_offer_we_dont_have():
    instructions = SalesAgent(JOB_LEAD).instructions
    assert Path(agent_module.__file__).with_name("objections.md").read_text() in instructions
    assert "money-back" not in instructions  # not in services.md, so the agent would be inventing it


async def _chunks(*texts):
    for text in texts:
        yield ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content=text))


@pytest.mark.asyncio
async def test_reply_stops_after_its_first_question():
    # gpt-4.1-mini sometimes asks two ("How often does it drop? Does it get slower first?"); callers answer one
    two = _chunks("[curious] How often does it drop", "? Does it get", " slower first?")
    assert "".join([c.delta.content async for c in agent_module.first_question_only(two)]) == "[curious] How often does it drop?"
    closed = []

    async def stream():
        try:
            yield ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content="Okay?"))
            yield ChatChunk(id="c", delta=ChoiceDelta(role="assistant", content=" More?"))
        finally:
            closed.append(True)

    _ = [c async for c in agent_module.first_question_only(stream())]
    assert closed  # an open LLM stream keeps generating, and billing, text nobody hears
    statement = _chunks("[warmly] Thanks, ", "goodbye!")
    assert "".join([c.delta.content async for c in agent_module.first_question_only(statement)]) == "[warmly] Thanks, goodbye!"


def test_transcript_keeps_only_spoken_turns():
    history = ChatContext()
    history.add_message(role="system", content="You are Sam.")
    history.add_message(role="assistant", content="[warmly] Hi, I'm an AI assistant.")
    history.add_message(role="user", content="Hello?")
    history.items.append(FunctionCall(call_id="1", name="record_outcome", arguments="{}"))
    started = history.items[1].created_at - 12.4
    lines = agent_module.transcript_lines(history.items, started)
    assert [(r, text) for r, text, _ in lines] == [("agent", "[warmly] Hi, I'm an AI assistant."), ("customer", "Hello?")]


def test_transcript_lines_carry_seconds_from_the_start():
    history = ChatContext()
    history.add_message(role="assistant", content="Hi")
    started = history.items[0].created_at - 12.4
    assert agent_module.transcript_lines(history.items, started) == [("agent", "Hi", 12)]


def test_agent_introduces_itself_on_behalf_of_the_business(monkeypatch):
    monkeypatch.setattr(agent_module, "BUSINESS_NAME", "Vodafone")
    assert "AI assistant calling on behalf of Vodafone" in SalesAgent(TEST_LEAD).instructions


def test_prompt_gives_the_summer_utc_offset():
    now = datetime(2027, 7, 1, 10, 0, tzinfo=agent_module.IRELAND)
    assert "+01:00" in agent_module.now_line(now)


@pytest.mark.asyncio
async def test_finished_call_has_outcome_and_transcript(call_rows, test_db):
    agent = SalesAgent({**TEST_LEAD, "job_id": "j1"})
    agent.reached = True
    await agent.finish([("agent", "Hi, I'm an AI assistant.", 0), ("customer", "Hello?", 4)])
    assert [r["outcome"] for r in call_rows()] == ["hung_up"]
    with psycopg.connect(test_db) as conn:
        assert conn.execute("select count(*) from transcripts").fetchone()[0] == 2



@pytest.mark.asyncio
async def test_finished_call_is_timed_analysed_and_sent_to_the_inbox(call_rows, test_db, monkeypatch):
    async def fake_llm(prompt):
        return json.dumps({"discovery": {}, "reason": "r", "objections": [], "sentiment": "positive",
                           "next_step": "n", "brief": "b", "questions": [], "unanswered": []})

    monkeypatch.setattr(analyse, "ask_llm", fake_llm)
    agent = SalesAgent({**TEST_LEAD, "job_id": "j1"})
    agent.reached = True
    await agent.save("interested", callback_time="Thu 10am")
    await agent.finish([("agent", "Hi", 0), ("customer", "Yes, ring me Thursday", 3)], latency_ms=900)
    row = call_rows()[0]
    assert row["avg_latency_ms"] == 900 and row["duration_s"] is not None
    with psycopg.connect(test_db) as conn:
        assert conn.execute("select score from analyses").fetchall() == [(10,)]  # nothing captured: only the agreed next step scores
        assert conn.execute("select count(*) from tasks").fetchone()[0] == 1

@pytest.mark.asyncio
async def test_unsaved_call_gets_no_transcript(call_rows, test_db):
    await SalesAgent({**TEST_LEAD, "job_id": "j1"}).finish([("agent", "Hi", 0)])  # never reached: setup error
    assert call_rows() == []


def test_silence_asks_once_then_ends():
    tracker = SilenceTracker()
    assert tracker.on_away() == "ask"
    assert tracker.on_away() == "end"


def test_silence_counter_resets_when_customer_speaks():
    tracker = SilenceTracker()
    tracker.on_away()
    tracker.on_speaking()
    assert tracker.on_away() == "ask"


@pytest.mark.asyncio
async def test_session_uses_vad_adaptive_interruptions_and_elevenlabs(monkeypatch):
    from livekit.plugins import elevenlabs, silero

    monkeypatch.setenv("ELEVEN_API_KEY", "test-key")
    session = build_session(vad=silero.VAD.load())
    assert session.vad is not None  # adaptive interruption needs a VAD
    assert isinstance(session.tts, elevenlabs.TTS)
    assert session.interruption_detection == "adaptive"
    assert session.options.user_away_timeout == 8.0
    # short answers ("Yeah.") made the agent wait the full max delay before replying
    assert session.options.endpointing["mode"] == "dynamic"
    assert session.options.endpointing["max_delay"] == 1.5
    assert session.options.preemptive_generation["preemptive_tts"] is True  # starts speaking sooner
    assert session.tts.model == "eleven_v4_turbo"



@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_interested_lead_is_recorded(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Hi, yes I have a few minutes.")
        await session.run(user_input="We're a dental practice, 8 staff, I make the decisions. Our office internet is really slow.")
        await session.run(user_input="Yes, that's right. I'd like a team member to call me about the internet.")
        result = await session.run(user_input="Tomorrow at 10am suits.")
        result.expect.contains_function_call(name="record_outcome", arguments={"outcome": "interested"})
    row = call_rows()[0]
    assert row["outcome"] == "interested"
    # "tomorrow" alone is useless once the call record is read a day later
    tomorrow = datetime.now(agent_module.IRELAND).date() + timedelta(days=1)
    assert f"{tomorrow.day} {tomorrow:%B}" in row["callback_time"], row["callback_time"]


@pytest.mark.asyncio
async def test_hang_up_recorded_only_after_customer_reached(call_rows):
    agent = SalesAgent(JOB_LEAD)
    await agent.record_hang_up()  # crash before dialing, e.g. bad trunk ID
    assert call_rows() == []
    agent.reached = True
    await agent.record_hang_up()
    assert [r["outcome"] for r in call_rows()] == ["hung_up"]


@pytest.mark.parametrize(
    "category,transcript,expected",
    [("human", "Hello?", False), ("uncertain", "Hello, who is this?", False), ("uncertain", "", True)],
)
def test_agent_opens_only_when_callee_is_silent(category, transcript, expected):
    # when the callee spoke, the session already replies to that turn
    assert agent_module.should_open_call(category, transcript) is expected


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_price_objection_is_acknowledged_without_pushing(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Sure, I have a minute.")
        await session.run(user_input="Our office internet keeps dropping.")
        result = await session.run(user_input="Honestly Vodafone is too expensive for us.")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm,
            intent="Acknowledges the price concern in one or two spoken sentences and asks one question, such as what they pay now. "
            "Promises no savings, lists no services and doesn't pressure the customer.",
        )


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_discovery_asks_before_selling(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yeah, I have a few minutes.")
        result = await session.run(user_input="It's fine, though our internet could be better.")
        event = result.expect.next_event().is_message(role="assistant")
        spoken = re.sub(r"\[[^\]]*\]", "", event.event().item.text_content)  # delivery tags are voice directions
        assert len(spoken.split()) <= 25, spoken
        await event.judge(llm, intent="Asks one question about their internet. Names no plan, speed or price.")


LIVE = pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")


def assistant_texts(*runs) -> list[str]:
    return [e.item.text_content for r in runs for e in r.events if e.type == "message" and e.item.role == "assistant"]


@LIVE
@pytest.mark.asyncio
async def test_greeting_alone_is_not_taken_as_permission(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.generate_reply(instructions="Open the call.")
        result = await session.run(user_input="Hi.")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm,
            intent="Its main point is checking again whether now is an okay time to talk. It does not ask how their business or services are going, and names no service or price.",
        )


@LIVE
@pytest.mark.asyncio
async def test_hinted_problem_gets_a_follow_up_before_any_pitch(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yeah, I have a minute.")
        await session.run(user_input="It's doing good, I guess.")
        result = await session.run(user_input="Not really, it's a bit of a hassle to be honest.")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm,
            intent="Asks what is going wrong or what makes it hard. Does not name a service, plan or price yet.",
        )


# The customer's lines from a real console call (28 Oct) that exposed these problems
CALL_LINES = ["Hi.", "That's yeah. It's doing good. Yeah.", "Not really.", "Sorry?", "Yeah.", "Maybe tomorrow?"]


@LIVE
@pytest.mark.asyncio
async def test_sorry_repeats_the_last_question(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yeah, I have a minute.")
        await session.run(user_input="Managing our team's phone lines is a pain, the bills are all over the place.")
        asked = assistant_texts(await session.run(user_input="Yeah, five of us, everyone has their own plan."))[-1]
        result = await session.run(user_input="Sorry?")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm,
            intent="Repeats the point of its previous message in fewer, simpler words, adds no new details, "
            f"and asks the same question again. Previous message was: {asked!r}",
        )


@LIVE
@pytest.mark.asyncio
async def test_real_call_never_hangs_up_on_an_open_question(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.generate_reply(instructions="Open the call.")
        for line in CALL_LINES:
            result = await session.run(user_input=line)
            texts = assistant_texts(result)
            assert all(t.count("?") <= 1 for t in texts), texts
            ended = any(e.type == "function_call" and e.item.name == "end_call" for e in result.events)
            assert not (ended and any("?" in t for t in texts)), f"hung up right after asking: {texts}"
            if ended:
                break


@pytest.mark.parametrize("outcome", ["interested", "callback"])
@pytest.mark.asyncio
async def test_follow_up_outcome_without_time_is_refused(call_rows, outcome):
    agent = SalesAgent(JOB_LEAD)
    reply = await agent.record_outcome(None, outcome=outcome)
    assert call_rows() == []  # nothing saved, so the agent can still ask and save later
    assert "when" in reply.lower()
    await agent.record_outcome(None, outcome=outcome, callback_time="tomorrow 10am", callback_at="2037-11-03T10:00:00+00:00")
    assert [r["callback_time"] for r in call_rows()] == ["tomorrow 10am"]


@pytest.mark.parametrize("callback_at", ["", "now", "2020-01-01T10:00:00+00:00"])
@pytest.mark.asyncio
async def test_follow_up_needs_a_real_future_time(call_rows, callback_at):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, outcome="interested", callback_time="now", callback_at=callback_at)
    assert call_rows() == []
    assert "Not saved" in reply


@pytest.mark.parametrize("given, saved_hour", [("03:00", 15), ("07:30", 19), ("08:00", 8), ("15:00", 15)])
@pytest.mark.asyncio
async def test_callback_before_8am_means_the_afternoon(call_rows, given, saved_hour):
    await SalesAgent(JOB_LEAD).record_outcome(
        None, outcome="callback", callback_time="at 3", callback_at=f"2037-11-03T{given}:00+00:00"
    )
    assert call_rows()[0]["callback_at"].astimezone(timezone.utc).hour == saved_hour


def test_opening_discloses_the_ai_and_uses_only_the_first_name(monkeypatch):
    monkeypatch.setattr(agent_module, "BUSINESS_NAME", "Vodafone")
    assert agent_module.opening_line({"name": "Ciaran Walsh"}).startswith(
        "Hi Ciaran, I'm an AI assistant calling on behalf of Vodafone."
    )
    assert agent_module.opening_line({"name": ""}).startswith("Hi, I'm an AI assistant")


def spoken(*lines: str) -> ChatContext:
    ctx = ChatContext.empty()
    for line in lines:
        ctx.add_message(role="assistant", content=line)
    return ctx


def test_first_reply_is_the_scripted_opening_even_if_the_customer_spoke_first():
    agent = SalesAgent(JOB_LEAD)
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="Hello?")
    assert agent.llm_node(ctx, [], None) == agent_module.opening_line(JOB_LEAD)


def test_opening_is_repeated_if_it_was_cut_off_before_the_disclosure():
    agent = SalesAgent(JOB_LEAD)
    assert agent.llm_node(spoken("Hi Ciaran, I'm"), [], None) == agent_module.opening_line(JOB_LEAD)


def test_after_the_disclosure_the_model_takes_over():
    agent = SalesAgent(JOB_LEAD)
    assert not isinstance(agent.llm_node(spoken(agent_module.opening_line(JOB_LEAD)), [], None), str)


def test_voicemail_gets_its_own_message_not_the_opening():
    agent = SalesAgent(JOB_LEAD)
    agent.on_voicemail = True
    assert not isinstance(agent.llm_node(ChatContext.empty(), [], None), str)


def test_ending_the_call_says_the_one_goodbye_with_the_agreed_time():
    tools = [t for t in SalesAgent(JOB_LEAD).tools if isinstance(t, agent_module.EndCallTool)]
    assert tools and "goodbye" in tools[0]._end_instructions and "callback time" in tools[0]._end_instructions
    assert "say nothing yourself" in SalesAgent(JOB_LEAD).instructions


def test_offer_prices_are_quoted_with_what_they_rise_to():
    for quote in ("forty euro a month for the first twelve months, then sixty euro",
                  "fifty euro a month for the first twelve months, then seventy euro",
                  "sixty-five euro a month for the first twelve months, then eighty-five euro"):
        assert quote in agent_module.SERVICES


@pytest.mark.asyncio
async def test_not_interested_needs_no_time(call_rows):
    await SalesAgent(JOB_LEAD).record_outcome(None, outcome="not_interested")
    assert [r["outcome"] for r in call_rows()] == ["not_interested"]


@LIVE
@pytest.mark.asyncio
async def test_agreeing_to_a_call_leads_to_asking_when(call_rows):
    # A "yes" to a follow-up is not a time: the 28 Oct console call saved a lead with no callback time
    lines = ["Yeah. Sure. Go ahead.", "A café, four of us, I decide. Eir broadband, it drops most afternoons.",
             "Yes, that's right.", "Yeah, a call would be good."]
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.generate_reply(instructions="Open the call.")
        for line in lines:
            result = await session.run(user_input=line)
        assert all(r["callback_time"] for r in call_rows())
        await result.expect[-1].is_message(role="assistant").judge(
            llm, intent="Asks when a team member should call, such as whether tomorrow or the next day suits. Does not say goodbye."
        )


@LIVE
@pytest.mark.asyncio
async def test_replies_carry_no_delivery_tags(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yes, go ahead.")
        result = await session.run(user_input="Honestly it's been a nightmare, our internet drops every single afternoon.")
        reply = result.expect.next_event().is_message(role="assistant").event().item.text_content
    assert "[" not in reply, reply  # everything the model writes is spoken


@pytest.mark.asyncio
async def test_callback_at_is_saved_for_the_dialler(call_rows):
    await SalesAgent(JOB_LEAD).record_outcome(
        None, outcome="callback", callback_time="tomorrow 3pm", callback_at="2037-11-03T15:00:00+00:00"
    )
    assert call_rows()[0]["callback_at"] == datetime(2037, 11, 3, 15, tzinfo=timezone.utc)


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_bad_time_books_an_exact_redial_time(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Hello?")
        await session.run(user_input="Ah no, I'm driving. Ring me back tomorrow at 3.")
    row = call_rows()[0]
    assert row["outcome"] == "callback"
    tomorrow = datetime.now(agent_module.IRELAND).date() + timedelta(days=1)
    local = row["callback_at"].astimezone(agent_module.IRELAND)
    assert (local.date(), local.hour) == (tomorrow, 15), row["callback_at"]


@pytest.mark.asyncio
async def test_voicemail_detector_uses_the_sessions_eu_speech_model():
    # AMD's own default, cartesia/ink-whisper, isn't served in LiveKit's EU region: every call skipped voicemail detection
    session = agent_module.build_session(vad=None)
    assert agent_module.build_amd(session, "sip-+1")._stt.model == "deepgram/nova-3"


def test_job_gets_time_to_save_and_analyse_after_hang_up():
    # LiveKit's default is 10 s; hang-up can spend 30 s on the analysis
    assert agent_module.server._shutdown_process_timeout >= 60


def test_opening_offers_a_free_connectivity_review(monkeypatch):
    monkeypatch.setattr(agent_module, "BUSINESS_NAME", "Vodafone")
    assert agent_module.opening_line({"name": "Ciaran Walsh"}) == (
        "Hi Ciaran, I'm an AI assistant calling on behalf of Vodafone. "
        "We're offering local businesses a free connectivity review. Have you got a few minutes?"
    )


FUTURE = "2037-11-03T10:00:00+00:00"
VISIT = {"outcome": "field_visit", "callback_time": "Tuesday 3 November at 10am", "callback_at": FUTURE,
         "visit_address": "Main Street, Cork", "contact_name": "Mary Byrne"}


@pytest.mark.parametrize("missing, asks", [("visit_address", "address"), ("contact_name", "name"), ("callback_at", "when")])
@pytest.mark.asyncio
async def test_field_visit_needs_an_address_a_contact_and_a_future_time(call_rows, missing, asks):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, **{**VISIT, missing: ""})
    assert call_rows() == []
    assert "Not saved" in reply and asks in reply.lower()


@pytest.mark.asyncio
async def test_field_visit_in_the_past_is_refused(call_rows):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, **{**VISIT, "callback_at": "2020-01-01T10:00:00+00:00"})
    assert call_rows() == [] and "Not saved" in reply


@pytest.mark.asyncio
async def test_field_visit_is_saved_as_a_request(call_rows):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, **VISIT, attendees="Mary and her partner")
    row = call_rows()[0]
    assert row["outcome"] == "field_visit"
    assert row["follow_up"] == {"reason": "field_visit", "visit_address": "Main Street, Cork",
                                "contact_name": "Mary Byrne", "attendees": "Mary and her partner"}
    assert "Saved" in reply


@pytest.mark.parametrize("email", ["", "mary", "mary@byrne", "mary at byrne dot ie", "mary@@byrne.ie"])
@pytest.mark.asyncio
async def test_email_request_needs_a_valid_email(call_rows, email):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, outcome="email_request", email=email, products="broadband")
    assert call_rows() == []
    assert "Not saved" in reply and "email" in reply.lower()


@pytest.mark.asyncio
async def test_email_request_needs_the_products(call_rows):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, outcome="email_request", email="mary@byrne.ie")
    assert call_rows() == [] and "services" in reply.lower()
    await SalesAgent(JOB_LEAD).record_outcome(None, outcome="email_request", email="mary@byrne.ie", products="broadband, mobiles")
    row = call_rows()[0]
    assert row["outcome"] == "email_request"
    assert row["follow_up"] == {"reason": "email", "email": "mary@byrne.ie", "products": "broadband, mobiles"}


RENEWAL = {"outcome": "callback", "reason": "renewal", "callback_time": "Tuesday 3 November at 10am", "callback_at": FUTURE}


@pytest.mark.asyncio
async def test_renewal_callback_after_the_contract_ends_is_refused(call_rows):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, **RENEWAL, contract_end="2037-10-01")
    assert call_rows() == []
    assert "Not saved" in reply and "before" in reply


@pytest.mark.parametrize("contract_end", ["2037-12-01", "unknown"])
@pytest.mark.asyncio
async def test_renewal_callback_before_the_end_or_with_an_unknown_end_is_saved(call_rows, contract_end):
    await SalesAgent(JOB_LEAD).record_outcome(None, **RENEWAL, contract_end=contract_end)
    row = call_rows()[0]
    assert row["outcome"] == "callback"
    assert row["follow_up"] == {"reason": "renewal", "contract_end": contract_end}


@pytest.mark.parametrize("contract_end", ["", "March 2037"])
@pytest.mark.asyncio
async def test_renewal_callback_asks_for_a_readable_contract_end(call_rows, contract_end):
    reply = await SalesAgent(JOB_LEAD).record_outcome(None, **RENEWAL, contract_end=contract_end)
    assert call_rows() == []
    assert "contract" in reply and "unknown" in reply


@pytest.mark.asyncio
async def test_agreed_callback_saves_its_reason_and_a_no_saves_none(call_rows):
    await SalesAgent({**JOB_LEAD, "job_id": "j1"}).record_outcome(
        None, outcome="interested", callback_time="Tuesday 3 November at 10am", callback_at=FUTURE)
    await SalesAgent({**JOB_LEAD, "job_id": "j2"}).record_outcome(None, outcome="not_interested")
    assert [r["follow_up"] for r in call_rows()] == [{"reason": "callback"}, None]


def test_prompt_covers_the_discovery_flow_and_never_claims_a_booking():
    instructions = SalesAgent(JOB_LEAD).instructions
    for phrase in ("Have I got that right?",
                   "who'll confirm the time", "a team member will email you", "transfer_to_specialist",
                   "Would tomorrow or the next day suit?", "Morning or afternoon?"):
        assert phrase in instructions, phrase
