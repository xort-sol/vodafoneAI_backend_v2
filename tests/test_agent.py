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


def test_tests_never_write_to_the_real_sheet():
    assert not os.getenv("RESULTS_WEBHOOK_URL")


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
    assert agent_module.now_line(now) == "Right now it's Friday 30 October 2026, 2:05 PM in Ireland."
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
    assert agent_module.transcript_lines(history.items) == [
        ("agent", "[warmly] Hi, I'm an AI assistant."),
        ("customer", "Hello?"),
    ]


@pytest.mark.asyncio
async def test_finished_call_has_outcome_and_transcript(call_rows, test_db):
    agent = SalesAgent({**TEST_LEAD, "job_id": "j1"})
    agent.reached = True
    await agent.finish([("agent", "Hi, I'm an AI assistant."), ("customer", "Hello?")])
    assert [r["outcome"] for r in call_rows()] == ["hung_up"]
    with psycopg.connect(test_db) as conn:
        assert conn.execute("select count(*) from transcripts").fetchone()[0] == 2



@pytest.mark.asyncio
async def test_finished_call_is_timed_analysed_and_sent_to_the_inbox(call_rows, test_db, monkeypatch):
    async def fake_llm(prompt):
        return json.dumps({"score": 82, "reason": "r", "objections": [], "sentiment": "positive",
                           "next_step": "n", "brief": "b"})

    monkeypatch.setattr(analyse, "ask_llm", fake_llm)
    agent = SalesAgent({**TEST_LEAD, "job_id": "j1"})
    agent.reached = True
    await agent.save("interested", callback_time="Thu 10am")
    await agent.finish([("agent", "Hi"), ("customer", "Yes, ring me Thursday")], latency_ms=900)
    row = call_rows()[0]
    assert row["avg_latency_ms"] == 900 and row["duration_s"] is not None
    with psycopg.connect(test_db) as conn:
        assert conn.execute("select score from analyses").fetchall() == [(82,)]
        assert conn.execute("select count(*) from tasks").fetchone()[0] == 1

@pytest.mark.asyncio
async def test_unsaved_call_gets_no_transcript(call_rows, test_db):
    await SalesAgent({**TEST_LEAD, "job_id": "j1"}).finish([("agent", "Hi")])  # never reached: setup error
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
    assert session.tts.model == "eleven_v4_turbo"  # reads [warmly]-style tags, so delivery follows the conversation



@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_interested_lead_is_recorded(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Hi, yes I have a couple of minutes.")
        await session.run(user_input="We have 8 staff and our office internet is really slow.")
        result = await session.run(
            user_input="That sounds great, I'm interested in the business internet. Have someone call me tomorrow at 10am."
        )
        result.expect.contains_function_call(name="record_outcome", arguments={"outcome": "interested"})
    row = call_rows()[0]
    assert row["outcome"] == "interested"
    # "tomorrow" alone is useless once the sheet is read a day later
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
        result = await session.run(user_input="Honestly that sounds too expensive for us.")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm,
            intent="Acknowledges the price concern and offers at most one short reframe, in one or two spoken sentences, "
            "without listing several services or pressuring the customer.",
        )


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_pitch_is_short_enough_to_follow_on_the_phone(call_rows):
    async with (
        inference.LLM(model=agent_module.LLM_MODEL) as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yeah, I have a minute.")
        await session.run(user_input="It's fine, though our internet could be better.")
        result = await session.run(user_input="Sure, tell me more.")
        reply = result.expect.next_event().is_message(role="assistant").event().item.text_content
        spoken = re.sub(r"\[[^\]]*\]", "", reply)  # delivery tags are voice directions, not words the caller hears
        assert len(spoken.split()) <= 30, reply


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
    await agent.record_outcome(None, outcome=outcome, callback_time="tomorrow 10am")
    assert [r["callback_time"] for r in call_rows()] == ["tomorrow 10am"]


@pytest.mark.asyncio
async def test_not_interested_needs_no_time(call_rows):
    await SalesAgent(JOB_LEAD).record_outcome(None, outcome="not_interested")
    assert [r["outcome"] for r in call_rows()] == ["not_interested"]


@LIVE
@pytest.mark.asyncio
async def test_okay_after_pitch_leads_to_asking_for_callback_time(call_rows):
    # The customer's lines from the 28 Oct console call where the lead was saved with no callback time
    lines = ["Yeah. Sure. Go ahead.", "Yeah. That's going good, but not up to the mark here.",
             "Maybe low speed. And sometime it it... It's not working.", "Yeah. Obviously.", "Yeah. Go ahead.", "Okay."]
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.generate_reply(instructions="Open the call.")
        for line in lines:
            result = await session.run(user_input=line)
        assert all(r["callback_time"] for r in call_rows())
        await result.expect[-1].is_message(role="assistant").judge(
            llm, intent="Asks when, or how, a team member should follow up with them. Does not say goodbye."
        )


@LIVE
@pytest.mark.asyncio
async def test_frustrated_customer_gets_an_empathetic_delivery_tag(call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent(JOB_LEAD))
        await session.run(user_input="Yes, go ahead.")
        result = await session.run(user_input="Honestly it's been a nightmare, our internet drops every single afternoon.")
        reply = result.expect.next_event().is_message(role="assistant").event().item.text_content
    # the voice only sounds sympathetic if the line carries a tag such as [empathetic]
    assert reply.lstrip().startswith("["), reply
    assert not reply.lstrip().startswith(("[excited]", "[cheerfully]")), reply  # match their mood


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
    # LiveKit's default is 10 s; hang-up can spend 15 s on the Google Sheet and 30 s on the analysis
    assert agent_module.server._shutdown_process_timeout >= 60
