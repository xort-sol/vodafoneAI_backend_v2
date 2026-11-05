import json
import os
from types import SimpleNamespace

import psycopg
import pytest

from dotenv import load_dotenv

from analysis.analyse import analyse_call, avg_latency_ms, build_prompt, is_hot, parse_analysis
from shared import db

load_dotenv()  # LiveKit credentials for the real-model test; conftest still pins DATABASE_URL to the test database

GOOD = {
    "score": 82,
    "reason": "Asked about price and booked a call.",
    "objections": ["price"],
    "sentiment": "positive",
    "next_step": "Call Thursday 10am about Business Broadband 500.",
    "brief": "Aoife, 8 staff, slow office internet; liked the 500 plan at forty euro; worried about price.",
}

LINES = [
    ("agent", "[warmly] Hi Aoife, I'm an AI assistant from Acme. Have you a quick minute?"),
    ("customer", "Yes, go on."),
    ("agent", "How's the office internet these days?"),
    ("customer", "Slow, honestly. Eight of us on it."),
    ("agent", "Business Broadband 500 starts at forty euro. Want to hear more?"),
    ("customer", "Bit pricey, but have someone ring me Thursday at 10."),
]


def test_prompt_carries_the_transcript_and_outcome_without_delivery_tags():
    prompt = build_prompt(LINES, "interested")
    assert "Customer: Slow, honestly." in prompt and "interested" in prompt
    assert "[warmly]" not in prompt


def test_parse_accepts_json_inside_chatter_or_a_code_fence():
    assert parse_analysis(json.dumps(GOOD)) == GOOD
    assert parse_analysis(f"Sure! Here it is:\n```json\n{json.dumps(GOOD)}\n```") == GOOD


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({**GOOD, "score": 140}),
        json.dumps({**GOOD, "score": "high"}),
        json.dumps({**GOOD, "sentiment": "ecstatic"}),
        json.dumps({**GOOD, "objections": "price"}),
        json.dumps({k: v for k, v in GOOD.items() if k != "brief"}),
    ],
)
def test_parse_rejects_anything_off_schema(bad):
    assert parse_analysis(bad) is None


def test_parse_trims_an_overlong_brief():
    assert len(parse_analysis(json.dumps({**GOOD, "brief": "x" * 2000}))["brief"]) == 600


@pytest.mark.parametrize(
    "outcome, score, hot",
    [("interested", 10, True), ("interested", None, True), ("hung_up", 75, True), ("hung_up", 69, False),
     ("hung_up", None, False), ("callback", 95, False), ("not_interested", 90, False), ("do_not_call", 100, False)],
)
def test_hot_lead_rule(outcome, score, hot):
    # a no or a do-not-call is never chased, whatever the model scored; a callback is re-dialled by the AI, not a person
    assert is_hot(outcome, score) is hot


def msg(role, **metrics):
    return SimpleNamespace(type="message", role=role, metrics=metrics)


def test_latency_is_end_of_turn_plus_first_token_plus_first_audio():
    items = [
        msg("assistant", llm_node_ttft=0.5, tts_node_ttfb=0.2),  # the opening line: no customer turn before it
        msg("user", end_of_turn_delay=0.3),
        msg("assistant", llm_node_ttft=0.4, tts_node_ttfb=0.3),
        msg("user", end_of_turn_delay=0.5),
        msg("assistant", llm_node_ttft=0.6, tts_node_ttfb=0.1),
    ]
    assert avg_latency_ms(items) == 1100  # (0.3+0.4+0.3 + 0.5+0.6+0.1) / 2 turns


def test_latency_is_none_without_answered_turns():
    assert avg_latency_ms([msg("assistant", llm_node_ttft=0.5)]) is None


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    async def __call__(self, prompt):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def saved(test_db):
    with psycopg.connect(test_db) as c:
        analyses = c.execute("select score, brief, model from analyses").fetchall()
        tasks = c.execute("select kind, due_at is not null from tasks").fetchall()
    return analyses, tasks


@pytest.mark.asyncio
async def test_hot_call_gets_an_analysis_and_one_inbox_task(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am",
                  "callback_at": "2026-11-05T10:00:00+00:00"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))  # retried: no duplicate
    analyses, tasks = saved(test_db)
    assert [(s, m) for s, _, m in analyses] == [(82, "openai/gpt-4.1")]
    assert tasks == [("follow_up", True)]


@pytest.mark.asyncio
async def test_interested_lead_reaches_the_inbox_even_when_analysis_fails(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(TimeoutError()))
    analyses, tasks = saved(test_db)
    assert analyses == [] and len(tasks) == 1


@pytest.mark.asyncio
async def test_cold_call_is_analysed_but_not_sent_to_the_inbox(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "not_interested"})
    await analyse_call("j1", LINES, "not_interested", ask=FakeLLM(json.dumps({**GOOD, "score": 20})))
    analyses, tasks = saved(test_db)
    assert len(analyses) == 1 and tasks == []


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_real_model_scores_a_warm_call_high(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am"})
    await analyse_call("j1", LINES, "interested")
    (score, brief, _), = saved(test_db)[0]
    assert score >= 60 and brief, (score, brief)


class MustNotBeCalled:
    def __init__(self):
        self.called = False

    async def __call__(self, prompt):
        self.called = True  # analyse_call swallows exceptions, so record the call instead of raising
        return json.dumps(GOOD)


@pytest.mark.parametrize("outcome", ["no_answer", "voicemail", "unreachable", "no_response"])
@pytest.mark.asyncio
async def test_unanswered_calls_are_not_analysed(call_rows, test_db, outcome):
    db.save_call({"job_id": "j1", "outcome": outcome})
    llm = MustNotBeCalled()
    await analyse_call("j1", [("agent", "Hi, you've reached Aoife, leave a message.")], outcome, ask=llm)
    assert not llm.called and saved(test_db) == ([], [])


@pytest.mark.asyncio
async def test_interested_lead_is_in_the_inbox_before_the_model_answers(call_rows, test_db):
    # LiveKit may end the job process mid-analysis: the lead must already be safe by then
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am"})

    async def check_task_first(prompt):
        assert len(saved(test_db)[1]) == 1, "task not saved before the LLM call"
        return json.dumps(GOOD)

    await analyse_call("j1", LINES, "interested", ask=check_task_first)
    analyses, tasks = saved(test_db)
    assert len(analyses) == 1 and len(tasks) == 1
