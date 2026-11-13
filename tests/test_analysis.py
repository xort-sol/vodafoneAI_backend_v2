import json
import os
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest

from dotenv import load_dotenv

from analysis import analyse
from analysis.analyse import FACTS, analyse_call, avg_latency_ms, build_prompt, is_hot, parse_analysis
from shared import db

load_dotenv()  # LiveKit credentials for the real-model test; conftest still pins DATABASE_URL to the test database

DISCOVERY = {
    "employees": {"status": "known", "value": {"exact": 8}, "source": "said"},
    "bb_issues": {"status": "known", "value": True, "source": "said"},
    "decision_maker": {"status": "known", "value": True, "source": "said"},
    "bb_spend": {"status": "declined", "value": None},
}

GOOD = {
    "discovery": DISCOVERY,
    "reason": "Asked about price and booked a call.",
    "objections": ["price"],
    "sentiment": "positive",
    "next_step": "Call Thursday 10am about Business Broadband 500.",
    "brief": "Aoife, 8 staff, slow office internet; liked the 500 plan at forty euro; worried about price.",
    "questions": ["Is it 5G?"],
    "unanswered": [],
}


LINES = [
    ("agent", "[warmly] Hi Aoife, I'm an AI assistant from Acme. Have you a quick minute?", 0),
    ("customer", "Yes, go on.", 0),
    ("agent", "How's the office internet these days?", 0),
    ("customer", "Slow, honestly. Eight of us on it.", 0),
    ("agent", "Business Broadband 500 starts at forty euro. Want to hear more?", 0),
    ("customer", "Bit pricey, but have someone ring me Thursday at 10.", 0),
]


def test_prompt_carries_the_transcript_and_outcome_without_delivery_tags():
    prompt = build_prompt(LINES, "interested")
    assert "Customer: Slow, honestly." in prompt and "interested" in prompt
    assert "[warmly]" not in prompt


def test_prompt_asks_for_discovery_facts_from_the_customers_words_only():
    prompt = build_prompt(LINES, "interested")
    for name in ("decision_maker", "bb_spend", "mob_connections", "considering_change", "other_suppliers"):
        assert f'"{name}"' in prompt
    assert "only what the customer said" in prompt and '"declined"' in prompt and '"score"' not in prompt


def test_prompt_has_todays_irish_date_and_the_real_call_rules():
    prompt = build_prompt(LINES, "interested")
    today = datetime.now(db.IRELAND).date()
    assert f"Today is {today:%A} {today.isoformat()} in Ireland" in prompt
    for phrase in ('{"month": m, "day": d}', "only when they stated or confirmed the year",
                   '"five hundred plus" is {"min": 500, "max": null}', '"exact" only for a figure they gave',
                   '"five people" is employees {"exact": 5}', "questions the customer asked, not the agent",
                   "agreeing to a callback is not enough"):
        assert phrase in prompt, phrase


def load_transcript(path) -> list[tuple[str, str, int]]:
    """The saved real-call transcript: '  52s cust: text' lines, as (agent or customer, text, seconds)."""
    lines = []
    for raw in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\d+)s (agen|cust): (.*)", raw)
        if m:
            lines.append(({"agen": "agent", "cust": "customer"}[m[2]], m[3], int(m[1])))
    return lines


# a real customer's call: kept out of git (.superpowers/sdd is ignored), so these tests skip on a clean checkout
ORLA = Path(__file__).parent / "fixtures" / "call22-orla.txt"  # the user's own rehearsal call, test-lead names


def test_orla_transcript_loads():
    lines = load_transcript(ORLA)
    assert lines[0] == ("customer", "No.", 8) and ("customer", "Five", 138) in lines


def test_parse_accepts_json_inside_chatter_or_a_code_fence():
    parsed = parse_analysis(json.dumps(GOOD))
    assert parse_analysis(f"Sure! Here it is:\n```json\n{json.dumps(GOOD)}\n```") == parsed
    assert {k: parsed[k] for k in ("reason", "sentiment", "next_step", "questions")} == {
        k: GOOD[k] for k in ("reason", "sentiment", "next_step", "questions")}


def test_parse_keeps_said_and_declined_facts_and_marks_the_rest_unknown():
    d = parse_analysis(json.dumps(GOOD))["discovery"]
    assert set(d) == set(FACTS)
    assert d["employees"] == {"status": "known", "value": {"exact": 8}}
    assert d["bb_spend"] == {"status": "declined", "value": None}
    assert d["mob_network"] == {"status": "unknown", "value": None}


@pytest.mark.parametrize("name, fact", [
    ("employees", {"status": "known", "value": 8, "source": "said"}),  # a count must be exact or a range
    ("employees", {"status": "known", "value": {"min": 10, "max": 5}, "source": "said"}),
    ("employees", {"status": "known", "value": {"exact": -1}, "source": "said"}),
    ("employees", {"status": "known", "value": {"exact": True}, "source": "said"}),
    ("decision_maker", {"status": "known", "value": "yes", "source": "said"}),
    ("business_type", {"status": "known", "value": "  ", "source": "said"}),
    ("bb_spend", {"status": "known", "value": {"exact": 60, "vat": "maybe"}, "source": "said"}),
    ("bb_spend", {"status": "known", "value": {"exact": 60, "per": "year"}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"date": "next spring"}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"months": 4, "type": "monthly"}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"month": 13, "day": 1}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"month": 2, "day": 30}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"month": 1, "day": True}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"month": 1}, "source": "said"}),
    ("bb_contract", {"status": "known", "value": {"month": 1, "day": 25, "year": 2027}, "source": "said"}),
    ("employees", {"status": "guessed", "value": None}),
    ("employees", {"status": "known", "value": {"exact": 8}}),  # a guess: nobody said it
    ("employees", {"status": "known", "value": {"exact": 8}, "source": "inferred"}),
    ("employees", "eight"),
    ("bb_spend", '{"status": "known", "value": {"exact": Infinity}, "source": "said"}'),
])
def test_parse_drops_a_malformed_or_unsaid_fact_but_keeps_the_analysis(name, fact):
    if isinstance(fact, str) and fact.startswith("{"):
        fact = json.loads(fact)  # Infinity: json.loads accepts it, jsonb does not
    parsed = parse_analysis(json.dumps({**GOOD, "discovery": {**DISCOVERY, name: fact}}))
    assert parsed["discovery"][name] == {"status": "unknown", "value": None}
    assert parsed["discovery"]["bb_issues"] == {"status": "known", "value": True}


@pytest.mark.parametrize("name, value", [
    ("employees", {"min": 50, "max": None}), ("bb_spend", {"min": 50, "max": 75, "vat": "excl"}),
    ("mob_spend", {"exact": 149.5}), ("bb_contract", {"date": "2027-03-01", "type": "fixed"}),
    ("mob_contract", {"months": 4}), ("mob_contract", {"month": 1, "day": 25}),
    ("bb_contract", {"month": 2, "day": 29, "type": "fixed"}), ("pain_point", "Drops during video calls"), ("m365", False),
])
def test_parse_keeps_every_valid_shape(name, value):
    fact = {"status": "known", "value": value, "source": "said"}
    assert parse_analysis(json.dumps({**GOOD, "discovery": {name: fact}}))["discovery"][name] == {"status": "known", "value": value}


def test_parse_ignores_unknown_fact_names_and_clears_values_that_were_not_said():
    parsed = parse_analysis(json.dumps({**GOOD, "discovery": {"shoe_size": {"status": "known", "value": 9, "source": "said"},
                                                              "m365": {"status": "n/a", "value": True}}}))
    assert "shoe_size" not in parsed["discovery"] and parsed["discovery"]["m365"] == {"status": "n/a", "value": None}


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({**GOOD, "sentiment": "ecstatic"}),
        json.dumps({**GOOD, "objections": "price"}),
        json.dumps({**GOOD, "discovery": "lots"}),
        json.dumps({k: v for k, v in GOOD.items() if k != "brief"}),
        json.dumps({k: v for k, v in GOOD.items() if k != "discovery"}),
    ],
)
def test_parse_rejects_anything_off_schema(bad):
    assert parse_analysis(bad) is None


def test_parse_trims_an_overlong_brief():
    assert len(parse_analysis(json.dumps({**GOOD, "brief": "x" * 2000}))["brief"]) == 600


@pytest.mark.parametrize(
    "outcome, classification, hot",
    [("interested", "low", True), ("interested", None, True), ("transferred", None, True),
     ("field_visit", None, True), ("email_request", "low", True), ("hung_up", "hot", True),
     ("callback", "hot", False), ("hung_up", "warm", False), ("hung_up", None, False),
     ("not_interested", "low", False), ("do_not_call", "low", False)],
)
def test_hot_lead_rule(outcome, classification, hot):
    # a person follows up an agreed next step; a callback is re-dialled by the AI, so it is never hot
    assert is_hot(outcome, classification) is hot


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
                  "callback_at": "2037-11-05T10:00:00+00:00"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))  # retried: no duplicate
    analyses, tasks = saved(test_db)
    assert [(s, m) for s, _, m in analyses] == [(35, "openai/gpt-4.1")]  # decision maker, 5+ staff, issues, next step
    assert tasks == [("follow_up", True)]


@pytest.mark.asyncio
async def test_analysis_saves_discovery_scores_and_updates_the_lead(call_rows, test_db):
    with psycopg.connect(test_db) as c:
        c.execute("truncate leads restart identity cascade")
        lead_id = c.execute("insert into leads (phone, status, attempts, consent) values ('+353870000001', 'dialling', 1, true) returning id").fetchone()[0]
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": "interested"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))
    with psycopg.connect(test_db) as c:
        assert c.execute("select score, sentiment, ai_summary from leads where id = %s", [lead_id]).fetchone() == (
            35, "positive", GOOD["brief"])
        questions, qualification, score, classification, discovery = c.execute(
            "select questions, qualification, score, classification, discovery from analyses").fetchone()
    assert questions == ["Is it 5G?"] and score == 35 and classification == "low"
    assert qualification["decision_maker"] == 10 and qualification["bb_spend_50"] == 0 and len(qualification) == 11
    assert discovery["employees"] == {"status": "known", "value": {"exact": 8}}
    assert discovery["bb_spend"]["status"] == "declined" and discovery["mob_network"]["status"] == "unknown"


@pytest.mark.parametrize("outcome", ["not_interested", "do_not_call", "hung_up"])
@pytest.mark.asyncio
async def test_an_opt_out_keeps_the_leads_score_but_the_analysis_keeps_the_total(call_rows, test_db, outcome):
    with psycopg.connect(test_db) as c:
        c.execute("truncate leads restart identity cascade")
        lead_id = c.execute("insert into leads (phone, status, attempts, consent, score) values ('+353870000001', 'dialling', 1, true, 12) "
                            "returning id").fetchone()[0]
    db.save_call({"job_id": "j1", "lead_id": lead_id, "outcome": outcome})
    await analyse_call("j1", LINES, outcome, ask=FakeLLM(json.dumps(GOOD)))
    with psycopg.connect(test_db) as c:
        assert c.execute("select score, sentiment, ai_summary from leads where id = %s", [lead_id]).fetchone() == (
            12, "positive", GOOD["brief"])
        assert c.execute("select score from analyses").fetchone() == (25,)  # no next step: 35 - 10


@pytest.mark.parametrize("outcome", ["transferred", "field_visit", "email_request"])
@pytest.mark.asyncio
async def test_follow_up_outcomes_reach_the_inbox_even_when_analysis_fails(call_rows, test_db, outcome):
    db.save_call({"job_id": "j1", "outcome": outcome})
    await analyse_call("j1", LINES, outcome, ask=FakeLLM("not json"))
    analyses, tasks = saved(test_db)
    assert analyses == [] and len(tasks) == 1


@pytest.mark.asyncio
async def test_a_scoring_error_still_saves_the_hot_lead(call_rows, test_db, monkeypatch):
    def broken(*args):
        raise ValueError("bad fact")

    monkeypatch.setattr(analyse, "score", broken)
    db.save_call({"job_id": "j1", "outcome": "interested"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(json.dumps(GOOD)))
    analyses, tasks = saved(test_db)
    assert analyses == [] and len(tasks) == 1


@pytest.mark.asyncio
async def test_interested_lead_reaches_the_inbox_even_when_analysis_fails(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am"})
    await analyse_call("j1", LINES, "interested", ask=FakeLLM(TimeoutError()))
    analyses, tasks = saved(test_db)
    assert analyses == [] and len(tasks) == 1


@pytest.mark.asyncio
async def test_cold_call_is_analysed_but_not_sent_to_the_inbox(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "not_interested"})
    await analyse_call("j1", LINES, "not_interested", ask=FakeLLM(json.dumps(GOOD)))
    analyses, tasks = saved(test_db)
    assert len(analyses) == 1 and tasks == []


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_real_model_captures_what_the_customer_said(call_rows, test_db):
    db.save_call({"job_id": "j1", "outcome": "interested", "callback_time": "Thu 10am"})
    await analyse_call("j1", LINES, "interested")
    with psycopg.connect(test_db) as c:
        discovery, brief = c.execute("select discovery, brief from analyses").fetchone()
    assert discovery["employees"]["value"] == {"exact": 8} and brief, discovery


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
    await analyse_call("j1", [("agent", "Hi, you've reached Aoife, leave a message.", 0)], outcome, ask=llm)
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


@pytest.mark.skipif(not os.getenv("LIVEKIT_API_KEY"), reason="needs LiveKit Cloud credentials for inference")
@pytest.mark.asyncio
async def test_real_model_scores_the_orla_call(call_rows, test_db):
    # call 22, 13 Nov 2026: a five-person physio on three sites; mobile spend "five hundred plus"; contract "January 25th" with no year
    db.save_call({"job_id": "orla", "outcome": "interested", "callback_time": "Saturday at 5pm"})
    await analyse_call("orla", load_transcript(ORLA), "interested")
    with psycopg.connect(test_db) as c:
        d, questions, score, classification = c.execute("select discovery, questions, score, classification from analyses").fetchone()
    value = lambda name: d[name]["value"] if d[name]["status"] == "known" else None
    assert value("employees") == {"exact": 5} and value("premises") == {"exact": 3}, d
    assert value("decision_maker") is True and value("bb_issues") is True and value("m365") is False, d
    assert value("mob_network").lower().startswith("eir") and value("mob_connections") == {"exact": 5}, d
    assert value("mob_spend") == {"min": 500, "max": None}, d
    assert value("bb_provider") is None and value("bb_spend") is None, d
    assert all(value(c) is None or "date" not in value(c) for c in ("bb_contract", "mob_contract")), d
    assert value("considering_change") is not True and questions == [], (d, questions)
    assert (score, classification) == (55, "developing"), d
