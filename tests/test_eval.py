"""Persona eval, the Phase 1 gate: uv run pytest -m eval. Re-run after every prompt change.

Every call must disclose the AI in the first message, invent no prices and ask at most one question per turn;
each persona can also expect an outcome and a judged last reply. Personas are in personas.py.
shortcut: customers are scripted lines, so they don't react to what the agent says; switch to an LLM-played
customer if scripted lines start missing real failures.
"""

import re
from datetime import datetime, timedelta

import pytest
from livekit.agents import AgentSession, inference

from agent import main as agent_module
from agent.main import TEST_LEAD, SalesAgent
from eval_checks import invented_prices
from personas import PERSONAS

CATALOGUE = 'RED Business Lite: "forty-five euro a month". Up to 2 gigabits: "sixty-five euro a month". Save ten euro.'
CATALOGUE += ' 4G: "twenty-four euro thirty-nine a month, then thirty-six fifty-nine".'
CATALOGUE += " Ninety-six percent are installed within fourteen days."


@pytest.mark.parametrize(
    "reply,invented",
    [
        ("[warmly] It's just... forty-five euro a month.", []),
        ("Our fastest is sixty-five euro a month, and you save ten euro.", []),
        ("It's only thirty euro a month.", ["thirty"]),
        ("It's sixty euro.", ["sixty"]),  # part of "sixty-five" in the catalogue, but a different price
        ("That's €45 a month.", ["€45"]),  # digits: the prompt says to speak prices, and we can't check them
        ("It takes two minutes, one question.", []),  # a number that isn't a price
        ("It's twenty-four thirty-nine a month.", []),  # euro-and-cents, said without "euro"
        ("Then thirty-six fifty-nine a month.", []),  # catalogue price not followed by "euro"
        ("Just fourteen euro a month.", ["fourteen"]),  # in the catalogue, but as days, not a price
    ],
)
def test_invented_prices_are_caught(reply, invented):
    assert invented_prices(reply, CATALOGUE) == invented


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("persona", PERSONAS, ids=lambda p: p["id"])
async def test_persona(persona, call_rows):
    async with inference.LLM(model=agent_module.LLM_MODEL) as llm, AgentSession(llm=llm) as session:
        await session.start(SalesAgent({**TEST_LEAD, "job_id": f"eval-{persona['id']}", **persona.get("lead", {})}))
        await session.generate_reply(instructions="Open the call.")
        for line in persona["lines"]:
            result = await session.run(user_input=line)
            if any(e.type == "function_call" and e.item.name == "end_call" for e in result.events):
                break
        replies = [i.text_content for i in session.history.items if i.type == "message" and i.role == "assistant"]
        if "intent" in persona:
            # the judge sees only the last reply; intents like "repeats its last point" need the one before
            intent = f"{persona['intent']} (Its previous message was: {replies[-2]!r})"
            await result.expect.contains_message(role="assistant").judge(llm, intent=intent)

    assert re.search(r"\bAI\b", replies[0]), f"no AI disclosure: {replies[0]}"
    assert persona.get("opening_has", "") in replies[0]
    catalogue = agent_module.SERVICES + agent_module.OBJECTIONS
    assert not [p for r in replies for p in invented_prices(r, catalogue)], replies
    assert not [r for r in replies if r.count("?") > 1], "more than one question in a turn"
    rows = call_rows()
    if "outcome" in persona:
        assert [r["outcome"] for r in rows] == [persona["outcome"]], replies
    if persona.get("callback_tomorrow"):
        tomorrow = datetime.now(agent_module.IRELAND).date() + timedelta(days=1)
        assert f"{tomorrow.day} {tomorrow:%B}" in rows[0]["callback_time"], rows[0]["callback_time"]
