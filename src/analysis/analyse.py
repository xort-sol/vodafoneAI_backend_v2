"""Post-call analysis: the discovery facts the customer gave, reason, objections, sentiment, next step and a brief for the salesperson.
The score is not the model's: analysis.score works it out from the facts."""

import asyncio
import json
import logging
import math
import re
from datetime import date

from livekit.agents import inference, llm

from analysis.score import score
from shared import db

logger = logging.getLogger("sales-agent")
MODEL = "openai/gpt-4.1"
TIMEOUT_S = 30
MAX_BRIEF = 600
SENTIMENTS = {"positive", "neutral", "negative"}
UNREACHED = {"no_answer", "voicemail", "unreachable", "no_response"}
ALWAYS_HOT = {"interested", "transferred", "field_visit", "email_request"}  # a person follows these up
TAG = re.compile(r"\[[a-z ]+\]\s*")
STATUSES = {"known", "unknown", "declined", "n/a"}
YES_NO = ("decision_maker", "bb_issues", "bb_peak_issues", "m365", "voice", "considering_change", "other_suppliers", "mob_coverage")
TEXT = ("business_type", "workforce", "other_decision_makers", "bb_provider", "bb_package", "bb_speed", "bb_needs",
        "mob_network", "mob_data_roaming", "mob_handsets", "pain_point", "impact", "desired_improvement", "timeline")
COUNTS = ("employees", "premises", "devices", "mob_connections")
SPEND = ("bb_spend", "mob_spend")
CONTRACT = ("bb_contract", "mob_contract")
FACTS = YES_NO + TEXT + COUNTS + SPEND + CONTRACT
UNKNOWN = {"status": "unknown", "value": None}


def build_prompt(lines: list[tuple[str, str, int]], outcome: str) -> str:
    transcript = "\n".join(f"{'Agent' if who == 'agent' else 'Customer'}: {TAG.sub('', text)}" for who, text, _ in lines)
    return f"""You review a sales call an AI agent made to a small business. The recorded outcome is "{outcome}".
Reply with JSON only, no other text, with exactly these keys:
- "discovery": object keyed by fact name, each fact {{"status": "known"|"unknown"|"declined"|"n/a", "value": ..., "source": "said"}}
- "reason": one sentence on how likely this customer is to buy, and why
- "objections": list of short phrases, the customer's objections (empty list if none)
- "sentiment": "positive", "neutral" or "negative", how the customer felt by the end
- "next_step": one sentence, what the salesperson should do next, with the time if one was agreed
- "brief": two or three sentences the salesperson reads before calling: who, their situation, what they liked, what worries them
- "questions": list of short phrases, what the customer asked (empty list if none)
- "unanswered": list of short phrases, the questions the agent could not answer (empty list if none)

Discovery facts. Use only what the customer said: a fact is "known" only when the customer said it, with "source": "said".
Anything not said is "unknown" (or leave it out), anything the customer refused to tell is "declined", and "n/a" when it does not apply.
"value" is set only when "known", otherwise null. Never guess or infer a value.
- Yes/no (true or false): "decision_maker", "bb_issues", "bb_peak_issues", "m365" (a Microsoft 365 need), "voice" (a phone system need), "considering_change", "other_suppliers" (uses more than one supplier), "mob_coverage" (has mobile coverage problems)
- Short text: "business_type", "workforce" (office, remote, field), "other_decision_makers", "bb_provider", "bb_package" (business or residential), "bb_speed", "bb_needs" (cloud, video, remote), "mob_network", "mob_data_roaming", "mob_handsets", "pain_point", "impact", "desired_improvement", "timeline"
- Counts: "employees", "premises", "devices", "mob_connections": {{"exact": n}} or {{"min": a, "max": b}} (max null for "50+")
- Spend in euro per month: "bb_spend", "mob_spend": {{"exact": n}} or {{"min": a, "max": b}}, plus "vat": "incl" or "excl" only if they said
- Contract end: "bb_contract", "mob_contract": {{"date": "YYYY-MM-DD"}} or {{"months": n}} from today ({date.today().isoformat()}), plus "type": "fixed" or "rolling" only if they said
Never invent prices, names or times.

Transcript:
{transcript}"""


def _number(n) -> bool:
    # finite: json.loads accepts Infinity, which jsonb would refuse, losing the whole analysis
    return isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) and n >= 0


def _amount(v: dict, extra: dict) -> bool:
    """{"exact": n} or {"min": a, "max": b or null}, plus only the optional keys in extra, each with an allowed value."""
    keys = set(v) - set(extra)
    if not all(v[k] in allowed for k, allowed in extra.items() if k in v):
        return False
    if keys == {"exact"}:
        return _number(v["exact"])
    return keys == {"min", "max"} and _number(v["min"]) and (v["max"] is None or _number(v["max"]) and v["max"] >= v["min"])


def _contract(v: dict) -> bool:
    keys = set(v) - {"type"}
    if v.get("type", "fixed") not in ("fixed", "rolling"):
        return False
    if keys == {"months"}:
        return isinstance(v["months"], int) and not isinstance(v["months"], bool) and v["months"] >= 0
    if keys == {"date"} and isinstance(v["date"], str):
        try:
            date.fromisoformat(v["date"])
            return True
        except ValueError:
            return False
    return False


def _valid_value(name: str, v) -> bool:
    if name in YES_NO:
        return isinstance(v, bool)
    if name in TEXT:
        return isinstance(v, str) and bool(v.strip())
    if not isinstance(v, dict):
        return False
    if name in COUNTS:
        return _amount(v, {}) and all(isinstance(n, int) for n in v.values() if n is not None)
    if name in SPEND:
        return _amount(v, {"vat": ("incl", "excl")})
    return _contract(v)


def parse_discovery(raw: dict) -> dict:
    """Every fact, in the stored shape; a malformed or unsaid fact is unknown, so one bad fact never loses the call."""
    facts = {}
    for name in FACTS:
        fact = raw.get(name)
        if not isinstance(fact, dict) or fact.get("status") not in STATUSES:
            facts[name] = UNKNOWN
        elif fact["status"] != "known":
            facts[name] = {"status": fact["status"], "value": None}
        elif fact.get("source") == "said" and _valid_value(name, fact.get("value")):
            facts[name] = {"status": "known", "value": fact["value"]}
        else:
            facts[name] = UNKNOWN
    return facts


def parse_analysis(text: str) -> dict | None:
    """The model's reply, validated; None when it doesn't match the schema."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    lists = [data.get(k) for k in ("objections", "questions", "unanswered")]
    strings = [data.get(k) for k in ("reason", "next_step", "brief")]
    if (
        not isinstance(data.get("discovery"), dict)
        or not all(isinstance(l, list) and all(isinstance(q, str) for q in l) for l in lists)
        or data.get("sentiment") not in SENTIMENTS
        or not all(isinstance(s, str) and s.strip() for s in strings)
    ):
        return None
    return {
        "discovery": parse_discovery(data["discovery"]),
        "reason": data["reason"],
        "objections": data["objections"],
        "sentiment": data["sentiment"],
        "next_step": data["next_step"],
        "brief": data["brief"][:MAX_BRIEF],
        "questions": data["questions"],
        "unanswered": data["unanswered"],
    }


def is_hot(outcome: str, classification: str | None) -> bool:
    # a callback is re-dialled by the AI, not a person
    return outcome in ALWAYS_HOT or (outcome != "callback" and classification == "hot")


def avg_latency_ms(items) -> int | None:
    """Mean time from the customer finishing a turn to the agent's first audio, over turns where all three parts were measured."""
    turns, end_of_turn = [], None
    for item in items:
        if getattr(item, "type", None) != "message":
            continue
        metrics = item.metrics or {}
        if item.role == "user":
            end_of_turn = metrics.get("end_of_turn_delay")
        elif item.role == "assistant":
            if end_of_turn is not None and "llm_node_ttft" in metrics and "tts_node_ttfb" in metrics:
                turns.append(end_of_turn + metrics["llm_node_ttft"] + metrics["tts_node_ttfb"])
            end_of_turn = None
    return round(1000 * sum(turns) / len(turns)) if turns else None


async def ask_llm(prompt: str) -> str:
    async with inference.LLM(model=MODEL) as model:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="user", content=prompt)
        return (await model.chat(chat_ctx=ctx).collect()).text


async def analyse_call(job_id: str, lines: list[tuple[str, str, int]], outcome: str, ask=None) -> None:
    """Analyse one saved call; a hot lead goes to the inbox even if the analysis fails, so it is never lost."""
    if outcome in UNREACHED or not any(who == "customer" for who, _, _ in lines):
        return  # nobody to analyse: unanswered, voicemail, or silence
    if outcome in ALWAYS_HOT:
        # before the model call: LiveKit may end the job process while it runs
        await asyncio.to_thread(db.save_analysis, job_id, None, MODEL, True)
    analysis = None
    try:
        reply = await asyncio.wait_for((ask or ask_llm)(build_prompt(lines, outcome)), TIMEOUT_S)
        parsed = parse_analysis(reply)
        if parsed is None:
            logger.error("analysis for job %s did not match the schema: %.300s", job_id, reply)
        else:
            s = score(parsed["discovery"], outcome)
            # set only once scored, so a scoring error can't break the save below
            analysis = parsed | {"qualification": s["criteria"], "score": s["total"], "classification": s["classification"]}
    except Exception as e:
        logger.error("analysis failed for job %s: %r", job_id, e)
    hot = is_hot(outcome, analysis["classification"] if analysis else None)
    await asyncio.to_thread(db.save_analysis, job_id, analysis, MODEL, hot)
