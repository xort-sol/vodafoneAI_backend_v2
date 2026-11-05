"""Post-call analysis: score, reason, objections, sentiment, next step and a brief for the salesperson."""

import asyncio
import json
import logging
import re

from livekit.agents import inference, llm

from shared import db

logger = logging.getLogger("sales-agent")
MODEL = "openai/gpt-4.1"
TIMEOUT_S = 30
HOT_SCORE = 70
MAX_BRIEF = 600
SENTIMENTS = {"positive", "neutral", "negative"}
UNREACHED = {"no_answer", "voicemail", "unreachable", "no_response"}
QUALIFICATION_KEYS = ("need", "budget", "timeline", "authority")
TAG = re.compile(r"\[[a-z ]+\]\s*")


def build_prompt(lines: list[tuple[str, str, int]], outcome: str) -> str:
    transcript = "\n".join(f"{'Agent' if who == 'agent' else 'Customer'}: {TAG.sub('', text)}" for who, text, _ in lines)
    return f"""You review a sales call an AI agent made to a small business. The recorded outcome is "{outcome}".
Reply with JSON only, no other text, with exactly these keys:
- "score": integer 0-100, how likely this customer is to buy if a salesperson follows up
- "reason": one sentence on why that score
- "objections": list of short phrases, the customer's objections (empty list if none)
- "sentiment": "positive", "neutral" or "negative", how the customer felt by the end
- "next_step": one sentence, what the salesperson should do next, with the time if one was agreed
- "brief": two or three sentences the salesperson reads before calling: who, their situation, what they liked, what worries them
- "questions": list of short phrases, what the customer asked (empty list if none)
- "unanswered": list of short phrases, the questions the agent could not answer (empty list if none)
- "qualification": object with integer keys "need", "budget", "timeline" and "authority", each 0-25: how clear their need is, whether they can afford it, how soon they would buy, and whether they decide
Use only what the transcript says. Never invent prices, names or times.

Transcript:
{transcript}"""


def parse_analysis(text: str) -> dict | None:
    """The model's reply, validated; None when it doesn't match the schema."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    score, objections = data.get("score"), data.get("objections")
    strings = [data.get(k) for k in ("reason", "next_step", "brief")]
    questions, unanswered, qualification = data.get("questions"), data.get("unanswered"), data.get("qualification")
    if (
        not all(isinstance(l, list) and all(isinstance(q, str) for q in l) for l in (questions, unanswered))
        or not isinstance(qualification, dict) or set(qualification) != set(QUALIFICATION_KEYS)
        or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 25 for v in qualification.values())
        or not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100
        or data.get("sentiment") not in SENTIMENTS
        or not isinstance(objections, list) or not all(isinstance(o, str) for o in objections)
        or not all(isinstance(s, str) and s.strip() for s in strings)
    ):
        return None
    return {
        "score": score,
        "reason": data["reason"],
        "objections": objections,
        "sentiment": data["sentiment"],
        "next_step": data["next_step"],
        "brief": data["brief"][:MAX_BRIEF],
        "questions": questions,
        "unanswered": unanswered,
        "qualification": qualification,
    }


def is_hot(outcome: str, score: int | None) -> bool:
    if outcome == "interested":
        return True
    if outcome in ("not_interested", "do_not_call", "callback"):
        return False
    return score is not None and score >= HOT_SCORE


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
    if outcome == "interested":
        # before the model call: LiveKit may end the job process while it runs
        await asyncio.to_thread(db.save_analysis, job_id, None, MODEL, True)
    analysis = None
    try:
        reply = await asyncio.wait_for((ask or ask_llm)(build_prompt(lines, outcome)), TIMEOUT_S)
        analysis = parse_analysis(reply)
        if analysis is None:
            logger.error("analysis for job %s did not match the schema: %.300s", job_id, reply)
    except Exception as e:
        logger.error("analysis failed for job %s: %r", job_id, e)
    hot = is_hot(outcome, analysis["score"] if analysis else None)
    await asyncio.to_thread(db.save_analysis, job_id, analysis, MODEL, hot)
