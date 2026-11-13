"""Management's 100-point qualification matrix, scored from the discovery facts a call captured. Pure: no I/O."""

import calendar
from datetime import date

NEXT_STEP = {"interested", "callback", "field_visit", "transferred", "email_request"}
NEVER_WARM = {"not_interested", "do_not_call", "hung_up"}  # the score is kept, but the lead is not chased


def _value(discovery: dict, name: str):
    fact = discovery.get(name) or {}
    return fact.get("value") if fact.get("status") == "known" else None


def _minimum(value) -> float | None:
    """The amount a threshold can rely on: the exact figure, or the bottom of a range."""
    if not isinstance(value, dict):
        return None
    return value.get("exact", value.get("min"))


def _at_least(value, threshold: float) -> bool:
    low = _minimum(value)
    return low is not None and low >= threshold


def _six_months_on(today: date) -> date:
    month = today.month + 6
    year, month = today.year + (month - 1) // 12, (month - 1) % 12 + 1
    return date(year, month, min(today.day, calendar.monthrange(year, month)[1]))


def _ends_within_six_months(contract, today: date) -> bool:
    if not isinstance(contract, dict):
        return False
    if "months" in contract:
        return contract["months"] <= 6
    if "date" in contract:
        # a date already past still counts: they are out of contract now
        return date.fromisoformat(contract["date"]) <= _six_months_on(today)
    return False


def classify(total: int, outcome: str) -> str:
    if outcome in NEVER_WARM or total < 40:
        return "low"
    return "hot" if total >= 80 else "warm" if total >= 60 else "developing"


def score(discovery: dict, outcome: str, today: date | None = None) -> dict:
    """{criteria: {name: points}, total, classification}; a point is awarded only when the threshold is clearly met."""
    today = today or date.today()
    yes = lambda name: _value(discovery, name) is True
    bb, mob = _value(discovery, "bb_provider"), _value(discovery, "mob_network")
    met = {
        "decision_maker": (10, yes("decision_maker")),
        "employees_5": (5, _at_least(_value(discovery, "employees"), 5)),
        "bb_spend_50": (10, _at_least(_value(discovery, "bb_spend"), 50)),
        "bb_contract_6m": (10, _ends_within_six_months(_value(discovery, "bb_contract"), today)),
        "bb_issues": (10, yes("bb_issues")),
        "mobiles_5": (10, _at_least(_value(discovery, "mob_connections"), 5)),
        "mob_spend_150": (10, _at_least(_value(discovery, "mob_spend"), 150)),
        "m365_need": (5, yes("m365")),
        "multiple_suppliers": (10, yes("other_suppliers") or bool(bb and mob and bb.strip().lower() != mob.strip().lower())),
        "considering_change": (10, yes("considering_change")),
        "next_step": (10, outcome in NEXT_STEP),
    }
    criteria = {name: points if ok else 0 for name, (points, ok) in met.items()}
    total = sum(criteria.values())
    return {"criteria": criteria, "total": total, "classification": classify(total, outcome)}
