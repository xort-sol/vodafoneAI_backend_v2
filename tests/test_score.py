from datetime import date

import pytest

from analysis.score import score

TODAY = date(2026, 11, 13)


def known(value):
    return {"status": "known", "value": value}


FULL = {
    "decision_maker": known(True),
    "employees": known({"exact": 8}),
    "bb_spend": known({"exact": 60, "vat": "incl"}),
    "bb_contract": known({"months": 3, "type": "fixed"}),
    "bb_issues": known(True),
    "mob_connections": known({"min": 5, "max": 10}),
    "mob_spend": known({"exact": 200}),
    "m365": known(True),
    "other_suppliers": known(True),
    "considering_change": known(True),
}


def points(discovery, outcome="interested", today=TODAY):
    return score(discovery, outcome, today)


def test_full_score_is_100_and_hot():
    s = points(FULL)
    assert s["total"] == 100 and s["classification"] == "hot"
    assert set(s["criteria"]) == {"decision_maker", "employees_5", "bb_spend_50", "bb_contract_6m", "bb_issues",
                                  "mobiles_5", "mob_spend_150", "m365_need", "multiple_suppliers",
                                  "considering_change", "next_step"}


def test_empty_score_is_0_and_low():
    s = points({}, "hung_up")
    assert s["total"] == 0 and s["classification"] == "low" and set(s["criteria"].values()) == {0}


def test_partial_score():
    s = points({"decision_maker": known(True), "employees": known({"exact": 3})}, "callback")
    assert s["criteria"]["decision_maker"] == 10 and s["criteria"]["employees_5"] == 0
    assert s["criteria"]["next_step"] == 10 and s["total"] == 20


@pytest.mark.parametrize("drop, total, classification", [
    # points come in fives, so 39, 59 and 79 are covered by test_classification_boundaries
    (["decision_maker", "employees", "bb_spend", "bb_contract", "bb_issues", "mob_connections", "mob_spend"], 35, "low"),
    (["decision_maker", "employees", "bb_spend", "bb_contract", "bb_issues", "mob_connections", "m365"], 40, "developing"),
    (["decision_maker", "bb_spend", "bb_contract", "bb_issues"], 60, "warm"),
    (["decision_maker", "employees", "bb_spend", "bb_contract", "bb_issues"], 55, "developing"),
    (["decision_maker", "employees", "bb_spend"], 75, "warm"),
    (["decision_maker", "bb_spend"], 80, "hot"),
    (["decision_maker", "employees"], 85, "hot"),
    (["decision_maker"], 90, "hot"),
])
def test_classification_bands(drop, total, classification):
    s = points({k: v for k, v in FULL.items() if k not in drop})
    assert (s["total"], s["classification"]) == (total, classification)


@pytest.mark.parametrize("total, classification", [(39, "low"), (40, "developing"), (59, "developing"),
                                                   (60, "warm"), (79, "warm"), (80, "hot")])
def test_classification_boundaries(total, classification):
    from analysis.score import classify
    assert classify(total, "interested") == classification


@pytest.mark.parametrize("value, pts", [({"exact": 50}, 10), ({"exact": 49}, 0), ({"min": 40, "max": 60}, 0),
                                        ({"min": 50, "max": 75}, 10), ({"min": 50, "max": None}, 10)])
def test_broadband_spend_reads_the_range_minimum(value, pts):
    assert points({"bb_spend": known(value)})["criteria"]["bb_spend_50"] == pts


@pytest.mark.parametrize("value, pts", [({"exact": 149}, 0), ({"exact": 150}, 10), ({"min": 140, "max": 160}, 0)])
def test_mobile_spend_threshold(value, pts):
    assert points({"mob_spend": known(value)})["criteria"]["mob_spend_150"] == pts


@pytest.mark.parametrize("value, pts", [({"months": 6}, 10), ({"months": 7}, 0), ({"months": 0}, 10),
                                        ({"date": "2027-05-13"}, 10), ({"date": "2027-05-14"}, 0),
                                        ({"date": "2027-01-01", "type": "fixed"}, 10),
                                        ({"date": "2026-01-01"}, 10),  # already ended: out of contract now
                                        ({"month": 1, "day": 25}, 0)])  # no year: unconfirmed, never scores
def test_broadband_contract_within_six_months(value, pts):
    assert points({"bb_contract": known(value)})["criteria"]["bb_contract_6m"] == pts


def test_six_months_from_month_end_clamps_the_day():
    # 31 August + 6 months is the end of February
    assert points({"bb_contract": known({"date": "2027-02-28"})}, today=date(2026, 8, 31))["criteria"]["bb_contract_6m"] == 10


@pytest.mark.parametrize("value, pts", [({"exact": 5}, 5), ({"exact": 4}, 0), ({"min": 3, "max": 8}, 0),
                                        ({"min": 50, "max": None}, 5)])
def test_employees_threshold(value, pts):
    assert points({"employees": known(value)})["criteria"]["employees_5"] == pts


@pytest.mark.parametrize("status", ["unknown", "declined", "n/a"])
def test_unknown_declined_and_na_score_nothing(status):
    discovery = {k: {"status": status, "value": None} for k in FULL}
    s = points(discovery, "not_interested")
    assert s["total"] == 0


def test_false_answers_score_nothing():
    s = points({k: known(False) for k in ("decision_maker", "bb_issues", "m365", "other_suppliers", "considering_change")})
    assert s["total"] == 10  # only the next step


def test_different_broadband_and_mobile_providers_count_as_multiple_suppliers():
    assert points({"bb_provider": known("Eir"), "mob_network": known("Three")})["criteria"]["multiple_suppliers"] == 10
    assert points({"bb_provider": known("Eir"), "mob_network": known("eir")})["criteria"]["multiple_suppliers"] == 0
    assert points({"bb_provider": known("Eir")})["criteria"]["multiple_suppliers"] == 0


@pytest.mark.parametrize("bb, mob", [("Eir", "eir Mobile"), ("Vodafone", "Vodafone Ireland"), (" Three ", "three")])
def test_the_same_supplier_named_two_ways_is_one_supplier(bb, mob):
    assert points({"bb_provider": known(bb), "mob_network": known(mob)})["criteria"]["multiple_suppliers"] == 0


@pytest.mark.parametrize("outcome, pts", [("interested", 10), ("callback", 10), ("field_visit", 10), ("transferred", 10),
                                          ("email_request", 10), ("not_interested", 0), ("hung_up", 0), ("do_not_call", 0)])
def test_next_step_follows_the_outcome(outcome, pts):
    assert points({}, outcome)["criteria"]["next_step"] == pts


@pytest.mark.parametrize("outcome", ["not_interested", "do_not_call", "hung_up"])
def test_a_no_with_a_high_score_is_low(outcome):
    s = points(FULL, outcome)
    assert s["total"] == 90 and s["classification"] == "low"
