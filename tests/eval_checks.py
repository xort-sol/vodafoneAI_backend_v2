import re

NUMBER_WORDS = set(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand".split()
)
DIGIT_PRICE = re.compile(r"€\s?\d[\d.,]*|\b\d[\d.,]*\s?euro", re.IGNORECASE)


def number_runs(text: str, prices_only: bool) -> list[str]:
    """Runs of number words. prices_only: just those said as a price, followed by "euro" or "a/per month",
    or after "then" ("then thirty-six fifty-nine"). "euro" between numbers is euro-and-cents
    ("twenty-four euro thirty-nine") and joins the run; punctuation ends it."""
    words = re.findall(r"[a-z]+|[.,;:!?\"]", text.lower())
    runs, run, before = [], [], ""
    for i, word in enumerate(words + [""]):
        after = words[i + 1] if i + 1 < len(words) else ""
        if word in NUMBER_WORDS or (run and word in ("and", "euro") and after in NUMBER_WORDS):
            if not run:
                before = words[i - 1] if i else ""
            if word != "euro":
                run.append(word)
            continue
        priced = word == "euro" or (word in ("a", "per") and after == "month") or before == "then"
        if run and (priced or not prices_only):
            runs.append(" ".join(run))
        run = []
    return runs


def invented_prices(reply: str, catalogue: str) -> list[str]:
    known = set(number_runs(catalogue, prices_only=True))
    return [p for p in number_runs(reply, prices_only=True) if p not in known] + DIGIT_PRICE.findall(reply)
