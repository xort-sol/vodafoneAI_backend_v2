"""Between rehearsals: make the allowlisted numbers callable again, as if never rung. Touches no other lead.

Usage (from the repo root): uv run python scripts/reset_demo.py, with the same DIAL_ALLOWLIST the demo uses.
"""

import os

from dotenv import load_dotenv

from dialler.main import parse_allowlist
from shared import db


def main() -> None:
    load_dotenv()
    numbers = parse_allowlist(os.getenv("DIAL_ALLOWLIST"))
    if numbers is None or "*" in numbers:
        raise SystemExit('DIAL_ALLOWLIST is "*": give the exact numbers to reset')
    print(f"Reset {db.reset_demo_leads(numbers)} leads for {sorted(numbers)}")


if __name__ == "__main__":
    main()
