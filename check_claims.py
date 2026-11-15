# claim-precheck-agent-v2 / check_claims.py
# Author:  Raghav Anand
# Created: 2026-10-31 (v1)
# Updated: 2026-11-14 (v2.0: scored against eval_cases.py)
#
# Purpose: The rules-only baseline. Plain Python checks, no AI at all.
#          It's the "before" number: what you get with hand-written rules
#          alone, so we can measure what the agent adds.
#
# What it checks (the same three things as v1):
#   1. Codes: does the procedure code fit the session length? (validate_codes)
#   2. Payer rules: video visits need the payer's modifier (payer_rules.json,
#      which only knows BlueSky Health)
#   3. Coverage: is the member active? (check_eligibility)
#
# What it can't do: read session notes, or know any rule that isn't
# hand-copied into payer_rules.json. That's what the agent is for.
#
# How to run (no API key needed):
#   python3 check_claims.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from pathlib import Path                 # builds file paths that work on any computer

from eval_cases import CASES             # v2: the answer key, for scoring
from tools import check_eligibility, lookup_payer_rules, read_claim, validate_codes


# ---------------------------------------------------------------------------
# Check one claim with plain rules
# ---------------------------------------------------------------------------
def precheck(claim_id):
    """Return (verdict, reasons) for one claim, using rules only."""
    claim = read_claim(claim_id)

    # 1. Code problems (wrong code for the minutes, missing diagnosis).
    problems = validate_codes(claim)

    # 2. Payer rule: video visits need the payer's modifier.
    #    lookup_payer_rules returns {} for every payer except BlueSky Health,
    #    so for Evergreen and Cascade this check silently does nothing.
    rules = lookup_payer_rules(claim["payer"])
    tag = rules.get("telehealth_modifier")
    if claim["telehealth"] and tag and tag not in claim["modifiers"]:
        problems.append(f"Video session needs modifier {tag}")

    # 3. Coverage: inactive coverage outranks everything else.
    member = check_eligibility(claim["member_id"])
    if not member["active"]:
        return "LIKELY DENIAL", ["Patient's coverage is not active"] + problems

    # Any other problem means the claim can be fixed before submitting.
    if problems:
        return "FIX NEEDED", problems
    return "READY", []


# ---------------------------------------------------------------------------
# Run every claim and score the verdicts against eval_cases.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    folder = Path(__file__).parent / "data" / "claims"

    # Every claim file, in order: claim_001 ... claim_014.
    claims = sorted(path.stem for path in folder.glob("*.json"))

    correct = 0
    for claim_id in claims:
        verdict, reasons = precheck(claim_id)
        expected = CASES[claim_id]["verdict"]

        # Count it if the verdict matches the answer key.
        # (Rules can't name fixes found only in policy text, so only
        # verdicts are scored here, not fixes or citations.)
        correct += verdict == expected

        # Show the expected verdict next to any miss.
        mark = "" if verdict == expected else f"   (expected {expected})"
        print(f"{claim_id}: {verdict}{mark}")
        for reason in reasons:
            print(f"    - {reason}")

    print(f"\nVerdicts correct: {correct}/{len(claims)}")