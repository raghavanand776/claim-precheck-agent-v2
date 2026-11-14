# claim-precheck-agent-v2 / tools.py
# Author:  Raghav Anand
# Created: 2026-10-31 (v1)   <- replace with the date from step 4a
# Updated: 2026-11-13 (v2.0: added search_payer_policy)
#
# Purpose: The tools the agent can call. Each one is plain Python that looks
#          something up or checks something, so the facts come from code and
#          data files, not from the AI's memory.
#
#   read_claim           - load one claim from data/claims/
#   validate_codes       - does the procedure code fit the session length?
#   lookup_payer_rules   - v1 only: one-line rules from payer_rules.json
#   check_eligibility    - is the member's coverage active?
#   search_payer_policy  - v2: search the payer's policy documents (RAG)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json                              # reads the .json data files
from pathlib import Path                 # builds file paths that work on any computer

from retriever import search_policy      # v2: the policy search from retriever.py

# Where the data files live: the folder this file is in, then data/.
DATA = Path(__file__).parent / "data"

# Which minutes each therapy code covers: (lowest, highest)
CPT_MINUTES = {
    "90832": (16, 37),     # 16-37 minutes
    "90834": (38, 52),     # 38-52 minutes
    "90837": (53, 999),    # 53 minutes or more (999 = no real upper limit)
}


# ---------------------------------------------------------------------------
# Tool: read one claim
# ---------------------------------------------------------------------------
def read_claim(claim_id):
    """Load a claim, e.g. read_claim("claim_001") reads data/claims/claim_001.json."""
    with open(DATA / "claims" / f"{claim_id}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool: check the codes on a claim
# ---------------------------------------------------------------------------
def validate_codes(claim):
    """Return a list of code problems. An empty list means the codes are fine."""
    problems = []
    code = claim["cpt_code"]                # e.g. "90834"
    minutes = claim["session_minutes"]      # e.g. 45

    if code not in CPT_MINUTES:
        # A code we don't know about at all.
        problems.append(f"Unknown code {code}")
    else:
        low, high = CPT_MINUTES[code]
        if not (low <= minutes <= high):
            # The code doesn't match the session length: find the code that does,
            # so the problem message can say exactly what to change.
            right = [c for c, (lo, hi) in CPT_MINUTES.items() if lo <= minutes <= hi]
            tip = f"; use {right[0]}" if right else ""
            problems.append(f"Code {code} doesn't fit a {minutes}-minute session{tip}")

    # Every claim needs a diagnosis code.
    if not claim.get("diagnosis_code"):
        problems.append("Missing diagnosis code")

    return problems


# ---------------------------------------------------------------------------
# Tool (v1 only): one-line payer rules
# ---------------------------------------------------------------------------
def lookup_payer_rules(payer):
    """v1's rules lookup. Kept so v1 can still run for comparison.

    Returns {} for any payer not in payer_rules.json, which is every payer
    except BlueSky Health. That gap is the reason v2 exists.
    """
    with open(DATA / "payer_rules.json") as f:
        return json.load(f).get(payer, {})


# ---------------------------------------------------------------------------
# Tool: is the member covered?
# ---------------------------------------------------------------------------
def check_eligibility(member_id):
    """Return {"active": True/False}. An unknown member counts as not covered."""
    with open(DATA / "eligibility.json") as f:
        return json.load(f).get(member_id, {"active": False})


# ---------------------------------------------------------------------------
# Tool (v2): search the payer's policy documents
# ---------------------------------------------------------------------------
def search_payer_policy(payer, query):
    """Return the 3 policy sections that best answer the query, for one payer.

    Each result has a section_id the agent must cite, e.g. EGM-TELEHEALTH.
    The search itself lives in retriever.py; this is the agent-facing wrapper.
    """
    return search_policy(payer, query, k=3)