# claim-precheck-agent-v2 / test_retriever.py
# Author:  Raghav Anand
# Created: 2026-11-14
# Version: 2.0
#
# Purpose: Automated tests for the policy search (retriever.py) and the
#          answer key (eval_cases.py). No API key needed: the search is plain
#          Python, so these run free, in under a second, on every push
#          (GitHub Actions).
#
# Run: python3 -m pytest -v test_retriever.py
#
# How pytest works: it finds every function whose name starts with "test_"
# and runs it. Each "assert" is a claim that must be true; if any assert is
# false, that test fails and pytest shows which line and why.

from pathlib import Path                 # builds file paths that work on any computer

from eval_cases import CASES, RETRIEVAL_CASES
from retriever import known_payers, load_index, search_policy, section_ids


# ---------------------------------------------------------------------------
# The policy library loads correctly
# ---------------------------------------------------------------------------
def test_every_section_has_an_id_payer_and_text():
    """All policy files were split into sections, and none is missing a part."""
    sections = load_index()[0]
    assert len(sections) >= 20                       # 3 payers x 7 sections = 21
    for s in sections:
        assert s["section_id"] and s["payer"] and s["text"]


def test_section_ids_are_unique():
    """No two sections share an ID, so every citation points to one place."""
    sections = load_index()[0]
    ids = [s["section_id"] for s in sections]
    assert len(ids) == len(set(ids))                 # a set drops duplicates


# ---------------------------------------------------------------------------
# The search behaves safely
# ---------------------------------------------------------------------------
def test_search_only_returns_the_requested_payer():
    """One payer's rule can never come back for another payer's claim."""
    for payer in known_payers():
        for hit in search_policy(payer, "telehealth modifier prior authorization phone"):
            assert hit["payer"] == payer


def test_unknown_payer_is_an_error_not_an_empty_list():
    """An empty list could be misread as 'no rule applies', so it must be an error."""
    result = search_policy("Acme Insurance", "telehealth")
    assert "error" in result


def test_results_are_best_first():
    """Scores come back highest first."""
    hits = search_policy("Evergreen Mutual", "video telehealth modifier")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# The retrieval eval: does the search find the right section?
# ---------------------------------------------------------------------------
def test_retrieval_eval_recall_at_3():
    """Every labeled question in eval_cases.py finds its section in the top 3.

    This is the same check you ran by hand in Step 5d, now permanent.
    "Recall at 3" = how often the right answer is somewhere in the top 3.
    """
    misses = []
    for payer, query, want in RETRIEVAL_CASES:
        ids = [h["section_id"] for h in search_policy(payer, query)]
        if want not in ids:
            misses.append((payer, query, want, ids))
    assert misses == []          # if not, pytest prints exactly which ones missed


def test_phone_matches_telephone():
    """The miss we found in Step 3: fails if the synonym list is ever removed."""
    hits = search_policy("Evergreen Mutual", "internet down so we did a phone call")
    assert hits and hits[0]["section_id"] == "EGM-AUDIO-ONLY"


# ---------------------------------------------------------------------------
# The answer key itself is valid
# ---------------------------------------------------------------------------
def test_answer_key_cites_real_sections():
    """Every must_cite in eval_cases.py is a real section (catches typos in the key)."""
    for claim_id, case in CASES.items():
        if "must_cite" in case:
            assert case["must_cite"] in section_ids(), claim_id


def test_answer_key_covers_every_claim_file():
    """Every claim file has an answer, and every answer has a claim file."""
    folder = Path(__file__).parent / "data" / "claims"
    assert sorted(p.stem for p in folder.glob("*.json")) == sorted(CASES)