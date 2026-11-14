# claim-precheck-agent-v2 / eval_cases.py
# Author:  Raghav Anand
# Created: 2026-11-14
# Version: 2.0
#
# Purpose: The answer key for evaluating v2. Kept in its own file so the
#          agent's scoring (agent.py) and the automated tests (test_*.py)
#          share one source of truth.
#
# Two answer keys live here:
#   CASES            - what the agent should conclude for each claim
#   RETRIEVAL_CASES  - what the search should find for each test question
#
# Testing retrieval separately matters: if the right policy section isn't
# found, no prompt can make the agent cite it. So we check the search first.

# ---------------------------------------------------------------------------
# Answer key 1: the agent's verdict for each claim
# ---------------------------------------------------------------------------
# For each claim:
#   verdict      - the expected verdict: READY, FIX NEEDED or LIKELY DENIAL
#   must_mention - the fix: AT LEAST ONE of these words must appear in the
#                  agent's reasons (so a right verdict with a vague or wrong
#                  fix still counts as wrong)
#   must_cite    - v2 only: the policy section the agent must cite, e.g.
#                  [EGM-TELEHEALTH]. Only set where a policy rule is the
#                  reason for the finding.
# A claim with only a verdict (e.g. READY) has no fix to check.

CASES = {
    # ---- v1 claims (001-007) ----
    "claim_001": {"verdict": "READY"},                                    # clean in-office claim
    "claim_002": {"verdict": "FIX NEEDED", "must_mention": ["90834"]},    # 40 min billed as 90837
    "claim_003": {"verdict": "FIX NEEDED", "must_mention": ["95"],        # video, no modifier
                  "must_cite": "BSH-TELEHEALTH"},
    "claim_004": {"verdict": "LIKELY DENIAL"},                            # coverage inactive
    "claim_005": {"verdict": "FIX NEEDED", "must_mention": ["95"],        # note: video from home
                  "must_cite": "BSH-TELEHEALTH"},
    "claim_006": {"verdict": "FIX NEEDED", "must_mention": ["90832"]},    # note: only 25 minutes
    "claim_007": {"verdict": "READY"},                                    # clean, with a note

    # ---- v2 claims (008-014): rules that exist only in policy documents ----
    "claim_008": {"verdict": "FIX NEEDED", "must_mention": ["GT"],        # Evergreen wants GT, not 95
                  "must_cite": "EGM-TELEHEALTH"},
    "claim_009": {"verdict": "FIX NEEDED", "must_mention": ["10"],        # video from home = POS 10
                  "must_cite": "BSH-TELEHEALTH"},
    "claim_010": {"verdict": "FIX NEEDED", "must_mention": ["authorization"],  # session 14, no auth number
                  "must_cite": "CCP-PRIOR-AUTH"},
    "claim_011": {"verdict": "LIKELY DENIAL",                             # phone session: not covered
                  "must_mention": ["audio", "phone", "telephone"],
                  "must_cite": "EGM-AUDIO-ONLY"},
    "claim_012": {"verdict": "READY"},                                    # clean: catches over-flagging
    "claim_013": {"verdict": "READY"},                                    # session 22 WITH auth number
    "claim_014": {"verdict": "FIX NEEDED", "must_mention": ["authorization"],  # 90837 needs auth
                  "must_cite": "EGM-90837-AUTH"},
}

# ---------------------------------------------------------------------------
# Answer key 2: the search, tested on its own
# ---------------------------------------------------------------------------
# Each entry: (payer, question, the section that must appear in the top 3).
# The questions are worded the way a session note or a person would say
# them, not copied from the policy text, so they test real-world wording.
# Note: I wrote both these questions and the synonym list in retriever.py,
# so this set flatters the search. A real test set should be written by
# someone else.

RETRIEVAL_CASES = [
    # BlueSky Health
    ("BlueSky Health", "video visit modifier", "BSH-TELEHEALTH"),
    ("BlueSky Health", "client joined from home place of service", "BSH-TELEHEALTH"),
    ("BlueSky Health", "do we need prior authorization for session 22", "BSH-PRIOR-AUTH"),
    ("BlueSky Health", "session by phone", "BSH-AUDIO-ONLY"),
    ("BlueSky Health", "which code for a 25 minute session", "BSH-TIME-CODES"),

    # Evergreen Mutual
    ("Evergreen Mutual", "video visit modifier", "EGM-TELEHEALTH"),
    ("Evergreen Mutual", "internet down so we did a phone call", "EGM-AUDIO-ONLY"),
    ("Evergreen Mutual", "is auth needed for 90837", "EGM-90837-AUTH"),
    ("Evergreen Mutual", "office visit place of service", "EGM-IN-OFFICE"),

    # Cascade Care
    ("Cascade Care", "session 14 this year authorization", "CCP-PRIOR-AUTH"),
    ("Cascade Care", "phone session no video", "CCP-AUDIO-ONLY"),
    ("Cascade Care", "how many days to submit a claim", "CCP-TIMELY-FILING"),
]