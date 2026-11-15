# claim-precheck-agent-v2 / test_agent.py
# Author:  Raghav Anand
# Created: 2026-11-14
# Version: 2.0
#
# Purpose: Automated tests for agent.py that run WITHOUT an API key.
#          A scripted "fake Claude" stands in for the real model, so these
#          tests check the code around the model: citation checking, grading
#          and the prompt's guardrails. They don't test the model's judgment;
#          that's what the eval (python3 agent.py) is for.
#
# Run: python3 -m pytest -v test_agent.py
#
# Why a fake Claude: real Claude costs money, needs a key, and gives slightly
# different answers each run. A fake one gives the exact answer we script,
# every time, so the tests are free, fast and repeatable.

import json
from types import SimpleNamespace        # makes simple objects with named fields

import agent


# ---------------------------------------------------------------------------
# Building blocks for the fake Claude
# ---------------------------------------------------------------------------
# Real Claude replies with "blocks": a tool request, or text. These two
# helpers build look-alike blocks with just the fields agent.py reads.

def tool_use(name, args, id_="t1"):
    """A fake 'Claude wants to call this tool' block."""
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


def text(payload):
    """A fake 'Claude's final answer' block, containing JSON."""
    return SimpleNamespace(type="text", text=json.dumps(payload))


class FakeClient:
    """Stands in for anthropic.Anthropic(). Replays scripted replies in order.

    Each reply is (stop_reason, blocks):
      ("tool_use", [...])  - Claude asks for tools; the agent loop runs them
      ("end_turn", [...])  - Claude gives its final answer
    """

    def __init__(self, responses):
        self.responses = list(responses)
        # agent.py calls client.messages.create(...), so provide that shape.
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        stop, blocks = self.responses.pop(0)     # next scripted reply
        return SimpleNamespace(stop_reason=stop, content=blocks)


def run(responses, claim_id="claim_008"):
    """Run the real agent loop with the fake Claude."""
    return agent.precheck(claim_id, "v2", client=FakeClient(responses), verbose=False)


# ---------------------------------------------------------------------------
# Citation checking
# ---------------------------------------------------------------------------
def test_citation_of_a_retrieved_section_is_grounded():
    """Fake Claude searches, then cites what it found: grounded and correct."""
    result = run([
        ("tool_use", [tool_use("search_payer_policy",
                               {"payer": "Evergreen Mutual", "query": "video telehealth modifier"})]),
        ("end_turn", [text({"verdict": "FIX NEEDED",
                            "reasons": ["Replace modifier 95 with GT [EGM-TELEHEALTH]"]})]),
    ])
    assert result["citations"] == ["EGM-TELEHEALTH"]
    assert result["ungrounded"] == []
    assert agent.grade("claim_008", result, "v2") == (True, True, [])


def test_citation_never_retrieved_is_flagged():
    """Fake Claude cites a REAL section it never searched for: caught in code."""
    result = run([
        # It searches for something else ("office visit")...
        ("tool_use", [tool_use("search_payer_policy",
                               {"payer": "Evergreen Mutual", "query": "office visit"})]),
        # ...then cites the telehealth section anyway.
        ("end_turn", [text({"verdict": "FIX NEEDED",
                            "reasons": ["Replace modifier 95 with GT [EGM-TELEHEALTH]"]})]),
    ])
    # Confirm the "office visit" search really didn't return EGM-TELEHEALTH.
    first_search = agent.search_payer_policy("Evergreen Mutual", "office visit")
    assert "EGM-TELEHEALTH" not in {h["section_id"] for h in first_search}
    # So the citation must be flagged: the answer is correct but not grounded.
    assert result["ungrounded"] == ["EGM-TELEHEALTH"]
    correct, grounded, _ = agent.grade("claim_008", result, "v2")
    assert correct and not grounded


def test_invented_section_is_flagged():
    """Fake Claude cites a section that doesn't exist: flagged."""
    result = run([
        ("end_turn", [text({"verdict": "FIX NEEDED", "reasons": ["Use GT [EGM-MADE-UP-RULE]"]})]),
    ])
    assert result["ungrounded"] == ["EGM-MADE-UP-RULE"]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def test_right_verdict_without_the_fix_is_wrong():
    """A vague answer with the right verdict still fails: it must name the fix (GT)."""
    result = run([
        ("end_turn", [text({"verdict": "FIX NEEDED", "reasons": ["Something is off with the modifier"]})]),
    ])
    correct, grounded, notes = agent.grade("claim_008", result, "v2")
    assert not correct and not grounded
    assert any("GT" in n for n in notes)


# ---------------------------------------------------------------------------
# Guardrails: fail loudly if someone weakens the setup
# ---------------------------------------------------------------------------
def test_system_prompt_keeps_its_guardrails():
    """Fails if the citation rules are ever deleted from the v2 prompt."""
    prompt = agent.SYSTEM_V2
    assert "cite its section_id" in prompt
    assert "Only cite section IDs that search_payer_policy returned" in prompt
    assert "do not guess a rule" in prompt


def test_v1_and_v2_tool_menus():
    """v1 and v2 differ only in the payer tool, so the comparison is fair."""
    v1 = {t["name"] for t in agent.TOOLS["v1"]}
    v2 = {t["name"] for t in agent.TOOLS["v2"]}
    assert "lookup_payer_rules" in v1 and "search_payer_policy" not in v1
    assert "search_payer_policy" in v2 and "lookup_payer_rules" not in v2
    assert v1 - {"lookup_payer_rules"} == v2 - {"search_payer_policy"}


def test_eligibility_tool_says_to_use_the_member_id():
    """The fix from the claim_008 bug: fails if that sentence is removed."""
    description = next(t for t in agent.TOOLS["v2"] if t["name"] == "check_eligibility")["description"]
    assert "not the claim ID" in description