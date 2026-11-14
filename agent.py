# claim-precheck-agent-v2 / agent.py
# Author:  Raghav Anand
# Created: 2026-10-31 (v1)
# Updated: 2026-11-14 (v2.0: policy search, cited findings, citation checks)
#
# Purpose: The claim pre-check agent. Claude reads a claim, calls tools to
#          check it, and returns a verdict (READY / FIX NEEDED / LIKELY DENIAL)
#          with the exact fixes.
#
# Two modes, so v1 and v2 can be compared on the same claims:
#   v2 (default) - Claude searches each payer's policy documents (RAG) and
#                  must cite the policy section behind every policy finding.
#                  Code then checks every citation: a section Claude cites but
#                  never retrieved is flagged as "ungrounded".
#   v1 (--v1)    - the original agent, with only the one-line payer rules lookup.
#
# How to run (needs ANTHROPIC_API_KEY in .env):
#   python3 agent.py                 # v2 on all 14 claims, with score
#   python3 agent.py claim_008       # v2 on one claim
#   python3 agent.py --v1            # v1 on all claims, for comparison
#
# File layout:
#   1. Imports and settings
#   2. The tool menu (what Claude is told each tool does)
#   3. The job description (system prompts for v1 and v2)
#   4. Citation checking (code, not the AI, decides if a citation is real)
#   5. The switchboard (turns Claude's tool request into a real function call)
#   6. The agent loop (ask Claude, run tools, repeat until it answers)
#   7. Grading against the answer key in eval_cases.py
#   8. Running it from Terminal

# ===========================================================================
# 1. Imports and settings
# ===========================================================================
import json                      # converts between Python data and JSON text
import re                        # regular expressions: finds [SECTION-ID] citations
import sys                       # reads what you typed after "python3 agent.py"

from dotenv import load_dotenv   # loads ANTHROPIC_API_KEY from the .env file

from eval_cases import CASES     # the answer key (Step 5d)
from retriever import section_ids  # every real policy section ID (Step 3)
from tools import (              # the tools Claude can call (Step 4)
    check_eligibility,
    lookup_payer_rules,
    read_claim,
    search_payer_policy,
    validate_codes,
)

load_dotenv()                    # read .env so the Anthropic client finds the key
MODEL = "claude-haiku-4-5-20251001"   # same model as v1


# ===========================================================================
# 2. The tool menu
# ===========================================================================
# Claude never sees our Python code. It only sees these descriptions: a name,
# what the tool does, and what inputs it needs (input_schema). Clear
# descriptions are how Claude knows which tool to call and when.

# Several tools take just a claim ID, so that input shape is defined once.
CLAIM_ID = {
    "type": "object",
    "properties": {"claim_id": {"type": "string"}},
    "required": ["claim_id"],
}

READ_CLAIM = {
    "name": "read_claim",
    "description": (
        "Read one claim. Returns its codes, minutes, payer, member ID, modifiers, "
        "place of service, sessions this year, prior authorization number and session note."
    ),
    "input_schema": CLAIM_ID,
}

VALIDATE_CODES = {
    "name": "validate_codes",
    "description": (
        "Check the claim's procedure code fits the session length and a diagnosis "
        "code is present. Returns a list of problems; empty means fine."
    ),
    "input_schema": CLAIM_ID,
}

CHECK_ELIGIBILITY = {
    "name": "check_eligibility",
       "description": (
        "Check whether a member's insurance coverage is active. Needs the member_id "
        "from the claim (e.g. 'M105'), not the claim ID, so call read_claim first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"member_id": {"type": "string"}},
        "required": ["member_id"],
    },
}

# v1 only: the one-line rules lookup (knows BlueSky Health only).
LOOKUP_PAYER_RULES = {
    "name": "lookup_payer_rules",
    "description": (
        "Get an insurance company's billing rules, such as the modifier required "
        "on video (telehealth) sessions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"payer": {"type": "string"}},
        "required": ["payer"],
    },
}

# v2 only: search the payer's policy documents.
SEARCH_PAYER_POLICY = {
    "name": "search_payer_policy",
    "description": (
        "Search one payer's billing policy documents. Returns the 3 most relevant "
        "sections, each with a section_id to cite, its title and its text. Use the "
        "payer name exactly as it appears on the claim. Ask one topic per search, "
        "e.g. 'video telehealth modifier and place of service', 'audio-only phone "
        "session', 'prior authorization session limit', 'rules for 90837'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "payer": {"type": "string"},
            "query": {"type": "string"},
        },
        "required": ["payer", "query"],
    },
}

# Which tools each mode gets. The only difference: v1 has the rules lookup,
# v2 has the policy search.
TOOLS = {
    "v1": [READ_CLAIM, VALIDATE_CODES, LOOKUP_PAYER_RULES, CHECK_ELIGIBILITY],
    "v2": [READ_CLAIM, VALIDATE_CODES, SEARCH_PAYER_POLICY, CHECK_ELIGIBILITY],
}


# ===========================================================================
# 3. The job description (system prompts)
# ===========================================================================
# v1's prompt, unchanged, so the comparison is fair.
SYSTEM_V1 = """You pre-check therapy insurance claims before a human submits them.
Use the tools to check the codes, the payer's rules and the member's coverage. Never guess a fact you can look up.
Always read the session_note. If it contradicts the claim's fields (for example, a different session length, or a video visit marked as in-person), trust the note.
Every reason must say exactly what to change, including the correct code or modifier. Code ranges: 90832 = 16-37 min, 90834 = 38-52 min, 90837 = 53+ min.
Verdict rules: coverage not active = LIKELY DENIAL. Any other problem = FIX NEEDED. No problems = READY.
When done, reply with only JSON: {"verdict": "...", "reasons": ["..."]}"""

# v2's prompt. What's new compared with v1:
#   - search the policy for every rule that could apply
#   - cite the section_id behind every policy finding, in [SQUARE-BRACKETS]
#   - only cite sections actually returned by the search; never guess a rule
#   - a service the policy says isn't covered = LIKELY DENIAL
# The code ranges were removed: they now come from validate_codes and the policy.
SYSTEM_V2 = """You pre-check therapy insurance claims before a human submits them.
Use the tools to check the codes, the member's coverage and the payer's policy. Never guess a fact you can look up.
Always read the session_note. If it contradicts the claim's fields (for example, a different session length, or a video visit marked as in-person), trust the note.

Payer policy:
- Search the payer's policy for every rule that could apply to this claim: how the session was delivered (in office, video or phone: modifier and place of service), prior authorization or session limits, and any rule specific to the procedure code.
- Base every policy finding on a section you retrieved, and cite its section_id in square brackets at the end of the reason, e.g. [BSH-TELEHEALTH].
- Only cite section IDs that search_payer_policy returned in this conversation. If no retrieved section covers a question, do not guess a rule; say the policy did not cover it.

Every reason must say exactly what to change: the correct code, modifier, place of service, or missing number.
Verdict rules: coverage not active, or the payer's policy says the service is not covered = LIKELY DENIAL. Any other problem = FIX NEEDED. No problems = READY.
When done, reply with only JSON: {"verdict": "...", "reasons": ["..."]}"""

SYSTEM = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


# ===========================================================================
# 4. Citation checking
# ===========================================================================
# A citation looks like [EGM-TELEHEALTH]: an opening bracket, capital letters,
# a dash, then capitals/digits/dashes, and a closing bracket. This pattern
# finds every one in the agent's reasons.
CITATION = re.compile(r"\[([A-Z]+-[A-Z0-9-]+)\]")


def retrieved_ids(output):
    """The section IDs in one search_payer_policy result.

    A successful search returns a list of sections; an error (unknown payer)
    returns a dict, which contains no sections.
    """
    if isinstance(output, list):
        return {hit["section_id"] for hit in output}
    return set()


def check_citations(reasons, retrieved):
    """Code, not the model, decides whether each citation is grounded.

    reasons   - the agent's list of reasons (text)
    retrieved - every section ID the search returned during this run

    Returns (cited, ungrounded):
      cited      - every section ID the agent cited
      ungrounded - cited IDs the agent never retrieved, or that don't exist
    """
    # Every [SECTION-ID] in every reason.
    cited = {cid for reason in reasons for cid in CITATION.findall(reason)}

    # A citation is ungrounded if it wasn't retrieved in this run,
    # or isn't a real section at all (made up).
    ungrounded = sorted(
        cid for cid in cited
        if cid not in retrieved or cid not in section_ids()
    )
    return sorted(cited), ungrounded


# ===========================================================================
# 5. The switchboard
# ===========================================================================
def run_tool(name, args):
    """Claude asks for a tool by name; this calls the matching Python function."""
    if name == "read_claim":
        return read_claim(args["claim_id"])
    if name == "validate_codes":
        return validate_codes(read_claim(args["claim_id"]))
    if name == "lookup_payer_rules":
        return lookup_payer_rules(args["payer"])
    if name == "search_payer_policy":
        return search_payer_policy(args["payer"], args["query"])
    if name == "check_eligibility":
        return check_eligibility(args["member_id"])
    # Claude asked for a tool that doesn't exist: tell it, don't crash.
    return {"error": f"unknown tool {name}"}


# ===========================================================================
# 6. The agent loop
# ===========================================================================
def precheck(claim_id, mode="v2", client=None, verbose=True):
    """Run the agent on one claim and return its answer.

    claim_id - e.g. "claim_008"
    mode     - "v2" (policy search) or "v1" (rules lookup)
    client   - normally None (a real Anthropic client is created). The tests
               pass in a fake client so they can run without an API key.
    verbose  - print each tool call as it happens

    Returns {"verdict", "reasons", "citations", "ungrounded"}.
    """
    import anthropic   # imported here so the tests can run without calling the API

    client = client or anthropic.Anthropic()

    # The conversation so far. It starts with one message from us.
    messages = [{"role": "user", "content": f"Pre-check claim {claim_id}."}]

    # Every section ID the search returns during this run, for citation checking.
    retrieved = set()

    # At most 12 rounds of tool calls, so a confused agent can't loop forever.
    for _ in range(12):
        # Send the conversation, the job description and the tool menu to Claude.
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM[mode],
            tools=TOOLS[mode],
            messages=messages,
        )
        # Add Claude's reply to the conversation.
        messages.append({"role": "assistant", "content": response.content})

        # If Claude didn't ask for a tool, it's done: read its final answer.
        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            # Pull out the JSON between the first { and the last }.
            result = json.loads(text[text.find("{"): text.rfind("}") + 1])
            # Check the citations in code before returning.
            result["citations"], result["ungrounded"] = check_citations(result["reasons"], retrieved)
            return result

        # Otherwise, run every tool Claude asked for and collect the results.
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"    Claude calls {block.name}({block.input})")
                output = run_tool(block.name, block.input)

                # Remember which sections the search returned.
                if block.name == "search_payer_policy":
                    retrieved |= retrieved_ids(output)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,          # links the result to Claude's request
                    "content": json.dumps(output),    # tool output, as JSON text
                })

        # Send the tool results back to Claude as the next message.
        messages.append({"role": "user", "content": results})

    raise RuntimeError("Agent took too many steps")


# ===========================================================================
# 7. Grading against the answer key
# ===========================================================================
def grade(claim_id, result, mode):
    """Compare one answer with eval_cases.py. Returns (correct, grounded, notes).

    correct  - right verdict AND the reasons name the right fix (both modes)
    grounded - v2 only: cites the expected section and makes no ungrounded
               citation. None in v1 mode, since v1 can't cite anything.
    notes    - what went wrong, in words
    """
    case = CASES[claim_id]
    notes = []
    text = " ".join(result["reasons"])   # all reasons as one string, for searching

    # Right verdict?
    correct = result["verdict"] == case["verdict"]
    if not correct:
        notes.append(f"expected {case['verdict']}")

    # Right fix? At least one of the must_mention words must appear (any case).
    needed = case.get("must_mention")
    if needed and not any(word.lower() in text.lower() for word in needed):
        correct = False
        notes.append(f"reasons don't mention {' / '.join(needed)}")

    # v1 doesn't cite sections, so there's nothing more to grade.
    if mode == "v1":
        return correct, None, notes

    # v2: no made-up or un-retrieved citations...
    grounded = not result["ungrounded"]
    if result["ungrounded"]:
        notes.append(f"ungrounded citation {', '.join(result['ungrounded'])}")

    # ...and the expected section must be among the citations.
    want = case.get("must_cite")
    if want and want not in result["citations"]:
        grounded = False
        notes.append(f"doesn't cite {want}")

    return correct, grounded, notes


# ===========================================================================
# 8. Running it from Terminal
# ===========================================================================
if __name__ == "__main__":
    args = sys.argv[1:]                               # what you typed after agent.py
    mode = "v1" if "--v1" in args else "v2"           # --v1 switches to v1 mode
    claim_ids = [a for a in args if not a.startswith("--")] or list(CASES)   # named claims, or all 14

    correct_total = grounded_total = 0
    for claim_id in claim_ids:
        print(f"\n{claim_id}")
        result = precheck(claim_id, mode)
        correct, grounded, notes = grade(claim_id, result, mode)
        correct_total += correct
        grounded_total += bool(grounded)

        # One status line per claim, then its reasons and any grading notes.
        status = "correct" if correct else "WRONG"
        if mode == "v2":
            status += ", grounded" if grounded else ", NOT grounded"
        print(f"  Verdict: {result['verdict']}  ({status})")
        for reason in result["reasons"]:
            print(f"    - {reason}")
        for note in notes:
            print(f"  ! {note}")

    # The scorecard.
    print(f"\nMode: {mode}")
    print(f"Correct (verdict + fix): {correct_total}/{len(claim_ids)}")
    if mode == "v2":
        print(f"Grounded (right citation, none invented): {grounded_total}/{len(claim_ids)}")