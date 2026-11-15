# Claim Pre-Check Agent v2.0 (RAG) & Rx Adjudication API

[![tests](https://github.com/raghavanand776/claim-precheck-agent-v2/actions/workflows/tests.yml/badge.svg)](https://github.com/raghavanand776/claim-precheck-agent-v2/actions/workflows/tests.yml)

A learning project exploring where AI helps, and where it doesn't, in healthcare claims.

**v2.0 adds retrieval over payer policy documents (RAG), with every citation checked in code.** v1 lives on unchanged at [claim-precheck-agent](https://github.com/raghavanand776/claim-precheck-agent).

It has two parts:

1. **Claim pre-check agent** – reviews therapy insurance claims *before* submission and flags what would get them denied.
2. **Rx adjudication API** – a small model of how a pharmacy benefit manager approves or rejects a prescription in real time.

All data is synthetic. No real patient information is used, and the payers and policies are invented.

---

## Part 1: Claim pre-check agent

**The problem:** a claim with a wrong procedure code, a missing telehealth modifier, or inactive coverage gets denied, and the provider waits weeks to fix and resubmit it. Many of these errors can be caught before submission.

### v1: tools and session notes

Claude (Haiku) gets four checks as tools (codes vs. session length, payer telehealth modifier, eligibility, reading the claim), decides which to call, and explains the fix. It scored **7/7** on the v1 eval, including problems that only appear in free-text session notes. Plain rules scored 1/3 on those.

### v2.0: payer policies with retrieval (RAG)

**The problem with v1:** payer rules lived in a one-line JSON file that only knew one payer. Real billing rules live in long policy documents that differ by payer. A rule that isn't hand-copied into code doesn't exist for the agent.

**What changed:**

- **Policy library** (`data/policies/`): billing policies for three synthetic payers, split into 21 sections with stable IDs such as `EGM-TELEHEALTH`. The payers deliberately disagree: modifier 95 vs. GT for video, phone sessions covered vs. not covered, prior authorization after 20 sessions, after 12, or for every 90837.
- **Retriever** (`retriever.py`): BM25 keyword ranking in plain Python, limited to the claim's payer. No extra dependencies, no API key, and the same answer every run, so it's tested in CI.
- **New tool:** `search_payer_policy` replaces the one-line rules lookup. The agent must cite the section behind every policy finding, e.g. `[EGM-TELEHEALTH]`.
- **Citations are checked in code, not trusted:** after the agent answers, every cited section is compared with the sections it actually retrieved in that run. A citation to a section it never retrieved, or one that doesn't exist, is flagged as ungrounded.
- **Seven new claims** (008–014), each turning on a rule found only in a policy document, plus two clean claims to catch false flags.

### Results (14 claims)

| Version | Correct (verdict + fix) | Grounded citations |
| --- | --- | --- |
| Rules only (`check_claims.py`) | 7/14 (verdicts only) | – |
| v1 agent (`python3 agent.py --v1`) | 9/14 | – |
| v2 agent, first run | 13/14 | 14/14 |
| v2 agent, final run (`python3 agent.py`) | **14/14** | **14/14** |

v1 got all of the original 7 claims right but missed every claim whose rule exists only in a policy document. On claim_011 it read the note correctly (a phone-only session) but couldn't know the payer doesn't cover phone sessions, so it suggested a fix instead of flagging a denial.

These are single runs on a small synthetic set I wrote. Results vary between runs; the first v2 run scored 13/14. Full output: [`eval_results_v1_mode.txt`](eval_results_v1_mode.txt), [`eval_results_v2_run1.txt`](eval_results_v2_run1.txt), [`eval_results_v2.txt`](eval_results_v2.txt).

### How it's tested

| What | How | Needs API key? |
| --- | --- | --- |
| Retrieval eval | 12 labeled questions, worded the way a note or a person would say them; the right section must be in the top 3. **12/12**, 11/12 ranked first | No (`pytest`) |
| Citation checks | A scripted fake model cites a section it never retrieved, and one that doesn't exist; both must be flagged | No (`pytest`) |
| Guardrails | Tests fail if the citation rules are removed from the prompt, if v1 and v2 differ in more than the payer tool, or if the eligibility fix is removed | No (`pytest`) |
| Agent eval | 14 labeled claims: right verdict, right fix, and (v2) the right section cited with nothing ungrounded | Yes (`python3 agent.py`) |

22 automated tests in total, run on every push with GitHub Actions.

### Lessons

- **Keyword search fails on wording.** "We did a phone call" found nothing, because the policy says "telephone". A test search caught it before the agent ever saw it; a short synonym list fixed it. The test questions and the synonyms were both written by me, so this set flatters the retriever.
- **Test retrieval first.** If the right section isn't retrieved, no prompt can make the agent cite it.
- **The eval caught a bug a demo would have missed.** On the first live run, the agent asked for eligibility before reading the claim and passed the claim ID instead of the member ID, so coverage came back "not active" and the verdict flipped to a denial. The policy reasoning was right; one sentence in the tool description fixed it (3/3 runs after).
- **Fix the cause, not the test.** The first full run missed claim_014: the policy says such claims "are denied", so the agent called a fixable problem a denial. Claim_008's policy uses the same wording and the agent got it right, so the answer key was consistent and the verdict rule was unclear. I clarified the rule (fixable before submission = FIX NEEDED) instead of changing the answer key, then checked that claim_011 (a truly uncovered service) was still a denial.
- **"Cite your source" in a prompt is a request; comparing citations with what was retrieved is a check.**

---

## Part 2: Rx adjudication API

At the pharmacy counter, a claim is decided in seconds. `rx_adjudicator.py` runs the checks in order and stops at the first failure:

| Check | Reject code if it fails |
| --- | --- |
| Patient covered? | 65 |
| Drug on the plan's formulary? | 70 |
| Prior authorization on file (if required)? | 75 |
| Too soon to refill (before 75% of last supply)? | 79 |
| Approved → patient pays the tier copay, never more than the drug's price | – |

`rx_api.py` puts it behind a **FastAPI** service:

- `POST /adjudicate` – send a claim, get approved (with amounts) or rejected (with code and reason)
- `GET /health` – for monitoring
- Malformed requests are rejected with **422** before reaching the logic

**Load test** (`load_test.py`, 1,000 sequential claims, one laptop, no network): **~1.3 ms average and p95.** Not a production number – the point is measuring p95, because the slow tail is the customer waiting at the counter. It's also why AI stays off the real-time path here.

### Sample request and response

```bash
curl -s -X POST localhost:8000/adjudicate \
  -H "Content-Type: application/json" \
  -d '{"claim_id":"rx_001","member_id":"M200","drug":"atorvastatin 20mg","fill_date":"2026-10-31"}'
```

```json
{"status": "APPROVED", "price": 12.0, "patient_pays": 10, "plan_pays": 2.0}
```

A rejected claim returns a standard reject code and a plain-language reason:

```json
{"status": "REJECTED", "reject_code": "79", "message": "Refill too soon: 10 days since last fill, earliest is day 23"}
```

Results for all 7 sample claims are in [`rx/sample_results.md`](rx/sample_results.md).

---

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 check_claims.py          # rules version, scored
python3 -m pytest -v             # 22 tests, no API key needed
python3 retriever.py "Evergreen Mutual" phone session   # try a policy search
python3 rx_adjudicator.py        # pharmacy adjudicator on sample claims

uvicorn rx_api:app --reload      # start the API, then open http://127.0.0.1:8000/docs
python3 load_test.py             # in a second terminal, while the API runs
```

The agent needs an Anthropic API key in a `.env` file:

```
ANTHROPIC_API_KEY=your-key-here
```

```bash
python3 agent.py                 # v2: all claims, with score
python3 agent.py claim_008       # v2: one claim
python3 agent.py --v1            # v1, for comparison
```

---

## What I'd do next

- Run each eval several times and report the spread, not a single run
- Grow the eval set from real denial patterns, with test questions written by someone else
- Compare BM25 with embedding search once the policy library is large enough for wording to drift
- Track policy versions by effective date, so a claim is checked against the policy in force on its date of service
- Add a code check that rejects a claim ID passed as a member ID, instead of relying on the tool description
- Run rules first on every claim, and send only claims that need judgment to the model, to control cost and latency

Built with Python, the Claude API (tool use), BM25 retrieval, FastAPI and pytest.

## License

MIT. See [LICENSE](LICENSE).