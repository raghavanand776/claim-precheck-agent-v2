# claim-precheck-agent-v2 / retriever.py
# Author:  Raghav Anand
# Created: 2026-11-13
# Version: 2.0
#
# Purpose: Search the payer policy documents in data/policies/ and return the
#          sections that best answer a question. This is the "R" (retrieval)
#          in RAG: the agent asks a question, this file finds the policy text,
#          and the agent bases its answer on that text and cites it.
#
# How it works, in short:
#   1. Each policy file is split into sections at every "## SECTION-ID Title"
#      heading, so every search result has an ID the agent can cite.
#   2. Sections are ranked with BM25, a standard keyword-ranking method used
#      by search engines. It's written out in plain Python below.
#   3. Only the claim's own payer is searched, so one payer's rule can never
#      be applied to another payer's claim.
#
# Why keyword search (BM25) and not AI "embeddings"?
#   - No extra libraries and no API key needed
#   - Same answer every run, so it can be tested automatically (GitHub Actions)
#   - Easy to see why a section ranked where it did
#
# Try it from Terminal:
#   python3 retriever.py "Evergreen Mutual" video telehealth modifier

# ---------------------------------------------------------------------------
# Imports: all built into Python, nothing new to install
# ---------------------------------------------------------------------------
import math                          # math.log, used in the BM25 rarity weight
import re                            # regular expressions: finding words and headings
from collections import Counter      # counts how many times each word appears
from functools import lru_cache      # remembers a function's result so it runs only once
from pathlib import Path             # builds file paths that work on any computer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Where the policy files live: the folder this file is in, then data/policies.
POLICY_DIR = Path(__file__).parent / "data" / "policies"

# Filler words we ignore. They appear in almost every section, so matching
# them adds noise instead of meaning.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "if", "in", "is", "it", "its", "of", "on", "or", "the", "their", "this",
    "to", "what", "when", "which", "with", "without", "per", "any", "all", "must",
}

# BM25's two settings, set to the standard defaults.
# K1: how quickly repeated words stop adding points (the 5th mention of a
#     word counts less than the 1st).
# B:  how much to adjust for section length (0 = no adjustment, 1 = full).
K1 = 1.5
B = 0.75


# ---------------------------------------------------------------------------
# Step 1: turn text into search words
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase words and codes ("90837", "95", "gt"), minus filler words.

    Example: "What modifier for a video visit?" -> ["modifier", "video", "visit"]
    """
    # Lowercase everything so "GT" and "gt" match, then pull out runs of
    # letters and digits. This keeps codes like 90837 whole and drops punctuation.
    words = re.findall(r"[a-z0-9]+", text.lower())

    # Drop the filler words.
    return [w for w in words if w not in STOPWORDS]

# ---------------------------------------------------------------------------
# Step 1b: add policy wording for everyday words (query expansion)
# ---------------------------------------------------------------------------
# Keyword search only matches exact words. A session note says "phone", but
# the policies say "telephone", so the search finds nothing. This short list
# maps everyday words to the words the policies actually use.
# Added after a test search ("we did a phone call") came back empty.
SYNONYMS = {
    "phone": ["telephone"],               # notes say phone, policies say telephone
    "call": ["telephone"],                # "a phone call"
    "zoom": ["video"],                    # people name the app, policies say video
    "pa": ["prior", "authorization"],     # common abbreviation for prior authorization
    "auth": ["authorization"],            # short for authorization
}


def expand(terms):
    """Add the policy's wording for any everyday words in the query.

    Example: ["phone", "session"] -> ["phone", "session", "telephone"]
    Only queries are expanded, never the policy text itself.
    """
    expanded = list(terms)                      # start with the original words
    for term in terms:
        expanded.extend(SYNONYMS.get(term, []))  # add synonyms, if this word has any
    return expanded


# ---------------------------------------------------------------------------
# Step 2: split a policy file into sections
# ---------------------------------------------------------------------------
def parse_policy(path):
    """Split one policy file into sections: {section_id, payer, title, text}."""
    # Read the whole file as one string.
    raw = Path(path).read_text()

    # The file starts with a header between two "---" lines (payer, title,
    # effective date). Cut at the first two "---": front = header, body = the rest.
    front, body = raw.split("---", 2)[1:]

    # Turn header lines like "payer: BlueSky Health" into a lookup,
    # so meta["payer"] gives "BlueSky Health".
    meta = dict(line.split(": ", 1) for line in front.strip().splitlines())

    sections = []
    # Cut the body wherever a line starts with "## ". [1:] skips the piece
    # before the first heading (the "Synthetic policy" note).
    for block in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
        # First line = heading, everything after it = the rule text.
        heading, _, text = block.partition("\n")

        # Split the heading at its first space:
        # "BSH-TELEHEALTH Telehealth by live video" -> ID + title.
        section_id, _, title = heading.partition(" ")

        sections.append({
            "section_id": section_id.strip(),   # e.g. BSH-TELEHEALTH (what the agent cites)
            "payer": meta["payer"],              # e.g. BlueSky Health
            "title": title.strip(),              # e.g. Telehealth by live video
            "text": text.strip(),                # the rule itself
        })
    return sections


# ---------------------------------------------------------------------------
# Step 3: read every policy once and prepare it for searching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)   # run once, then reuse the result instead of re-reading files
def load_index():
    """Read every policy once and pre-compute what BM25 needs."""
    # Collect the sections from every .md file in the policies folder.
    # sorted() keeps the order the same on every computer.
    sections = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        sections.extend(parse_policy(path))

    # Each section's searchable words. The title is included, so a heading
    # like "Telehealth by live video" counts as content too.
    docs = [tokenize(s["title"] + " " + s["text"]) for s in sections]

    # In how many sections does each word appear? A word in 1 section is rare
    # (strong evidence); a word in 20 sections is common (weak evidence).
    doc_freq = Counter(term for doc in docs for term in set(doc))

    # Average section length in words, used to adjust for long sections.
    avg_len = sum(len(d) for d in docs) / len(docs)

    return sections, docs, doc_freq, avg_len


# ---------------------------------------------------------------------------
# Step 4: score one section against the query (BM25)
# ---------------------------------------------------------------------------
def bm25(query_terms, doc, doc_freq, n_docs, avg_len):
    """Score one section for a query. Higher = better match.

    Three ideas, all in the formula below:
      1. More mentions of a query word = higher score, with diminishing returns
      2. Rare words count more than common ones
      3. Long sections don't win just for being long
    """
    counts = Counter(doc)   # how many times each word appears in this section
    score = 0.0

    for term in query_terms:
        if term not in counts:
            continue        # word not in this section: adds nothing

        # Idea 2, rarity weight ("IDF"): big for rare words, small for common ones.
        idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))

        # Idea 1, term frequency: how often the word appears in this section.
        tf = counts[term]

        # Combine them. tf / (tf + K1 * ...) makes repeats count less and less;
        # len(doc) / avg_len is the length adjustment (idea 3).
        score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(doc) / avg_len))

    return score


# ---------------------------------------------------------------------------
# Helper: which payers have a policy on file?
# ---------------------------------------------------------------------------
def known_payers():
    """Sorted list of payer names, e.g. ['BlueSky Health', 'Cascade Care', ...]."""
    sections = load_index()[0]
    return sorted({s["payer"] for s in sections})


# ---------------------------------------------------------------------------
# Step 5: the search the agent will call
# ---------------------------------------------------------------------------
def search_policy(payer, query, k=3):
    """Top-k policy sections for one payer, best match first.

    Only that payer's sections are searched, so one payer's rule can never be
    cited for another payer's claim. An unknown payer returns an error rather
    than an empty list, so the agent knows the lookup failed.
    """
    sections, docs, doc_freq, avg_len = load_index()

    # Unknown payer: say so clearly, and list the payers we do have.
    # (An empty list could be misread as "no rule applies".)
    if payer not in known_payers():
        return {"error": f"No policy on file for payer '{payer}'", "known_payers": known_payers()}

    # Break the question into search words, then add policy wording
    # for everyday words (e.g. "phone" -> "telephone").
    terms = expand(tokenize(query))
    
    scored = []
    for section, doc in zip(sections, docs):
        # Search only this payer's sections.
        if section["payer"] != payer:
            continue

        score = bm25(terms, doc, doc_freq, len(docs), avg_len)

        # Keep only sections that matched at least one search word.
        if score > 0:
            scored.append((score, section))

    # Highest score first.
    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Return the top k, each with its score rounded to 2 decimals.
    return [dict(section, score=round(score, 2)) for score, section in scored[:k]]


# ---------------------------------------------------------------------------
# Helper: every valid section ID (used later to catch made-up citations)
# ---------------------------------------------------------------------------
def section_ids():
    """Every citable section ID in the library, e.g. {'BSH-TELEHEALTH', ...}."""
    return {s["section_id"] for s in load_index()[0]}


# ---------------------------------------------------------------------------
# Try it by hand. Runs only when you run this file directly:
#   python3 retriever.py "Evergreen Mutual" video telehealth modifier
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # First word after the file name = payer; the rest = the question.
    # With nothing typed, it uses an example search.
    payer = sys.argv[1] if len(sys.argv) > 1 else "BlueSky Health"
    query = " ".join(sys.argv[2:]) or "video telehealth modifier"

    results = search_policy(payer, query)

    if isinstance(results, dict):       # unknown payer
        print(results["error"])
    elif not results:                   # nothing matched
        print("No matching sections.")
    else:                               # one line per result: score, ID, title
        for hit in results:
            print(f"{hit['score']:>6}  {hit['section_id']:<20} {hit['title']}")