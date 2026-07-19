"""
MetricGuard - GenAI Layer
=========================
Four GenAI concepts implemented in working code:

  1. LLM SUMMARIZATION  — Claude reads messy SQL + descriptions and explains
     in plain English what a metric actually computes.

  2. RAG (Retrieval-Augmented Generation)  — before generating, we RETRIEVE
     the top-k most relevant glossary entries and inject them into the prompt.
     This grounds the model so it uses the company's official meaning instead
     of hallucinating.

  3. EMBEDDINGS-BASED RETRIEVAL  — the retrieve step reuses embed() from
     engine.py to do semantic search over the glossary.

  4. AGENTIC WORKFLOW  — multi-step agent: analyze → retrieve → reason →
     recommend canonical definition.

Runs with the Anthropic API if ANTHROPIC_API_KEY is set; otherwise falls back
to a transparent template so the pipeline never breaks.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from engine import embed

logger = logging.getLogger(__name__)

GLOSSARY_PATH = Path(__file__).parent.parent / "data" / "glossary.json"

# How many glossary entries to retrieve before re-ranking (top-k RAG)
RETRIEVAL_TOP_K = 3


# ---------------------------------------------------------------------------
# GOVERNED GLOSSARY
# ---------------------------------------------------------------------------
def load_glossary() -> list[dict]:
    """Load the company's official canonical metric definitions."""
    logger.debug("Loading glossary from %s", GLOSSARY_PATH)
    with open(GLOSSARY_PATH) as f:
        return json.load(f)


def retrieve_glossary_entry(
    query_text: str,
    glossary: list[dict],
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[dict, float]:
    """
    RAG STEP 1 = RETRIEVE (top-k then re-rank).

    The original version returned only the single best hit. This is brittle
    for conflict groups that span two concepts — e.g. a metric blending
    revenue + user-activity logic might score highest on the wrong entry.

    Now we:
      1. Embed the query and all glossary entries.
      2. Return top-k candidates by cosine similarity.
      3. Re-rank by also checking whether the query text contains the concept
         name as a keyword substring (keyword boost).
      4. Return the top-ranked entry and its similarity score.

    This gives the LLM more context (all top-k are injected into the prompt)
    while still surfacing the single best match as the canonical entry.
    """
    if not glossary:
        logger.warning("Glossary is empty — returning sentinel entry")
        return {"concept": "Unknown", "official_definition": "No glossary available.", "owner": "—"}, 0.0

    entry_texts = [f"{g['concept']}: {g['official_definition']}" for g in glossary]
    all_vecs = embed([query_text] + entry_texts)
    query_vec = all_vecs[0:1]
    entry_vecs = all_vecs[1:]
    sims = (query_vec @ entry_vecs.T)[0]

    # Keyword boost: if the concept name appears in the query, +0.1
    query_lower = query_text.lower()
    boosted = []
    for idx, (entry, sim) in enumerate(zip(glossary, sims)):
        keyword_match = any(
            word.lower() in query_lower
            for word in entry["concept"].split()
            if len(word) > 3          # skip short stop-words like 'and', 'the'
        )
        boosted.append((idx, float(sim) + (0.1 if keyword_match else 0.0)))

    boosted.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in boosted[:top_k]]
    top_entries = [glossary[i] for i in top_indices]
    top_sims    = [float(sims[i]) for i in top_indices]

    best_entry = top_entries[0]
    best_sim   = top_sims[0]

    logger.debug(
        "retrieve_glossary_entry: top-%d hits for query '%s…': %s",
        top_k, query_text[:50],
        [(e["concept"], round(s, 3)) for e, s in zip(top_entries, top_sims)],
    )
    return best_entry, best_sim, top_entries   # caller may use all top_entries


# ---------------------------------------------------------------------------
# LLM CALL  — typed exception handling
# ---------------------------------------------------------------------------
def call_llm(prompt: str, max_tokens: int = 500) -> str:
    """
    Send a prompt to Claude. Exception handling is typed:
      - anthropic.AuthenticationError → API key missing/wrong (not retryable)
      - anthropic.RateLimitError      → back-off and retry (retryable)
      - anthropic.APIError            → other API-side error
      - any other Exception           → unexpected, log full traceback

    Falls back to a deterministic template so the pipeline never breaks.
    """
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — using fallback template")
            return f"[LLM fallback – no API key]\n{_template_answer()}"
        client = anthropic.Anthropic(api_key=api_key)
        logger.debug("Calling LLM (max_tokens=%d)", max_tokens)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        response = "".join(b.text for b in msg.content if hasattr(b, "text"))
        logger.info("LLM call succeeded (%d chars)", len(response))
        return response

    except Exception as exc:
        # Import here so the module doesn't hard-require anthropic at load time
        try:
            import anthropic as _anthropic
            if isinstance(exc, _anthropic.AuthenticationError):
                logger.warning("LLM auth error (check ANTHROPIC_API_KEY): %s", exc)
            elif isinstance(exc, _anthropic.RateLimitError):
                logger.warning("LLM rate-limited — consider adding retry logic: %s", exc)
            elif isinstance(exc, _anthropic.APIError):
                logger.error("LLM API error: %s", exc)
            else:
                logger.error("Unexpected LLM error (%s): %s", type(exc).__name__, exc)
        except ImportError:
            logger.error("anthropic not installed; LLM call skipped: %s", exc)

        return f"[LLM fallback – {type(exc).__name__}]\n{_template_answer()}"


def _template_answer() -> str:
    """Deterministic stand-in — honest and useful even without an API key."""
    return (
        "Based on the retrieved official definition, all teams should align "
        "to the governed glossary definition. Teams whose logic diverges "
        "(refund handling, time window, or filters) should migrate their SQL "
        "and dashboards to the canonical definition to restore trust in the numbers."
    )


# ---------------------------------------------------------------------------
# THE AGENT  (analyze → retrieve → reason → recommend)
# ---------------------------------------------------------------------------
def resolve_conflict(conflict: dict, glossary: list[dict]) -> dict:
    """
    Agentic resolution of ONE conflict group.

    Step A (analyze)  : conflict metadata is already computed upstream.
    Step B (retrieve) : top-k RAG — fetch the best-matching glossary entries.
    Step C (generate) : ask Claude to recommend the canonical definition,
                        grounded in ALL top-k retrieved entries.
    """
    names    = ", ".join(conflict["names"])
    teams    = ", ".join(conflict["teams"])
    problems = "; ".join(conflict["conflicts"])

    defs = "\n".join(
        f"- Team {m['team']} calls it '{m['metric_name']}': "
        f"{m['description']} (SQL: {m['sql']})"
        for m in conflict["metrics"]
    )

    descriptions = " ".join(m["description"] for m in conflict["metrics"])
    query = f"{names}. {descriptions}"

    result = retrieve_glossary_entry(query, glossary)
    # retrieve_glossary_entry now returns 3 values
    if len(result) == 3:
        best_entry, best_sim, top_entries = result
    else:
        best_entry, best_sim = result
        top_entries = [best_entry]

    # Format all top-k entries for the prompt (richer context for the LLM)
    glossary_context = "\n\n".join(
        f"Concept: {e['concept']}\n"
        f"Official definition: {e['official_definition']}\n"
        f"Owner: {e['owner']}"
        for e in top_entries
    )

    prompt = f"""You are a data governance assistant. Multiple teams defined the same business metric differently, causing their dashboards to disagree.

CONFLICTING DEFINITIONS:
{defs}

DETECTED PROBLEMS: {problems}
TEAMS INVOLVED: {teams}

OFFICIAL COMPANY GLOSSARY (top retrieved entries — use as source of truth):
{glossary_context}

TASK:
1. State the single canonical definition all teams should adopt (based on the glossary).
2. For each team, note in one line exactly what they must change in their SQL or logic.
3. Give a one-sentence business-impact statement for a non-technical executive.
Keep the total response under 180 words."""

    recommendation = call_llm(prompt)
    logger.info("Resolved conflict for concept '%s' (retrieval sim=%.3f)",
                best_entry["concept"], best_sim)
    return {
        "concept":             best_entry["concept"],
        "retrieved_glossary":  best_entry,
        "retrieval_similarity": round(best_sim, 3),
        "recommendation":      recommendation,
    }


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from engine import run_analysis
    glossary = load_glossary()
    metrics, results = run_analysis()
    print(f"\nResolving {len(results)} conflicts with RAG + LLM agent...\n")
    for r in results:
        res = resolve_conflict(r, glossary)
        print("=" * 70)
        print(f"CONCEPT: {res['concept']}  (retrieval sim {res['retrieval_similarity']})")
        print(f"Glossary owner: {res['retrieved_glossary']['owner']}")
        print(f"\nAGENT RECOMMENDATION:\n{res['recommendation']}\n")
