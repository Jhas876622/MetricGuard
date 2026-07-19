"""
MetricGuard - Report Generator
================================
Orchestrates the full pipeline and writes:
  - output/results.json  (machine-readable, full detail)
  - output/data.js       (dashboard entry-point: const MG_DATA = {...})

Also prints headline KPIs to stdout.

Usage:
    python src/report.py
    python src/report.py --no-llm    # skip LLM calls (faster, offline)
"""

import json
import logging
import sys
from pathlib import Path

from engine import run_analysis, load_metrics
from genai import load_glossary, resolve_conflict

logger = logging.getLogger(__name__)

OUT = Path(__file__).parent.parent / "output"


def main(use_llm: bool = True) -> dict:
    logger.info("MetricGuard report starting (use_llm=%s)", use_llm)
    OUT.mkdir(exist_ok=True)

    metrics, results = run_analysis()
    glossary = load_glossary()
    logger.info("Pipeline produced %d conflict groups from %d metrics",
                len(results), len(metrics))

    enriched = []
    for idx, r in enumerate(results):
        logger.info("Enriching conflict group %d/%d: %s",
                    idx + 1, len(results), r["names"])
        item = {
            "names":        r["names"],
            "teams":        r["teams"],
            "metric_ids":   r["metric_ids"],
            "conflicts":    r["conflicts"],
            "trust_risk":   r["trust_risk"],
            "avg_similarity": round(r["avg_similarity"], 3),
            "definitions": [
                {
                    "team":        m["team"],
                    "name":        m["metric_name"],
                    "description": m["description"],
                    "sql":         m["sql"],
                }
                for m in r["metrics"]
            ],
        }
        if use_llm:
            try:
                res = resolve_conflict(r, glossary)
                item["recommended_concept"]  = res["concept"]
                item["glossary_owner"]       = res["retrieved_glossary"]["owner"]
                item["retrieval_similarity"] = res["retrieval_similarity"]
                item["recommendation"]       = res["recommendation"]
            except Exception as exc:
                logger.error("resolve_conflict failed for group %s: %s", r["names"], exc)
                item["recommendation"] = f"[resolution error: {type(exc).__name__}]"
        enriched.append(item)

    # -----------------------------------------------------------------------
    # Headline KPIs
    # -----------------------------------------------------------------------
    total_defs      = len(metrics)
    conflicting_defs = sum(len(r["metric_ids"]) for r in results)
    teams_affected  = len({t for r in results for t in r["teams"]})
    kpis = {
        "total_definitions_scanned":    total_defs,
        "conflicting_definitions_found": conflicting_defs,
        "conflict_groups":              len(results),
        "teams_affected":               teams_affected,
        "pct_definitions_in_conflict":  round(100 * conflicting_defs / total_defs, 1)
                                        if total_defs else 0.0,
        "highest_trust_risk":           max((r["trust_risk"] for r in results), default=0),
    }

    payload = {"kpis": kpis, "conflicts": enriched}

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    results_path = OUT / "results.json"
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", results_path)

    # dashboard.html loads data.js as a <script> tag — must stay in sync
    data_js_path = OUT / "data.js"
    with open(data_js_path, "w") as f:
        f.write("const MG_DATA = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
    logger.info("Wrote %s  ← dashboard entry-point", data_js_path)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\nHEADLINE KPIs")
    print("-" * 40)
    for k, v in kpis.items():
        print(f"  {k:<38} {v}")
    print(f"\nWrote {results_path}")
    print(f"Wrote {data_js_path}  ← open output/dashboard.html in a browser")

    return payload


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    use_llm = "--no-llm" not in sys.argv
    main(use_llm=use_llm)
