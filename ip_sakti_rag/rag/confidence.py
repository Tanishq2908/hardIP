"""Confidence score (0..1) from retrieval quality, CRAG outcome and citation support."""
from __future__ import annotations

CRAG_VALUE = {"correct": 1.0, "ambiguous": 0.6, "incorrect": 0.0}


def compute_confidence(passages_used: list[dict], crag_status: str, verdicts: list[str],
                       high: float = 0.75, medium: float = 0.55) -> dict:
    top = sorted((p.get("rerank_score", 0.0) for p in passages_used), reverse=True)[:3]
    retrieval = sum(top) / len(top) if top else 0.0
    crag = CRAG_VALUE.get(crag_status, 0.0)
    if verdicts:
        support = (verdicts.count("supported") + 0.5 * verdicts.count("partial")) / len(verdicts)
    else:
        support = 0.0
    all_current = all(p["metadata"].get("is_current", True) and p["metadata"].get("source_type") != "web"
                      for p in passages_used) if passages_used else False
    currency = 1.0 if all_current else 0.7

    score = 0.35 * retrieval + 0.25 * crag + 0.30 * support + 0.10 * currency
    level = "high" if score >= high else "medium" if score >= medium else "low"
    return {
        "score": round(score, 3), "level": level,
        "components": {"retrieval": round(retrieval, 3), "crag": crag,
                       "citation_support": round(support, 3), "currency": currency},
    }
