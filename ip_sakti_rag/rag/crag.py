"""Corrective RAG.

retrieve -> rerank -> grade each passage with the LLM
  correct    : enough relevant passages          -> use them
  ambiguous  : only partial matches              -> rewrite query, retry, merge
  incorrect  : nothing relevant                  -> rewrite query, retry
After retries: optional web search on official domains only, else refuse/escalate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .query import rewrite_query

log = logging.getLogger(__name__)

GRADER_SYSTEM = """You are a strict relevance grader for a legal search system.
For each passage decide whether it helps answer the question.
- "relevant": states the rule, requirement, exclusion or definition the question needs.
- "partial": related background, but does not answer on its own.
- "irrelevant": does not help.
Return JSON only: {"grades": [{"id": "P1", "grade": "relevant"}, ...]} covering every passage."""


@dataclass
class CragResult:
    status: str                               # correct | ambiguous | incorrect
    passages: list[dict] = field(default_factory=list)
    queries_tried: list[str] = field(default_factory=list)
    attempts: int = 0
    used_web: bool = False


def grade_passages(llm, question: str, passages: list[dict]) -> dict[str, str]:
    if not passages:
        return {}
    blocks = [f"[P{i}] {p['text'][:900]}" for i, p in enumerate(passages, start=1)]
    data = llm.json(GRADER_SYSTEM, f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(blocks),
                    max_tokens=600, default={}) or {}
    grades = {}
    for g in data.get("grades", []):
        pid, grade = str(g.get("id", "")).strip(), str(g.get("grade", "")).lower().strip()
        if pid.startswith("P") and grade in ("relevant", "partial", "irrelevant"):
            grades[pid] = grade
    # Fallback for anything the grader skipped: trust the reranker
    for i, p in enumerate(passages, start=1):
        pid = f"P{i}"
        if pid not in grades:
            s = p.get("rerank_score", 0)
            grades[pid] = "relevant" if s >= 0.7 else "partial" if s >= 0.35 else "irrelevant"
    return grades


def decide(passages: list[dict]) -> str:
    rel = [p for p in passages if p.get("grade") == "relevant"]
    par = [p for p in passages if p.get("grade") == "partial"]
    top = max((p.get("rerank_score", 0) for p in rel), default=0)
    if len(rel) >= 2 or (rel and top >= 0.6):
        return "correct"
    if rel or par:
        return "ambiguous"
    return "incorrect"


def web_fallback(question: str, api_key: str, domains: tuple, k: int = 4) -> list[dict]:
    """Search official sites only (Tavily). Returns passages shaped like retrieval results."""
    try:
        from tavily import TavilyClient
    except ImportError:
        log.warning("tavily-python not installed; web fallback skipped")
        return []
    try:
        res = TavilyClient(api_key=api_key).search(query=question, include_domains=list(domains), max_results=k)
    except Exception as exc:
        log.warning("Web fallback failed: %s", exc)
        return []
    out = []
    for i, r in enumerate(res.get("results", [])):
        url = r.get("url", "")
        intl = any(d in url for d in ("wipo.int", "cbd.int", "epo.org", "uspto.gov"))
        out.append({
            "id": f"web::{i}", "text": r.get("content", ""), "rerank_score": float(r.get("score", 0.5)),
            "grade": "partial",
            "metadata": {"title": r.get("title", url), "short_title": r.get("title", url)[:70],
                         "source_url": url, "jurisdiction": "INTL" if intl else "IN", "section_label": "",
                         "pages": "", "is_current": True, "source_type": "web"},
        })
    return out


class CorrectiveRAG:
    def __init__(self, llm, retriever, reranker, rs):
        self.llm, self.retriever, self.reranker, self.rs = llm, retriever, reranker, rs

    def _retrieve(self, question: str, queries: list[str], jurisdiction: str | None) -> list[dict]:
        cands = self.retriever.search(queries, jurisdiction, k_each=self.rs.k_each, top_n=self.rs.fused_top_n)
        top = self.reranker.rerank(question, cands, top_k=self.rs.rerank_top_k)
        grades = grade_passages(self.llm, question, top)
        for i, p in enumerate(top, start=1):
            p["grade"] = grades.get(f"P{i}", "irrelevant")
        return top

    def run(self, question: str, queries: list[str], jurisdiction: str | None) -> CragResult:
        pool: dict[str, dict] = {}
        tried = list(queries)
        status = "incorrect"
        attempts = 0
        for attempt in range(self.rs.crag_max_retries + 1):
            attempts += 1
            for p in self._retrieve(question, queries, jurisdiction):
                old = pool.get(p["id"])
                if not old or p.get("rerank_score", 0) > old.get("rerank_score", 0):
                    pool[p["id"]] = p
            status = decide(list(pool.values()))
            log.info("CRAG attempt %d: %s", attempt + 1, status)
            if status == "correct" or attempt == self.rs.crag_max_retries:
                break
            new_q = rewrite_query(self.llm, question, tried)
            tried.append(new_q)
            queries = [new_q, question]

        used_web = False
        if status == "incorrect" and self.rs.web_fallback and self.rs.tavily_api_key:
            web = web_fallback(question, self.rs.tavily_api_key, self.rs.official_domains)
            if web:
                grades = grade_passages(self.llm, question, web)
                for i, p in enumerate(web, start=1):
                    p["grade"] = grades.get(f"P{i}", "irrelevant")
                web = [p for p in web if p["grade"] != "irrelevant"]
                if web:
                    used_web, status = True, "ambiguous"
                    for p in web:
                        pool[p["id"]] = p

        order = {"relevant": 0, "partial": 1, "irrelevant": 2}
        kept = sorted((p for p in pool.values() if p.get("grade") in ("relevant", "partial")),
                      key=lambda p: (order[p["grade"]], -p.get("rerank_score", 0)))
        return CragResult(status=status, passages=kept, queries_tried=tried, attempts=attempts, used_web=used_web)
