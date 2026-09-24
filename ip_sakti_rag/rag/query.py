"""Multi-query expansion and CRAG query rewriting (Groq fast model)."""
from __future__ import annotations

TOPICS = ["patents", "biodiversity_abs", "drugs", "advertising", "food", "cosmetics",
          "trademarks", "traditional_knowledge", "general"]

EXPAND_SYSTEM = """You turn a user's question about intellectual property and regulatory law for Ayurveda products into search queries for a legal document database.
Return JSON only:
{"queries": ["..."], "jurisdiction": "IN" | "INTL" | "BOTH", "topic": "<one topic>"}
Rules:
- Write exactly {n} different search queries in English, each under 20 words.
- Use precise legal vocabulary (Act names, section numbers if implied, legal concepts like "prior informed consent", "not inventions", "benefit sharing").
- One query should be a plain restatement; the others should cover different angles or related provisions.
- jurisdiction: "IN" for Indian law, "INTL" for foreign or international regimes (WIPO, Nagoya, EPO, USPTO), "BOTH" if both are asked.
- topic must be one of: {topics}.
- Do not answer the question."""

REWRITE_SYSTEM = """The search queries below failed to find relevant legal provisions for the user's question.
Write ONE better search query in English (under 25 words) using different legal terms, likely Act or section names, or broader wording.
Return JSON only: {"query": "..."}"""


def expand_query(llm, question_en: str, n: int = 3, product_category: str | None = None) -> dict:
    context = f"\nProduct category: {product_category}" if product_category else ""
    data = llm.json(EXPAND_SYSTEM.replace("{n}", str(n)).replace("{topics}", ", ".join(TOPICS)),
                    f"Question: {question_en}{context}", default={}) or {}
    queries = [question_en]
    for q in data.get("queries", []):
        q = str(q).strip()
        if q and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)
    jur = str(data.get("jurisdiction", "IN")).upper()
    topic = data.get("topic") if data.get("topic") in TOPICS else "general"
    return {"queries": queries[: n + 1], "jurisdiction": jur if jur in ("IN", "INTL", "BOTH") else "IN",
            "topic": topic}


def rewrite_query(llm, question_en: str, tried: list[str]) -> str:
    data = llm.json(REWRITE_SYSTEM, f"Question: {question_en}\nFailed queries:\n- " + "\n- ".join(tried),
                    default={}) or {}
    q = str(data.get("query", "")).strip()
    return q or question_en
