"""Hybrid retrieval: dense (ChromaDB) + keyword (BM25), fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from .bm25 import BM25Index
from .vectorstore import VectorStore, build_where


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Each list is ranked best-first. Score = sum over lists of 1 / (k + rank)."""
    fused: dict[str, dict] = {}
    for lst in result_lists:
        for rank, item in enumerate(lst, start=1):
            entry = fused.setdefault(item["id"], {**item, "rrf_score": 0.0, "hits": 0})
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["hits"] += 1
            for key in ("dense_score", "bm25_score"):
                if key in item:
                    entry[key] = max(entry.get(key, float("-inf")), item[key])
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, bm25: BM25Index | None = None):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Index(store)

    def search(self, queries: list[str], jurisdiction: str | None = None, k_each: int = 20,
               top_n: int = 25, extra_where: dict | None = None) -> list[dict]:
        jur = None if not jurisdiction or jurisdiction == "BOTH" else jurisdiction
        where = build_where(jurisdiction=jur, extra=extra_where)
        lists: list[list[dict]] = []
        for q in queries:
            lists.append(self.store.dense_search(self.embedder.embed_query(q), k=k_each, where=where))
            lists.append(self.bm25.search(q, k=k_each, jurisdiction=jur))
        return reciprocal_rank_fusion([lst for lst in lists if lst])[:top_n]
