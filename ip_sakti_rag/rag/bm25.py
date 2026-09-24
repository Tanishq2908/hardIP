"""Keyword (BM25) index built from whatever is in ChromaDB.

Because it is rebuilt from the collection, it always matches the vector store:
no separate files to keep in sync. It rebuilds automatically when the
collection size changes.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the this
to was were will with which shall may any such under within other than their there these those
been being can do does if not no so also""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


class BM25Index:
    def __init__(self, store):
        self.store = store
        self._size = -1
        self.bm25 = None
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metas: list[dict] = []

    def invalidate(self) -> None:
        self._size = -1

    def _ensure(self) -> None:
        size = self.store.count()
        if size == self._size and self.bm25 is not None:
            return
        data = self.store.get_all(where={"is_current": True})
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if self.docs else None
        self._size = size

    def search(self, query: str, k: int = 20, jurisdiction: str | None = None) -> list[dict]:
        self._ensure()
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        out = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] <= 0:
                break
            meta = self.metas[i]
            if jurisdiction and meta.get("jurisdiction") != jurisdiction:
                continue
            out.append({"id": self.ids[i], "text": self.docs[i], "metadata": meta, "bm25_score": float(scores[i])})
            if len(out) >= k:
                break
        return out
