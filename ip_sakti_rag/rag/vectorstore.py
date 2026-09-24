"""ChromaDB wrapper.

Versioning model
- Every chunk carries doc_id, version (file hash) and is_current.
- Re-ingesting a changed file adds the new version and marks the old
  chunks is_current=False (they are kept for "what was the law on date X").
- Retrieval filters on is_current=True by default.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import chromadb

log = logging.getLogger(__name__)
BATCH = 256


def build_where(*, jurisdiction: str | None = None, current_only: bool = True,
                extra: dict | None = None) -> dict | None:
    """Combine simple equality filters into ChromaDB's where syntax."""
    clauses: list[dict] = []
    if current_only:
        clauses.append({"is_current": True})
    if jurisdiction:
        clauses.append({"jurisdiction": jurisdiction.upper()})
    for k, v in (extra or {}).items():
        clauses.append({k: v})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

    # ---------------------------------------------------------------- write
    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict],
               embeddings: list[list[float]]) -> None:
        for i in range(0, len(ids), BATCH):
            self.collection.upsert(ids=ids[i:i + BATCH], documents=documents[i:i + BATCH],
                                   metadatas=metadatas[i:i + BATCH], embeddings=embeddings[i:i + BATCH])

    def current_version(self, doc_id: str) -> str | None:
        res = self.collection.get(where=build_where(extra={"doc_id": doc_id}), limit=1, include=["metadatas"])
        return res["metadatas"][0].get("version") if res["ids"] else None

    def retire_old_versions(self, doc_id: str, keep_version: str) -> int:
        """Mark every current chunk of doc_id that is not keep_version as superseded."""
        res = self.collection.get(where=build_where(extra={"doc_id": doc_id}), include=["metadatas"])
        stale = [(i, m) for i, m in zip(res["ids"], res["metadatas"]) if m.get("version") != keep_version]
        if not stale:
            return 0
        stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        ids = [i for i, _ in stale]
        metas = [{**m, "is_current": False, "superseded_at": stamp, "superseded_by": keep_version}
                 for _, m in stale]
        for i in range(0, len(ids), BATCH):
            self.collection.update(ids=ids[i:i + BATCH], metadatas=metas[i:i + BATCH])
        return len(ids)

    def update_metadatas(self, ids: list[str], metadatas: list[dict]) -> None:
        for i in range(0, len(ids), BATCH):
            self.collection.update(ids=ids[i:i + BATCH], metadatas=metadatas[i:i + BATCH])

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def reset(self) -> None:
        name, meta = self.collection.name, self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name, metadata=meta)

    # ---------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def dense_search(self, query_embedding: list[float], k: int = 20,
                     where: dict | None = None) -> list[dict[str, Any]]:
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k, where=where,
                                    include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            out.append({"id": cid, "text": doc, "metadata": meta, "dense_score": 1.0 - dist})
        return out

    def get_all(self, where: dict | None = None) -> dict[str, list]:
        """All chunks (ids, documents, metadatas). Used to build the BM25 index."""
        ids, docs, metas, offset, page = [], [], [], 0, 5000
        while True:
            res = self.collection.get(where=where, limit=page, offset=offset,
                                      include=["documents", "metadatas"])
            if not res["ids"]:
                break
            ids += res["ids"]; docs += res["documents"]; metas += res["metadatas"]
            offset += len(res["ids"])
            if len(res["ids"]) < page:
                break
        return {"ids": ids, "documents": docs, "metadatas": metas}

    def list_documents(self) -> list[dict]:
        data = self.get_all(where=None)
        seen: dict[tuple, dict] = {}
        for m in data["metadatas"]:
            key = (m.get("doc_id"), m.get("version"))
            if key not in seen:
                seen[key] = {"doc_id": m.get("doc_id"), "title": m.get("title"),
                             "version": m.get("version"), "is_current": m.get("is_current"),
                             "jurisdiction": m.get("jurisdiction"), "regime": m.get("regime"),
                             "doc_type": m.get("doc_type"), "source_file": m.get("source_file"),
                             "source_url": m.get("source_url", ""), "num_pages": m.get("num_pages"),
                             "effective_from": m.get("effective_from", ""), "ingested_at": m.get("ingested_at"),
                             "chunks": 0}
            seen[key]["chunks"] += 1
        return sorted(seen.values(), key=lambda d: (d["doc_id"] or "", not d["is_current"]))
