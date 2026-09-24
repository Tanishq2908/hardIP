"""Sentence-transformers embedder with the right query/passage prefixes.

- intfloat/multilingual-e5-*   needs "query: " / "passage: " prefixes
- BAAI/bge-m3                  needs no prefix
Vectors are L2-normalised so cosine similarity works in ChromaDB.
"""
from __future__ import annotations

import logging
from functools import cached_property

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        name = model_name.lower()
        if "e5" in name:
            self.query_prefix, self.passage_prefix = "query: ", "passage: "
        else:
            self.query_prefix, self.passage_prefix = "", ""

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading embedding model %s on %s", self.model_name, device)
        return SentenceTransformer(self.model_name, device=device)

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vecs = self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=show_progress, convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(self.query_prefix + text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()
