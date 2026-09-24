"""Cross-encoder reranking. Scores are mapped to 0..1 so they feed the confidence score."""
from __future__ import annotations

import logging
from functools import cached_property

import numpy as np

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, max_length: int = 512, device: str | None = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device

    @cached_property
    def model(self):
        import torch
        from sentence_transformers import CrossEncoder

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading reranker %s on %s", self.model_name, device)
        return CrossEncoder(self.model_name, max_length=self.max_length, device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=float)
        if scores.min() < 0 or scores.max() > 1:  # raw logits -> probabilities
            scores = 1.0 / (1.0 + np.exp(-scores))
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
