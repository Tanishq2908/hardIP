"""End-to-end pipeline.

question (any language)
  -> detect language, translate to English
  -> multi-query expansion + jurisdiction/topic routing
  -> CRAG loop (hybrid search -> rerank -> grade -> rewrite/retry -> optional official-web fallback)
  -> cited answer (Groq) -> claim-by-claim citation check -> drop unsupported claims
  -> confidence score -> translate back -> escalate if low
"""
from __future__ import annotations

import logging
import re
import time

from .confidence import compute_confidence
from .crag import CorrectiveRAG
from .embeddings import Embedder
from .generator import generate_answer, label_for
from .language import LANG_NAMES, detect_query_language, make_translator
from .llm import LLM
from .query import expand_query
from .reranker import Reranker
from .retriever import HybridRetriever
from .vectorstore import VectorStore
from .verify import verify_sentences

log = logging.getLogger(__name__)

REFUSAL_EN = ("I could not find this in the verified legal sources, so I will not guess. "
              "Try adding your product type and ingredient, or send the question to a verified expert.")


class RAGPipeline:
    def __init__(self, settings, rs, store=None, embedder=None, reranker=None, llm=None, translator=None):
        self.s, self.rs = settings, rs
        self.store = store or VectorStore(settings.chroma_dir, settings.collection_name)
        self.embedder = embedder or Embedder(settings.embed_model, settings.embed_batch_size)
        self.reranker = reranker or Reranker(settings.reranker_model)
        self.llm = llm or LLM(settings.groq_api_key, settings.groq_model, settings.groq_fast_model)
        self.translator = translator or make_translator(self.llm)
        self.retriever = HybridRetriever(self.store, self.embedder)
        self.crag = CorrectiveRAG(self.llm, self.retriever, self.reranker, rs)

    def warm_up(self) -> None:
        """Load models once at startup so the first question is not slow."""
        self.embedder.embed_query("warm up")
        self.reranker.rerank("warm up", [{"id": "x", "text": "warm up"}], top_k=1)
        self.retriever.bm25._ensure()

    # ------------------------------------------------------------------ main
    def answer(self, question: str, jurisdiction: str | None = None, language: str | None = None,
               product_category: str | None = None) -> dict:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def tick(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = round(now - start, 2)
            return now

        lang = detect_query_language(question, language)
        q_en = question if lang == "en" else self.translator.to_english(question, lang)
        t = tick("translate_in", t0)

        exp = expand_query(self.llm, q_en, self.rs.num_query_variants, product_category)
        jur = (jurisdiction or exp["jurisdiction"] or "IN").upper()
        t = tick("multi_query", t)

        crag = self.crag.run(q_en, exp["queries"], jur)
        t = tick("retrieve_rerank_grade", t)

        base = {"question": question, "question_en": q_en, "language": lang,
                "language_name": LANG_NAMES.get(lang, lang), "jurisdiction": jur, "topic": exp["topic"],
                "queries": crag.queries_tried, "crag_status": crag.status, "crag_attempts": crag.attempts,
                "used_web_fallback": crag.used_web}

        if crag.status == "incorrect" or not crag.passages:
            return self._refusal(base, lang, timings, t0)

        passages = crag.passages[: self.rs.max_context_chunks]
        gen = generate_answer(self.llm, q_en, passages, jur, product_category)
        t = tick("generate", t)
        if not gen["sentences"]:
            return self._refusal(base, lang, timings, t0)

        verdicts = verify_sentences(self.llm, gen["sentences"], passages)
        t = tick("verify", t)

        kept = [(s, v) for s, v in zip(gen["sentences"], verdicts) if v != "unsupported"]
        removed = len(gen["sentences"]) - len(kept)
        if not kept:
            return self._refusal(base, lang, timings, t0)

        # Renumber sources in order of first use: S3 -> [1], S1 -> [2] ...
        order: list[str] = []
        for s, _ in kept:
            for c in s["citations"]:
                if c not in order and re.fullmatch(r"S\d+", c) and int(c[1:]) <= len(passages):
                    order.append(c)
        num = {sid: i for i, sid in enumerate(order, start=1)}
        used = [passages[int(sid[1:]) - 1] for sid in order]

        sentences_out = []
        for s, v in kept:
            nums = sorted({num[c] for c in s["citations"] if c in num})
            sentences_out.append({"text": s["text"], "citations": nums, "verdict": v})
        answer_en = " ".join(s["text"] + "".join(f" [{n}]" for n in s["citations"]) for s in sentences_out)

        conf = compute_confidence(used, crag.status, [v for _, v in kept],
                                  self.rs.high_confidence, self.rs.medium_confidence)
        escalate = conf["level"] == "low" or gen["insufficient"]

        answer = answer_en if lang == "en" else self.translator.from_english(answer_en, lang)
        tick("translate_out", t)
        timings["total"] = round(time.perf_counter() - t0, 2)

        return {
            **base,
            "answer": answer,
            "answer_en": answer_en,
            "sentences": sentences_out,
            "sources": [self._source(i, p) for i, p in enumerate(used, start=1)],
            "confidence": conf,
            "escalate": escalate,
            "insufficient": gen["insufficient"],
            "missing": gen["missing"],
            "removed_unsupported_claims": removed,
            "timings": timings,
        }

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _source(n: int, p: dict) -> dict:
        m = p["metadata"]
        return {
            "n": n, "id": p["id"], "label": label_for(m), "title": m.get("title", ""),
            "section": m.get("section_label", ""), "chapter": m.get("chapter", ""), "pages": m.get("pages", ""),
            "jurisdiction": m.get("jurisdiction", ""), "regime": m.get("regime", ""),
            "is_current": m.get("is_current", True), "effective_from": m.get("effective_from", ""),
            "url": m.get("source_url", ""), "source_file": m.get("source_file", ""),
            "source_type": m.get("source_type", "corpus"),
            "rerank_score": round(p.get("rerank_score", 0.0), 3), "grade": p.get("grade", ""),
            "excerpt": re.sub(r"^\[[^\]]*\]\n", "", p["text"])[:500],
        }

    def _refusal(self, base: dict, lang: str, timings: dict, t0: float) -> dict:
        text = REFUSAL_EN if lang == "en" else self.translator.from_english(REFUSAL_EN, lang)
        timings["total"] = round(time.perf_counter() - t0, 2)
        return {**base, "answer": text, "answer_en": REFUSAL_EN, "sentences": [], "sources": [],
                "confidence": {"score": 0.0, "level": "low", "components": {}}, "escalate": True,
                "insufficient": True, "missing": "", "removed_unsupported_claims": 0, "timings": timings}


def build_pipeline() -> RAGPipeline:
    from config import retrieval, settings

    return RAGPipeline(settings, retrieval)
