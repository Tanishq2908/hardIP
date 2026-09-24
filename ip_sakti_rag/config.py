"""Central configuration. Every value can be overridden in .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else BASE_DIR / p


def _bool(env: str, default: bool) -> bool:
    return os.getenv(env, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Paths
    raw_dir: Path = _path("RAW_DIR", "data/raw")
    manifest_path: Path = _path("MANIFEST_PATH", "data/manifest.yaml")
    chroma_dir: Path = _path("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("COLLECTION", "ip_sakti_legal")

    # Embeddings
    embed_model: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "80"))

    # OCR
    enable_ocr: bool = _bool("ENABLE_OCR", False)
    ocr_langs: str = os.getenv("OCR_LANGS", "eng+hin")

    # LLM (Groq), used from Chunk 2
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_fast_model: str = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

    # Reranker, used from Chunk 2
    # bge-reranker-base is laptop-friendly; use BAAI/bge-reranker-v2-m3 on a GPU
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


settings = Settings()


@dataclass(frozen=True)
class RetrievalSettings:
    k_each: int = int(os.getenv("K_EACH", "20"))                 # results per query per method
    fused_top_n: int = int(os.getenv("FUSED_TOP_N", "25"))       # candidates sent to the reranker
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "8"))      # passages sent to the CRAG grader
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
    num_query_variants: int = int(os.getenv("NUM_QUERY_VARIANTS", "3"))
    crag_max_retries: int = int(os.getenv("CRAG_MAX_RETRIES", "1"))
    high_confidence: float = float(os.getenv("HIGH_CONFIDENCE", "0.75"))
    medium_confidence: float = float(os.getenv("MEDIUM_CONFIDENCE", "0.55"))
    # Optional web fallback, restricted to official domains (needs TAVILY_API_KEY)
    web_fallback: bool = _bool("WEB_FALLBACK", False)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    official_domains: tuple = tuple(os.getenv(
        "OFFICIAL_DOMAINS",
        "ipindia.gov.in,nbaindia.org,ayush.gov.in,fssai.gov.in,cdsco.gov.in,egazette.gov.in,"
        "indiacode.nic.in,wipo.int,cbd.int,epo.org,uspto.gov").split(","))


retrieval = RetrievalSettings()
