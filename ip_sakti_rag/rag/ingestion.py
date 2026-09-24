"""Ingestion pipeline: file -> pages -> chunks -> embeddings -> ChromaDB."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, chunk_document
from .embeddings import Embedder
from .loaders import iter_files, load_document
from .metadata import build_doc_metadata, load_manifest, sanitize
from .vectorstore import VectorStore

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_written: int = 0


def context_header(meta: dict, chunk: Chunk) -> str:
    """Short header prepended to each chunk. Helps both dense and BM25 retrieval."""
    bits = [meta.get("short_title") or meta.get("title", "")]
    if chunk.chapter:
        bits.append(chunk.chapter)
    if chunk.section_label:
        bits.append(chunk.section_label)
    return "[" + " | ".join(b for b in bits if b) + "]"


def chunk_metadata(doc_meta: dict, chunk: Chunk) -> dict:
    pages = str(chunk.page_start) if chunk.page_start == chunk.page_end else f"{chunk.page_start}-{chunk.page_end}"
    return sanitize({
        **doc_meta,
        "chunk_index": chunk.index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "pages": pages,
        "section_kind": chunk.section_kind,
        "section_label": chunk.section_label,
        "chapter": chunk.chapter,
        "is_current": True,
    })


class Ingestor:
    def __init__(self, settings, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.s = settings
        self.store = store or VectorStore(settings.chroma_dir, settings.collection_name)
        self.embedder = embedder or Embedder(settings.embed_model, settings.embed_batch_size)
        self.manifest = load_manifest(settings.manifest_path)

    def prepare(self, path: Path):
        """Load + chunk one file without touching the database (used by --dry-run)."""
        doc = load_document(path, self.s.enable_ocr, self.s.ocr_langs)
        meta = build_doc_metadata(doc, self.manifest, self.s.raw_dir)
        chunks = chunk_document(doc, self.s.chunk_size, self.s.chunk_overlap, self.s.min_chunk_chars)
        return doc, meta, chunks

    def ingest_file(self, path: Path, force: bool = False, report: IngestReport | None = None) -> IngestReport:
        report = report or IngestReport()
        try:
            doc, meta, chunks = self.prepare(path)
            if not chunks:
                report.failed.append((path.name, "no extractable text (scanned PDF? enable OCR)"))
                return report

            doc_id, version = meta["doc_id"], meta["version"]
            existing = self.store.current_version(doc_id)
            if existing == version and not force:
                report.skipped.append(path.name)
                return report

            ids = [f"{doc_id}::{version}::{c.index:05d}" for c in chunks]
            documents = [f"{context_header(meta, c)}\n{c.text}" for c in chunks]
            metadatas = [chunk_metadata(meta, c) for c in chunks]
            embeddings = self.embedder.embed_documents(documents)
            self.store.upsert(ids, documents, metadatas, embeddings)
            retired = self.store.retire_old_versions(doc_id, version)

            (report.updated if existing else report.added).append(path.name)
            report.chunks_written += len(chunks)
            log.info("%s: %d chunks (retired %d old)", path.name, len(chunks), retired)
        except Exception as exc:  # keep going on bad files
            log.exception("Failed to ingest %s", path)
            report.failed.append((path.name, str(exc)))
        return report

    def refresh_metadata(self, path: Path) -> int:
        """Re-apply manifest + inference to already-stored chunks (no re-embedding)."""
        doc = load_document(path, False, self.s.ocr_langs)
        meta = build_doc_metadata(doc, self.manifest, self.s.raw_dir)
        res = self.store.collection.get(where={"version": meta["version"]}, include=["metadatas"])
        if not res["ids"]:
            return 0
        doc_level = {k: v for k, v in meta.items() if k != "ingested_at"}
        self.store.update_metadatas(res["ids"], [{**m, **doc_level} for m in res["metadatas"]])
        return len(res["ids"])

    def ingest_path(self, root: Path, force: bool = False, progress: bool = True) -> IngestReport:
        report = IngestReport()
        files = list(iter_files(root))
        iterator = files
        if progress:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Ingesting", unit="file")
        for f in iterator:
            self.ingest_file(f, force=force, report=report)
        return report
