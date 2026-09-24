"""CLI for building the legal corpus.

Examples
  python ingest.py                          # ingest everything in data/raw
  python ingest.py --path data/raw/patents  # one folder or one file
  python ingest.py --dry-run                # show how files would be chunked, no DB writes
  python ingest.py --force                  # re-embed even unchanged files
  python ingest.py --reset                  # wipe the collection first
  python ingest.py --list                   # show what is in the database
  python ingest.py --refresh-metadata       # re-apply manifest/metadata fixes without re-embedding
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    ap.add_argument("--path", type=Path, default=settings.raw_dir)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--refresh-metadata", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.dry_run:
        from rag.chunker import chunk_document
        from rag.loaders import iter_files, load_document
        from rag.metadata import build_doc_metadata, load_manifest

        manifest = load_manifest(settings.manifest_path)
        for f in iter_files(args.path):
            doc = load_document(f, settings.enable_ocr, settings.ocr_langs)
            meta = build_doc_metadata(doc, manifest, settings.raw_dir)
            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars)
            labels = sorted({c.section_label for c in chunks if c.section_label})
            print(f"\n{f.name}\n  title: {meta['title']}\n  doc_id: {meta['doc_id']}  version: {meta['version']}"
                  f"\n  jurisdiction: {meta['jurisdiction']}  regime: {meta['regime']}  type: {meta['doc_type']}"
                  f"  lang: {meta['language']}  ({meta['metadata_source']})"
                  f"\n  pages: {meta['num_pages']}  chunks: {len(chunks)}  sections found: {len(labels)}")
            for c in chunks[:3]:
                preview = c.text[:110].replace("\n", " ")
                print(f"   - #{c.index} p.{c.page_start}-{c.page_end} [{c.section_label or '-'}] {preview}...")
        return

    from rag.ingestion import Ingestor

    ing = Ingestor(settings)
    if args.list:
        for d in ing.store.list_documents():
            flag = "current " if d["is_current"] else "old     "
            print(f"{flag}{d['doc_id']:<45} v{d['version']}  {d['jurisdiction']:<4} {d['regime']:<18} {d['chunks']:>4} chunks")
        print(f"\nTotal chunks: {ing.store.count()}")
        return
    if args.refresh_metadata:
        from rag.loaders import iter_files
        for f in iter_files(args.path):
            n = ing.refresh_metadata(f)
            print(f"{'updated' if n else 'not stored'}  {f.name}  ({n} chunks)")
        return
    if args.reset:
        ing.store.reset()
        print("Collection reset.")

    report = ing.ingest_path(args.path, force=args.force)
    print(f"\nAdded: {len(report.added)}  Updated: {len(report.updated)}  "
          f"Unchanged: {len(report.skipped)}  Failed: {len(report.failed)}  "
          f"Chunks written: {report.chunks_written}  Total in DB: {ing.store.count()}")
    for name, err in report.failed:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
