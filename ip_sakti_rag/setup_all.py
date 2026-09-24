"""One command to prepare the whole legal corpus.

  python setup_all.py

What it does, in order:
  1. Checks that data/manifest.yaml exists
  2. Moves very large non-legal books (Ayurvedic Pharmacopoeia) to data/later
  3. Checks every file can be read (reports scanned PDFs)
  4. Ingests all new or changed files into ChromaDB
  5. Applies clean titles and tags to files already stored
  6. Prints the final list of documents
"""
from __future__ import annotations

import logging
import shutil
import sys
import time

from config import settings

SKIP_PATTERNS = ["pharmacopoeia"]          # moved to data/later (add them back later if you want)


def line(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from rag.chunker import chunk_document
    from rag.ingestion import Ingestor
    from rag.loaders import iter_files, load_document

    raw = settings.raw_dir
    later = raw.parent / "later"

    line("Step 1/6  Manifest")
    if settings.manifest_path.exists():
        print(f"OK  {settings.manifest_path}")
    else:
        print(f"Not found: {settings.manifest_path}. Titles will be guessed automatically.")

    line("Step 2/6  Setting aside very large reference books")
    moved = 0
    for f in list(iter_files(raw)):
        if any(p in f.name.lower() for p in SKIP_PATTERNS):
            later.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(later / f.name))
            print(f"Moved to data/later: {f.name}")
            moved += 1
    if not moved:
        print("Nothing to move.")

    line("Step 3/6  Checking every file can be read")
    files = list(iter_files(raw))
    unreadable = []
    for f in files:
        try:
            doc = load_document(f, settings.enable_ocr, settings.ocr_langs)
            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars)
            status = f"{len(doc.pages):>4} pages  {len(chunks):>5} chunks"
            if not chunks:
                unreadable.append(f.name)
                status = "NO TEXT (scanned PDF?)"
        except Exception as exc:
            unreadable.append(f.name)
            status = f"ERROR: {exc}"
        print(f"  {status:<26} {f.name}")
    if unreadable:
        print("\nThese files have no readable text and will be skipped:")
        for n in unreadable:
            print(f"  - {n}")
        print("Fix: download a text (not scanned) version from the official site, "
              "or set ENABLE_OCR=true in .env after installing Tesseract.")

    line("Step 4/6  Ingesting new and changed files (this can take a while)")
    ing = Ingestor(settings)
    t0 = time.time()
    report = ing.ingest_path(raw)
    print(f"\nAdded: {len(report.added)}  Updated: {len(report.updated)}  "
          f"Unchanged: {len(report.skipped)}  Failed: {len(report.failed)}  "
          f"Chunks written: {report.chunks_written}  ({time.time() - t0:.0f}s)")
    for name, err in report.failed:
        print(f"  FAILED {name}: {err}")

    line("Step 5/6  Applying clean titles and tags")
    for f in iter_files(raw):
        n = ing.refresh_metadata(f)
        print(f"  {'updated' if n else 'skipped'}  {f.name}")

    line("Step 6/6  Documents in the database")
    docs = [d for d in ing.store.list_documents() if d["is_current"]]
    for d in docs:
        print(f"  {d['jurisdiction'] or '?':<5} {str(d['regime'] or ''):<22} {d['chunks']:>5} chunks  {d['title']}")
    print(f"\nTotal: {len(docs)} documents, {ing.store.count()} chunks.")
    print("\nAll done. Start the app with:  uvicorn api:app --port 8000")


if __name__ == "__main__":
    main()
