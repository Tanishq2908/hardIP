"""Load any supported document into page-level text.

Supported: .pdf, .docx, .txt, .md, .html, .htm
Every loader returns a LoadedDocument with a list of PageText, so the rest of
the pipeline never cares about the original format.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


@dataclass
class PageText:
    page: int          # 1-based page number (1 for formats without pages)
    text: str


@dataclass
class LoadedDocument:
    path: Path
    file_type: str
    sha256: str
    title: str
    pages: list[PageText] = field(default_factory=list)
    pdf_metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


# ---------------------------------------------------------------- helpers
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Normalise unicode and whitespace without damaging Indic scripts."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")                      # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # re-join hyphenated words
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_lines(pages: list[PageText], min_pages: int = 4) -> list[PageText]:
    """Drop running headers/footers (short lines that repeat on most pages)."""
    if len(pages) < min_pages:
        return pages
    counts: Counter[str] = Counter()
    for p in pages:
        lines = {ln.strip() for ln in p.text.splitlines() if 0 < len(ln.strip()) < 80}
        counts.update(lines)
    threshold = len(pages) * 0.5
    repeated = {ln for ln, c in counts.items() if c >= threshold and not re.fullmatch(r"\d+", ln)}
    page_num = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
    out = []
    for p in pages:
        kept = [ln for ln in p.text.splitlines()
                if ln.strip() not in repeated and not page_num.match(ln.strip())]
        out.append(PageText(p.page, "\n".join(kept)))
    return out


def _first_meaningful_line(text: str) -> str:
    for ln in text.splitlines():
        ln = ln.strip().lstrip("#").strip()
        if len(ln) > 8 and not re.fullmatch(r"[\W\d_]+", ln):
            return ln[:160]
    return ""


# ---------------------------------------------------------------- loaders
def _ocr_page(page, langs: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log.warning("OCR requested but pytesseract/Pillow not installed; skipping OCR.")
        return ""
    pix = page.get_pixmap(dpi=220)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=langs)


def load_pdf(path: Path, enable_ocr: bool = False, ocr_langs: str = "eng+hin"):
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz

    pages: list[PageText] = []
    with fitz.open(path) as doc:
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if len(text.strip()) < 25 and enable_ocr:
                text = _ocr_page(page, ocr_langs)
            pages.append(PageText(i, clean_text(text)))
    return pages, meta


def load_docx(path: Path):
    import docx

    d = docx.Document(str(path))
    parts: list[str] = []
    for para in d.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(dict.fromkeys(cells)))  # dedupe merged cells
    meta = {}
    cp = d.core_properties
    if cp.title:
        meta["title"] = cp.title
    return [PageText(1, clean_text("\n".join(parts)))], meta


def load_text(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return [PageText(1, clean_text(raw.decode(enc)))], {}
        except UnicodeDecodeError:
            continue
    return [PageText(1, clean_text(raw.decode("utf-8", errors="ignore")))], {}


def load_html(path: Path):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    meta = {"title": soup.title.get_text(strip=True)} if soup.title else {}
    return [PageText(1, clean_text(soup.get_text("\n")))], meta


def load_document(path: Path, enable_ocr: bool = False, ocr_langs: str = "eng+hin") -> LoadedDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages, meta = load_pdf(path, enable_ocr, ocr_langs)
    elif ext == ".docx":
        pages, meta = load_docx(path)
    elif ext in {".txt", ".md"}:
        pages, meta = load_text(path)
    elif ext in {".html", ".htm"}:
        pages, meta = load_html(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    pages = remove_repeated_lines(pages)
    pages = [p for p in pages if p.text.strip()]
    title = (meta.get("title") or "").strip() or _first_meaningful_line(
        "\n".join(p.text for p in pages[:2])) or path.stem
    return LoadedDocument(path=path, file_type=ext.lstrip("."), sha256=file_sha256(path),
                          title=title, pages=pages, pdf_metadata=meta)


def iter_files(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith((".", "~$")):
            yield p
