"""Legal-aware chunking.

1. Detect structural headings (Chapter, Section, Article, Rule, Regulation, and
   the Indian-statute style "3. What are not inventions.—").
2. Split the document into those sections so a chunk never mixes two sections.
3. Pack paragraphs/sentences into chunks of ~chunk_size characters with overlap.
4. Track the page each chunk starts and ends on, for citations.

If a document has no recognisable headings, it falls back to plain
paragraph-based chunking, so any document still works.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from .loaders import LoadedDocument

# (regex, kind). Group 1 is the label number when present.
HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(?:CHAPTER|Chapter)\s+([IVXLC]+[A-Z]?|\d+[A-Z]?)\b"), "chapter"),
    (re.compile(r"^\s*(?:PART|Part)\s+([IVXLC]+[A-Z]?|\d+[A-Z]?)\b"), "part"),
    (re.compile(r"^\s*(?:Section|SECTION|Sec\.)\s+(\d+[A-Z]{0,2})\b"), "section"),
    (re.compile(r"^\s*(?:Article|ARTICLE|Art\.)\s+(\d+[a-zA-Z]{0,2})\b"), "article"),
    (re.compile(r"^\s*(?:Rule|RULE)\s+(\d+[A-Z]{0,2})\b"), "rule"),
    (re.compile(r"^\s*(?:Regulation|REGULATION)\s+(\d+[A-Z]{0,2})\b"), "regulation"),
    (re.compile(r"^\s*(?:Schedule|SCHEDULE)\s*([IVXLC]+|[A-Z]|\d+)?\b"), "schedule"),
    # Indian statute style: "3. What are not inventions.—" / "10A. Something.–"
    (re.compile(r"^\s*(\d{1,3}[A-Z]{0,2})\.\s+[A-Z][^\n]{2,120}?(?:[—–]|\.-)"), "section"),
    # Markdown headings in .md files
    (re.compile(r"^\s*#{1,4}\s+(.{2,100})$"), "heading"),
    # Hindi headings
    (re.compile(r"^\s*(?:धारा|अनुच्छेद|नियम|अध्याय)\s+(\d+[A-Z]{0,2})"), "section"),
]

KIND_LABEL = {"chapter": "Chapter", "part": "Part", "section": "Section", "article": "Article",
              "rule": "Rule", "regulation": "Regulation", "schedule": "Schedule"}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?।;])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    page_start: int
    page_end: int
    section_kind: str      # section / article / rule / ... / "" if none
    section_label: str     # e.g. "Section 3" ("" if none)
    chapter: str           # e.g. "Chapter II" ("" if none)
    char_start: int


def _build_text_with_offsets(doc: LoadedDocument) -> tuple[str, list[int], list[int]]:
    parts, starts, pages, pos = [], [], [], 0
    for p in doc.pages:
        starts.append(pos)
        pages.append(p.page)
        parts.append(p.text)
        pos += len(p.text) + 2
    return "\n\n".join(parts), starts, pages


def _page_at(offset: int, starts: list[int], pages: list[int]) -> int:
    i = bisect.bisect_right(starts, offset) - 1
    return pages[max(i, 0)] if pages else 1


def _find_headings(text: str) -> list[tuple[int, str, str]]:
    """Return [(char_offset, kind, label)] for every heading line."""
    found, offset = [], 0
    for line in text.split("\n"):
        stripped = line.strip()
        if 0 < len(stripped) < 220:
            for pattern, kind in HEADING_PATTERNS:
                m = pattern.match(line)
                if m:
                    num = (m.group(1) or "").strip() if m.groups() else ""
                    label = num[:80] if kind == "heading" else f"{KIND_LABEL.get(kind, kind.title())} {num}".strip()
                    found.append((offset, kind, label))
                    break
        offset += len(line) + 1
    return found


def _units(text: str, base: int, max_len: int) -> list[tuple[int, str]]:
    """Break text into (offset, piece) units no longer than max_len."""
    units: list[tuple[int, str]] = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):  # paragraphs
        para, pstart = m.group(0), base + m.start()
        if len(para) <= max_len:
            units.append((pstart, para))
            continue
        cursor = 0
        for sent in SENTENCE_SPLIT.split(para):
            idx = para.find(sent, cursor)
            idx = cursor if idx < 0 else idx
            cursor = idx + len(sent)
            if len(sent) <= max_len:
                units.append((pstart + idx, sent))
            else:  # very long sentence: hard window split on spaces
                s = 0
                while s < len(sent):
                    e = min(s + max_len, len(sent))
                    if e < len(sent):
                        sp = sent.rfind(" ", s, e)
                        e = sp if sp > s + max_len // 2 else e
                    units.append((pstart + idx + s, sent[s:e].strip()))
                    s = e
    return [(o, u) for o, u in units if u.strip()]


def _pack(units: list[tuple[int, str]], size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Greedy packing with tail overlap. Returns (start_offset, end_offset, text)."""
    chunks, cur = [], []
    cur_len = 0

    def emit():
        if cur:
            start = cur[0][0]
            end = cur[-1][0] + len(cur[-1][1])
            chunks.append((start, end, "\n".join(u for _, u in cur)))

    for off, u in units:
        if cur and cur_len + len(u) + 1 > size:
            emit()
            tail, tlen = [], 0
            for prev in reversed(cur):
                if tlen + len(prev[1]) > overlap:
                    break
                tail.insert(0, prev)
                tlen += len(prev[1]) + 1
            cur, cur_len = tail, tlen
        cur.append((off, u))
        cur_len += len(u) + 1
    emit()
    return chunks


def chunk_document(doc: LoadedDocument, chunk_size: int = 1200, chunk_overlap: int = 200,
                   min_chars: int = 80) -> list[Chunk]:
    text, starts, pages = _build_text_with_offsets(doc)
    if not text.strip():
        return []

    headings = _find_headings(text)
    # Segments: (start, end, kind, label, chapter)
    segments = []
    if headings:
        if headings[0][0] > 0:
            segments.append((0, headings[0][0], "", "Preamble", ""))
        chapter = ""
        for i, (off, kind, label) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            if kind in ("chapter", "part"):
                chapter = label
            segments.append((off, end, kind, label, chapter))
    else:
        segments.append((0, len(text), "", "", ""))

    # Merge tiny segments (e.g. a bare "CHAPTER II" line) into the next one
    merged: list[list] = []
    for seg in segments:
        seg = list(seg)
        if merged and (merged[-1][1] - merged[-1][0]) < min_chars:
            prev = merged.pop()
            seg[0] = prev[0]
            if not seg[4]:
                seg[4] = prev[4]
        merged.append(seg)

    chunks: list[Chunk] = []
    for s_start, s_end, kind, label, chapter in merged:
        seg_text = text[s_start:s_end]
        for c_start, c_end, c_text in _pack(_units(seg_text, s_start, chunk_size), chunk_size, chunk_overlap):
            if len(c_text.strip()) < min_chars and chunks:
                chunks[-1].text += "\n" + c_text
                chunks[-1].page_end = max(chunks[-1].page_end, _page_at(c_end, starts, pages))
                continue
            chunks.append(Chunk(
                text=c_text.strip(), index=len(chunks),
                page_start=_page_at(c_start, starts, pages),
                page_end=_page_at(max(c_end - 1, c_start), starts, pages),
                section_kind=kind if kind not in ("chapter", "part") else "",
                section_label=label if kind not in ("chapter", "part") else (label if label == "Preamble" else ""),
                chapter=chapter, char_start=c_start,
            ))
    return chunks
