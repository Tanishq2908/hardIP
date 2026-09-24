"""Document-level metadata.

Two sources, in priority order:
1. data/manifest.yaml  (optional, per file, written by you: most accurate)
2. Automatic inference from the title and first pages (so any new file still works)

Everything is flattened into ChromaDB-safe values (str, int, float, bool).
"""
from __future__ import annotations

import fnmatch
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .loaders import LoadedDocument

# Keyword signals -------------------------------------------------------------
INTL_SIGNALS = [
    "wipo", "world intellectual property organization", "nagoya protocol",
    "convention on biological diversity", "european patent", "epo", "uspto",
    "united states patent", "trips", "patent cooperation treaty", "pct",
    "treaty", "united nations", "european union", "regulation (eu)",
]
IN_SIGNALS = [
    "india", "gazette of india", "ministry of", "government of india", "ayush",
    "fssai", "cdsco", "national biodiversity authority", "controller general",
    "parliament", "lok sabha", "rajya sabha", "state biodiversity board",
]
REGIME_SIGNALS = {
    "patents": ["patent", "invention", "prior art", "novelty"],
    "biodiversity_abs": ["biological diversity", "biodiversity", "benefit sharing",
                         "access and benefit", "nagoya", "genetic resource"],
    "drugs": ["drugs and cosmetics", "ayurvedic, siddha", "asu drug", "new drugs and clinical",
              "phytopharmaceutical", "licensing authority"],
    "advertising": ["advertisement", "magic remedies", "misleading", "objectionable"],
    "food": ["food safety", "fssai", "ayurveda aahar", "nutraceutical"],
    "cosmetics": ["cosmetic"],
    "trademarks": ["trade mark", "trademark"],
    "traditional_knowledge": ["traditional knowledge", "tkdl"],
}
DOC_TYPE_SIGNALS = [
    ("treaty", ["treaty", "protocol", "convention"]),
    ("rules", ["rules, 19", "rules, 20", " rules"]),
    ("regulations", ["regulations, 20", "regulation"]),
    ("act", ["act, 19", "act, 20", " act"]),
    ("notification", ["notification", "g.s.r", "s.o."]),
    ("guideline", ["guideline", "manual", "advisory", "circular"]),
]
SCRIPTS = {
    "hi": (0x0900, 0x097F), "gu": (0x0A80, 0x0AFF), "ta": (0x0B80, 0x0BFF),
    "bn": (0x0980, 0x09FF), "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F), "pa": (0x0A00, 0x0A7F), "or": (0x0B00, 0x0B7F),
}


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:80] or "doc"


def detect_language(text: str) -> str:
    sample = text[:5000]
    letters = [ch for ch in sample if ch.isalpha()]
    if not letters:
        return "unknown"
    counts = {lang: sum(lo <= ord(ch) <= hi for ch in letters) for lang, (lo, hi) in SCRIPTS.items()}
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n / len(letters) > 0.3 else "en"


def _score(text: str, words: list[str]) -> int:
    """Whole-word counts, so "epo" does not match inside "deposit"."""
    return sum(len(re.findall(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", text)) for w in words)


INTL_TITLE_WORDS = ["treaty", "protocol", "convention", "wipo", "european", "regulation (eu)", "united nations"]


def infer_metadata(doc: LoadedDocument) -> dict[str, Any]:
    head = (doc.title + "\n" + "\n".join(p.text for p in doc.pages[:3])).lower()
    title_l = doc.title.lower()

    indian_title = bool(re.search(r"\b(act|rules|regulations|order)\b,?\s*(18|19|20)\d\d", title_l)) and \
        not any(w in title_l for w in INTL_TITLE_WORDS)
    intl = _score(head, INTL_SIGNALS) + 5 * _score(title_l, INTL_SIGNALS)
    ind = _score(head, IN_SIGNALS) + 5 * _score(title_l, IN_SIGNALS) + (15 if indian_title else 0)
    jurisdiction = "INTL" if intl > ind else "IN"

    regimes = [r for r, words in REGIME_SIGNALS.items() if _score(head, words) > 0]
    regimes.sort(key=lambda r: -_score(head, REGIME_SIGNALS[r]))

    doc_type = next((t for t, words in DOC_TYPE_SIGNALS if any(w in title_l for w in words)), "document")

    year = None
    m = re.search(r"\b(19[4-9]\d|20[0-4]\d)\b", doc.title)
    if m:
        year = int(m.group(1))

    return {
        "jurisdiction": jurisdiction,
        "regime": regimes[0] if regimes else "general",
        "regimes": ",".join(regimes[:4]),
        "doc_type": doc_type,
        "year": year,
        "language": detect_language(doc.full_text),
        "metadata_source": "inferred",
    }


# Manifest ---------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, dict]:
    """Manifest format (YAML): a mapping of filename or glob -> fields."""
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): (v or {}) for k, v in data.get("documents", data).items()}


def manifest_entry(manifest: dict[str, dict], file_path: Path, raw_root: Path) -> dict:
    try:
        rel = file_path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        rel = file_path.name
    for key in (rel, file_path.name):
        if key in manifest:
            return dict(manifest[key])
    for pattern, entry in manifest.items():
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(file_path.name, pattern):
            return dict(entry)
    return {}


def _date_int(value: Any) -> int | None:
    """Dates become YYYYMMDD ints so ChromaDB can filter with $lte / $gte."""
    if value in (None, ""):
        return None
    if isinstance(value, (date, datetime)):
        return int(value.strftime("%Y%m%d"))
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y"):
        try:
            d = datetime.strptime(s, fmt)
            return int(d.strftime("%Y%m%d"))
        except ValueError:
            continue
    return None


def build_doc_metadata(doc: LoadedDocument, manifest: dict[str, dict], raw_root: Path) -> dict[str, Any]:
    meta = infer_metadata(doc)
    entry = manifest_entry(manifest, doc.path, raw_root)
    if entry:
        meta["metadata_source"] = "manifest"
    meta.update({k: v for k, v in entry.items() if v is not None})

    meta.setdefault("title", doc.title)
    meta["title"] = entry.get("title") or doc.title
    meta["short_title"] = entry.get("short_title") or meta["title"][:70]
    meta["doc_id"] = entry.get("doc_id") or slugify(doc.path.stem)
    meta["source_file"] = doc.path.name
    meta["file_type"] = doc.file_type
    meta["version"] = doc.sha256[:16]
    meta["num_pages"] = len(doc.pages)
    meta["ingested_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["jurisdiction"] = str(meta.get("jurisdiction", "IN")).upper()

    for key in ("effective_from", "effective_to"):
        if key in meta:
            meta[f"{key}_int"] = _date_int(meta[key])
    return sanitize(meta)


def sanitize(meta: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB accepts only str/int/float/bool and rejects None."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple, set)):
            out[k] = ",".join(map(str, v))
        elif isinstance(v, (date, datetime)):
            out[k] = v.isoformat()
        else:
            out[k] = str(v)
    return out
