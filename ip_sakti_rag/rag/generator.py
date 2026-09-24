"""Grounded answer generation with mandatory per-sentence citations."""
from __future__ import annotations

JURISDICTION_RULES = {
    "IN": "Answer only under Indian law. Do not use international sources unless the question asks about them.",
    "INTL": "Answer only under international or foreign regimes (WIPO, Nagoya Protocol, EPO, USPTO and similar). "
            "Do not describe Indian law unless the question asks.",
    "BOTH": "If both Indian and international sources are relevant, describe the Indian position first, then the "
            "international position, and say clearly which is which. Never mix them in one sentence.",
}

GEN_SYSTEM = """You are IP-SAKTI Sahayak, a legal information assistant on intellectual property and regulatory compliance for Ayurveda products.

Rules you must follow:
1. Use ONLY the numbered sources given. Never use outside knowledge. Never invent sections, dates, fees, forms or deadlines.
2. Every sentence must cite at least one source id, e.g. ["S1"] or ["S1","S3"].
3. {jurisdiction_rule}
4. Mention the Act and section, article or rule number whenever the source shows it.
5. Write for a non-lawyer: short, plain sentences. 3 to 8 sentences in total.
6. If the sources do not contain the answer, say what is missing and set "insufficient" to true. Do not guess.
7. This is legal information, not legal advice.

Return JSON only:
{{"sentences": [{{"text": "...", "citations": ["S1"]}}], "insufficient": false, "missing": ""}}"""


def label_for(meta: dict) -> str:
    bits = [meta.get("short_title") or meta.get("title", "Source")]
    if meta.get("section_label") and meta["section_label"] != "Preamble":
        bits.append(meta["section_label"])
    return ", ".join(bits)


def build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        m = p["metadata"]
        status = "current" if m.get("is_current", True) else "superseded"
        pages = f"pages {m['pages']}" if m.get("pages") else ""
        head = " | ".join(x for x in [label_for(m), pages, m.get("jurisdiction", ""), status] if x)
        blocks.append(f"[S{i}] {head}\n{p['text']}")
    return "\n\n".join(blocks)


def generate_answer(llm, question: str, passages: list[dict], jurisdiction: str,
                    product_category: str | None = None) -> dict:
    system = GEN_SYSTEM.format(jurisdiction_rule=JURISDICTION_RULES.get(jurisdiction, JURISDICTION_RULES["BOTH"]))
    extra = f"\nThe user's product category: {product_category}." if product_category else ""
    user = f"Question: {question}{extra}\n\nSources:\n\n{build_context(passages)}"
    data = llm.json(system, user, fast=False, max_tokens=1200, default={}) or {}
    sentences = []
    for s in data.get("sentences", []):
        text = str(s.get("text", "")).strip()
        cites = [str(c).strip().strip("[]") for c in (s.get("citations") or [])]
        if text:
            sentences.append({"text": text, "citations": [c for c in cites if c]})
    return {"sentences": sentences, "insufficient": bool(data.get("insufficient", False)),
            "missing": str(data.get("missing", "") or "")}
