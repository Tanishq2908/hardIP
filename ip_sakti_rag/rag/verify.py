"""Check every generated sentence against the exact source text it cites."""
from __future__ import annotations

VERIFY_SYSTEM = """You check claims against legal source text.
For each claim, decide using ONLY the cited source text shown with it:
- "supported": the source states it or it follows directly.
- "partial": some of it is supported, some is not stated.
- "unsupported": the source does not say this.
Return JSON only: {"results": [{"id": 1, "verdict": "supported"}, ...]} covering every claim."""


def verify_sentences(llm, sentences: list[dict], passages: list[dict]) -> list[str]:
    """Returns one verdict per sentence."""
    id_to_text = {f"S{i}": p["text"] for i, p in enumerate(passages, start=1)}
    verdicts: list[str | None] = []
    blocks = []
    for n, s in enumerate(sentences, start=1):
        valid = [c for c in s["citations"] if c in id_to_text]
        if not valid:
            verdicts.append("unsupported")  # no citation or citation to a non-existent source
            continue
        verdicts.append(None)
        src = "\n---\n".join(id_to_text[c][:1400] for c in valid)
        blocks.append(f"Claim {n}: {s['text']}\nCited source text:\n{src}")
    if blocks:
        data = llm.json(VERIFY_SYSTEM, "\n\n=====\n\n".join(blocks), max_tokens=500, default={}) or {}
        got = {int(r.get("id", 0)): str(r.get("verdict", "")).lower() for r in data.get("results", [])
               if str(r.get("id", "")).isdigit()}
        for n in range(1, len(sentences) + 1):
            if verdicts[n - 1] is None:
                v = got.get(n, "partial")  # grader skipped it: neither trust nor drop
                verdicts[n - 1] = v if v in ("supported", "partial", "unsupported") else "partial"
    return [v or "partial" for v in verdicts]
