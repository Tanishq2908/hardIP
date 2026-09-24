"""Evaluate the RAG pipeline on a golden question set.

  python eval.py                                 # uses data/eval_questions.yaml
  python eval.py --file my_questions.yaml

Metrics
- Source hit: the expected document (and section) is among the cited sources
- Keyword recall: share of expected keywords present in the English answer
- Refusal accuracy: out-of-scope questions are refused, in-scope ones are not
- Average confidence and latency
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from config import BASE_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=BASE_DIR / "data" / "eval_questions.yaml")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between questions (Groq rate limits)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    items = (yaml.safe_load(args.file.read_text(encoding="utf-8")) or {}).get("questions", [])
    if not items:
        sys.exit(f"No questions found in {args.file}")

    from rag.pipeline import build_pipeline

    print("Loading models...")
    pipe = build_pipeline()
    pipe.warm_up()

    rows = []
    for i, it in enumerate(items, start=1):
        q = it["q"]
        res = pipe.answer(q, it.get("jurisdiction"), it.get("language"))
        refused = not res["sources"]
        expect_refusal = bool(it.get("expect_refusal", False))

        doc_ids = [s["id"].split("::")[0] for s in res["sources"]]
        sections = [s["section"] for s in res["sources"]]
        hit = None
        if not expect_refusal and it.get("expect_doc"):
            want_doc, want_sec = it["expect_doc"].lower(), (it.get("expect_section") or "").lower()
            hit = any(want_doc in d.lower() and (not want_sec or sec.lower().startswith(want_sec))
                      for d, sec in zip(doc_ids, sections))

        kws = [k.lower() for k in it.get("keywords", [])]
        ans = res.get("answer_en", "").lower()
        recall = (sum(k in ans for k in kws) / len(kws)) if kws else None

        row = {"q": q, "refused": refused, "refusal_correct": refused == expect_refusal, "source_hit": hit,
               "keyword_recall": recall, "confidence": res["confidence"]["score"],
               "level": res["confidence"]["level"], "crag": res["crag_status"],
               "latency_s": res["timings"].get("total"), "sources": [s["label"] for s in res["sources"]],
               "answer_en": res.get("answer_en", "")}
        rows.append(row)
        mark = "PASS" if row["refusal_correct"] and hit is not False else "FAIL"
        print(f"\n[{i}/{len(items)}] {mark}  {q}")
        print(f"   hit={hit}  keywords={recall if recall is None else round(recall, 2)}  "
              f"conf={row['confidence']} ({row['level']})  crag={row['crag']}  {row['latency_s']}s")
        print(f"   sources: {row['sources'][:3]}")
        time.sleep(args.pause)

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "questions": len(rows),
        "source_hit_rate": avg([1.0 if r["source_hit"] else 0.0 for r in rows if r["source_hit"] is not None]),
        "keyword_recall": avg([r["keyword_recall"] for r in rows]),
        "refusal_accuracy": avg([1.0 if r["refusal_correct"] else 0.0 for r in rows]),
        "avg_confidence": avg([r["confidence"] for r in rows if not r["refused"]]),
        "avg_latency_s": avg([r["latency_s"] for r in rows]),
    }
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"{k:<20} {v}")
    out = BASE_DIR / "data" / "eval_results.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved details to {out}")


if __name__ == "__main__":
    main()
