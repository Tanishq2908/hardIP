"""Ask questions from the terminal.

  python ask.py "Can I patent a classical Ayurvedic formulation?"
  python ask.py "क्या मैं शास्त्रीय फ़ॉर्मूलेशन का पेटेंट करा सकता हूँ?" --jurisdiction IN
  python ask.py            # interactive mode
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def show(res: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    c = res["confidence"]
    print("\n" + "=" * 78)
    print(res["answer"])
    print("-" * 78)
    print(f"Confidence: {c['level'].upper()} ({c['score']})   CRAG: {res['crag_status']}   "
          f"Jurisdiction: {res['jurisdiction']}   Language: {res['language_name']}")
    if res.get("escalate"):
        print(">> Recommended: send to a human expert for review.")
    if res.get("removed_unsupported_claims"):
        print(f"(Removed {res['removed_unsupported_claims']} claim(s) not supported by sources.)")
    for s in res["sources"]:
        flag = "" if s["is_current"] else "  [SUPERSEDED]"
        print(f"  [{s['n']}] {s['label']}  p.{s['pages']}  {s['jurisdiction']}  score {s['rerank_score']}{flag}")
        if s["url"]:
            print(f"      {s['url']}")
    print(f"Queries used: {res['queries']}")
    print(f"Timings (s): {res['timings']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--jurisdiction", choices=["IN", "INTL", "BOTH"], default=None)
    ap.add_argument("--lang", default=None, help="force language code, e.g. mr for Marathi")
    ap.add_argument("--category", default=None, help="product category, e.g. 'classical ASU drug'")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hindi/Tamil output on Windows

    from rag.pipeline import build_pipeline

    print("Loading models...")
    pipe = build_pipeline()
    pipe.warm_up()

    def run(q: str) -> None:
        show(pipe.answer(q, args.jurisdiction, args.lang, args.category), args.json)

    if args.question:
        run(" ".join(args.question))
        return
    print("Type a question (empty line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
