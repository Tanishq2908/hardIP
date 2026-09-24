# IP-SAKTI Sahayak: RAG backend

Source-cited legal assistant for IP and regulatory guidance in Ayurveda.
LLM: Groq. Vector DB: ChromaDB. Works on any PDF, DOCX, TXT, MD or HTML you add.

## Build plan

| Chunk | What it adds | Status |
|---|---|---|
| 1 | Ingestion: loaders, legal-aware chunking, metadata, versioning, ChromaDB | Done |
| 2 | Retrieval: multi-query (Groq), hybrid search (dense + BM25, RRF), jurisdiction filters, cross-encoder reranking | Done |
| 3 | CRAG: relevance grading, query rewrite and retry, official-web fallback, refuse or escalate | Done |
| 4 | Generation: cited answers, claim-by-claim citation check, confidence score, answer in user's language | Done |
| 5 | FastAPI backend (ask, documents, escalate) + Bhashini hook | Done |
| 6 | Web UI connected to the API + evaluation script | Done |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your GROQ_API_KEY
```

The first run downloads the embedding model (about 1 GB for multilingual-e5-base).

## Add documents

Put files anywhere under `data/raw/` (subfolders are fine):

```
data/raw/
  patents/patents_act_1970.pdf
  patents/patents_amendment_rules_2024.pdf
  biodiversity/biological_diversity_act_2002.pdf
  treaties/wipo_gratk_treaty_2024.pdf
```

Optional but recommended: copy `data/manifest.example.yaml` to `data/manifest.yaml`
and describe your key files (title, jurisdiction, regime, effective dates, official URL).
Files not in the manifest still work; their metadata is inferred automatically.

## Ingest

```bash
python ingest.py --dry-run     # preview titles, jurisdiction, sections and chunks (no DB writes)
python ingest.py               # embed and store everything in data/raw
python ingest.py --list        # see what is stored, with versions
```

Re-running is safe: unchanged files are skipped. If a file changes (for example an
amended Act), the new version is stored and the old chunks are kept but marked
`is_current=False`, so answers use current law while history stays available.

## What each chunk stores

Text: `[Short title | Chapter | Section]` header followed by the chunk text.

Metadata used later for filtering and citations:
`doc_id, title, short_title, jurisdiction (IN/INTL), regime, doc_type, language,
section_label, chapter, page_start, page_end, pages, source_url, effective_from,
effective_from_int, version, is_current, source_file`

## Ask questions (terminal)

```bash
python ask.py "Can I patent a classical Ayurvedic formulation?"
python ask.py "Can I patent it abroad?" --jurisdiction INTL
python ask.py "क्या शास्त्रीय फ़ॉर्मूलेशन का पेटेंट हो सकता है?"
python ask.py                      # interactive mode
```

## Run the API

```bash
uvicorn api:app --port 8000
```

Open http://localhost:8000 for the web app (served from `static/index.html`)
and http://localhost:8000/docs to try the API directly.

Web app features: live server status, cited answers with confidence, source viewer
that opens the original PDF at the cited page, "How this answer was found" panel,
feedback buttons, saved question history, document library, expert desk for
escalated questions, product classifier, and five interface languages.

API endpoints: `/api/ask`, `/api/health`, `/api/documents`, `/api/files/{name}`,
`/api/escalate`, `/api/escalations`, `/api/escalations/{id}/resolve`, `/api/feedback`.

## Evaluate

```bash
python eval.py
```

Runs the questions in `data/eval_questions.yaml` and reports source hit rate,
keyword recall, refusal accuracy, average confidence and latency.

## How one question flows

1. Detect the language; translate to English (Groq now, Bhashini later).
2. Multi-query: Groq writes 3 extra search queries and guesses jurisdiction and topic.
3. Hybrid search: every query runs on ChromaDB (meaning) and BM25 (exact words like "3(p)"); results are merged with Reciprocal Rank Fusion.
4. Rerank: a cross-encoder rescores the top 25 against the question and keeps 8.
5. CRAG: Groq grades each passage. Weak results trigger a rewritten query and a retry; if nothing is found, the system refuses and recommends an expert instead of guessing.
6. Generate: Groq answers only from the passages, citing a source for every sentence.
7. Verify: each sentence is checked against the exact text it cites; unsupported sentences are removed.
8. Confidence: 35% reranker score, 25% CRAG result, 30% citation support, 10% whether sources are current.
9. Translate the answer back and flag it for escalation if confidence is low.

## Project layout

```
config.py            settings from .env
ingest.py            CLI
rag/loaders.py       PDF / DOCX / TXT / MD / HTML to page text (+ optional OCR)
rag/metadata.py      manifest + automatic jurisdiction / regime / language inference
rag/chunker.py       section-aware chunking with page tracking
rag/embeddings.py    multilingual embedder
rag/vectorstore.py   ChromaDB wrapper with versioning and filters
rag/ingestion.py     the ingestion pipeline
rag/llm.py           Groq client with retries and JSON mode
rag/language.py      language detection, Groq translator, Bhashini placeholder
rag/query.py         multi-query expansion and CRAG query rewriting
rag/bm25.py          keyword index built from ChromaDB
rag/retriever.py     hybrid search + Reciprocal Rank Fusion
rag/reranker.py      cross-encoder reranking
rag/crag.py          grading, retry, web fallback, refusal
rag/generator.py     cited answer generation
rag/verify.py        claim-by-claim citation check
rag/confidence.py    confidence score
rag/pipeline.py      everything above, end to end
ask.py               terminal Q&A
api.py               FastAPI server
```

## Tips

- Scanned gazette PDFs: set `ENABLE_OCR=true` and install Tesseract with `eng` and `hin` language packs.
- On a GPU, switch `EMBED_MODEL=BAAI/bge-m3` for stronger Indic retrieval. Changing the model means re-ingesting with `--reset`.
- Keep chunk size around 1000 to 1500 characters for statutes; sections are never mixed in one chunk.
