"""FastAPI backend.

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BASE_DIR, retrieval, settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="IP-SAKTI Sahayak API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipeline = None
_lock = threading.Lock()
ESCALATIONS = BASE_DIR / "data" / "escalations.jsonl"
FEEDBACK = BASE_DIR / "data" / "feedback.jsonl"


def get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is None:
            from rag.pipeline import RAGPipeline
            _pipeline = RAGPipeline(settings, retrieval)
            _pipeline.warm_up()
    return _pipeline


@app.on_event("startup")
def _startup() -> None:
    try:
        get_pipeline()
        log.info("Pipeline ready: %d chunks in store", _pipeline.store.count())
    except Exception as exc:  # start anyway so /health can report the problem
        log.error("Pipeline failed to load: %s", exc)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    jurisdiction: Optional[Literal["IN", "INTL", "BOTH"]] = "IN"
    language: Optional[str] = Field(None, description="Optional hint, e.g. 'mr' for Marathi")
    product_category: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: Optional[str] = None
    rating: Literal["up", "down"]
    confidence: Optional[float] = None
    language: Optional[str] = None


class ResolveRequest(BaseModel):
    reply: str = Field(..., min_length=1, max_length=5000)


class EscalateRequest(BaseModel):
    question: str
    answer: Optional[str] = None
    expert_type: Optional[Literal["ip_attorney", "regulatory", "abs"]] = "ip_attorney"
    contact: Optional[str] = None
    language: Optional[str] = None
    confidence: Optional[float] = None


@app.get("/api/health")
def health():
    try:
        p = get_pipeline()
        return {"status": "ok", "chunks": p.store.count(), "llm": settings.groq_model,
                "embed_model": settings.embed_model, "reranker": settings.reranker_model}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return get_pipeline().answer(req.question, req.jurisdiction, req.language, req.product_category)
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/documents")
def documents():
    return get_pipeline().store.list_documents()


@app.post("/api/escalate")
def escalate(req: EscalateRequest):
    ticket = {"ticket_id": uuid.uuid4().hex[:10].upper(), "created_at": datetime.now().isoformat(timespec="seconds"),
              "status": "open", **req.model_dump()}
    _append(ESCALATIONS, ticket)
    return {"ticket_id": ticket["ticket_id"], "message": "An expert will review this within 48 to 72 hours."}


@app.get("/api/escalations")
def list_escalations():
    if not ESCALATIONS.exists():
        return []
    return [json.loads(line) for line in ESCALATIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    _append(FEEDBACK, {"created_at": datetime.now().isoformat(timespec="seconds"), **req.model_dump()})
    return {"ok": True}


@app.post("/api/escalations/{ticket_id}/resolve")
def resolve(ticket_id: str, req: ResolveRequest):
    if not ESCALATIONS.exists():
        raise HTTPException(404, "No tickets")
    rows = [json.loads(l) for l in ESCALATIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    found = False
    for r in rows:
        if r.get("ticket_id") == ticket_id:
            r.update(status="resolved", reply=req.reply, resolved_at=datetime.now().isoformat(timespec="seconds"))
            found = True
    if not found:
        raise HTTPException(404, "Ticket not found")
    ESCALATIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"ok": True}


@app.get("/api/files/{filename}")
def get_file(filename: str):
    """Serve an original document from data/raw so sources can be opened at the cited page."""
    raw = settings.raw_dir.resolve()
    for p in raw.rglob("*"):
        if p.is_file() and p.name == filename:
            if raw not in p.resolve().parents:
                break
            media = "application/pdf" if p.suffix.lower() == ".pdf" else None
            return FileResponse(p, media_type=media, filename=None if media else p.name,
                                headers={"Content-Disposition": f'inline; filename="{p.name}"'} if media else None)
    raise HTTPException(404, "File not found")


# Serve the web UI (put index.html in ./static) at http://localhost:8000/
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
