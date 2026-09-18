"""FastAPI service for voice-rag-agent.

Demo v1 status: this is a real FastAPI skeleton with typed request/response
models and input validation. The /ingest and /ask handlers call into
src/rag.py, whose embedding/retrieval/answer functions are themselves
skeletons (see rag.py's TODOs and README.md's "Demo scope / roadmap").
There is no auth on these endpoints — this app assumes a trusted local/dev
caller, per the README's honesty note. Do not deploy this as-is.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.rag import Answer, Citation, answer_with_citations, embed_query, retrieve_top_k

logger = logging.getLogger("voice_rag_agent")

app = FastAPI(
    title="voice-rag-agent",
    description="Demo v1 — ask a PDF by phone or web. See README.md for scope.",
    version="0.1.0",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB — arbitrary v1 ceiling, not tuned


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None


class CitationResponse(BaseModel):
    document_id: str
    page_number: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


def _citation_to_response(citation: Citation) -> CitationResponse:
    return CitationResponse(document_id=citation.document_id, page_number=citation.page_number)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF, chunk it, embed the chunks, and store them in pgvector.

    TODO (v1 demo): extract per-page text from the uploaded PDF (e.g. with
    pypdf), call chunk_text per page, call embed_chunks on the result, and
    persist the embedded chunks to the pgvector-backed table. Currently this
    handler validates input and returns a clear 501 rather than pretending
    the pipeline is wired.
    """
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the demo size limit")

    logger.info("received PDF upload: %s (%d bytes)", file.filename, len(contents))

    # TODO: extract_pdf_pages(contents) -> list[(page_number, text)]
    # TODO: chunks = [c for page_num, text in pages for c in chunk_text(text, document_id=doc_id, page_number=page_num)]
    # TODO: embedded = embed_chunks(chunks)
    # TODO: persist embedded chunks to pgvector, return the real document_id/chunk_count
    raise HTTPException(
        status_code=501,
        detail=(
            "Ingestion pipeline is a demo v1 skeleton — PDF text extraction, "
            "embedding, and pgvector storage are not yet wired. See rag.py TODOs."
        ),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question grounded in a previously ingested PDF.

    Shared by both front doors: called directly by the web client, and called
    as a tool by the ElevenLabs Conversational AI agent mid-call for the
    phone front door.

    TODO (v1 demo): call embed_query, then retrieve_top_k, then
    answer_with_citations, and return the real result. Currently returns a
    clear 501 rather than a fabricated answer.
    """
    logger.info("ask: document_id=%s question_len=%d", request.document_id, len(request.question))

    try:
        query_embedding = embed_query(request.question)
        retrieved = retrieve_top_k(query_embedding, document_id=request.document_id)
        result: Answer = answer_with_citations(request.question, retrieved)
        return AskResponse(
            answer=result.text,
            citations=[_citation_to_response(c) for c in result.citations],
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"RAG pipeline is a demo v1 skeleton: {exc}",
        ) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check — safe to call without any external dependencies."""
    return {"status": "ok"}
