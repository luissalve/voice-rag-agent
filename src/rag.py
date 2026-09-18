"""Retrieval-augmented generation core for voice-rag-agent.

This module is the single place that decides what "grounded" means for both
front doors (web /ask and the voice webhook called by the ElevenLabs agent).

Demo v1 status: `chunk_text` is a real, dependency-free implementation and is
unit-tested in `tests/test_rag.py`. The embedding, retrieval, and answer
functions below are typed skeletons with clear TODOs — they define the
contract the rest of the app relies on, but do not yet call a live embedding
model, a live database, or the live Anthropic API. See README.md's
"Demo scope / roadmap" section for the honest real-vs-demo breakdown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single chunk of extracted PDF text, before embedding."""

    document_id: str
    page_number: int
    text: str
    chunk_index: int


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk plus its embedding vector, as stored in / retrieved from pgvector."""

    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class Citation:
    """A page-level pointer back to the source material for an answer."""

    document_id: str
    page_number: int


@dataclass(frozen=True)
class Answer:
    """The final grounded answer returned to either front door."""

    text: str
    citations: list[Citation]


def chunk_text(
    text: str,
    *,
    document_id: str,
    page_number: int,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Split raw page text into overlapping chunks.

    Pure function, no I/O — this is deliberately the one function this v1
    scaffold unit-tests directly (see tests/test_rag.py), since it has no
    external dependencies and is the cheapest place to catch regressions.

    Args:
        text: Raw extracted text for a single PDF page.
        document_id: Identifier of the parent document.
        page_number: 1-indexed page number this text came from.
        chunk_size: Target number of characters per chunk.
        overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        A list of Chunk objects. Empty input yields an empty list.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is negative or
            overlap >= chunk_size (which would prevent forward progress).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stripped = text.strip()
    if not stripped:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(stripped):
        end = min(start + chunk_size, len(stripped))
        chunk_text_value = stripped[start:end].strip()
        if chunk_text_value:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    page_number=page_number,
                    text=chunk_text_value,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        if end == len(stripped):
            break
        start += step

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Result is in [-1.0, 1.0]. A zero vector on either side returns 0.0 rather
    than raising — keeps ranking well-defined when a stub embedder emits a
    zero vector for an unknown token.

    Raises:
        ValueError: If `a` and `b` have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(
    query_embedding: list[float],
    candidates: Iterable[EmbeddedChunk],
) -> list[EmbeddedChunk]:
    """Rank `EmbeddedChunk`s by cosine similarity to the query, descending.

    Pure function — no DB, no network. Designed to sit between
    `retrieve_top_k` (which fetches candidates from pgvector) and
    `answer_with_citations` (which asks Claude). Splitting ranking out this
    way makes it trivially unit-testable with a deterministic fake embedder
    (see tests/conftest.py).

    Stable: candidates with identical similarity scores keep their input
    order in the output, so retrieval results stay reproducible.
    """
    scored = [(cosine_similarity(query_embedding, c.embedding), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed a batch of chunks with BGE-M3.

    TODO (v1 demo): call the actual BGE-M3 model (local via sentence-transformers
    or a hosted embedding endpoint, per EMBEDDING_MODEL in .env) instead of
    raising. Batch the calls; do not embed one chunk per request in production.
    """
    raise NotImplementedError(
        "embed_chunks: wire the BGE-M3 embedding client here (see README roadmap)"
    )


def embed_query(question: str) -> list[float]:
    """Embed a single user question with the same model used for chunks.

    TODO (v1 demo): must use the identical embedding model/config as
    embed_chunks, or cosine similarity against stored vectors is meaningless.
    """
    raise NotImplementedError(
        "embed_query: wire the BGE-M3 embedding client here (see README roadmap)"
    )


def retrieve_top_k(
    query_embedding: list[float],
    *,
    document_id: str | None = None,
    k: int = 5,
) -> list[EmbeddedChunk]:
    """Run a pgvector cosine-distance query and return the top-k closest chunks.

    TODO (v1 demo): wire the actual SQL query against the pgvector-backed
    table, e.g. `ORDER BY embedding <=> %s LIMIT %s`. Optionally filter by
    document_id once multi-document support exists (see README roadmap —
    out of scope for v1).
    """
    raise NotImplementedError(
        "retrieve_top_k: wire the pgvector query here (see README roadmap)"
    )


def answer_with_citations(question: str, retrieved: list[EmbeddedChunk]) -> Answer:
    """Ask Claude to answer strictly from the retrieved chunks, with citations.

    TODO (v1 demo): build a prompt that includes each retrieved chunk's text
    and page number, instructs Claude to answer only from that context and to
    say when it doesn't know, then call the Anthropic API and parse out which
    page numbers were actually used.
    """
    raise NotImplementedError(
        "answer_with_citations: wire the Anthropic Claude call here (see README roadmap)"
    )
