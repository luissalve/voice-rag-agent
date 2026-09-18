"""Shared pytest fixtures for voice-rag-agent tests.

Kept dependency-light on purpose: anything that would force tests to hit a
real embedding model, a real database, or any paid API lives here behind a
fake, so the test path runs offline (no API keys, no network).
"""

from __future__ import annotations

import pytest

from src.rag import Chunk, EmbeddedChunk


class FakeEmbedder:
    """Deterministic, hand-coded embedder used by retrieval-ranking tests.

    Maps a small fixed vocabulary of keywords to orthogonal 4-D vectors so
    we can assert exactly which chunk a query should rank first. Anything
    outside the vocabulary maps to the zero vector (cosine similarity 0.0).

    The point is not to be a realistic embedder — it is to make retrieval
    ranking fully deterministic and reproducible, so unit tests can pin
    down the ranking contract without network or API keys.
    """

    VOCAB: dict[str, list[float]] = {
        "refund": [1.0, 0.0, 0.0, 0.0],
        "policy": [0.0, 1.0, 0.0, 0.0],
        "shipping": [0.0, 0.0, 1.0, 0.0],
        "warranty": [0.0, 0.0, 0.0, 1.0],
    }
    DIM = 4

    @classmethod
    def embed(cls, text: str) -> list[float]:
        lowered = text.lower()
        for keyword, vector in cls.VOCAB.items():
            if keyword in lowered:
                return list(vector)
        return [0.0] * cls.DIM


def make_chunk(
    text: str,
    *,
    document_id: str = "doc-1",
    page_number: int = 1,
    chunk_index: int = 0,
) -> Chunk:
    """Build a Chunk with sensible defaults for tests."""
    return Chunk(
        document_id=document_id,
        page_number=page_number,
        text=text,
        chunk_index=chunk_index,
    )


def make_embedded(
    text: str,
    *,
    document_id: str = "doc-1",
    page_number: int = 1,
    chunk_index: int = 0,
) -> EmbeddedChunk:
    """Build an EmbeddedChunk whose embedding comes from FakeEmbedder."""
    return EmbeddedChunk(
        chunk=make_chunk(
            text,
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
        ),
        embedding=FakeEmbedder.embed(text),
    )


@pytest.fixture
def fake_embedder() -> type[FakeEmbedder]:
    """The FakeEmbedder class, exposed as a fixture."""
    return FakeEmbedder


@pytest.fixture
def make_embedded():
    """Factory fixture: returns a function that builds EmbeddedChunks with FakeEmbedder embeddings.

    Usage::

        def test_x(make_embedded):
            chunk = make_embedded("refund info")
    """

    def factory(
        text: str,
        *,
        document_id: str = "doc-1",
        page_number: int = 1,
        chunk_index: int = 0,
    ) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk=Chunk(
                document_id=document_id,
                page_number=page_number,
                text=text,
                chunk_index=chunk_index,
            ),
            embedding=FakeEmbedder.embed(text),
        )

    return factory


@pytest.fixture
def sample_chunks(make_embedded) -> list[EmbeddedChunk]:
    """Three mutually-orthogonal embedded chunks, useful for ranking tests."""
    return [
        make_embedded("refund policy details"),
        make_embedded("shipping policy details"),
        make_embedded("warranty policy details"),
    ]
