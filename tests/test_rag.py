"""Unit tests for src/rag.py.

Demo v1 status: this covers `chunk_text` (pure / dependency-free) and the
proposed `cosine_similarity` + `rank_by_similarity` ranking core, both of
which are deterministic and need no API keys, no network, and no database.
The embedding/retrieval/answer functions in src/rag.py are unimplemented
skeletons (NotImplementedError) and are not exercised here.
"""

from __future__ import annotations

import pytest

from src.rag import Chunk, chunk_text, cosine_similarity, rank_by_similarity


# --- chunk_text -----------------------------------------------------------


def test_chunk_text_splits_long_text_into_overlapping_chunks() -> None:
    """A page longer than chunk_size should split into multiple chunks that
    preserve original content and carry the right document/page metadata."""
    text = "A" * 500 + "B" * 500  # 1000 chars, clearly longer than chunk_size

    chunks = chunk_text(
        text,
        document_id="doc-1",
        page_number=3,
        chunk_size=600,
        overlap=100,
    )

    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.page_number == 3 for c in chunks)

    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # every chunk should be non-empty and within the requested size
    for c in chunks:
        assert 0 < len(c.text) <= 600

    # the tail of the text should show up in the last chunk (nothing lost off the end)
    assert chunks[-1].text.endswith("B")


def test_chunk_text_returns_empty_list_for_blank_input() -> None:
    assert chunk_text("   \n\t  ", document_id="doc-1", page_number=1) == []


def test_chunk_text_returns_single_chunk_for_short_input() -> None:
    chunks = chunk_text(
        "short page text", document_id="doc-2", page_number=1, chunk_size=800, overlap=150
    )
    assert len(chunks) == 1
    assert chunks[0].text == "short page text"
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),      # chunk_size must be positive
        (100, -1),   # overlap must not be negative
        (100, 100),  # overlap must be smaller than chunk_size
        (100, 150),  # overlap larger than chunk_size
    ],
)
def test_chunk_text_rejects_invalid_size_overlap_combinations(
    chunk_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_id="doc-1",
            page_number=1,
            chunk_size=chunk_size,
            overlap=overlap,
        )


# --- cosine_similarity ---------------------------------------------------


def test_cosine_similarity_identical_vectors_returns_one() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_returns_zero() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_returns_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_returns_zero_when_either_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0


def test_cosine_similarity_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# --- rank_by_similarity --------------------------------------------------


def test_rank_by_similarity_orders_chunks_by_descending_score(
    sample_chunks, fake_embedder
) -> None:
    """A 'refund' query should rank the refund chunk first, not shipping or warranty."""
    query = fake_embedder.embed("refund")

    ranked = rank_by_similarity(query, sample_chunks)

    assert [c.chunk.text for c in ranked] == [
        "refund policy details",
        "shipping policy details",
        "warranty policy details",
    ]


def test_rank_by_similarity_returns_empty_for_empty_candidates(fake_embedder) -> None:
    assert rank_by_similarity(fake_embedder.embed("refund"), []) == []


def test_rank_by_similarity_is_stable_for_identical_scores(sample_chunks) -> None:
    """When all candidates score 0.0 (zero-vector query hits every norm-zero branch),
    the input order should be preserved — retrieval results stay reproducible."""
    ranked = rank_by_similarity([0.0, 0.0, 0.0, 0.0], sample_chunks)
    assert [c.chunk.text for c in ranked] == [c.chunk.text for c in sample_chunks]


def test_rank_by_similarity_accepts_an_iterator(make_embedded) -> None:
    """Accepting an Iterable (not just a list) lets retrieve_top_k stream
    results from a DB cursor without materializing the full candidate set."""
    candidates = iter(
        [
            make_embedded("warranty info"),
            make_embedded("refund info"),
        ]
    )
    ranked = rank_by_similarity([1.0, 0.0, 0.0, 0.0], candidates)
    assert [c.chunk.text for c in ranked] == ["refund info", "warranty info"]
