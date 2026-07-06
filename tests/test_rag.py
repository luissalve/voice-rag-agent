"""Unit tests for src/rag.py.

Demo v1 status: this covers `chunk_text`, the one pure/dependency-free
function in rag.py. It is a real, passing test — not a coverage claim for
the whole module. The embedding/retrieval/answer functions are unimplemented
skeletons (see rag.py) and are not exercised here because they have no
real behavior yet to assert against.
"""

from __future__ import annotations

import pytest

from src.rag import Chunk, chunk_text


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
    chunks = chunk_text("short page text", document_id="doc-2", page_number=1, chunk_size=800, overlap=150)
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
def test_chunk_text_rejects_invalid_size_overlap_combinations(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", document_id="doc-1", page_number=1, chunk_size=chunk_size, overlap=overlap)
