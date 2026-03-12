import pytest
from unittest.mock import MagicMock
from ingestion.chunker import (
    _count_tokens,
    _build_chunk,
    _build_metadata,
    _split_text,
    _exceeds_token_limit,
    _flush_buffer,
    chunk_parent,
    chunk_children,
    ENCODER,
)
from ingestion.schemas import Chunk, ChunkMetadata


# ── helpers ──────────────────────────────────────────────────────

def _make_element(text: str, filename="test.pdf", filetype="application/pdf", page_number=1):
    """Create a mock Element with the given text and metadata."""
    element = MagicMock()
    element.text = text
    element.metadata.filename = filename
    element.metadata.filetype = filetype
    element.metadata.page_number = page_number
    return element


def _make_chunk(text: str, is_parent=True, parent_id=None):
    """Create a real Chunk for use in chunk_children tests."""
    return Chunk(
        text=text,
        chunk_id="parent-id-123",
        is_parent=is_parent,
        parent_id=parent_id,
        metadata=ChunkMetadata(
            filename="test.pdf",
            filetype="application/pdf",
            page_number=1,
        ),
    )


def _generate_text(token_count: int) -> str:
    """Generate a string that is exactly `token_count` tokens long.

    Uses space-separated 'word' repetitions: each 'word' = 1 token
    in cl100k_base when space-separated.
    """
    return " ".join(["word"] * token_count)


# ── _count_tokens ────────────────────────────────────────────────

class TestCountTokens:

    def test_empty_string(self):
        assert _count_tokens("") == 0

    def test_single_word(self):
        assert _count_tokens("hello") > 0

    def test_consistent_results(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert _count_tokens(text) == _count_tokens(text)

    def test_longer_text_has_more_tokens(self):
        short = "hello"
        long = "hello world this is a longer sentence with more tokens"
        assert _count_tokens(long) > _count_tokens(short)

    def test_known_token_count(self):
        text = _generate_text(10)
        assert _count_tokens(text) == 10


# ── _build_metadata ─────────────────────────────────────────────

class TestBuildMetadata:

    def test_extracts_filename(self):
        element = _make_element("text", filename="report.pdf")
        metadata = _build_metadata(element)
        assert metadata.filename == "report.pdf"

    def test_extracts_filetype(self):
        element = _make_element("text", filetype="application/pdf")
        metadata = _build_metadata(element)
        assert metadata.filetype == "application/pdf"

    def test_extracts_page_number(self):
        element = _make_element("text", page_number=5)
        metadata = _build_metadata(element)
        assert metadata.page_number == 5

    def test_returns_chunk_metadata_instance(self):
        element = _make_element("text")
        metadata = _build_metadata(element)
        assert isinstance(metadata, ChunkMetadata)


# ── _build_chunk ─────────────────────────────────────────────────

class TestBuildChunk:

    def test_sets_text(self):
        metadata = ChunkMetadata(filename="f", filetype="t", page_number=1)
        chunk = _build_chunk("hello world", metadata)
        assert chunk.text == "hello world"

    def test_generates_unique_ids(self):
        metadata = ChunkMetadata(filename="f", filetype="t", page_number=1)
        chunk1 = _build_chunk("a", metadata)
        chunk2 = _build_chunk("b", metadata)
        assert chunk1.chunk_id != chunk2.chunk_id

    def test_defaults_to_parent(self):
        metadata = ChunkMetadata(filename="f", filetype="t", page_number=1)
        chunk = _build_chunk("text", metadata)
        assert chunk.is_parent is True
        assert chunk.parent_id is None

    def test_child_chunk(self):
        metadata = ChunkMetadata(filename="f", filetype="t", page_number=1)
        chunk = _build_chunk("text", metadata, parent_id="p-123", is_parent=False)
        assert chunk.is_parent is False
        assert chunk.parent_id == "p-123"

    def test_returns_chunk_instance(self):
        metadata = ChunkMetadata(filename="f", filetype="t", page_number=1)
        chunk = _build_chunk("text", metadata)
        assert isinstance(chunk, Chunk)


# ── _split_text ──────────────────────────────────────────────────

class TestSplitText:

    def test_short_text_returns_single_segment(self):
        segments = _split_text("hello", size=100)
        assert len(segments) == 1
        assert segments[0] == "hello"

    def test_splits_at_size_boundary(self):
        text = _generate_text(200)
        segments = _split_text(text, size=100)
        assert len(segments) == 2

    def test_each_segment_within_size(self):
        text = _generate_text(250)
        segments = _split_text(text, size=100)
        for segment in segments:
            assert _count_tokens(segment) <= 100

    def test_preserves_all_content(self):
        text = _generate_text(150)
        segments = _split_text(text, size=100)
        total_tokens = sum(_count_tokens(s) for s in segments)
        assert total_tokens == 150

    def test_exact_multiple_of_size(self):
        text = _generate_text(200)
        segments = _split_text(text, size=100)
        assert len(segments) == 2
        assert _count_tokens(segments[0]) == 100
        assert _count_tokens(segments[1]) == 100

    def test_non_exact_multiple(self):
        text = _generate_text(150)
        segments = _split_text(text, size=100)
        assert len(segments) == 2
        assert _count_tokens(segments[0]) == 100
        assert _count_tokens(segments[1]) == 50


# ── _exceeds_token_limit ─────────────────────────────────────────

class TestExceedsTokenLimit:

    def test_short_text_does_not_exceed(self):
        assert _exceeds_token_limit("hello") is False

    def test_long_text_exceeds(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 5)
        text = _generate_text(10)
        assert _exceeds_token_limit(text) is True

    def test_exact_limit_does_not_exceed(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 10)
        text = _generate_text(10)
        assert _exceeds_token_limit(text) is False


# ── _flush_buffer ────────────────────────────────────────────────

class TestFlushBuffer:

    def test_below_limit_returns_text_and_none(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        element = _make_element("text")
        result_text, chunk = _flush_buffer("short text", element)
        assert result_text == "short text"
        assert chunk is None

    def test_at_limit_flushes(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 5)
        text = _generate_text(5)
        element = _make_element("text")
        result_text, chunk = _flush_buffer(text, element)
        assert result_text == ""
        assert chunk is not None
        assert isinstance(chunk, Chunk)

    def test_above_limit_flushes(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 5)
        text = _generate_text(10)
        element = _make_element("text")
        result_text, chunk = _flush_buffer(text, element)
        assert result_text == ""
        assert chunk is not None

    def test_flushed_chunk_has_correct_metadata(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 5)
        text = _generate_text(10)
        element = _make_element("text", filename="doc.pdf", page_number=3)
        _, chunk = _flush_buffer(text, element)
        assert chunk.metadata.filename == "doc.pdf"
        assert chunk.metadata.page_number == 3

    def test_flushed_chunk_contains_buffer_text(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 5)
        text = _generate_text(10)
        element = _make_element("text")
        _, chunk = _flush_buffer(text, element)
        assert chunk.text == text


# ── chunk_parent ─────────────────────────────────────────────────

class TestChunkParentValidInput:

    def test_single_small_element(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        elements = [_make_element("Hello world")]
        chunks = chunk_parent(elements)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_multiple_small_elements_merge(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        elements = [_make_element("First"), _make_element("Second"), _make_element("Third")]
        chunks = chunk_parent(elements)
        assert len(chunks) == 1
        assert "First" in chunks[0].text
        assert "Second" in chunks[0].text
        assert "Third" in chunks[0].text

    def test_elements_joined_with_double_newline(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        elements = [_make_element("AAA"), _make_element("BBB")]
        chunks = chunk_parent(elements)
        assert chunks[0].text == "AAA\n\nBBB"

    def test_large_elements_split_into_multiple_chunks(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        text = _generate_text(120)
        elements = [_make_element(text)]
        chunks = chunk_parent(elements)
        assert len(chunks) > 1

    def test_all_chunks_are_parents(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        text = _generate_text(200)
        elements = [_make_element(text)]
        chunks = chunk_parent(elements)
        assert all(c.is_parent for c in chunks)
        assert all(c.parent_id is None for c in chunks)

    def test_all_chunks_have_unique_ids(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        elements = [_make_element(_generate_text(100)) for _ in range(3)]
        chunks = chunk_parent(elements)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_oversized_single_element_is_split(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        text = _generate_text(120)
        elements = [_make_element(text)]
        chunks = chunk_parent(elements)
        assert len(chunks) >= 2


class TestChunkParentMetadata:

    def test_metadata_from_anchor_element(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        elements = [_make_element("text", filename="report.pdf", page_number=7)]
        chunks = chunk_parent(elements)
        assert chunks[0].metadata.filename == "report.pdf"
        assert chunks[0].metadata.page_number == 7

    def test_anchor_element_metadata_used_for_merged_chunk(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        e1 = _make_element("First", filename="a.pdf", page_number=1)
        e2 = _make_element("Second", filename="a.pdf", page_number=2)
        chunks = chunk_parent([e1, e2])
        assert chunks[0].metadata.page_number == 1


class TestChunkParentInvalidInput:

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="elements must not be empty"):
            chunk_parent([])


class TestChunkParentEdgeCases:

    def test_single_element_exactly_at_limit(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        text = _generate_text(50)
        elements = [_make_element(text)]
        chunks = chunk_parent(elements)
        # exactly at limit — flush triggers, so we get 1 chunk
        assert len(chunks) == 1

    def test_buffer_remainder_is_flushed(self, monkeypatch):
        """Elements that don't fill a full chunk should still be returned."""
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 1000)
        elements = [_make_element("leftover")]
        chunks = chunk_parent(elements)
        assert len(chunks) == 1
        assert chunks[0].text == "leftover"

    def test_many_tiny_elements_accumulate(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        elements = [_make_element("hi") for _ in range(100)]
        chunks = chunk_parent(elements)
        assert len(chunks) >= 2


# ── chunk_children ───────────────────────────────────────────────

class TestChunkChildrenValidInput:

    def test_produces_children(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        assert len(children) > 1

    def test_children_reference_parent_id(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        assert all(c.parent_id == parent.chunk_id for c in children)

    def test_children_are_not_parents(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        assert all(c.is_parent is False for c in children)

    def test_children_inherit_metadata(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        for child in children:
            assert child.metadata.filename == parent.metadata.filename
            assert child.metadata.filetype == parent.metadata.filetype
            assert child.metadata.page_number == parent.metadata.page_number

    def test_children_have_unique_ids(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        ids = [c.chunk_id for c in children]
        assert len(ids) == len(set(ids))

    def test_small_parent_produces_single_child(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 1000)
        parent = _make_chunk("Short text")
        children = chunk_children(parent)
        assert len(children) == 1

    def test_child_tokens_within_limit(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)
        parent = _make_chunk(_generate_text(50))
        children = chunk_children(parent)
        for child in children:
            assert _count_tokens(child.text) <= 10


# ── Integration: parent → children pipeline ─────────────────────

class TestParentChildPipeline:

    def test_full_pipeline(self, monkeypatch):
        monkeypatch.setattr("ingestion.chunker.settings.parent_chunk_size", 50)
        monkeypatch.setattr("ingestion.chunker.settings.child_chunk_size", 10)

        elements = [_make_element(_generate_text(120))]
        parents = chunk_parent(elements)
        assert len(parents) >= 2

        all_children = []
        for parent in parents:
            children = chunk_children(parent)
            all_children.extend(children)

        assert len(all_children) > len(parents)
        for child in all_children:
            assert child.is_parent is False
            assert child.parent_id in [p.chunk_id for p in parents]
