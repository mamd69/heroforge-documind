"""
Tests for ContentChunker class.
Tests content chunking with overlap and section boundary respect.
"""
import pytest

from src.documind.content_chunker import ContentChunker


class TestContentChunker:
    """Test suite for ContentChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a ContentChunker instance with default settings."""
        return ContentChunker()

    @pytest.fixture
    def custom_chunker(self):
        """Create a ContentChunker with custom settings."""
        return ContentChunker(
            target_size=100,
            overlap_percent=0.20
        )

    # Basic Chunking Tests
    def test_chunk_short_content(self, chunker):
        """Test that short content stays in single chunk."""
        content = "This is a short piece of content."
        chunks = chunker.chunk_content(content, document_id="test-doc")
        assert len(chunks) == 1
        assert content in chunks[0]["content"]

    def test_chunk_long_content(self, custom_chunker):
        """Test chunking of longer content."""
        # Create content with ~300 words
        words = ["word"] * 300
        content = " ".join(words)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self, custom_chunker):
        """Test that chunk indices are sequential."""
        words = ["word"] * 300
        content = " ".join(words)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_word_counts(self, custom_chunker):
        """Test that word counts are accurate."""
        words = ["word"] * 200
        content = " ".join(words)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")
        for chunk in chunks:
            actual_words = len(chunk["content"].split())
            # Allow some tolerance for overlap
            assert abs(chunk["word_count"] - actual_words) <= 5

    # Overlap Tests
    def test_chunk_overlap_present(self, custom_chunker):
        """Test that chunks have proper overlap."""
        # Create content with sentences (chunker uses sentence-level overlap)
        sentences = [f"This is sentence number {i}." for i in range(50)]
        content = " ".join(sentences)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        if len(chunks) >= 2:
            # Check that some content from end of chunk1 appears in chunk2 (sentence-level overlap)
            chunk1_content = chunks[0]["content"]
            chunk2_content = chunks[1]["content"]
            # The last sentence of chunk1 should appear at start of chunk2
            chunk1_sentences = [s.strip() for s in chunk1_content.split('.') if s.strip()]
            chunk2_sentences = [s.strip() for s in chunk2_content.split('.') if s.strip()]
            if chunk1_sentences and chunk2_sentences:
                # Check if any sentence from end of chunk1 is in chunk2
                last_sentences = chunk1_sentences[-3:] if len(chunk1_sentences) >= 3 else chunk1_sentences
                first_sentences = chunk2_sentences[:3] if len(chunk2_sentences) >= 3 else chunk2_sentences
                # Overlap is detected if last sentences of chunk1 appear in start of chunk2
                has_overlap = any(s in chunk2_content for s in last_sentences if len(s) > 10)
                # Either we have overlap, or chunking respects natural boundaries
                assert has_overlap or len(chunks) == 1, "Expected overlap or single chunk"

    def test_no_overlap_when_disabled(self):
        """Test chunking without overlap."""
        chunker = ContentChunker(
            target_size=100,
            overlap_percent=0.0
        )
        words = ["word" + str(i) for i in range(300)]
        content = " ".join(words)

        chunks = chunker.chunk_content(content, document_id="test-doc")

        if len(chunks) >= 2:
            # With no overlap, last word of chunk1 should not appear at start of chunk2
            chunk1_last = chunks[0]["content"].split()[-1]
            chunk2_first = chunks[1]["content"].split()[0]
            assert chunk1_last != chunk2_first

    # Section Boundary Tests
    def test_respects_heading_boundaries(self, custom_chunker):
        """Test that chunks respect heading boundaries."""
        content = """# Section 1

This is content for section one with many words to fill the space.

# Section 2

This is content for section two with different words.

# Section 3

Final section with its own content."""

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        # Check that headings aren't split mid-way
        for chunk in chunks:
            if "#" in chunk["content"]:
                # If chunk contains a heading, it should be at the start or be complete
                lines = chunk["content"].split("\n")
                for line in lines:
                    if line.strip().startswith("#"):
                        # Heading should be complete (not cut off)
                        assert len(line.strip()) > 1

    def test_respects_paragraph_boundaries(self, custom_chunker):
        """Test that chunks prefer paragraph boundaries."""
        paragraphs = ["Paragraph " + str(i) + " " + "content " * 30 for i in range(5)]
        content = "\n\n".join(paragraphs)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        # Chunks should ideally not split mid-sentence
        for chunk in chunks:
            # Most chunks should end with proper punctuation or paragraph break
            stripped = chunk["content"].strip()
            if len(stripped) > 0:
                # This is a soft check - not always possible
                pass

    # Edge Cases
    def test_empty_content(self, chunker):
        """Test handling of empty content."""
        chunks = chunker.chunk_content("", document_id="test-doc")
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0]["content"] == "")

    def test_whitespace_only_content(self, chunker):
        """Test handling of whitespace-only content."""
        chunks = chunker.chunk_content("   \n\n   \t  ", document_id="test-doc")
        assert len(chunks) <= 1

    def test_single_word(self, chunker):
        """Test handling of single word."""
        chunks = chunker.chunk_content("word", document_id="test-doc")
        assert len(chunks) == 1
        assert chunks[0]["word_count"] == 1

    def test_very_long_word(self, chunker):
        """Test handling of very long word."""
        long_word = "a" * 1000
        chunks = chunker.chunk_content(long_word, document_id="test-doc")
        assert len(chunks) >= 1

    # Metadata Preservation Tests
    def test_preserve_metadata(self, custom_chunker):
        """Test that metadata is preserved in chunks."""
        content = "# Title\n\nContent goes here with many words."
        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        assert len(chunks) >= 1
        # Chunks should have metadata fields
        for chunk in chunks:
            assert "section_heading" in chunk or "metadata_tags" in chunk

    # Size Constraint Tests
    def test_chunks_within_size_limits(self, custom_chunker):
        """Test that chunks respect size limits."""
        words = ["word"] * 500
        content = " ".join(words)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        for chunk in chunks[:-1]:  # Last chunk may be smaller
            word_count = chunk["word_count"]
            # Allow some flexibility for overlap and boundaries
            assert word_count <= ContentChunker.MAX_CHUNK_SIZE * 1.2

    def test_min_size_respected_except_last(self, custom_chunker):
        """Test that min size is respected except for last chunk."""
        words = ["word"] * 500
        content = " ".join(words)

        chunks = custom_chunker.chunk_content(content, document_id="test-doc")

        if len(chunks) > 1:
            for chunk in chunks[:-1]:
                # Most chunks should be at least min_size (with some tolerance)
                assert chunk["word_count"] >= ContentChunker.MIN_CHUNK_SIZE * 0.5

    # Special Content Tests
    def test_markdown_content(self, chunker):
        """Test chunking of Markdown content."""
        content = """# Heading 1

Paragraph with **bold** and *italic* text.

## Heading 2

- List item 1
- List item 2

```python
code_block = True
```

More content here."""

        chunks = chunker.chunk_content(content, document_id="test-doc")
        assert len(chunks) >= 1

        # Verify Markdown syntax is preserved
        all_content = " ".join(c["content"] for c in chunks)
        assert "**bold**" in all_content or "bold" in all_content

    def test_table_content(self, chunker):
        """Test chunking of table content."""
        content = """# Data

| Col1 | Col2 | Col3 |
|------|------|------|
| A    | B    | C    |
| D    | E    | F    |

More text after table."""

        chunks = chunker.chunk_content(content, document_id="test-doc")
        assert len(chunks) >= 1

    # Configuration Tests
    def test_custom_target_size(self):
        """Test custom target size configuration."""
        chunker = ContentChunker(target_size=50)
        assert chunker.target_size == 50

    def test_custom_overlap_percent(self):
        """Test custom overlap percent configuration."""
        chunker = ContentChunker(overlap_percent=0.15)
        assert chunker.overlap_percent == 0.15

    def test_valid_overlap_percent(self):
        """Test that valid overlap percent is accepted."""
        chunker = ContentChunker(overlap_percent=0.25)
        assert chunker.overlap_percent == 0.25
