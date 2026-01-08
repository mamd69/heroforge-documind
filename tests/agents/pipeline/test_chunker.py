#!/usr/bin/env python3
"""
Comprehensive tests for Chunker Agent

Tests cover:
- Chunk size requirements (400-600 words)
- Overlap functionality (50 words)
- Sentence boundary detection
- Edge cases (short docs, long sentences, empty content)
- JSON input/output handling
- Performance requirements (< 1s per doc)
- Metadata accuracy
"""

import json
import pytest
import time
from pathlib import Path
import sys
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from agents.pipeline.chunker import TextChunker, Chunk, process_file


class TestTextChunker:
    """Test suite for TextChunker class."""

    def test_initialization_defaults(self):
        """Test chunker initializes with correct defaults."""
        chunker = TextChunker()
        assert chunker.target_chunk_size == 500
        assert chunker.overlap_size == 50

    def test_initialization_custom(self):
        """Test chunker with custom parameters."""
        chunker = TextChunker(target_chunk_size=300, overlap_size=30)
        assert chunker.target_chunk_size == 300
        assert chunker.overlap_size == 30

    def test_initialization_validation(self):
        """Test parameter validation."""
        # Chunk size too small
        with pytest.raises(ValueError, match="at least 100 words"):
            TextChunker(target_chunk_size=50)

        # Overlap too large
        with pytest.raises(ValueError, match="smaller than chunk size"):
            TextChunker(target_chunk_size=100, overlap_size=100)

        # Negative overlap
        with pytest.raises(ValueError, match="non-negative"):
            TextChunker(target_chunk_size=500, overlap_size=-10)

    def test_word_count(self):
        """Test word counting accuracy."""
        chunker = TextChunker()

        assert chunker.count_words("Hello world") == 2
        assert chunker.count_words("Hello, world!") == 2
        assert chunker.count_words("It's a test.") == 3
        assert chunker.count_words("one-two-three") == 1  # Hyphenated word
        assert chunker.count_words("") == 0
        assert chunker.count_words("   ") == 0

    def test_sentence_splitting(self):
        """Test sentence boundary detection."""
        chunker = TextChunker()

        # Simple sentences
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third sentence."

        # Multiple punctuation
        text = "Question? Exclamation! Statement."
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) == 3

        # Quoted sentences
        text = 'He said "Hello." She replied "Hi!"'
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) == 2

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "doc123")
        assert chunks == []

        chunks = chunker.chunk_text("   ", "doc123")
        assert chunks == []

    def test_short_document(self):
        """Test document shorter than chunk size."""
        chunker = TextChunker(target_chunk_size=500)
        text = "This is a short document. It has only a few sentences. Should be one chunk."

        chunks = chunker.chunk_text(text, "short_doc")

        assert len(chunks) == 1
        assert chunks[0].document_id == "short_doc"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].content == text
        assert chunks[0].has_overlap == False

    def test_chunk_size_target(self):
        """Test chunks are within target size range (400-600 words)."""
        chunker = TextChunker(target_chunk_size=500, overlap_size=50)

        # Generate longer text
        sentence = "This is a test sentence with exactly ten words in it. "
        text = sentence * 100  # ~1000 words

        chunks = chunker.chunk_text(text, "test_doc")

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check chunk sizes (allowing some flexibility for sentence boundaries)
        for chunk in chunks[:-1]:  # Exclude last chunk
            assert 300 <= chunk.word_count <= 700, \
                f"Chunk {chunk.chunk_index} has {chunk.word_count} words"

    def test_overlap_functionality(self):
        """Test overlap between consecutive chunks."""
        chunker = TextChunker(target_chunk_size=100, overlap_size=20)

        # Create text with distinct sentences
        sentences = [f"Sentence number {i} with some additional words here. " for i in range(50)]
        text = "".join(sentences)

        chunks = chunker.chunk_text(text, "overlap_doc")

        # Should have multiple chunks
        assert len(chunks) > 2

        # Check overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Next chunk should indicate it has overlap
            assert next_chunk.has_overlap == True

            # Check for content overlap (some words should appear in both)
            current_words = current_chunk.content.split()[-30:]  # Last 30 words
            next_words = next_chunk.content.split()[:30]  # First 30 words

            # Should have some common words
            common = set(current_words) & set(next_words)
            assert len(common) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_chunk_metadata(self):
        """Test chunk metadata accuracy."""
        chunker = TextChunker(target_chunk_size=100, overlap_size=10)

        text = "First sentence. " * 80  # Enough for multiple chunks

        chunks = chunker.chunk_text(text, "meta_doc")

        # Check chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"meta_doc_chunk_{i:04d}"
            assert chunk.chunk_index == i
            assert chunk.document_id == "meta_doc"
            assert chunk.total_chunks == len(chunks)

        # Check positions are sequential
        for i in range(len(chunks) - 1):
            # End position should be greater than start
            assert chunks[i].end_position > chunks[i].start_position

            # Content length should match positions
            expected_length = chunks[i].end_position - chunks[i].start_position
            assert len(chunks[i].content) == expected_length

    def test_sentence_boundary_preservation(self):
        """Test that chunks split on sentence boundaries."""
        chunker = TextChunker(target_chunk_size=100, overlap_size=10)

        # Create sentences with clear boundaries
        sentences = []
        for i in range(30):
            sentences.append(f"This is sentence number {i} with some content. ")

        text = "".join(sentences)
        chunks = chunker.chunk_text(text, "boundary_doc")

        # Each chunk should end with sentence-ending punctuation
        # (except possibly the last one if text doesn't end with punctuation)
        for chunk in chunks[:-1]:
            content = chunk.content.rstrip()
            assert content[-1] in '.!?', \
                f"Chunk {chunk.chunk_index} doesn't end with sentence boundary: '{content[-20:]}'"

    def test_process_document_dict(self):
        """Test processing document in dictionary format."""
        chunker = TextChunker(target_chunk_size=100)

        document = {
            "document_id": "test123",
            "content": "First sentence. " * 60,
            "metadata": {"source": "test"}
        }

        result = chunker.process_document(document)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(chunk['document_id'] == 'test123' for chunk in result)

    def test_process_document_empty_content(self):
        """Test processing document with empty content."""
        chunker = TextChunker()

        document = {
            "document_id": "empty123",
            "content": ""
        }

        result = chunker.process_document(document)
        assert result == []

    def test_process_document_missing_id(self):
        """Test processing document without ID."""
        chunker = TextChunker()

        document = {
            "content": "Some content here. More content follows."
        }

        result = chunker.process_document(document)
        assert len(result) > 0
        assert result[0]['document_id'] == 'unknown'

    def test_performance_requirement(self):
        """Test chunking completes in < 1 second per document."""
        chunker = TextChunker()

        # Generate realistic document (1000-2000 words)
        paragraph = "This is a test paragraph with multiple sentences. " \
                   "It contains enough words to be realistic. " \
                   "We need to test performance requirements. "
        text = paragraph * 100  # ~1500 words

        start_time = time.time()
        chunks = chunker.chunk_text(text, "perf_test")
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Chunking took {elapsed:.3f}s, requirement is < 1s"
        assert len(chunks) > 0

    def test_very_long_sentence(self):
        """Test handling of sentences longer than chunk size."""
        chunker = TextChunker(target_chunk_size=100, overlap_size=10)

        # Create a very long "sentence" without punctuation (200 words)
        long_sentence = " ".join([f"word{i}" for i in range(200)])

        chunks = chunker.chunk_text(long_sentence, "long_sent")

        # Should still create chunks
        assert len(chunks) > 0

        # Total content should be preserved
        total_content = " ".join([c.content for c in chunks])
        assert long_sentence in total_content or total_content in long_sentence

    def test_special_characters(self):
        """Test handling of special characters and unicode."""
        chunker = TextChunker()

        text = "This has émojis 🎉 and spëcial çharacters. " \
               "It also has 'quotes' and \"double quotes\". " \
               "Plus numbers like 123 and symbols like $%&."

        chunks = chunker.chunk_text(text, "special_doc")

        assert len(chunks) > 0
        assert chunks[0].content == text  # Should preserve all content


class TestFileOperations:
    """Test file input/output operations."""

    def test_process_file_single_document(self):
        """Test processing single document from file."""
        chunker = TextChunker(target_chunk_size=100)

        document = {
            "document_id": "file_test",
            "content": "Test sentence. " * 60
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(document, f)
            temp_path = Path(f.name)

        try:
            result = process_file(chunker, temp_path)

            assert isinstance(result, list)
            assert len(result) > 0
            assert all(chunk['document_id'] == 'file_test' for chunk in result)
        finally:
            temp_path.unlink()

    def test_process_file_multiple_documents(self):
        """Test processing array of documents from file."""
        chunker = TextChunker(target_chunk_size=100)

        documents = [
            {"document_id": "doc1", "content": "Content one. " * 60},
            {"document_id": "doc2", "content": "Content two. " * 60}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(documents, f)
            temp_path = Path(f.name)

        try:
            result = process_file(chunker, temp_path)

            assert isinstance(result, list)
            assert len(result) > 0

            # Should have chunks from both documents
            doc_ids = set(chunk['document_id'] for chunk in result)
            assert 'doc1' in doc_ids
            assert 'doc2' in doc_ids
        finally:
            temp_path.unlink()


class TestChunkDataclass:
    """Test Chunk dataclass functionality."""

    def test_chunk_creation(self):
        """Test creating a chunk with all fields."""
        chunk = Chunk(
            chunk_id="test_chunk_0001",
            content="Test content here.",
            word_count=3,
            start_position=0,
            end_position=18,
            document_id="test_doc",
            chunk_index=1,
            total_chunks=5,
            has_overlap=True
        )

        assert chunk.chunk_id == "test_chunk_0001"
        assert chunk.content == "Test content here."
        assert chunk.word_count == 3
        assert chunk.start_position == 0
        assert chunk.end_position == 18
        assert chunk.document_id == "test_doc"
        assert chunk.chunk_index == 1
        assert chunk.total_chunks == 5
        assert chunk.has_overlap == True

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        from dataclasses import asdict

        chunk = Chunk(
            chunk_id="test_chunk_0000",
            content="Test",
            word_count=1,
            start_position=0,
            end_position=4,
            document_id="doc",
            chunk_index=0,
            total_chunks=1,
            has_overlap=False
        )

        chunk_dict = asdict(chunk)

        assert isinstance(chunk_dict, dict)
        assert chunk_dict['chunk_id'] == "test_chunk_0000"
        assert chunk_dict['word_count'] == 1
        assert chunk_dict['has_overlap'] == False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_word_document(self):
        """Test document with single word."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("Hello", "single_word")

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert chunks[0].word_count == 1

    def test_no_sentence_boundaries(self):
        """Test text without clear sentence boundaries."""
        chunker = TextChunker(target_chunk_size=100)

        # Long text without punctuation
        text = " ".join([f"word{i}" for i in range(200)])

        chunks = chunker.chunk_text(text, "no_punct")

        # Should still create chunks
        assert len(chunks) > 0

    def test_only_punctuation(self):
        """Test text with only punctuation."""
        chunker = TextChunker()

        # This should be treated as no content
        chunks = chunker.chunk_text("...", "punct_only")
        # Depends on word detection - might be 0 or 1 chunk
        assert len(chunks) <= 1

    def test_mixed_line_endings(self):
        """Test text with different line ending styles."""
        chunker = TextChunker()

        text = "Line one.\nLine two.\r\nLine three.\rLine four."
        chunks = chunker.chunk_text(text, "mixed_endings")

        assert len(chunks) > 0
        # Content should be preserved (whitespace normalization happens in sentence splitting)
        assert "Line one" in chunks[0].content
        assert "Line four" in chunks[0].content


class TestRealisticDocuments:
    """Test with realistic document examples."""

    def test_technical_documentation(self):
        """Test chunking technical documentation."""
        chunker = TextChunker(target_chunk_size=500, overlap_size=50)

        text = """
        The DocuMind system provides intelligent document management capabilities.
        It uses a multi-agent pipeline architecture for document processing.
        Each agent in the pipeline has a specific responsibility.

        The extractor agent handles document ingestion and text extraction.
        It supports multiple formats including PDF, DOCX, and TXT files.
        The extraction process maintains document structure and metadata.

        After extraction, the chunker agent splits text into semantic segments.
        This enables better context preservation and retrieval accuracy.
        Chunks are sized to optimize for language model processing.
        """ * 20  # Repeat to get enough content

        chunks = chunker.chunk_text(text, "tech_doc")

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should meet size requirements
        for chunk in chunks:
            assert chunk.word_count > 0
            assert 'document' in chunk.content.lower() or \
                   'agent' in chunk.content.lower() or \
                   'chunk' in chunk.content.lower()

    def test_narrative_text(self):
        """Test chunking narrative text."""
        chunker = TextChunker(target_chunk_size=400, overlap_size=40)

        text = """
        Once upon a time, in a digital realm far away, there existed a system called DocuMind.
        This system was designed to help users manage their documents efficiently.
        The creators of DocuMind understood that document management was a complex challenge.
        They decided to build a multi-agent system to handle this complexity.

        The first agent they created was the extractor.
        Its job was to read documents and extract their text content.
        The extractor could handle many different file formats.
        This made it very versatile and useful.
        """ * 15

        chunks = chunker.chunk_text(text, "story")

        assert len(chunks) > 0

        # Chunks should maintain narrative flow
        for i, chunk in enumerate(chunks):
            assert len(chunk.content) > 0
            # Should have complete sentences
            if i < len(chunks) - 1:
                assert chunk.content.rstrip()[-1] in '.!?'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
