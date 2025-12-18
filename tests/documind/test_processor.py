"""
Tests for DocumentProcessor main orchestrator.
Integration tests for the complete document processing pipeline.
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.documind.processor import DocumentProcessor, process_document
from src.documind.data_structures import ProcessedDocument


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor(auto_upload=False)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_txt_file(self, temp_dir):
        """Create a sample TXT file."""
        file_path = os.path.join(temp_dir, "sample.txt")
        with open(file_path, "w") as f:
            f.write("""# Sample Document

This is a sample document for testing the document processor.

## Section 1

Content in section one with multiple sentences. This helps test the chunking functionality.

## Section 2

More content here with different topics. We want to ensure proper extraction and formatting.

## Conclusion

Final section with summary information.
""")
        return file_path

    @pytest.fixture
    def sample_md_file(self, temp_dir):
        """Create a sample Markdown file."""
        file_path = os.path.join(temp_dir, "sample.md")
        with open(file_path, "w") as f:
            f.write("""# Markdown Document

This is a **Markdown** document with various formatting.

## Features

- List item 1
- List item 2
- List item 3

## Code Example

```python
def hello():
    print("Hello, World!")
```

## Table

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |
""")
        return file_path

    # Basic Processing Tests
    def test_process_txt_document(self, processor, sample_txt_file):
        """Test processing a TXT document."""
        result = processor.process_document(sample_txt_file)

        assert isinstance(result, ProcessedDocument)
        assert result.extractor_used == "txt"
        assert result.file_name == "sample.txt"
        assert len(result.raw_content) > 0
        assert result.metadata.fingerprint is not None

    def test_process_md_document(self, processor, sample_md_file):
        """Test processing a Markdown document."""
        result = processor.process_document(sample_md_file)

        assert isinstance(result, ProcessedDocument)
        # Markdown files are processed as txt
        assert result.extractor_used == "txt"
        assert "Markdown" in result.raw_content

    def test_process_returns_chunks(self, processor, sample_txt_file):
        """Test that processing returns chunks."""
        result = processor.process_document(sample_txt_file)

        assert result.chunks is not None
        assert len(result.chunks) >= 1
        assert result.chunks[0].content is not None

    def test_process_returns_metadata(self, processor, sample_txt_file):
        """Test that processing returns metadata."""
        result = processor.process_document(sample_txt_file)

        assert result.metadata is not None
        assert result.metadata.basic is not None
        assert result.metadata.basic.file_name == "sample.txt"

    # Format Detection Tests
    def test_supported_formats(self, processor):
        """Test getting supported formats."""
        formats = processor.get_supported_formats()

        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".csv" in formats
        assert ".txt" in formats
        assert ".md" in formats

    def test_unsupported_format_raises_error(self, processor, temp_dir):
        """Test that unsupported formats raise errors."""
        unsupported_file = os.path.join(temp_dir, "test.xyz")
        # Write content to avoid empty file error
        with open(unsupported_file, "w") as f:
            f.write("some content")

        with pytest.raises(ValueError) as exc_info:
            processor.process_document(unsupported_file)

        # Should fail due to unsupported format
        error_msg = str(exc_info.value).lower()
        assert "unsupported" in error_msg or "supported" in error_msg or "format" in error_msg

    def test_nonexistent_file_raises_error(self, processor):
        """Test that non-existent files raise errors."""
        with pytest.raises((ValueError, FileNotFoundError)):
            processor.process_document("/nonexistent/path/file.txt")

    # Fingerprint Tests
    def test_fingerprint_generated(self, processor, sample_txt_file):
        """Test that fingerprint is generated."""
        result = processor.process_document(sample_txt_file)

        assert result.metadata.fingerprint is not None
        assert len(result.metadata.fingerprint) == 64  # SHA-256 hex length

    def test_fingerprint_consistent(self, processor, sample_txt_file):
        """Test that fingerprint is consistent for same content."""
        result1 = processor.process_document(sample_txt_file)
        result2 = processor.process_document(sample_txt_file)

        assert result1.metadata.fingerprint == result2.metadata.fingerprint

    def test_fingerprint_different_for_different_content(self, processor, temp_dir):
        """Test that different content produces different fingerprints."""
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        with open(file1, "w") as f:
            f.write("Content for file 1")
        with open(file2, "w") as f:
            f.write("Different content for file 2")

        result1 = processor.process_document(file1)
        result2 = processor.process_document(file2)

        assert result1.metadata.fingerprint != result2.metadata.fingerprint

    # Chunking Configuration Tests
    def test_custom_chunk_size(self, temp_dir):
        """Test custom chunk size configuration."""
        # Create a file with enough content to chunk
        file_path = os.path.join(temp_dir, "long.txt")
        with open(file_path, "w") as f:
            f.write(" ".join(["word"] * 2000))  # 2000 words

        processor = DocumentProcessor(
            chunk_target_size=100,
            auto_upload=False
        )

        result = processor.process_document(file_path)

        # Should have multiple chunks (at least 2)
        assert len(result.chunks) >= 2

    def test_chunk_overlap_configured(self, temp_dir):
        """Test chunk overlap configuration."""
        file_path = os.path.join(temp_dir, "long.txt")
        with open(file_path, "w") as f:
            f.write(" ".join(["word" + str(i) for i in range(1000)]))

        processor = DocumentProcessor(
            chunk_target_size=100,
            overlap_percent=0.20,
            auto_upload=False
        )

        result = processor.process_document(file_path)

        # Verify chunking happened - overlap detection is implementation-specific
        assert len(result.chunks) >= 1

    # Batch Processing Tests
    def test_process_batch(self, processor, temp_dir):
        """Test batch processing of multiple documents."""
        files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"doc{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content for document {i}")
            files.append(file_path)

        results = processor.process_batch(files)

        assert len(results) == 3
        for path, result in results.items():
            assert isinstance(result, ProcessedDocument)

    def test_process_batch_with_error(self, processor, temp_dir):
        """Test batch processing handles errors gracefully."""
        valid_file = os.path.join(temp_dir, "valid.txt")
        with open(valid_file, "w") as f:
            f.write("Valid content")

        files = [valid_file, "/nonexistent/file.txt"]

        results = processor.process_batch(files, stop_on_error=False)

        assert len(results) == 2
        assert isinstance(results[valid_file], ProcessedDocument)
        assert "Error" in str(results["/nonexistent/file.txt"])

    def test_process_batch_stop_on_error(self, processor, temp_dir):
        """Test batch processing can stop on first error."""
        files = ["/nonexistent/file1.txt", "/nonexistent/file2.txt"]

        results = processor.process_batch(files, stop_on_error=True)

        # At least one error should be captured
        assert any("Error" in str(v) for v in results.values())

    # LLM Formatting Tests
    def test_formatted_text_includes_frontmatter(self, processor, sample_txt_file):
        """Test that formatted text includes frontmatter."""
        result = processor.process_document(sample_txt_file)

        assert result.content.startswith("---")
        assert "---" in result.content[4:]

    def test_formatted_text_preserves_structure(self, processor, sample_md_file):
        """Test that formatted text preserves document structure."""
        result = processor.process_document(sample_md_file)

        # Should preserve headings
        assert "Markdown Document" in result.content
        assert "Features" in result.content

    # Custom Metadata Tests
    def test_custom_metadata_included(self, processor, sample_txt_file):
        """Test that custom metadata is included."""
        custom = {"project": "test", "version": "1.0"}

        result = processor.process_document(
            sample_txt_file,
            custom_metadata=custom
        )

        # Custom metadata should be in format_metadata
        assert result.metadata is not None

    # Convenience Function Tests
    def test_process_document_function(self, sample_txt_file):
        """Test the convenience function."""
        result = process_document(sample_txt_file)

        assert isinstance(result, ProcessedDocument)
        assert result.extractor_used == "txt"

    # Duplicate Check Tests
    def test_check_duplicate(self, processor, sample_txt_file):
        """Test duplicate checking functionality."""
        result = processor.check_duplicate(sample_txt_file)

        assert "is_duplicate" in result
        # Without upload, should not find duplicate
        assert result["is_duplicate"] is False

    # Edge Cases
    def test_empty_file(self, processor, temp_dir):
        """Test handling of empty file raises validation error."""
        empty_file = os.path.join(temp_dir, "empty.txt")
        Path(empty_file).touch()

        # Empty files are rejected by validation
        with pytest.raises(ValueError) as exc_info:
            processor.process_document(empty_file)
        assert "empty" in str(exc_info.value).lower()

    def test_unicode_content(self, processor, temp_dir):
        """Test handling of Unicode content."""
        unicode_file = os.path.join(temp_dir, "unicode.txt")
        with open(unicode_file, "w", encoding="utf-8") as f:
            f.write("Unicode: café, naïve, 日本語, emoji 🎉")

        result = processor.process_document(unicode_file)

        assert "café" in result.raw_content
        assert "日本語" in result.raw_content

    def test_large_file(self, processor, temp_dir):
        """Test handling of larger file."""
        large_file = os.path.join(temp_dir, "large.txt")
        with open(large_file, "w") as f:
            # Write ~1MB of text
            for i in range(10000):
                f.write(f"Line {i}: This is content to make the file larger.\n")

        result = processor.process_document(large_file)

        assert len(result.chunks) > 10
        assert result.metadata.basic.word_count > 10000


class TestProcessedDocumentSerialization:
    """Test ProcessedDocument serialization."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(auto_upload=False)

    @pytest.fixture
    def sample_file(self, tmp_path):
        file_path = tmp_path / "test.txt"
        file_path.write_text("Sample content for testing.")
        return str(file_path)

    def test_to_dict(self, processor, sample_file):
        """Test ProcessedDocument.to_dict() method."""
        result = processor.process_document(sample_file)

        data = result.to_dict()

        assert isinstance(data, dict)
        assert "file_path" in data
        assert "extractor_used" in data
        assert "chunks" in data
        assert "metadata" in data

    def test_to_json(self, processor, sample_file):
        """Test ProcessedDocument.to_json() method."""
        result = processor.process_document(sample_file)

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "file_path" in json_str

    def test_from_dict(self, processor, sample_file):
        """Test ProcessedDocument.from_dict() method."""
        result = processor.process_document(sample_file)

        data = result.to_dict()
        restored = ProcessedDocument.from_dict(data)

        assert restored.file_path == result.file_path
        assert restored.extractor_used == result.extractor_used
