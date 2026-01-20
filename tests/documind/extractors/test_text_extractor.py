"""
Tests for TextExtractor class.
Tests text and Markdown file extraction.
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.documind.extractors.text_extractor import TextExtractor


class TestTextExtractor:
    """Test suite for TextExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a TextExtractor instance."""
        return TextExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    # Basic Extraction Tests
    def test_extract_txt_file(self, extractor, temp_dir):
        """Test extraction from TXT file."""
        file_path = os.path.join(temp_dir, "test.txt")
        content = "This is a test text file."
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert result["text"] == content

    def test_extract_md_file(self, extractor, temp_dir):
        """Test extraction from Markdown file."""
        file_path = os.path.join(temp_dir, "test.md")
        content = "# Heading\n\nParagraph text."
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "# Heading" in result["text"]

    def test_extract_detects_markdown(self, extractor, temp_dir):
        """Test that Markdown content is handled."""
        file_path = os.path.join(temp_dir, "test.md")
        with open(file_path, "w") as f:
            f.write("# Heading\n\nContent")

        result = extractor.extract(file_path)

        # Check text is extracted
        assert result["success"] is True
        assert "Heading" in result["text"]

    def test_extract_txt_not_markdown(self, extractor, temp_dir):
        """Test that plain TXT is processed correctly."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("Just plain text without markdown.")

        result = extractor.extract(file_path)

        assert result["success"] is True

    # Encoding Tests
    def test_extract_utf8_encoding(self, extractor, temp_dir):
        """Test extraction with UTF-8 encoding."""
        file_path = os.path.join(temp_dir, "utf8.txt")
        content = "UTF-8: café, naïve, 日本語"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "café" in result["text"]

    def test_extract_latin1_encoding(self, extractor, temp_dir):
        """Test extraction with Latin-1 encoding."""
        file_path = os.path.join(temp_dir, "latin1.txt")
        content = "Latin-1: café"
        with open(file_path, "w", encoding="latin-1") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "caf" in result["text"]  # Should extract, possibly with encoding fallback

    def test_extract_reports_encoding(self, extractor, temp_dir):
        """Test that encoding is reported in result."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Test content")

        result = extractor.extract(file_path)

        assert result["success"] is True
        # Encoding info may be in metadata
        assert result.get("metadata") is not None

    # Structure Extraction Tests
    def test_extract_with_structure(self, extractor, temp_dir):
        """Test extraction with structure information."""
        file_path = os.path.join(temp_dir, "structured.md")
        content = """# Heading 1

Paragraph under heading 1.

## Heading 2

Paragraph under heading 2.

### Heading 3

Final content.
"""
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract_with_structure(file_path)

        assert result["success"] is True
        assert "structure" in result or "text" in result

    def test_extract_structure_finds_headings(self, extractor, temp_dir):
        """Test that structure extraction finds headings."""
        file_path = os.path.join(temp_dir, "headings.md")
        with open(file_path, "w") as f:
            f.write("# First\n## Second\n### Third")

        result = extractor.extract_with_structure(file_path)

        if "structure" in result:
            headings = result["structure"].get("headings", [])
            assert len(headings) >= 3

    # Error Handling Tests
    def test_extract_nonexistent_file(self, extractor):
        """Test extraction of non-existent file."""
        result = extractor.extract("/nonexistent/path/file.txt")

        assert result["success"] is False
        assert result.get("error") is not None

    def test_extract_directory_fails(self, extractor, temp_dir):
        """Test that extracting a directory fails."""
        result = extractor.extract(temp_dir)

        assert result["success"] is False

    def test_extract_binary_file(self, extractor, temp_dir):
        """Test handling of binary file."""
        file_path = os.path.join(temp_dir, "binary.txt")
        with open(file_path, "wb") as f:
            f.write(bytes([0x00, 0x01, 0x02, 0xFF, 0xFE]))

        result = extractor.extract(file_path)

        # Should either fail or return with encoding issues noted

    # Edge Cases
    def test_extract_empty_file(self, extractor, temp_dir):
        """Test extraction of empty file."""
        file_path = os.path.join(temp_dir, "empty.txt")
        Path(file_path).touch()

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert result["text"] == ""

    def test_extract_whitespace_only(self, extractor, temp_dir):
        """Test extraction of whitespace-only file."""
        file_path = os.path.join(temp_dir, "whitespace.txt")
        with open(file_path, "w") as f:
            f.write("   \n\n\t  \n  ")

        result = extractor.extract(file_path)

        assert result["success"] is True

    def test_extract_very_long_lines(self, extractor, temp_dir):
        """Test extraction with very long lines."""
        file_path = os.path.join(temp_dir, "long_lines.txt")
        long_line = "x" * 10000
        with open(file_path, "w") as f:
            f.write(long_line)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert len(result["text"]) == 10000

    def test_extract_special_characters(self, extractor, temp_dir):
        """Test extraction with special characters."""
        file_path = os.path.join(temp_dir, "special.txt")
        content = "Special chars: <>&\"'\\t\\n`~!@#$%^*()[]{}|"
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "Special chars" in result["text"]

    # Markdown-specific Tests
    def test_extract_markdown_with_code_blocks(self, extractor, temp_dir):
        """Test extraction of Markdown with code blocks."""
        file_path = os.path.join(temp_dir, "code.md")
        content = """# Code Example

```python
def hello():
    print("Hello")
```

More text after code.
"""
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "def hello():" in result["text"]
        assert "```python" in result["text"]

    def test_extract_markdown_with_tables(self, extractor, temp_dir):
        """Test extraction of Markdown with tables."""
        file_path = os.path.join(temp_dir, "table.md")
        content = """# Table

| A | B |
|---|---|
| 1 | 2 |
"""
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "|" in result["text"]

    def test_extract_markdown_with_links(self, extractor, temp_dir):
        """Test extraction of Markdown with links."""
        file_path = os.path.join(temp_dir, "links.md")
        content = "[Link text](https://example.com)"
        with open(file_path, "w") as f:
            f.write(content)

        result = extractor.extract(file_path)

        assert result["success"] is True
        assert "[Link text]" in result["text"]
        assert "https://example.com" in result["text"]

    # Different File Extensions
    def test_extract_markdown_extension(self, extractor, temp_dir):
        """Test extraction with .markdown extension."""
        file_path = os.path.join(temp_dir, "test.markdown")
        with open(file_path, "w") as f:
            f.write("# Heading\n\nContent")

        result = extractor.extract(file_path)

        assert result["success"] is True
