"""
Tests for MetadataExtractor class.
Tests metadata extraction, fingerprinting, and entity detection.
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.documind.extractors.metadata_extractor import MetadataExtractor


class TestMetadataExtractor:
    """Test suite for MetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a MetadataExtractor instance."""
        return MetadataExtractor()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample file for testing."""
        file_path = os.path.join(temp_dir, "sample.txt")
        with open(file_path, "w") as f:
            f.write("Sample content for testing metadata extraction.")
        return file_path

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return """# Security Policy

This document outlines security guidelines for employees.

## Password Requirements

1. Minimum 12 characters
2. Use special characters
3. Change every 90 days

Contact: security@example.com
Phone: 555-123-4567
Website: https://security.example.com

Last updated: 2025-01-15
"""

    # Basic Metadata Tests
    def test_extract_basic_metadata(self, extractor, sample_file):
        """Test basic metadata extraction."""
        with open(sample_file) as f:
            content = f.read()

        result = extractor.extract_basic_metadata(sample_file, content)

        assert "file_name" in result
        assert result["file_name"] == "sample.txt"
        assert "file_path" in result
        assert "file_size_bytes" in result
        assert "file_type" in result
        assert result["file_type"] == ".txt"

    def test_extract_word_count(self, extractor, sample_file):
        """Test word count extraction."""
        content = "One two three four five"

        result = extractor.extract_basic_metadata(sample_file, content)

        assert result["word_count"] == 5

    def test_extract_character_count(self, extractor, sample_file):
        """Test character count extraction."""
        content = "Hello World"

        result = extractor.extract_basic_metadata(sample_file, content)

        assert result["character_count"] == 11

    def test_extract_line_count(self, extractor, sample_file):
        """Test line count extraction."""
        content = "Line 1\nLine 2\nLine 3"

        result = extractor.extract_basic_metadata(sample_file, content)

        assert result["line_count"] == 3

    def test_extract_read_time(self, extractor, sample_file):
        """Test estimated read time calculation."""
        # 400 words should be ~2 minutes at 200 wpm
        content = " ".join(["word"] * 400)

        result = extractor.extract_basic_metadata(sample_file, content)

        assert result["estimated_read_time_minutes"] == 2

    # Structure Extraction Tests
    def test_extract_structure(self, extractor, sample_content):
        """Test structure extraction."""
        result = extractor.extract_structure(sample_content)

        assert "heading_count" in result
        assert "headings" in result
        assert result["heading_count"] >= 2

    def test_extract_headings(self, extractor):
        """Test heading extraction."""
        content = "# Heading 1\n\nContent\n\n## Heading 2\n\nMore content"

        result = extractor.extract_structure(content)

        assert len(result["headings"]) == 2
        assert result["headings"][0]["text"] == "Heading 1"
        assert result["headings"][0]["level"] == 1
        assert result["headings"][1]["level"] == 2

    def test_extract_list_items(self, extractor, sample_content):
        """Test list item detection."""
        result = extractor.extract_structure(sample_content)

        # Sample content has numbered list
        assert result["numbered_lists"] >= 3

    def test_detect_tables(self, extractor):
        """Test table detection."""
        content_with_table = "| A | B |\n|---|---|\n| 1 | 2 |"
        content_without_table = "Just regular text"

        result_with = extractor.extract_structure(content_with_table)
        result_without = extractor.extract_structure(content_without_table)

        assert result_with["has_tables"] is True
        assert result_without["has_tables"] is False

    # Entity Extraction Tests
    def test_extract_emails(self, extractor, sample_content):
        """Test email extraction."""
        result = extractor.extract_entities(sample_content)

        assert "emails" in result
        assert "security@example.com" in result["emails"]

    def test_extract_urls(self, extractor, sample_content):
        """Test URL extraction."""
        result = extractor.extract_entities(sample_content)

        assert "urls" in result
        assert "https://security.example.com" in result["urls"]

    def test_extract_phone_numbers(self, extractor, sample_content):
        """Test phone number extraction."""
        result = extractor.extract_entities(sample_content)

        assert "phone_numbers" in result
        assert any("555" in phone for phone in result["phone_numbers"])

    def test_extract_dates(self, extractor, sample_content):
        """Test date extraction."""
        result = extractor.extract_entities(sample_content)

        assert "dates" in result
        assert "2025-01-15" in result["dates"]

    def test_extract_various_date_formats(self, extractor):
        """Test extraction of various date formats."""
        content = """
        ISO: 2025-01-15
        US: 01/15/2025
        Written: January 15, 2025
        """

        result = extractor.extract_entities(content)

        assert len(result["dates"]) >= 3

    def test_entity_count(self, extractor, sample_content):
        """Test entity count calculation."""
        result = extractor.extract_entities(sample_content)

        assert "entity_count" in result
        assert result["entity_count"] > 0

    # Topic Extraction Tests
    def test_extract_topics(self, extractor, sample_content):
        """Test topic extraction."""
        result = extractor.extract_topics(sample_content)

        assert "suggested_topics" in result
        assert "security" in result["suggested_topics"]

    def test_extract_tags(self, extractor, sample_content):
        """Test tag extraction."""
        result = extractor.extract_topics(sample_content)

        assert "suggested_tags" in result
        assert len(result["suggested_tags"]) > 0

    def test_topic_keywords_detected(self, extractor):
        """Test various topic keywords are detected."""
        hr_content = "Employee benefits and vacation policy"
        eng_content = "API development and code deployment"

        hr_result = extractor.extract_topics(hr_content)
        eng_result = extractor.extract_topics(eng_content)

        assert "hr" in hr_result["suggested_topics"]
        assert "engineering" in eng_result["suggested_topics"]

    # Fingerprint Tests
    def test_generate_fingerprint(self, extractor):
        """Test fingerprint generation."""
        content = "Test content for fingerprinting"

        fingerprint = extractor.generate_fingerprint(content)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA-256 hex length

    def test_fingerprint_consistent(self, extractor):
        """Test fingerprint consistency."""
        content = "Same content"

        fp1 = extractor.generate_fingerprint(content)
        fp2 = extractor.generate_fingerprint(content)

        assert fp1 == fp2

    def test_fingerprint_different_for_different_content(self, extractor):
        """Test different content produces different fingerprints."""
        fp1 = extractor.generate_fingerprint("Content A")
        fp2 = extractor.generate_fingerprint("Content B")

        assert fp1 != fp2

    def test_fingerprint_normalization(self, extractor):
        """Test fingerprint normalization."""
        content1 = "Hello  World"  # Extra space
        content2 = "hello world"   # Different case

        fp1 = extractor.generate_fingerprint(content1, normalize=True)
        fp2 = extractor.generate_fingerprint(content2, normalize=True)

        assert fp1 == fp2

    def test_fingerprint_without_normalization(self, extractor):
        """Test fingerprint without normalization."""
        content1 = "Hello  World"
        content2 = "hello world"

        fp1 = extractor.generate_fingerprint(content1, normalize=False)
        fp2 = extractor.generate_fingerprint(content2, normalize=False)

        assert fp1 != fp2

    # Extract All Tests
    def test_extract_all(self, extractor, sample_file, sample_content):
        """Test complete metadata extraction."""
        result = extractor.extract_all(sample_file, sample_content)

        assert "basic" in result
        assert "structure" in result
        assert "entities" in result
        assert "topics" in result
        assert "fingerprint" in result

    def test_extract_all_with_format_metadata(self, extractor, sample_file, sample_content):
        """Test extract_all with format-specific metadata."""
        format_meta = {"page_count": 5, "author": "Test Author"}

        result = extractor.extract_all(sample_file, sample_content, format_meta)

        assert "format_metadata" in result
        assert result["format_metadata"]["page_count"] == 5

    def test_extract_all_without_format_metadata(self, extractor, sample_file, sample_content):
        """Test extract_all without format metadata."""
        result = extractor.extract_all(sample_file, sample_content)

        assert result["format_metadata"] == {}

    # Edge Cases
    def test_empty_content(self, extractor, sample_file):
        """Test handling of empty content."""
        result = extractor.extract_all(sample_file, "")

        assert result["basic"]["word_count"] == 0
        assert result["basic"]["character_count"] == 0

    def test_unicode_content(self, extractor, sample_file):
        """Test handling of Unicode content."""
        content = "Unicode: café, naïve, 日本語"

        result = extractor.extract_all(sample_file, content)

        assert result["basic"]["word_count"] > 0
        assert result["fingerprint"] is not None

    def test_special_characters_in_content(self, extractor, sample_file):
        """Test handling of special characters."""
        content = "Special: <>&\"'`~!@#$%^*()[]{}"

        result = extractor.extract_all(sample_file, content)

        assert result is not None
        assert result["fingerprint"] is not None
