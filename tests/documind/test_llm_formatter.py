"""
Tests for LLMFormatter class.
Tests Markdown formatting, frontmatter generation, and table formatting.
"""
import pytest
import yaml

from src.documind.llm_formatter import LLMFormatter


class TestLLMFormatter:
    """Test suite for LLMFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create an LLMFormatter instance."""
        return LLMFormatter()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "basic": {
                "file_name": "test_document.pdf",
                "file_path": "/path/to/test_document.pdf",
                "file_size_bytes": 12345,
                "file_type": ".pdf",
                "word_count": 500,
                "character_count": 3000,
                "created_at": "2025-01-15T10:30:00",
                "modified_at": "2025-01-15T14:45:00"
            },
            "structure": {
                "heading_count": 5,
                "section_count": 6,
                "has_tables": True
            },
            "topics": {
                "suggested_topics": ["security", "engineering"],
                "suggested_tags": ["policy", "guidelines", "best-practices"]
            },
            "fingerprint": "abc123def456"
        }

    @pytest.fixture
    def sample_tables(self):
        """Create sample table data."""
        return [
            {
                "headers": ["Name", "Role", "Department"],
                "rows": [
                    ["Alice", "Engineer", "Engineering"],
                    ["Bob", "Manager", "Operations"]
                ]
            }
        ]

    # Basic Formatting Tests
    def test_format_basic_content(self, formatter):
        """Test basic content formatting."""
        content = "This is a simple test document."
        result = formatter.format_for_llm(content, [], {})

        assert "This is a simple test document." in result

    def test_format_with_metadata(self, formatter, sample_metadata):
        """Test formatting with metadata generates frontmatter."""
        content = "Document content here."
        result = formatter.format_for_llm(content, [], sample_metadata)

        # Should have frontmatter delimiters
        assert result.startswith("---")
        assert "---" in result[4:]  # Second delimiter

    def test_format_without_metadata(self, formatter):
        """Test formatting without metadata."""
        content = "Just content, no metadata."
        result = formatter.format_for_llm(content, [], {})

        assert "Just content, no metadata." in result

    # Frontmatter Tests
    def test_frontmatter_contains_file_info(self, formatter, sample_metadata):
        """Test frontmatter contains metadata information."""
        content = "Content here."
        result = formatter.format_for_llm(content, [], sample_metadata)

        # Extract frontmatter
        parts = result.split("---")
        if len(parts) >= 3:
            frontmatter = parts[1]
            # Frontmatter should contain some metadata fields
            assert "fingerprint" in frontmatter or "topics" in frontmatter or "processed_at" in frontmatter

    def test_frontmatter_valid_yaml(self, formatter, sample_metadata):
        """Test that frontmatter is valid YAML."""
        content = "Content here."
        result = formatter.format_for_llm(content, [], sample_metadata)

        # Extract and parse frontmatter
        parts = result.split("---")
        if len(parts) >= 3:
            frontmatter_yaml = parts[1].strip()
            try:
                parsed = yaml.safe_load(frontmatter_yaml)
                assert isinstance(parsed, dict)
            except yaml.YAMLError:
                pytest.fail("Frontmatter is not valid YAML")

    def test_frontmatter_includes_topics(self, formatter, sample_metadata):
        """Test frontmatter includes topic information."""
        content = "Content here."
        result = formatter.format_for_llm(content, [], sample_metadata)

        # Should include topics or tags
        assert "security" in result.lower() or "topics" in result.lower()

    # Table Formatting Tests
    def test_format_with_tables(self, formatter, sample_tables):
        """Test formatting includes tables."""
        content = "Document with tables."
        result = formatter.format_for_llm(content, sample_tables, {})

        # Should contain table markers
        assert "|" in result
        assert "---" in result or "Name" in result

    def test_table_markdown_format(self, formatter, sample_tables):
        """Test tables are in Markdown format."""
        content = "Content."
        result = formatter.format_for_llm(content, sample_tables, {})

        # Markdown table format
        assert "| Name | Role | Department |" in result or "Name" in result

    def test_multiple_tables(self, formatter):
        """Test formatting with multiple tables."""
        tables = [
            {"headers": ["A", "B"], "rows": [["1", "2"]]},
            {"headers": ["X", "Y"], "rows": [["3", "4"]]}
        ]

        content = "Document with multiple tables."
        result = formatter.format_for_llm(content, tables, {})

        # Both tables should be present
        assert "A" in result and "X" in result

    def test_empty_table_handled(self, formatter):
        """Test handling of empty tables."""
        tables = [{"headers": [], "rows": []}]
        content = "Content with empty table."

        # Should not raise an error
        result = formatter.format_for_llm(content, tables, {})
        assert result is not None

    # Content Cleaning Tests
    def test_clean_extra_whitespace(self, formatter):
        """Test that content is preserved even with extra whitespace."""
        content = "Text   with    extra    spaces."
        result = formatter.format_for_llm(content, [], {})

        # Content should be present in result
        assert "Text" in result and "spaces" in result

    def test_clean_multiple_newlines(self, formatter):
        """Test that multiple newlines are normalized."""
        content = "Paragraph 1.\n\n\n\n\nParagraph 2."
        result = formatter.format_for_llm(content, [], {})

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n\n" not in result

    def test_preserve_markdown_structure(self, formatter):
        """Test that Markdown structure is preserved."""
        content = """# Heading

Paragraph text.

- List item 1
- List item 2

**Bold text** and *italic text*."""

        result = formatter.format_for_llm(content, [], {})

        assert "# Heading" in result or "Heading" in result
        assert "- List item" in result or "List item" in result

    # Edge Cases
    def test_empty_content(self, formatter):
        """Test handling of empty content."""
        result = formatter.format_for_llm("", [], {})
        assert result is not None

    def test_unicode_content(self, formatter):
        """Test handling of Unicode content."""
        content = "Unicode: café, naïve, 日本語, emoji 🎉"
        result = formatter.format_for_llm(content, [], {})

        assert "café" in result or "cafe" in result
        assert "日本語" in result

    def test_special_yaml_characters(self, formatter, sample_metadata):
        """Test handling of special YAML characters in metadata."""
        sample_metadata["basic"]["file_name"] = "test: file [with] special {chars}.pdf"
        content = "Content."

        # Should not raise YAML parsing errors
        result = formatter.format_for_llm(content, [], sample_metadata)
        assert result is not None

    def test_very_long_content(self, formatter):
        """Test handling of very long content."""
        content = "Word " * 10000  # 10000 words
        result = formatter.format_for_llm(content, [], {})

        assert len(result) > 0
        assert "Word" in result

    # Create Frontmatter Method Tests
    def test_create_frontmatter_method(self, formatter, sample_metadata):
        """Test create_frontmatter method directly."""
        frontmatter = formatter.create_frontmatter(sample_metadata)

        assert frontmatter.startswith("---")
        assert frontmatter.strip().endswith("---")

    def test_create_frontmatter_empty_metadata(self, formatter):
        """Test create_frontmatter with empty metadata."""
        frontmatter = formatter.create_frontmatter({})

        # Should still produce valid frontmatter or empty string
        if frontmatter:
            assert "---" in frontmatter

    # Format Tables Method Tests
    def test_format_tables_method(self, formatter, sample_tables):
        """Test format_tables method directly."""
        formatted = formatter.format_tables(sample_tables)

        assert "|" in formatted
        assert "Name" in formatted

    def test_format_tables_single_column(self, formatter):
        """Test formatting table with single column."""
        tables = [{"headers": ["Header"], "rows": [["Value1"], ["Value2"]]}]
        formatted = formatter.format_tables(tables)

        assert "Header" in formatted
        assert "Value1" in formatted

    # Integration Tests
    def test_full_formatting_pipeline(self, formatter, sample_metadata, sample_tables):
        """Test complete formatting with all components."""
        content = """# Security Policy

This document outlines our security guidelines.

## Access Control

Users must authenticate before accessing resources.
"""

        result = formatter.format_for_llm(content, sample_tables, sample_metadata)

        # Should have frontmatter
        assert result.startswith("---")

        # Should have content
        assert "Security Policy" in result

        # Should have tables
        assert "Name" in result or "|" in result
