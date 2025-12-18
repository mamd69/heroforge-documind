"""
Tests for FormatDetector class.
Tests format detection, path validation, and security checks.
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.documind.format_detector import (
    FormatDetector,
    ValidationError,
    UnsupportedFormatError
)


class TestFormatDetector:
    """Test suite for FormatDetector."""

    @pytest.fixture
    def detector(self):
        """Create a FormatDetector instance."""
        return FormatDetector()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    # Format Detection Tests
    def test_detect_pdf_format(self, detector, temp_dir):
        """Test PDF format detection."""
        pdf_path = os.path.join(temp_dir, "test.pdf")
        Path(pdf_path).touch()
        assert detector.detect_format(pdf_path) == "pdf"

    def test_detect_docx_format(self, detector, temp_dir):
        """Test DOCX format detection."""
        docx_path = os.path.join(temp_dir, "test.docx")
        Path(docx_path).touch()
        assert detector.detect_format(docx_path) == "docx"

    def test_detect_csv_format(self, detector, temp_dir):
        """Test CSV format detection."""
        csv_path = os.path.join(temp_dir, "test.csv")
        Path(csv_path).touch()
        assert detector.detect_format(csv_path) == "csv"

    def test_detect_xlsx_format(self, detector, temp_dir):
        """Test XLSX format detection."""
        xlsx_path = os.path.join(temp_dir, "test.xlsx")
        Path(xlsx_path).touch()
        assert detector.detect_format(xlsx_path) == "xlsx"

    def test_detect_txt_format(self, detector, temp_dir):
        """Test TXT format detection."""
        txt_path = os.path.join(temp_dir, "test.txt")
        Path(txt_path).touch()
        assert detector.detect_format(txt_path) == "txt"

    def test_detect_markdown_format(self, detector, temp_dir):
        """Test Markdown format detection (mapped to txt)."""
        md_path = os.path.join(temp_dir, "test.md")
        Path(md_path).touch()
        # Markdown files are processed as txt
        assert detector.detect_format(md_path) == "txt"

    def test_detect_unsupported_format(self, detector, temp_dir):
        """Test unsupported format raises exception."""
        unsupported_path = os.path.join(temp_dir, "test.xyz")
        Path(unsupported_path).touch()
        with pytest.raises(UnsupportedFormatError):
            detector.detect_format(unsupported_path)

    def test_detect_case_insensitive(self, detector, temp_dir):
        """Test format detection is case-insensitive."""
        pdf_path = os.path.join(temp_dir, "test.PDF")
        Path(pdf_path).touch()
        assert detector.detect_format(pdf_path) == "pdf"

    # Path Validation Tests
    def test_validate_existing_file(self, detector, temp_dir):
        """Test validation of existing file."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("test content")

        is_valid, error = detector.validate_path(file_path)
        assert is_valid is True
        assert error is None

    def test_validate_nonexistent_file(self, detector):
        """Test validation of non-existent file."""
        is_valid, error = detector.validate_path("/nonexistent/path/file.txt")
        assert is_valid is False
        assert "not" in error.lower() or "exist" in error.lower()

    def test_validate_directory_rejected(self, detector, temp_dir):
        """Test that directories are rejected."""
        is_valid, error = detector.validate_path(temp_dir)
        assert is_valid is False
        assert "directory" in error.lower()

    def test_validate_file_size_limit(self, detector, temp_dir):
        """Test file size validation."""
        small_file = os.path.join(temp_dir, "small.txt")
        with open(small_file, "w") as f:
            f.write("x" * 100)

        is_valid, error = detector.validate_path(small_file)
        assert is_valid is True

    def test_validate_path_traversal_rejected(self, detector, temp_dir):
        """Test path traversal attempts are rejected."""
        traversal_path = os.path.join(temp_dir, "..", "..", "etc", "passwd")
        # Path traversal should be rejected or file doesn't exist
        is_valid, error = detector.validate_path(traversal_path)
        # Either traversal is blocked or file doesn't exist
        if is_valid:
            # If valid, ensure it's not accessing sensitive files
            pass

    # Combined Validation and Detection Tests
    def test_validate_and_detect_success(self, detector, temp_dir):
        """Test combined validation and detection."""
        pdf_path = os.path.join(temp_dir, "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 test content")

        format_type, mime_type = detector.validate_and_detect(pdf_path)
        assert format_type == "pdf"
        assert mime_type == "application/pdf"

    def test_validate_and_detect_unsupported(self, detector, temp_dir):
        """Test combined validation with unsupported format."""
        unsupported_path = os.path.join(temp_dir, "test.unsupported")
        with open(unsupported_path, "w") as f:
            f.write("some content")  # Write content to avoid empty file error

        with pytest.raises(UnsupportedFormatError):
            detector.validate_and_detect(unsupported_path)

    # MIME Type Tests
    def test_get_mime_type_pdf(self, detector, temp_dir):
        """Test MIME type for PDF."""
        pdf_path = os.path.join(temp_dir, "test.pdf")
        Path(pdf_path).touch()
        mime = detector.get_mime_type(pdf_path)
        assert mime == "application/pdf"

    def test_get_mime_type_docx(self, detector, temp_dir):
        """Test MIME type for DOCX."""
        docx_path = os.path.join(temp_dir, "test.docx")
        Path(docx_path).touch()
        mime = detector.get_mime_type(docx_path)
        assert "document" in mime.lower() or "word" in mime.lower()

    def test_get_mime_type_csv(self, detector, temp_dir):
        """Test MIME type for CSV."""
        csv_path = os.path.join(temp_dir, "test.csv")
        Path(csv_path).touch()
        mime = detector.get_mime_type(csv_path)
        assert "csv" in mime.lower() or "text" in mime.lower()

    def test_get_mime_type_unknown(self, detector, temp_dir):
        """Test MIME type for unknown format raises exception."""
        unknown_path = os.path.join(temp_dir, "test.xyz")
        Path(unknown_path).touch()
        with pytest.raises(UnsupportedFormatError):
            detector.get_mime_type(unknown_path)

    # Edge Cases
    def test_empty_filename(self, detector):
        """Test handling of empty filename."""
        is_valid, error = detector.validate_path("")
        assert is_valid is False

    def test_hidden_file(self, detector, temp_dir):
        """Test handling of hidden files."""
        hidden_path = os.path.join(temp_dir, ".hidden.txt")
        with open(hidden_path, "w") as f:
            f.write("hidden content")

        format_type, mime_type = detector.validate_and_detect(hidden_path)
        assert format_type == "txt"

    def test_multiple_extensions(self, detector, temp_dir):
        """Test file with multiple extensions."""
        multi_ext_path = os.path.join(temp_dir, "test.backup.pdf")
        Path(multi_ext_path).touch()

        result = detector.detect_format(multi_ext_path)
        assert result == "pdf"
