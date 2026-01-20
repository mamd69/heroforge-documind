"""
Format Detection and Path Validation
Handles file format detection and security validation
"""
from pathlib import Path
from typing import Tuple, Optional
import os


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""
    pass


class ValidationError(Exception):
    """Raised when file validation fails."""
    pass


class FormatDetector:
    """Detect and validate document formats."""

    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".xls": "xlsx",
        ".txt": "txt",
        ".md": "txt",
        ".markdown": "txt"
    }

    MIME_TYPES = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt": "text/plain"
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def detect_format(self, file_path: str) -> str:
        """
        Detect document format from file extension.

        Args:
            file_path: Path to document

        Returns:
            Format string (pdf, docx, csv, xlsx, txt)

        Raises:
            UnsupportedFormatError: If format not supported
            ValidationError: If file path is invalid
        """
        if not file_path:
            raise ValidationError("File path cannot be empty")

        # Convert to Path object for reliable extension handling
        path = Path(file_path)

        # Get extension in lowercase for case-insensitive matching
        extension = path.suffix.lower()

        if not extension:
            raise UnsupportedFormatError(
                f"File '{file_path}' has no extension. "
                f"Supported formats: {', '.join(sorted(set(self.SUPPORTED_FORMATS.values())))}"
            )

        # Look up format
        format_type = self.SUPPORTED_FORMATS.get(extension)

        if format_type is None:
            supported_exts = ', '.join(sorted(self.SUPPORTED_FORMATS.keys()))
            raise UnsupportedFormatError(
                f"Unsupported file format '{extension}'. "
                f"Supported extensions: {supported_exts}"
            )

        return format_type

    def validate_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path for security and existence.

        Args:
            file_path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)

        Security Checks:
            - Rejects path traversal attempts (..)
            - Rejects symbolic links
            - Validates file exists and is readable
            - Enforces size limits
        """
        if not file_path:
            return False, "File path cannot be empty"

        try:
            # Convert to absolute path to normalize
            abs_path = os.path.abspath(file_path)
            path = Path(abs_path)

            # Security: Check for path traversal attempts
            # Normalize the path and ensure it matches the absolute path
            if ".." in file_path or ".." in str(path):
                return False, "Path traversal detected: '..' not allowed in file paths"

            # Check if file exists
            if not path.exists():
                return False, f"File does not exist: {file_path}"

            # Check it's not a directory
            if path.is_dir():
                return False, f"Path is a directory, not a file: {file_path}"

            # Security: Reject symbolic links
            if path.is_symlink():
                return False, "Symbolic links are not allowed for security reasons"

            # Check file is readable
            if not os.access(abs_path, os.R_OK):
                return False, f"File is not readable: {file_path}"

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                return False, (
                    f"File size ({actual_mb:.2f}MB) exceeds maximum "
                    f"allowed size ({max_mb}MB)"
                )

            # Check for zero-byte files
            if file_size == 0:
                return False, "File is empty (0 bytes)"

            return True, None

        except OSError as e:
            return False, f"Error accessing file: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error validating path: {str(e)}"

    def get_mime_type(self, file_path: str) -> str:
        """
        Get MIME type for the file format.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string

        Raises:
            UnsupportedFormatError: If format not supported
        """
        format_type = self.detect_format(file_path)
        mime_type = self.MIME_TYPES.get(format_type)

        if mime_type is None:
            raise UnsupportedFormatError(
                f"No MIME type mapping for format: {format_type}"
            )

        return mime_type

    def validate_and_detect(self, file_path: str) -> Tuple[str, str]:
        """
        Combined validation and format detection.

        Args:
            file_path: Path to validate and detect

        Returns:
            Tuple of (format_type, mime_type)

        Raises:
            ValidationError: If validation fails
            UnsupportedFormatError: If format not supported
        """
        # Validate path first
        is_valid, error_msg = self.validate_path(file_path)
        if not is_valid:
            raise ValidationError(error_msg)

        # Detect format
        format_type = self.detect_format(file_path)
        mime_type = self.get_mime_type(file_path)

        return format_type, mime_type

    def is_supported(self, file_path: str) -> bool:
        """
        Check if file format is supported without raising exceptions.

        Args:
            file_path: Path to check

        Returns:
            True if format is supported, False otherwise
        """
        try:
            self.detect_format(file_path)
            return True
        except (UnsupportedFormatError, ValidationError):
            return False

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """
        Get list of all supported file extensions.

        Returns:
            Sorted list of supported extensions (with dots)
        """
        return sorted(cls.SUPPORTED_FORMATS.keys())

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """
        Get list of unique supported format types.

        Returns:
            Sorted list of format types
        """
        return sorted(set(cls.SUPPORTED_FORMATS.values()))
