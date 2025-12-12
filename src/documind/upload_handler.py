"""
DocuMind Document Upload Handler
Demonstrates Skills, Subagents, and Hooks integration
"""

import json
from datetime import datetime
from pathlib import Path

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# Base directory for uploads - all file operations restricted to this directory
BASE_UPLOAD_DIR = Path.cwd()


def validate_file_path(file_path, base_dir=None):
    """
    Validates that file path is safe and exists.

    Args:
        file_path: Path to the file to validate
        base_dir: Base directory to restrict file access (defaults to BASE_UPLOAD_DIR)

    Returns:
        tuple: (is_valid: bool, error_message: str or None)

    Security checks:
    - Input type validation
    - Path traversal prevention via resolve()
    - Symlink rejection
    - Allowed file extensions
    - File existence
    - File size limits
    """
    # Input type validation
    if not isinstance(file_path, (str, Path)):
        return False, "Invalid input: file_path must be a string or Path"

    if base_dir is None:
        base_dir = BASE_UPLOAD_DIR

    base_dir = Path(base_dir).resolve()
    path = Path(file_path)

    # Resolve to absolute path to prevent traversal attacks
    try:
        resolved_path = path.resolve()
    except (OSError, ValueError) as e:
        return False, f"Invalid path: {e}"

    # Verify path is within allowed base directory
    try:
        resolved_path.relative_to(base_dir)
    except ValueError:
        return False, "Access denied: path outside allowed directory"

    # Reject symlinks to prevent symlink attacks
    if path.is_symlink():
        return False, "Symlinks are not allowed"

    # Validate file extension
    if resolved_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Invalid extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check if file exists
    if not resolved_path.exists():
        return False, "File does not exist"

    if not resolved_path.is_file():
        return False, "Path is not a file"

    # Check file size
    file_size = resolved_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB"

    return True, None


def read_document(file_path):
    """
    Reads document contents safely.

    Args:
        file_path: Path to the file to read

    Returns:
        tuple: (contents: str or None, error_message: str or None)

    Handles UTF-8 encoding with fallback to latin-1.
    """
    path = Path(file_path)

    # Handle PDF files differently (return placeholder for binary)
    if path.suffix.lower() == ".pdf":
        return "[PDF binary content - requires PDF parser]", None

    # Try UTF-8 first, then fallback to latin-1
    encodings = ["utf-8", "latin-1"]

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                contents = f.read()
            return contents, None
        except UnicodeDecodeError:
            continue
        except IOError as e:
            return None, f"Error reading file: {str(e)}"

    return None, "Unable to decode file with supported encodings"


def extract_metadata(file_path, contents):
    """
    Extracts metadata from document.

    Args:
        file_path: Path to the file
        contents: File contents as string

    Returns:
        dict: Metadata including file info and content statistics
    """
    path = Path(file_path)
    stat = path.stat()

    # Calculate content statistics
    lines = contents.split("\n") if contents else []
    words = contents.split() if contents else []

    metadata = {
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": stat.st_size,
        "file_size_human": _format_size(stat.st_size),
        "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "line_count": len(lines),
        "word_count": len(words),
        "character_count": len(contents) if contents else 0,
    }

    return metadata


def _format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def analyze_document(file_path):
    """
    Main function: orchestrates document analysis.

    Args:
        file_path: Path to the document to analyze

    Returns:
        dict: JSON-serializable analysis result with status, metadata, and content preview
    """
    result = {
        "status": "success",
        "file_path": file_path,
        "timestamp": datetime.now().isoformat(),
        "validation": None,
        "metadata": None,
        "content_preview": None,
        "error": None,
    }

    # Step 1: Validate file path
    is_valid, error = validate_file_path(file_path)
    result["validation"] = {"valid": is_valid, "error": error}

    if not is_valid:
        result["status"] = "error"
        result["error"] = f"Validation failed: {error}"
        return result

    # Step 2: Read document contents
    contents, error = read_document(file_path)

    if error:
        result["status"] = "error"
        result["error"] = f"Read failed: {error}"
        return result

    # Step 3: Extract metadata
    result["metadata"] = extract_metadata(file_path, contents)

    # Step 4: Add content preview (first 500 chars)
    if contents:
        preview_length = min(500, len(contents))
        result["content_preview"] = contents[:preview_length]
        if len(contents) > preview_length:
            result["content_preview"] += "..."

    return result


# Test the function
if __name__ == "__main__":
    # Create a sample document for testing
    test_doc = "test_document.txt"
    with open(test_doc, "w") as f:
        f.write("Sample document for DocuMind testing.\nThis is line 2.")

    # Analyze it
    result = analyze_document(test_doc)
    print(json.dumps(result, indent=2))
