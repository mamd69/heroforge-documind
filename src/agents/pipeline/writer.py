#!/usr/bin/env python3
"""
Database Writer Agent

Writes document chunks and vector embeddings to Supabase database with
transaction safety and error handling.

Schema:
    documents:
        - id (UUID)
        - title (TEXT)
        - content (TEXT)
        - file_type (VARCHAR)
        - metadata (JSONB)
        - created_at (TIMESTAMPTZ)

    document_chunks:
        - id (UUID)
        - document_id (UUID, FK to documents)
        - chunk_text (TEXT)
        - chunk_index (INTEGER)
        - embedding (vector 1536)
        - metadata (JSONB)
        - created_at (TIMESTAMPTZ)

Usage:
    python writer.py <input_json_file>

Input JSON format:
    {
        "document": {
            "title": "Document Title",
            "content": "Full document text...",
            "file_type": "pdf",
            "metadata": {...}
        },
        "chunks": [
            {
                "chunk_text": "Chunk content...",
                "chunk_index": 0,
                "embedding": [0.1, 0.2, ...],  # 1536 dimensions
                "metadata": {...}
            },
            ...
        ]
    }

Returns JSON:
    {
        "success": bool,
        "document_id": str (UUID),
        "chunks_written": int,
        "write_time_ms": float,
        "error": str (if success=False)
    }

Environment Variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_ANON_KEY: Supabase anonymous/service key
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid


def get_supabase_client():
    """
    Initialize Supabase client from environment variables.

    Returns:
        Supabase client instance

    Raises:
        ImportError: If supabase library not installed
        ValueError: If environment variables not set
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "Missing supabase library. Install with: pip install supabase"
        )

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url:
        raise ValueError(
            "SUPABASE_URL environment variable not set. "
            "Set it in .env file or export SUPABASE_URL=https://your-project.supabase.co"
        )

    if not key:
        raise ValueError(
            "SUPABASE_ANON_KEY environment variable not set. "
            "Set it in .env file or export SUPABASE_ANON_KEY=your-anon-key"
        )

    return create_client(url, key)


def validate_embedding(embedding: List[float]) -> bool:
    """
    Validate embedding dimensions and values.

    Args:
        embedding: List of float values representing vector

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(embedding, list):
        return False

    if len(embedding) != 1536:
        return False

    if not all(isinstance(x, (int, float)) for x in embedding):
        return False

    return True


def validate_input(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate input data structure.

    Args:
        data: Input dictionary with document and chunks

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check document exists
    if "document" not in data:
        return False, "Missing 'document' field in input"

    doc = data["document"]

    # Validate required document fields
    required_doc_fields = ["title", "content", "file_type"]
    for field in required_doc_fields:
        if field not in doc:
            return False, f"Missing required document field: {field}"

    # Check chunks exist
    if "chunks" not in data:
        return False, "Missing 'chunks' field in input"

    if not isinstance(data["chunks"], list):
        return False, "'chunks' must be a list"

    if len(data["chunks"]) == 0:
        return False, "No chunks provided"

    # Validate each chunk
    for i, chunk in enumerate(data["chunks"]):
        required_chunk_fields = ["chunk_text", "chunk_index", "embedding"]

        for field in required_chunk_fields:
            if field not in chunk:
                return False, f"Chunk {i}: missing required field '{field}'"

        # Validate embedding dimensions
        if not validate_embedding(chunk["embedding"]):
            return False, f"Chunk {i}: invalid embedding (must be 1536 floats)"

    return True, None


def write_to_database(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write document and chunks to Supabase with transaction safety.

    This function performs the following operations:
    1. Validates input data
    2. Connects to Supabase
    3. Inserts document record
    4. Inserts all chunk records with embeddings
    5. Returns success status and document ID

    Transaction behavior:
    - Supabase client automatically handles transactions
    - If document insert fails, nothing is committed
    - If chunk insert fails, entire transaction rolls back
    - On success, all records are committed atomically

    Args:
        data: Dictionary containing document and chunks data

    Returns:
        Dictionary with:
        - success: bool
        - document_id: str (UUID) if successful
        - chunks_written: int
        - write_time_ms: float
        - error: str if failed
    """
    start_time = time.time()

    # Validate input
    is_valid, error = validate_input(data)
    if not is_valid:
        return {
            "success": False,
            "error": f"Validation failed: {error}"
        }

    # Initialize client
    try:
        client = get_supabase_client()
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to connect to Supabase: {str(e)}"
        }

    document_id = None

    try:
        # Generate document ID
        document_id = str(uuid.uuid4())

        # Prepare document record
        doc_data = {
            "id": document_id,
            "title": data["document"]["title"],
            "content": data["document"]["content"],
            "file_type": data["document"]["file_type"],
            "metadata": data["document"].get("metadata", {})
        }

        # Insert document (first transaction)
        doc_response = client.table("documents").insert(doc_data).execute()

        if not doc_response.data:
            raise Exception("Document insert failed: no data returned")

        # Prepare chunk records (matching document_chunks schema)
        chunk_records = []
        for chunk in data["chunks"]:
            chunk_record = {
                "document_id": document_id,
                "content": chunk.get("chunk_text", chunk.get("text", "")),
                "chunk_index": chunk["chunk_index"],
                "embedding": chunk["embedding"],
                "word_count": chunk.get("word_count", len(chunk.get("chunk_text", chunk.get("text", "")).split())),
                "metadata": chunk.get("metadata", {})
            }
            chunk_records.append(chunk_record)

        # Batch insert chunks (second transaction)
        # Supabase handles batch inserts efficiently
        chunk_response = client.table("document_chunks").insert(chunk_records).execute()

        if not chunk_response.data:
            raise Exception("Chunk insert failed: no data returned")

        chunks_written = len(chunk_response.data)

        # Calculate write time
        write_time_ms = (time.time() - start_time) * 1000

        return {
            "success": True,
            "document_id": document_id,
            "chunks_written": chunks_written,
            "write_time_ms": round(write_time_ms, 2)
        }

    except Exception as e:
        error_msg = str(e)

        # If document was created but chunks failed, try to clean up
        if document_id:
            try:
                client.table("documents").delete().eq("id", document_id).execute()
                error_msg += " (document record rolled back)"
            except:
                error_msg += " (WARNING: orphaned document record may exist)"

        return {
            "success": False,
            "error": f"Database write failed: {error_msg}"
        }


def load_input_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON input file.

    Args:
        file_path: Path to JSON input file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """
    CLI entry point for standalone execution.

    Usage:
        python writer.py <input_json_file>

    Input JSON should contain:
        - document: {title, content, file_type, metadata}
        - chunks: [{chunk_text, chunk_index, embedding, metadata}, ...]

    Outputs JSON to stdout with write results.
    """
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python writer.py <input_json_file>"
        }), file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        # Load input data
        data = load_input_file(input_file)

        # Write to database
        result = write_to_database(data)

        # Output result as JSON
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)

    except FileNotFoundError as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }), file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(json.dumps({
            "success": False,
            "error": f"Invalid JSON in input file: {str(e)}"
        }), file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
