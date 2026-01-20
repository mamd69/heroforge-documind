"""
Document Fingerprinting
Generates SHA-256 hashes for duplicate detection
"""
import hashlib
from typing import Optional

def generate_fingerprint(content: str, normalize: bool = True) -> str:
    """
    Generate SHA-256 fingerprint of document content.

    Args:
        content: Document text content
        normalize: Whether to normalize whitespace before hashing

    Returns:
        Hexadecimal SHA-256 hash string
    """
    if normalize:
        # Normalize: lowercase, remove extra whitespace
        content = ' '.join(content.lower().split())

    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID."""
    combined = f"{document_id}:chunk:{chunk_index}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]

def content_hash(content: str, algorithm: str = 'sha256') -> str:
    """
    Generate hash of content using specified algorithm.

    Args:
        content: Text content
        algorithm: Hash algorithm (sha256, md5, sha1)

    Returns:
        Hexadecimal hash string
    """
    if algorithm == 'sha256':
        return hashlib.sha256(content.encode()).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(content.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(content.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
