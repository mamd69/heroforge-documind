#!/usr/bin/env python3
"""
Chunker Agent - DocuMind Document Processing Pipeline

Splits extracted text into semantically meaningful chunks with overlap for better
context preservation. Optimized for 400-600 word chunks with sentence boundary
awareness.

Usage:
    # From stdin
    echo '{"document_id": "doc123", "content": "..."}' | python chunker.py

    # From file
    python chunker.py --input extracted.json

    # With custom parameters
    python chunker.py --chunk-size 500 --overlap 50 --input data.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class Chunk:
    """Represents a single document chunk with metadata."""
    chunk_id: str
    content: str
    word_count: int
    start_position: int
    end_position: int
    document_id: str
    chunk_index: int
    total_chunks: int
    has_overlap: bool


class TextChunker:
    """
    Intelligent text chunker that splits documents into semantic chunks.

    Features:
    - Sentence boundary detection
    - Configurable chunk size and overlap
    - Preserves context with overlap
    - Handles edge cases (short docs, long sentences)
    """

    # Sentence boundary patterns (supports multiple punctuation marks)
    SENTENCE_END_PATTERN = re.compile(
        r'([.!?]+[\s\'")\]]*)'  # Match sentence endings with optional quotes/brackets
    )

    # Word boundary pattern (handles contractions and hyphenated words)
    WORD_PATTERN = re.compile(r'\b[\w\'-]+\b')

    def __init__(self, target_chunk_size: int = 500, overlap_size: int = 50):
        """
        Initialize the text chunker.

        Args:
            target_chunk_size: Target number of words per chunk (default: 500)
            overlap_size: Number of words to overlap between chunks (default: 50)
        """
        if target_chunk_size < 100:
            raise ValueError("Chunk size must be at least 100 words")
        if overlap_size >= target_chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        if overlap_size < 0:
            raise ValueError("Overlap must be non-negative")

        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size

    def count_words(self, text: str) -> int:
        """Count words in text using regex pattern."""
        return len(self.WORD_PATTERN.findall(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving sentence boundaries.

        Args:
            text: Input text to split

        Returns:
            List of sentences with whitespace normalized
        """
        # Split on sentence boundaries
        parts = self.SENTENCE_END_PATTERN.split(text)

        # Recombine sentence content with its ending punctuation
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]

            # Clean and normalize whitespace
            sentence = ' '.join(sentence.split())
            if sentence:
                sentences.append(sentence)

        # Handle last part if no ending punctuation
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(' '.join(parts[-1].split()))

        return sentences

    def create_chunk(
        self,
        content: str,
        start_pos: int,
        chunk_idx: int,
        document_id: str,
        total_chunks: int,
        has_overlap: bool
    ) -> Chunk:
        """
        Create a Chunk object with metadata.

        Args:
            content: Chunk text content
            start_pos: Character position where chunk starts in original document
            chunk_idx: Zero-based index of this chunk
            document_id: ID of the source document
            total_chunks: Total number of chunks (placeholder, will be updated)
            has_overlap: Whether this chunk has overlap from previous chunk

        Returns:
            Chunk object with all metadata
        """
        chunk_id = f"{document_id}_chunk_{chunk_idx:04d}"
        word_count = self.count_words(content)
        end_pos = start_pos + len(content)

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            word_count=word_count,
            start_position=start_pos,
            end_position=end_pos,
            document_id=document_id,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            has_overlap=has_overlap
        )

    def chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into semantically meaningful chunks.

        Algorithm:
        1. Split text into sentences
        2. Group sentences into chunks targeting the desired word count
        3. Add overlap from previous chunk for context preservation
        4. Handle edge cases (very short documents, single long sentences)

        Args:
            text: Input text to chunk
            document_id: Unique identifier for the source document

        Returns:
            List of Chunk objects with metadata
        """
        # Handle empty or very short text
        text = text.strip()
        if not text:
            return []

        word_count = self.count_words(text)

        # If document is shorter than target chunk size, return as single chunk
        if word_count <= self.target_chunk_size:
            return [self.create_chunk(
                content=text,
                start_pos=0,
                chunk_idx=0,
                document_id=document_id,
                total_chunks=1,
                has_overlap=False
            )]

        # Split into sentences
        sentences = self.split_into_sentences(text)

        # Handle edge case: no sentences detected
        if not sentences:
            sentences = [text]

        chunks: List[Chunk] = []
        current_chunk_sentences: List[str] = []
        current_word_count = 0
        char_position = 0
        overlap_sentences: List[str] = []
        overlap_word_count = 0

        for sentence in sentences:
            sentence_word_count = self.count_words(sentence)

            # Check if adding this sentence would exceed target size
            if current_word_count > 0 and current_word_count + sentence_word_count > self.target_chunk_size:
                # Create chunk from accumulated sentences
                chunk_content = ' '.join(current_chunk_sentences)
                chunk = self.create_chunk(
                    content=chunk_content,
                    start_pos=char_position,
                    chunk_idx=len(chunks),
                    document_id=document_id,
                    total_chunks=0,  # Will be updated later
                    has_overlap=len(chunks) > 0
                )
                chunks.append(chunk)

                # Calculate overlap sentences for next chunk
                overlap_sentences = []
                overlap_word_count = 0

                # Take sentences from end of current chunk for overlap
                for overlap_sentence in reversed(current_chunk_sentences):
                    overlap_wc = self.count_words(overlap_sentence)
                    if overlap_word_count + overlap_wc <= self.overlap_size:
                        overlap_sentences.insert(0, overlap_sentence)
                        overlap_word_count += overlap_wc
                    else:
                        break

                # Move position forward (excluding overlap)
                char_position += len(chunk_content) - len(' '.join(overlap_sentences))
                if overlap_sentences:
                    char_position -= 1  # Account for space

                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences.copy()
                current_word_count = overlap_word_count

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count

        # Add final chunk if there are remaining sentences
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences)
            chunk = self.create_chunk(
                content=chunk_content,
                start_pos=char_position,
                chunk_idx=len(chunks),
                document_id=document_id,
                total_chunks=0,
                has_overlap=len(chunks) > 0
            )
            chunks.append(chunk)

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks

        return chunks

    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document from extractor output format.

        Expected input format:
        {
            "document_id": "unique_id",
            "content": "extracted text...",
            "metadata": {...}  # optional
        }

        Args:
            document: Document dictionary from extractor

        Returns:
            List of chunk dictionaries ready for JSON serialization
        """
        document_id = document.get('document_id', 'unknown')
        content = document.get('content', '')

        if not content:
            return []

        chunks = self.chunk_text(content, document_id)
        return [asdict(chunk) for chunk in chunks]


def process_stdin(chunker: TextChunker) -> List[Dict[str, Any]]:
    """Process document from stdin."""
    try:
        data = json.load(sys.stdin)

        # Handle single document or array of documents
        if isinstance(data, dict):
            return chunker.process_document(data)
        elif isinstance(data, list):
            all_chunks = []
            for doc in data:
                all_chunks.extend(chunker.process_document(doc))
            return all_chunks
        else:
            raise ValueError("Invalid input format: expected dict or list")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)


def process_file(chunker: TextChunker, file_path: Path) -> List[Dict[str, Any]]:
    """Process document from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle single document or array of documents
        if isinstance(data, dict):
            return chunker.process_document(data)
        elif isinstance(data, list):
            all_chunks = []
            for doc in data:
                all_chunks.extend(chunker.process_document(doc))
            return all_chunks
        else:
            raise ValueError("Invalid input format: expected dict or list")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for chunker agent."""
    parser = argparse.ArgumentParser(
        description='Chunk extracted text into semantic segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        type=Path,
        help='Input JSON file (reads from stdin if not provided)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file (writes to stdout if not provided)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Target chunk size in words (default: 500)'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Overlap size in words (default: 50)'
    )

    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics to stderr'
    )

    args = parser.parse_args()

    # Create chunker with specified parameters
    try:
        chunker = TextChunker(
            target_chunk_size=args.chunk_size,
            overlap_size=args.overlap
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Process input
    if args.input:
        chunks = process_file(chunker, args.input)
    else:
        chunks = process_stdin(chunker)

    # Print statistics if requested
    if args.stats and chunks:
        word_counts = [c['word_count'] for c in chunks]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)

        print(f"\n=== Chunking Statistics ===", file=sys.stderr)
        print(f"Total chunks: {len(chunks)}", file=sys.stderr)
        print(f"Average words per chunk: {avg_words:.1f}", file=sys.stderr)
        print(f"Min words: {min_words}", file=sys.stderr)
        print(f"Max words: {max_words}", file=sys.stderr)

        # Document-level stats
        docs = set(c['document_id'] for c in chunks)
        print(f"Documents processed: {len(docs)}", file=sys.stderr)

    # Output results
    indent = 2 if args.pretty else None
    output_json = json.dumps(chunks, indent=indent, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == '__main__':
    main()
