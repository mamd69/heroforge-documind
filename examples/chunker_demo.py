#!/usr/bin/env python3
"""
Chunker Agent Demo

Demonstrates the chunker agent's capabilities with various examples.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agents.pipeline.chunker import TextChunker


def demo_basic_chunking():
    """Demonstrate basic text chunking."""
    print("=" * 70)
    print("Demo 1: Basic Text Chunking")
    print("=" * 70)

    chunker = TextChunker(target_chunk_size=100, overlap_size=20)

    text = """
    DocuMind is an intelligent document management system. It uses a multi-agent
    pipeline architecture to process documents efficiently. The pipeline consists
    of several specialized agents, each handling a specific task.

    The extractor agent reads documents and extracts text content. It supports
    multiple file formats including PDF, DOCX, and plain text files. The
    extraction process preserves document structure and metadata.

    After extraction, the chunker agent splits the text into semantic chunks.
    This is important for efficient processing by downstream components. The
    chunker respects sentence boundaries to maintain semantic coherence.

    The chunker also implements overlap between consecutive chunks. This overlap
    ensures that context is preserved across chunk boundaries. It's particularly
    useful for tasks like semantic search and question answering.

    Finally, the chunks are stored with comprehensive metadata. Each chunk includes
    its position in the original document, word count, and relationships to other
    chunks. This metadata enables sophisticated retrieval and analysis operations.
    """

    chunks = chunker.chunk_text(text.strip(), "demo_doc_001")

    print(f"\nOriginal text length: {len(text)} characters")
    print(f"Total word count: {chunker.count_words(text)}")
    print(f"Number of chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}:")
        print(f"  - Chunk ID: {chunk.chunk_id}")
        print(f"  - Word count: {chunk.word_count}")
        print(f"  - Has overlap: {chunk.has_overlap}")
        print(f"  - Position: {chunk.start_position}-{chunk.end_position}")
        print(f"  - Preview: {chunk.content[:100]}...")
        print()


def demo_sentence_boundaries():
    """Demonstrate sentence boundary detection."""
    print("=" * 70)
    print("Demo 2: Sentence Boundary Preservation")
    print("=" * 70)

    chunker = TextChunker(target_chunk_size=100, overlap_size=20)

    # Create text with clear sentences
    sentences = [
        "This is the first sentence.",
        "Here comes the second sentence.",
        "The third sentence is here.",
        "Fourth sentence follows.",
        "Fifth sentence arrives.",
        "Sixth sentence appears.",
        "Seventh sentence is present.",
        "Eighth sentence shows up.",
        "Ninth sentence emerges.",
        "The tenth and final sentence concludes."
    ]

    text = " ".join(sentences)
    chunks = chunker.chunk_text(text, "sentence_demo")

    print(f"\nOriginal sentences: {len(sentences)}")
    print(f"Chunks created: {len(chunks)}\n")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index + 1}:")
        print(f"  Words: {chunk.word_count}")
        print(f"  Content: {chunk.content}")
        print(f"  Ends with punctuation: {chunk.content.rstrip()[-1] in '.!?'}")
        print()


def demo_overlap():
    """Demonstrate overlap between chunks."""
    print("=" * 70)
    print("Demo 3: Chunk Overlap")
    print("=" * 70)

    chunker = TextChunker(target_chunk_size=150, overlap_size=30)

    text = """
    The overlap feature is crucial for maintaining context. When text is split
    into chunks, information near chunk boundaries can be fragmented. Overlap
    solves this problem by including the last few sentences of each chunk in
    the beginning of the next chunk. This ensures that important context is
    not lost during chunking. The overlap size can be configured based on
    specific requirements. For most applications, 10-15% overlap is sufficient.
    """

    chunks = chunker.chunk_text(text.strip(), "overlap_demo")

    print(f"Chunks created: {len(chunks)}")
    print(f"Overlap size: {chunker.overlap_size} words\n")

    for i in range(len(chunks) - 1):
        current = chunks[i]
        next_chunk = chunks[i + 1]

        print(f"Overlap between Chunk {i + 1} and Chunk {i + 2}:")
        print(f"  Chunk {i + 1} ends with: ...{current.content[-80:]}")
        print(f"  Chunk {i + 2} starts with: {next_chunk.content[:80]}...")

        # Find common words
        current_words = set(current.content.split()[-30:])
        next_words = set(next_chunk.content.split()[:30])
        common = current_words & next_words

        print(f"  Common words: {len(common)}")
        print()


def demo_json_io():
    """Demonstrate JSON input/output format."""
    print("=" * 70)
    print("Demo 4: JSON Input/Output")
    print("=" * 70)

    chunker = TextChunker(target_chunk_size=100, overlap_size=20)

    # Simulate document from extractor
    document = {
        "document_id": "json_demo_001",
        "content": """
        This demonstrates the JSON format used in the pipeline. The chunker
        accepts documents in JSON format with a document_id and content field.
        It processes the content and returns an array of chunks, each with
        complete metadata. This format enables seamless integration with other
        pipeline agents. The JSON output can be directly consumed by downstream
        components like the embedder and indexer agents.
        """.strip(),
        "metadata": {
            "source": "demo",
            "format": "text/plain"
        }
    }

    print("Input document:")
    print(json.dumps(document, indent=2))
    print()

    # Process document
    chunks = chunker.process_document(document)

    print(f"Output ({len(chunks)} chunks):")
    print(json.dumps(chunks[0], indent=2))  # Show first chunk as example
    print(f"... and {len(chunks) - 1} more chunks")
    print()


def demo_edge_cases():
    """Demonstrate handling of edge cases."""
    print("=" * 70)
    print("Demo 5: Edge Cases")
    print("=" * 70)

    chunker = TextChunker()

    # Test cases
    test_cases = [
        ("Empty text", ""),
        ("Single word", "Hello"),
        ("Short sentence", "This is short."),
        ("No punctuation", " ".join([f"word{i}" for i in range(20)])),
        ("Special chars", "Hello! How are you? I'm fine. 😊"),
    ]

    for name, text in test_cases:
        chunks = chunker.chunk_text(text, f"edge_{name.replace(' ', '_')}")
        print(f"{name}:")
        print(f"  Input: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Chunks created: {len(chunks)}")
        if chunks:
            print(f"  First chunk words: {chunks[0].word_count}")
        print()


def demo_performance():
    """Demonstrate performance characteristics."""
    print("=" * 70)
    print("Demo 6: Performance Test")
    print("=" * 70)

    import time

    chunker = TextChunker(target_chunk_size=500, overlap_size=50)

    # Generate realistic document (2000 words)
    paragraph = """
    This is a performance test for the chunker agent. We're generating a
    realistic document to measure processing speed. The chunker should be
    able to process this document in less than one second. This requirement
    ensures that the pipeline can handle large document volumes efficiently.
    The chunker uses optimized algorithms for sentence detection and word
    counting to achieve high performance.
    """

    # Repeat to create ~2000 word document
    text = (paragraph * 50).strip()

    print(f"Test document size: {len(text)} characters")
    print(f"Test document words: {chunker.count_words(text)}")

    # Measure performance
    start_time = time.time()
    chunks = chunker.chunk_text(text, "perf_test")
    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Processing time: {elapsed:.3f} seconds")
    print(f"  Performance requirement: < 1.0 second")
    print(f"  Status: {'✓ PASS' if elapsed < 1.0 else '✗ FAIL'}")

    # Show chunk size distribution
    word_counts = [c.word_count for c in chunks]
    print(f"\nChunk size statistics:")
    print(f"  Average: {sum(word_counts) / len(word_counts):.1f} words")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    print(f"  Target: 400-600 words")


def main():
    """Run all demonstrations."""
    demos = [
        demo_basic_chunking,
        demo_sentence_boundaries,
        demo_overlap,
        demo_json_io,
        demo_edge_cases,
        demo_performance,
    ]

    for i, demo in enumerate(demos):
        demo()
        if i < len(demos) - 1:
            print("\n" * 2)


if __name__ == '__main__':
    main()
