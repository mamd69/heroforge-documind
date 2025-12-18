"""
Intelligent Content Chunking
Splits content into overlapping chunks for RAG systems
"""
from typing import List, Dict, Optional
import re
import uuid


class ContentChunker:
    """Split content into intelligent chunks with overlap."""

    DEFAULT_TARGET_SIZE = 750  # words
    MIN_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1000
    OVERLAP_PERCENT = 0.10  # 10% overlap

    def __init__(self, target_size: int = None, overlap_percent: float = None):
        self.target_size = target_size or self.DEFAULT_TARGET_SIZE
        self.overlap_percent = overlap_percent or self.OVERLAP_PERCENT

    def chunk_content(self, content: str, document_id: str) -> List[Dict]:
        """
        Split content into overlapping chunks.

        Args:
            content: Markdown-formatted content
            document_id: Parent document ID

        Returns:
            List of Chunk dictionaries
        """
        # 1. Split into elements (sentences and headings)
        elements = self._split_into_elements(content)

        # 2. Build chunks respecting boundaries
        chunks = self._build_chunks(elements, document_id)

        return chunks

    def _split_into_elements(self, content: str) -> List[Dict]:
        """
        Split content into atomic elements (sentences, headings).

        Each element has:
        - text: str
        - type: 'heading' | 'sentence' | 'paragraph'
        - word_count: int
        - heading_level: int (for headings)
        """
        elements = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if it's a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2)
                elements.append({
                    'text': line,
                    'type': 'heading',
                    'word_count': len(heading_text.split()),
                    'heading_level': level,
                    'heading_text': heading_text
                })
                continue

            # Split line into sentences
            sentences = self._split_into_sentences(line)
            for sentence in sentences:
                if sentence.strip():
                    word_count = len(sentence.split())
                    elements.append({
                        'text': sentence,
                        'type': 'sentence',
                        'word_count': word_count,
                        'heading_level': None
                    })

        return elements

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, respecting common abbreviations.
        """
        # Handle common abbreviations that shouldn't trigger splits
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.', r'\1<DOT>', text)

        # Split on sentence boundaries (. ! ?)
        # Keep the punctuation with the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviation dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        return sentences

    def _build_chunks(self, elements: List[Dict], document_id: str) -> List[Dict]:
        """
        Build chunks from elements with overlap.

        Algorithm:
        1. Accumulate elements until target size reached
        2. If heading encountered and chunk is large enough, finalize chunk
        3. Add overlap from previous chunk
        4. Track current section heading
        """
        chunks = []
        current_chunk = []
        current_word_count = 0
        current_heading = None
        position = 0
        previous_chunk_elements = []

        for i, element in enumerate(elements):
            # Update current heading if we encounter one
            if element['type'] == 'heading':
                current_heading = element.get('heading_text', element['text'])

                # If we have accumulated content and hit a heading, finalize the chunk
                if current_chunk and current_word_count >= self.MIN_CHUNK_SIZE:
                    chunk = self._create_chunk(
                        current_chunk,
                        document_id,
                        len(chunks),
                        0,  # Will update total_chunks later
                        current_heading,
                        position,
                        len(previous_chunk_elements) > 0
                    )
                    chunks.append(chunk)
                    position += len(chunk['content']) + 1

                    # Save elements for overlap
                    previous_chunk_elements = self._get_overlap_elements(current_chunk)

                    # Start new chunk with overlap
                    current_chunk = previous_chunk_elements.copy()
                    current_word_count = sum(e['word_count'] for e in current_chunk)

            # Add current element
            current_chunk.append(element)
            current_word_count += element['word_count']

            # Check if chunk is ready to finalize
            if current_word_count >= self.target_size:
                # Look ahead for a good breaking point (heading or end of section)
                should_break = False

                # If we're at max size, break regardless
                if current_word_count >= self.MAX_CHUNK_SIZE:
                    should_break = True
                # If next element is a heading, break here
                elif i + 1 < len(elements) and elements[i + 1]['type'] == 'heading':
                    should_break = True
                # If we've reached target size and it's a natural break, break here
                elif current_word_count >= self.target_size and element['type'] == 'sentence':
                    should_break = True

                if should_break:
                    chunk = self._create_chunk(
                        current_chunk,
                        document_id,
                        len(chunks),
                        0,  # Will update total_chunks later
                        current_heading,
                        position,
                        len(previous_chunk_elements) > 0
                    )
                    chunks.append(chunk)
                    position += len(chunk['content']) + 1

                    # Save elements for overlap
                    previous_chunk_elements = self._get_overlap_elements(current_chunk)

                    # Start new chunk with overlap
                    current_chunk = previous_chunk_elements.copy()
                    current_word_count = sum(e['word_count'] for e in current_chunk)

        # Finalize last chunk if any content remains
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                document_id,
                len(chunks),
                0,  # Will update total_chunks later
                current_heading,
                position,
                len(previous_chunk_elements) > 0
            )
            chunks.append(chunk)

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks

        return chunks

    def _create_chunk(self, elements: List[Dict], document_id: str,
                     chunk_index: int, total_chunks: int,
                     section_heading: Optional[str],
                     start_pos: int, has_overlap: bool) -> Dict:
        """Create a chunk dictionary from accumulated elements."""
        # Join elements with appropriate spacing
        content_parts = []
        for element in elements:
            if element['type'] == 'heading':
                # Add extra newline before headings for readability
                if content_parts:
                    content_parts.append('\n')
                content_parts.append(element['text'])
            else:
                content_parts.append(element['text'])

        content = ' '.join(content_parts)
        word_count = sum(e['word_count'] for e in elements)

        # Extract metadata tags from headings in this chunk
        metadata_tags = []
        for element in elements:
            if element['type'] == 'heading':
                # Add heading text as a tag
                heading_text = element.get('heading_text', element['text'].lstrip('#').strip())
                metadata_tags.append(f"section:{heading_text}")

        return {
            "chunk_id": str(uuid.uuid4()),
            "document_id": document_id,
            "content": content,
            "word_count": word_count,
            "start_position": start_pos,
            "end_position": start_pos + len(content),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "has_overlap": has_overlap,
            "section_heading": section_heading,
            "metadata_tags": metadata_tags
        }

    def _get_overlap_elements(self, elements: List[Dict]) -> List[Dict]:
        """Get elements for overlap (last ~10% of words)."""
        total_words = sum(e['word_count'] for e in elements)
        target_overlap_words = int(total_words * self.overlap_percent)

        # Work backwards to accumulate overlap elements
        overlap_elements = []
        overlap_word_count = 0

        for element in reversed(elements):
            # Don't include headings in overlap
            if element['type'] == 'heading':
                break

            overlap_elements.insert(0, element)
            overlap_word_count += element['word_count']

            if overlap_word_count >= target_overlap_words:
                break

        return overlap_elements

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Calculate statistics about the chunks.

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_word_count': 0,
                'min_word_count': 0,
                'max_word_count': 0,
                'chunks_with_overlap': 0,
                'total_words': 0
            }

        word_counts = [c['word_count'] for c in chunks]
        chunks_with_overlap = sum(1 for c in chunks if c['has_overlap'])

        return {
            'total_chunks': len(chunks),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'chunks_with_overlap': chunks_with_overlap,
            'total_words': sum(word_counts)
        }
