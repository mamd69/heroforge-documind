"""
Data structures for the Document Processing System.
All structures are JSON-serializable for database storage.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import hashlib


@dataclass
class BasicMetadata:
    """
    Basic file and content metadata.

    Attributes:
        file_name: Name of the file
        file_path: Full path to the file
        file_size_bytes: Size in bytes
        file_type: File extension (pdf, docx, csv, txt)
        created_at: Creation timestamp in ISO format
        modified_at: Last modification timestamp in ISO format
        word_count: Total number of words
        character_count: Total number of characters
        line_count: Total number of lines
        estimated_read_time_minutes: Estimated reading time
    """
    file_name: str
    file_path: str
    file_size_bytes: int
    file_type: str
    created_at: str
    modified_at: str
    word_count: int
    character_count: int
    line_count: int
    estimated_read_time_minutes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BasicMetadata':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class HeadingInfo:
    """
    Information about a document heading.

    Attributes:
        text: Heading text content
        level: Heading level (1-6 for H1-H6)
        line_number: Line number where heading appears
    """
    text: str
    level: int
    line_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeadingInfo':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class StructureMetadata:
    """
    Document structure analysis.

    Attributes:
        heading_count: Total number of headings
        headings: List of all headings with details
        section_count: Number of distinct sections
        list_items: Total list items (bulleted + numbered)
        numbered_lists: Count of numbered list items
        has_tables: Whether document contains tables
        table_count: Number of tables found
        max_heading_level: Deepest heading level used
    """
    heading_count: int
    headings: List[HeadingInfo]
    section_count: int
    list_items: int
    numbered_lists: int
    has_tables: bool
    table_count: int = 0
    max_heading_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'heading_count': self.heading_count,
            'headings': [h.to_dict() for h in self.headings],
            'section_count': self.section_count,
            'list_items': self.list_items,
            'numbered_lists': self.numbered_lists,
            'has_tables': self.has_tables,
            'table_count': self.table_count,
            'max_heading_level': self.max_heading_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureMetadata':
        """Create instance from dictionary."""
        headings_data = data.pop('headings', [])
        headings = [HeadingInfo.from_dict(h) for h in headings_data]
        return cls(headings=headings, **data)


@dataclass
class EntityMetadata:
    """
    Extracted entities from document content.

    Attributes:
        emails: List of email addresses found
        urls: List of URLs found
        dates: List of date strings found
        phone_numbers: List of phone numbers found
        entity_count: Total count of all entities
    """
    emails: List[str]
    urls: List[str]
    dates: List[str]
    phone_numbers: List[str]
    entity_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityMetadata':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class TopicMetadata:
    """
    Topic classification and tagging.

    Attributes:
        suggested_topics: AI-suggested topics based on content
        suggested_tags: Metadata tags for categorization
        auto_category: Automatically assigned category
    """
    suggested_topics: List[str]
    suggested_tags: List[str]
    auto_category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopicMetadata':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class EnrichedMetadata:
    """
    Comprehensive document metadata combining all metadata types.

    Attributes:
        basic: Basic file and content metadata
        structure: Document structure analysis
        entities: Extracted entities
        topics: Topic classification
        fingerprint: SHA-256 hash of content for deduplication
        format_metadata: Format-specific metadata (PDF pages, DOCX styles, etc.)
    """
    basic: BasicMetadata
    structure: StructureMetadata
    entities: EntityMetadata
    topics: TopicMetadata
    fingerprint: str
    format_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'basic': self.basic.to_dict(),
            'structure': self.structure.to_dict(),
            'entities': self.entities.to_dict(),
            'topics': self.topics.to_dict(),
            'fingerprint': self.fingerprint,
            'format_metadata': self.format_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedMetadata':
        """Create instance from dictionary."""
        return cls(
            basic=BasicMetadata.from_dict(data['basic']),
            structure=StructureMetadata.from_dict(data['structure']),
            entities=EntityMetadata.from_dict(data['entities']),
            topics=TopicMetadata.from_dict(data['topics']),
            fingerprint=data['fingerprint'],
            format_metadata=data.get('format_metadata', {})
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'EnrichedMetadata':
        """Create instance from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TableData:
    """
    Extracted table with metadata.

    Attributes:
        table_id: Unique identifier for the table
        page_or_location: Page number or location in document
        rows: Number of rows
        columns: Number of columns
        headers: Column headers
        data: 2D list of cell values
        markdown: Markdown representation of table
    """
    table_id: str
    page_or_location: int
    rows: int
    columns: int
    headers: List[str]
    data: List[List[str]]
    markdown: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableData':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Chunk:
    """
    Content chunk with metadata for vector storage.

    Attributes:
        chunk_id: Unique identifier for the chunk
        document_id: Parent document identifier
        content: Chunk text content
        word_count: Number of words in chunk
        start_position: Starting character position in document
        end_position: Ending character position in document
        chunk_index: Index of this chunk in sequence
        total_chunks: Total number of chunks in document
        has_overlap: Whether this chunk overlaps with adjacent chunks
        section_heading: Heading of section containing this chunk
        metadata_tags: Additional tags for filtering/search
    """
    chunk_id: str
    document_id: str
    content: str
    word_count: int
    start_position: int
    end_position: int
    chunk_index: int
    total_chunks: int
    has_overlap: bool
    section_heading: Optional[str] = None
    metadata_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class UploadResult:
    """
    Result of database upload operation.

    Attributes:
        success: Whether upload succeeded
        document_id: Identifier of uploaded document
        chunk_ids: List of uploaded chunk identifiers
        upload_time_seconds: Time taken for upload
        error: Error message if upload failed
    """
    success: bool
    document_id: str
    chunk_ids: List[str] = field(default_factory=list)
    upload_time_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UploadResult':
        """Create instance from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ExtractionResult:
    """
    Result from content extraction operation.

    Attributes:
        success: Whether extraction succeeded
        text: Extracted text content
        tables: List of extracted tables
        metadata: Format-specific metadata
        error: Error message if extraction failed
    """
    success: bool
    text: str = ""
    tables: List[TableData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'text': self.text,
            'tables': [t.to_dict() for t in self.tables],
            'metadata': self.metadata,
            'error': self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create instance from dictionary."""
        tables_data = data.pop('tables', [])
        tables = [TableData.from_dict(t) for t in tables_data]
        return cls(tables=tables, **data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ProcessedDocument:
    """
    Complete processed document with all metadata and content.

    Attributes:
        document_id: Unique document identifier
        file_path: Path to source file
        file_name: Name of source file
        content: Markdown-formatted content
        raw_content: Original extracted text
        tables: Extracted tables
        metadata: Comprehensive metadata
        chunks: Content chunks for vector storage
        processed_at: Processing timestamp in ISO format
        processing_time_seconds: Time taken to process
        extractor_used: Name of extractor that processed this document
        success: Whether processing succeeded
        errors: List of errors encountered during processing
    """
    document_id: str
    file_path: str
    file_name: str
    content: str
    raw_content: str
    tables: List[TableData]
    metadata: EnrichedMetadata
    chunks: List[Chunk] = field(default_factory=list)
    processed_at: str = ""
    processing_time_seconds: float = 0.0
    extractor_used: str = ""
    success: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document_id': self.document_id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'content': self.content,
            'raw_content': self.raw_content,
            'tables': [t.to_dict() for t in self.tables],
            'metadata': self.metadata.to_dict(),
            'chunks': [c.to_dict() for c in self.chunks],
            'processed_at': self.processed_at,
            'processing_time_seconds': self.processing_time_seconds,
            'extractor_used': self.extractor_used,
            'success': self.success,
            'errors': self.errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create instance from dictionary."""
        tables_data = data.pop('tables', [])
        tables = [TableData.from_dict(t) for t in tables_data]

        metadata_data = data.pop('metadata', {})
        metadata = EnrichedMetadata.from_dict(metadata_data)

        chunks_data = data.pop('chunks', [])
        chunks = [Chunk.from_dict(c) for c in chunks_data]

        return cls(
            tables=tables,
            metadata=metadata,
            chunks=chunks,
            **data
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessedDocument':
        """Create instance from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_content_hash(self) -> str:
        """Generate SHA-256 hash of document content."""
        content_bytes = self.raw_content.encode('utf-8')
        return hashlib.sha256(content_bytes).hexdigest()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the document."""
        return {
            'document_id': self.document_id,
            'file_name': self.file_name,
            'word_count': self.metadata.basic.word_count,
            'character_count': self.metadata.basic.character_count,
            'line_count': self.metadata.basic.line_count,
            'table_count': len(self.tables),
            'chunk_count': len(self.chunks),
            'heading_count': self.metadata.structure.heading_count,
            'entity_count': self.metadata.entities.entity_count,
            'processing_time_seconds': self.processing_time_seconds,
            'extractor_used': self.extractor_used,
            'success': self.success,
            'error_count': len(self.errors)
        }


def create_document_fingerprint(content: str) -> str:
    """
    Create SHA-256 fingerprint of document content.

    Args:
        content: Document content to fingerprint

    Returns:
        Hexadecimal SHA-256 hash string
    """
    content_bytes = content.encode('utf-8')
    return hashlib.sha256(content_bytes).hexdigest()


def estimate_read_time(word_count: int, words_per_minute: int = 200) -> int:
    """
    Estimate reading time in minutes.

    Args:
        word_count: Number of words in document
        words_per_minute: Average reading speed (default: 200)

    Returns:
        Estimated reading time in minutes (minimum 1)
    """
    if word_count <= 0:
        return 0
    minutes = max(1, round(word_count / words_per_minute))
    return minutes
