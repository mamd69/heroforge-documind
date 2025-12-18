"""
DocuMind - AI-Powered Knowledge Management System

This package contains the core DocuMind application built during
the AI-Powered Software Development course.

Includes production-ready document processing pipeline for:
- Auto-detecting file formats (PDF, DOCX, CSV, XLSX, TXT, MD)
- Extracting text, tables, and metadata
- Formatting output for LLM consumption
- Chunking content intelligently with overlap
- Integrating with DocuMind database via MCP
"""

__version__ = "1.0.0"
__author__ = "HeroForge Course Students"

# Main processor
from .processor import DocumentProcessor, process_document

# Data structures
from .data_structures import (
    ProcessedDocument,
    ExtractionResult,
    Chunk,
    EnrichedMetadata,
    BasicMetadata,
    StructureMetadata,
    EntityMetadata,
    TopicMetadata,
    TableData,
    UploadResult,
)

# Core components
from .format_detector import FormatDetector
from .llm_formatter import LLMFormatter
from .content_chunker import ContentChunker
from .documind_uploader import DocuMindUploader

__all__ = [
    # Main
    "DocumentProcessor",
    "process_document",
    # Data structures
    "ProcessedDocument",
    "ExtractionResult",
    "Chunk",
    "EnrichedMetadata",
    "BasicMetadata",
    "StructureMetadata",
    "EntityMetadata",
    "TopicMetadata",
    "TableData",
    "UploadResult",
    # Components
    "FormatDetector",
    "LLMFormatter",
    "ContentChunker",
    "DocuMindUploader",
]
