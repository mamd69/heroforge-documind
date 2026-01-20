"""
Document API Schemas

Pydantic models for document management endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class FileType(str, Enum):
    """Supported document file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    XLSX = "xlsx"


class UploadStatus(str, Enum):
    """Status of document upload"""
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    PROCESSING = "processing"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata information"""
    word_count: Optional[int] = Field(None, description="Total word count")
    page_count: Optional[int] = Field(None, description="Number of pages (if applicable)")
    language: Optional[str] = Field(None, description="Detected language")
    topics: List[str] = Field(default_factory=list, description="Detected topics")
    entities: List[str] = Field(default_factory=list, description="Named entities found")
    fingerprint: Optional[str] = Field(None, description="Content fingerprint for deduplication")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class DocumentUploadResponse(BaseModel):
    """Response from document upload"""
    status: UploadStatus = Field(..., description="Upload status")
    document_id: Optional[str] = Field(None, description="UUID of uploaded document")
    existing_id: Optional[str] = Field(None, description="ID if duplicate detected")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Detected file type")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    message: Optional[str] = Field(None, description="Status message")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "company_policies.pdf",
                "file_type": "pdf",
                "chunks_created": 12,
                "message": "Document uploaded and processed successfully",
                "processing_time_ms": 2340.5
            }
        }


class Document(BaseModel):
    """Full document details"""
    id: str = Field(..., description="Document UUID")
    title: str = Field(..., description="Document title")
    file_type: FileType = Field(..., description="File type")
    file_name: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Upload timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    content_preview: Optional[str] = Field(None, description="Preview of document content")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Company Policies",
                "file_type": "pdf",
                "file_name": "company_policies.pdf",
                "chunk_count": 12,
                "created_at": "2024-01-15T10:30:00Z",
                "metadata": {
                    "word_count": 5420,
                    "page_count": 15,
                    "topics": ["HR", "Benefits", "Policies"]
                },
                "content_preview": "This document outlines the company's policies..."
            }
        }


class DocumentSummary(BaseModel):
    """Document summary for listing"""
    id: str = Field(..., description="Document UUID")
    title: str = Field(..., description="Document title")
    file_type: FileType = Field(..., description="File type")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Upload timestamp")


class DocumentList(BaseModel):
    """Paginated list of documents"""
    items: List[DocumentSummary] = Field(..., description="Documents in this page")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    has_more: bool = Field(..., description="Whether more pages exist")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "title": "Company Policies",
                        "file_type": "pdf",
                        "chunk_count": 12,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                ],
                "total": 25,
                "page": 1,
                "limit": 20,
                "has_more": True
            }
        }


class DocumentChunk(BaseModel):
    """A chunk of document content"""
    chunk_id: str = Field(..., description="Chunk UUID")
    document_id: str = Field(..., description="Parent document UUID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Position in document")
    word_count: int = Field(..., description="Word count")
    section_heading: Optional[str] = Field(None, description="Section this chunk belongs to")
    has_overlap: bool = Field(default=False, description="Whether chunk overlaps with previous")
    metadata_tags: List[str] = Field(default_factory=list, description="Content tags")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "content": "Section 3: Vacation Policy\n\nAll full-time employees are entitled to...",
                "chunk_index": 2,
                "word_count": 750,
                "section_heading": "Vacation Policy",
                "has_overlap": True,
                "metadata_tags": ["policy", "vacation", "benefits"]
            }
        }
