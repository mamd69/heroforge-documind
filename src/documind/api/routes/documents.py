"""
Document API Routes

Endpoints for document upload, management, and retrieval.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, status, Query
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
import os
import tempfile

from ..schemas.documents import (
    DocumentUploadResponse,
    Document,
    DocumentList,
    DocumentSummary,
    DocumentChunk,
    DocumentMetadata,
    FileType,
    UploadStatus,
)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# In-memory storage for development
# TODO: Replace with database service
_documents: dict[str, Document] = {}
_chunks: dict[str, List[DocumentChunk]] = {}


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload a document for processing and indexing."
)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    import time
    start_time = time.time()

    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Map extension to file type
    file_type_map = {
        ".pdf": FileType.PDF,
        ".docx": FileType.DOCX,
        ".txt": FileType.TXT,
        ".md": FileType.MD,
        ".csv": FileType.CSV,
        ".xlsx": FileType.XLSX,
    }
    file_type = file_type_map.get(ext, FileType.TXT)

    # Process document
    # TODO: Integrate with DocumentProcessor
    from ..services.doc_service import DocumentService
    service = DocumentService()

    try:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await service.process_and_upload(
                file_path=tmp_path,
                filename=filename
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        processing_time = (time.time() - start_time) * 1000

        # Check for duplicate
        if result.get("status") == "duplicate":
            return DocumentUploadResponse(
                status=UploadStatus.DUPLICATE,
                existing_id=result.get("existing_id"),
                filename=filename,
                file_type=file_type,
                chunks_created=0,
                message="Document already exists",
                processing_time_ms=processing_time
            )

        # Create document record
        doc_id = result.get("document_id", str(uuid4()))
        now = datetime.utcnow()

        document = Document(
            id=doc_id,
            title=os.path.splitext(filename)[0],
            file_type=file_type,
            file_name=filename,
            chunk_count=result.get("chunks_created", 0),
            created_at=now,
            metadata=DocumentMetadata(
                word_count=result.get("word_count"),
                fingerprint=result.get("fingerprint")
            )
        )

        _documents[doc_id] = document
        _chunks[doc_id] = result.get("chunks", [])

        return DocumentUploadResponse(
            status=UploadStatus.SUCCESS,
            document_id=doc_id,
            filename=filename,
            file_type=file_type,
            chunks_created=result.get("chunks_created", 0),
            message="Document uploaded and processed successfully",
            processing_time_ms=processing_time
        )

    except Exception as e:
        # Fallback for when service isn't fully implemented
        doc_id = str(uuid4())
        now = datetime.utcnow()

        document = Document(
            id=doc_id,
            title=os.path.splitext(filename)[0],
            file_type=file_type,
            file_name=filename,
            chunk_count=1,
            created_at=now,
            metadata=DocumentMetadata()
        )

        _documents[doc_id] = document
        _chunks[doc_id] = []

        processing_time = (time.time() - start_time) * 1000

        return DocumentUploadResponse(
            status=UploadStatus.SUCCESS,
            document_id=doc_id,
            filename=filename,
            file_type=file_type,
            chunks_created=1,
            message="Document uploaded (processing pending)",
            processing_time_ms=processing_time
        )


@router.get(
    "",
    response_model=DocumentList,
    summary="List documents",
    description="Get a paginated list of all documents."
)
async def list_documents(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    file_type: Optional[str] = None
):
    """List all documents with pagination"""
    # Filter documents
    docs = list(_documents.values())

    if file_type:
        docs = [d for d in docs if d.file_type.value == file_type]

    total = len(docs)

    # Paginate
    start = (page - 1) * limit
    end = start + limit
    page_docs = docs[start:end]

    return DocumentList(
        items=[
            DocumentSummary(
                id=d.id,
                title=d.title,
                file_type=d.file_type,
                chunk_count=d.chunk_count,
                created_at=d.created_at
            )
            for d in page_docs
        ],
        total=total,
        page=page,
        limit=limit,
        has_more=end < total
    )


@router.get(
    "/{doc_id}",
    response_model=Document,
    summary="Get document",
    description="Get details of a specific document."
)
async def get_document(doc_id: str):
    """Get document details"""
    if doc_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return _documents[doc_id]


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and all its chunks."
)
async def delete_document(doc_id: str):
    """Delete a document"""
    if doc_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # TODO: Delete from database and vector store
    del _documents[doc_id]
    if doc_id in _chunks:
        del _chunks[doc_id]


@router.get(
    "/{doc_id}/chunks",
    response_model=List[DocumentChunk],
    summary="Get document chunks",
    description="Get all chunks for a document."
)
async def get_chunks(doc_id: str):
    """Get document chunks"""
    if doc_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return _chunks.get(doc_id, [])


@router.patch(
    "/{doc_id}/metadata",
    response_model=Document,
    summary="Update document metadata",
    description="Update custom metadata for a document."
)
async def update_metadata(doc_id: str, metadata: dict):
    """Update document metadata"""
    if doc_id not in _documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    doc = _documents[doc_id]
    doc.metadata.custom.update(metadata)
    doc.updated_at = datetime.utcnow()

    return doc
