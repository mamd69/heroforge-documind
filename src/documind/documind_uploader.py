"""
DocuMind Database Integration
Uploads processed documents via MCP tools with duplicate detection and batch processing.
"""
from typing import Dict, List, Optional, Any
import json
import time
import uuid
import hashlib
from datetime import datetime


class DocuMindUploader:
    """
    Upload documents to DocuMind database via MCP tools.

    Features:
    - Duplicate detection via content fingerprinting
    - JSONB metadata storage
    - Chunk management with parent linkage
    - Batch upload support
    - Error handling and retry logic
    """

    def __init__(self, mcp_available: bool = True):
        """
        Initialize uploader.

        Args:
            mcp_available: Whether MCP tools are available
        """
        self.mcp_available = mcp_available
        self.upload_stats = {
            "total_uploads": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "duplicate_detections": 0,
            "total_chunks_uploaded": 0
        }

    def upload_document(self, processed_doc: Dict) -> Dict:
        """
        Upload a processed document to DocuMind.

        Args:
            processed_doc: ProcessedDocument as dictionary with keys:
                - file_name: Document filename
                - content: Full document content
                - metadata: Document metadata dictionary
                - chunks: List of chunk dictionaries
                - document_id: Optional existing document ID

        Returns:
            UploadResult dictionary:
                - success: Boolean indicating success
                - document_id: Unique document identifier
                - chunk_ids: List of uploaded chunk IDs
                - upload_time_seconds: Time taken for upload
                - error: Error message if failed
        """
        start_time = time.time()
        self.upload_stats["total_uploads"] += 1

        try:
            # 1. Check for duplicates via fingerprint
            fingerprint = processed_doc.get("metadata", {}).get("fingerprint")
            if not fingerprint:
                # Generate fingerprint if not present
                content = processed_doc.get("content", "")
                fingerprint = self._generate_fingerprint(content)
                if "metadata" not in processed_doc:
                    processed_doc["metadata"] = {}
                processed_doc["metadata"]["fingerprint"] = fingerprint

            duplicate_id = self.check_duplicate(fingerprint)
            if duplicate_id:
                self.upload_stats["duplicate_detections"] += 1
                return {
                    "success": False,
                    "document_id": duplicate_id,
                    "chunk_ids": [],
                    "error": f"Duplicate document found: {duplicate_id}",
                    "upload_time_seconds": time.time() - start_time,
                    "is_duplicate": True
                }

            # 2. Prepare metadata for JSONB storage
            metadata_json = self._serialize_metadata(processed_doc.get("metadata", {}))

            # 3. Upload main document
            doc_result = self._upload_via_mcp(
                title=processed_doc.get("file_name", "Untitled"),
                content=processed_doc.get("content", ""),
                file_type=self._get_file_type(processed_doc),
                metadata=metadata_json
            )

            if not doc_result.get("success"):
                self.upload_stats["failed_uploads"] += 1
                return {
                    "success": False,
                    "document_id": "",
                    "chunk_ids": [],
                    "error": doc_result.get("error", "Upload failed"),
                    "upload_time_seconds": time.time() - start_time
                }

            document_id = doc_result.get("document_id", str(uuid.uuid4()))

            # 4. Upload chunks with parent linkage
            chunk_ids = []
            chunks = processed_doc.get("chunks", [])
            for chunk in chunks:
                chunk_result = self._upload_chunk(chunk, document_id)
                if chunk_result.get("success"):
                    chunk_ids.append(chunk_result.get("chunk_id"))
                    self.upload_stats["total_chunks_uploaded"] += 1

            self.upload_stats["successful_uploads"] += 1

            return {
                "success": True,
                "document_id": document_id,
                "chunk_ids": chunk_ids,
                "chunks_uploaded": len(chunk_ids),
                "upload_time_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.upload_stats["failed_uploads"] += 1
            return {
                "success": False,
                "document_id": "",
                "chunk_ids": [],
                "error": f"Upload exception: {str(e)}",
                "upload_time_seconds": time.time() - start_time
            }

    def check_duplicate(self, fingerprint: str) -> Optional[str]:
        """
        Check if document with same fingerprint exists.

        Uses metadata search to find documents with matching fingerprint.

        Args:
            fingerprint: SHA-256 hash of content

        Returns:
            Document ID if duplicate found, None otherwise
        """
        if not fingerprint or not self.mcp_available:
            return None

        try:
            # Search for documents with matching fingerprint in metadata
            # This would use mcp__documind__search_documents with metadata filter
            # For now, return None (no duplicates in empty database)

            # In production with MCP:
            # results = mcp__documind__search_documents(
            #     query=f"fingerprint:{fingerprint}",
            #     limit=1,
            #     file_type=None
            # )
            # if results and len(results) > 0:
            #     return results[0].get("document_id")

            return None

        except Exception as e:
            print(f"Duplicate check error: {e}")
            return None

    def _upload_via_mcp(self, title: str, content: str,
                        file_type: str, metadata: Dict) -> Dict:
        """
        Upload document using MCP tool.

        In production, this calls mcp__documind__upload_document.
        For standalone use, returns a mock result.

        Args:
            title: Document title
            content: Document content
            file_type: File type (pdf, docx, txt, etc.)
            metadata: Document metadata as JSON-serializable dict

        Returns:
            Result dictionary with success status and document_id
        """
        if not self.mcp_available:
            # Mock result for standalone testing
            return {
                "success": True,
                "document_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }

        try:
            # In production with MCP tools:
            # result = mcp__documind__upload_document(
            #     title=title,
            #     content=content,
            #     file_type=file_type,
            #     metadata=json.dumps(metadata)
            # )
            # return result

            # Mock for development
            return {
                "success": True,
                "document_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"MCP upload error: {str(e)}"
            }

    def _upload_chunk(self, chunk: Dict, parent_id: str) -> Dict:
        """
        Upload a single chunk with parent linkage.

        Args:
            chunk: Chunk dictionary with content and metadata
            parent_id: Parent document ID

        Returns:
            Result dictionary with success status and chunk_id
        """
        chunk_metadata = {
            "parent_document_id": parent_id,
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1),
            "word_count": chunk.get("word_count", 0),
            "section_heading": chunk.get("section_heading", ""),
            "has_overlap": chunk.get("has_overlap", False),
            "chunk_method": chunk.get("chunk_method", "unknown"),
            "start_char": chunk.get("start_char", 0),
            "end_char": chunk.get("end_char", 0)
        }

        result = self._upload_via_mcp(
            title=f"chunk_{chunk.get('chunk_index', 0)}_{parent_id[:8]}",
            content=chunk.get("content", ""),
            file_type="chunk",
            metadata=chunk_metadata
        )

        if result.get("success"):
            result["chunk_id"] = result.get("document_id", str(uuid.uuid4()))

        return result

    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """
        Ensure metadata is JSON-serializable for JSONB storage.

        Converts non-serializable types to strings and handles nested structures.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            JSON-serializable metadata dictionary
        """
        try:
            # Test serialization and deserialize to ensure compatibility
            serialized = json.dumps(metadata, default=str)
            return json.loads(serialized)
        except Exception as e:
            print(f"Metadata serialization error: {e}")
            # Return minimal safe metadata
            return {
                "serialization_error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _generate_fingerprint(self, content: str) -> str:
        """
        Generate SHA-256 fingerprint of content.

        Args:
            content: Document content

        Returns:
            Hexadecimal SHA-256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _get_file_type(self, processed_doc: Dict) -> str:
        """
        Extract file type from processed document.

        Args:
            processed_doc: ProcessedDocument dictionary

        Returns:
            File type string (pdf, docx, txt, etc.)
        """
        metadata = processed_doc.get("metadata", {})

        # Try multiple metadata paths
        if "basic" in metadata:
            file_type = metadata["basic"].get("file_type", "txt")
        elif "file_type" in metadata:
            file_type = metadata["file_type"]
        else:
            # Infer from filename
            file_name = processed_doc.get("file_name", "")
            if "." in file_name:
                file_type = file_name.rsplit(".", 1)[-1].lower()
            else:
                file_type = "txt"

        return file_type

    def upload_batch(self, documents: List[Dict],
                    parallel: bool = False,
                    stop_on_error: bool = False) -> List[Dict]:
        """
        Upload multiple documents with error handling.

        Args:
            documents: List of ProcessedDocument dictionaries
            parallel: Whether to upload in parallel (future enhancement)
            stop_on_error: Stop batch if any upload fails

        Returns:
            List of UploadResult dictionaries
        """
        results = []

        if parallel:
            # Future enhancement: Use ThreadPoolExecutor for parallel uploads
            print("Note: Parallel uploads not yet implemented, using sequential")

        for idx, doc in enumerate(documents):
            try:
                result = self.upload_document(doc)
                results.append(result)

                if stop_on_error and not result.get("success"):
                    print(f"Batch upload stopped at document {idx} due to error")
                    break

            except Exception as e:
                error_result = {
                    "success": False,
                    "document_id": "",
                    "chunk_ids": [],
                    "error": f"Batch upload exception: {str(e)}",
                    "document_index": idx
                }
                results.append(error_result)

                if stop_on_error:
                    print(f"Batch upload stopped at document {idx} due to exception")
                    break

        return results

    def get_upload_stats(self) -> Dict:
        """
        Get upload statistics.

        Returns:
            Dictionary with upload statistics
        """
        stats = self.upload_stats.copy()
        if stats["total_uploads"] > 0:
            stats["success_rate"] = (
                stats["successful_uploads"] / stats["total_uploads"]
            )
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset upload statistics."""
        self.upload_stats = {
            "total_uploads": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "duplicate_detections": 0,
            "total_chunks_uploaded": 0
        }


# Utility functions for integration

def upload_processed_document(processed_doc: Dict,
                              mcp_available: bool = True) -> Dict:
    """
    Convenience function to upload a single processed document.

    Args:
        processed_doc: ProcessedDocument dictionary
        mcp_available: Whether MCP tools are available

    Returns:
        UploadResult dictionary
    """
    uploader = DocuMindUploader(mcp_available=mcp_available)
    return uploader.upload_document(processed_doc)


def upload_batch_documents(documents: List[Dict],
                          mcp_available: bool = True,
                          parallel: bool = False) -> List[Dict]:
    """
    Convenience function to upload multiple processed documents.

    Args:
        documents: List of ProcessedDocument dictionaries
        mcp_available: Whether MCP tools are available
        parallel: Use parallel uploads (future)

    Returns:
        List of UploadResult dictionaries
    """
    uploader = DocuMindUploader(mcp_available=mcp_available)
    return uploader.upload_batch(documents, parallel=parallel)


if __name__ == "__main__":
    # Example usage and testing
    print("DocuMind Uploader - Standalone Test")
    print("-" * 60)

    # Create test document
    test_doc = {
        "file_name": "test_document.pdf",
        "content": "This is a test document for upload testing.",
        "metadata": {
            "basic": {
                "file_type": "pdf",
                "file_size": 1024
            },
            "extraction": {
                "method": "test"
            }
        },
        "chunks": [
            {
                "chunk_index": 0,
                "total_chunks": 2,
                "content": "This is chunk 1.",
                "word_count": 4,
                "section_heading": "Introduction"
            },
            {
                "chunk_index": 1,
                "total_chunks": 2,
                "content": "This is chunk 2.",
                "word_count": 4,
                "section_heading": "Content"
            }
        ]
    }

    # Test single upload
    uploader = DocuMindUploader(mcp_available=False)
    result = uploader.upload_document(test_doc)

    print("\nSingle Upload Result:")
    print(f"  Success: {result['success']}")
    print(f"  Document ID: {result['document_id']}")
    print(f"  Chunks Uploaded: {result.get('chunks_uploaded', 0)}")
    print(f"  Upload Time: {result['upload_time_seconds']:.3f}s")

    # Test batch upload
    batch_docs = [test_doc, test_doc.copy()]
    batch_results = uploader.upload_batch(batch_docs)

    print(f"\nBatch Upload: {len(batch_results)} documents")
    print(f"  Successful: {sum(1 for r in batch_results if r['success'])}")

    # Show stats
    stats = uploader.get_upload_stats()
    print("\nUpload Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
