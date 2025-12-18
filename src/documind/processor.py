"""
DocumentProcessor - Main Orchestrator
Production-ready document processing pipeline that:
- Auto-detects file formats (PDF, DOCX, CSV, XLSX, TXT, MD)
- Extracts text, tables, and metadata
- Formats output for LLM consumption
- Chunks content intelligently with overlap
- Integrates with DocuMind database via MCP
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .format_detector import FormatDetector
from .extractors.text_extractor import TextExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DocxExtractor
from .extractors.spreadsheet_extractor import SpreadsheetExtractor
from .extractors.metadata_extractor import MetadataExtractor
from .llm_formatter import LLMFormatter
from .content_chunker import ContentChunker
from .documind_uploader import DocuMindUploader
from .data_structures import (
    ProcessedDocument,
    ExtractionResult,
    Chunk,
    EnrichedMetadata
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main orchestrator for document processing pipeline.

    Workflow:
    1. Validate and detect file format
    2. Route to appropriate extractor
    3. Enrich with metadata
    4. Format for LLM consumption
    5. Chunk content with overlap
    6. Optionally upload to DocuMind

    Example:
        processor = DocumentProcessor()
        result = processor.process_document("document.pdf")
        print(result.chunks)
    """

    def __init__(
        self,
        chunk_target_size: int = 750,
        overlap_percent: float = 0.10,
        auto_upload: bool = False,
        max_workers: int = 4
    ):
        """
        Initialize the document processor.

        Args:
            chunk_target_size: Target words per chunk (default 750)
            overlap_percent: Overlap between chunks (default 10%)
            auto_upload: Automatically upload to DocuMind (default False)
            max_workers: Max threads for batch processing (default 4)
        """
        # Initialize all components
        self.format_detector = FormatDetector()
        self.text_extractor = TextExtractor()
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        self.spreadsheet_extractor = SpreadsheetExtractor()
        self.metadata_extractor = MetadataExtractor()
        self.llm_formatter = LLMFormatter()
        self.content_chunker = ContentChunker(
            target_size=chunk_target_size,
            overlap_percent=overlap_percent
        )
        self.uploader = DocuMindUploader()

        self.auto_upload = auto_upload
        self.max_workers = max_workers

        # Map formats to extractors
        self._extractors = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".doc": self._extract_docx,
            ".csv": self._extract_spreadsheet,
            ".xlsx": self._extract_spreadsheet,
            ".xls": self._extract_spreadsheet,
            ".txt": self._extract_text,
            ".md": self._extract_text,
            ".markdown": self._extract_text,
        }

    def process_document(
        self,
        file_path: Union[str, Path],
        upload: Optional[bool] = None,
        custom_metadata: Optional[Dict] = None
    ) -> ProcessedDocument:
        """
        Process a single document through the full pipeline.

        Args:
            file_path: Path to the document file
            upload: Override auto_upload setting
            custom_metadata: Additional metadata to include

        Returns:
            ProcessedDocument with all extracted data

        Raises:
            ValueError: If file format is unsupported or validation fails
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        logger.info(f"Processing document: {file_path}")

        # Step 1: Validate and detect format
        try:
            format_type, mime_type = self.format_detector.validate_and_detect(str(file_path))
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")

        # Convert format_type to extension format (e.g., "pdf" -> ".pdf")
        detected_format = f".{format_type}" if not format_type.startswith('.') else format_type
        logger.debug(f"Detected format: {detected_format}")

        # Step 2: Extract content using appropriate extractor
        extraction_result = self._extract_content(str(file_path), detected_format)

        if not extraction_result.success:
            raise ValueError(f"Extraction failed: {extraction_result.error}")

        # Step 3: Enrich with metadata
        format_metadata = extraction_result.metadata or {}
        if custom_metadata:
            format_metadata.update(custom_metadata)

        metadata = self.metadata_extractor.extract_all(
            str(file_path),
            extraction_result.text,
            format_metadata
        )

        # Convert to EnrichedMetadata dataclass
        enriched_metadata = EnrichedMetadata.from_dict(metadata)

        # Step 4: Format for LLM consumption
        # Convert tables to expected format with headers and rows
        formatted_tables = self._format_tables_for_llm(extraction_result.tables)
        llm_formatted = self.llm_formatter.format_for_llm(
            content=extraction_result.text,
            tables=formatted_tables,
            metadata=metadata
        )

        # Step 5: Chunk the content
        # Generate document ID from fingerprint
        document_id = metadata["fingerprint"][:16]
        chunks = self.content_chunker.chunk_content(
            content=llm_formatted,
            document_id=document_id
        )

        # Convert to Chunk dataclasses
        chunk_objects = [
            Chunk(
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                document_id=document_id,
                content=c["content"],
                word_count=c["word_count"],
                start_position=c.get("start_position", 0),
                end_position=c.get("end_position", len(c["content"])),
                chunk_index=c["chunk_index"],
                total_chunks=c.get("total_chunks", len(chunks)),
                has_overlap=c.get("has_overlap", False),
                section_heading=c.get("section_heading"),
                metadata_tags=c.get("metadata_tags", [])
            )
            for i, c in enumerate(chunks)
        ]

        # Step 6: Create processed document
        import uuid
        from datetime import datetime
        processed = ProcessedDocument(
            document_id=str(uuid.uuid4()),
            file_path=str(file_path.absolute()),
            file_name=file_path.name,
            content=llm_formatted,
            raw_content=extraction_result.text,
            tables=[],  # Tables already included in content
            metadata=enriched_metadata,
            chunks=chunk_objects,
            processed_at=datetime.utcnow().isoformat() + "Z",
            extractor_used=detected_format.strip('.')
        )

        # Step 7: Optionally upload to DocuMind
        should_upload = upload if upload is not None else self.auto_upload
        if should_upload:
            upload_result = self.uploader.upload_document(
                title=file_path.stem,
                content=llm_formatted,
                metadata=metadata,
                file_type=detected_format
            )
            processed.upload_result = upload_result
            logger.info(f"Uploaded to DocuMind: {upload_result.document_id}")

        logger.info(f"Successfully processed: {file_path.name} ({len(chunk_objects)} chunks)")
        return processed

    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        upload: Optional[bool] = None,
        stop_on_error: bool = False
    ) -> Dict[str, Union[ProcessedDocument, str]]:
        """
        Process multiple documents in parallel.

        Args:
            file_paths: List of file paths to process
            upload: Override auto_upload setting
            stop_on_error: Stop processing if any document fails

        Returns:
            Dictionary mapping file paths to ProcessedDocument or error string
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_document, path, upload): path
                for path in file_paths
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                path_str = str(path)

                try:
                    results[path_str] = future.result()
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to process {path}: {error_msg}")
                    results[path_str] = f"Error: {error_msg}"

                    if stop_on_error:
                        # Cancel remaining futures
                        for f in future_to_path:
                            f.cancel()
                        break

        return results

    def _extract_content(self, file_path: str, format: str) -> ExtractionResult:
        """Route to appropriate extractor based on format."""
        extractor_func = self._extractors.get(format)

        if not extractor_func:
            return ExtractionResult(
                success=False,
                text="",
                error=f"No extractor available for format: {format}"
            )

        try:
            return extractor_func(file_path)
        except Exception as e:
            logger.exception(f"Extraction error for {file_path}")
            return ExtractionResult(
                success=False,
                text="",
                error=str(e)
            )

    def _extract_pdf(self, file_path: str) -> ExtractionResult:
        """Extract content from PDF file."""
        text_result = self.pdf_extractor.extract_text(file_path)

        if not text_result.get("success", False):
            return ExtractionResult(
                success=False,
                text="",
                error=text_result.get("error", "PDF text extraction failed")
            )

        tables_result = self.pdf_extractor.extract_tables(file_path)

        # Convert tables to standard format
        table_data = []
        if tables_result.get("success", False):
            for table in tables_result.get("tables", []):
                table_data.append({
                    "table_number": table.get("table_number", 0),
                    "page": table.get("page", 0),
                    "data": table.get("data", []),
                    "headers": table.get("headers", [])
                })

        return ExtractionResult(
            success=True,
            text=text_result.get("text", ""),
            tables=table_data,
            metadata={
                "page_count": text_result.get("page_count", 0),
                "pdf_metadata": text_result.get("metadata", {})
            }
        )

    def _extract_docx(self, file_path: str) -> ExtractionResult:
        """Extract content from Word document."""
        result = self.docx_extractor.extract(file_path)

        if not result.get("success", False):
            return ExtractionResult(
                success=False,
                text="",
                error=result.get("error", "Unknown DOCX extraction error")
            )

        # Convert tables to standard format
        table_data = []
        for table in result.get("tables", []):
            table_data.append({
                "table_number": table["table_number"],
                "rows": table["rows"],
                "columns": table["columns"],
                "data": table["data"]
            })

        return ExtractionResult(
            success=True,
            text=result["text"],
            tables=table_data,
            metadata=result.get("metadata", {})
        )

    def _extract_spreadsheet(self, file_path: str) -> ExtractionResult:
        """Extract content from CSV or Excel file."""
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            result = self.spreadsheet_extractor.extract_csv(file_path)
        else:
            result = self.spreadsheet_extractor.extract_excel(file_path)

        if not result.get("success", False):
            return ExtractionResult(
                success=False,
                text="",
                error=result.get("error", "Unknown spreadsheet extraction error")
            )

        # Format as LLM-friendly text
        formatted = self.spreadsheet_extractor.format_for_llm(file_path)

        # Build table data
        table_data = []
        if ext == ".csv":
            table_data.append({
                "table_number": 1,
                "rows": result["rows"],
                "columns": result["columns"],
                "column_names": result["column_names"],
                "data": result["data"]
            })
        else:
            for sheet_name, sheet in result.get("sheets", {}).items():
                table_data.append({
                    "sheet_name": sheet_name,
                    "rows": sheet["rows"],
                    "columns": sheet["columns"],
                    "column_names": sheet["column_names"],
                    "data": sheet["data"]
                })

        return ExtractionResult(
            success=True,
            text=formatted,
            tables=table_data,
            metadata={
                "sheet_count": result.get("sheet_count", 1),
                "total_rows": result.get("rows", sum(t.get("rows", 0) for t in table_data))
            }
        )

    def _extract_text(self, file_path: str) -> ExtractionResult:
        """Extract content from plain text or Markdown file."""
        result = self.text_extractor.extract(file_path)

        if not result.get("success", False):
            return ExtractionResult(
                success=False,
                text="",
                error=result.get("error", "Unknown text extraction error")
            )

        return ExtractionResult(
            success=True,
            text=result.get("text", result.get("content", "")),
            metadata={
                "encoding": result.get("encoding", result.get("metadata", {}).get("encoding", "utf-8")),
                "is_markdown": result.get("is_markdown", False)
            }
        )

    def _format_tables_for_llm(self, tables: List[Dict]) -> List[Dict]:
        """
        Convert tables from various formats to LLMFormatter expected format.

        Expected format: {'headers': [...], 'rows': [...]}
        """
        if not tables:
            return []

        formatted = []
        for table in tables:
            # Handle different table formats
            if 'headers' in table and 'rows' in table:
                # Already in correct format
                formatted.append(table)
            elif 'data' in table:
                # Format with data array (first row is header)
                data = table['data']
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        # data is list of lists (rows)
                        formatted.append({
                            'headers': [str(cell) for cell in data[0]],
                            'rows': [[str(cell) for cell in row] for row in data[1:]]
                        })
                    else:
                        # data is list of dicts (records)
                        if data and isinstance(data[0], dict):
                            headers = list(data[0].keys())
                            rows = [[str(row.get(h, '')) for h in headers] for row in data]
                            formatted.append({
                                'headers': headers,
                                'rows': rows
                            })
            elif 'column_names' in table and 'data' in table:
                # Spreadsheet format with column_names
                headers = table['column_names']
                data = table['data']
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    rows = [[str(row.get(h, '')) for h in headers] for row in data]
                else:
                    rows = []
                formatted.append({
                    'headers': headers,
                    'rows': rows
                })

        return formatted

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self._extractors.keys())

    def check_duplicate(self, file_path: Union[str, Path]) -> Dict:
        """
        Check if a document already exists in DocuMind.

        Args:
            file_path: Path to document file

        Returns:
            Dict with 'is_duplicate' and 'existing_id' if found
        """
        file_path = Path(file_path)

        # Extract content for fingerprinting
        try:
            format_type, mime_type = self.format_detector.validate_and_detect(str(file_path))
        except Exception as e:
            return {"is_duplicate": False, "error": str(e)}

        detected_format = f".{format_type}" if not format_type.startswith('.') else format_type
        extraction = self._extract_content(str(file_path), detected_format)
        if not extraction.success:
            return {"is_duplicate": False, "error": extraction.error}

        fingerprint = self.metadata_extractor.generate_fingerprint(extraction.text)

        existing_id = self.uploader.check_duplicate(fingerprint)
        if existing_id:
            return {"is_duplicate": True, "existing_id": existing_id}
        return {"is_duplicate": False, "fingerprint": fingerprint}


# Convenience function for quick processing
def process_document(file_path: str, **kwargs) -> ProcessedDocument:
    """
    Convenience function to process a single document.

    Args:
        file_path: Path to document file
        **kwargs: Additional arguments passed to DocumentProcessor

    Returns:
        ProcessedDocument with all extracted data
    """
    processor = DocumentProcessor(**kwargs)
    return processor.process_document(file_path)


# Test
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with sample documents
    sample_docs = [
        "docs/workshops/S7-sample-docs/simple_security_policy.pdf",
        "docs/workshops/S7-sample-docs/meeting_notes.docx",
        "docs/workshops/S7-sample-docs/employee_data.csv",
    ]

    processor = DocumentProcessor()

    print("=" * 60)
    print("DOCUMENT PROCESSOR TEST")
    print("=" * 60)
    print(f"\nSupported formats: {processor.get_supported_formats()}")

    for doc_path in sample_docs:
        path = Path(doc_path)
        if not path.exists():
            print(f"\n⚠️  Skipping (not found): {doc_path}")
            continue

        print(f"\n📄 Processing: {path.name}")
        print("-" * 40)

        try:
            result = processor.process_document(doc_path)
            print(f"  Format: {result.format}")
            print(f"  Fingerprint: {result.fingerprint[:16]}...")
            print(f"  Raw text length: {len(result.raw_text)} chars")
            print(f"  Chunks: {len(result.chunks)}")
            print(f"  Tables: {len(result.tables)}")

            if result.chunks:
                print(f"\n  First chunk preview ({result.chunks[0].word_count} words):")
                preview = result.chunks[0].content[:200]
                print(f"  {preview}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
