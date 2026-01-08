# DocuMind Agent Interfaces

## Coordinator Agent
**Role**: Orchestrates document processing pipeline

### Methods
- `process_document(file_path: str) -> dict`
  - Coordinates full pipeline
  - Returns: Processing report with status, chunks created, errors

- `process_batch(file_paths: list[str]) -> dict`
  - Processes multiple documents in parallel
  - Returns: Batch processing report

### Communication
- Delegates to: Extractor
- Receives from: Extractor, Chunker, Embedder, Writer
- Shared Memory Keys:
  - `pipeline/status/{document_id}` - Current stage
  - `pipeline/errors/{document_id}` - Error log

---

## Extractor Agent
**Role**: Extract text from various file formats

### Methods
- `extract_text(file_path: str) -> dict`
  - Returns: `{text: str, metadata: dict, format: str}`

- `supported_formats() -> list[str]`
  - Returns: [".pdf", ".docx", ".xlsx", ".txt", ".md"]

### Communication
- Receives from: Coordinator
- Delegates to: Chunker
- Shared Memory Keys:
  - `extraction/raw_text/{document_id}` - Extracted text
  - `extraction/metadata/{document_id}` - File metadata

---

## Chunker Agent
**Role**: Split text into semantic chunks

### Methods
- `chunk_text(text: str, strategy: str = "semantic") -> list[dict]`
  - Returns: Array of chunks with metadata
  - Strategies: "semantic", "fixed-size", "sentence"

- `optimize_chunk_size(text: str) -> int`
  - Returns: Recommended chunk size for this document

### Communication
- Receives from: Extractor
- Delegates to: Embedder
- Shared Memory Keys:
  - `chunking/chunks/{document_id}` - Generated chunks
  - `chunking/strategy/{document_id}` - Strategy used

---

## Embedder Agent
**Role**: Generate vector embeddings

### Methods
- `generate_embeddings(chunks: list[str]) -> list[list[float]]`
  - Batch generates embeddings
  - Returns: Array of 1536-dimensional vectors

- `batch_optimize(chunks: list[str]) -> int`
  - Returns: Optimal batch size for API efficiency

### Communication
- Receives from: Chunker
- Delegates to: Writer
- Shared Memory Keys:
  - `embeddings/vectors/{document_id}` - Generated embeddings
  - `embeddings/model/{document_id}` - Model used

---

## Writer Agent
**Role**: Store chunks and embeddings in Supabase

### Methods
- `write_chunks(document_id: str, chunks: list[dict]) -> bool`
  - Writes to document_chunks table
  - Returns: Success status

- `transaction_safe_write(data: dict) -> bool`
  - Writes with transaction rollback on error

### Communication
- Receives from: Embedder
- Reports to: Coordinator
- Shared Memory Keys:
  - `database/write_status/{document_id}` - Write confirmation
  - `database/record_ids/{document_id}` - Created record IDs