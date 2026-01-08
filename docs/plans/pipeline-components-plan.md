# DocuMind Pipeline Components - GOAP Implementation Plan

**Document Version**: 1.0
**Created**: 2025-12-12
**Planning Method**: Goal-Oriented Action Planning (GOAP)

---

## Executive Summary

This plan defines the implementation of DocuMind's 5-agent document processing pipeline using Goal-Oriented Action Planning methodology. The pipeline transforms uploaded documents into searchable, vector-embedded chunks stored in Supabase for RAG-based retrieval.

**Goal State**: Fully functional multi-agent document processing pipeline capable of handling 100+ documents/hour with parallel processing, error recovery, and comprehensive monitoring.

**Current State**: Database schema exists, MCP servers configured, basic project structure in place.

**Gap Analysis**: Need to implement 5 specialized agents with coordination logic, error handling, and performance optimization.

---

## 1. Pipeline Architecture Overview

```
User Upload (PDF/DOCX/XLSX/TXT/MD)
         ↓
    COORDINATOR ←───────── Error Recovery & Status Tracking
         ↓
    EXTRACTOR ←────────── File Format Detection & Text Extraction
         ↓                 (Parallel: multiple files)
    CHUNKER ←──────────── Semantic Segmentation (~500 words)
         ↓                 (Parallel: multiple documents)
    EMBEDDER ←─────────── OpenAI Embedding Generation (1536-dim)
         ↓                 (Parallel: batch processing)
    WRITER ←───────────── Supabase Storage with Transactions
         ↓                 (Parallel: database writes)
    COORDINATOR ────────► Processing Report & Metrics
```

**Topology**: Hierarchical Pipeline with stage-level parallelism
**Technology**: Python 3.10+, asyncio, OpenAI API, Supabase, pgvector
**Success Criteria**: < 3 seconds per document, 90%+ success rate, graceful error handling

---

## 2. Component Specifications (GOAP Action Definitions)

### 2.1 COORDINATOR AGENT

**Role**: Orchestrates document processing pipeline and manages workflow state

**Preconditions**:
- `supabase_connection_available: true`
- `environment_variables_configured: true`
- `agent_registry_initialized: true`

**Actions**:
1. `initialize_pipeline(topology: hierarchical, max_agents: 5)`
2. `process_document(file_path: str) -> processing_report`
3. `process_batch(file_paths: list[str]) -> batch_report`
4. `handle_error(agent_id: str, error: Exception) -> recovery_action`
5. `generate_status_report() -> metrics_summary`

**Effects** (State Changes):
- `pipeline_initialized: true`
- `document_processing_started: true`
- `processing_status_tracked: true`
- `errors_logged_and_recovered: true`

**Success Criteria**:
- Pipeline initializes in < 1 second
- Coordinates 10+ concurrent documents
- 100% error tracking coverage
- Generates comprehensive status reports

**Implementation File**: `app/src/agents/coordinator.py`

**Dependencies**:
- All other agents (Extractor, Chunker, Embedder, Writer)
- Supabase connection
- Memory/coordination system

**Cost**: 1 (orchestration overhead)

**Tool Groups**: [claude-flow MCP, memory management, error tracking]

---

### 2.2 EXTRACTOR AGENT

**Role**: Extract raw text from various file formats

**Preconditions**:
- `file_path_exists: true`
- `file_format_supported: true` (PDF/DOCX/XLSX/TXT/MD)
- `read_permissions_granted: true`

**Actions**:
1. `detect_file_format(file_path: str) -> format_type`
2. `extract_text_from_pdf(file_path: str) -> extracted_text`
3. `extract_text_from_docx(file_path: str) -> extracted_text`
4. `extract_text_from_xlsx(file_path: str) -> extracted_text`
5. `extract_text_from_plaintext(file_path: str) -> extracted_text`
6. `extract_metadata(file_path: str) -> metadata_dict`

**Effects** (State Changes):
- `file_format_detected: true`
- `text_extracted: true`
- `metadata_captured: true`
- `extraction_logged: true`

**Success Criteria**:
- Supports 5 file formats (PDF, DOCX, XLSX, TXT, MD)
- < 2 seconds extraction time per document
- 95%+ text accuracy for simple formats
- Preserves metadata (title, author, date)

**Implementation File**: `app/src/agents/extractor.py`

**Dependencies**:
- Python libraries: PyPDF2, pdfplumber, python-docx, openpyxl, pandas
- File system access
- Memory coordination for storing raw text

**Cost**: 3 (I/O and parsing overhead)

**Tool Groups**: [file_parsers, metadata_extractors]

**Execution Mode**: Code (deterministic file operations)

**Technology Requirements**:
```python
# Libraries
PyPDF2>=3.0.0          # Simple PDF extraction
pdfplumber>=0.10.0     # Complex PDF layouts
python-docx>=1.0.0     # Word documents
openpyxl>=3.1.0        # Excel files
pandas>=2.0.0          # Spreadsheet processing
```

---

### 2.3 CHUNKER AGENT

**Role**: Split extracted text into semantic chunks (~500 words)

**Preconditions**:
- `text_extracted: true`
- `text_length_sufficient: true` (> 50 words)
- `chunking_strategy_defined: true`

**Actions**:
1. `analyze_document_structure(text: str) -> structure_metadata`
2. `chunk_by_semantic_boundaries(text: str) -> list[chunk]`
3. `chunk_by_fixed_size(text: str, size: int = 500) -> list[chunk]`
4. `chunk_by_sentence_groups(text: str) -> list[chunk]`
5. `optimize_chunk_size(text: str, target: int = 500) -> optimal_size`
6. `add_chunk_metadata(chunks: list) -> enriched_chunks`

**Effects** (State Changes):
- `document_structure_analyzed: true`
- `text_chunked: true`
- `chunks_optimized_for_embedding: true`
- `chunk_metadata_added: true`

**Success Criteria**:
- Average chunk size: 400-600 words
- Respects semantic boundaries (paragraphs, sections)
- < 1 second chunking time per document
- Maintains source document reference

**Implementation File**: `app/src/agents/chunker.py`

**Dependencies**:
- Extractor agent (requires extracted text)
- NLTK or spaCy for sentence detection
- Memory coordination for storing chunks

**Cost**: 2 (text processing overhead)

**Tool Groups**: [text_processors, boundary_detectors]

**Execution Mode**: Hybrid (LLM for semantic understanding, code for splitting)

**Technology Requirements**:
```python
# Libraries
nltk>=3.8              # Sentence tokenization
spacy>=3.7             # NLP processing (optional)
langchain>=0.1.0       # Text splitters (optional)
```

**Chunking Strategy**:
```python
# Default: Semantic + Fixed-Size Hybrid
- Target: 500 words per chunk
- Overlap: 50 words between chunks
- Boundaries: Respect paragraph/section breaks
- Metadata: chunk_index, document_id, source_page
```

---

### 2.4 EMBEDDER AGENT

**Role**: Generate vector embeddings using OpenAI API

**Preconditions**:
- `chunks_available: true`
- `openai_api_key_configured: true`
- `embedding_model_accessible: true` (text-embedding-3-small)
- `chunk_text_valid: true` (non-empty, < 8191 tokens)

**Actions**:
1. `validate_chunk_text(chunk: str) -> bool`
2. `generate_single_embedding(text: str) -> vector[1536]`
3. `generate_batch_embeddings(texts: list[str]) -> list[vector[1536]]`
4. `optimize_batch_size(num_chunks: int) -> optimal_batch`
5. `handle_rate_limiting() -> retry_strategy`
6. `cache_embeddings(chunk_id: str, embedding: vector)`

**Effects** (State Changes):
- `chunks_validated: true`
- `embeddings_generated: true`
- `batch_processing_optimized: true`
- `embeddings_cached: true`

**Success Criteria**:
- 100% embedding generation success rate
- < 1 second per embedding (batch optimized)
- Handles rate limiting gracefully
- 1536-dimensional vectors (OpenAI standard)

**Implementation File**: `app/src/agents/embedder.py`

**Dependencies**:
- Chunker agent (requires chunks)
- OpenAI API access
- Rate limiting and retry logic
- Memory coordination for caching

**Cost**: 5 (API calls and network latency)

**Tool Groups**: [openai_api, batch_processors, rate_limiters]

**Execution Mode**: Code (API interactions)

**Technology Requirements**:
```python
# Libraries
openai>=1.0.0          # OpenAI Python SDK
tenacity>=8.0.0        # Retry logic
aiohttp>=3.9.0         # Async HTTP for batch processing
```

**API Configuration**:
```python
# OpenAI Settings
model = "text-embedding-3-small"
dimensions = 1536
batch_size = 100       # Optimal for rate limits
max_retries = 3
timeout = 30           # seconds
```

---

### 2.5 WRITER AGENT

**Role**: Store chunks and embeddings in Supabase with pgvector

**Preconditions**:
- `embeddings_generated: true`
- `supabase_connection_active: true`
- `database_schema_created: true`
- `pgvector_extension_enabled: true`

**Actions**:
1. `validate_embedding_dimensions(embedding: vector) -> bool`
2. `write_single_chunk(chunk: dict, embedding: vector) -> record_id`
3. `write_batch_chunks(chunks: list[dict]) -> list[record_id]`
4. `transaction_safe_write(data: list) -> success_status`
5. `handle_write_failures(failed_chunks: list) -> retry_result`
6. `update_document_status(document_id: str, status: str)`

**Effects** (State Changes):
- `chunks_stored_in_database: true`
- `embeddings_indexed_with_pgvector: true`
- `document_status_updated: true`
- `write_errors_handled: true`

**Success Criteria**:
- 100% successful writes (with transaction safety)
- < 500ms write time per chunk (batch optimized)
- Automatic rollback on failures
- HNSW index for vector search

**Implementation File**: `app/src/agents/writer.py`

**Dependencies**:
- Embedder agent (requires embeddings)
- Supabase database connection
- Transaction management
- Memory coordination for status updates

**Cost**: 4 (database I/O overhead)

**Tool Groups**: [supabase_mcp, transaction_managers, index_optimizers]

**Execution Mode**: Code (database operations)

**Technology Requirements**:
```python
# Libraries
supabase>=2.0.0        # Supabase Python client
psycopg2-binary>=2.9.0 # PostgreSQL adapter
asyncpg>=0.29.0        # Async PostgreSQL
```

**Database Operations**:
```sql
-- Insert with transaction safety
BEGIN;
  INSERT INTO documents (title, content, file_type, metadata)
  VALUES ($1, $2, $3, $4)
  RETURNING id;

  INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding, metadata)
  VALUES ($1, $2, $3, $4, $5);
COMMIT;

-- HNSW index creation
CREATE INDEX document_chunks_embedding_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 3. Implementation Order (Dependency-Based)

### Phase 1: Foundation (No Dependencies)
**Duration**: 2 hours
**Components**: Database setup, environment configuration

**Actions**:
1. ✅ Verify Supabase connection
2. ✅ Enable pgvector extension
3. ✅ Create database schema (documents, document_chunks)
4. ✅ Configure environment variables (API keys)
5. ✅ Initialize project structure (`app/src/agents/`)

**Preconditions**: None (starting state)
**Effects**: `database_ready: true`, `environment_configured: true`
**Success Criteria**: Schema created, pgvector enabled, connections tested

---

### Phase 2: Core Agents (Sequential Dependencies)
**Duration**: 6 hours
**Components**: Extractor → Chunker → Embedder → Writer

#### Step 2.1: Implement EXTRACTOR Agent
**Duration**: 1.5 hours
**Preconditions**: `environment_configured: true`

**Actions**:
```python
# Create app/src/agents/extractor.py
class ExtractorAgent:
    def extract_text(self, file_path: str) -> dict:
        """Extract text from PDF/DOCX/XLSX/TXT/MD"""
        format = self.detect_format(file_path)

        if format == "pdf":
            return self._extract_pdf(file_path)
        elif format == "docx":
            return self._extract_docx(file_path)
        elif format == "xlsx":
            return self._extract_xlsx(file_path)
        else:
            return self._extract_plaintext(file_path)

    def _extract_pdf(self, path: str) -> dict:
        # PyPDF2 for simple PDFs, pdfplumber for complex
        pass

    # Similar methods for other formats
```

**Effects**: `text_extraction_working: true`
**Success Criteria**: All 5 formats supported, < 2s per file

---

#### Step 2.2: Implement CHUNKER Agent
**Duration**: 1.5 hours
**Preconditions**: `text_extraction_working: true`

**Actions**:
```python
# Create app/src/agents/chunker.py
class ChunkerAgent:
    def chunk_text(self, text: str, strategy: str = "semantic") -> list[dict]:
        """Split text into ~500 word chunks"""
        if strategy == "semantic":
            return self._semantic_chunk(text)
        elif strategy == "fixed-size":
            return self._fixed_size_chunk(text, size=500)
        else:
            return self._sentence_chunk(text)

    def _semantic_chunk(self, text: str) -> list[dict]:
        # Use NLTK for sentence detection
        # Group sentences into ~500 word chunks
        # Respect paragraph boundaries
        pass
```

**Effects**: `text_chunking_working: true`
**Success Criteria**: 400-600 word chunks, semantic boundaries respected

---

#### Step 2.3: Implement EMBEDDER Agent
**Duration**: 2 hours
**Preconditions**: `text_chunking_working: true`, `openai_api_configured: true`

**Actions**:
```python
# Create app/src/agents/embedder.py
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbedderAgent:
    def __init__(self):
        self.client = openai.OpenAI()
        self.model = "text-embedding-3-small"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Batch generate embeddings with retry logic"""
        response = await self.client.embeddings.create(
            input=chunks,
            model=self.model,
            dimensions=1536
        )
        return [item.embedding for item in response.data]
```

**Effects**: `embedding_generation_working: true`
**Success Criteria**: < 1s per embedding, batch processing, rate limit handling

---

#### Step 2.4: Implement WRITER Agent
**Duration**: 1 hour
**Preconditions**: `embedding_generation_working: true`, `database_ready: true`

**Actions**:
```python
# Create app/src/agents/writer.py
from supabase import create_client

class WriterAgent:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client = create_client(supabase_url, supabase_key)

    async def write_chunks(self, document_id: str, chunks: list[dict]) -> bool:
        """Write chunks with transaction safety"""
        try:
            # Insert document record
            doc_result = self.client.table("documents").insert({
                "id": document_id,
                "title": chunks[0]["metadata"]["title"],
                "content": chunks[0]["metadata"]["full_text"],
                "file_type": chunks[0]["metadata"]["format"]
            }).execute()

            # Insert chunks with embeddings
            chunk_records = [
                {
                    "document_id": document_id,
                    "chunk_text": chunk["text"],
                    "chunk_index": chunk["index"],
                    "embedding": chunk["embedding"],
                    "metadata": chunk["metadata"]
                }
                for chunk in chunks
            ]

            self.client.table("document_chunks").insert(chunk_records).execute()
            return True

        except Exception as e:
            # Transaction rollback handled by Supabase
            raise
```

**Effects**: `database_writes_working: true`
**Success Criteria**: < 500ms per chunk, transaction safety, rollback on errors

---

### Phase 3: Orchestration (Requires All Agents)
**Duration**: 3 hours
**Components**: Coordinator Agent + Pipeline Integration

#### Step 3.1: Implement COORDINATOR Agent
**Duration**: 2 hours
**Preconditions**: All 4 agents implemented

**Actions**:
```python
# Create app/src/agents/coordinator.py
import asyncio
from typing import Dict, List

class CoordinatorAgent:
    def __init__(self):
        self.extractor = ExtractorAgent()
        self.chunker = ChunkerAgent()
        self.embedder = EmbedderAgent()
        self.writer = WriterAgent()
        self.status = {}

    async def process_document(self, file_path: str) -> dict:
        """Orchestrate full pipeline for single document"""
        doc_id = self._generate_doc_id(file_path)

        try:
            # Stage 1: Extract
            self._update_status(doc_id, "extracting")
            extracted = await self.extractor.extract_text(file_path)

            # Stage 2: Chunk
            self._update_status(doc_id, "chunking")
            chunks = await self.chunker.chunk_text(extracted["text"])

            # Stage 3: Embed
            self._update_status(doc_id, "embedding")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = await self.embedder.generate_embeddings(chunk_texts)

            # Attach embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]

            # Stage 4: Write
            self._update_status(doc_id, "writing")
            await self.writer.write_chunks(doc_id, chunks)

            self._update_status(doc_id, "completed")
            return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}

        except Exception as e:
            self._update_status(doc_id, f"failed: {str(e)}")
            return {"status": "error", "doc_id": doc_id, "error": str(e)}

    async def process_batch(self, file_paths: List[str]) -> dict:
        """Process multiple documents in parallel"""
        tasks = [self.process_document(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful

        return {
            "total": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results
        }
```

**Effects**: `pipeline_orchestration_working: true`
**Success Criteria**: Coordinates all 4 agents, handles errors, reports status

---

#### Step 3.2: Pipeline Integration & Testing
**Duration**: 1 hour
**Preconditions**: `pipeline_orchestration_working: true`

**Actions**:
```python
# Create app/src/pipeline.py - Main entry point
import asyncio
from agents.coordinator import CoordinatorAgent

async def main():
    coordinator = CoordinatorAgent()

    # Single document test
    result = await coordinator.process_document("test_docs/sample.pdf")
    print(f"Single document: {result}")

    # Batch processing test
    files = ["doc1.pdf", "doc2.docx", "doc3.xlsx"]
    batch_result = await coordinator.process_batch(files)
    print(f"Batch processing: {batch_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Effects**: `end_to_end_pipeline_working: true`
**Success Criteria**: 100+ documents/hour, < 3s per document, 90%+ success rate

---

### Phase 4: Error Handling & Monitoring (Refinement)
**Duration**: 2 hours
**Components**: Logging, metrics, error recovery

**Actions**:
1. Add comprehensive logging to all agents
2. Implement retry logic for transient failures
3. Create status dashboard (CLI or web)
4. Add performance metrics tracking
5. Implement graceful degradation

**Effects**: `production_ready: true`
**Success Criteria**: All errors logged, automatic recovery, real-time monitoring

---

## 4. Technology Requirements

### Python Environment
```bash
# Python 3.10+
python --version  # Should be >= 3.10

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Required Dependencies
```txt
# app/requirements.txt

# Document Processing
PyPDF2>=3.0.0
pdfplumber>=0.10.0
python-docx>=1.0.0
openpyxl>=3.1.0
pandas>=2.0.0

# Text Processing
nltk>=3.8
spacy>=3.7.0  # Optional for advanced chunking

# AI & Embeddings
openai>=1.0.0
anthropic>=0.18.0  # For future RAG implementation

# Database
supabase>=2.0.0
psycopg2-binary>=2.9.0
asyncpg>=0.29.0

# Async & HTTP
aiohttp>=3.9.0
asyncio>=3.4.3

# Utilities
tenacity>=8.0.0  # Retry logic
python-dotenv>=1.0.0  # Environment variables
pydantic>=2.0.0  # Data validation

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### Environment Variables
```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-xxxxx
OPENAI_API_KEY=sk-xxxxx
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJxxxxx
SUPABASE_SERVICE_KEY=eyJxxxxx

# Pipeline Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_PARALLEL_DOCUMENTS=10
BATCH_EMBEDDING_SIZE=100
```

### Database Schema Verification
```sql
-- Run in Supabase SQL Editor

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify tables exist
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- Verify pgvector index
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks';
```

---

## 5. Success Validation Checklist

### Unit Tests (Per Agent)
- [ ] Extractor: Test all 5 file formats
- [ ] Extractor: Verify metadata extraction
- [ ] Chunker: Validate chunk sizes (400-600 words)
- [ ] Chunker: Test semantic boundary detection
- [ ] Embedder: Confirm 1536-dimensional vectors
- [ ] Embedder: Validate batch processing
- [ ] Writer: Test transaction rollback
- [ ] Writer: Verify pgvector storage

### Integration Tests (Pipeline)
- [ ] Single document: PDF → chunks → embeddings → database
- [ ] Batch processing: 10 documents in parallel
- [ ] Error handling: Invalid file, API failure, database error
- [ ] Performance: < 3 seconds per document
- [ ] Concurrency: 10+ simultaneous documents

### System Tests (End-to-End)
- [ ] Upload 100 documents, verify all processed
- [ ] Check database contains expected number of chunks
- [ ] Verify embeddings are searchable via pgvector
- [ ] Test error recovery (deliberate failures)
- [ ] Monitor memory usage during batch processing

### Performance Benchmarks
- [ ] Document processing speed: 100+ docs/hour
- [ ] Embedding generation: < 1 second per chunk (batch)
- [ ] Database write: < 500ms per chunk (batch)
- [ ] End-to-end: < 3 seconds per document
- [ ] Parallel throughput: 10 concurrent documents

---

## 6. Error Handling & Recovery Strategies

### Extractor Failures
**Scenarios**: Corrupted file, unsupported format, read permissions
**Recovery**:
1. Log error with file path and format
2. Skip to next document
3. Report failure in batch summary
4. Store failed file path for manual review

### Chunker Failures
**Scenarios**: Empty text, invalid encoding, extremely large documents
**Recovery**:
1. Validate text before chunking
2. Use fallback fixed-size strategy if semantic fails
3. Split large documents into sub-batches
4. Log warning and continue

### Embedder Failures
**Scenarios**: Rate limiting, API timeout, invalid text
**Recovery**:
1. Implement exponential backoff retry (3 attempts)
2. Batch processing: continue with remaining chunks
3. Cache successful embeddings before failure
4. Queue failed chunks for later retry

### Writer Failures
**Scenarios**: Database connection lost, constraint violation, disk full
**Recovery**:
1. Transaction rollback (automatic)
2. Retry write with exponential backoff
3. Store failed records in dead-letter queue
4. Alert administrator if repeated failures

### Coordinator Failures
**Scenarios**: Agent crash, memory overflow, infinite loop
**Recovery**:
1. Timeout protection (max 5 minutes per document)
2. Graceful shutdown of all agents
3. Persist processing state for resume
4. Generate failure report with diagnostics

---

## 7. Performance Optimization Techniques

### Parallel Processing (asyncio)
```python
# Process 10 documents concurrently
async def process_batch(file_paths: list) -> dict:
    semaphore = asyncio.Semaphore(10)  # Limit concurrency

    async def process_with_limit(path):
        async with semaphore:
            return await process_document(path)

    tasks = [process_with_limit(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Batch Embedding Generation
```python
# Process 100 chunks in single API call
async def generate_embeddings_batch(chunks: list[str]) -> list[vector]:
    BATCH_SIZE = 100
    batches = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    all_embeddings = []
    for batch in batches:
        embeddings = await openai_client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        all_embeddings.extend([e.embedding for e in embeddings.data])

    return all_embeddings
```

### Database Batch Writes
```python
# Insert 50 chunks in single transaction
async def write_chunks_batch(chunks: list[dict]):
    BATCH_SIZE = 50
    batches = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    for batch in batches:
        supabase.table("document_chunks").insert(batch).execute()
```

### Memory Management
```python
# Stream large documents instead of loading fully
def extract_large_pdf(path: str) -> Iterator[str]:
    """Yield pages instead of loading entire document"""
    with open(path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            yield page.extract_text()
```

---

## 8. Monitoring & Observability

### Key Metrics to Track
```python
# Metrics dictionary
metrics = {
    "documents_processed": 0,
    "chunks_created": 0,
    "embeddings_generated": 0,
    "database_writes": 0,
    "errors": {
        "extraction": 0,
        "chunking": 0,
        "embedding": 0,
        "writing": 0
    },
    "performance": {
        "avg_extraction_time": 0.0,
        "avg_chunking_time": 0.0,
        "avg_embedding_time": 0.0,
        "avg_write_time": 0.0,
        "total_pipeline_time": 0.0
    }
}
```

### Logging Strategy
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(agent)s - %(message)s',
    handlers=[
        logging.FileHandler('app/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

# Agent-specific loggers
extractor_logger = logging.getLogger('ExtractorAgent')
chunker_logger = logging.getLogger('ChunkerAgent')
embedder_logger = logging.getLogger('EmbedderAgent')
writer_logger = logging.getLogger('WriterAgent')
coordinator_logger = logging.getLogger('CoordinatorAgent')
```

### Status Dashboard (CLI)
```python
# Real-time processing status
def display_status(coordinator: CoordinatorAgent):
    """
    Processing Status:
    ==================
    Documents: 45/100 (45% complete)
    Current Stage: Embedding

    Agent Status:
    - Extractor: IDLE
    - Chunker: IDLE
    - Embedder: BUSY (3 tasks)
    - Writer: BUSY (2 tasks)

    Errors: 2 (view details with --errors)
    ETA: 3 minutes 12 seconds
    """
    pass
```

---

## 9. Testing Strategy

### Test Data Preparation
```bash
# Create test document corpus
app/tests/fixtures/
├── sample.pdf          # Simple PDF
├── complex.pdf         # Multi-column, tables
├── document.docx       # Word doc
├── spreadsheet.xlsx    # Excel file
├── readme.md           # Markdown
└── notes.txt           # Plain text
```

### Unit Test Example
```python
# app/tests/test_extractor.py
import pytest
from agents.extractor import ExtractorAgent

@pytest.fixture
def extractor():
    return ExtractorAgent()

def test_extract_pdf(extractor):
    result = extractor.extract_text("tests/fixtures/sample.pdf")
    assert result["format"] == "pdf"
    assert len(result["text"]) > 100
    assert "metadata" in result

def test_extract_docx(extractor):
    result = extractor.extract_text("tests/fixtures/document.docx")
    assert result["format"] == "docx"
    assert result["metadata"]["title"] is not None
```

### Integration Test Example
```python
# app/tests/test_pipeline.py
import pytest
from agents.coordinator import CoordinatorAgent

@pytest.mark.asyncio
async def test_full_pipeline():
    coordinator = CoordinatorAgent()
    result = await coordinator.process_document("tests/fixtures/sample.pdf")

    assert result["status"] == "success"
    assert result["chunks"] > 0

    # Verify database storage
    chunks = supabase.table("document_chunks").select("*").eq("document_id", result["doc_id"]).execute()
    assert len(chunks.data) == result["chunks"]
    assert chunks.data[0]["embedding"] is not None
```

---

## 10. Implementation Timeline

### Week 1: Foundation & Core Agents
**Days 1-2**: Environment setup, database schema, configuration
**Days 3-4**: Implement Extractor + Chunker agents
**Day 5**: Implement Embedder agent

### Week 2: Pipeline Integration
**Days 1-2**: Implement Writer agent
**Days 3-4**: Implement Coordinator + pipeline orchestration
**Day 5**: Integration testing & bug fixes

### Week 3: Optimization & Production Readiness
**Days 1-2**: Error handling, retry logic, monitoring
**Days 3-4**: Performance optimization (batch processing, parallelism)
**Day 5**: End-to-end testing with 100+ documents

---

## 11. Next Steps After Pipeline Completion

Once the 5-agent pipeline is operational, the following enhancements become possible:

### Session 6: RAG Implementation
**Preconditions**: `pipeline_complete: true`, `chunks_stored_with_embeddings: true`

**Actions**:
1. Implement semantic search query
2. Build context assembly from top-K chunks
3. Integrate OpenRouter for answer generation
4. Add citation tracking

### Session 7: Advanced Parsing
**Enhancements to Extractor**:
- Complex PDF layouts (multi-column, tables)
- Preserve document structure (headings, sections)
- Extract images and diagrams
- Handle scanned PDFs (OCR)

### Session 8: Semantic Search Optimization
**Enhancements to Embedder & Writer**:
- HNSW index tuning (m, ef_construction)
- Hybrid search (semantic + keyword)
- Reranking models (Cohere, Voyage AI)
- Query expansion

### Session 9: Learning System
**New Agents**:
- Feedback collector
- Learning algorithm (adjust retrieval)
- Personalization engine
- Memory manager

### Session 10: Quality Assurance
**Evaluation Framework**:
- RAGAS metrics (faithfulness, relevance)
- TruLens monitoring
- A/B testing (Claude vs GPT-4)
- Production quality gates

---

## 12. Conclusion

This GOAP implementation plan provides a complete roadmap for building DocuMind's 5-agent document processing pipeline. By following the dependency-based implementation order and adhering to the defined preconditions, actions, and success criteria, the pipeline will achieve:

✅ **100+ documents/hour processing speed**
✅ **< 3 seconds per document end-to-end**
✅ **90%+ success rate with error recovery**
✅ **Parallel processing at each pipeline stage**
✅ **Production-ready monitoring and observability**

**Total Estimated Implementation Time**: 13 hours (across 3 weeks)

**Success Metric**: Pipeline processes 100 test documents with 90%+ success rate in under 5 minutes.

---

**Next Action**: Begin Phase 1 (Foundation) by verifying Supabase connection and enabling pgvector extension.

**Planning Methodology**: Goal-Oriented Action Planning (GOAP) with A* pathfinding for optimal action sequences.

**Continuous Improvement**: Monitor metrics, adjust batch sizes, optimize based on real-world performance data.
