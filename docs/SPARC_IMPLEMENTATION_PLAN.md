# SPARC Implementation Plan: Production-Ready Document Processing System

**Project**: DocuMind Unified Document Processor
**Methodology**: SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)
**Objective**: Build a production-ready document processing system with auto-detection, metadata enrichment, and DocuMind database integration

---

## Executive Summary

This plan follows the SPARC methodology to build a unified document processor that automatically detects file formats, extracts content with tables, enriches metadata, formats output for LLM consumption, and integrates with the DocuMind database via MCP tools.

### Key Deliverables
1. **Unified Processor Class** - Single entry point for all document types
2. **Metadata Enrichment Engine** - Comprehensive metadata extraction and tagging
3. **LLM-Optimized Formatter** - Markdown-formatted output with intelligent chunking
4. **Database Integration** - MCP-based upload with full-text search support
5. **Test Suite** - Complete unit and integration tests
6. **Documentation** - Usage examples and API reference

---

## Phase S: Specification

**Objective**: Define exact requirements, interfaces, data structures, and acceptance criteria

### S1: System Requirements Definition

#### S1.1 Input Requirements
- **Supported Formats**: PDF, DOCX, CSV, XLSX, TXT
- **File Size Limits**: Up to 50MB per document
- **Character Encoding**: UTF-8, Latin-1, CP1252 (auto-detect)
- **Input Validation**: Path sanitization, extension verification, file existence checks

#### S1.2 Output Requirements
- **Content Format**: Clean Markdown with preserved structure
- **Table Format**: Markdown tables with headers and alignment
- **Metadata Structure**: JSONB-compatible nested dictionary
- **Chunk Format**: 500-1000 words with 10% overlap, respecting section boundaries

#### S1.3 Functional Requirements

**FR-1: Document Processing**
```python
process_document(file_path: str) -> ProcessedDocument
```
- Auto-detect file format from extension
- Route to appropriate extractor (PDF, DOCX, CSV, TXT)
- Extract text, tables, and embedded metadata
- Generate unified output structure

**FR-2: Metadata Enrichment**
```python
enrich_metadata(file_path: str, content: str, format_specific_metadata: Dict) -> EnrichedMetadata
```
- **Basic Metadata**: filename, size, dates, word count, read time
- **Structure Metadata**: heading count, section count, list items, table presence
- **Entity Extraction**: emails, URLs, dates, phone numbers
- **Topic Detection**: keyword-based categorization, auto-tagging
- **Document Fingerprint**: SHA-256 hash for duplicate detection

**FR-3: LLM Optimization**
```python
format_for_llm(content: str, tables: List[Dict], metadata: Dict) -> str
```
- Convert to clean Markdown format
- Preserve heading hierarchy (#, ##, ###)
- Format tables with proper alignment
- Include metadata header
- Normalize whitespace and encoding

**FR-4: Content Chunking**
```python
chunk_content(text: str, metadata: Dict) -> List[Chunk]
```
- Target: 500-1000 words per chunk
- 10% overlap between chunks (50-100 words)
- Respect section boundaries (don't split mid-heading)
- Preserve sentence integrity
- Include chunk metadata (index, position, overlap indicator)

**FR-5: Database Integration**
```python
upload_to_documind(processed_doc: ProcessedDocument, chunks: List[Chunk]) -> UploadResult
```
- Use MCP `upload_document` tool
- Store metadata in JSONB column
- Store chunks with position markers
- Enable full-text search
- Support batch uploads

#### S1.4 Non-Functional Requirements

**Performance**
- Process 100-page PDF in < 5 seconds
- Extract metadata in < 1 second
- Support concurrent processing (thread-safe)

**Reliability**
- 99% extraction success rate for supported formats
- Graceful degradation for corrupted files
- Comprehensive error messages

**Maintainability**
- Modular architecture (< 500 lines per file)
- Type hints for all public methods
- Docstrings with examples
- 80%+ test coverage

**Security**
- Path traversal protection
- File size validation
- Extension whitelist enforcement
- No code execution from documents

### S2: Interface Specifications

#### S2.1 Core Data Structures

```python
@dataclass
class ProcessedDocument:
    """Complete processed document with all metadata and content."""

    # Identifiers
    document_id: str  # UUID
    file_path: str
    file_name: str

    # Content
    content: str  # Markdown-formatted
    raw_content: str  # Original extracted text
    tables: List[TableData]

    # Metadata
    metadata: EnrichedMetadata

    # Processing info
    processed_at: datetime
    processing_time_seconds: float
    extractor_used: str
    success: bool
    errors: List[str]


@dataclass
class EnrichedMetadata:
    """Comprehensive document metadata."""

    # Basic metadata
    basic: BasicMetadata

    # Structure analysis
    structure: StructureMetadata

    # Entity extraction
    entities: EntityMetadata

    # Topic classification
    topics: TopicMetadata

    # Document fingerprint
    fingerprint: str  # SHA-256 hash

    # Format-specific metadata (PDF, DOCX, etc.)
    format_metadata: Dict[str, Any]


@dataclass
class BasicMetadata:
    """Basic file and content metadata."""
    file_name: str
    file_path: str
    file_size_bytes: int
    file_type: str  # pdf, docx, csv, txt
    created_at: datetime
    modified_at: datetime
    word_count: int
    character_count: int
    line_count: int
    estimated_read_time_minutes: int


@dataclass
class StructureMetadata:
    """Document structure analysis."""
    heading_count: int
    headings: List[HeadingInfo]
    section_count: int
    list_items: int
    numbered_lists: int
    has_tables: bool
    table_count: int
    max_heading_level: int


@dataclass
class EntityMetadata:
    """Extracted entities."""
    emails: List[str]
    urls: List[str]
    dates: List[str]
    phone_numbers: List[str]
    entity_count: int


@dataclass
class TopicMetadata:
    """Topic classification and tagging."""
    suggested_topics: List[str]  # hr, security, engineering, finance, legal, sales
    suggested_tags: List[str]  # top 5 frequent meaningful words
    auto_category: str  # primary category


@dataclass
class TableData:
    """Extracted table with metadata."""
    table_id: str
    page_or_location: int
    rows: int
    columns: int
    headers: List[str]
    data: List[List[str]]
    markdown: str


@dataclass
class Chunk:
    """Content chunk with metadata."""
    chunk_id: str
    document_id: str
    content: str
    word_count: int
    start_position: int
    end_position: int
    chunk_index: int
    total_chunks: int
    has_overlap: bool
    section_heading: Optional[str]
    metadata_tags: List[str]


@dataclass
class UploadResult:
    """Result of database upload."""
    success: bool
    document_id: str
    chunk_ids: List[str]
    upload_time_seconds: float
    error: Optional[str]
```

#### S2.2 API Methods

```python
class DocumentProcessor:
    """Unified document processor."""

    def process_document(self, file_path: str) -> ProcessedDocument:
        """Main entry point - process any supported document."""

    def detect_format(self, file_path: str) -> str:
        """Detect document format from extension."""

    def extract_content(self, file_path: str, format: str) -> ExtractionResult:
        """Extract raw content using format-specific extractor."""

    def enrich_metadata(self, file_path: str, content: str,
                       format_metadata: Dict) -> EnrichedMetadata:
        """Generate comprehensive metadata."""

    def format_for_llm(self, content: str, tables: List[TableData],
                      metadata: EnrichedMetadata) -> str:
        """Format content as clean Markdown for LLM consumption."""

    def chunk_content(self, markdown_content: str,
                     document_id: str) -> List[Chunk]:
        """Split content into overlapping chunks."""

    def upload_to_documind(self, processed_doc: ProcessedDocument) -> UploadResult:
        """Upload to DocuMind database via MCP."""

    def process_batch(self, file_paths: List[str],
                     parallel: bool = True) -> List[ProcessedDocument]:
        """Process multiple documents."""
```

### S3: Acceptance Criteria

#### AC-1: Format Detection
- [x] Correctly identifies PDF, DOCX, CSV, XLSX, TXT from extensions
- [x] Handles case-insensitive extensions (.PDF, .Pdf, .pdf)
- [x] Rejects unsupported formats with clear error message
- [x] Validates file existence before processing

#### AC-2: Content Extraction
- [x] PDF: Extracts text from all pages with pdfplumber
- [x] PDF: Extracts tables with structure preservation
- [x] DOCX: Extracts paragraphs with heading styles
- [x] DOCX: Extracts tables with row/column structure
- [x] CSV: Loads with pandas, handles delimiters correctly
- [x] TXT: Handles UTF-8, Latin-1, CP1252 encodings
- [x] All formats: Returns empty content on failure (not crash)

#### AC-3: Metadata Enrichment
- [x] Basic: Extracts file size, dates, word count
- [x] Structure: Detects headings, sections, lists
- [x] Entities: Extracts emails, URLs, dates, phone numbers
- [x] Topics: Auto-categorizes into predefined topics
- [x] Fingerprint: Generates SHA-256 hash
- [x] All metadata is JSON-serializable

#### AC-4: LLM Formatting
- [x] Outputs valid Markdown syntax
- [x] Preserves heading hierarchy
- [x] Tables formatted with | separators and alignment
- [x] Metadata included as YAML frontmatter or header
- [x] Whitespace normalized (single blank lines)
- [x] Special characters properly escaped

#### AC-5: Content Chunking
- [x] Chunks are 500-1000 words (target: 750)
- [x] 10% overlap between consecutive chunks
- [x] No mid-sentence splits
- [x] Respects section boundaries (headings)
- [x] Each chunk has position metadata
- [x] Last chunk can be smaller than target size

#### AC-6: Database Upload
- [x] Successfully calls MCP `upload_document` tool
- [x] Metadata stored in JSONB column
- [x] Full-text search enabled on content
- [x] Chunks linked to parent document
- [x] Duplicate detection via fingerprint
- [x] Batch upload supports 10+ documents

#### AC-7: Error Handling
- [x] Corrupted files don't crash the system
- [x] Missing files return clear error messages
- [x] Unsupported formats are rejected gracefully
- [x] Partial extraction succeeds (e.g., some tables fail)
- [x] All errors are logged with context

#### AC-8: Testing
- [x] Unit tests for each extractor (PDF, DOCX, CSV, TXT)
- [x] Unit tests for metadata extraction
- [x] Unit tests for chunking algorithm
- [x] Integration test processing all 4 sample documents
- [x] Test error conditions (missing file, corrupted file)
- [x] Test coverage > 80%

---

## Phase P: Pseudocode

**Objective**: Design algorithms for document routing, chunking, metadata extraction

### P1: Main Processing Algorithm

```
FUNCTION process_document(file_path: str) -> ProcessedDocument:
    START_TIMER

    # Step 1: Validation and Format Detection
    VALIDATE_FILE_PATH(file_path)
    format = DETECT_FORMAT(file_path)
    document_id = GENERATE_UUID()

    # Step 2: Content Extraction
    extraction_result = EXTRACT_CONTENT(file_path, format)
    IF extraction_result.success == False:
        RETURN FAILURE_DOCUMENT(document_id, extraction_result.error)

    raw_content = extraction_result.text
    tables = extraction_result.tables
    format_metadata = extraction_result.metadata

    # Step 3: Metadata Enrichment
    enriched_metadata = ENRICH_METADATA(
        file_path, raw_content, format_metadata
    )

    # Step 4: LLM Formatting
    markdown_content = FORMAT_FOR_LLM(
        raw_content, tables, enriched_metadata
    )

    # Step 5: Content Chunking
    chunks = CHUNK_CONTENT(markdown_content, document_id)

    # Step 6: Build Result
    processed_doc = ProcessedDocument(
        document_id = document_id,
        file_path = file_path,
        content = markdown_content,
        raw_content = raw_content,
        tables = tables,
        metadata = enriched_metadata,
        chunks = chunks,
        processing_time = STOP_TIMER(),
        success = True
    )

    RETURN processed_doc
END FUNCTION
```

### P2: Format Detection Algorithm

```
FUNCTION detect_format(file_path: str) -> str:
    extension = GET_FILE_EXTENSION(file_path).lower()

    SWITCH extension:
        CASE ".pdf":
            RETURN "pdf"
        CASE ".docx":
            RETURN "docx"
        CASE ".csv":
            RETURN "csv"
        CASE ".xlsx", ".xls":
            RETURN "xlsx"
        CASE ".txt", ".md":
            RETURN "txt"
        DEFAULT:
            RAISE UnsupportedFormatError(extension)
END FUNCTION
```

### P3: Content Extraction Routing

```
FUNCTION extract_content(file_path: str, format: str) -> ExtractionResult:
    SWITCH format:
        CASE "pdf":
            RETURN PDF_EXTRACTOR.extract(file_path)
        CASE "docx":
            RETURN DOCX_EXTRACTOR.extract(file_path)
        CASE "csv":
            RETURN CSV_EXTRACTOR.extract(file_path)
        CASE "xlsx":
            RETURN XLSX_EXTRACTOR.extract(file_path)
        CASE "txt":
            RETURN TXT_EXTRACTOR.extract(file_path)
        DEFAULT:
            RETURN FAILURE_RESULT("Unknown format: " + format)
END FUNCTION
```

### P4: Metadata Enrichment Algorithm

```
FUNCTION enrich_metadata(file_path, content, format_metadata) -> EnrichedMetadata:
    # Basic metadata from filesystem
    basic = EXTRACT_BASIC_METADATA(file_path, content)

    # Structure analysis
    headings = FIND_HEADINGS(content)
    structure = StructureMetadata(
        heading_count = LENGTH(headings),
        headings = headings,
        section_count = LENGTH(headings) + 1,
        list_items = COUNT_LIST_ITEMS(content),
        numbered_lists = COUNT_NUMBERED_LISTS(content),
        has_tables = DETECT_TABLES(content),
        table_count = COUNT_TABLES(content)
    )

    # Entity extraction using regex
    entities = EntityMetadata(
        emails = EXTRACT_EMAILS(content),
        urls = EXTRACT_URLS(content),
        dates = EXTRACT_DATES(content),
        phone_numbers = EXTRACT_PHONE_NUMBERS(content)
    )

    # Topic classification
    topics = CLASSIFY_TOPICS(content)

    # Document fingerprint
    fingerprint = SHA256_HASH(content)

    RETURN EnrichedMetadata(
        basic, structure, entities, topics, fingerprint, format_metadata
    )
END FUNCTION
```

### P5: Topic Classification Algorithm

```
FUNCTION classify_topics(content: str) -> TopicMetadata:
    content_lower = LOWERCASE(content)

    # Define topic keywords
    topic_keywords = {
        "hr": ["employee", "benefit", "vacation", "policy", "hire", "salary"],
        "security": ["password", "encryption", "firewall", "authentication"],
        "engineering": ["code", "software", "api", "database", "deployment"],
        "finance": ["budget", "expense", "revenue", "cost", "invoice"],
        "legal": ["contract", "agreement", "liability", "compliance"],
        "sales": ["customer", "revenue", "deal", "quota", "pipeline"]
    }

    # Find matching topics
    matched_topics = []
    FOR each topic, keywords IN topic_keywords:
        FOR each keyword IN keywords:
            IF keyword IN content_lower:
                ADD topic TO matched_topics
                BREAK

    # Extract frequent words for tags
    words = EXTRACT_WORDS(content_lower, min_length=4)
    word_freq = COUNT_FREQUENCIES(words)
    REMOVE_STOPWORDS(word_freq)
    top_tags = GET_TOP_N(word_freq, n=5)

    # Determine primary category
    primary_category = matched_topics[0] IF matched_topics ELSE "general"

    RETURN TopicMetadata(
        suggested_topics = matched_topics,
        suggested_tags = top_tags,
        auto_category = primary_category
    )
END FUNCTION
```

### P6: LLM Formatting Algorithm

```
FUNCTION format_for_llm(content, tables, metadata) -> str:
    output = []

    # Add YAML frontmatter with key metadata
    output.ADD("---")
    output.ADD("title: " + metadata.basic.file_name)
    output.ADD("type: " + metadata.basic.file_type)
    output.ADD("topics: " + JOIN(metadata.topics.suggested_topics, ", "))
    output.ADD("word_count: " + metadata.basic.word_count)
    output.ADD("---")
    output.ADD("")

    # Process content line by line
    FOR each line IN content.split("\n"):
        # Normalize whitespace
        line = STRIP_WHITESPACE(line)

        # Skip empty lines (will add single blank lines between sections)
        IF line == "":
            IF last_line_was_not_empty:
                output.ADD("")
            CONTINUE

        # Detect and format headings
        IF IS_HEADING(line):
            level = DETECT_HEADING_LEVEL(line)
            text = EXTRACT_HEADING_TEXT(line)
            output.ADD("#" * level + " " + text)
        ELSE:
            output.ADD(line)

    # Add tables in Markdown format
    IF tables NOT EMPTY:
        output.ADD("")
        output.ADD("## Tables")
        output.ADD("")
        FOR each table IN tables:
            output.ADD(table.markdown)
            output.ADD("")

    RETURN JOIN(output, "\n")
END FUNCTION
```

### P7: Content Chunking Algorithm

```
FUNCTION chunk_content(markdown_content, document_id) -> List[Chunk]:
    TARGET_SIZE = 750  # words
    OVERLAP_PERCENT = 0.10
    MIN_CHUNK_SIZE = 500
    MAX_CHUNK_SIZE = 1000

    # Split into sentences while preserving headings
    elements = SPLIT_INTO_ELEMENTS(markdown_content)  # headings + sentences

    chunks = []
    current_chunk = []
    current_word_count = 0
    current_heading = None
    overlap_elements = []

    FOR each element IN elements:
        element_word_count = COUNT_WORDS(element)

        # Check if element is a heading
        IF IS_HEADING(element):
            # If we have accumulated content, create chunk before heading
            IF current_word_count >= MIN_CHUNK_SIZE:
                chunk_text = JOIN(current_chunk, " ")
                chunk = CREATE_CHUNK(
                    document_id, chunk_text, LENGTH(chunks),
                    current_heading, has_overlap=(LENGTH(chunks) > 0)
                )
                chunks.ADD(chunk)

                # Calculate overlap
                overlap_elements = GET_LAST_N_WORDS(current_chunk, TARGET_SIZE * OVERLAP_PERCENT)
                current_chunk = overlap_elements
                current_word_count = COUNT_WORDS(JOIN(overlap_elements))

            current_heading = EXTRACT_HEADING_TEXT(element)
            current_chunk.ADD(element)
            current_word_count += element_word_count

        # Regular sentence
        ELSE:
            # Check if adding this would exceed max size
            IF current_word_count + element_word_count > MAX_CHUNK_SIZE:
                # Create chunk
                chunk_text = JOIN(current_chunk, " ")
                chunk = CREATE_CHUNK(
                    document_id, chunk_text, LENGTH(chunks),
                    current_heading, has_overlap=(LENGTH(chunks) > 0)
                )
                chunks.ADD(chunk)

                # Calculate overlap
                overlap_elements = GET_LAST_N_WORDS(current_chunk, TARGET_SIZE * OVERLAP_PERCENT)
                current_chunk = overlap_elements
                current_word_count = COUNT_WORDS(JOIN(overlap_elements))

            current_chunk.ADD(element)
            current_word_count += element_word_count

    # Add final chunk if remaining content
    IF current_chunk:
        chunk_text = JOIN(current_chunk, " ")
        chunk = CREATE_CHUNK(
            document_id, chunk_text, LENGTH(chunks),
            current_heading, has_overlap=(LENGTH(chunks) > 0)
        )
        chunks.ADD(chunk)

    RETURN chunks
END FUNCTION
```

### P8: Database Upload Algorithm

```
FUNCTION upload_to_documind(processed_doc) -> UploadResult:
    START_TIMER

    TRY:
        # Prepare metadata for JSONB storage
        metadata_json = SERIALIZE_TO_JSON(processed_doc.metadata)

        # Call MCP upload_document tool
        mcp_result = MCP_CALL(
            tool = "upload_document",
            params = {
                "title": processed_doc.metadata.basic.file_name,
                "content": processed_doc.content,
                "file_type": processed_doc.metadata.basic.file_type,
                "metadata": metadata_json
            }
        )

        document_id = mcp_result.document_id

        # Upload chunks as separate records
        chunk_ids = []
        FOR each chunk IN processed_doc.chunks:
            chunk_result = MCP_CALL(
                tool = "upload_document",
                params = {
                    "title": chunk.chunk_id,
                    "content": chunk.content,
                    "file_type": "chunk",
                    "metadata": {
                        "parent_document_id": document_id,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "word_count": chunk.word_count,
                        "section_heading": chunk.section_heading
                    }
                }
            )
            chunk_ids.ADD(chunk_result.document_id)

        RETURN UploadResult(
            success = True,
            document_id = document_id,
            chunk_ids = chunk_ids,
            upload_time = STOP_TIMER()
        )

    CATCH error:
        RETURN UploadResult(
            success = False,
            error = error.message,
            upload_time = STOP_TIMER()
        )
END FUNCTION
```

---

## Phase A: Architecture

**Objective**: Design system components, class hierarchy, data flow, error handling

### A1: System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DocumentProcessor                        │
│                   (Orchestrator/Facade)                      │
│  - process_document()                                        │
│  - process_batch()                                           │
│  - upload_to_documind()                                      │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├─────> FormatDetector
           │       - detect_format()
           │       - validate_path()
           │
           ├─────> ExtractionRouter
           │       │
           │       ├─> PDFExtractor (existing)
           │       │   - extract_text()
           │       │   - extract_tables()
           │       │
           │       ├─> DocxExtractor (existing)
           │       │   - extract()
           │       │   - format_for_llm()
           │       │
           │       ├─> SpreadsheetExtractor (existing)
           │       │   - extract_csv()
           │       │   - extract_excel()
           │       │
           │       └─> TextExtractor (new)
           │           - extract()
           │
           ├─────> MetadataEnricher (extends existing)
           │       - enrich_metadata()
           │       - extract_basic_metadata()
           │       - extract_structure()
           │       - extract_entities()
           │       - extract_topics()
           │       - generate_fingerprint()
           │
           ├─────> LLMFormatter (new)
           │       - format_for_llm()
           │       - format_tables()
           │       - normalize_whitespace()
           │       - create_frontmatter()
           │
           ├─────> ContentChunker (extends existing)
           │       - chunk_text()
           │       - split_into_elements()
           │       - create_chunk()
           │       - calculate_overlap()
           │
           └─────> DocuMindUploader (new)
                   - upload_document()
                   - upload_chunks()
                   - check_duplicate()
```

### A2: Class Hierarchy

```python
# src/documind/processor.py
class DocumentProcessor:
    """Main orchestrator for document processing."""

    def __init__(self):
        self.format_detector = FormatDetector()
        self.extraction_router = ExtractionRouter()
        self.metadata_enricher = MetadataEnricher()
        self.llm_formatter = LLMFormatter()
        self.content_chunker = ContentChunker()
        self.documind_uploader = DocuMindUploader()

    def process_document(self, file_path: str) -> ProcessedDocument:
        """Main entry point."""
        ...

    def process_batch(self, file_paths: List[str],
                     parallel: bool = True) -> List[ProcessedDocument]:
        """Process multiple documents."""
        ...

    def upload_to_documind(self, processed_doc: ProcessedDocument) -> UploadResult:
        """Upload to database."""
        ...


# src/documind/format_detector.py
class FormatDetector:
    """Detect and validate document formats."""

    SUPPORTED_FORMATS = {"pdf", "docx", "csv", "xlsx", "xls", "txt", "md"}

    def detect_format(self, file_path: str) -> str:
        """Detect format from extension."""
        ...

    def validate_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate file path and return error if invalid."""
        ...


# src/documind/extraction_router.py
class ExtractionRouter:
    """Route extraction requests to appropriate extractor."""

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        self.spreadsheet_extractor = SpreadsheetExtractor()
        self.text_extractor = TextExtractor()

    def extract(self, file_path: str, format: str) -> ExtractionResult:
        """Route to appropriate extractor."""
        ...


# src/documind/llm_formatter.py
class LLMFormatter:
    """Format extracted content for LLM consumption."""

    def format_for_llm(self, content: str, tables: List[TableData],
                      metadata: EnrichedMetadata) -> str:
        """Convert to clean Markdown with metadata."""
        ...

    def create_frontmatter(self, metadata: EnrichedMetadata) -> str:
        """Create YAML frontmatter."""
        ...

    def format_tables(self, tables: List[TableData]) -> str:
        """Format tables as Markdown."""
        ...

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (single blank lines)."""
        ...


# src/documind/documind_uploader.py
class DocuMindUploader:
    """Upload documents to DocuMind database via MCP."""

    def upload_document(self, processed_doc: ProcessedDocument) -> UploadResult:
        """Upload document with metadata."""
        ...

    def upload_chunks(self, chunks: List[Chunk],
                     parent_document_id: str) -> List[str]:
        """Upload chunks separately."""
        ...

    def check_duplicate(self, fingerprint: str) -> Optional[str]:
        """Check if document already exists."""
        ...
```

### A3: Data Flow

```
┌──────────────┐
│  User Input  │
│  file_path   │
└──────┬───────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 1. VALIDATION & DETECTION                                │
│    - Validate file path                                  │
│    - Detect format from extension                        │
│    - Generate document ID                                │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 2. CONTENT EXTRACTION                                    │
│    Route to: PDF / DOCX / CSV / TXT extractor           │
│    Extract: text, tables, format-specific metadata      │
│    Result: ExtractionResult(text, tables, metadata)     │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 3. METADATA ENRICHMENT                                   │
│    - Basic: file info, word count, dates                │
│    - Structure: headings, sections, lists               │
│    - Entities: emails, URLs, dates, phones              │
│    - Topics: auto-categorization, tagging               │
│    - Fingerprint: SHA-256 hash                          │
│    Result: EnrichedMetadata                             │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 4. LLM FORMATTING                                        │
│    - Add YAML frontmatter                               │
│    - Convert to clean Markdown                          │
│    - Format tables with | separators                    │
│    - Normalize whitespace                               │
│    Result: markdown_content (str)                       │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 5. CONTENT CHUNKING                                      │
│    - Split into 500-1000 word chunks                    │
│    - Add 10% overlap between chunks                     │
│    - Respect section boundaries                         │
│    - Preserve sentence integrity                        │
│    Result: List[Chunk]                                  │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 6. PROCESSED DOCUMENT ASSEMBLY                           │
│    - Combine all results into ProcessedDocument         │
│    - Calculate processing time                          │
│    - Set success/error flags                            │
│    Result: ProcessedDocument                            │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│ 7. DATABASE UPLOAD (optional)                            │
│    - Check for duplicates via fingerprint               │
│    - Upload main document via MCP                       │
│    - Upload chunks with parent linkage                  │
│    Result: UploadResult                                 │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌─────────────────┐
│  Final Output   │
│  ProcessedDoc   │
│  UploadResult   │
└─────────────────┘
```

### A4: Error Handling Strategy

```python
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class ValidationError(ProcessingError):
    """File validation errors."""
    pass

class ExtractionError(ProcessingError):
    """Content extraction errors."""
    pass

class FormattingError(ProcessingError):
    """Formatting errors."""
    pass

class UploadError(ProcessingError):
    """Database upload errors."""
    pass


# Error handling pattern
def process_document(file_path: str) -> ProcessedDocument:
    errors = []

    try:
        # Validation
        is_valid, error = validate_path(file_path)
        if not is_valid:
            raise ValidationError(error)

        # Extraction (partial success allowed)
        try:
            extraction_result = extract_content(file_path)
        except ExtractionError as e:
            errors.append(f"Extraction warning: {e}")
            extraction_result = ExtractionResult(
                success=False,
                text="[Extraction failed]",
                error=str(e)
            )

        # Continue processing even with extraction warnings...

    except ProcessingError as e:
        # Return failure document
        return ProcessedDocument(
            success=False,
            errors=[str(e)],
            ...
        )

    return ProcessedDocument(
        success=True,
        errors=errors,  # May contain warnings
        ...
    )
```

### A5: File Structure

```
src/documind/
├── __init__.py
├── config.py                    # Existing configuration
├── processor.py                 # NEW: Main DocumentProcessor class
├── format_detector.py           # NEW: Format detection and validation
├── extraction_router.py         # NEW: Route to extractors
├── llm_formatter.py             # NEW: LLM-optimized formatting
├── documind_uploader.py         # NEW: MCP-based database upload
├── data_structures.py           # NEW: All dataclasses
├── extractors/
│   ├── __init__.py
│   ├── pdf_extractor.py         # Existing - use as-is
│   ├── docx_extractor.py        # Existing - use as-is
│   ├── spreadsheet_extractor.py # Existing - use as-is
│   ├── metadata_extractor.py    # Existing - extend for fingerprint
│   └── text_extractor.py        # NEW: TXT/MD extraction
└── utils/
    ├── __init__.py
    ├── hashing.py               # SHA-256 fingerprint generation
    └── validation.py            # Path validation utilities

tests/documind/
├── __init__.py
├── test_processor.py            # Integration tests
├── test_format_detector.py
├── test_extraction_router.py
├── test_llm_formatter.py
├── test_documind_uploader.py
├── test_metadata_enricher.py
├── test_content_chunker.py
└── fixtures/
    ├── sample.pdf
    ├── sample.docx
    ├── sample.csv
    └── sample.txt

examples/
├── basic_usage.py               # Simple example
├── batch_processing.py          # Batch example
└── custom_pipeline.py           # Advanced customization
```

### A6: Dependencies

```toml
# pyproject.toml additions
[project]
dependencies = [
    # Existing
    "pdfplumber>=0.10.0",
    "python-docx>=1.0.0",
    "pandas>=2.0.0",
    "openpyxl>=3.1.0",

    # New for processing
    "pydantic>=2.0.0",  # Data validation
    "tiktoken>=0.5.0",  # Token counting
    "python-magic>=0.4.27",  # MIME type detection (optional)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.1",
    "black>=23.7.0",
    "mypy>=1.4.1",
    "ruff>=0.0.282",
]
```

---

## Phase R: Refinement (TDD)

**Objective**: Write tests first, then implement

### R1: Test Strategy

#### R1.1 Test Categories

1. **Unit Tests** (80% coverage)
   - Test each component in isolation
   - Mock external dependencies
   - Fast execution (< 1s per test)

2. **Integration Tests** (critical paths)
   - Test complete workflows
   - Use real sample documents
   - Verify end-to-end processing

3. **Edge Case Tests**
   - Empty files
   - Corrupted files
   - Extremely large files
   - Special characters
   - Unicode handling

#### R1.2 TDD Workflow

For each component:
1. **RED**: Write failing test
2. **GREEN**: Implement minimal code to pass
3. **REFACTOR**: Clean up implementation
4. **REPEAT**: Next test

### R2: Test Implementation Tasks

#### Task R2.1: Format Detector Tests
**File**: `tests/documind/test_format_detector.py`

```python
# Test cases to write FIRST:
def test_detect_format_pdf():
    """Should return 'pdf' for .pdf files"""

def test_detect_format_case_insensitive():
    """Should handle .PDF, .Pdf, .pdf"""

def test_detect_format_unsupported():
    """Should raise UnsupportedFormatError for .exe files"""

def test_validate_path_file_not_found():
    """Should return (False, error) for missing files"""

def test_validate_path_path_traversal():
    """Should reject ../../../etc/passwd"""

def test_validate_path_symlink():
    """Should reject symlinks for security"""
```

**Implementation**: After tests pass, create `src/documind/format_detector.py`

#### Task R2.2: PDF Extractor Tests
**File**: `tests/documind/extractors/test_pdf_extractor.py`

```python
# Tests for existing PDFExtractor
def test_extract_text_from_simple_pdf():
    """Should extract text from simple_security_policy.pdf"""

def test_extract_tables_from_pdf():
    """Should extract tables from employee_directory.pdf"""

def test_extract_metadata_from_pdf():
    """Should extract author, title, creation date"""

def test_handle_corrupted_pdf():
    """Should return error result, not crash"""

def test_extract_text_from_multipage_pdf():
    """Should concatenate text from all pages"""
```

**Implementation**: Existing `PDFExtractor` - add any missing error handling

#### Task R2.3: DOCX Extractor Tests
**File**: `tests/documind/extractors/test_docx_extractor.py`

```python
def test_extract_paragraphs_from_docx():
    """Should extract all paragraphs from meeting_notes.docx"""

def test_extract_tables_from_docx():
    """Should extract tables from employee_handbook.docx"""

def test_detect_heading_styles():
    """Should identify Heading 1, Heading 2, etc."""

def test_format_for_llm_preserves_structure():
    """Should convert headings to Markdown # syntax"""

def test_handle_empty_docx():
    """Should handle documents with no content"""
```

**Implementation**: Existing `DocxExtractor` - verify tests pass

#### Task R2.4: Spreadsheet Extractor Tests
**File**: `tests/documind/extractors/test_spreadsheet_extractor.py`

```python
def test_extract_csv_with_headers():
    """Should extract employee_data.csv with column names"""

def test_extract_csv_with_special_characters():
    """Should handle commas in quoted fields"""

def test_format_csv_for_llm():
    """Should format as readable markdown table"""

def test_handle_malformed_csv():
    """Should handle inconsistent column counts"""
```

**Implementation**: Existing `SpreadsheetExtractor` - verify tests pass

#### Task R2.5: Text Extractor Tests
**File**: `tests/documind/extractors/test_text_extractor.py`

```python
def test_extract_utf8_text():
    """Should read UTF-8 text files"""

def test_extract_latin1_text():
    """Should fallback to latin-1 encoding"""

def test_extract_markdown_preserves_formatting():
    """Should keep Markdown syntax intact"""

def test_handle_binary_file_gracefully():
    """Should return error for non-text files"""
```

**Implementation**: Create `src/documind/extractors/text_extractor.py`

#### Task R2.6: Metadata Enricher Tests
**File**: `tests/documind/test_metadata_enricher.py`

```python
def test_extract_basic_metadata():
    """Should extract file size, dates, word count"""

def test_extract_structure_metadata():
    """Should count headings, sections, lists"""

def test_extract_entities():
    """Should find emails, URLs, dates, phone numbers"""

def test_classify_topics():
    """Should auto-categorize into topics"""

def test_generate_fingerprint():
    """Should generate consistent SHA-256 hash"""

def test_duplicate_fingerprints():
    """Same content should produce same fingerprint"""
```

**Implementation**: Extend `src/documind/extractors/metadata_extractor.py`

#### Task R2.7: LLM Formatter Tests
**File**: `tests/documind/test_llm_formatter.py`

```python
def test_create_frontmatter():
    """Should create valid YAML frontmatter"""

def test_format_tables_as_markdown():
    """Should use | separators and --- alignment"""

def test_normalize_whitespace():
    """Should replace multiple blank lines with single"""

def test_preserve_heading_hierarchy():
    """Should convert to #, ##, ### format"""

def test_escape_special_characters():
    """Should escape Markdown special chars in content"""
```

**Implementation**: Create `src/documind/llm_formatter.py`

#### Task R2.8: Content Chunker Tests
**File**: `tests/documind/test_content_chunker.py`

```python
def test_chunk_size_within_range():
    """Chunks should be 500-1000 words"""

def test_chunk_overlap_is_10_percent():
    """Adjacent chunks should overlap by ~10%"""

def test_respect_section_boundaries():
    """Should not split mid-heading"""

def test_preserve_sentence_integrity():
    """Should not split mid-sentence"""

def test_handle_short_documents():
    """Documents < 500 words should be single chunk"""

def test_chunk_metadata_correctness():
    """Chunk indices, positions, IDs should be correct"""
```

**Implementation**: Extend existing `src/agents/pipeline/chunker.py`

#### Task R2.9: DocuMind Uploader Tests
**File**: `tests/documind/test_documind_uploader.py`

```python
def test_upload_document_via_mcp(mock_mcp):
    """Should call mcp__documind__upload_document"""

def test_upload_chunks_separately(mock_mcp):
    """Should upload each chunk with parent linkage"""

def test_check_duplicate_by_fingerprint(mock_mcp):
    """Should query for existing fingerprint"""

def test_handle_upload_failure(mock_mcp):
    """Should return UploadResult with error"""

def test_metadata_serializes_to_json():
    """Metadata should be JSON-serializable"""
```

**Implementation**: Create `src/documind/documind_uploader.py`

#### Task R2.10: Integration Tests
**File**: `tests/documind/test_processor_integration.py`

```python
def test_process_pdf_end_to_end():
    """Process simple_security_policy.pdf through full pipeline"""

def test_process_docx_end_to_end():
    """Process employee_handbook.docx through full pipeline"""

def test_process_csv_end_to_end():
    """Process employee_data.csv through full pipeline"""

def test_process_txt_end_to_end():
    """Process plain text file through full pipeline"""

def test_process_batch_multiple_formats():
    """Process all 4 sample documents in batch"""

def test_upload_to_documind_integration(mock_mcp):
    """Full process + upload workflow"""
```

**Implementation**: Create `src/documind/processor.py`

### R3: Implementation Order (TDD Sequence)

1. **Sprint 1**: Foundation (Days 1-2)
   - R2.1: Format Detector (tests → implementation)
   - R2.5: Text Extractor (tests → implementation)
   - R2.6: Metadata Enricher extensions (tests → implementation)

2. **Sprint 2**: Formatting & Chunking (Days 3-4)
   - R2.7: LLM Formatter (tests → implementation)
   - R2.8: Content Chunker extensions (tests → implementation)

3. **Sprint 3**: Integration (Days 5-6)
   - R2.9: DocuMind Uploader (tests → implementation)
   - R2.10: Main Processor (tests → implementation)

4. **Sprint 4**: Validation (Day 7)
   - R2.2-R2.4: Verify existing extractors pass tests
   - R2.10: Integration tests
   - Fix any failing tests

### R4: Test Data Setup

Create test fixtures in `tests/documind/fixtures/`:

```python
# tests/documind/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_pdf_path():
    """Path to simple_security_policy.pdf"""
    return Path("docs/workshops/S7-sample-docs/simple_security_policy.pdf")

@pytest.fixture
def sample_docx_path():
    """Path to employee_handbook.docx"""
    return Path("docs/workshops/S7-sample-docs/employee_handbook.docx")

@pytest.fixture
def sample_csv_path():
    """Path to employee_data.csv"""
    return Path("docs/workshops/S7-sample-docs/employee_data.csv")

@pytest.fixture
def sample_txt_path(tmp_path):
    """Create temporary text file for testing"""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Sample text content for testing.\nSecond line.")
    return txt_file

@pytest.fixture
def mock_mcp_upload(mocker):
    """Mock MCP upload_document tool"""
    return mocker.patch("src.documind.documind_uploader.mcp_upload_document")
```

---

## Phase C: Completion

**Objective**: Integration testing, documentation, performance optimization

### C1: Integration & Quality Assurance

#### Task C1.1: End-to-End Testing
- [ ] Run full test suite (`pytest`)
- [ ] Verify 80%+ code coverage (`pytest --cov`)
- [ ] Test with all 4 sample document types
- [ ] Test batch processing (10+ documents)
- [ ] Performance benchmark: 100-page PDF in < 5s

#### Task C1.2: Error Handling Validation
- [ ] Test corrupted files (manually corrupt a PDF)
- [ ] Test missing files
- [ ] Test unsupported formats
- [ ] Test oversized files (> 50MB)
- [ ] Verify error messages are clear and actionable

#### Task C1.3: Security Audit
- [ ] Path traversal protection works
- [ ] Symlink rejection works
- [ ] File size limits enforced
- [ ] Extension whitelist enforced
- [ ] No code execution from documents

### C2: Documentation

#### Task C2.1: API Documentation
**File**: `docs/api/processor.md`

```markdown
# DocumentProcessor API Reference

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from documind.processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document("path/to/document.pdf")

if result.success:
    print(f"Extracted {result.metadata.basic.word_count} words")
    print(f"Generated {len(result.chunks)} chunks")
else:
    print(f"Error: {result.errors}")
```

## API Methods

### process_document(file_path: str) -> ProcessedDocument
...
```

#### Task C2.2: Usage Examples
**File**: `examples/basic_usage.py`

```python
#!/usr/bin/env python3
"""Basic usage example for DocumentProcessor"""

from documind.processor import DocumentProcessor

def main():
    # Initialize processor
    processor = DocumentProcessor()

    # Process a PDF
    result = processor.process_document("sample.pdf")

    if result.success:
        # Access content
        print("Content:", result.content[:500])

        # Access metadata
        print("Topics:", result.metadata.topics.suggested_topics)

        # Access chunks
        print(f"Created {len(result.chunks)} chunks")
        for chunk in result.chunks[:3]:
            print(f"  Chunk {chunk.chunk_index}: {chunk.word_count} words")

        # Upload to DocuMind
        upload_result = processor.upload_to_documind(result)
        if upload_result.success:
            print(f"Uploaded as document ID: {upload_result.document_id}")
    else:
        print("Processing failed:", result.errors)

if __name__ == "__main__":
    main()
```

#### Task C2.3: Batch Processing Example
**File**: `examples/batch_processing.py`

```python
#!/usr/bin/env python3
"""Batch processing example"""

from documind.processor import DocumentProcessor
from pathlib import Path
import time

def main():
    processor = DocumentProcessor()

    # Find all documents in a directory
    doc_dir = Path("docs/workshops/S7-sample-docs")
    file_paths = [
        str(p) for p in doc_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".docx", ".csv"}
    ]

    print(f"Processing {len(file_paths)} documents...")
    start_time = time.time()

    # Process in parallel
    results = processor.process_batch(file_paths, parallel=True)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.success)

    print(f"Completed in {elapsed:.2f}s")
    print(f"Successful: {successful}/{len(results)}")

    # Upload all successful results
    for result in results:
        if result.success:
            processor.upload_to_documind(result)

if __name__ == "__main__":
    main()
```

#### Task C2.4: README Update
**File**: `README.md` (add section)

```markdown
## Document Processing

DocuMind includes a production-ready document processor that handles PDF, DOCX, CSV, and TXT files.

### Features
- Auto-format detection
- Table extraction
- Comprehensive metadata enrichment
- LLM-optimized Markdown output
- Intelligent chunking with overlap
- Duplicate detection via fingerprinting

### Quick Start
```python
from documind.processor import DocumentProcessor

processor = DocumentProcessor()
result = processor.process_document("document.pdf")
processor.upload_to_documind(result)
```

See [examples/](examples/) for more usage patterns.
```

### C3: Performance Optimization

#### Task C3.1: Profiling
```bash
# Profile processor on large PDF
python -m cProfile -o profile.stats -m examples.basic_usage
python -m pstats profile.stats

# Identify bottlenecks
# Common targets: PDF parsing, entity extraction, chunking
```

#### Task C3.2: Optimization Targets
- [ ] PDF extraction: Use pdfplumber only (remove PyPDF2 fallback if slower)
- [ ] Regex compilation: Compile patterns once in `__init__`
- [ ] Metadata extraction: Parallelize independent operations
- [ ] Chunking: Optimize sentence splitting regex
- [ ] Batch processing: Use thread pool (4-8 workers)

#### Task C3.3: Caching Strategy
```python
# Add optional caching for repeated processing
from functools import lru_cache
import hashlib

class DocumentProcessor:
    def __init__(self, enable_cache=False):
        self.enable_cache = enable_cache

    def process_document(self, file_path: str) -> ProcessedDocument:
        if self.enable_cache:
            file_hash = self._compute_file_hash(file_path)
            cached = self._check_cache(file_hash)
            if cached:
                return cached

        result = self._do_process_document(file_path)

        if self.enable_cache:
            self._save_cache(file_hash, result)

        return result
```

### C4: Deployment Checklist

#### Task C4.1: Production Readiness
- [ ] All tests pass (`pytest`)
- [ ] Code coverage > 80% (`pytest --cov`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Security scan passes (`bandit -r src/`)
- [ ] Documentation complete
- [ ] Examples tested

#### Task C4.2: Package Configuration
```toml
# pyproject.toml
[project]
name = "documind"
version = "0.2.0"
description = "Production-ready document processing for AI applications"
readme = "README.md"
requires-python = ">=3.9"

[project.scripts]
documind-process = "documind.cli:main"
```

#### Task C4.3: CI/CD Setup
**File**: `.github/workflows/test-processor.yml`

```yaml
name: Test Document Processor

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e .[dev]

      - name: Run tests
        run: |
          pytest --cov --cov-report=xml

      - name: Type check
        run: |
          mypy src/documind

      - name: Lint
        run: |
          ruff check src/
```

---

## Milestones & Timeline

### Week 1: Foundation (Phase S, P, A)
- **Days 1-2**: Complete Specification phase
  - Define all data structures
  - Write acceptance criteria
  - Create interface specifications

- **Days 3-4**: Complete Pseudocode phase
  - Design all algorithms
  - Document logic flow
  - Identify edge cases

- **Days 5-7**: Complete Architecture phase
  - Design class hierarchy
  - Plan data flow
  - Set up file structure

### Week 2: Implementation (Phase R)
- **Days 8-9**: Sprint 1 - Foundation
  - TDD: Format Detector
  - TDD: Text Extractor
  - TDD: Metadata Enricher extensions

- **Days 10-11**: Sprint 2 - Formatting & Chunking
  - TDD: LLM Formatter
  - TDD: Content Chunker extensions

- **Days 12-13**: Sprint 3 - Integration
  - TDD: DocuMind Uploader
  - TDD: Main Processor

- **Day 14**: Sprint 4 - Validation
  - Integration tests
  - Fix failing tests
  - Code review

### Week 3: Completion (Phase C)
- **Days 15-16**: Quality Assurance
  - End-to-end testing
  - Security audit
  - Performance benchmarking

- **Days 17-18**: Documentation
  - API documentation
  - Usage examples
  - README updates

- **Days 19-20**: Optimization & Polish
  - Performance profiling
  - Optimization implementation
  - Final testing

- **Day 21**: Deployment
  - Package configuration
  - CI/CD setup
  - Production release

---

## Risk Assessment & Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Existing extractors have bugs** | Medium | Medium | Add comprehensive tests to catch issues early |
| **PDF tables extraction inconsistent** | High | Medium | Implement fallback logic, document limitations |
| **MCP DocuMind tool unavailable** | Low | High | Implement mock for testing, document API requirements |
| **Performance below targets** | Medium | Medium | Profile early, optimize hot paths, add caching |
| **Metadata enrichment too slow** | Low | Low | Optimize regex patterns, parallelize operations |
| **Chunking algorithm too complex** | Medium | Low | Start simple, add sophistication incrementally |
| **Test coverage difficult to achieve** | Low | Medium | Write tests first (TDD), prioritize critical paths |
| **Integration issues with existing code** | Medium | High | Test early, maintain backward compatibility |

### Mitigation Details

**Existing Extractors Have Bugs**
- Strategy: Add comprehensive unit tests for all existing extractors
- Action: Create tests in R2.2-R2.4 before integration
- Fallback: Fix bugs in existing code as needed

**PDF Tables Extraction Inconsistent**
- Strategy: Document known limitations, provide best-effort extraction
- Action: Test with variety of PDF formats, handle gracefully
- Fallback: Allow users to skip table extraction if needed

**MCP DocuMind Tool Unavailable**
- Strategy: Implement comprehensive mocks for testing
- Action: Create mock MCP client in `tests/conftest.py`
- Fallback: Document API requirements, allow users to implement custom uploaders

**Performance Below Targets**
- Strategy: Profile early and often
- Action: Add performance benchmarks in C3.1
- Fallback: Increase target times, add optional caching

**Metadata Enrichment Too Slow**
- Strategy: Optimize regex patterns, compile once
- Action: Profile entity extraction specifically
- Fallback: Make some metadata extraction optional

**Chunking Algorithm Too Complex**
- Strategy: Start with simple sentence-based chunking
- Action: Add sophistication (section boundaries, overlap) incrementally
- Fallback: Use existing chunker with minimal modifications

---

## Appendix: GitHub Issue Template

```markdown
# Production-Ready Document Processing System - Implementation

**Epic**: DocuMind Document Processor
**Methodology**: SPARC
**Estimated Effort**: 3 weeks

## Objective
Build a unified document processor that automatically detects file formats, extracts content with metadata, formats for LLM consumption, and integrates with DocuMind database.

## Success Criteria
- [ ] Process PDF, DOCX, CSV, TXT files automatically
- [ ] Extract comprehensive metadata (basic, structure, entities, topics)
- [ ] Format output as clean Markdown
- [ ] Generate 500-1000 word chunks with 10% overlap
- [ ] Upload to DocuMind via MCP with full-text search
- [ ] Test coverage > 80%
- [ ] Process 100-page PDF in < 5 seconds

## Phase S: Specification
- [ ] S1: Define system requirements
- [ ] S2: Specify interfaces and data structures
- [ ] S3: Write acceptance criteria

## Phase P: Pseudocode
- [ ] P1: Main processing algorithm
- [ ] P2-P3: Format detection and extraction routing
- [ ] P4-P5: Metadata enrichment and topic classification
- [ ] P6-P7: LLM formatting and chunking
- [ ] P8: Database upload

## Phase A: Architecture
- [ ] A1: Design system architecture diagram
- [ ] A2: Define class hierarchy
- [ ] A3: Document data flow
- [ ] A4: Plan error handling strategy
- [ ] A5: Organize file structure
- [ ] A6: Define dependencies

## Phase R: Refinement (TDD)

### Sprint 1: Foundation
- [ ] R2.1: Format Detector (tests + implementation)
- [ ] R2.5: Text Extractor (tests + implementation)
- [ ] R2.6: Metadata Enricher extensions (tests + implementation)

### Sprint 2: Formatting & Chunking
- [ ] R2.7: LLM Formatter (tests + implementation)
- [ ] R2.8: Content Chunker extensions (tests + implementation)

### Sprint 3: Integration
- [ ] R2.9: DocuMind Uploader (tests + implementation)
- [ ] R2.10: Main Processor (tests + implementation)

### Sprint 4: Validation
- [ ] R2.2: Verify PDF extractor tests pass
- [ ] R2.3: Verify DOCX extractor tests pass
- [ ] R2.4: Verify spreadsheet extractor tests pass
- [ ] R2.10: Run integration tests
- [ ] Fix all failing tests

## Phase C: Completion

### Quality Assurance
- [ ] C1.1: End-to-end testing
- [ ] C1.2: Error handling validation
- [ ] C1.3: Security audit

### Documentation
- [ ] C2.1: API documentation
- [ ] C2.2: Basic usage example
- [ ] C2.3: Batch processing example
- [ ] C2.4: README update

### Optimization
- [ ] C3.1: Profile performance
- [ ] C3.2: Optimize bottlenecks
- [ ] C3.3: Implement caching

### Deployment
- [ ] C4.1: Production readiness checklist
- [ ] C4.2: Package configuration
- [ ] C4.3: CI/CD setup

## Timeline
- **Week 1**: Specification, Pseudocode, Architecture
- **Week 2**: TDD Implementation (4 sprints)
- **Week 3**: Completion (QA, docs, optimization, deployment)

## Dependencies
- Existing extractors: `pdf_extractor.py`, `docx_extractor.py`, `spreadsheet_extractor.py`, `metadata_extractor.py`
- MCP DocuMind tool: `upload_document`
- Sample documents: `docs/workshops/S7-sample-docs/`

## Related Issues
- #XXX: S7 Workshop Preparation
- #XXX: Document Processing Pipeline

---

**Note**: This issue follows the SPARC methodology for systematic development with Test-Driven Development (TDD) approach.
```

---

## Summary

This SPARC implementation plan provides a comprehensive, step-by-step approach to building a production-ready document processing system. The plan:

1. **Specification (S)**: Defines exact requirements, interfaces, and acceptance criteria
2. **Pseudocode (P)**: Designs algorithms before implementation
3. **Architecture (A)**: Plans system structure and data flow
4. **Refinement (R)**: Uses TDD to implement with tests first
5. **Completion (C)**: Ensures quality, documentation, and deployment readiness

The plan leverages existing extractors while adding the missing pieces (unified processor, LLM formatting, chunking, database integration) to create a complete, production-ready system.

**Key Strengths:**
- Systematic methodology reduces risk
- TDD ensures code quality
- Modular design enables maintenance
- Comprehensive testing provides confidence
- Clear milestones track progress
- Risk mitigation addresses potential issues

**Next Steps:**
1. Review and approve this plan
2. Create GitHub issue using template
3. Begin Phase S (Specification)
4. Execute plan following TDD workflow
