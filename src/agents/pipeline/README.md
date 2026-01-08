# DocuMind Pipeline Orchestrator

Coordinates the 4-stage document processing pipeline for DocuMind.

## Pipeline Architecture

```
Document Input
      ↓
┌─────────────┐
│  EXTRACT    │  Stage 1: Extract text from PDF/DOCX/XLSX/TXT/MD
└─────────────┘
      ↓
┌─────────────┐
│   CHUNK     │  Stage 2: Split text into semantic chunks (~500 words)
└─────────────┘
      ↓
┌─────────────┐
│   EMBED     │  Stage 3: Generate 1536-dim embeddings (OpenAI)
└─────────────┘
      ↓
┌─────────────┐
│   WRITE     │  Stage 4: Store in Supabase with pgvector
└─────────────┘
      ↓
  Database
```

## Features

- ✅ **Parallel Processing**: Process 10+ documents concurrently
- ✅ **Error Recovery**: Continue-on-error mode for batch processing
- ✅ **Progress Indicators**: Real-time status updates
- ✅ **Comprehensive Metrics**: Stage-by-stage performance tracking
- ✅ **JSON Export**: Machine-readable reports
- ✅ **CLI Support**: Easy command-line usage

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (if using real agents)
export OPENAI_API_KEY=sk-xxx
export SUPABASE_URL=https://xxx.supabase.co
export SUPABASE_ANON_KEY=xxx
```

## Usage

### Process Single Files

```bash
python orchestrate.py document.pdf
python orchestrate.py file1.md file2.txt file3.docx
```

### Process Directory (Recursive)

```bash
python orchestrate.py -d demo-docs/
python orchestrate.py --directory /path/to/documents/
```

### Advanced Options

```bash
# Increase parallelism
python orchestrate.py -d demo-docs/ --max-parallel 20

# Stop on first error
python orchestrate.py docs/*.pdf --no-continue-on-error

# Quiet mode
python orchestrate.py -d docs/ --quiet

# Save JSON report
python orchestrate.py -d docs/ --json-output report.json
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `files` | File paths to process (supports globs) | - |
| `-d, --directory` | Process all files in directory | - |
| `--max-parallel` | Max concurrent documents | 10 |
| `--no-continue-on-error` | Stop on first error | False |
| `--quiet` | Suppress progress output | False |
| `--json-output` | Save JSON report to file | - |

## Output Report

The orchestrator generates a comprehensive report with:

### Summary Metrics
- Total documents processed
- Success/failure counts
- Success rate percentage

### Output Metrics
- Total chunks created
- Total embeddings generated
- Average chunks per document

### Performance Metrics
- Total processing time
- Average time per document
- Throughput (docs/second)
- Stage-by-stage timing breakdown

### Error Tracking
- Errors by pipeline stage
- Failed document details (up to 10 shown)

### Example Report

```
╔══════════════════════════════════════════════════════════════╗
║           DOCUMIND PIPELINE PROCESSING REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📊 SUMMARY
────────────────────────────────────────────────────────────
Total Documents:        15
✅ Successful:          15 (100.0%)
❌ Failed:              0 (0.0%)

📦 OUTPUT
────────────────────────────────────────────────────────────
Total Chunks Created:   45
Total Embeddings:       45
Avg Chunks/Document:    3.0

⏱️  PERFORMANCE
────────────────────────────────────────────────────────────
Total Time:             12.34s
Avg Time/Document:      0.82s
Throughput:             1.2 docs/second

⚙️  STAGE BREAKDOWN (Average Times)
────────────────────────────────────────────────────────────
Extract:                0.234s
Chunk:                  0.156s
Embed:                  0.389s
Write:                  0.041s

❌ ERRORS BY STAGE
────────────────────────────────────────────────────────────
Extract failures:       0
Chunk failures:         0
Embed failures:         0
Write failures:         0
```

## JSON Report Format

When using `--json-output`, the report includes:

```json
{
  "metrics": {
    "total_documents": 15,
    "successful": 15,
    "failed": 0,
    "total_chunks": 45,
    "total_embeddings": 45,
    "total_time": 12.34,
    "stage_times": {
      "extract": 3.51,
      "chunk": 2.34,
      "embed": 5.84,
      "write": 0.62
    },
    "errors_by_stage": {
      "extract": 0,
      "chunk": 0,
      "embed": 0,
      "write": 0
    }
  },
  "results": [
    {
      "file_path": "demo-docs/sample1.md",
      "status": "success",
      "stage_completed": "write",
      "chunks_created": 3,
      "embeddings_generated": 3,
      "processing_time": 0.82,
      "error_message": null,
      "metadata": {
        "document_id": "abc123",
        "file_name": "sample1.md",
        "file_type": "md"
      }
    }
  ]
}
```

## Architecture Details

### Orchestrator Class

The `PipelineOrchestrator` class coordinates all 4 pipeline stages:

```python
orchestrator = PipelineOrchestrator(
    max_parallel=10,           # Concurrent document limit
    continue_on_error=True,    # Keep going on failures
    verbose=True               # Print progress
)

# Process single document
result = await orchestrator.process_document("file.pdf")

# Process batch
results = await orchestrator.process_batch(["f1.pdf", "f2.md"])

# Process directory
results = await orchestrator.process_directory("docs/")

# Generate report
report = orchestrator.generate_report(results)
```

### Mock Agents

The orchestrator includes mock implementations of all 4 agents for testing without dependencies:

- **MockExtractorAgent**: Reads file content with encoding detection
- **MockChunkerAgent**: Simple word-based chunking
- **MockEmbedderAgent**: Generates deterministic random embeddings
- **MockWriterAgent**: Returns mock document IDs

These are automatically used if the real agents are not implemented.

### Real Agent Integration

To use real agents, implement these classes in separate files:

```python
# extract.py
class ExtractorAgent:
    async def extract_text(self, file_path: str) -> Dict[str, Any]:
        # Implementation using PyPDF2, python-docx, etc.
        pass

# chunk.py
class ChunkerAgent:
    async def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        # Implementation using NLTK, spaCy, etc.
        pass

# embed.py
class EmbedderAgent:
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Implementation using OpenAI API
        pass

# write.py
class WriterAgent:
    async def write_chunks(self, file_path: str, chunks: List[Dict], metadata: Dict) -> str:
        # Implementation using Supabase
        pass
```

The orchestrator will automatically use real agents when available, falling back to mocks otherwise.

## Error Handling

The orchestrator implements robust error handling:

### Continue-on-Error Mode (Default)

```bash
python orchestrate.py -d docs/
# Processes all files, reports failures in summary
```

Behavior:
- One document fails → others continue processing
- Errors tracked per stage
- Failed documents listed in report
- Exit code 1 if any failures

### Stop-on-Error Mode

```bash
python orchestrate.py -d docs/ --no-continue-on-error
# Stops immediately on first error
```

Behavior:
- First error → pipeline stops
- Partial results discarded
- Exception raised with stack trace
- Exit code 1

### Error Tracking

Errors are categorized by pipeline stage:

```python
errors_by_stage = {
    "extract": 2,   # File read/parse errors
    "chunk": 0,     # Text splitting errors
    "embed": 1,     # API/network errors
    "write": 0      # Database errors
}
```

## Performance Optimization

### Parallelism

Control concurrent document processing:

```bash
# Low parallelism (conservative)
python orchestrate.py -d docs/ --max-parallel 5

# High parallelism (aggressive)
python orchestrate.py -d docs/ --max-parallel 20
```

**Guidelines**:
- CPU-bound work: parallelism = CPU cores
- I/O-bound work: parallelism = 2-4x CPU cores
- API rate limits: adjust to avoid throttling

### Batch Processing

The orchestrator uses `asyncio.gather()` for efficient parallel execution:

```python
# All documents processed concurrently (up to max_parallel)
tasks = [process_document(path) for path in file_paths]
results = await asyncio.gather(*tasks)
```

### Semaphore Limiting

Prevents resource exhaustion with semaphore:

```python
semaphore = asyncio.Semaphore(max_parallel)

async with semaphore:
    # Only max_parallel documents in flight
    await process_document(file_path)
```

## Success Criteria (from GOAP Plan)

- ✅ Coordinates all 4 agents sequentially
- ✅ Processes 10+ concurrent documents
- ✅ Comprehensive metrics reporting
- ✅ Continue-on-error support
- ✅ Progress indicators
- ✅ Stage-level timing breakdown
- ✅ JSON export capability
- ✅ CLI with directory/file list support

## Examples

### Example 1: Basic Usage

```bash
python orchestrate.py demo-docs/sample1.md demo-docs/sample2.txt
```

Output:
```
🚀 Starting pipeline for 2 documents
   Max parallel: 10
   Continue on error: True

📄 Processing: sample1.md
📄 Processing: sample2.txt
  ✅ Success: 3 chunks, 3 embeddings
  ✅ Success: 2 chunks, 2 embeddings

[Report...]
```

### Example 2: Directory Processing

```bash
python orchestrate.py -d demo-docs/
```

Processes all `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx` files recursively.

### Example 3: High-Throughput Mode

```bash
python orchestrate.py -d large-corpus/ --max-parallel 50 --quiet --json-output results.json
```

- 50 concurrent documents
- No console output
- Results saved to JSON

### Example 4: Error Testing

```bash
# Test with missing file
python orchestrate.py good.pdf missing.pdf another.md

# Report shows:
# ✅ Successful: 2 (66.7%)
# ❌ Failed: 1 (33.3%)
# Failed Documents: missing.pdf - File not found
```

## Testing

Run the orchestrator with mock agents (no dependencies required):

```bash
# Create test documents
mkdir -p demo-docs
echo "Test document" > demo-docs/test.txt

# Run orchestrator
python orchestrate.py demo-docs/test.txt
```

The mock agents provide realistic timing simulation for testing the orchestration logic.

## Future Enhancements

Planned features:
- [ ] Resume interrupted batches from checkpoint
- [ ] Distributed processing across multiple workers
- [ ] Real-time WebSocket progress updates
- [ ] Prometheus metrics export
- [ ] Document priority queues
- [ ] Retry queue for failed documents
- [ ] Cost tracking (API usage)
- [ ] Memory usage monitoring

## Troubleshooting

### "File not found" errors

Ensure file paths are correct:
```bash
ls -la demo-docs/  # Verify files exist
python orchestrate.py demo-docs/*.md  # Use absolute paths if needed
```

### Import errors for agents

Mock agents are used automatically:
```
MockExtractorAgent - using mock implementation
```

Real agents require implementation in separate files.

### Rate limiting errors (OpenAI)

Reduce parallelism or implement backoff:
```bash
python orchestrate.py -d docs/ --max-parallel 3
```

### Memory errors with large documents

Process in smaller batches or increase system memory.

## Contributing

When implementing real agents, follow this interface:

1. **Extractor**: `async def extract_text(file_path: str) -> Dict`
2. **Chunker**: `async def chunk_text(text: str, metadata: Dict) -> List[Dict]`
3. **Embedder**: `async def generate_embeddings(texts: List[str]) -> List[List[float]]`
4. **Writer**: `async def write_chunks(file_path: str, chunks: List[Dict], metadata: Dict) -> str`

All methods must be async and return the specified types.

## License

Part of the DocuMind project - Session 5 implementation.
