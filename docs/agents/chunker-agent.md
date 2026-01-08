# Chunker Agent

The Chunker Agent is a key component of the DocuMind document processing pipeline. It splits extracted text into semantically meaningful chunks with configurable overlap for optimal context preservation.

## Features

- **Intelligent Chunking**: Splits text at sentence boundaries for semantic coherence
- **Configurable Parameters**: Adjustable chunk size (default 500 words) and overlap (default 50 words)
- **Overlap Support**: Maintains context across chunk boundaries with configurable overlap
- **Edge Case Handling**: Gracefully handles short documents, long sentences, and special characters
- **Performance Optimized**: Processes documents in < 1 second
- **JSON I/O**: Compatible with pipeline data format
- **Comprehensive Metadata**: Each chunk includes position, word count, and relationship data

## Installation

No additional dependencies required beyond Python 3.8+.

## Usage

### Command Line

#### From stdin:
```bash
echo '{"document_id": "doc123", "content": "Your text here..."}' | python src/agents/pipeline/chunker.py
```

#### From file:
```bash
python src/agents/pipeline/chunker.py --input extracted.json --output chunks.json
```

#### With custom parameters:
```bash
python src/agents/pipeline/chunker.py \
  --input document.json \
  --output chunks.json \
  --chunk-size 400 \
  --overlap 40 \
  --pretty \
  --stats
```

### Python API

```python
from agents.pipeline.chunker import TextChunker

# Initialize chunker
chunker = TextChunker(target_chunk_size=500, overlap_size=50)

# Process text
chunks = chunker.chunk_text(
    text="Your document text here...",
    document_id="doc123"
)

# Process from document dict
document = {
    "document_id": "doc123",
    "content": "Your text here...",
    "metadata": {}
}
chunks = chunker.process_document(document)

# Each chunk is a Chunk object with:
# - chunk_id: Unique identifier
# - content: Chunk text
# - word_count: Number of words
# - start_position: Character position in original
# - end_position: Character end position
# - document_id: Source document ID
# - chunk_index: Zero-based chunk number
# - total_chunks: Total number of chunks
# - has_overlap: Whether chunk includes overlap
```

## Input Format

### Single Document:
```json
{
  "document_id": "unique_id",
  "content": "The extracted text content...",
  "metadata": {
    "source": "optional metadata"
  }
}
```

### Multiple Documents:
```json
[
  {
    "document_id": "doc1",
    "content": "First document..."
  },
  {
    "document_id": "doc2",
    "content": "Second document..."
  }
]
```

## Output Format

```json
[
  {
    "chunk_id": "doc123_chunk_0000",
    "content": "First chunk text...",
    "word_count": 487,
    "start_position": 0,
    "end_position": 2847,
    "document_id": "doc123",
    "chunk_index": 0,
    "total_chunks": 3,
    "has_overlap": false
  },
  {
    "chunk_id": "doc123_chunk_0001",
    "content": "Second chunk with overlap...",
    "word_count": 502,
    "start_position": 2650,
    "end_position": 5789,
    "document_id": "doc123",
    "chunk_index": 1,
    "total_chunks": 3,
    "has_overlap": true
  }
]
```

## Configuration

### Parameters

| Parameter | Default | Min | Description |
|-----------|---------|-----|-------------|
| `target_chunk_size` | 500 | 100 | Target words per chunk |
| `overlap_size` | 50 | 0 | Words to overlap between chunks |

### Recommendations

- **General purpose**: 500 words, 50 overlap
- **Long-form content**: 600-800 words, 60-80 overlap
- **Short-form content**: 300-400 words, 30-40 overlap
- **Semantic search**: 400-500 words, 50-75 overlap
- **Q&A systems**: 300-400 words, 40-50 overlap

## Algorithm

1. **Sentence Detection**: Split text using regex pattern matching for `.!?` with proper handling of quotes and brackets
2. **Chunk Assembly**: Group sentences until target word count is reached
3. **Overlap Calculation**: Take last N words from previous chunk as overlap for next chunk
4. **Metadata Generation**: Calculate positions, word counts, and relationships
5. **Edge Case Handling**: Special logic for very short documents and long sentences

## Performance

- **Throughput**: ~500,000 words/second
- **Latency**: < 1 second per document (2000-3000 words)
- **Memory**: O(n) where n is document size
- **Chunk Size Accuracy**: 400-600 words (target 500)

## Edge Cases

### Handled Automatically:
- Empty documents → Returns empty array
- Single word/sentence → Returns single chunk
- No sentence boundaries → Splits on word boundaries
- Very long sentences → Creates chunks as needed
- Special characters/Unicode → Preserved correctly
- Mixed line endings → Normalized

## Pipeline Integration

### Input from Extractor:
```bash
python src/agents/pipeline/extractor.py document.pdf | \
  python src/agents/pipeline/chunker.py
```

### Output to Embedder:
```bash
python src/agents/pipeline/chunker.py --input extracted.json | \
  python src/agents/pipeline/embedder.py
```

### Full Pipeline:
```bash
python src/agents/pipeline/extractor.py document.pdf | \
  python src/agents/pipeline/chunker.py --stats | \
  python src/agents/pipeline/embedder.py | \
  python src/agents/pipeline/indexer.py
```

## Testing

```bash
# Run all tests
python -m pytest tests/agents/pipeline/test_chunker.py -v

# Run specific test category
python -m pytest tests/agents/pipeline/test_chunker.py::TestTextChunker -v

# Run with coverage
python -m pytest tests/agents/pipeline/test_chunker.py --cov=src/agents/pipeline/chunker
```

## Examples

See `examples/chunker_demo.py` for comprehensive demonstrations:

```bash
python examples/chunker_demo.py
```

## Troubleshooting

### Issue: Chunks too large/small
**Solution**: Adjust `target_chunk_size` parameter

### Issue: Lost context at boundaries
**Solution**: Increase `overlap_size` parameter

### Issue: Chunks split mid-sentence
**Solution**: Check input text has proper punctuation

### Issue: Performance slow
**Solution**: Reduce chunk size or process in batches

## API Reference

### TextChunker Class

#### `__init__(target_chunk_size=500, overlap_size=50)`
Initialize chunker with parameters.

**Raises**: `ValueError` if parameters are invalid

#### `chunk_text(text: str, document_id: str) -> List[Chunk]`
Split text into chunks.

**Parameters**:
- `text`: Input text to chunk
- `document_id`: Unique identifier for source document

**Returns**: List of Chunk objects

#### `process_document(document: Dict) -> List[Dict]`
Process document from pipeline format.

**Parameters**:
- `document`: Dict with 'document_id' and 'content' keys

**Returns**: List of chunk dictionaries

#### `count_words(text: str) -> int`
Count words in text using regex pattern.

#### `split_into_sentences(text: str) -> List[str]`
Split text into sentences at punctuation boundaries.

### Chunk Dataclass

**Fields**:
- `chunk_id: str` - Unique identifier (format: `{doc_id}_chunk_{index:04d}`)
- `content: str` - Chunk text content
- `word_count: int` - Number of words in chunk
- `start_position: int` - Character position where chunk starts
- `end_position: int` - Character position where chunk ends
- `document_id: str` - Source document identifier
- `chunk_index: int` - Zero-based index of chunk
- `total_chunks: int` - Total number of chunks in document
- `has_overlap: bool` - Whether chunk includes overlap from previous

## Contributing

When modifying the chunker:
1. Ensure all tests pass
2. Add tests for new features
3. Update this documentation
4. Verify performance requirements still met
5. Test with realistic documents

## License

Part of the DocuMind project.
