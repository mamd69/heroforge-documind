# Embedder Agent Documentation

## Overview

The Embedder Agent generates vector embeddings for text chunks using OpenAI's `text-embedding-3-small` model. It handles batching, rate limiting, and provides robust error handling with exponential backoff retry logic.

## Features

- **1536-dimensional embeddings** using OpenAI's latest embedding model
- **Batch processing** for efficient API usage (up to 100 chunks per batch)
- **Exponential backoff** retry logic for rate limit handling
- **Robust error handling** with detailed error messages
- **Performance optimized** (<1s per embedding when batched)
- **Standalone execution** via CLI or programmatic usage

## Installation

### Requirements

```bash
pip install openai
```

### Environment Setup

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

## Usage

### Command Line Interface

#### From stdin
```bash
echo '{"chunks": [{"chunk_id": "1", "text": "Sample text"}]}' | python src/agents/pipeline/embedder.py
```

#### From file
```bash
python src/agents/pipeline/embedder.py --input chunks.json --output embeddings.json
```

#### With custom batch size
```bash
python src/agents/pipeline/embedder.py --input chunks.json --batch-size 50
```

#### Verbose logging
```bash
python src/agents/pipeline/embedder.py --input chunks.json --verbose
```

### Programmatic Usage

```python
from agents.pipeline.embedder import EmbedderAgent, EmbeddingConfig

# Basic usage
agent = EmbedderAgent()
chunks = [
    {'chunk_id': 'chunk-1', 'text': 'First chunk'},
    {'chunk_id': 'chunk-2', 'text': 'Second chunk'}
]
result = agent.process(chunks)

if result['success']:
    embeddings = result['embeddings']
    print(f"Generated {len(embeddings)} embeddings")
else:
    print(f"Error: {result['error']}")
```

### Custom Configuration

```python
from agents.pipeline.embedder import EmbedderAgent, EmbeddingConfig

config = EmbeddingConfig(
    model='text-embedding-3-small',
    dimensions=1536,
    batch_size=50,  # Process 50 chunks per batch
    max_retries=5,
    initial_retry_delay=1.0,
    max_retry_delay=60.0
)

agent = EmbedderAgent(config)
result = agent.process(chunks)
```

## Input Format

### Chunk Object

```json
{
  "chunk_id": "unique-identifier",
  "text": "The text content to embed",
  "metadata": {
    "source": "optional metadata",
    "tokens": 42
  }
}
```

### Alternative Keys

The agent accepts alternative field names:
- `chunk_id` or `id`
- `text` or `content`

### Full Input Format

```json
{
  "chunks": [
    {
      "chunk_id": "doc1-chunk-0",
      "text": "First chunk of text...",
      "metadata": {"tokens": 128}
    },
    {
      "chunk_id": "doc1-chunk-1",
      "text": "Second chunk of text...",
      "metadata": {"tokens": 142}
    }
  ]
}
```

## Output Format

### Success Response

```json
{
  "success": true,
  "embeddings": [
    {
      "chunk_id": "doc1-chunk-0",
      "vector": [0.0123, -0.0456, 0.0789, ...],
      "model": "text-embedding-3-small",
      "dimensions": 1536
    }
  ],
  "metadata": {
    "total_chunks": 2,
    "total_batches": 1,
    "model": "text-embedding-3-small",
    "dimensions": 1536
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Detailed error message",
  "embeddings": []
}
```

## Performance

### Timing Benchmarks

- **Single embedding**: ~0.2-0.5s
- **Batch of 10**: ~0.8-1.2s (~0.1s per embedding)
- **Batch of 100**: ~3-5s (~0.04s per embedding)

### Optimization Tips

1. **Batch size**: Use larger batches (50-100) for better throughput
2. **Concurrent processing**: Process multiple documents in parallel
3. **Caching**: Cache embeddings for frequently accessed chunks
4. **Compression**: Consider dimensionality reduction for storage

## Rate Limiting

### Default Retry Strategy

- **Initial delay**: 1 second
- **Backoff multiplier**: 2x (exponential)
- **Max delay**: 60 seconds
- **Max retries**: 5 attempts

### Retry Behavior

```
Attempt 1: Immediate
Attempt 2: 1s delay
Attempt 3: 2s delay
Attempt 4: 4s delay
Attempt 5: 8s delay
Attempt 6: 16s delay (or max 60s)
```

### Rate Limit Guidelines

OpenAI rate limits vary by tier:
- **Free tier**: 3 RPM, 150,000 TPM
- **Tier 1**: 500 RPM, 1,000,000 TPM
- **Tier 2**: 5,000 RPM, 5,000,000 TPM

Adjust batch sizes and delays accordingly.

## Error Handling

### Common Errors

#### Missing API Key
```json
{
  "success": false,
  "error": "OPENAI_API_KEY environment variable is required",
  "embeddings": []
}
```

#### Empty Text
```json
{
  "success": false,
  "error": "Chunk chunk-1 has no text content",
  "embeddings": []
}
```

#### Rate Limit Exceeded
```json
{
  "success": false,
  "error": "Max retries (5) exceeded for rate limit",
  "embeddings": []
}
```

#### API Error
```json
{
  "success": false,
  "error": "API error: [detailed message]",
  "embeddings": []
}
```

## Integration with Pipeline

### With Chunker

```bash
# Chain chunker → embedder
python src/agents/pipeline/chunker.py --input doc.txt | \
  python src/agents/pipeline/embedder.py --output embeddings.json
```

### With Storage

```python
from agents.pipeline.embedder import EmbedderAgent
from agents.pipeline.storage import store_embeddings

agent = EmbedderAgent()
result = agent.process(chunks)

if result['success']:
    store_embeddings(result['embeddings'], database='documind')
```

## Testing

### Run Tests

```bash
# All tests
python -m pytest tests/test_embedder.py -v

# Specific test
python -m pytest tests/test_embedder.py::TestEmbedderAgent::test_process_chunks_success -v

# With coverage
python -m pytest tests/test_embedder.py --cov=src/agents/pipeline/embedder
```

### Example Tests

```python
import unittest
from agents.pipeline.embedder import EmbedderAgent

class TestEmbedder(unittest.TestCase):
    def test_basic_embedding(self):
        agent = EmbedderAgent()
        chunks = [{'chunk_id': '1', 'text': 'Test'}]
        result = agent.process(chunks)

        self.assertTrue(result['success'])
        self.assertEqual(len(result['embeddings']), 1)
        self.assertEqual(result['embeddings'][0]['dimensions'], 1536)
```

## Advanced Usage

### Custom Retry Logic

```python
from agents.pipeline.embedder import EmbedderAgent, EmbeddingConfig

config = EmbeddingConfig(
    max_retries=3,
    initial_retry_delay=0.5,
    max_retry_delay=30.0
)

agent = EmbedderAgent(config)
```

### Progress Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
agent = EmbedderAgent()

# Logs will show batch progress:
# Processing batch 1/5
# Processed batch of 100 chunks in 3.2s (0.032s per embedding)
```

### Memory-Efficient Processing

```python
def process_large_dataset(chunks, batch_size=1000):
    """Process large datasets in memory-efficient batches"""
    agent = EmbedderAgent()
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = agent.process(batch)

        if result['success']:
            all_embeddings.extend(result['embeddings'])
        else:
            print(f"Error in batch {i}: {result['error']}")

    return all_embeddings
```

## API Reference

### Classes

#### `EmbedderAgent`

Main agent class for generating embeddings.

**Methods:**
- `__init__(config: Optional[EmbeddingConfig] = None)`: Initialize agent
- `process(chunks: List[Dict[str, Any]]) -> Dict[str, Any]`: Process chunks and generate embeddings

#### `EmbeddingConfig`

Configuration dataclass for embedding generation.

**Fields:**
- `model: str` - Model name (default: "text-embedding-3-small")
- `dimensions: int` - Vector dimensions (default: 1536)
- `batch_size: int` - Chunks per batch (default: 100)
- `max_retries: int` - Maximum retry attempts (default: 5)
- `initial_retry_delay: float` - Initial retry delay in seconds (default: 1.0)
- `max_retry_delay: float` - Maximum retry delay in seconds (default: 60.0)
- `timeout: int` - API timeout in seconds (default: 30)

#### `Chunk`

Dataclass representing a text chunk.

**Fields:**
- `chunk_id: str` - Unique identifier
- `text: str` - Text content
- `metadata: Optional[Dict[str, Any]]` - Optional metadata

#### `Embedding`

Dataclass representing an embedding result.

**Fields:**
- `chunk_id: str` - Chunk identifier
- `vector: List[float]` - Embedding vector
- `model: str` - Model used
- `dimensions: int` - Vector dimensions

## Troubleshooting

### Issue: API Key Not Found

**Solution**: Set the environment variable
```bash
export OPENAI_API_KEY='your-key-here'
```

### Issue: Rate Limits

**Solution**: Reduce batch size or increase retry delays
```python
config = EmbeddingConfig(batch_size=25, initial_retry_delay=2.0)
```

### Issue: Out of Memory

**Solution**: Process in smaller batches
```python
config = EmbeddingConfig(batch_size=10)
```

### Issue: Slow Performance

**Solution**: Increase batch size
```python
config = EmbeddingConfig(batch_size=200)
```

## Best Practices

1. **Use batching**: Always process multiple chunks in batches
2. **Handle errors gracefully**: Check `success` field before using embeddings
3. **Monitor performance**: Use verbose logging to track timing
4. **Cache results**: Store embeddings to avoid re-processing
5. **Validate input**: Ensure all chunks have non-empty text
6. **Set appropriate batch sizes**: Balance between throughput and memory
7. **Configure retries**: Adjust retry settings based on your tier

## Security Considerations

1. **Never commit API keys**: Use environment variables
2. **Rotate keys regularly**: Update OPENAI_API_KEY periodically
3. **Monitor usage**: Track API costs and usage
4. **Validate input**: Sanitize text chunks before processing
5. **Secure storage**: Encrypt embeddings at rest

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [heroforge-documind/issues](https://github.com/your-org/heroforge-documind/issues)
- Documentation: [docs/agents/](https://github.com/your-org/heroforge-documind/tree/main/docs/agents)
