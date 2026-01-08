#!/usr/bin/env python3
"""
Example usage of the Embedder Agent

Demonstrates:
- Basic embedding generation
- Batch processing
- Error handling
- Integration with chunker output
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.pipeline.embedder import EmbedderAgent, EmbeddingConfig


def example_basic_usage():
    """Basic embedding generation example"""
    print("\n=== Basic Usage Example ===\n")

    # Sample chunks
    chunks = [
        {
            'chunk_id': 'chunk-1',
            'text': 'Artificial intelligence is transforming healthcare through early disease detection.'
        },
        {
            'chunk_id': 'chunk-2',
            'text': 'Machine learning models can analyze medical images with high accuracy.'
        },
        {
            'chunk_id': 'chunk-3',
            'text': 'Natural language processing helps extract insights from medical records.'
        }
    ]

    # Create agent
    agent = EmbedderAgent()

    # Process chunks
    result = agent.process(chunks)

    if result['success']:
        print(f"✓ Successfully generated {len(result['embeddings'])} embeddings")
        print(f"  Model: {result['metadata']['model']}")
        print(f"  Dimensions: {result['metadata']['dimensions']}")
        print(f"\nFirst embedding preview:")
        print(f"  Chunk ID: {result['embeddings'][0]['chunk_id']}")
        print(f"  Vector (first 5): {result['embeddings'][0]['vector'][:5]}")
    else:
        print(f"✗ Error: {result['error']}")


def example_large_batch():
    """Large batch processing example"""
    print("\n=== Large Batch Processing Example ===\n")

    # Generate 150 chunks
    chunks = [
        {
            'chunk_id': f'chunk-{i}',
            'text': f'This is sample text chunk number {i} for testing batch processing.'
        }
        for i in range(150)
    ]

    # Create agent with custom batch size
    config = EmbeddingConfig(batch_size=50)
    agent = EmbedderAgent(config)

    print(f"Processing {len(chunks)} chunks with batch_size={config.batch_size}")

    result = agent.process(chunks)

    if result['success']:
        print(f"✓ Successfully processed in {result['metadata']['total_batches']} batches")
        print(f"  Total embeddings: {len(result['embeddings'])}")
        print(f"  Average vector magnitude: {sum(sum(e['vector']) for e in result['embeddings']) / len(result['embeddings']):.4f}")
    else:
        print(f"✗ Error: {result['error']}")


def example_error_handling():
    """Error handling example"""
    print("\n=== Error Handling Example ===\n")

    # Invalid chunks (no text)
    chunks = [
        {'chunk_id': 'chunk-1', 'text': ''},
        {'chunk_id': 'chunk-2', 'text': 'Valid text'}
    ]

    agent = EmbedderAgent()
    result = agent.process(chunks)

    if not result['success']:
        print(f"✓ Error properly handled: {result['error']}")
    else:
        print("✗ Should have raised an error")


def example_chunker_integration():
    """Example integrating with chunker output"""
    print("\n=== Chunker Integration Example ===\n")

    # Simulated chunker output format
    chunker_output = {
        'success': True,
        'chunks': [
            {
                'chunk_id': 'doc1-chunk-0',
                'text': 'Introduction to machine learning and artificial intelligence.',
                'metadata': {
                    'start_char': 0,
                    'end_char': 63,
                    'tokens': 12
                }
            },
            {
                'chunk_id': 'doc1-chunk-1',
                'text': 'Deep learning uses neural networks with multiple layers.',
                'metadata': {
                    'start_char': 64,
                    'end_char': 120,
                    'tokens': 10
                }
            }
        ],
        'metadata': {
            'total_chunks': 2,
            'chunk_size': 512,
            'overlap': 50
        }
    }

    # Process chunks from chunker
    agent = EmbedderAgent()
    result = agent.process(chunker_output['chunks'])

    if result['success']:
        print(f"✓ Processed {len(result['embeddings'])} chunks from chunker output")
        print("\nEmbedding details:")
        for emb in result['embeddings']:
            print(f"  {emb['chunk_id']}: {emb['dimensions']}-dim vector")
    else:
        print(f"✗ Error: {result['error']}")


def example_save_to_file():
    """Example saving embeddings to file"""
    print("\n=== Save to File Example ===\n")

    chunks = [
        {'chunk_id': f'chunk-{i}', 'text': f'Sample text {i}'}
        for i in range(5)
    ]

    agent = EmbedderAgent()
    result = agent.process(chunks)

    if result['success']:
        output_file = '/tmp/embeddings.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Saved embeddings to {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    else:
        print(f"✗ Error: {result['error']}")


def main():
    """Run all examples"""
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠ OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning examples with mock data (will fail without real API key)...")

    try:
        example_basic_usage()
        example_large_batch()
        example_error_handling()
        example_chunker_integration()
        example_save_to_file()

        print("\n=== All Examples Complete ===\n")

    except Exception as e:
        print(f"\n✗ Example failed: {str(e)}")
        print("  Make sure OPENAI_API_KEY is set and valid")


if __name__ == '__main__':
    main()
