#!/usr/bin/env python3
"""
Embedder Agent for DocuMind Pipeline

Generates embeddings for text chunks using OpenAI's text-embedding-3-small model.
Handles batching, rate limiting, and exponential backoff for robust operation.

Usage:
    # From stdin
    echo '{"chunks": [...]}' | python embedder.py

    # From file
    python embedder.py --input chunks.json

    # As module
    from agents.pipeline.embedder import EmbedderAgent
    agent = EmbedderAgent()
    result = agent.process(chunks)
"""

import sys
import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError:
    print(json.dumps({
        "success": False,
        "error": "OpenAI package not installed. Run: pip install openai",
        "embeddings": []
    }))
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100  # OpenAI allows up to 2048 inputs per request
    max_retries: int = 5
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    timeout: int = 30


@dataclass
class Chunk:
    """Represents a text chunk to be embedded"""
    chunk_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create Chunk from dictionary"""
        return cls(
            chunk_id=data.get('chunk_id', data.get('id', '')),
            text=data.get('text', data.get('content', '')),
            metadata=data.get('metadata', {})
        )


@dataclass
class Embedding:
    """Represents an embedding result"""
    chunk_id: str
    vector: List[float]
    model: str
    dimensions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk_id,
            'vector': self.vector,
            'model': self.model,
            'dimensions': self.dimensions
        }


class EmbedderAgent:
    """
    Agent responsible for generating embeddings from text chunks.

    Features:
    - Batch processing for efficiency
    - Exponential backoff for rate limit handling
    - Robust error handling and retry logic
    - 1536-dimensional embeddings
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the Embedder Agent.

        Args:
            config: Optional configuration object. Uses defaults if not provided.

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        self.config = config or EmbeddingConfig()

        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )

        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized EmbedderAgent with model: {self.config.model}")

    def _batch_chunks(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """
        Split chunks into batches for efficient processing.

        Args:
            chunks: List of chunks to batch

        Returns:
            List of chunk batches
        """
        batches = []
        for i in range(0, len(chunks), self.config.batch_size):
            batches.append(chunks[i:i + self.config.batch_size])
        return batches

    def _embed_batch_with_retry(
        self,
        texts: List[str],
        retry_count: int = 0
    ) -> List[List[float]]:
        """
        Embed a batch of texts with exponential backoff retry logic.

        Args:
            texts: List of text strings to embed
            retry_count: Current retry attempt number

        Returns:
            List of embedding vectors

        Raises:
            Exception: If max retries exceeded or non-retryable error occurs
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model,
                dimensions=self.config.dimensions
            )

            # Extract vectors in the same order as input
            embeddings = [item.embedding for item in response.data]
            return embeddings

        except RateLimitError as e:
            if retry_count >= self.config.max_retries:
                logger.error(f"Max retries ({self.config.max_retries}) exceeded for rate limit")
                raise

            # Calculate exponential backoff delay
            delay = min(
                self.config.initial_retry_delay * (2 ** retry_count),
                self.config.max_retry_delay
            )

            logger.warning(
                f"Rate limit hit. Retrying in {delay:.2f}s "
                f"(attempt {retry_count + 1}/{self.config.max_retries})"
            )
            time.sleep(delay)

            return self._embed_batch_with_retry(texts, retry_count + 1)

        except (APIError, APIConnectionError) as e:
            if retry_count >= self.config.max_retries:
                logger.error(f"Max retries ({self.config.max_retries}) exceeded for API error")
                raise

            delay = min(
                self.config.initial_retry_delay * (2 ** retry_count),
                self.config.max_retry_delay
            )

            logger.warning(
                f"API error: {str(e)}. Retrying in {delay:.2f}s "
                f"(attempt {retry_count + 1}/{self.config.max_retries})"
            )
            time.sleep(delay)

            return self._embed_batch_with_retry(texts, retry_count + 1)

    def _process_batch(self, batch: List[Chunk]) -> List[Embedding]:
        """
        Process a single batch of chunks.

        Args:
            batch: List of chunks to embed

        Returns:
            List of Embedding objects
        """
        start_time = time.time()

        # Extract texts
        texts = [chunk.text for chunk in batch]

        # Get embeddings with retry logic
        vectors = self._embed_batch_with_retry(texts)

        # Create Embedding objects
        embeddings = []
        for chunk, vector in zip(batch, vectors):
            embeddings.append(Embedding(
                chunk_id=chunk.chunk_id,
                vector=vector,
                model=self.config.model,
                dimensions=len(vector)
            ))

        elapsed = time.time() - start_time
        avg_time = elapsed / len(batch)

        logger.info(
            f"Processed batch of {len(batch)} chunks in {elapsed:.2f}s "
            f"({avg_time:.3f}s per embedding)"
        )

        return embeddings

    def process(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process chunks and generate embeddings.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'text' fields

        Returns:
            Dictionary with success status and embeddings array
        """
        if not chunks:
            logger.warning("No chunks provided")
            return {
                'success': True,
                'embeddings': [],
                'metadata': {
                    'total_chunks': 0,
                    'total_batches': 0,
                    'model': self.config.model
                }
            }

        try:
            # Convert to Chunk objects
            chunk_objects = [Chunk.from_dict(c) for c in chunks]

            # Validate chunks
            for chunk in chunk_objects:
                if not chunk.text:
                    raise ValueError(f"Chunk {chunk.chunk_id} has no text content")

            # Batch chunks
            batches = self._batch_chunks(chunk_objects)
            logger.info(
                f"Processing {len(chunk_objects)} chunks in {len(batches)} batches "
                f"(batch_size={self.config.batch_size})"
            )

            # Process batches
            all_embeddings = []
            for i, batch in enumerate(batches, 1):
                logger.info(f"Processing batch {i}/{len(batches)}")
                batch_embeddings = self._process_batch(batch)
                all_embeddings.extend(batch_embeddings)

            # Convert to dictionaries
            embeddings_dicts = [emb.to_dict() for emb in all_embeddings]

            return {
                'success': True,
                'embeddings': embeddings_dicts,
                'metadata': {
                    'total_chunks': len(chunk_objects),
                    'total_batches': len(batches),
                    'model': self.config.model,
                    'dimensions': self.config.dimensions
                }
            }

        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'embeddings': []
            }


def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(
        description='Generate embeddings for text chunks using OpenAI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From stdin
  echo '{"chunks": [...]}' | python embedder.py

  # From file
  python embedder.py --input chunks.json

  # Custom batch size
  python embedder.py --input chunks.json --batch-size 50

Environment Variables:
  OPENAI_API_KEY: Required. Your OpenAI API key.
        """
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        help='Input JSON file path (uses stdin if not provided)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output JSON file path (uses stdout if not provided)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of chunks to process per batch (default: 100)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Read input
        if args.input:
            with open(args.input, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json.load(sys.stdin)

        # Extract chunks array
        if isinstance(input_data, dict):
            chunks = input_data.get('chunks', [])
        elif isinstance(input_data, list):
            chunks = input_data
        else:
            raise ValueError("Input must be a JSON object with 'chunks' array or a JSON array")

        # Create agent with custom config
        config = EmbeddingConfig(batch_size=args.batch_size)
        agent = EmbedderAgent(config)

        # Process chunks
        result = agent.process(chunks)

        # Write output
        output_json = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            logger.info(f"Results written to {args.output}")
        else:
            print(output_json)

        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)

    except FileNotFoundError as e:
        error_result = {
            'success': False,
            'error': f'File not found: {str(e)}',
            'embeddings': []
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except json.JSONDecodeError as e:
        error_result = {
            'success': False,
            'error': f'Invalid JSON input: {str(e)}',
            'embeddings': []
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except ValueError as e:
        error_result = {
            'success': False,
            'error': str(e),
            'embeddings': []
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except KeyboardInterrupt:
        error_result = {
            'success': False,
            'error': 'Operation cancelled by user',
            'embeddings': []
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        error_result = {
            'success': False,
            'error': f'Unexpected error: {str(e)}',
            'embeddings': []
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == '__main__':
    main()
