#!/usr/bin/env python3
"""
Tests for Embedder Agent

Tests embedding generation, batching, retry logic, and error handling.
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.pipeline.embedder import (
    EmbedderAgent,
    EmbeddingConfig,
    Chunk,
    Embedding,
    RateLimitError,
    APIError
)


class TestChunk(unittest.TestCase):
    """Test Chunk dataclass"""

    def test_from_dict_full(self):
        """Test creating Chunk from complete dictionary"""
        data = {
            'chunk_id': 'chunk-1',
            'text': 'Test content',
            'metadata': {'source': 'test'}
        }
        chunk = Chunk.from_dict(data)
        self.assertEqual(chunk.chunk_id, 'chunk-1')
        self.assertEqual(chunk.text, 'Test content')
        self.assertEqual(chunk.metadata, {'source': 'test'})

    def test_from_dict_alternative_keys(self):
        """Test creating Chunk with alternative key names"""
        data = {
            'id': 'chunk-2',
            'content': 'Alternative keys',
        }
        chunk = Chunk.from_dict(data)
        self.assertEqual(chunk.chunk_id, 'chunk-2')
        self.assertEqual(chunk.text, 'Alternative keys')

    def test_from_dict_minimal(self):
        """Test creating Chunk with minimal data"""
        data = {}
        chunk = Chunk.from_dict(data)
        self.assertEqual(chunk.chunk_id, '')
        self.assertEqual(chunk.text, '')
        self.assertEqual(chunk.metadata, {})


class TestEmbedding(unittest.TestCase):
    """Test Embedding dataclass"""

    def test_to_dict(self):
        """Test converting Embedding to dictionary"""
        embedding = Embedding(
            chunk_id='chunk-1',
            vector=[0.1, 0.2, 0.3],
            model='text-embedding-3-small',
            dimensions=3
        )
        result = embedding.to_dict()

        self.assertEqual(result['chunk_id'], 'chunk-1')
        self.assertEqual(result['vector'], [0.1, 0.2, 0.3])
        self.assertEqual(result['model'], 'text-embedding-3-small')
        self.assertEqual(result['dimensions'], 3)


class TestEmbedderAgent(unittest.TestCase):
    """Test EmbedderAgent functionality"""

    def setUp(self):
        """Set up test environment"""
        # Mock the OpenAI API key
        os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    def tearDown(self):
        """Clean up test environment"""
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

    def test_init_no_api_key(self):
        """Test initialization fails without API key"""
        del os.environ['OPENAI_API_KEY']

        with self.assertRaises(ValueError) as context:
            EmbedderAgent()

        self.assertIn('OPENAI_API_KEY', str(context.exception))

    @patch('agents.pipeline.embedder.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        """Test successful initialization with API key"""
        agent = EmbedderAgent()

        self.assertIsNotNone(agent.client)
        self.assertEqual(agent.config.model, 'text-embedding-3-small')
        self.assertEqual(agent.config.dimensions, 1536)
        mock_openai.assert_called_once_with(api_key='test-key-12345')

    @patch('agents.pipeline.embedder.OpenAI')
    def test_batch_chunks(self, mock_openai):
        """Test chunk batching logic"""
        config = EmbeddingConfig(batch_size=2)
        agent = EmbedderAgent(config)

        chunks = [
            Chunk(chunk_id=f'chunk-{i}', text=f'Text {i}')
            for i in range(5)
        ]

        batches = agent._batch_chunks(chunks)

        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)
        self.assertEqual(len(batches[2]), 1)

    @patch('agents.pipeline.embedder.OpenAI')
    def test_embed_batch_success(self, mock_openai):
        """Test successful batch embedding"""
        # Mock the embeddings response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]

        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = EmbedderAgent()
        texts = ['Text 1', 'Text 2']

        embeddings = agent._embed_batch_with_retry(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 1536)
        self.assertEqual(embeddings[0][0], 0.1)
        self.assertEqual(embeddings[1][0], 0.2)

    @patch('agents.pipeline.embedder.OpenAI')
    @patch('time.sleep')
    def test_embed_batch_rate_limit_retry(self, mock_sleep, mock_openai):
        """Test retry logic on rate limit error"""
        mock_client = Mock()

        # First call raises RateLimitError, second succeeds
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]

        mock_client.embeddings.create.side_effect = [
            RateLimitError('Rate limit exceeded'),
            mock_response
        ]
        mock_openai.return_value = mock_client

        config = EmbeddingConfig(initial_retry_delay=0.1)
        agent = EmbedderAgent(config)
        texts = ['Text 1']

        embeddings = agent._embed_batch_with_retry(texts)

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(mock_client.embeddings.create.call_count, 2)
        mock_sleep.assert_called_once()

    @patch('agents.pipeline.embedder.OpenAI')
    @patch('time.sleep')
    def test_embed_batch_max_retries_exceeded(self, mock_sleep, mock_openai):
        """Test max retries exceeded"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = RateLimitError('Rate limit exceeded')
        mock_openai.return_value = mock_client

        config = EmbeddingConfig(max_retries=2, initial_retry_delay=0.01)
        agent = EmbedderAgent(config)
        texts = ['Text 1']

        with self.assertRaises(RateLimitError):
            agent._embed_batch_with_retry(texts)

        # Should try initial + 2 retries = 3 total
        self.assertEqual(mock_client.embeddings.create.call_count, 3)

    @patch('agents.pipeline.embedder.OpenAI')
    def test_process_empty_chunks(self, mock_openai):
        """Test processing empty chunk list"""
        agent = EmbedderAgent()
        result = agent.process([])

        self.assertTrue(result['success'])
        self.assertEqual(len(result['embeddings']), 0)
        self.assertEqual(result['metadata']['total_chunks'], 0)

    @patch('agents.pipeline.embedder.OpenAI')
    def test_process_chunks_success(self, mock_openai):
        """Test successful chunk processing"""
        # Mock the embeddings response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]

        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        agent = EmbedderAgent()
        chunks = [
            {'chunk_id': 'chunk-1', 'text': 'Test text 1'},
            {'chunk_id': 'chunk-2', 'text': 'Test text 2'}
        ]

        result = agent.process(chunks)

        self.assertTrue(result['success'])
        self.assertEqual(len(result['embeddings']), 2)
        self.assertEqual(result['embeddings'][0]['chunk_id'], 'chunk-1')
        self.assertEqual(result['embeddings'][0]['dimensions'], 1536)
        self.assertEqual(result['embeddings'][0]['model'], 'text-embedding-3-small')
        self.assertEqual(result['metadata']['total_chunks'], 2)

    @patch('agents.pipeline.embedder.OpenAI')
    def test_process_chunks_with_empty_text(self, mock_openai):
        """Test processing chunks with empty text"""
        agent = EmbedderAgent()
        chunks = [
            {'chunk_id': 'chunk-1', 'text': ''},
        ]

        result = agent.process(chunks)

        self.assertFalse(result['success'])
        self.assertIn('no text content', result['error'].lower())

    @patch('agents.pipeline.embedder.OpenAI')
    def test_process_chunks_api_error(self, mock_openai):
        """Test processing with API error"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = APIError('API Error')
        mock_openai.return_value = mock_client

        config = EmbeddingConfig(max_retries=1, initial_retry_delay=0.01)
        agent = EmbedderAgent(config)
        chunks = [
            {'chunk_id': 'chunk-1', 'text': 'Test text'},
        ]

        result = agent.process(chunks)

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertEqual(len(result['embeddings']), 0)

    @patch('agents.pipeline.embedder.OpenAI')
    def test_exponential_backoff_calculation(self, mock_openai):
        """Test exponential backoff delay calculation"""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = RateLimitError('Rate limit')
        mock_openai.return_value = mock_client

        config = EmbeddingConfig(
            max_retries=3,
            initial_retry_delay=1.0,
            max_retry_delay=10.0
        )
        agent = EmbedderAgent(config)

        with patch('time.sleep') as mock_sleep:
            try:
                agent._embed_batch_with_retry(['test'])
            except RateLimitError:
                pass

            # Check exponential backoff: 1s, 2s, 4s
            calls = [call[0][0] for call in mock_sleep.call_args_list]
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0], 1.0)
            self.assertEqual(calls[1], 2.0)
            self.assertEqual(calls[2], 4.0)


class TestEmbedderIntegration(unittest.TestCase):
    """Integration tests for Embedder Agent"""

    def setUp(self):
        """Set up test environment"""
        os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    def tearDown(self):
        """Clean up test environment"""
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

    @patch('agents.pipeline.embedder.OpenAI')
    def test_large_batch_processing(self, mock_openai):
        """Test processing large number of chunks with batching"""
        # Create 250 chunks (should create 3 batches with batch_size=100)
        chunks = [
            {'chunk_id': f'chunk-{i}', 'text': f'Test text {i}'}
            for i in range(250)
        ]

        # Mock responses for 3 batches
        def create_mock_response(count):
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1 * (i + 1)] * 1536)
                for i in range(count)
            ]
            return mock_response

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = [
            create_mock_response(100),
            create_mock_response(100),
            create_mock_response(50)
        ]
        mock_openai.return_value = mock_client

        config = EmbeddingConfig(batch_size=100)
        agent = EmbedderAgent(config)

        result = agent.process(chunks)

        self.assertTrue(result['success'])
        self.assertEqual(len(result['embeddings']), 250)
        self.assertEqual(result['metadata']['total_batches'], 3)
        self.assertEqual(mock_client.embeddings.create.call_count, 3)


if __name__ == '__main__':
    unittest.main()
