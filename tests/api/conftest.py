"""
Pytest Configuration and Shared Fixtures for API Tests

This module provides common fixtures and configuration for all API tests.
"""

import pytest
import os
from unittest.mock import Mock, patch
from io import BytesIO


# Set test environment
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-key")


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application for testing"""
    from src.documind.api.main import app
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for isolated tests"""
    with patch("src.documind.api.services.chat_service.get_supabase") as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client for embedding generation"""
    with patch("src.documind.api.services.doc_service.get_openai") as mock:
        mock_client = Mock()
        # Return fake embedding
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_production_qa():
    """Mock ProductionQA for chat tests"""
    with patch("src.documind.api.services.chat_service.ProductionQA") as mock:
        mock_qa = Mock()
        mock_qa.query.return_value = Mock(
            answer="This is a test answer.",
            sources=[],
            search_time=100,
            generation_time=500
        )
        mock.return_value = mock_qa
        yield mock_qa


@pytest.fixture
def sample_text_content():
    """Sample text content for testing"""
    return """
    DocuMind User Guide

    Chapter 1: Introduction
    DocuMind is an AI-powered knowledge management system that helps
    you search and query your documents using natural language.

    Chapter 2: Getting Started
    To begin using DocuMind, first upload your documents using the
    upload interface or CLI tool.

    Chapter 3: Querying Documents
    Once your documents are uploaded, you can ask questions in natural
    language and receive AI-generated answers with citations.
    """


@pytest.fixture
def sample_pdf_bytes():
    """Create minimal valid PDF bytes"""
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test Document) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing"""
    return b"""name,department,salary
Alice,Engineering,85000
Bob,Marketing,72000
Charlie,Engineering,90000
Diana,Sales,78000"""


@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    return {"Authorization": "Bearer test-token"}


# Markers for test categorization

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may use real services"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
