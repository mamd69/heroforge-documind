"""
Comprehensive Test Suite for Production Q&A System

Tests cover:
- Enhanced search with re-ranking and deduplication
- Query method with citations and fallback
- Model comparison functionality
- Query logging
- Analytics generation

Run with: pytest tests/rag/test_production_qa.py -v
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the module under test
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from documind.rag.production_qa import ProductionQA


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def qa_system():
    """Create ProductionQA instance with logging disabled for testing."""
    return ProductionQA(enable_logging=False)


@pytest.fixture
def qa_system_with_logging():
    """Create ProductionQA instance with logging enabled."""
    return ProductionQA(enable_logging=True)


@pytest.fixture
def sample_documents():
    """Sample document results for testing."""
    return [
        {
            "id": "doc-1",
            "content": "Employees are entitled to 15 days of vacation per year. New employees receive their full allocation on day one.",
            "metadata": {"document_name": "hr_policies.txt"},
            "similarity": 0.92,
            "document_name": "hr_policies.txt",
            "chunk_index": 0,
        },
        {
            "id": "doc-2",
            "content": "Vacation requests must be submitted at least two weeks in advance through the HR portal.",
            "metadata": {"document_name": "hr_policies.txt"},
            "similarity": 0.85,
            "document_name": "hr_policies.txt",
            "chunk_index": 1,
        },
        {
            "id": "doc-3",
            "content": "Unused vacation days can be carried over to the next year, up to a maximum of 5 days.",
            "metadata": {"document_name": "benefits_guide.txt"},
            "similarity": 0.78,
            "document_name": "benefits_guide.txt",
            "chunk_index": 3,
        },
        {
            "id": "doc-4",
            "content": "Employees are entitled to 15 days of vacation per year. New employees receive their full allocation on day one.",
            "metadata": {"document_name": "employee_handbook.txt"},
            "similarity": 0.88,
            "document_name": "employee_handbook.txt",
            "chunk_index": 5,
        },
    ]


@pytest.fixture
def mock_search_response(sample_documents):
    """Mock search_documents response."""
    return sample_documents


@pytest.fixture
def mock_llm_response():
    """Mock LLM response object."""
    mock = Mock()
    mock.choices = [
        Mock(
            message=Mock(
                content="According to [Source 1], employees receive 15 days of vacation per year. "
                "New employees get their full allocation immediately [Source 1]. "
                "Vacation requests should be submitted two weeks in advance [Source 2]."
            )
        )
    ]
    mock.usage = Mock(prompt_tokens=500, completion_tokens=100, total_tokens=600)
    return mock


# =============================================================================
# TEST: ENHANCED SEARCH
# =============================================================================


class TestEnhancedSearch:
    """Tests for enhanced search functionality."""

    def test_enhanced_search_empty_query_raises_error(self, qa_system):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            qa_system.enhanced_search("")

        with pytest.raises(ValueError, match="empty"):
            qa_system.enhanced_search("   ")

    @patch("documind.rag.production_qa.search_documents")
    def test_enhanced_search_basic(self, mock_search, qa_system, sample_documents):
        """Test basic enhanced search returns results."""
        mock_search.return_value = sample_documents

        results = qa_system.enhanced_search("vacation policy", top_k=5)

        assert len(results) <= 5
        assert all("similarity" in r for r in results)
        assert all("content" in r for r in results)
        mock_search.assert_called_once()

    @patch("documind.rag.production_qa.search_documents")
    def test_enhanced_search_deduplication(self, mock_search, qa_system, sample_documents):
        """Test deduplication removes similar chunks."""
        mock_search.return_value = sample_documents

        # With deduplication - should remove duplicate content
        results_deduped = qa_system.enhanced_search(
            "vacation policy", top_k=10, deduplicate=True
        )

        # doc-1 and doc-4 have identical content, one should be removed
        assert len(results_deduped) < len(sample_documents)

    @patch("documind.rag.production_qa.search_documents")
    def test_enhanced_search_reranking(self, mock_search, qa_system, sample_documents):
        """Test re-ranking adds final_score."""
        mock_search.return_value = sample_documents

        results = qa_system.enhanced_search(
            "vacation policy", top_k=5, rerank=True, deduplicate=False
        )

        # All results should have final_score
        assert all("final_score" in r for r in results)

        # Results should be sorted by final_score (descending)
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("documind.rag.production_qa.search_documents")
    def test_enhanced_search_citation_enrichment(
        self, mock_search, qa_system, sample_documents
    ):
        """Test that results include citation metadata."""
        mock_search.return_value = sample_documents[:2]

        results = qa_system.enhanced_search("vacation policy", top_k=2)

        for result in results:
            assert "citation_number" in result
            assert "document_link" in result
            assert "citation_format" in result
            assert "highlighted_content" in result

    @patch("documind.rag.production_qa.hybrid_search")
    def test_enhanced_search_hybrid_mode(self, mock_hybrid, qa_system, sample_documents):
        """Test hybrid search mode."""
        mock_hybrid.return_value = sample_documents

        results = qa_system.enhanced_search("vacation policy", use_hybrid=True)

        mock_hybrid.assert_called_once()
        assert len(results) > 0


# =============================================================================
# TEST: QUERY METHOD
# =============================================================================


class TestQuery:
    """Tests for main query method."""

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_basic(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test basic query returns answer with sources."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.query("What is the vacation policy?")

        assert "answer" in result
        assert "sources" in result
        assert "model" in result
        assert "timing" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_includes_citations(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test that answer includes proper citations."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.query("What is the vacation policy?")

        # Answer should reference sources
        assert "[Source" in result["answer"]

        # Sources should have required fields
        for source in result["sources"]:
            assert "citation_number" in source
            assert "document" in source
            assert "was_cited" in source

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_timing_metrics(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test that timing metrics are captured."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.query("What is the vacation policy?")

        timing = result["timing"]
        assert "search" in timing
        assert "generation" in timing
        assert "total" in timing
        assert timing["total"] >= timing["search"] + timing["generation"] - 0.01

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_with_specific_model(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test query with specific model selection."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.query(
            "What is the vacation policy?", model="anthropic/claude-3.5-haiku"
        )

        # Verify the model was passed to the API
        call_args = mock_client.return_value.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "anthropic/claude-3.5-haiku"

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_fallback_mechanism(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test fallback to alternative model on failure."""
        mock_search.return_value = sample_documents

        # First call fails, second succeeds
        mock_client.return_value.chat.completions.create.side_effect = [
            Exception("Primary model failed"),
            mock_llm_response,
        ]

        result = qa_system.query("What is the vacation policy?", enable_fallback=True)

        assert "answer" in result
        assert result["fallback_used"] is True

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_without_fallback_fails(
        self, mock_search, mock_client, qa_system, sample_documents
    ):
        """Test that query fails without fallback when model errors."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.side_effect = Exception(
            "Model failed"
        )

        with pytest.raises(Exception, match="failed"):
            qa_system.query("What is the vacation policy?", enable_fallback=False)

    def test_query_empty_question_raises_error(self, qa_system):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            qa_system.query("")

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_query_complexity_analysis(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test query complexity analysis."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        # Simple query
        result1 = qa_system.query("What is vacation?")
        assert result1["complexity"] == "simple"

        # Complex query
        result2 = qa_system.query(
            "Compare and analyze the differences between vacation and sick leave policies"
        )
        assert result2["complexity"] == "complex"


# =============================================================================
# TEST: MODEL COMPARISON
# =============================================================================


class TestModelComparison:
    """Tests for multi-model comparison."""

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_compare_models_basic(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test basic model comparison."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=["google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"],
        )

        assert "query" in result
        assert "models_compared" in result
        assert "results" in result
        assert "analysis" in result
        assert "sources" in result

        # Should have results for both models
        assert len(result["results"]) == 2

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_compare_models_analysis(
        self, mock_search, mock_client, qa_system, sample_documents
    ):
        """Test that comparison includes analysis."""
        mock_search.return_value = sample_documents

        # Create different responses for different models
        mock1 = Mock()
        mock1.choices = [Mock(message=Mock(content="Short answer."))]
        mock1.usage = Mock(prompt_tokens=100, completion_tokens=10, total_tokens=110)

        mock2 = Mock()
        mock2.choices = [
            Mock(message=Mock(content="This is a much longer and more detailed answer."))
        ]
        mock2.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        mock_client.return_value.chat.completions.create.side_effect = [mock1, mock2]

        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=["model1", "model2"],
            parallel=False,
        )

        analysis = result["analysis"]
        assert "fastest_model" in analysis
        assert "slowest_model" in analysis
        assert "shortest_answer" in analysis
        assert "longest_answer" in analysis

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_compare_models_handles_failures(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test graceful handling of model failures."""
        mock_search.return_value = sample_documents

        # First model fails, second succeeds
        mock_client.return_value.chat.completions.create.side_effect = [
            Exception("Model unavailable"),
            mock_llm_response,
        ]

        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=["model1", "model2"],
            parallel=False,
        )

        # Should have error for failed model
        assert "error" in result["results"]["model1"]

        # Should have answer for working model
        assert "answer" in result["results"]["model2"]


# =============================================================================
# TEST: QUERY LOGGING
# =============================================================================


class TestQueryLogging:
    """Tests for query logging functionality."""

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_log_query_stores_data(self, mock_supabase, qa_system_with_logging):
        """Test that queries are logged to database."""
        mock_response = Mock()
        mock_response.data = [{"id": 123}]
        mock_supabase.return_value.table.return_value.insert.return_value.execute.return_value = (
            mock_response
        )

        query_data = {
            "query": "What is the vacation policy?",
            "answer": "According to [Source 1], 15 days.",
            "model": "google/gemini-2.5-flash-lite",
            "sources": [
                {
                    "document": "hr_policies.txt",
                    "chunk_index": 0,
                    "similarity": 0.92,
                    "was_cited": True,
                }
            ],
            "timing": {"total": 1.5},
            "complexity": "simple",
            "fallback_used": False,
        }

        log_id = qa_system_with_logging.log_query(query_data)

        assert log_id == 123
        mock_supabase.return_value.table.assert_called_with("query_logs")

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_log_query_handles_failure_gracefully(self, mock_supabase, qa_system_with_logging):
        """Test that logging failures don't raise exceptions."""
        mock_supabase.return_value.table.return_value.insert.return_value.execute.side_effect = (
            Exception("Database error")
        )

        query_data = {
            "query": "Test",
            "answer": "Answer",
            "model": "test",
            "timing": {"total": 1.0},
        }

        # Should not raise, just return None
        result = qa_system_with_logging.log_query(query_data)
        assert result is None

    def test_pii_redaction(self, qa_system):
        """Test PII redaction in queries."""
        qa = ProductionQA(enable_pii_redaction=True, enable_logging=False)

        text_with_pii = "Contact john@example.com or call 555-123-4567"
        redacted = qa._redact_pii(text_with_pii)

        assert "john@example.com" not in redacted
        assert "[REDACTED_EMAIL]" in redacted
        assert "555-123-4567" not in redacted
        assert "[REDACTED_PHONE]" in redacted


# =============================================================================
# TEST: ANALYTICS
# =============================================================================


class TestAnalytics:
    """Tests for analytics functionality."""

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_get_analytics_basic(self, mock_supabase, qa_system):
        """Test basic analytics retrieval."""
        mock_response = Mock()
        mock_response.data = [
            {
                "question": "What is vacation?",
                "model": "google/gemini-2.5-flash-lite",
                "response_time": 1.5,
                "fallback_used": False,
            },
            {
                "question": "How many sick days?",
                "model": "google/gemini-2.5-flash-lite",
                "response_time": 1.2,
                "fallback_used": False,
            },
        ]
        mock_supabase.return_value.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        analytics = qa_system.get_analytics()

        assert "total_queries" in analytics
        assert analytics["total_queries"] == 2
        assert "performance" in analytics
        assert "models" in analytics
        assert "insights" in analytics

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_get_analytics_performance_metrics(self, mock_supabase, qa_system):
        """Test performance metrics calculation."""
        mock_response = Mock()
        mock_response.data = [
            {"response_time": 1.0, "model": "test", "question": "q1"},
            {"response_time": 2.0, "model": "test", "question": "q2"},
            {"response_time": 3.0, "model": "test", "question": "q3"},
        ]
        mock_supabase.return_value.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        analytics = qa_system.get_analytics()

        perf = analytics["performance"]
        assert "avg_response_time" in perf
        assert perf["avg_response_time"] == 2.0  # (1+2+3)/3
        assert perf["fastest_query"] == 1.0
        assert perf["slowest_query"] == 3.0

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_get_analytics_model_usage(self, mock_supabase, qa_system):
        """Test model usage statistics."""
        mock_response = Mock()
        mock_response.data = [
            {"model": "model-a", "response_time": 1.0, "fallback_used": False, "question": "q1"},
            {"model": "model-a", "response_time": 1.5, "fallback_used": False, "question": "q2"},
            {"model": "model-b", "response_time": 2.0, "fallback_used": True, "question": "q3"},
        ]
        mock_supabase.return_value.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        analytics = qa_system.get_analytics()

        models = analytics["models"]
        assert "model-a" in models
        assert models["model-a"]["usage_count"] == 2
        assert models["model-b"]["usage_count"] == 1
        assert models["model-b"]["fallback_rate"] == 100.0

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_get_analytics_insights(self, mock_supabase, qa_system):
        """Test automatic insight generation."""
        mock_response = Mock()
        mock_response.data = [
            {"model": "test", "response_time": 1.5, "question": "vacation policy query"},
        ]
        mock_supabase.return_value.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        analytics = qa_system.get_analytics()

        assert "insights" in analytics
        assert isinstance(analytics["insights"], list)
        assert len(analytics["insights"]) > 0

    @patch("documind.rag.production_qa._get_supabase_client")
    def test_get_analytics_empty_results(self, mock_supabase, qa_system):
        """Test analytics with no data."""
        mock_response = Mock()
        mock_response.data = []
        mock_supabase.return_value.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = (
            mock_response
        )

        analytics = qa_system.get_analytics()

        assert analytics["total_queries"] == 0
        assert "No queries found" in analytics["insights"][0]


# =============================================================================
# TEST: HELPER METHODS
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_analyze_complexity_simple(self, qa_system):
        """Test simple query complexity detection."""
        assert qa_system._analyze_complexity("What is vacation?") == "simple"
        assert qa_system._analyze_complexity("How many days?") == "simple"

    def test_analyze_complexity_medium(self, qa_system):
        """Test medium query complexity detection."""
        # 9+ words without complex terms = medium
        assert (
            qa_system._analyze_complexity("What is the vacation policy for all new full-time employees?")
            == "medium"
        )

    def test_analyze_complexity_complex(self, qa_system):
        """Test complex query complexity detection."""
        assert (
            qa_system._analyze_complexity(
                "Compare the differences between vacation and sick leave policies"
            )
            == "complex"
        )
        assert (
            qa_system._analyze_complexity("Analyze our benefits in detail")
            == "complex"
        )

    def test_highlight_query_terms(self, qa_system):
        """Test query term highlighting."""
        content = "Employees receive vacation days annually."
        query = "vacation days"

        highlighted = qa_system._highlight_query_terms(content, query)

        assert "**vacation**" in highlighted
        assert "**days**" in highlighted

    def test_generate_document_link(self, qa_system):
        """Test document link generation."""
        doc = {
            "document_name": "hr_policies.txt",
            "chunk_index": 5,
            "id": "abc123",
        }

        link = qa_system._generate_document_link(doc)

        assert "hr_policies.txt" in link
        assert "chunk-5" in link
        assert "abc123" in link


# =============================================================================
# TEST: INTEGRATION
# =============================================================================


class TestIntegration:
    """Integration tests (require mocking external services)."""

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    def test_complete_rag_pipeline(
        self, mock_search, mock_client, qa_system, sample_documents, mock_llm_response
    ):
        """Test complete RAG pipeline from query to answer."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response

        result = qa_system.query("What is the vacation policy for new employees?")

        # Verify all components worked
        assert result["answer"]
        assert len(result["sources"]) > 0
        assert result["timing"]["total"] > 0
        assert any(src["was_cited"] for src in result["sources"])

    @patch("documind.rag.production_qa._get_openrouter_client")
    @patch("documind.rag.production_qa.search_documents")
    @patch("documind.rag.production_qa._get_supabase_client")
    def test_query_with_logging_integration(
        self,
        mock_supabase,
        mock_search,
        mock_client,
        sample_documents,
        mock_llm_response,
    ):
        """Test query with logging enabled."""
        mock_search.return_value = sample_documents
        mock_client.return_value.chat.completions.create.return_value = mock_llm_response
        mock_log_response = Mock()
        mock_log_response.data = [{"id": 456}]
        mock_supabase.return_value.table.return_value.insert.return_value.execute.return_value = (
            mock_log_response
        )

        qa = ProductionQA(enable_logging=True)
        result = qa.query("What is vacation?")

        assert result["answer"]
        mock_supabase.return_value.table.assert_called_with("query_logs")


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
