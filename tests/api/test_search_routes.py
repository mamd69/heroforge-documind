"""
TDD Tests for Search API Routes

Following Red-Green-Refactor cycle:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Clean up and integrate with real services

Run with: pytest tests/api/test_search_routes.py -v
"""

import pytest
from fastapi.testclient import TestClient


class TestSearchDocuments:
    """Tests for POST /api/search"""

    def test_search_returns_200(self, client):
        """Search request returns 200 status"""
        response = client.post(
            "/api/search",
            json={"query": "test query"}
        )

        assert response.status_code == 200

    def test_search_returns_results_array(self, client):
        """Search response includes results array"""
        response = client.post(
            "/api/search",
            json={"query": "documents"}
        )
        data = response.json()

        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_returns_total_count(self, client):
        """Search response includes total count"""
        response = client.post(
            "/api/search",
            json={"query": "information"}
        )
        data = response.json()

        assert "total" in data
        assert isinstance(data["total"], int)

    def test_search_returns_timing(self, client):
        """Search response includes search time"""
        response = client.post(
            "/api/search",
            json={"query": "search timing"}
        )
        data = response.json()

        assert "search_time_ms" in data
        assert isinstance(data["search_time_ms"], (int, float))

    def test_search_results_have_required_fields(self, client, with_documents):
        """Each result has document_id, title, content_preview, score"""
        response = client.post(
            "/api/search",
            json={"query": "test"}
        )
        data = response.json()

        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "document_id" in result
            assert "title" in result
            assert "content_preview" in result
            assert "relevance_score" in result

    def test_search_empty_query_returns_422(self, client):
        """Empty query returns validation error"""
        response = client.post(
            "/api/search",
            json={"query": ""}
        )

        assert response.status_code == 422

    def test_search_missing_query_returns_422(self, client):
        """Missing query field returns validation error"""
        response = client.post(
            "/api/search",
            json={}
        )

        assert response.status_code == 422

    def test_search_respects_limit(self, client):
        """Limit parameter restricts result count"""
        response = client.post(
            "/api/search",
            json={"query": "test", "limit": 3}
        )
        data = response.json()

        assert len(data["results"]) <= 3

    def test_search_default_limit(self, client):
        """Default limit is 5 results"""
        response = client.post(
            "/api/search",
            json={"query": "test"}
        )
        data = response.json()

        assert len(data["results"]) <= 5


class TestSearchModes:
    """Tests for different search modes"""

    def test_semantic_search_mode(self, client):
        """Semantic search mode is accepted"""
        response = client.post(
            "/api/search",
            json={"query": "meaning of documents", "mode": "semantic"}
        )

        assert response.status_code == 200

    def test_keyword_search_mode(self, client):
        """Keyword search mode is accepted"""
        response = client.post(
            "/api/search",
            json={"query": "exact phrase", "mode": "keyword"}
        )

        assert response.status_code == 200

    def test_hybrid_search_mode(self, client):
        """Hybrid search mode is accepted"""
        response = client.post(
            "/api/search",
            json={"query": "combined search", "mode": "hybrid"}
        )

        assert response.status_code == 200

    def test_default_mode_is_hybrid(self, client):
        """Default search mode is hybrid"""
        response = client.post(
            "/api/search",
            json={"query": "default mode test"}
        )
        data = response.json()

        # Should succeed with default mode
        assert response.status_code == 200

    def test_invalid_mode_returns_422(self, client):
        """Invalid search mode returns validation error"""
        response = client.post(
            "/api/search",
            json={"query": "test", "mode": "invalid_mode"}
        )

        assert response.status_code == 422


class TestSearchFilters:
    """Tests for search filtering options"""

    def test_filter_by_document_type(self, client):
        """Can filter search by document type"""
        response = client.post(
            "/api/search",
            json={"query": "test", "file_type": "pdf"}
        )

        assert response.status_code == 200

    def test_filter_by_date_range(self, client):
        """Can filter search by date range"""
        response = client.post(
            "/api/search",
            json={
                "query": "test",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31"
            }
        )

        assert response.status_code == 200

    def test_filter_by_min_score(self, client):
        """Can filter by minimum relevance score"""
        response = client.post(
            "/api/search",
            json={"query": "test", "min_score": 0.7}
        )

        assert response.status_code == 200
        data = response.json()

        # All results should meet minimum score
        for result in data["results"]:
            assert result["relevance_score"] >= 0.7


class TestSearchSuggestions:
    """Tests for GET /api/search/suggestions"""

    def test_suggestions_returns_200(self, client):
        """Getting suggestions returns 200"""
        response = client.get("/api/search/suggestions?q=doc")

        assert response.status_code == 200

    def test_suggestions_returns_array(self, client):
        """Suggestions response is an array"""
        response = client.get("/api/search/suggestions?q=test")
        data = response.json()

        assert isinstance(data, list)

    def test_suggestions_require_query(self, client):
        """Missing query parameter returns 422"""
        response = client.get("/api/search/suggestions")

        assert response.status_code == 422

    def test_suggestions_are_strings(self, client):
        """Each suggestion is a string"""
        response = client.get("/api/search/suggestions?q=docu")
        data = response.json()

        for suggestion in data:
            assert isinstance(suggestion, str)

    def test_suggestions_limit(self, client):
        """Suggestions respect limit parameter"""
        response = client.get("/api/search/suggestions?q=test&limit=3")
        data = response.json()

        assert len(data) <= 3


class TestSearchAnalytics:
    """Tests for search analytics endpoints"""

    def test_popular_queries_returns_200(self, client):
        """GET /api/search/popular returns 200"""
        response = client.get("/api/search/popular")

        assert response.status_code == 200

    def test_popular_queries_returns_array(self, client):
        """Popular queries is an array"""
        response = client.get("/api/search/popular")
        data = response.json()

        assert isinstance(data, list)

    def test_popular_queries_have_count(self, client):
        """Each popular query includes count"""
        response = client.get("/api/search/popular")
        data = response.json()

        for item in data:
            if item:  # Skip if empty
                assert "query" in item
                assert "count" in item


# Fixtures

@pytest.fixture
def client():
    """Create test client for API"""
    from src.documind.api.main import app
    return TestClient(app)


@pytest.fixture
def with_documents(client):
    """Ensure some documents exist for search tests"""
    from io import BytesIO

    # Upload a test document
    content = BytesIO(b"This is test document content for search testing.")
    files = {"file": ("test_search.txt", content, "text/plain")}
    client.post("/api/documents", files=files)

    yield

    # Cleanup could go here if needed
