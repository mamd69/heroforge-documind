"""
Search API Schemas

Pydantic models for search endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum


class SearchMode(str, Enum):
    """Search algorithm mode"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Search request parameters"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search algorithm to use"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum results to return"
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Minimum relevance score (0-1)"
    )
    file_type: Optional[str] = Field(
        None,
        description="Filter by file type (pdf, docx, etc.)"
    )
    date_from: Optional[date] = Field(
        None,
        description="Filter documents uploaded after this date"
    )
    date_to: Optional[date] = Field(
        None,
        description="Filter documents uploaded before this date"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "vacation policy",
                "mode": "hybrid",
                "limit": 5,
                "min_score": 0.7
            }
        }


class SearchResult(BaseModel):
    """A single search result"""
    document_id: str = Field(..., description="Document UUID")
    chunk_id: str = Field(..., description="Chunk UUID")
    title: str = Field(..., description="Document title")
    file_type: str = Field(..., description="Document file type")
    content_preview: str = Field(..., description="Relevant content snippet")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    section_heading: Optional[str] = Field(None, description="Section heading")
    highlights: List[str] = Field(
        default_factory=list,
        description="Highlighted matching phrases"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                "title": "Company Policies",
                "file_type": "pdf",
                "content_preview": "...employees are entitled to 15 days of paid vacation...",
                "relevance_score": 0.92,
                "section_heading": "Vacation Policy",
                "highlights": ["15 days", "paid vacation"]
            }
        }


class SearchResponse(BaseModel):
    """Search results response"""
    results: List[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    search_time_ms: float = Field(..., description="Search execution time")
    mode_used: SearchMode = Field(..., description="Search mode that was used")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                        "title": "Company Policies",
                        "file_type": "pdf",
                        "content_preview": "...employees are entitled to 15 days...",
                        "relevance_score": 0.92
                    }
                ],
                "total": 3,
                "search_time_ms": 145.2,
                "mode_used": "hybrid"
            }
        }


class SearchSuggestion(BaseModel):
    """Search suggestion/autocomplete item"""
    text: str = Field(..., description="Suggested search text")
    score: Optional[float] = Field(None, description="Suggestion relevance")


class PopularQuery(BaseModel):
    """Popular search query"""
    query: str = Field(..., description="Search query text")
    count: int = Field(..., description="Number of times searched")


class SearchAnalytics(BaseModel):
    """Search analytics summary"""
    total_searches: int = Field(..., description="Total searches performed")
    unique_queries: int = Field(..., description="Unique query count")
    avg_results_per_search: float = Field(..., description="Average results returned")
    popular_queries: List[PopularQuery] = Field(..., description="Most popular queries")
