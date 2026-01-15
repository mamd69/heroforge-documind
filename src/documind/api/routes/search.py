"""
Search API Routes

Endpoints for semantic and hybrid search across documents.
"""

from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
import time

from ..schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchMode,
    PopularQuery,
)

router = APIRouter(
    prefix="/api/search",
    tags=["Search"]
)

# Track search queries for analytics
_search_history: List[dict] = []


@router.post(
    "",
    response_model=SearchResponse,
    summary="Search documents",
    description="Search across all documents using semantic, keyword, or hybrid search."
)
async def search_documents(request: SearchRequest):
    """Execute a search query"""
    start_time = time.time()

    # TODO: Integrate with HybridSearch service
    from ..services.search_service import SearchService
    service = SearchService()

    try:
        results = await service.search(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            min_score=request.min_score,
            file_type=request.file_type,
            date_from=request.date_from,
            date_to=request.date_to
        )

        search_time = (time.time() - start_time) * 1000

        # Track search for analytics
        _search_history.append({
            "query": request.query,
            "mode": request.mode,
            "results_count": len(results),
            "timestamp": time.time()
        })

        return SearchResponse(
            results=results,
            total=len(results),
            search_time_ms=search_time,
            mode_used=request.mode
        )

    except Exception as e:
        # Fallback with empty results
        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=[],
            total=0,
            search_time_ms=search_time,
            mode_used=request.mode
        )


@router.get(
    "/suggestions",
    response_model=List[str],
    summary="Get search suggestions",
    description="Get autocomplete suggestions for a search query."
)
async def get_suggestions(
    q: str = Query(..., min_length=1, description="Query prefix"),
    limit: int = Query(default=5, ge=1, le=10)
):
    """Get search suggestions"""
    # TODO: Implement proper suggestion service
    # For now, return based on search history

    suggestions = []
    seen = set()

    for item in reversed(_search_history):
        query = item["query"]
        if query.lower().startswith(q.lower()) and query not in seen:
            suggestions.append(query)
            seen.add(query)
            if len(suggestions) >= limit:
                break

    return suggestions


@router.get(
    "/popular",
    response_model=List[PopularQuery],
    summary="Get popular queries",
    description="Get the most popular search queries."
)
async def get_popular_queries(
    limit: int = Query(default=10, ge=1, le=50)
):
    """Get popular search queries"""
    # Count queries
    query_counts: dict[str, int] = {}

    for item in _search_history:
        query = item["query"]
        query_counts[query] = query_counts.get(query, 0) + 1

    # Sort by count
    sorted_queries = sorted(
        query_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:limit]

    return [
        PopularQuery(query=q, count=c)
        for q, c in sorted_queries
    ]


@router.get(
    "/recent",
    response_model=List[dict],
    summary="Get recent searches",
    description="Get recent search queries for the current session."
)
async def get_recent_searches(
    limit: int = Query(default=10, ge=1, le=50)
):
    """Get recent searches"""
    return [
        {
            "query": item["query"],
            "results_count": item["results_count"]
        }
        for item in reversed(_search_history[-limit:])
    ]
