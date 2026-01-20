"""
Search Service

Business logic for semantic and hybrid search.
Integrates with existing HybridSearch functionality.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import date
import time

from ..schemas.search import SearchResult, SearchMode

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for search operations.

    Integrates with:
    - HybridSearch for semantic + keyword search
    - Vector database for similarity search
    """

    def __init__(self):
        self._hybrid_search = None
        self._semantic_search = None

    @property
    def hybrid_search_fn(self):
        """Lazy-load hybrid_search function"""
        if self._hybrid_search is None:
            try:
                from ...hybrid_search import hybrid_search
                self._hybrid_search = hybrid_search
            except ImportError:
                logger.warning("HybridSearch not available")
                self._hybrid_search = None
        return self._hybrid_search

    @property
    def semantic_search_fn(self):
        """Lazy-load semantic search function"""
        if self._semantic_search is None:
            try:
                from ...rag.search import search_documents
                self._semantic_search = search_documents
            except ImportError:
                logger.warning("Semantic search not available")
                self._semantic_search = None
        return self._semantic_search

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 5,
        min_score: Optional[float] = None,
        file_type: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None
    ) -> List[SearchResult]:
        """
        Execute a search query.

        Args:
            query: Search query string
            mode: Search algorithm (semantic, keyword, hybrid)
            limit: Maximum results to return
            min_score: Minimum relevance score (0-1)
            file_type: Filter by document type
            date_from: Filter by upload date (after)
            date_to: Filter by upload date (before)

        Returns:
            List of SearchResult objects
        """
        try:
            # Choose search function based on mode
            if mode == SearchMode.HYBRID and self.hybrid_search_fn:
                raw_results = await self._execute_hybrid_search(
                    query, limit * 2  # Get extra for filtering
                )
            elif mode == SearchMode.SEMANTIC and self.semantic_search_fn:
                raw_results = await self._execute_semantic_search(
                    query, limit * 2
                )
            elif mode == SearchMode.KEYWORD:
                raw_results = await self._execute_keyword_search(
                    query, limit * 2
                )
            else:
                # Fallback to any available search
                raw_results = await self._execute_fallback_search(
                    query, limit * 2
                )

            # Apply filters
            filtered_results = self._apply_filters(
                raw_results,
                min_score=min_score,
                file_type=file_type,
                date_from=date_from,
                date_to=date_to
            )

            # Convert to SearchResult objects
            results = [
                self._to_search_result(r)
                for r in filtered_results[:limit]
            ]

            return results

        except Exception as e:
            logger.exception(f"Search error: {e}")
            return []

    async def _execute_hybrid_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute hybrid (semantic + keyword) search"""
        try:
            if self.hybrid_search_fn:
                results = self.hybrid_search_fn(
                    query=query,
                    top_k=limit
                )
                return self._normalize_results(results)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
        return []

    async def _execute_semantic_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute semantic (vector) search"""
        try:
            if self.semantic_search_fn:
                results = self.semantic_search_fn(
                    query=query,
                    top_k=limit
                )
                return self._normalize_results(results)
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        return []

    async def _execute_keyword_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute keyword-based search"""
        # TODO: Implement BM25 or full-text search
        return []

    async def _execute_fallback_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback search when main services unavailable"""
        # Try hybrid first, then semantic
        results = await self._execute_hybrid_search(query, limit)
        if not results:
            results = await self._execute_semantic_search(query, limit)
        return results

    def _normalize_results(
        self,
        results: Any
    ) -> List[Dict[str, Any]]:
        """Normalize results to consistent dict format"""
        normalized = []

        if not results:
            return normalized

        for r in results:
            try:
                if hasattr(r, '__dict__'):
                    normalized.append({
                        'document_id': getattr(r, 'document_id', 'unknown'),
                        'chunk_id': getattr(r, 'chunk_id', 'unknown'),
                        'title': getattr(r, 'title', getattr(r, 'document_title', 'Untitled')),
                        'file_type': getattr(r, 'file_type', 'unknown'),
                        'content': getattr(r, 'content', ''),
                        'score': getattr(r, 'score', getattr(r, 'similarity', 0.5)),
                        'section_heading': getattr(r, 'section_heading', None),
                        'created_at': getattr(r, 'created_at', None)
                    })
                elif isinstance(r, dict):
                    normalized.append({
                        'document_id': r.get('document_id', 'unknown'),
                        'chunk_id': r.get('chunk_id', 'unknown'),
                        'title': r.get('title', r.get('document_title', 'Untitled')),
                        'file_type': r.get('file_type', 'unknown'),
                        'content': r.get('content', ''),
                        'score': r.get('score', r.get('similarity', 0.5)),
                        'section_heading': r.get('section_heading'),
                        'created_at': r.get('created_at')
                    })
            except Exception as e:
                logger.warning(f"Error normalizing result: {e}")

        return normalized

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        min_score: Optional[float] = None,
        file_type: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered = results

        if min_score is not None:
            filtered = [r for r in filtered if r.get('score', 0) >= min_score]

        if file_type:
            filtered = [
                r for r in filtered
                if r.get('file_type', '').lower() == file_type.lower()
            ]

        # TODO: Implement date filtering when created_at is available

        return filtered

    def _to_search_result(self, data: Dict[str, Any]) -> SearchResult:
        """Convert normalized dict to SearchResult"""
        content = data.get('content', '')
        preview = content[:300] + '...' if len(content) > 300 else content

        return SearchResult(
            document_id=data.get('document_id', 'unknown'),
            chunk_id=data.get('chunk_id', 'unknown'),
            title=data.get('title', 'Untitled'),
            file_type=data.get('file_type', 'unknown'),
            content_preview=preview,
            relevance_score=min(1.0, max(0.0, data.get('score', 0.5))),
            section_heading=data.get('section_heading'),
            highlights=self._extract_highlights(data.get('content', ''))
        )

    def _extract_highlights(self, content: str) -> List[str]:
        """Extract highlighted phrases from content"""
        # Simple implementation - return first few sentences
        sentences = content.split('.')[:2]
        return [s.strip() + '.' for s in sentences if s.strip()]
