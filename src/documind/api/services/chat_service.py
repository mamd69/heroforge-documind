"""
Chat Service

Business logic for chat/Q&A functionality.
Integrates with existing ProductionQA and HybridSearch.
"""

import logging
from typing import List, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for handling chat/Q&A operations.

    Integrates with:
    - ProductionQA for answer generation
    - HybridSearch for document retrieval
    - ConversationMemory for context management
    """

    def __init__(self):
        self._qa = None
        self._search = None

    @property
    def qa(self):
        """Lazy-load ProductionQA"""
        if self._qa is None:
            try:
                from ...rag.production_qa import ProductionQA
                self._qa = ProductionQA()
            except ImportError:
                logger.warning("ProductionQA not available, using fallback")
                self._qa = None
        return self._qa

    @property
    def search(self):
        """Lazy-load HybridSearcher"""
        if self._search is None:
            try:
                from ...hybrid_search import HybridSearcher
                self._searcher = HybridSearcher(semantic_weight=0.7)
                logger.info("HybridSearcher initialized successfully")
                # Create a search function wrapper for compatibility
                def search_func(query: str, top_k: int = 5, min_score: float = 0.5):
                    logger.info(f"Search wrapper called: query='{query[:50]}...', top_k={top_k}, min_score={min_score}")
                    results = self._searcher.search_hybrid(query, top_k=top_k)
                    logger.info(f"Raw search returned {len(results)} results")
                    for i, r in enumerate(results[:3]):
                        score = r.get('combined_score', r.get('semantic_score', 0))
                        logger.info(f"  Result {i}: combined_score={score}")
                    # Filter by minimum score
                    filtered = [r for r in results if r.get('combined_score', r.get('semantic_score', 0)) >= min_score]
                    logger.info(f"After filtering: {len(filtered)} results")
                    return filtered
                self._search = search_func
            except ImportError as e:
                logger.warning(f"HybridSearch not available: {e}, using fallback")
                self._search = None
            except Exception as e:
                logger.error(f"Error initializing HybridSearch: {e}")
                self._search = None
        return self._search

    async def process_message(
        self,
        message: str,
        conversation_id: str,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            message: User's question
            conversation_id: ID of the conversation
            history: Previous messages in the conversation

        Returns:
            Dict containing:
            - content: The generated answer
            - citations: List of source citations
            - suggested_questions: Follow-up suggestions
            - metrics: Performance metrics
        """
        start_time = time.time()
        search_time = 0
        generation_time = 0
        chunks_retrieved = 0

        try:
            # Step 1: Search for relevant context
            search_start = time.time()
            context_chunks = []

            if self.search:
                try:
                    search_results = self.search(
                        query=message,
                        top_k=5,
                        min_score=0.4  # Lowered to catch more semantic matches
                    )
                    context_chunks = search_results if search_results else []
                    chunks_retrieved = len(context_chunks)
                    logger.info(f"Initial search found {chunks_retrieved} chunks")
                except Exception as e:
                    logger.error(f"Search failed: {e}")
            else:
                logger.warning("No search function available")

            search_time = (time.time() - search_start) * 1000

            # Step 2: Generate answer using ProductionQA
            gen_start = time.time()

            if self.qa:
                try:
                    # ProductionQA.query() does its own search internally
                    # Use use_hybrid=True to leverage our hybrid search
                    result = self.qa.query(
                        question=message,
                        top_k=5,
                        use_hybrid=True,
                        include_sources=True
                    )

                    # ProductionQA returns a dict with 'answer' and 'sources'
                    if isinstance(result, dict):
                        content = result.get('answer', str(result))
                        sources = result.get('sources', [])
                        citations = self._extract_citations(sources) if sources else []
                        chunks_retrieved = len(sources)
                    else:
                        content = str(result)
                        citations = []

                except Exception as e:
                    logger.error(f"QA generation failed: {e}")
                    # Fallback to just returning context if ProductionQA fails
                    content = self._generate_fallback_response(message, context_chunks)
                    citations = self._extract_citations(context_chunks) if context_chunks else []
            elif context_chunks:
                # No QA but we have search results - show what we found
                content = self._generate_context_response(message, context_chunks)
                citations = self._extract_citations(context_chunks)
            else:
                # Fallback when no services available
                content = self._generate_fallback_response(message, [])
                citations = []

            generation_time = (time.time() - gen_start) * 1000

            # Step 3: Generate suggested follow-up questions
            suggested = self._generate_suggestions(message, content)

            total_time = (time.time() - start_time) * 1000

            return {
                "content": content,
                "citations": citations,
                "suggested_questions": suggested,
                "metrics": {
                    "search_time_ms": round(search_time, 2),
                    "generation_time_ms": round(generation_time, 2),
                    "total_time_ms": round(total_time, 2),
                    "chunks_retrieved": chunks_retrieved,
                    "model_used": self._get_model_name()
                }
            }

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            raise

    def _extract_citations(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Extract citation information from chunks/sources"""
        citations = []

        for chunk in chunks[:5]:  # Limit to top 5 citations
            try:
                if isinstance(chunk, dict):
                    # Handle search results and ProductionQA source format
                    # Search results have: id, document_id, content, document_name, section_heading, semantic_score, metadata
                    # ProductionQA has: id, citation_number, document, chunk_index, similarity, link, preview, was_cited
                    document_name = chunk.get('document_name', chunk.get('document', chunk.get('title', 'Unknown Document')))
                    preview = chunk.get('content', chunk.get('preview', chunk.get('content_preview', '')))

                    citation = {
                        "document_id": chunk.get('id', chunk.get('document_id', 'unknown')),
                        "document_title": document_name,
                        "chunk_id": str(chunk.get('chunk_index', chunk.get('chunk_id', 'unknown'))),
                        "content_preview": self._truncate(preview, 200) if preview else "",
                        "relevance_score": round(chunk.get('semantic_score', chunk.get('combined_score', chunk.get('similarity', chunk.get('score', 0.8)))), 3),
                        "link": chunk.get('link', ''),
                        "citation_number": chunk.get('citation_number'),
                        "section_heading": chunk.get('section_heading')
                    }
                elif hasattr(chunk, '__dict__'):
                    citation = {
                        "document_id": getattr(chunk, 'id', getattr(chunk, 'document_id', 'unknown')),
                        "document_title": getattr(chunk, 'document', getattr(chunk, 'document_title', 'Unknown Document')),
                        "chunk_id": str(getattr(chunk, 'chunk_index', getattr(chunk, 'chunk_id', 'unknown'))),
                        "content_preview": self._truncate(
                            getattr(chunk, 'preview', getattr(chunk, 'content', ''))[:200]
                        ),
                        "relevance_score": round(getattr(chunk, 'similarity', getattr(chunk, 'score', 0.8)), 3)
                    }
                else:
                    continue

                citations.append(citation)

            except Exception as e:
                logger.warning(f"Error extracting citation: {e}")

        return citations

    def _generate_context_response(
        self,
        message: str,
        context: List[Any]
    ) -> str:
        """Generate a response from context chunks when QA is unavailable"""
        if not context:
            return self._generate_fallback_response(message, context)

        # Build response from the first few chunks
        response_parts = [
            f"Based on your documents, I found {len(context)} relevant section(s) about your question:\n"
        ]

        for i, chunk in enumerate(context[:3], 1):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')[:300]
            elif hasattr(chunk, 'content'):
                content = getattr(chunk, 'content', '')[:300]
            else:
                content = str(chunk)[:300]

            if content:
                response_parts.append(f"\n**Source {i}:**\n{content}...\n")

        return "".join(response_parts)

    def _generate_fallback_response(
        self,
        message: str,
        context: List[Any]
    ) -> str:
        """Generate a fallback response when full pipeline isn't available"""
        if context:
            return (
                f"Based on your documents, I found {len(context)} relevant sections "
                f"related to your question about '{message[:50]}...'. "
                "The full AI-powered answer generation will provide more detailed "
                "responses once the service is fully configured."
            )
        else:
            return (
                f"I received your question: '{message}'. "
                "To provide accurate answers, please ensure documents are uploaded "
                "and the search service is properly configured. "
                "You can upload documents using the Documents section."
            )

    def _generate_suggestions(
        self,
        question: str,
        answer: str
    ) -> List[str]:
        """Generate suggested follow-up questions"""
        # Simple rule-based suggestions
        # TODO: Use LLM to generate contextual suggestions

        suggestions = []

        # Add generic follow-ups based on question type
        if "what" in question.lower():
            suggestions.append("Can you provide more details?")
            suggestions.append("Are there any related topics?")
        elif "how" in question.lower():
            suggestions.append("What are the steps involved?")
            suggestions.append("Are there any prerequisites?")
        elif "why" in question.lower():
            suggestions.append("What are the benefits?")
            suggestions.append("Are there alternatives?")
        else:
            suggestions.append("Can you elaborate on this?")
            suggestions.append("What else should I know?")

        suggestions.append("What documents contain this information?")

        return suggestions[:3]

    def _get_model_name(self) -> str:
        """Get the name of the model being used"""
        if self.qa and hasattr(self.qa, 'model'):
            return self.qa.model
        return "gemini-2.5-flash-lite"

    @staticmethod
    def _truncate(text: str, max_length: int = 200) -> str:
        """Truncate text to max length with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
