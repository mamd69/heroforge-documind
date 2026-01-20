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
        """Lazy-load HybridSearch"""
        if self._search is None:
            try:
                from ...hybrid_search import hybrid_search
                self._search = hybrid_search
            except ImportError:
                logger.warning("HybridSearch not available, using fallback")
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
                        min_score=0.6
                    )
                    context_chunks = search_results if search_results else []
                    chunks_retrieved = len(context_chunks)
                except Exception as e:
                    logger.error(f"Search failed: {e}")

            search_time = (time.time() - search_start) * 1000

            # Step 2: Generate answer using ProductionQA
            gen_start = time.time()

            if self.qa and context_chunks:
                try:
                    result = self.qa.query(
                        question=message,
                        context=context_chunks
                    )

                    content = result.answer if hasattr(result, 'answer') else str(result)
                    citations = self._extract_citations(context_chunks)

                except Exception as e:
                    logger.error(f"QA generation failed: {e}")
                    content = self._generate_fallback_response(message, context_chunks)
                    citations = []
            else:
                # Fallback when services aren't available
                content = self._generate_fallback_response(message, context_chunks)
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
        """Extract citation information from chunks"""
        citations = []

        for chunk in chunks[:3]:  # Limit to top 3 citations
            try:
                if hasattr(chunk, '__dict__'):
                    citation = {
                        "document_id": getattr(chunk, 'document_id', 'unknown'),
                        "document_title": getattr(chunk, 'document_title', 'Unknown Document'),
                        "chunk_id": getattr(chunk, 'chunk_id', 'unknown'),
                        "content_preview": self._truncate(
                            getattr(chunk, 'content', '')[:200]
                        ),
                        "relevance_score": getattr(chunk, 'score', 0.8)
                    }
                elif isinstance(chunk, dict):
                    citation = {
                        "document_id": chunk.get('document_id', 'unknown'),
                        "document_title": chunk.get('title', 'Unknown Document'),
                        "chunk_id": chunk.get('chunk_id', 'unknown'),
                        "content_preview": self._truncate(
                            chunk.get('content', '')[:200]
                        ),
                        "relevance_score": chunk.get('score', 0.8)
                    }
                else:
                    continue

                citations.append(citation)

            except Exception as e:
                logger.warning(f"Error extracting citation: {e}")

        return citations

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
