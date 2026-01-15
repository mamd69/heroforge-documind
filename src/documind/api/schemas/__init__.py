"""
Pydantic schemas for API request/response validation.
"""

from .chat import (
    ChatRequest,
    ChatResponse,
    Message,
    Conversation,
    ConversationSummary,
    Citation,
    FeedbackRequest,
    ResponseMetrics,
)

from .documents import (
    DocumentUploadResponse,
    Document,
    DocumentList,
    DocumentChunk,
    DocumentMetadata,
)

from .search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchMode,
)

__all__ = [
    # Chat
    "ChatRequest",
    "ChatResponse",
    "Message",
    "Conversation",
    "ConversationSummary",
    "Citation",
    "FeedbackRequest",
    "ResponseMetrics",
    # Documents
    "DocumentUploadResponse",
    "Document",
    "DocumentList",
    "DocumentChunk",
    "DocumentMetadata",
    # Search
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchMode",
]
