"""
Chat API Schemas

Pydantic models for chat/conversation endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Citation(BaseModel):
    """Source citation for an answer"""
    document_id: str = Field(..., description="UUID of source document")
    document_title: str = Field(..., description="Title of source document")
    chunk_id: str = Field(..., description="UUID of specific chunk")
    content_preview: str = Field(..., description="Preview of cited content", max_length=500)
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score 0-1")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_title": "Company Policies.pdf",
                "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                "content_preview": "The vacation policy allows for 15 days of PTO per year...",
                "relevance_score": 0.92
            }
        }


class ResponseMetrics(BaseModel):
    """Performance metrics for a response"""
    search_time_ms: float = Field(..., description="Time spent searching in milliseconds")
    generation_time_ms: float = Field(..., description="Time spent generating response")
    total_time_ms: float = Field(..., description="Total processing time")
    chunks_retrieved: int = Field(..., description="Number of chunks used for context")
    model_used: str = Field(..., description="LLM model used for generation")


class ChatRequest(BaseModel):
    """Request to send a chat message"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question or message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the vacation policy?"
            }
        }


class ChatResponse(BaseModel):
    """Response from the chat endpoint"""
    message_id: str = Field(..., description="UUID of this message")
    content: str = Field(..., description="AI-generated response")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions",
        max_length=5
    )
    metrics: Optional[ResponseMetrics] = Field(None, description="Performance metrics")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "770e8400-e29b-41d4-a716-446655440002",
                "content": "According to the company policies, you are entitled to 15 days of paid time off (PTO) per year...",
                "citations": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_title": "Company Policies.pdf",
                        "chunk_id": "660e8400-e29b-41d4-a716-446655440001",
                        "content_preview": "The vacation policy allows for 15 days...",
                        "relevance_score": 0.92
                    }
                ],
                "suggested_questions": [
                    "How do I request time off?",
                    "Can I carry over unused PTO?",
                    "What is the sick leave policy?"
                ]
            }
        }


class Message(BaseModel):
    """A single message in a conversation"""
    id: str = Field(..., description="Message UUID")
    role: MessageRole = Field(..., description="Who sent the message")
    content: str = Field(..., description="Message content")
    created_at: datetime = Field(..., description="When message was created")
    citations: Optional[List[Citation]] = Field(None, description="Citations (for assistant messages)")
    metrics: Optional[ResponseMetrics] = Field(None, description="Response metrics")


class Conversation(BaseModel):
    """A conversation with messages"""
    id: str = Field(..., description="Conversation UUID")
    created_at: datetime = Field(..., description="When conversation started")
    updated_at: datetime = Field(..., description="When last message was added")
    message_count: int = Field(default=0, description="Number of messages")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing"""
    id: str = Field(..., description="Conversation UUID")
    created_at: datetime = Field(..., description="When conversation started")
    updated_at: datetime = Field(..., description="When last message was added")
    message_count: int = Field(default=0, description="Number of messages")
    preview: Optional[str] = Field(None, description="First message preview")


class FeedbackRequest(BaseModel):
    """User feedback on a response"""
    rating: int = Field(
        ...,
        ge=-1,
        le=1,
        description="Rating: -1 (thumbs down), 0 (neutral), 1 (thumbs up)"
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional feedback comment"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rating": 1,
                "comment": "Very helpful answer!"
            }
        }
