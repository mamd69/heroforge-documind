"""
Chat API Routes

Endpoints for conversation management and Q&A with documents.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
from uuid import uuid4
from datetime import datetime

from ..schemas.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    Conversation,
    ConversationSummary,
    Citation,
    FeedbackRequest,
    ResponseMetrics,
    MessageRole,
)

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)

# In-memory storage for development
# TODO: Replace with database service
_conversations: dict[str, Conversation] = {}
_messages: dict[str, List[Message]] = {}


@router.post(
    "/conversations",
    response_model=Conversation,
    status_code=status.HTTP_201_CREATED,
    summary="Create new conversation",
    description="Start a new conversation session for Q&A."
)
async def create_conversation():
    """Create a new conversation"""
    conv_id = str(uuid4())
    now = datetime.utcnow()

    conversation = Conversation(
        id=conv_id,
        created_at=now,
        updated_at=now,
        message_count=0,
        messages=[]
    )

    _conversations[conv_id] = conversation
    _messages[conv_id] = []

    return conversation


@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List conversations",
    description="Get a list of all conversation sessions."
)
async def list_conversations(limit: int = 20):
    """List all conversations"""
    conversations = list(_conversations.values())[:limit]

    return [
        ConversationSummary(
            id=c.id,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=c.message_count,
            preview=_get_preview(c.id)
        )
        for c in conversations
    ]


@router.get(
    "/conversations/{conv_id}",
    response_model=Conversation,
    summary="Get conversation",
    description="Get details of a specific conversation."
)
async def get_conversation(conv_id: str):
    """Get a specific conversation"""
    if conv_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    conv = _conversations[conv_id]
    conv.messages = _messages.get(conv_id, [])
    return conv


@router.post(
    "/conversations/{conv_id}/messages",
    response_model=ChatResponse,
    summary="Send message",
    description="Send a message to the conversation and receive an AI response."
)
async def send_message(conv_id: str, request: ChatRequest):
    """Send a message and get AI response"""
    if conv_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    now = datetime.utcnow()

    # Store user message
    user_message = Message(
        id=str(uuid4()),
        role=MessageRole.USER,
        content=request.message,
        created_at=now
    )

    if conv_id not in _messages:
        _messages[conv_id] = []
    _messages[conv_id].append(user_message)

    # TODO: Integrate with ProductionQA service
    # For now, return placeholder response
    from ..services.chat_service import ChatService
    service = ChatService()

    try:
        result = await service.process_message(
            message=request.message,
            conversation_id=conv_id,
            history=_messages[conv_id][:-1]  # Exclude current message
        )
    except Exception as e:
        # Fallback response if service fails
        result = {
            "content": f"I received your question: '{request.message}'. The full RAG pipeline will be integrated soon.",
            "citations": [],
            "suggested_questions": [
                "What documents are available?",
                "How do I upload new documents?",
                "Can you summarize my documents?"
            ],
            "metrics": {
                "search_time_ms": 0,
                "generation_time_ms": 0,
                "total_time_ms": 0,
                "chunks_retrieved": 0,
                "model_used": "placeholder"
            }
        }

    # Create assistant message
    assistant_message_id = str(uuid4())
    citations = [Citation(**c) for c in result.get("citations", [])]

    assistant_message = Message(
        id=assistant_message_id,
        role=MessageRole.ASSISTANT,
        content=result["content"],
        created_at=datetime.utcnow(),
        citations=citations
    )
    _messages[conv_id].append(assistant_message)

    # Update conversation metadata
    _conversations[conv_id].updated_at = datetime.utcnow()
    _conversations[conv_id].message_count = len(_messages[conv_id])

    return ChatResponse(
        message_id=assistant_message_id,
        content=result["content"],
        citations=citations,
        suggested_questions=result.get("suggested_questions", []),
        metrics=ResponseMetrics(**result["metrics"]) if result.get("metrics") else None
    )


@router.get(
    "/conversations/{conv_id}/messages",
    response_model=List[Message],
    summary="Get conversation history",
    description="Get all messages in a conversation."
)
async def get_messages(conv_id: str):
    """Get conversation message history"""
    if conv_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    return _messages.get(conv_id, [])


@router.post(
    "/messages/{message_id}/feedback",
    summary="Submit feedback",
    description="Submit feedback (rating) for an AI response."
)
async def submit_feedback(message_id: str, request: FeedbackRequest):
    """Submit feedback for a message"""
    # TODO: Store feedback in database
    # For now, just acknowledge receipt

    return {
        "status": "received",
        "message_id": message_id,
        "rating": request.rating
    }


@router.delete(
    "/conversations/{conv_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete conversation",
    description="Delete a conversation and all its messages."
)
async def delete_conversation(conv_id: str):
    """Delete a conversation"""
    if conv_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    del _conversations[conv_id]
    if conv_id in _messages:
        del _messages[conv_id]


def _get_preview(conv_id: str) -> str | None:
    """Get preview of first user message"""
    messages = _messages.get(conv_id, [])
    for msg in messages:
        if msg.role == MessageRole.USER:
            return msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
    return None
