"""
TDD Tests for Chat API Routes

Following Red-Green-Refactor cycle:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Clean up and integrate with real services

Run with: pytest tests/api/test_chat_routes.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import uuid


class TestCreateConversation:
    """Tests for POST /api/chat/conversations"""

    def test_create_conversation_returns_201(self, client):
        """Creating a new conversation returns 201 status"""
        response = client.post("/api/chat/conversations")

        assert response.status_code == 201

    def test_create_conversation_returns_id(self, client):
        """New conversation includes a UUID id"""
        response = client.post("/api/chat/conversations")
        data = response.json()

        assert "id" in data
        # Validate it's a valid UUID
        uuid.UUID(data["id"])

    def test_create_conversation_returns_timestamp(self, client):
        """New conversation includes created_at timestamp"""
        response = client.post("/api/chat/conversations")
        data = response.json()

        assert "created_at" in data
        assert isinstance(data["created_at"], str)

    def test_create_conversation_initializes_empty_messages(self, client):
        """New conversation starts with no messages"""
        response = client.post("/api/chat/conversations")
        data = response.json()

        assert data.get("message_count", 0) == 0


class TestListConversations:
    """Tests for GET /api/chat/conversations"""

    def test_list_conversations_returns_200(self, client):
        """Listing conversations returns 200 status"""
        response = client.get("/api/chat/conversations")

        assert response.status_code == 200

    def test_list_conversations_returns_array(self, client):
        """Conversations list is an array"""
        response = client.get("/api/chat/conversations")
        data = response.json()

        assert isinstance(data, list)

    def test_list_conversations_includes_created(self, client):
        """Listed conversations include ones we created"""
        # Create a conversation
        create_response = client.post("/api/chat/conversations")
        conv_id = create_response.json()["id"]

        # List and find it
        list_response = client.get("/api/chat/conversations")
        ids = [c["id"] for c in list_response.json()]

        assert conv_id in ids

    def test_list_conversations_respects_limit(self, client):
        """Limit parameter restricts result count"""
        # Create several conversations
        for _ in range(5):
            client.post("/api/chat/conversations")

        response = client.get("/api/chat/conversations?limit=2")
        data = response.json()

        assert len(data) <= 2


class TestSendMessage:
    """Tests for POST /api/chat/conversations/{id}/messages"""

    def test_send_message_returns_200(self, client, conversation_id):
        """Sending a message returns 200 status"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "Hello, DocuMind!"}
        )

        assert response.status_code == 200

    def test_send_message_returns_message_id(self, client, conversation_id):
        """Response includes a message_id"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "What is DocuMind?"}
        )
        data = response.json()

        assert "message_id" in data
        uuid.UUID(data["message_id"])

    def test_send_message_returns_content(self, client, conversation_id):
        """Response includes non-empty content"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "Tell me about documents"}
        )
        data = response.json()

        assert "content" in data
        assert len(data["content"]) > 0

    def test_send_message_returns_citations(self, client, conversation_id):
        """Response includes citations array"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "What documents are available?"}
        )
        data = response.json()

        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_send_message_returns_suggested_questions(self, client, conversation_id):
        """Response includes suggested follow-up questions"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "How do I upload files?"}
        )
        data = response.json()

        assert "suggested_questions" in data
        assert isinstance(data["suggested_questions"], list)

    def test_send_empty_message_returns_422(self, client, conversation_id):
        """Empty message returns validation error"""
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": ""}
        )

        assert response.status_code == 422

    def test_send_message_too_long_returns_422(self, client, conversation_id):
        """Message over 2000 chars returns validation error"""
        long_message = "x" * 2001
        response = client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": long_message}
        )

        assert response.status_code == 422

    def test_send_message_to_nonexistent_conversation_returns_404(self, client):
        """Sending to non-existent conversation returns 404"""
        fake_id = str(uuid.uuid4())
        response = client.post(
            f"/api/chat/conversations/{fake_id}/messages",
            json={"message": "Hello?"}
        )

        assert response.status_code == 404


class TestGetConversationHistory:
    """Tests for GET /api/chat/conversations/{id}/messages"""

    def test_get_history_returns_200(self, client, conversation_id):
        """Getting history returns 200 status"""
        response = client.get(
            f"/api/chat/conversations/{conversation_id}/messages"
        )

        assert response.status_code == 200

    def test_get_history_returns_array(self, client, conversation_id):
        """History is an array of messages"""
        response = client.get(
            f"/api/chat/conversations/{conversation_id}/messages"
        )
        data = response.json()

        assert isinstance(data, list)

    def test_get_history_includes_sent_messages(self, client, conversation_id):
        """History includes messages we sent"""
        # Send a message
        client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "Test message"}
        )

        # Get history
        response = client.get(
            f"/api/chat/conversations/{conversation_id}/messages"
        )
        messages = response.json()

        # Should have user message and assistant response
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_get_history_messages_have_required_fields(self, client, conversation_id):
        """Each message has role, content, created_at"""
        client.post(
            f"/api/chat/conversations/{conversation_id}/messages",
            json={"message": "Hello"}
        )

        response = client.get(
            f"/api/chat/conversations/{conversation_id}/messages"
        )
        messages = response.json()

        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert "created_at" in msg

    def test_get_history_nonexistent_conversation_returns_404(self, client):
        """Getting history of non-existent conversation returns 404"""
        fake_id = str(uuid.uuid4())
        response = client.get(
            f"/api/chat/conversations/{fake_id}/messages"
        )

        assert response.status_code == 404


class TestFeedback:
    """Tests for POST /api/chat/messages/{id}/feedback"""

    def test_submit_feedback_returns_200(self, client, message_with_response):
        """Submitting feedback returns 200"""
        message_id = message_with_response["message_id"]

        response = client.post(
            f"/api/chat/messages/{message_id}/feedback",
            json={"rating": 1, "comment": "Helpful!"}
        )

        assert response.status_code == 200

    def test_submit_thumbs_up_feedback(self, client, message_with_response):
        """Thumbs up (rating=1) is accepted"""
        message_id = message_with_response["message_id"]

        response = client.post(
            f"/api/chat/messages/{message_id}/feedback",
            json={"rating": 1}
        )

        assert response.status_code == 200

    def test_submit_thumbs_down_feedback(self, client, message_with_response):
        """Thumbs down (rating=-1) is accepted"""
        message_id = message_with_response["message_id"]

        response = client.post(
            f"/api/chat/messages/{message_id}/feedback",
            json={"rating": -1}
        )

        assert response.status_code == 200

    def test_invalid_rating_returns_422(self, client, message_with_response):
        """Invalid rating value returns 422"""
        message_id = message_with_response["message_id"]

        response = client.post(
            f"/api/chat/messages/{message_id}/feedback",
            json={"rating": 5}  # Only -1, 0, 1 allowed
        )

        assert response.status_code == 422


# Fixtures

@pytest.fixture
def client():
    """Create test client for API"""
    from src.documind.api.main import app
    return TestClient(app)


@pytest.fixture
def conversation_id(client):
    """Create a conversation and return its ID"""
    response = client.post("/api/chat/conversations")
    return response.json()["id"]


@pytest.fixture
def message_with_response(client, conversation_id):
    """Send a message and return the response"""
    response = client.post(
        f"/api/chat/conversations/{conversation_id}/messages",
        json={"message": "Test question for feedback"}
    )
    return response.json()
