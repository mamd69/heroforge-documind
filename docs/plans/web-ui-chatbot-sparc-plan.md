# DocuMind Web UI Chatbot - SPARC Development Plan

## Executive Summary

Transform DocuMind from a CLI-only tool into a full-featured web application with a conversational chatbot interface. This plan follows SPARC methodology (Specification, Pseudocode, Architecture, Refinement, Completion) with Test-Driven Development (TDD) practices.

**Target**: Production-ready web UI chatbot for document Q&A

---

## Phase 1: SPECIFICATION

### 1.1 Functional Requirements

#### FR-1: Chat Interface
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1.1 | Users can type questions in natural language | P0 | Text input accepts min 1, max 2000 characters |
| FR-1.2 | System displays AI responses with citations | P0 | Response shows source documents with links |
| FR-1.3 | Conversation history persists in session | P0 | Previous messages visible on scroll |
| FR-1.4 | Users can start new conversations | P0 | Clear button resets chat state |
| FR-1.5 | Typing indicator shows when AI is processing | P1 | Animation displays during API call |
| FR-1.6 | Users can copy responses to clipboard | P1 | Copy button on each response |
| FR-1.7 | Markdown rendering for formatted responses | P1 | Code blocks, lists, tables render correctly |
| FR-1.8 | Mobile-responsive design | P1 | Usable on screens 320px+ |

#### FR-2: Document Management
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-2.1 | Upload documents via drag-and-drop | P0 | Accepts PDF, DOCX, CSV, XLSX, TXT, MD |
| FR-2.2 | View list of uploaded documents | P0 | Shows name, type, upload date, chunk count |
| FR-2.3 | Delete documents | P1 | Confirms before deletion |
| FR-2.4 | Search/filter documents | P2 | Filter by type, date, keyword |
| FR-2.5 | View document details/preview | P2 | Shows metadata and first chunk |

#### FR-3: Search & Discovery
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-3.1 | Semantic search across all documents | P0 | Returns top-5 relevant chunks |
| FR-3.2 | Hybrid search (semantic + keyword) | P1 | Toggle between search modes |
| FR-3.3 | Search results show relevance score | P1 | Percentage or score displayed |

#### FR-4: User Experience
| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-4.1 | Feedback mechanism (thumbs up/down) | P1 | Stores rating with response ID |
| FR-4.2 | Suggested follow-up questions | P2 | 3 related questions after response |
| FR-4.3 | Dark/light mode toggle | P2 | Persists preference |

### 1.2 Non-Functional Requirements

#### NFR-1: Performance
| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| NFR-1.1 | API response time (search) | < 500ms | P95 latency |
| NFR-1.2 | API response time (Q&A) | < 5s | P95 latency |
| NFR-1.3 | Document upload processing | < 30s/doc | Average time |
| NFR-1.4 | Frontend initial load | < 2s | First Contentful Paint |
| NFR-1.5 | Time to Interactive | < 3s | TTI metric |

#### NFR-2: Reliability
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-2.1 | API availability | 99.5% uptime |
| NFR-2.2 | Error handling | All errors return structured JSON |
| NFR-2.3 | Rate limiting | 60 requests/minute per IP |
| NFR-2.4 | Request timeout | 30s max |

#### NFR-3: Security
| ID | Requirement | Implementation |
|----|-------------|----------------|
| NFR-3.1 | Input validation | All endpoints validate payloads |
| NFR-3.2 | File upload validation | Type, size (50MB max), content checks |
| NFR-3.3 | CORS configuration | Allowlist frontend origins |
| NFR-3.4 | Rate limiting | Prevent abuse |

### 1.3 User Stories

```gherkin
Feature: Chat with Documents
  As a knowledge worker
  I want to ask questions about my documents
  So that I can quickly find information

  Scenario: Ask a simple question
    Given I have uploaded documents about "company policies"
    When I type "What is the vacation policy?"
    Then I receive an answer within 5 seconds
    And the answer includes citations to source documents

  Scenario: Follow-up question
    Given I asked "What is the vacation policy?"
    And received an answer
    When I ask "How do I request time off?"
    Then the system considers conversation context
    And provides a contextually relevant answer

Feature: Document Upload
  As a document administrator
  I want to upload company documents
  So that they become searchable

  Scenario: Upload a PDF
    Given I am on the upload page
    When I drag a PDF file into the upload zone
    Then I see upload progress
    And the document appears in the list when complete
    And it shows the number of chunks created
```

### 1.4 Constraints & Assumptions

**Constraints:**
- Must reuse existing Python backend (DocumentProcessor, ProductionQA)
- Must use Supabase as database (already configured)
- Must support same file formats as CLI
- Budget: Use free tier services where possible

**Assumptions:**
- Single-tenant initially (no multi-user auth in v1)
- Documents are not sensitive (no encryption at rest)
- Users have modern browsers (ES2020+)
- Network latency is acceptable (< 100ms to server)

---

## Phase 2: PSEUDOCODE

### 2.1 Chat Flow Algorithm

```
FUNCTION handle_chat_message(user_message, conversation_id):
    # Input validation
    IF length(user_message) < 1 OR length(user_message) > 2000:
        RETURN error("Message must be 1-2000 characters")

    # Get or create conversation
    conversation = get_or_create_conversation(conversation_id)

    # Store user message
    store_message(conversation.id, role="user", content=user_message)

    # Build context from conversation history
    history = get_recent_messages(conversation.id, limit=10)
    context_prompt = build_context_prompt(history, user_message)

    # Execute RAG pipeline
    TRY:
        # Semantic search for relevant chunks
        search_results = hybrid_search(
            query=user_message,
            top_k=5,
            min_score=0.7
        )

        # Generate answer with LLM
        response = production_qa.query(
            question=user_message,
            context_chunks=search_results,
            conversation_history=history
        )

        # Extract citations
        citations = extract_citations(response, search_results)

        # Store assistant message
        message_id = store_message(
            conversation.id,
            role="assistant",
            content=response.answer,
            metadata={
                "citations": citations,
                "model": response.model,
                "search_time_ms": response.search_time,
                "generation_time_ms": response.generation_time
            }
        )

        RETURN {
            "message_id": message_id,
            "content": response.answer,
            "citations": citations,
            "suggested_questions": generate_follow_ups(response)
        }

    CATCH error:
        log_error(error)
        RETURN error("Failed to process question. Please try again.")


FUNCTION hybrid_search(query, top_k, min_score):
    # Generate query embedding
    embedding = openai.embed(query, model="text-embedding-3-small")

    # Semantic search
    semantic_results = supabase.rpc(
        "match_documents",
        query_embedding=embedding,
        match_count=top_k * 2,
        threshold=min_score
    )

    # Keyword search for diversity
    keyword_results = supabase.rpc(
        "search_document_chunks_keyword",
        query=query,
        limit=top_k
    )

    # Merge and rerank
    combined = merge_results(semantic_results, keyword_results)
    reranked = rerank_by_relevance(combined, query)

    RETURN top_k(reranked)


FUNCTION build_context_prompt(history, current_message):
    system_prompt = """
    You are DocuMind, an AI assistant that helps users find information
    in their documents. Answer based on the provided context. If the
    answer isn't in the context, say so. Always cite sources.
    """

    messages = [{"role": "system", "content": system_prompt}]

    FOR message IN history:
        messages.append({
            "role": message.role,
            "content": message.content
        })

    RETURN messages
```

### 2.2 Document Upload Flow

```
FUNCTION handle_document_upload(file, filename):
    # Validate file
    IF file.size > MAX_FILE_SIZE (50MB):
        RETURN error("File too large. Maximum 50MB.")

    IF NOT is_supported_format(filename):
        RETURN error(f"Unsupported format. Supported: {FORMATS}")

    # Generate unique ID
    document_id = generate_uuid()
    temp_path = save_temp_file(file, document_id, filename)

    TRY:
        # Check for duplicates
        fingerprint = generate_fingerprint(temp_path)
        existing = find_by_fingerprint(fingerprint)

        IF existing:
            cleanup(temp_path)
            RETURN {
                "status": "duplicate",
                "existing_id": existing.id,
                "message": "Document already exists"
            }

        # Process document
        processor = DocumentProcessor()
        result = processor.process_document(temp_path, upload=True)

        # Generate embeddings for chunks
        chunks_with_embeddings = []
        FOR chunk IN result.chunks:
            embedding = openai.embed(chunk.content)
            chunks_with_embeddings.append({
                ...chunk,
                "embedding": embedding
            })

        # Store in database
        doc_id = store_document(result, chunks_with_embeddings)

        cleanup(temp_path)

        RETURN {
            "status": "success",
            "document_id": doc_id,
            "filename": filename,
            "chunks_created": len(result.chunks),
            "metadata": result.metadata
        }

    CATCH ProcessingError AS e:
        cleanup(temp_path)
        RETURN error(f"Processing failed: {e.message}")

    CATCH DatabaseError AS e:
        cleanup(temp_path)
        RETURN error("Failed to store document")
```

### 2.3 Frontend State Machine

```
STATE_MACHINE ChatInterface:

    STATES:
        IDLE          # Waiting for user input
        TYPING        # User is typing
        SUBMITTING    # Request sent, waiting for response
        STREAMING     # Receiving streamed response
        ERROR         # Error occurred

    INITIAL_STATE: IDLE

    TRANSITIONS:
        IDLE -> TYPING:
            TRIGGER: user_starts_typing
            ACTION: show_input_focus()

        TYPING -> SUBMITTING:
            TRIGGER: user_submits_message
            GUARD: message.length >= 1
            ACTION:
                add_user_message_to_chat()
                show_typing_indicator()
                send_request_to_api()

        SUBMITTING -> STREAMING:
            TRIGGER: response_starts
            ACTION:
                hide_typing_indicator()
                start_streaming_response()

        STREAMING -> IDLE:
            TRIGGER: response_complete
            ACTION:
                add_assistant_message_to_chat()
                scroll_to_bottom()
                focus_input()

        SUBMITTING -> ERROR:
            TRIGGER: request_fails
            ACTION:
                hide_typing_indicator()
                show_error_message()
                enable_retry_button()

        ERROR -> SUBMITTING:
            TRIGGER: user_clicks_retry
            ACTION: resend_last_request()

        ERROR -> IDLE:
            TRIGGER: user_dismisses_error
            ACTION: clear_error()
```

---

## Phase 3: ARCHITECTURE

### 3.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    React Application                        ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ ││
│  │  │ ChatPanel    │  │ DocumentList │  │ UploadZone       │ ││
│  │  │ - Messages   │  │ - Items      │  │ - Drag&Drop      │ ││
│  │  │ - Input      │  │ - Search     │  │ - Progress       │ ││
│  │  │ - Citations  │  │ - Actions    │  │ - Validation     │ ││
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ ││
│  │            │               │                  │            ││
│  │            └───────────────┼──────────────────┘            ││
│  │                            │                                ││
│  │                    ┌───────▼───────┐                       ││
│  │                    │  API Client   │                       ││
│  │                    │  (TanStack Q) │                       ││
│  │                    └───────────────┘                       ││
│  └─────────────────────────────────────────────────────────────┘│
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP/REST
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     API Routes                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ ││
│  │  │ /api/chat    │  │ /api/docs    │  │ /api/search      │ ││
│  │  │ - POST /ask  │  │ - GET /      │  │ - POST /query    │ ││
│  │  │ - GET /hist  │  │ - POST /     │  │ - GET /suggest   │ ││
│  │  │ - POST /fb   │  │ - DELETE /:id│  │                  │ ││
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ ││
│  │            │               │                  │            ││
│  │            └───────────────┼──────────────────┘            ││
│  │                            │                                ││
│  │                    ┌───────▼───────┐                       ││
│  │                    │ Service Layer │                       ││
│  │                    └───────────────┘                       ││
│  │            ┌───────────────┼───────────────┐               ││
│  │            ▼               ▼               ▼               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    ││
│  │  │ ProductionQA │  │ DocProcessor │  │ HybridSearch │    ││
│  │  │ (existing)   │  │ (existing)   │  │ (existing)   │    ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└───────────────────────────────┬─────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Supabase   │     │   OpenAI    │     │ OpenRouter  │
    │  PostgreSQL │     │ Embeddings  │     │    LLMs     │
    │  + pgvector │     │             │     │             │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### 3.2 Directory Structure

```
heroforge-documind/
├── src/
│   ├── documind/                    # Existing Python core
│   │   ├── ...                      # (unchanged)
│   │   └── api/                     # NEW: FastAPI layer
│   │       ├── __init__.py
│   │       ├── main.py              # FastAPI app entry
│   │       ├── routes/
│   │       │   ├── __init__.py
│   │       │   ├── chat.py          # Chat endpoints
│   │       │   ├── documents.py     # Document CRUD
│   │       │   └── search.py        # Search endpoints
│   │       ├── schemas/
│   │       │   ├── __init__.py
│   │       │   ├── chat.py          # Pydantic models
│   │       │   ├── documents.py
│   │       │   └── search.py
│   │       ├── services/
│   │       │   ├── __init__.py
│   │       │   ├── chat_service.py  # Business logic
│   │       │   ├── doc_service.py
│   │       │   └── search_service.py
│   │       └── middleware/
│   │           ├── __init__.py
│   │           ├── cors.py
│   │           ├── rate_limit.py
│   │           └── error_handler.py
│   │
│   └── web/                         # NEW: React frontend
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts
│       ├── index.html
│       ├── public/
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── components/
│           │   ├── chat/
│           │   │   ├── ChatPanel.tsx
│           │   │   ├── MessageList.tsx
│           │   │   ├── MessageBubble.tsx
│           │   │   ├── ChatInput.tsx
│           │   │   ├── TypingIndicator.tsx
│           │   │   └── CitationCard.tsx
│           │   ├── documents/
│           │   │   ├── DocumentList.tsx
│           │   │   ├── DocumentCard.tsx
│           │   │   ├── UploadZone.tsx
│           │   │   └── UploadProgress.tsx
│           │   ├── search/
│           │   │   ├── SearchBar.tsx
│           │   │   └── SearchResults.tsx
│           │   └── common/
│           │       ├── Layout.tsx
│           │       ├── Sidebar.tsx
│           │       ├── Header.tsx
│           │       └── ErrorBoundary.tsx
│           ├── hooks/
│           │   ├── useChat.ts
│           │   ├── useDocuments.ts
│           │   └── useSearch.ts
│           ├── api/
│           │   ├── client.ts
│           │   ├── chat.ts
│           │   ├── documents.ts
│           │   └── search.ts
│           ├── store/
│           │   └── chatStore.ts
│           ├── types/
│           │   └── index.ts
│           └── styles/
│               └── globals.css
│
├── tests/
│   ├── api/                         # NEW: API tests
│   │   ├── test_chat_routes.py
│   │   ├── test_document_routes.py
│   │   └── test_search_routes.py
│   └── web/                         # NEW: Frontend tests
│       ├── components/
│       │   ├── ChatPanel.test.tsx
│       │   └── DocumentList.test.tsx
│       └── hooks/
│           └── useChat.test.ts
```

### 3.3 API Contract Specification

```yaml
openapi: 3.0.3
info:
  title: DocuMind Web API
  version: 1.0.0

paths:
  /api/chat/conversations:
    post:
      summary: Create new conversation
      responses:
        201:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Conversation'
    get:
      summary: List conversations
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ConversationSummary'

  /api/chat/conversations/{id}/messages:
    post:
      summary: Send message and get response
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatRequest'
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatResponse'
    get:
      summary: Get conversation history
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Message'

  /api/documents:
    get:
      summary: List all documents
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentList'
    post:
      summary: Upload document
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
      responses:
        201:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UploadResult'

  /api/documents/{id}:
    get:
      summary: Get document details
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Document'
    delete:
      summary: Delete document
      responses:
        204:
          description: Deleted successfully

  /api/search:
    post:
      summary: Search documents
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SearchRequest'
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SearchResults'

components:
  schemas:
    ChatRequest:
      type: object
      required:
        - message
      properties:
        message:
          type: string
          minLength: 1
          maxLength: 2000

    ChatResponse:
      type: object
      properties:
        message_id:
          type: string
          format: uuid
        content:
          type: string
        citations:
          type: array
          items:
            $ref: '#/components/schemas/Citation'
        suggested_questions:
          type: array
          items:
            type: string
        metrics:
          $ref: '#/components/schemas/ResponseMetrics'

    Citation:
      type: object
      properties:
        document_id:
          type: string
        document_title:
          type: string
        chunk_id:
          type: string
        content_preview:
          type: string
        relevance_score:
          type: number

    Document:
      type: object
      properties:
        id:
          type: string
          format: uuid
        title:
          type: string
        file_type:
          type: string
        chunk_count:
          type: integer
        created_at:
          type: string
          format: date-time
        metadata:
          type: object

    SearchRequest:
      type: object
      required:
        - query
      properties:
        query:
          type: string
        mode:
          type: string
          enum: [semantic, keyword, hybrid]
          default: hybrid
        limit:
          type: integer
          default: 5

    SearchResults:
      type: object
      properties:
        results:
          type: array
          items:
            $ref: '#/components/schemas/SearchResult'
        total:
          type: integer
        search_time_ms:
          type: number
```

### 3.4 Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React 18 + TypeScript | Modern, typed, large ecosystem |
| **Build Tool** | Vite | Fast HMR, ESM-native |
| **Styling** | Tailwind CSS | Utility-first, rapid prototyping |
| **State** | Zustand | Lightweight, hooks-based |
| **Data Fetching** | TanStack Query | Caching, mutations, devtools |
| **Backend** | FastAPI | Async, auto-docs, Pydantic |
| **Database** | Supabase (existing) | Already configured |
| **Embeddings** | OpenAI (existing) | Already integrated |
| **LLM** | OpenRouter (existing) | Multi-model support |

---

## Phase 4: REFINEMENT (TDD Implementation)

### 4.1 TDD Red-Green-Refactor Cycles

#### Cycle 1: API Chat Endpoint

**RED - Write Failing Test**
```python
# tests/api/test_chat_routes.py
import pytest
from fastapi.testclient import TestClient
from src.documind.api.main import app

client = TestClient(app)

class TestChatEndpoints:

    def test_create_conversation_returns_201(self):
        """POST /api/chat/conversations creates new conversation"""
        response = client.post("/api/chat/conversations")
        assert response.status_code == 201
        assert "id" in response.json()
        assert "created_at" in response.json()

    def test_send_message_returns_response(self):
        """POST /api/chat/conversations/{id}/messages returns AI response"""
        # Create conversation first
        conv = client.post("/api/chat/conversations").json()

        response = client.post(
            f"/api/chat/conversations/{conv['id']}/messages",
            json={"message": "What is DocuMind?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data
        assert "content" in data
        assert len(data["content"]) > 0

    def test_send_empty_message_returns_422(self):
        """Empty message returns validation error"""
        conv = client.post("/api/chat/conversations").json()

        response = client.post(
            f"/api/chat/conversations/{conv['id']}/messages",
            json={"message": ""}
        )

        assert response.status_code == 422

    def test_send_message_includes_citations(self):
        """Response includes source citations"""
        conv = client.post("/api/chat/conversations").json()

        response = client.post(
            f"/api/chat/conversations/{conv['id']}/messages",
            json={"message": "What documents do you have?"}
        )

        data = response.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)
```

**GREEN - Implement Minimally**
```python
# src/documind/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    message_id: str
    content: str
    citations: list = []
    suggested_questions: list = []

conversations = {}  # In-memory for now

@router.post("/conversations", status_code=201)
async def create_conversation():
    conv_id = str(uuid4())
    conversations[conv_id] = {
        "id": conv_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    return conversations[conv_id]

@router.post("/conversations/{conv_id}/messages")
async def send_message(conv_id: str, request: ChatRequest):
    if conv_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # TODO: Integrate with ProductionQA
    response = ChatResponse(
        message_id=str(uuid4()),
        content="This is a placeholder response.",
        citations=[],
        suggested_questions=[]
    )
    return response
```

**REFACTOR - Integrate Real Services**
```python
# src/documind/api/services/chat_service.py
from src.documind.rag.production_qa import ProductionQA
from src.documind.hybrid_search import hybrid_search

class ChatService:
    def __init__(self):
        self.qa = ProductionQA()

    async def process_message(self, message: str, history: list) -> dict:
        # Search for relevant context
        search_results = hybrid_search(message, top_k=5)

        # Generate response
        response = self.qa.query(
            question=message,
            context=search_results
        )

        return {
            "content": response.answer,
            "citations": self._extract_citations(search_results),
            "metrics": {
                "search_time_ms": response.search_time,
                "generation_time_ms": response.generation_time
            }
        }
```

#### Cycle 2: Document Upload Endpoint

**RED - Write Failing Test**
```python
# tests/api/test_document_routes.py
import pytest
from fastapi.testclient import TestClient
from io import BytesIO

class TestDocumentEndpoints:

    def test_upload_pdf_returns_201(self, client):
        """POST /api/documents uploads PDF successfully"""
        pdf_content = b"%PDF-1.4 test content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert "chunks_created" in data
        assert data["chunks_created"] > 0

    def test_upload_unsupported_format_returns_400(self, client):
        """Unsupported file format returns error"""
        files = {"file": ("test.exe", BytesIO(b"binary"), "application/octet-stream")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_upload_large_file_returns_413(self, client):
        """File over 50MB returns error"""
        large_content = b"x" * (51 * 1024 * 1024)  # 51MB
        files = {"file": ("large.pdf", BytesIO(large_content), "application/pdf")}

        response = client.post("/api/documents", files=files)

        assert response.status_code == 413

    def test_list_documents_returns_paginated(self, client):
        """GET /api/documents returns paginated list"""
        response = client.get("/api/documents?page=1&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data

    def test_delete_document_returns_204(self, client):
        """DELETE /api/documents/{id} removes document"""
        # Upload first
        files = {"file": ("test.txt", BytesIO(b"test"), "text/plain")}
        upload = client.post("/api/documents", files=files).json()

        # Delete
        response = client.delete(f"/api/documents/{upload['document_id']}")

        assert response.status_code == 204

        # Verify deleted
        get_response = client.get(f"/api/documents/{upload['document_id']}")
        assert get_response.status_code == 404
```

#### Cycle 3: React Chat Component

**RED - Write Failing Test**
```typescript
// tests/web/components/ChatPanel.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ChatPanel } from '@/components/chat/ChatPanel';
import { vi } from 'vitest';

const queryClient = new QueryClient();

const wrapper = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    {children}
  </QueryClientProvider>
);

describe('ChatPanel', () => {

  it('renders input field and send button', () => {
    render(<ChatPanel />, { wrapper });

    expect(screen.getByPlaceholderText(/ask a question/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('disables send button when input is empty', () => {
    render(<ChatPanel />, { wrapper });

    const sendButton = screen.getByRole('button', { name: /send/i });
    expect(sendButton).toBeDisabled();
  });

  it('sends message when form submitted', async () => {
    const mockSend = vi.fn().mockResolvedValue({
      message_id: '123',
      content: 'Test response'
    });

    render(<ChatPanel onSendMessage={mockSend} />, { wrapper });

    const input = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.submit(screen.getByRole('form'));

    await waitFor(() => {
      expect(mockSend).toHaveBeenCalledWith('Test question');
    });
  });

  it('shows typing indicator while waiting for response', async () => {
    render(<ChatPanel />, { wrapper });

    const input = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.change(input, { target: { value: 'Question' } });
    fireEvent.submit(screen.getByRole('form'));

    expect(await screen.findByTestId('typing-indicator')).toBeInTheDocument();
  });

  it('displays user message in chat history', async () => {
    render(<ChatPanel />, { wrapper });

    const input = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.change(input, { target: { value: 'My question' } });
    fireEvent.submit(screen.getByRole('form'));

    expect(await screen.findByText('My question')).toBeInTheDocument();
  });

  it('displays assistant response with citations', async () => {
    const mockResponse = {
      message_id: '123',
      content: 'The answer is 42.',
      citations: [
        { document_title: 'Guide.pdf', content_preview: '...' }
      ]
    };

    render(<ChatPanel initialMessages={[mockResponse]} />, { wrapper });

    expect(screen.getByText('The answer is 42.')).toBeInTheDocument();
    expect(screen.getByText(/Guide.pdf/)).toBeInTheDocument();
  });
});
```

**GREEN - Implement Component**
```typescript
// src/web/src/components/chat/ChatPanel.tsx
import { useState, FormEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { TypingIndicator } from './TypingIndicator';
import { sendMessage, getMessages } from '@/api/chat';

interface ChatPanelProps {
  conversationId?: string;
}

export function ChatPanel({ conversationId }: ChatPanelProps) {
  const [input, setInput] = useState('');

  const { data: messages = [] } = useQuery({
    queryKey: ['messages', conversationId],
    queryFn: () => getMessages(conversationId!),
    enabled: !!conversationId
  });

  const mutation = useMutation({
    mutationFn: (message: string) => sendMessage(conversationId!, message),
    onSuccess: () => setInput('')
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      mutation.mutate(input);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} />

      {mutation.isPending && <TypingIndicator data-testid="typing-indicator" />}

      <form onSubmit={handleSubmit} role="form" className="p-4 border-t">
        <ChatInput
          value={input}
          onChange={setInput}
          disabled={mutation.isPending}
        />
        <button
          type="submit"
          disabled={!input.trim() || mutation.isPending}
          className="btn-primary"
        >
          Send
        </button>
      </form>
    </div>
  );
}
```

### 4.2 Test Coverage Targets

| Component | Target | Metrics |
|-----------|--------|---------|
| API Routes | 90% | Lines, branches |
| Services | 95% | Lines, branches |
| React Components | 85% | Lines, branches |
| Hooks | 90% | Lines, branches |
| Integration | 80% | E2E flows |

### 4.3 Iteration Schedule

| Iteration | Focus | Tests | Duration |
|-----------|-------|-------|----------|
| 1 | API scaffolding + Chat endpoint | 15 tests | Day 1 |
| 2 | Document upload/list endpoints | 12 tests | Day 2 |
| 3 | Search endpoint + hybrid search | 8 tests | Day 3 |
| 4 | React Chat components | 20 tests | Day 4-5 |
| 5 | React Document components | 15 tests | Day 6 |
| 6 | Integration + E2E | 10 tests | Day 7 |
| 7 | Polish + error handling | 10 tests | Day 8 |

---

## Phase 5: COMPLETION

### 5.1 Definition of Done Checklist

#### Backend API
- [ ] All endpoints return correct status codes
- [ ] Request validation with Pydantic
- [ ] Error responses follow consistent format
- [ ] Rate limiting middleware active
- [ ] CORS configured for frontend origin
- [ ] OpenAPI docs generated at /docs
- [ ] Health check endpoint at /health
- [ ] All tests pass (90%+ coverage)

#### Frontend
- [ ] Chat panel fully functional
- [ ] Document upload with progress
- [ ] Document list with pagination
- [ ] Search results display
- [ ] Responsive on mobile (320px+)
- [ ] Loading states for all async ops
- [ ] Error boundaries catch crashes
- [ ] All tests pass (85%+ coverage)

#### Integration
- [ ] Chat queries return real answers
- [ ] Document upload creates embeddings
- [ ] Citations link to source documents
- [ ] Conversation history persists
- [ ] Feedback submissions stored

#### DevOps
- [ ] Docker compose for local dev
- [ ] Environment variable documentation
- [ ] Build scripts in package.json
- [ ] CI pipeline runs tests

### 5.2 Deployment Checklist

```bash
# Local Development
npm run dev:api      # Start FastAPI on :8000
npm run dev:web      # Start Vite on :5173

# Production Build
npm run build:api    # No build needed (Python)
npm run build:web    # Vite production build

# Docker
docker-compose up    # Full stack local

# Environment
cp .env.example .env
# Fill in: SUPABASE_*, OPENAI_*, OPENROUTER_*
```

### 5.3 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| User can ask question | 100% success | Manual test |
| Response includes citations | 95% of responses | Automated check |
| Upload accepts all formats | 6/6 formats | Test suite |
| Page loads under 3s | P95 < 3s | Lighthouse |
| No console errors | 0 errors | E2E test |

---

## Appendix A: File-by-File Implementation Order

### Backend (Python/FastAPI)

1. `src/documind/api/__init__.py` - Package init
2. `src/documind/api/main.py` - FastAPI app
3. `src/documind/api/schemas/chat.py` - Chat Pydantic models
4. `src/documind/api/routes/chat.py` - Chat endpoints
5. `src/documind/api/services/chat_service.py` - Chat business logic
6. `src/documind/api/schemas/documents.py` - Document models
7. `src/documind/api/routes/documents.py` - Document endpoints
8. `src/documind/api/services/doc_service.py` - Document logic
9. `src/documind/api/schemas/search.py` - Search models
10. `src/documind/api/routes/search.py` - Search endpoints
11. `src/documind/api/middleware/cors.py` - CORS config
12. `src/documind/api/middleware/rate_limit.py` - Rate limiting
13. `src/documind/api/middleware/error_handler.py` - Error handling

### Frontend (React/TypeScript)

1. `src/web/package.json` - Dependencies
2. `src/web/vite.config.ts` - Build config
3. `src/web/src/main.tsx` - Entry point
4. `src/web/src/App.tsx` - Root component
5. `src/web/src/api/client.ts` - API client setup
6. `src/web/src/types/index.ts` - TypeScript types
7. `src/web/src/components/common/Layout.tsx` - App layout
8. `src/web/src/components/chat/ChatPanel.tsx` - Main chat
9. `src/web/src/components/chat/MessageList.tsx` - Messages
10. `src/web/src/components/chat/MessageBubble.tsx` - Single message
11. `src/web/src/components/chat/ChatInput.tsx` - Input field
12. `src/web/src/components/chat/CitationCard.tsx` - Citation display
13. `src/web/src/components/documents/DocumentList.tsx` - Doc list
14. `src/web/src/components/documents/UploadZone.tsx` - Upload area
15. `src/web/src/hooks/useChat.ts` - Chat hook
16. `src/web/src/hooks/useDocuments.ts` - Documents hook

---

## Appendix B: Test File Mapping

| Source File | Test File |
|-------------|-----------|
| `api/routes/chat.py` | `tests/api/test_chat_routes.py` |
| `api/routes/documents.py` | `tests/api/test_document_routes.py` |
| `api/routes/search.py` | `tests/api/test_search_routes.py` |
| `api/services/chat_service.py` | `tests/api/test_chat_service.py` |
| `api/services/doc_service.py` | `tests/api/test_doc_service.py` |
| `web/components/chat/ChatPanel.tsx` | `tests/web/components/ChatPanel.test.tsx` |
| `web/components/chat/MessageList.tsx` | `tests/web/components/MessageList.test.tsx` |
| `web/components/documents/UploadZone.tsx` | `tests/web/components/UploadZone.test.tsx` |
| `web/hooks/useChat.ts` | `tests/web/hooks/useChat.test.ts` |

---

## Appendix C: Dependencies to Add

### Python (requirements.txt additions)
```
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
slowapi>=0.1.9
```

### Node.js (package.json additions)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.17.0",
    "zustand": "^4.4.7",
    "react-markdown": "^9.0.1",
    "react-dropzone": "^14.2.3"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0",
    "tailwindcss": "^3.4.0",
    "vitest": "^1.2.0",
    "@testing-library/react": "^14.1.0"
  }
}
```

---

*Document Version: 1.0*
*Created: Following SPARC Methodology*
*Author: Claude Code with TDD Practices*
