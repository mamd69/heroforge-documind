# HeroForge.AI Course: AI-Powered Software Development
## Lesson 6 Workshop: RAG and CAG - Retrieval Augmented Generation

**Estimated Time:** 45-60 minutes\
**Difficulty:** Intermediate\
**Prerequisites:** 
1. Completed Sessions 1-5 (Vector embeddings, Supabase integration, document processing)
2. If your S5 pipeline is not writing to the database, run the db/seeds/S6-demo.sql script now to populate your database for this session. (pull from upstream HeroForge DocuMind repo)

---

## Workshop Objectives

By completing this workshop, you will:
- [x] Understand the RAG (Retrieval-Augmented Generation) pipeline architecture
- [x] Implement semantic search over your document corpus using pgvector
- [x] Build a complete Q&A system with OpenRouter integration
- [x] Add source citations and attribution to AI-generated answers
- [x] Distinguish between RAG and CAG (Context-Augmented Generation)
- [x] Optimize retrieval for accuracy and relevance

---

## Module 1: RAG Fundamentals (15 minutes)

### Concept Review

**What is RAG (Retrieval-Augmented Generation)?**

RAG is an AI architecture pattern that combines document retrieval with language generation to produce accurate, grounded answers. Instead of relying solely on the LLM's training data, RAG retrieves relevant information from your document corpus and uses it to generate contextually accurate responses.

**The RAG Pipeline:**
```
User Query → Embed Query → Vector Search → Retrieve Top-K Docs → Assemble Context → Generate Answer → Return with Citations
```

**Why Use RAG?**
1. **Grounding**: Answers based on your actual documents, not hallucinated
2. **Recency**: Access to up-to-date information beyond model training cutoff
3. **Attribution**: Every answer includes source citations for verification
4. **Privacy**: Your data stays in your database, not in model training
5. **Domain-Specific**: Works with specialized knowledge not in general models

**RAG vs Other Approaches:**
| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Pure LLM** | General questions | Fast, simple | Hallucinations, outdated info |
| **Fine-tuning** | Domain adaptation | Model learns domain | Expensive, inflexible, no citations |
| **RAG** | Q&A over documents | Grounded, recent, cited | Requires vector DB, more complex |
| **CAG** | Small knowledge base | Simple, direct | Limited by context window |

---

### Exercise 1.1: Implement Semantic Search

**Task:** Build a semantic search function that finds relevant documents for a query.

**Instructions:**

**Step 1: Create Search Module (3 mins)**

```bash
# Create the RAG module directory
mkdir -p src/documind/rag
touch src/documind/rag/__init__.py
touch src/documind/rag/search.py
```

**Step 2: Implement Semantic Search (7 mins)**

## Exercise Prompt: Create Semantic Search Module

### Natural Language Instruction to Claude

```
Create a semantic search module for DocuMind at src/documind/rag/search.py.

Implement the following functions:

1. get_query_embedding(query: str) -> List[float]:
   - Generate embedding for search query using OpenAI
   - Use model "text-embedding-3-small"
   - Return the embedding vector

2. search_documents(query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
   - Generate query embedding using get_query_embedding()
   - Search using Supabase RPC function 'match_documents'
   - Pass query_embedding, match_count (top_k), and similarity_threshold
   - Format results with id, content, metadata, similarity, document_name, chunk_index
   - Return list of matching documents

3. hybrid_search(query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
   - Combine semantic search with keyword search
   - Get semantic_results from search_documents()
   - Get keyword_results using Supabase full-text search on content
   - Merge results avoiding duplicates
   - Return combined top_k results

Include:
- Import os, typing (List, Dict, Any), openai, supabase
- Initialize clients using environment variables (SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY)
- Add comprehensive docstrings
- Add test code in __main__ block that searches for "What is our vacation policy?" and prints results

Use proper type hints and error handling.
```

### Review: Generated Semantic Search Module

After Claude generates the code, review the file:

```python
"""
Semantic Search Implementation
Uses OpenAI embeddings + Supabase pgvector for similarity search
"""
import os
from typing import List, Dict, Any
import openai
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
openai.api_key = OPENAI_API_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for search query using OpenAI.

    Args:
        query: User's search query

    Returns:
        List of floats representing the query embedding
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def search_documents(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search for documents relevant to the query using semantic similarity.

    Args:
        query: User's search query
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of document chunks with metadata and similarity scores
    """
    # Step 1: Generate query embedding
    query_embedding = get_query_embedding(query)

    # Step 2: Search using Supabase RPC function
    # This calls the match_documents function we created in Session 5
    results = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_count': top_k,
            'similarity_threshold': similarity_threshold
        }
    ).execute()

    # Step 3: Format and return results
    documents = []
    for row in results.data:
        documents.append({
            'id': row['id'],
            'content': row['content'],
            'metadata': row['metadata'],
            'similarity': row['similarity'],
            'document_name': row.get('document_name', 'Unknown'),
            'chunk_index': row.get('chunk_index', 0)
        })

    return documents

def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Combine semantic search with keyword search for better results.

    Args:
        query: User's search query
        top_k: Number of results to return
        semantic_weight: Weight for semantic search (0-1), remainder is keyword

    Returns:
        List of documents ranked by hybrid score
    """
    # Semantic search
    semantic_results = search_documents(query, top_k=top_k * 2)

    # Keyword search using PostgreSQL full-text search
    keyword_results = supabase.from_('document_chunks') \
        .select('*') \
        .text_search('content', query) \
        .limit(top_k * 2) \
        .execute()

    # Combine and re-rank
    # In production, implement proper scoring fusion (RRF, CombSUM, etc.)
    # For now, prioritize semantic results
    seen_ids = set()
    combined = []

    for doc in semantic_results:
        if doc['id'] not in seen_ids:
            doc['search_type'] = 'semantic'
            combined.append(doc)
            seen_ids.add(doc['id'])

    for row in keyword_results.data:
        if row['id'] not in seen_ids:
            combined.append({
                'id': row['id'],
                'content': row['content'],
                'metadata': row['metadata'],
                'similarity': 0.5,  # Default score for keyword matches
                'document_name': row.get('document_name', 'Unknown'),
                'chunk_index': row.get('chunk_index', 0),
                'search_type': 'keyword'
            })
            seen_ids.add(row['id'])

    return combined[:top_k]

# Test function
if __name__ == "__main__":
    import json

    # Test semantic search
    query = "What is our vacation policy?"
    print(f"Searching for: {query}")
    print("=" * 60)

    results = search_documents(query, top_k=3)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Document: {doc['document_name']}")
        print(f"Similarity: {doc['similarity']:.3f}")
        print(f"Content Preview: {doc['content'][:150]}...")
        print("-" * 60)
```

**Step 3: Test Semantic Search (5 mins)**

```bash
# Set environment variables (if not already in .env)
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"
export OPENAI_API_KEY="your_openai_key"

# Run the search test
python src/documind/rag/search.py
```

**Expected Output:**
```
Searching for: What is our vacation policy?
============================================================

Result 1:
Document: employee_handbook.pdf
Similarity: 0.892
Content Preview: Vacation Policy: All full-time employees receive 15 days of paid vacation per year. Vacation accrues at a rate of 1.25 days per month...
------------------------------------------------------------

Result 2:
Document: hr_policies.md
Similarity: 0.845
Content Preview: Time Off Benefits: Our company offers comprehensive time-off benefits including vacation, sick leave, and personal days...
------------------------------------------------------------

Result 3:
Document: benefits_guide.pdf
Similarity: 0.789
Content Preview: Annual Leave: Employees are entitled to paid annual leave which increases with tenure...
------------------------------------------------------------
```

---

### ⚠️ Verification Checkpoint: Natural Language Search Testing

**Your search MUST work with natural language queries, not just keywords.**

Run these test cases to verify your search is production-ready:

```python
# Test script: test_search_queries.py
from src.documind.rag.search import search_documents

test_cases = [
    # (query, minimum_expected_results)
    ("remote", 1),                                    # Keyword - should work
    ("What is our remote work policy?", 1),           # Natural language - MUST work
    ("How many PTO days do employees get?", 1),       # Question format - MUST work
    ("vacation benefits", 1),                         # Two keywords
    ("Can I work from home on Fridays?", 1),          # Conversational
]

print("Search Query Testing")
print("=" * 60)

all_passed = True
for query, min_results in test_cases:
    results = search_documents(query, top_k=5)
    count = len(results)
    passed = count >= min_results
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: '{query[:40]}...' returned {count} results (need {min_results}+)")
    if not passed:
        all_passed = False

print("=" * 60)
if all_passed:
    print("✅ All search tests passed!")
else:
    print("❌ Some tests failed - check your search implementation")
```

**If natural language fails but keywords work**, your search may need:
1. **Stop word removal** - Remove "what", "is", "our", "the" before searching
2. **Query preprocessing** - Extract meaningful terms from questions
3. **Lower similarity threshold** - Natural language queries have lower exact match scores
4. **Hybrid search** - Combine semantic + keyword search

**All tests must PASS before continuing to the Q&A module.**

---

### Quiz 1:

**Question 1:** What is the primary advantage of RAG over using a pure LLM?\
   a) RAG grounds answers in actual documents, reducing hallucinations and providing source citations\
   b) RAG is always faster than pure LLM queries\
   c) RAG requires less computational power\
   d) RAG works without an internet connection

**Question 2:** In the RAG pipeline, what happens after the query is embedded?\
   a) Vector similarity search finds the most relevant document chunks from the database\
   b) The query is sent directly to the LLM\
   c) The documents are re-indexed\
   d) The user receives an immediate response

**Question 3:** What does the `similarity_threshold` parameter do in semantic search?\
   a) It filters out results below a certain relevance score, ensuring only sufficiently similar documents are retrieved\
   b) It sets the maximum number of results\
   c) It determines the embedding model size\
   d) It controls the API rate limit

**Answers:**
1. **a)** RAG retrieves from actual documents, providing grounded, cited answers
2. **a)** Vector search retrieves the top-K most similar document chunks
3. **a)** Similarity threshold ensures minimum relevance quality

---

## Module 2: Building the Q&A Pipeline (15 minutes)

### Concept Review

**The Complete Q&A Pipeline:**

1. **Query Processing**: Understand user intent, clean and prepare query
2. **Retrieval**: Use semantic search to find relevant chunks
3. **Context Assembly**: Select and order chunks to fit LLM context window
4. **Prompt Engineering**: Format context and query into effective prompt
5. **Generation**: Call LLM (via OpenRouter) to generate answer
6. **Citation**: Add source references to the answer
7. **Logging**: Track queries and responses for improvement

**Why OpenRouter?**
- Multi-model access (Claude, GPT-4, Gemini, etc.)
- Single API for many providers
- Cost optimization (choose cheapest/best model)
- Easy A/B testing between models
- Fallback support (if one model is down, use another)

---

### Exercise 2.1: Implement Q&A Pipeline

**Task:** Build the complete question-answering system with OpenRouter integration.

**Instructions:**

**Step 1: Install OpenRouter (2 mins)**

```bash
pip install openai  # OpenRouter uses OpenAI-compatible API
```

**Step 2: Create Q&A Module (8 mins)**

## Exercise Prompt: Build Q&A Pipeline with OpenRouter

### Natural Language Instruction to Claude

```
Create a complete Q&A pipeline with OpenRouter integration at src/documind/rag/qa_pipeline.py.

Implement the following functions:

1. assemble_context(documents: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
   - Format retrieved documents into context string
   - Include source citations like "[Source {i}: {document_name}, chunk {chunk_index}]"
   - Respect max_tokens limit (use ~4 chars per token estimate)
   - Separate documents with "---"

2. build_qa_prompt(query: str, context: str) -> str:
   - Create prompt with instructions to:
     - Answer using ONLY provided context
     - Say "I don't have enough information" if answer not in context
     - Include source references using [Source X] format
     - Be concise but comprehensive
   - Format with CONTEXT and QUESTION sections

3. generate_answer(query: str, model: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.1, max_tokens: int = 500) -> Dict[str, Any]:
   - Step 1: Retrieve documents using search_documents(query, top_k=5)
   - Step 2: Assemble context from documents
   - Step 3: Build prompt using build_qa_prompt()
   - Step 4: Generate answer using OpenRouter client
   - Step 5: Format sources with id, document, chunk, similarity, preview
   - Return dict with answer, sources, query, model, context_chunks, timestamp

4. compare_models(query: str, models: List[str]) -> Dict[str, Any]:
   - Query multiple models and compare responses
   - Handle errors gracefully

Include:
- Import os, typing, datetime, json, openai
- Import search_documents and get_query_embedding from .search
- Initialize OpenRouter client with base_url "https://openrouter.ai/api/v1"
- Add __main__ block with CLI interface for testing
- Comprehensive docstrings and type hints

The OpenRouter client should use OpenAI-compatible API.
```

### Review: Generated Q&A Pipeline

After Claude generates the code, review the implementation:

```python
"""
Q&A Pipeline with OpenRouter Integration
Complete RAG implementation for question answering
"""
import os
from typing import List, Dict, Any
from datetime import datetime
import json
from openai import OpenAI

from .search import search_documents, get_query_embedding

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter client (uses OpenAI-compatible API)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def assemble_context(documents: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
    """
    Assemble retrieved documents into a context string for the LLM.

    Args:
        documents: List of retrieved document chunks
        max_tokens: Maximum tokens to use for context (rough estimate)

    Returns:
        Formatted context string with source citations
    """
    context_parts = []
    char_count = 0
    max_chars = max_tokens * 4  # Rough conversion: 1 token ≈ 4 characters

    for i, doc in enumerate(documents, 1):
        source_info = f"[Source {i}: {doc['document_name']}, chunk {doc['chunk_index']}]"
        doc_text = f"{source_info}\n{doc['content']}\n"

        # Check if adding this document exceeds limit
        if char_count + len(doc_text) > max_chars:
            break

        context_parts.append(doc_text)
        char_count += len(doc_text)

    return "\n---\n".join(context_parts)

def build_qa_prompt(query: str, context: str) -> str:
    """
    Build the prompt for the LLM with retrieved context.

    Args:
        query: User's question
        context: Assembled context from retrieved documents

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a helpful AI assistant answering questions based on the provided documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information in the provided context
2. If the answer is not in the context, say "I don't have enough information to answer that question"
3. Include specific references to sources in your answer using [Source X] format
4. Be concise but comprehensive
5. If multiple sources provide relevant information, synthesize them

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    return prompt

def generate_answer(
    query: str,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Generate answer using OpenRouter with RAG.

    Args:
        query: User's question
        model: OpenRouter model identifier
        temperature: Generation randomness (0-1)
        max_tokens: Maximum response length

    Returns:
        Dictionary with answer, sources, and metadata
    """
    # Step 1: Retrieve relevant documents
    documents = search_documents(query, top_k=5)

    if not documents:
        return {
            'answer': "I couldn't find any relevant information to answer your question.",
            'sources': [],
            'query': query,
            'model': model,
            'timestamp': datetime.now().isoformat()
        }

    # Step 2: Assemble context
    context = assemble_context(documents)

    # Step 3: Build prompt
    prompt = build_qa_prompt(query, context)

    # Step 4: Generate answer via OpenRouter
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that answers questions based on provided documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer_text = response.choices[0].message.content

    # Step 5: Format sources
    sources = []
    for i, doc in enumerate(documents, 1):
        sources.append({
            'id': i,
            'document': doc['document_name'],
            'chunk': doc['chunk_index'],
            'similarity': doc['similarity'],
            'preview': doc['content'][:200] + "..."
        })

    # Step 6: Return complete response
    return {
        'answer': answer_text,
        'sources': sources,
        'query': query,
        'model': model,
        'context_chunks': len(documents),
        'timestamp': datetime.now().isoformat()
    }

def compare_models(query: str, models: List[str]) -> Dict[str, Any]:
    """
    Compare responses from multiple models on the same query.

    Args:
        query: User's question
        models: List of OpenRouter model identifiers

    Returns:
        Dictionary with responses from each model
    """
    results = {}

    for model in models:
        print(f"Querying {model}...")
        try:
            response = generate_answer(query, model=model)
            results[model] = response
        except Exception as e:
            results[model] = {'error': str(e)}

    return results

# Test and CLI interface
if __name__ == "__main__":
    import sys

    # Check for command-line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask a question: ")

    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")

    # Generate answer
    result = generate_answer(query)

    print(f"Answer:\n{result['answer']}\n")
    print(f"{'='*60}")
    print(f"Sources ({len(result['sources'])}):\n")

    for source in result['sources']:
        print(f"[{source['id']}] {source['document']} (chunk {source['chunk']})")
        print(f"    Similarity: {source['similarity']:.3f}")
        print(f"    Preview: {source['preview']}")
        print()

    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
```

**Step 3: Test the Q&A System (5 mins)**

```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY="your_openrouter_key"

# Test with a question
python src/documind/rag/qa_pipeline.py "What is our vacation policy?"

# Or interactive mode
python src/documind/rag/qa_pipeline.py
```

**Expected Output:**
```
============================================================
Question: What is our vacation policy?
============================================================

Answer:
Based on the provided documents, the vacation policy is as follows:

All full-time employees receive 15 days of paid vacation per year [Source 1]. Vacation time accrues at a rate of 1.25 days per month [Source 1]. Employees must submit vacation requests at least two weeks in advance through the HR portal [Source 2]. Vacation days can be carried over to the next year, but only up to a maximum of 5 days [Source 3].

============================================================
Sources (5):

[1] employee_handbook.pdf (chunk 12)
    Similarity: 0.892
    Preview: Vacation Policy: All full-time employees receive 15 days of paid vacation per year. Vacation accrues at a rate of 1.25 days per month...

[2] hr_policies.md (chunk 5)
    Similarity: 0.845
    Preview: Time Off Benefits: Our company offers comprehensive time-off benefits. Vacation requests must be submitted at least two weeks in advance...

[3] benefits_guide.pdf (chunk 8)
    Similarity: 0.789
    Preview: Annual Leave: Employees are entitled to paid annual leave. Unused vacation can be carried over, up to 5 days maximum...

Model: anthropic/claude-3.5-sonnet
Timestamp: 2025-11-24T14:30:00.123456
```

---

### Quiz 2:

**Question 1:** What is the purpose of the `assemble_context` function?\
   a) To combine retrieved document chunks into a formatted context string that fits within the LLM's token limit\
   b) To search for documents in the database\
   c) To generate embeddings for the query\
   d) To save the conversation history

**Question 2:** Why does the prompt instruct the LLM to "use ONLY the information in the provided context"?\
   a) To prevent hallucinations and ensure answers are grounded in the retrieved documents\
   b) To make the response shorter\
   c) To reduce API costs\
   d) To make the LLM work faster

**Question 3:** What is the main advantage of using OpenRouter instead of calling Claude API directly?\
   a) OpenRouter provides multi-model access, allowing easy switching and comparison between different LLMs\
   b) OpenRouter is always free\
   c) OpenRouter has better quality models\
   d) OpenRouter doesn't require an API key

**Answers:**
1. **a)** Context assembly formats chunks and manages token limits
2. **a)** Constraining to context prevents hallucinations and ensures grounding
3. **a)** OpenRouter provides unified access to multiple LLM providers

---

## Module 3: RAG vs CAG - Understanding the Difference (15 minutes)

### Concept Review

**RAG (Retrieval-Augmented Generation):**
- Retrieves specific, relevant chunks from large corpus
- Uses vector search to find semantic matches
- Scales to millions of documents
- Better for large, diverse knowledge bases
- More complex pipeline (embedding, search, assembly)

**CAG (Context-Augmented Generation):**
- Provides entire knowledge base in context window
- No retrieval step needed
- Simple architecture (just prompt + LLM)
- Limited by context window size (typically 100K-200K tokens)
- Better for small, focused knowledge bases

**When to Use Each:**

| Scenario | RAG | CAG |
|----------|-----|-----|
| **Knowledge base size** | 100+ documents | < 20 documents |
| **Total content size** | > 200K tokens | < 100K tokens |
| **Query patterns** | Diverse, unpredictable | Focused, predictable |
| **Update frequency** | High (documents change often) | Low (stable content) |
| **Latency requirements** | Moderate (retrieval adds ~500ms) | Low (direct LLM call) |
| **Cost considerations** | Lower per query (only relevant chunks) | Higher (full context each time) |

---

### Exercise 3.1: Implement CAG for Comparison

**Task:** Build a simple CAG system to compare with RAG.

**Instructions:**

**Step 1: Create CAG Module (5 mins)**

## Exercise Prompt: Implement CAG for Comparison

### Natural Language Instruction to Claude

```
Create a CAG (Context-Augmented Generation) implementation at src/documind/rag/cag_pipeline.py.

Implement the following functions:

1. load_all_documents(max_docs: int = 20) -> str:
   - Fetch all document chunks from Supabase (limit to max_docs)
   - Format each as "[Document: {document_name}]\n{content}\n"
   - Join with "---" separator
   - Return concatenated string

2. generate_answer_cag(query: str, model: str = "anthropic/claude-3.5-sonnet", temperature: float = 0.1, max_tokens: int = 500) -> Dict[str, Any]:
   - Load ALL documents into full_context using load_all_documents()
   - Build prompt with full context (not retrieval-based)
   - Instructions: answer using documents, be concise, reference specific documents
   - Generate answer using OpenRouter client
   - Return dict with answer, method='CAG', query, model, timestamp

Include:
- Import os, typing, datetime, openai, supabase
- Initialize Supabase client and OpenRouter client
- Add __main__ block for testing
- Comprehensive docstrings
- Type hints

Note: CAG loads entire knowledge base into context - only practical for small knowledge bases (<20 documents).
```

### Review: Generated CAG Implementation

After Claude generates the code, review the differences from RAG:

```python
"""
CAG (Context-Augmented Generation) Implementation
For small knowledge bases that fit in context window
"""
import os
from typing import List, Dict, Any
from datetime import datetime
from openai import OpenAI
from supabase import create_client, Client

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def load_all_documents(max_docs: int = 20) -> str:
    """
    Load all documents from database and concatenate.
    Only practical for small knowledge bases.

    Args:
        max_docs: Maximum number of documents to load

    Returns:
        Concatenated document content
    """
    # Fetch all document chunks
    result = supabase.from_('document_chunks') \
        .select('*') \
        .limit(max_docs) \
        .execute()

    documents = []
    for row in result.data:
        doc_text = f"[Document: {row.get('document_name', 'Unknown')}]\n{row['content']}\n"
        documents.append(doc_text)

    return "\n---\n".join(documents)

def generate_answer_cag(
    query: str,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Generate answer using CAG (full context, no retrieval).

    Args:
        query: User's question
        model: OpenRouter model identifier
        temperature: Generation randomness
        max_tokens: Maximum response length

    Returns:
        Dictionary with answer and metadata
    """
    # Load ALL documents into context
    full_context = load_all_documents()

    # Build prompt with full context
    prompt = f"""You are a helpful AI assistant answering questions based on the provided documents.

INSTRUCTIONS:
1. Answer the question using the information in the provided documents
2. Be concise but comprehensive
3. Reference specific documents when relevant

ALL DOCUMENTS:
{full_context}

QUESTION: {query}

ANSWER:"""

    # Generate answer
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer_text = response.choices[0].message.content

    return {
        'answer': answer_text,
        'method': 'CAG',
        'query': query,
        'model': model,
        'timestamp': datetime.now().isoformat()
    }

# Test function
if __name__ == "__main__":
    import sys

    query = input("Ask a question (CAG mode): ")

    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"Method: CAG (Context-Augmented Generation)")
    print(f"{'='*60}\n")

    result = generate_answer_cag(query)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
```

**Step 2: Compare RAG vs CAG (10 mins)**

## Exercise Prompt: Build RAG vs CAG Comparison Tool

### Natural Language Instruction to Claude

```
Create a comparison tool at src/documind/rag/compare.py that compares RAG vs CAG approaches.

Implement compare_approaches(query: str) function that:
- Prints comparison header
- Tests RAG: time the rag_answer() call, capture result
- Tests CAG: time the cag_answer() call, capture result
- Display both answers side-by-side
- Show performance metrics (latency for each)
- Highlight differences in approach and results

Include:
- Import time module
- Import generate_answer from qa_pipeline as rag_answer
- Import generate_answer_cag from cag_pipeline as cag_answer
- Add __main__ block with test questions
- Print formatted comparison results

Show execution time, answer quality, and method differences clearly.
```

### Review: Generated Comparison Tool

After Claude generates the code, examine how it compares the two approaches:

```python
"""
Compare RAG vs CAG performance and quality
"""
import time
from qa_pipeline import generate_answer as rag_answer
from cag_pipeline import generate_answer_cag as cag_answer

def compare_approaches(query: str):
    """
    Compare RAG and CAG on the same query.

    Args:
        query: Test question
    """
    print(f"{'='*70}")
    print(f"COMPARISON: RAG vs CAG")
    print(f"{'='*70}")
    print(f"Query: {query}\n")

    # Test RAG
    print("Testing RAG (Retrieval-Augmented Generation)...")
    start = time.time()
    rag_result = rag_answer(query)
    rag_time = time.time() - start

    print(f"✓ RAG completed in {rag_time:.2f}s")
    print(f"  Retrieved {rag_result['context_chunks']} chunks")
    print(f"  Answer length: {len(rag_result['answer'])} characters\n")

    # Test CAG
    print("Testing CAG (Context-Augmented Generation)...")
    start = time.time()
    cag_result = cag_answer(query)
    cag_time = time.time() - start

    print(f"✓ CAG completed in {cag_time:.2f}s")
    print(f"  Answer length: {len(cag_result['answer'])} characters\n")

    # Display results
    print(f"{'='*70}")
    print("RAG ANSWER:")
    print(f"{'='*70}")
    print(rag_result['answer'])
    print(f"\nSources: {len(rag_result['sources'])} documents retrieved")

    print(f"\n{'='*70}")
    print("CAG ANSWER:")
    print(f"{'='*70}")
    print(cag_result['answer'])

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    print(f"Speed: RAG {rag_time:.2f}s | CAG {cag_time:.2f}s")
    if rag_time < cag_time:
        print(f"  → RAG was {cag_time/rag_time:.1f}x faster")
    else:
        print(f"  → CAG was {rag_time/cag_time:.1f}x faster")

    print(f"\nRAG Advantages:")
    print(f"  • Specific retrieval (only relevant chunks)")
    print(f"  • Source citations for verification")
    print(f"  • Scales to large document sets")
    print(f"  • Lower cost (fewer tokens per query)")

    print(f"\nCAG Advantages:")
    print(f"  • Simpler architecture")
    print(f"  • Access to full context")
    print(f"  • No retrieval errors")
    print(f"  • Better for small, stable knowledge bases")

if __name__ == "__main__":
    # Test queries
    queries = [
        "What is our vacation policy?",
        "How do I request time off?",
        "What are the remote work guidelines?"
    ]

    for query in queries:
        compare_approaches(query)
        print("\n" + "="*70 + "\n")
```

**Step 3: Run the Comparison**

```bash
python src/documind/rag/compare.py
```

---

### Quiz 3:

**Question 1:** What is the main limitation of CAG compared to RAG?\
   a) CAG is limited by the LLM's context window size and cannot scale to large document collections\
   b) CAG is always slower than RAG\
   c) CAG cannot provide accurate answers\
   d) CAG requires more complex infrastructure

**Question 2:** When is CAG a better choice than RAG?\
   a) When the entire knowledge base is small enough to fit in the context window (< 100K tokens)\
   b) When you have millions of documents\
   c) When you need source citations\
   d) When documents change frequently

**Question 3:** What is the cost trade-off between RAG and CAG?\
   a) RAG uses fewer tokens per query (only relevant chunks), making it more cost-effective for large knowledge bases\
   b) CAG is always cheaper than RAG\
   c) RAG and CAG cost exactly the same\
   d) Cost doesn't depend on the approach

**Answers:**
1. **a)** CAG is limited by context window size and doesn't scale
2. **a)** CAG works well for small knowledge bases that fit in context
3. **a)** RAG is more cost-effective for large document sets

---

## Module 4: Challenge Project - Complete DocuMind Q&A System (15 minutes)

### Challenge Overview

Build a production-ready Q&A system for DocuMind that combines all RAG concepts.

**Your Mission:**
Create a complete question-answering system with:
1. Semantic search with relevance filtering
2. Multi-model support (compare Claude, GPT-4, Gemini)
3. Source citations and attribution
4. Query logging for analytics
5. Simple CLI or web interface

---

### Challenge Requirements

**Feature:** Production-Ready RAG Q&A System

**What to Build:**

1. **Enhanced Search** (Python)
   - Semantic search with configurable threshold
   - Result re-ranking by relevance
   - Deduplication of similar chunks

2. **Multi-Model Q&A**
   - Support 3+ models via OpenRouter
   - Side-by-side comparison mode
   - Automatic fallback if model fails

3. **Citation System**
   - Track which chunks contributed to answer
   - Provide document links/references
   - Highlight relevant passages

4. **Query Logging**
   - Store all queries and responses
   - Track response times
   - Record user feedback (optional)

5. **User Interface**
   - CLI with rich formatting
   - OR simple web interface (Flask/Streamlit)
   - Show sources alongside answers

---

### Starter Code

Create `src/documind/rag/production_qa.py`:

```python
"""
Production-Ready Q&A System
Complete DocuMind RAG implementation
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from openai import OpenAI
from supabase import create_client, Client

from .search import search_documents, hybrid_search

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Available models
MODELS = {
    'claude': 'anthropic/claude-3.5-sonnet',
    'gpt4': 'openai/gpt-4-turbo',
    'gemini': 'google/gemini-pro'
}

class ProductionQA:
    """Production-ready Q&A system with logging and monitoring."""

    def __init__(self, default_model: str = 'claude'):
        self.default_model = MODELS.get(default_model, MODELS['claude'])

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = False
    ) -> Dict[str, Any]:
        """
        TODO: Implement complete query pipeline

        Steps:
        1. Retrieve relevant documents (semantic or hybrid search)
        2. Re-rank results by relevance
        3. Assemble context
        4. Generate answer
        5. Add citations
        6. Log query and response
        7. Return formatted result
        """
        pass

    def compare_models(
        self,
        question: str,
        models: List[str] = ['claude', 'gpt4', 'gemini']
    ) -> Dict[str, Any]:
        """
        TODO: Compare responses from multiple models

        Returns side-by-side comparison with:
        - Each model's answer
        - Response times
        - Token usage
        - Cost estimates
        """
        pass

    def log_query(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        model: str,
        response_time: float
    ) -> None:
        """
        TODO: Log query to database for analytics

        Store in query_logs table:
        - question
        - answer
        - model used
        - sources retrieved
        - response time
        - timestamp
        """
        pass

    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        TODO: Get query analytics

        Returns:
        - Total queries
        - Average response time
        - Most common questions
        - Most frequently retrieved documents
        """
        pass

# CLI Interface
def main():
    """
    TODO: Implement CLI interface

    Features:
    - Interactive Q&A loop
    - Model selection
    - Show sources
    - Save conversation history
    """
    print("DocuMind Q&A System")
    print("=" * 60)

    qa = ProductionQA()

    # Your implementation here
    pass

if __name__ == "__main__":
    main()
```

---

### Your Task

**Step 1: Implement the ProductionQA Class (10 mins)**

Complete the TODOs in `production_qa.py`:
1. Implement `query()` method with full RAG pipeline
2. Implement `compare_models()` for multi-model testing
3. Implement `log_query()` to store queries in Supabase
4. Implement `get_analytics()` to analyze query patterns

**Step 2: Create the Database Table (2 mins)**

```sql
-- Run in Supabase SQL Editor
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model VARCHAR(100),
    sources JSONB,
    response_time FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for analytics
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
```

**Step 3: Test the System (3 mins)**

```bash
# Run the Q&A system
python src/documind/rag/production_qa.py

# Test queries:
# - "What is our vacation policy?"
# - "How do I submit an expense report?"
# - "What are the remote work guidelines?"
```

---

### Success Criteria

Your implementation is complete when:

- [ ] `query()` retrieves documents, generates answer, adds citations
- [ ] `compare_models()` tests multiple models side-by-side
- [ ] `log_query()` stores queries in database successfully
- [ ] `get_analytics()` returns meaningful query statistics
- [ ] CLI provides interactive Q&A experience
- [ ] Sources are displayed with each answer
- [ ] Response times are < 3 seconds
- [ ] Answers include proper citations like [Source 1]

---

## Answer Key

### Exercise 1.1 Solution

The semantic search implementation is provided in the exercise. Key points:
- Uses `text-embedding-3-small` for query embeddings (cost-effective)
- Calls Supabase RPC function `match_documents` for vector search
- Returns top-K results with similarity scores
- Filters by similarity threshold to ensure quality

### Exercise 2.1 Solution

The Q&A pipeline implementation is complete in the exercise. Key components:
- Context assembly manages token limits
- Prompt engineering constrains LLM to provided context
- OpenRouter integration allows multi-model access
- Source citation tracks document attribution

### Module 4 Challenge Solution

Complete implementation of `src/documind/rag/production_qa.py`:

```python
"""
Production-Ready Q&A System - COMPLETE SOLUTION
"""
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from openai import OpenAI
from supabase import create_client, Client

from .search import search_documents, hybrid_search

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Available models
MODELS = {
    'claude': 'anthropic/claude-3.5-sonnet',
    'gpt4': 'openai/gpt-4-turbo',
    'gemini': 'google/gemini-pro'
}

class ProductionQA:
    """Production-ready Q&A system with logging and monitoring."""

    def __init__(self, default_model: str = 'claude'):
        self.default_model = MODELS.get(default_model, MODELS['claude'])

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        top_k: int = 5,
        use_hybrid: bool = False
    ) -> Dict[str, Any]:
        """Complete query pipeline with logging."""
        start_time = time.time()

        # Use default model if not specified
        model_id = model or self.default_model
        if model in MODELS:
            model_id = MODELS[model]

        # Step 1: Retrieve documents
        if use_hybrid:
            documents = hybrid_search(question, top_k=top_k)
        else:
            documents = search_documents(question, top_k=top_k)

        if not documents:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'model': model_id,
                'response_time': time.time() - start_time
            }

        # Step 2: Assemble context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Source {i}: {doc['document_name']}]\n{doc['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Build prompt
        prompt = f"""You are a helpful AI assistant answering questions based on provided documents.

INSTRUCTIONS:
1. Answer using ONLY the provided context
2. Include source references like [Source 1]
3. Be concise but comprehensive
4. If answer not in context, say so

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # Step 4: Generate answer
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        response_time = time.time() - start_time

        # Step 5: Format sources
        sources = [
            {
                'id': i,
                'document': doc['document_name'],
                'similarity': doc['similarity'],
                'preview': doc['content'][:150] + "..."
            }
            for i, doc in enumerate(documents, 1)
        ]

        result = {
            'answer': answer,
            'sources': sources,
            'model': model_id,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }

        # Step 6: Log query
        self.log_query(question, answer, sources, model_id, response_time)

        return result

    def compare_models(
        self,
        question: str,
        models: List[str] = ['claude', 'gpt4', 'gemini']
    ) -> Dict[str, Any]:
        """Compare responses from multiple models."""
        results = {}

        for model_name in models:
            print(f"Querying {model_name}...")
            try:
                result = self.query(question, model=model_name)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {'error': str(e)}

        # Analysis
        fastest = min(
            [(m, r['response_time']) for m, r in results.items() if 'response_time' in r],
            key=lambda x: x[1]
        )

        return {
            'results': results,
            'fastest_model': fastest[0],
            'fastest_time': fastest[1]
        }

    def log_query(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        model: str,
        response_time: float
    ) -> None:
        """Log query to database."""
        try:
            supabase.from_('query_logs').insert({
                'question': question,
                'answer': answer,
                'model': model,
                'sources': json.dumps(sources),
                'response_time': response_time
            }).execute()
        except Exception as e:
            print(f"Warning: Failed to log query: {e}")

    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get query analytics."""
        cutoff = datetime.now() - timedelta(days=days)

        # Fetch logs
        result = supabase.from_('query_logs') \
            .select('*') \
            .gte('created_at', cutoff.isoformat()) \
            .execute()

        logs = result.data

        if not logs:
            return {'message': 'No queries in time period'}

        # Calculate statistics
        total_queries = len(logs)
        avg_response_time = sum(log['response_time'] for log in logs) / total_queries

        # Most common questions
        questions = {}
        for log in logs:
            q = log['question']
            questions[q] = questions.get(q, 0) + 1

        top_questions = sorted(questions.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'total_queries': total_queries,
            'avg_response_time': f"{avg_response_time:.2f}s",
            'top_questions': [
                {'question': q, 'count': c} for q, c in top_questions
            ]
        }

# CLI Interface
def main():
    """Interactive CLI interface."""
    print("="*70)
    print("DocuMind Q&A System (Production)")
    print("="*70)
    print("\nCommands:")
    print("  /help     - Show help")
    print("  /compare  - Compare multiple models")
    print("  /stats    - Show analytics")
    print("  /quit     - Exit")
    print("\nOr just type your question!\n")

    qa = ProductionQA()

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question == '/quit':
                print("Goodbye!")
                break

            if question == '/help':
                print("\nCommands:")
                print("  /help     - Show this help")
                print("  /compare  - Compare multiple models")
                print("  /stats    - Show query analytics")
                print("  /quit     - Exit")
                continue

            if question == '/stats':
                stats = qa.get_analytics()
                print(f"\n{'='*70}")
                print("Analytics (Last 7 Days)")
                print(f"{'='*70}")
                print(f"Total Queries: {stats.get('total_queries', 0)}")
                print(f"Avg Response Time: {stats.get('avg_response_time', 'N/A')}")
                print("\nTop Questions:")
                for i, item in enumerate(stats.get('top_questions', []), 1):
                    print(f"  {i}. {item['question']} ({item['count']} times)")
                continue

            if question == '/compare':
                question = input("Question to compare: ").strip()
                print("\nComparing models...")
                comparison = qa.compare_models(question)

                print(f"\n{'='*70}")
                print(f"Fastest: {comparison['fastest_model']} ({comparison['fastest_time']:.2f}s)")
                print(f"{'='*70}\n")

                for model_name, result in comparison['results'].items():
                    print(f"{model_name.upper()}:")
                    if 'error' in result:
                        print(f"  Error: {result['error']}")
                    else:
                        print(f"  Time: {result['response_time']:.2f}s")
                        print(f"  Answer: {result['answer'][:200]}...")
                    print()
                continue

            # Regular query
            print("\nSearching...")
            result = qa.query(question)

            print(f"\n{'='*70}")
            print("Answer:")
            print(f"{'='*70}")
            print(result['answer'])

            print(f"\n{'='*70}")
            print(f"Sources ({len(result['sources'])}):")
            print(f"{'='*70}")
            for source in result['sources']:
                print(f"[{source['id']}] {source['document']} (similarity: {source['similarity']:.3f})")
                print(f"    {source['preview']}")

            print(f"\nModel: {result['model']}")
            print(f"Response Time: {result['response_time']:.2f}s")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
======================================================================
DocuMind Q&A System (Production)
======================================================================

Commands:
  /help     - Show help
  /compare  - Compare multiple models
  /stats    - Show analytics
  /quit     - Exit

Or just type your question!

You: What is our vacation policy?

Searching...

======================================================================
Answer:
======================================================================
Based on the provided documents, the vacation policy includes:

All full-time employees receive 15 days of paid vacation annually [Source 1]. Vacation accrues at 1.25 days per month [Source 1]. Requests must be submitted at least two weeks in advance through the HR portal [Source 2]. Up to 5 unused days can be carried over to the next year [Source 3].

======================================================================
Sources (3):
======================================================================
[1] employee_handbook.pdf (similarity: 0.892)
    Vacation Policy: All full-time employees receive 15 days of paid vacation per year. Vacation accrues at a rate of 1.25 days per month...
[2] hr_policies.md (similarity: 0.845)
    Time Off Benefits: Our company offers comprehensive time-off benefits. Vacation requests must be submitted at least two weeks in advance...
[3] benefits_guide.pdf (similarity: 0.789)
    Annual Leave: Employees are entitled to paid annual leave. Unused vacation can be carried over, up to 5 days maximum...

Model: anthropic/claude-3.5-sonnet
Response Time: 2.34s
```

---

## Additional Challenges (Optional)

### Challenge 1: Advanced Retrieval
Implement advanced retrieval techniques:
- Reciprocal Rank Fusion (RRF) for hybrid search
- Query expansion (generate multiple versions of query)
- Maximal Marginal Relevance (MMR) for diverse results
- Parent document retrieval (retrieve full doc from matched chunk)

### Challenge 2: Streaming Responses
Implement streaming for better UX:
- Stream answer tokens as they're generated
- Show "Searching..." → "Thinking..." → "Answer:" states
- Progressive source loading

### Challenge 3: Conversation Context
Add multi-turn conversation support:
- Track conversation history
- Use previous questions/answers for context
- Resolve pronoun references (it, that, them)

### Challenge 4: Evaluation Suite
Build automated evaluation:
- Create test dataset of Q&A pairs
- Implement RAGAS metrics (faithfulness, relevance)
- Compare RAG vs CAG quantitatively
- A/B test different retrieval strategies

---

## Key Takeaways

By completing this workshop, you've learned:

1. **RAG Architecture**: Query → Embed → Retrieve → Assemble → Generate → Cite
2. **Semantic Search**: Vector embeddings + cosine similarity for relevance
3. **OpenRouter**: Multi-model access for flexibility and cost optimization
4. **RAG vs CAG**: RAG scales to large corpora, CAG works for small knowledge bases
5. **Production Systems**: Logging, monitoring, and analytics are critical

**The RAG Formula:**
```
Relevance = Retrieval Quality × Generation Quality × Context Assembly
```

Good retrieval is worthless without good generation, and vice versa!

---

## Git: Creating Pull Requests (10 minutes)

### Concept: Code Review Before Merging

Pull Requests (PRs) are how professional teams:
- **Review code quality** before it enters main branch
- **Discuss implementation** decisions
- **Catch bugs early** through peer review
- **Share knowledge** across the team
- **Maintain code standards**

**The Flow:**
```
Your Branch → Pull Request → Code Review → Merge to Main
```

### Exercise 0.3: Create Your First Pull Request

**Step 1: Ensure Everything is Committed and Pushed (2 mins)**

```bash
# Check for uncommitted changes
git status

# If you have changes, commit them
git add .
git commit -m "feat: complete RAG Q&A system with citations

Implemented full RAG pipeline:
- Semantic document retrieval
- Context augmentation
- LLM answer generation with OpenRouter
- Source citation tracking

All tests passing. Ready for review.

Closes #15"

# Push to GitHub
git push
```

**Step 2: Create Pull Request on GitHub (3 mins)**

1. **Go to your repository on GitHub**
2. **Click "Pull requests" tab**
3. **Click "New pull request" button**
4. **Set base and compare:**
   - Base: `main`
   - Compare: `issue-15-rag-qa-system` (your branch)
5. **Click "Create pull request"**

**Step 3: Write PR Description (5 mins)**

Use this template:

```markdown
## Summary
Implements complete RAG Q&A system for DocuMind with semantic search and citation tracking.

## Changes Made
- ✅ Created `src/rag/embeddings.py` - OpenAI embedding generator
- ✅ Created `src/rag/retriever.py` - Semantic document search
- ✅ Created `src/rag/llm_client.py` - OpenRouter client wrapper
- ✅ Created `src/rag/qa_system.py` - Complete RAG pipeline
- ✅ Created `src/rag/demo.py` - Interactive demo script
- ✅ Added test cases for different question types
- ✅ Documented API usage and configuration

## Testing Done
- [x] Manual testing with 5+ sample queries
- [x] Verified semantic search finds relevant documents
- [x] Confirmed citations are accurate
- [x] Tested out-of-scope question handling

## Test Plan for Reviewers
1. Run `python3 src/rag/demo.py`
2. Try asking: "How many vacation days do employees get?"
3. Verify answer includes specific numbers (15/20/25 days)
4. Check that sources are listed at bottom

## Related Issues
Closes #15

## Checklist
- [x] Code follows project style guidelines
- [x] Comments added for complex logic
- [x] All tests passing
- [x] Documentation updated
- [x] No secrets (API keys) committed
- [x] Ready for review
```

**Click "Create pull request"**

### PR Best Practices

**DO:**
- ✅ Write clear, descriptive PR titles
- ✅ Explain WHY you made changes, not just WHAT
- ✅ Include test instructions for reviewers
- ✅ Reference related issues with `Closes #<number>`
- ✅ Keep PRs focused (one feature per PR)
- ✅ Respond to review comments promptly

**DON'T:**
- ❌ Create massive PRs (>500 lines changed)
- ❌ Leave PR description empty
- ❌ Merge your own PR without review (in team settings)
- ❌ Include unrelated changes
- ❌ Commit broken code

### Merging Your PR

**After approval:**

1. **Ensure all checks pass** (we'll add CI/CD in Session 10)
2. **Resolve any merge conflicts** (if any)
3. **Click "Merge pull request"** button
4. **Choose merge strategy:**
   - "Create a merge commit" (default - keeps all history)
   - "Squash and merge" (combines all commits into one)
5. **Click "Confirm merge"**
6. **Delete the branch** (GitHub will prompt you)

**After merging:**

```bash
# Switch back to main branch
git checkout main

# Pull the merged changes
git pull origin main

# Delete local feature branch (it's merged now)
git branch -d issue-15-rag-qa-system

# Verify it's gone
git branch
# Should only show: * main
```

### Git Flow Summary

```
1. Create Issue → Get issue # (e.g., #15)
2. Create Branch → git checkout -b issue-15-feature-name
3. Make Changes → Edit files with Claude Code
4. Commit Locally → git add . && git commit -m "feat: ..."
5. Push to Remote → git push -u origin issue-15-feature-name
6. Create PR → On GitHub, compare branch to main
7. Request Review → Tag teammate or instructor
8. Address Feedback → Make requested changes
9. Merge PR → Click "Merge pull request"
10. Clean Up → Delete branch, pull main
11. Repeat → Start next feature with new issue!
```

This is the professional Git workflow used by companies like Anthropic, OpenAI, and Google!

---

## Testing Your RAG Pipeline (10 minutes)

### Concept: Why Test AI Systems?

AI systems have unique testing challenges:
- **Non-deterministic outputs:** Same input → different outputs
- **Quality vs correctness:** "Good" answers are subjective
- **Context dependency:** Retrieval quality affects generation

**What to test:**
1. ✅ **Integration tests:** Does the full pipeline work?
2. ✅ **Retrieval tests:** Are relevant documents found?
3. ✅ **Regression tests:** Are previous bugs fixed?
4. ❌ **NOT unit tests:** Too brittle for AI systems

### Exercise: Create Integration Test Suite

**Step 1: Create Test File (5 mins)**

Ask Claude Code:

```
Create an integration test suite at tests/test_rag_integration.py.

Requirements:
- Test that RAG pipeline answers questions correctly
- Test that retrieval finds relevant documents
- Test that citations are included
- Test that out-of-scope questions are handled gracefully
- Use pytest framework
- Each test should be independent

Include:
- Test fixture for RAGQASystem
- 5 test cases covering different scenarios
- Assertions checking answer quality (length, keywords, sources)
- Clear test names: test_<what>_<expected>
```

**Step 2: Run Tests (2 mins)**

```bash
# Install pytest if needed
pip install pytest

# Run tests
pytest tests/test_rag_integration.py -v

# Expected output:
# tests/test_rag_integration.py::test_vacation_policy_returns_days PASSED
# tests/test_rag_integration.py::test_retrieval_finds_relevant_docs PASSED
# tests/test_rag_integration.py::test_citations_included PASSED
# tests/test_rag_integration.py::test_out_of_scope_handled PASSED
# ==================== 4 passed in 12.43s ====================
```

**Step 3: Commit Tests (3 mins)**

```bash
git add tests/test_rag_integration.py
git commit -m "test: add integration tests for RAG pipeline

Tests cover:
- Question answering with specific details
- Document retrieval relevance
- Citation inclusion
- Out-of-scope question handling

Relates to #15"

git push
```

### Testing Best Practices for AI Systems

**DO:**
- ✅ Test integration points (full pipeline)
- ✅ Use real examples from production
- ✅ Check for presence of key information, not exact text
- ✅ Test edge cases (empty input, malformed queries)
- ✅ Run tests frequently

**DON'T:**
- ❌ Assert exact output text (too brittle)
- ❌ Mock LLM responses (defeats the purpose)
- ❌ Test internal LLM behavior (black box it)
- ❌ Aim for 100% coverage (not practical for AI)

---

## Documenting Your RAG System (5 minutes)

### Concept: READMEs are Your First Impression

When someone discovers your project (instructor, classmate, future employer), they read the README first.

**A good README answers:**
1. What does this do?
2. How do I use it?
3. How do I set it up?
4. Where can I learn more?

### Exercise: Update Project README

**Ask Claude Code:**

```
Update the README.md file to document the RAG Q&A system.

Add a section called "## RAG Q&A System" that includes:
1. Overview: What the RAG system does
2. Architecture: List of components (embeddings, retriever, llm_client, qa_system)
3. Quick Start: How to run the demo
4. Configuration: Environment variables needed
5. Example Usage: Code snippet showing RAGQASystem
6. Testing: How to run tests

Use clear headings, code blocks, and bullet points.
```

### Documentation Best Practices

**DO:**
- ✅ Update README when adding features
- ✅ Include code examples that actually work
- ✅ Document prerequisites clearly
- ✅ Add troubleshooting for common issues
- ✅ Keep it concise (people skim, don't read)

**DON'T:**
- ❌ Let README get stale (update it!)
- ❌ Write a novel (too much text)
- ❌ Skip setup instructions
- ❌ Use outdated examples

**Commit the README:**

```bash
git add README.md
git commit -m "docs: add RAG Q&A system documentation

Documented architecture, usage, and troubleshooting.
Includes code examples and performance metrics.

Relates to #15"
```

---

## Next Session Preview

In **Session 7: Advanced Data Extraction**, we'll:
- Extract text from complex PDFs (tables, images, layouts)
- Parse Word documents and spreadsheets
- Handle OCR for scanned documents
- Build multi-format document ingestion pipelines
- Preserve structure and metadata during extraction

**Preparation:**
1. Install: `pip install pypdf2 pdfplumber python-docx openpyxl`
2. Gather sample PDFs, DOCX, XLSX files for testing
3. Review the DocuMind demo from Session 3

See you in Session 7!

---

**Workshop Complete! 🎉**

You've built a production-ready RAG system for DocuMind. Your documents are now searchable, and users can get AI-powered answers with source citations!
