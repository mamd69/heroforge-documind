"""
DocuMind Semantic Search Module

Provides semantic and hybrid search capabilities for document retrieval
using OpenAI embeddings and Supabase vector search.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)


# Initialize clients using environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Client instances (lazy initialization)
_openai_client: Optional[OpenAI] = None
_supabase_client: Optional[Client] = None


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_supabase_client() -> Client:
    """Get or create Supabase client instance."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a search query using OpenAI.

    Uses the text-embedding-3-small model to create a vector representation
    of the query text for semantic similarity search.

    Args:
        query: The search query text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        ValueError: If OPENAI_API_KEY is not configured.
        openai.OpenAIError: If the API call fails.

    Example:
        >>> embedding = get_query_embedding("What is our vacation policy?")
        >>> len(embedding)
        1536
    """
    client = _get_openai_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return response.data[0].embedding


def search_documents(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Search documents using semantic similarity.

    Generates an embedding for the query and searches for similar documents
    using Supabase's vector similarity search via the match_documents RPC function.

    Args:
        query: The search query text.
        top_k: Maximum number of results to return. Defaults to 5.
        similarity_threshold: Minimum similarity score (0-1) for results.
            Defaults to 0.7.

    Returns:
        A list of dictionaries containing matching documents with keys:
            - id: Document chunk unique identifier
            - content: The text content of the chunk
            - metadata: Additional document metadata
            - similarity: Cosine similarity score (0-1)
            - document_name: Name of the source document
            - chunk_index: Index of this chunk within the document

    Raises:
        ValueError: If required environment variables are not configured.
        Exception: If the database query fails.

    Example:
        >>> results = search_documents("vacation policy", top_k=3)
        >>> for doc in results:
        ...     print(f"{doc['document_name']}: {doc['similarity']:.2f}")
    """
    # Generate query embedding
    query_embedding = get_query_embedding(query)

    # Get Supabase client
    supabase = _get_supabase_client()

    # Call the match_documents RPC function
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "similarity_threshold": similarity_threshold
        }
    ).execute()

    # Format results
    results = []
    for item in response.data or []:
        metadata = item.get("metadata", {})
        # Try multiple field names for document name
        doc_name = (
            metadata.get("document_name") or
            metadata.get("file_name") or
            metadata.get("title") or
            "Unknown"
        )
        results.append({
            "id": item.get("id"),
            "content": item.get("content"),
            "metadata": metadata,
            "similarity": item.get("similarity"),
            "document_name": doc_name,
            "chunk_index": metadata.get("chunk_index", 0)
        })

    return results


def hybrid_search(
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic and keyword search.

    Combines vector similarity search with full-text keyword search to
    improve recall and handle both conceptual and exact-match queries.

    Args:
        query: The search query text.
        top_k: Maximum number of results to return. Defaults to 5.
        semantic_weight: Weight given to semantic results (0-1).
            Keyword results receive weight of (1 - semantic_weight).
            Defaults to 0.7.

    Returns:
        A list of dictionaries containing matching documents with keys:
            - id: Document chunk unique identifier
            - content: The text content of the chunk
            - metadata: Additional document metadata
            - similarity: Combined score (0-1)
            - document_name: Name of the source document
            - chunk_index: Index of this chunk within the document
            - search_type: "semantic", "keyword", or "both"

    Raises:
        ValueError: If required environment variables are not configured.
        Exception: If database queries fail.

    Example:
        >>> results = hybrid_search("annual leave policy", top_k=5)
        >>> for doc in results:
        ...     print(f"[{doc['search_type']}] {doc['document_name']}")
    """
    # Get semantic search results
    semantic_results = search_documents(
        query,
        top_k=top_k,
        similarity_threshold=0.5  # Lower threshold for hybrid search
    )

    # Get Supabase client for keyword search
    supabase = _get_supabase_client()

    # Perform keyword search using Supabase full-text search
    keyword_response = supabase.table("document_chunks").select(
        "id, content, metadata"
    ).ilike("content", f"%{query}%").limit(top_k).execute()

    # Process keyword results
    keyword_results = []
    for item in keyword_response.data or []:
        metadata = item.get("metadata", {})
        doc_name = (
            metadata.get("document_name") or
            metadata.get("file_name") or
            metadata.get("title") or
            "Unknown"
        )
        keyword_results.append({
            "id": item.get("id"),
            "content": item.get("content"),
            "metadata": metadata,
            "similarity": 1.0,  # Full match score for keyword results
            "document_name": doc_name,
            "chunk_index": metadata.get("chunk_index", 0)
        })

    # Merge results avoiding duplicates
    seen_ids = set()
    combined_results = []

    # Create lookup for keyword result IDs
    keyword_ids = {r["id"] for r in keyword_results}

    # Process semantic results
    for result in semantic_results:
        doc_id = result["id"]
        seen_ids.add(doc_id)

        # Check if also found in keyword results
        if doc_id in keyword_ids:
            # Found in both - combine scores
            keyword_weight = 1 - semantic_weight
            combined_score = (result["similarity"] * semantic_weight) + (1.0 * keyword_weight)
            result["similarity"] = min(combined_score, 1.0)
            result["search_type"] = "both"
        else:
            # Semantic only - apply weight
            result["similarity"] = result["similarity"] * semantic_weight
            result["search_type"] = "semantic"

        combined_results.append(result)

    # Add keyword-only results
    keyword_weight = 1 - semantic_weight
    for result in keyword_results:
        if result["id"] not in seen_ids:
            result["similarity"] = 1.0 * keyword_weight
            result["search_type"] = "keyword"
            combined_results.append(result)

    # Sort by combined similarity score and return top_k
    combined_results.sort(key=lambda x: x["similarity"], reverse=True)

    return combined_results[:top_k]


if __name__ == "__main__":
    # Test code for the search module
    print("=" * 60)
    print("DocuMind Semantic Search Test")
    print("=" * 60)

    test_query = "What is our vacation policy?"

    print(f"\nSearch Query: '{test_query}'")
    print("-" * 60)

    try:
        # Test semantic search
        print("\n📊 Semantic Search Results:")
        print("-" * 40)

        results = search_documents(test_query, top_k=5, similarity_threshold=0.3)

        if not results:
            print("  No results found.")
        else:
            for i, doc in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Document: {doc['document_name']}")
                print(f"    Similarity: {doc['similarity']:.4f}")
                print(f"    Chunk Index: {doc['chunk_index']}")
                print(f"    Content Preview: {doc['content'][:150]}...")

        # Test hybrid search
        print("\n\n🔀 Hybrid Search Results:")
        print("-" * 40)

        hybrid_results = hybrid_search(test_query, top_k=5)

        if not hybrid_results:
            print("  No results found.")
        else:
            for i, doc in enumerate(hybrid_results, 1):
                print(f"\n  Result {i}:")
                print(f"    Document: {doc['document_name']}")
                print(f"    Score: {doc['similarity']:.4f}")
                print(f"    Search Type: {doc['search_type']}")
                print(f"    Content Preview: {doc['content'][:150]}...")

        print("\n" + "=" * 60)
        print("✅ Search test completed successfully!")
        print("=" * 60)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("   Please set the required environment variables:")
        print("   - OPENAI_API_KEY")
        print("   - SUPABASE_URL")
        print("   - SUPABASE_ANON_KEY (or SUPABASE_KEY)")
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        raise
