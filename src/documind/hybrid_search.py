"""
Hybrid Search: Combine semantic (vector) and keyword (BM25) search
"""
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Lazy imports to handle missing dependencies gracefully
_supabase_client = None
_openai_client = None


def _get_supabase_client():
    """Get or create Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        _supabase_client = create_client(url, key)
    return _supabase_client


def _get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


class EmbeddingsClient:
    """Simple embeddings client using OpenAI."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        client = _get_openai_client()
        response = client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

class HybridSearcher:
    """Combines vector and keyword search with reranking"""

    def __init__(self, semantic_weight: float = 0.7):
        """
        Initialize hybrid searcher.

        Args:
            semantic_weight: Weight for semantic search (0-1)
                           1.0 = pure semantic, 0.0 = pure keyword
        """
        self.embeddings_client = EmbeddingsClient()
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        return _get_supabase_client()

    def search_semantic(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search using vector embeddings"""
        # Generate query embedding
        query_embedding = self.embeddings_client.generate_embedding(query)

        # Search with pgvector using match_documents function
        result = self.supabase.rpc("match_documents", {
            "query_embedding": query_embedding,
            "similarity_threshold": threshold,
            "match_count": top_k
        }).execute()

        return [{
            "id": row.get("id", ""),
            "document_id": row.get("document_id", row.get("id", "")),
            "content": row.get("content", ""),
            "semantic_score": row.get("similarity", 0.0)
        } for row in (result.data or [])]

    def search_keyword(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Keyword search using ILIKE pattern matching"""
        # Use simple ILIKE for keyword matching
        # This is a fallback since full-text search RPC may not exist
        try:
            result = self.supabase.table("document_chunks").select(
                "id, content, metadata"
            ).ilike("content", f"%{query}%").limit(top_k).execute()

            results = []
            for i, row in enumerate(result.data or []):
                # Calculate a simple rank based on position
                rank = 1.0 - (i * 0.05)  # Decreasing score
                results.append({
                    "id": row.get("id", ""),
                    "document_id": row.get("metadata", {}).get("document_id", row.get("id", "")),
                    "content": row.get("content", ""),
                    "keyword_score": max(0.1, rank)
                })
            return results
        except Exception as e:
            print(f"   Keyword search error: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rerank_method: str = "linear"
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            rerank_method: Reranking method ("linear" or "rrf")

        Returns:
            List of results sorted by combined score
        """
        print(f"\n🔍 Hybrid Search: '{query}'")
        print(f"   Weights: {self.semantic_weight:.1f} semantic + {self.keyword_weight:.1f} keyword")
        print(f"   Reranking: {rerank_method}")

        # Fetch results from both search methods
        semantic_results = self.search_semantic(query, top_k=20)
        keyword_results = self.search_keyword(query, top_k=20)

        print(f"   Semantic results: {len(semantic_results)}")
        print(f"   Keyword results:  {len(keyword_results)}")

        # Merge and rerank
        if rerank_method == "linear":
            results = self._rerank_linear(semantic_results, keyword_results)
        elif rerank_method == "rrf":
            results = self._rerank_rrf(semantic_results, keyword_results)
        else:
            raise ValueError(f"Unknown rerank method: {rerank_method}")

        # Return top K
        return results[:top_k]

    def _rerank_linear(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Linear combination of scores"""
        # Normalize scores to 0-1 range
        semantic_scores = self._normalize_scores(
            [r["semantic_score"] for r in semantic_results]
        )
        keyword_scores = self._normalize_scores(
            [r["keyword_score"] for r in keyword_results]
        )

        # Update results with normalized scores
        for result, norm_score in zip(semantic_results, semantic_scores):
            result["semantic_score_norm"] = norm_score

        for result, norm_score in zip(keyword_results, keyword_scores):
            result["keyword_score_norm"] = norm_score

        # Merge by ID
        merged = defaultdict(lambda: {"semantic_score_norm": 0, "keyword_score_norm": 0})

        for result in semantic_results:
            merged[result["id"]].update(result)

        for result in keyword_results:
            chunk_id = result["id"]
            if chunk_id in merged:
                merged[chunk_id]["keyword_score_norm"] = result["keyword_score_norm"]
            else:
                merged[chunk_id].update(result)

        # Calculate combined scores
        final_results = []
        for chunk_id, data in merged.items():
            combined_score = (
                self.semantic_weight * data.get("semantic_score_norm", 0) +
                self.keyword_weight * data.get("keyword_score_norm", 0)
            )
            data["combined_score"] = combined_score
            data["id"] = chunk_id
            final_results.append(data)

        # Sort by combined score
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return final_results

    def _rerank_rrf(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) reranking.

        RRF_score(d) = Σ 1 / (k + rank_i(d))

        Where rank_i(d) is the rank of document d in ranking i.
        k is a constant (typically 60).
        """
        # Create rank dictionaries
        semantic_ranks = {r["id"]: i + 1 for i, r in enumerate(semantic_results)}
        keyword_ranks = {r["id"]: i + 1 for i, r in enumerate(keyword_results)}

        # Collect all unique chunk IDs
        all_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_ids:
            semantic_rank = semantic_ranks.get(chunk_id, 999)  # High rank if not found
            keyword_rank = keyword_ranks.get(chunk_id, 999)

            rrf_score = (
                1.0 / (k + semantic_rank) +
                1.0 / (k + keyword_rank)
            )
            rrf_scores[chunk_id] = rrf_score

        # Merge results
        results_dict = {}
        for result in semantic_results + keyword_results:
            chunk_id = result["id"]
            if chunk_id not in results_dict:
                results_dict[chunk_id] = result
                results_dict[chunk_id]["rrf_score"] = rrf_scores[chunk_id]

        # Sort by RRF score
        final_results = list(results_dict.values())
        final_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

        return final_results

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]


# Test hybrid search
if __name__ == "__main__":
    searcher = HybridSearcher(semantic_weight=0.7)

    test_queries = [
        "What is the vacation policy?",
        "How many sick days do I get?",
        "What are the employee benefits?",
        "How does parental leave work?"
    ]

    for query in test_queries:
        print("\n" + "=" * 70)

        # Compare pure semantic, pure keyword, and hybrid
        print("\n📊 Comparison:")

        # Pure semantic
        semantic_only = HybridSearcher(semantic_weight=1.0)
        results_semantic = semantic_only.search_hybrid(query, top_k=3, rerank_method="linear")

        print("\n1️⃣  Pure Semantic (100% vector):")
        for i, r in enumerate(results_semantic, 1):
            print(f"   {i}. [{r.get('combined_score', 0):.4f}] {r['content'][:60]}...")

        # Pure keyword
        keyword_only = HybridSearcher(semantic_weight=0.0)
        results_keyword = keyword_only.search_hybrid(query, top_k=3, rerank_method="linear")

        print("\n2️⃣  Pure Keyword (100% BM25):")
        for i, r in enumerate(results_keyword, 1):
            print(f"   {i}. [{r.get('combined_score', 0):.4f}] {r['content'][:60]}...")

        # Hybrid
        results_hybrid = searcher.search_hybrid(query, top_k=3, rerank_method="linear")

        print("\n3️⃣  Hybrid (70% semantic + 30% keyword):")
        for i, r in enumerate(results_hybrid, 1):
            print(f"   {i}. [{r.get('combined_score', 0):.4f}] {r['content'][:60]}...")

        # RRF
        results_rrf = searcher.search_hybrid(query, top_k=3, rerank_method="rrf")

        print("\n4️⃣  RRF (Reciprocal Rank Fusion):")
        for i, r in enumerate(results_rrf, 1):
            print(f"   {i}. [{r.get('rrf_score', 0):.4f}] {r['content'][:60]}...")