"""
DocuMind Production-Ready Q&A System

Complete RAG implementation with multi-model support, citations, query logging,
and comprehensive analytics. Designed for production deployment with:

- Enhanced semantic search with re-ranking and deduplication
- Multi-model support via OpenRouter with automatic fallback
- Citation tracking and attribution system
- Query logging to Supabase for analytics
- Performance monitoring and insights

Usage:
    python src/documind/rag/production_qa.py "What is the vacation policy?"
    python src/documind/rag/production_qa.py --compare "Benefits overview"
    python src/documind/rag/production_qa.py --analytics
    python src/documind/rag/production_qa.py --interactive
"""

import os
import time
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# Support both module and direct execution
try:
    from .search import search_documents, hybrid_search, get_query_embedding
    from .qa_pipeline import (
        MODELS,
        MODEL_INFO,
        assemble_context,
        build_qa_prompt,
        _get_openrouter_client,
    )
except ImportError:
    # Direct execution - use absolute imports
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from src.documind.rag.search import search_documents, hybrid_search, get_query_embedding
    from src.documind.rag.qa_pipeline import (
        MODELS,
        MODEL_INFO,
        assemble_context,
        build_qa_prompt,
        _get_openrouter_client,
    )

# Load environment variables
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

# Lazy-initialized Supabase client
_supabase_client: Optional[Client] = None


def _get_supabase_client() -> Client:
    """Get or create Supabase client instance."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set"
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# =============================================================================
# PRODUCTION QA CLASS
# =============================================================================


class ProductionQA:
    """
    Production-ready Q&A system with advanced RAG capabilities.

    Features:
    - Enhanced semantic search with re-ranking and deduplication
    - Multi-model support with automatic fallback
    - Citation tracking and attribution
    - Query logging and analytics
    - Performance optimization

    Example:
        >>> qa = ProductionQA()
        >>> result = qa.query("What is the vacation policy?")
        >>> print(result["answer"])
        >>> for src in result["sources"]:
        ...     print(f"  - {src['document']}")
    """

    # Default fallback models in priority order
    DEFAULT_FALLBACK_MODELS = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-haiku",
        "deepseek/deepseek-chat",
    ]

    def __init__(
        self,
        default_model: str = "google/gemini-2.5-flash-lite",
        fallback_models: Optional[List[str]] = None,
        enable_logging: bool = True,
        enable_pii_redaction: bool = False,
    ):
        """
        Initialize ProductionQA system.

        Args:
            default_model: Primary model for queries. Can be model ID or alias
                from MODELS dict (default, premium, budget, opensource, quality).
            fallback_models: List of models to try if primary fails.
                Defaults to [gpt-4o-mini, claude-3.5-haiku, deepseek-chat].
            enable_logging: Whether to log queries to database. Defaults to True.
            enable_pii_redaction: Whether to redact PII in logs. Defaults to False.
        """
        # Resolve model alias if provided
        if default_model in MODELS:
            self.default_model = MODELS[default_model]
        else:
            self.default_model = default_model

        # Set fallback models
        self.fallback_models = fallback_models or self.DEFAULT_FALLBACK_MODELS

        # Configuration
        self.enable_logging = enable_logging
        self.enable_pii_redaction = enable_pii_redaction

        # Embedding cache for performance
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_max_size = 100

    # =========================================================================
    # ENHANCED SEARCH
    # =========================================================================

    def enhanced_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        rerank: bool = True,
        deduplicate: bool = True,
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with re-ranking and deduplication.

        Args:
            query: Search query text.
            top_k: Maximum results to return. Defaults to 10.
            similarity_threshold: Minimum similarity score. Defaults to 0.3.
            rerank: Whether to re-rank results. Defaults to True.
            deduplicate: Whether to remove similar chunks. Defaults to True.
            use_hybrid: Use hybrid (semantic + keyword) search. Defaults to False.

        Returns:
            List of enhanced document results with:
                - id, content, metadata, similarity
                - document_name, chunk_index
                - final_score (if reranked)
                - document_link, citation_format

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Get more results than needed for re-ranking/deduplication
        fetch_k = top_k * 2 if (rerank or deduplicate) else top_k

        # Execute search
        if use_hybrid:
            results = hybrid_search(query, top_k=fetch_k)
        else:
            results = search_documents(
                query, top_k=fetch_k, similarity_threshold=similarity_threshold
            )

        # Deduplication
        if deduplicate and results:
            results = self._deduplicate_results(results)

        # Re-ranking
        if rerank and results:
            results = self._rerank_results(results, query)

        # Limit to top_k
        results = results[:top_k]

        # Enrich results with citation info
        for i, doc in enumerate(results, 1):
            doc["citation_number"] = i
            doc["document_link"] = self._generate_document_link(doc)
            doc["citation_format"] = f"[Source {i}]"
            doc["highlighted_content"] = self._highlight_query_terms(
                doc.get("content", ""), query
            )

        return results

    def _deduplicate_results(
        self, results: List[Dict[str, Any]], threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate results based on content similarity."""
        if not results:
            return results

        unique_results = []
        seen_hashes = set()

        for doc in results:
            content = doc.get("content", "")
            # Create a normalized hash of the content
            normalized = " ".join(content.lower().split())
            content_hash = hashlib.md5(normalized[:500].encode()).hexdigest()

            if content_hash not in seen_hashes:
                unique_results.append(doc)
                seen_hashes.add(content_hash)

        return unique_results

    def _rerank_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Re-rank results using multiple signals."""
        query_terms = set(query.lower().split())

        for doc in results:
            base_score = doc.get("similarity", 0.5)
            content = doc.get("content", "").lower()

            # Term frequency boost
            term_matches = sum(1 for term in query_terms if term in content)
            term_boost = min(term_matches / max(len(query_terms), 1) * 0.1, 0.1)

            # Length penalty (prefer medium-length chunks)
            content_len = len(content)
            if 100 <= content_len <= 1000:
                length_boost = 0.05
            else:
                length_boost = 0

            # Calculate final score
            doc["final_score"] = base_score + term_boost + length_boost

        # Sort by final score
        results.sort(key=lambda x: x.get("final_score", x.get("similarity", 0)), reverse=True)

        return results

    def _generate_document_link(self, doc: Dict[str, Any]) -> str:
        """Generate a reference link for a document."""
        doc_name = doc.get("document_name", "Unknown")
        chunk_idx = doc.get("chunk_index", 0)
        doc_id = doc.get("id", "")
        return f"doc://{doc_name}#chunk-{chunk_idx}?id={doc_id}"

    def _highlight_query_terms(self, content: str, query: str) -> str:
        """Highlight query terms in content using **bold** markers."""
        if not content or not query:
            return content

        highlighted = content
        for term in query.lower().split():
            if len(term) > 2:  # Skip short words
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f"**{term}**", highlighted)

        return highlighted

    # =========================================================================
    # MAIN QUERY METHOD
    # =========================================================================

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        enable_fallback: bool = True,
        include_sources: bool = True,
        log_query: bool = True,
        top_k: int = 5,
        use_hybrid: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline with citations and fallback.

        Args:
            question: The user's question.
            model: Model to use. Defaults to self.default_model.
            enable_fallback: Try fallback models on failure. Defaults to True.
            include_sources: Include source documents. Defaults to True.
            log_query: Log to database. Defaults to True.
            top_k: Number of documents to retrieve. Defaults to 5.
            use_hybrid: Use hybrid search. Defaults to False.

        Returns:
            Dictionary containing:
                - answer: Generated answer with citations
                - sources: List of source documents
                - model: Model used
                - timing: Performance metrics
                - complexity: Query complexity estimate
                - query: Original question
                - timestamp: ISO timestamp

        Raises:
            ValueError: If question is empty.
            Exception: If all models fail.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Start timing
        start_time = time.perf_counter()
        timing = {"embedding": 0, "search": 0, "generation": 0, "total": 0}

        # Select model
        target_model = model if model else self.default_model
        if target_model in MODELS:
            target_model = MODELS[target_model]

        # Analyze query complexity
        complexity = self._analyze_complexity(question)

        # Step 1: Retrieve documents
        search_start = time.perf_counter()
        documents = self.enhanced_search(
            question,
            top_k=top_k,
            rerank=True,
            deduplicate=True,
            use_hybrid=use_hybrid,
        )
        timing["search"] = time.perf_counter() - search_start

        # Step 2: Build context with citations
        context, citation_map = self._build_cited_context(documents)

        # Step 3: Build prompt
        prompt = self._build_production_prompt(question, context, citation_map)

        # Step 4: Generate answer with fallback
        gen_start = time.perf_counter()
        answer, model_used, fallback_used = self._generate_with_fallback(
            prompt, target_model, enable_fallback
        )
        timing["generation"] = time.perf_counter() - gen_start
        timing["total"] = time.perf_counter() - start_time

        # Step 5: Extract cited sources
        cited_sources = self._extract_cited_sources(answer, documents, citation_map)

        # Step 6: Format response
        result = {
            "answer": answer,
            "sources": cited_sources if include_sources else [],
            "model": model_used,
            "timing": timing,
            "complexity": complexity,
            "query": question,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback_used": fallback_used,
            "context_chunks": len(documents),
        }

        # Step 7: Log query
        if log_query and self.enable_logging:
            try:
                self.log_query(result)
            except Exception as e:
                # Don't fail the query if logging fails
                result["logging_error"] = str(e)

        return result

    def _analyze_complexity(self, question: str) -> str:
        """Analyze query complexity: simple, medium, or complex."""
        words = question.split()
        word_count = len(words)

        # Complex indicators
        complex_words = ["compare", "analyze", "explain", "difference", "relationship"]
        has_complex_terms = any(w.lower() in complex_words for w in words)

        if word_count > 15 or has_complex_terms:
            return "complex"
        elif word_count > 8:
            return "medium"
        else:
            return "simple"

    def _build_cited_context(
        self, documents: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, int]]:
        """Build context string with citation markers."""
        if not documents:
            return "No relevant documents found.", {}

        context_parts = []
        citation_map = {}

        for i, doc in enumerate(documents, 1):
            doc_name = doc.get("document_name", "Unknown")
            chunk_idx = doc.get("chunk_index", 0)
            content = doc.get("content", "")
            similarity = doc.get("similarity", 0)

            # Create citation header
            header = f"[Source {i}: {doc_name}, chunk {chunk_idx}] (relevance: {similarity:.2f})"

            context_parts.append(f"{header}\n{content}")
            citation_map[doc.get("id", str(i))] = i

        return "\n\n---\n\n".join(context_parts), citation_map

    def _build_production_prompt(
        self, question: str, context: str, citation_map: Dict[str, int]
    ) -> str:
        """Build production-grade RAG prompt with citation instructions."""
        system_instructions = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer using ONLY information from the CONTEXT below
2. If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."
3. ALWAYS cite your sources using [Source X] format when referencing information
4. Be concise but comprehensive - include all relevant details from the context
5. If multiple sources contain relevant information, synthesize them and cite all
6. Do not make up or infer information not explicitly stated in the context
7. Structure your answer clearly with bullet points or numbered lists when appropriate"""

        return f"""{system_instructions}

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER (remember to cite sources with [Source X]):"""

    def _generate_with_fallback(
        self, prompt: str, primary_model: str, enable_fallback: bool
    ) -> Tuple[str, str, bool]:
        """Generate answer with fallback to alternative models."""
        client = _get_openrouter_client()
        last_error = None

        # Try primary model
        try:
            response = client.chat.completions.create(
                model=primary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                timeout=60.0,
            )
            return response.choices[0].message.content, primary_model, False
        except Exception as e:
            last_error = e

        # Try fallback models
        if enable_fallback:
            for fallback_model in self.fallback_models:
                if fallback_model == primary_model:
                    continue
                try:
                    response = client.chat.completions.create(
                        model=fallback_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=500,
                        timeout=60.0,
                    )
                    return response.choices[0].message.content, fallback_model, True
                except Exception:
                    continue

        raise Exception(f"All models failed. Last error: {last_error}")

    def _extract_cited_sources(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        citation_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Extract and format sources that were cited in the answer."""
        # Find all citation references in the answer
        cited_numbers = set(
            int(m) for m in re.findall(r"\[Source\s*(\d+)\]", answer)
        )

        formatted_sources = []
        for doc in documents:
            citation_num = doc.get("citation_number", 0)
            was_cited = citation_num in cited_numbers

            content = doc.get("content", "")
            formatted_sources.append({
                "id": doc.get("id"),
                "citation_number": citation_num,
                "document": doc.get("document_name", "Unknown"),
                "chunk_index": doc.get("chunk_index", 0),
                "similarity": round(doc.get("similarity", 0), 4),
                "link": doc.get("document_link", ""),
                "content": content,  # Full content for evaluation
                "preview": content[:200] + "..." if len(content) > 200 else content,
                "was_cited": was_cited,
            })

        return formatted_sources

    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================

    def compare_models(
        self,
        question: str,
        models: Optional[List[str]] = None,
        parallel: bool = True,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Compare responses from multiple models side-by-side.

        Args:
            question: The question to answer.
            models: List of models to compare. Defaults to [default, premium, budget].
            parallel: Execute in parallel. Defaults to True.
            timeout: Maximum time for all responses. Defaults to 60s.

        Returns:
            Dictionary containing:
                - query: Original question
                - models_compared: List of models tested
                - results: Dict mapping model to response
                - analysis: Performance analysis
                - sources: Shared source documents
                - timestamp: ISO timestamp
        """
        # Default models
        if models is None:
            models = ["default", "premium", "budget"]

        # Resolve aliases
        resolved_models = []
        for m in models:
            if m in MODELS:
                resolved_models.append(MODELS[m])
            else:
                resolved_models.append(m)

        # Get shared context
        documents = self.enhanced_search(question, top_k=5)
        context, citation_map = self._build_cited_context(documents)
        prompt = self._build_production_prompt(question, context, citation_map)

        # Execute queries
        results = {}
        client = _get_openrouter_client()

        def query_model(model: str) -> Tuple[str, Dict[str, Any]]:
            start = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=timeout,
                )
                latency = int((time.perf_counter() - start) * 1000)

                result = {
                    "answer": response.choices[0].message.content,
                    "latency_ms": latency,
                }

                if hasattr(response, "usage") and response.usage:
                    result["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                return model, result

            except Exception as e:
                latency = int((time.perf_counter() - start) * 1000)
                return model, {"error": str(e), "latency_ms": latency}

        if parallel:
            with ThreadPoolExecutor(max_workers=len(resolved_models)) as executor:
                futures = [executor.submit(query_model, m) for m in resolved_models]
                for future in as_completed(futures, timeout=timeout + 10):
                    try:
                        model, result = future.result()
                        results[model] = result
                    except Exception as e:
                        pass
        else:
            for model in resolved_models:
                _, result = query_model(model)
                results[model] = result

        # Analyze results
        analysis = self._analyze_comparison(results)

        # Format sources
        sources = []
        for doc in documents:
            content = doc.get("content", "")
            sources.append({
                "document": doc.get("document_name", "Unknown"),
                "chunk_index": doc.get("chunk_index", 0),
                "similarity": round(doc.get("similarity", 0), 4),
                "preview": content[:150] + "..." if len(content) > 150 else content,
            })

        return {
            "query": question,
            "models_compared": resolved_models,
            "results": results,
            "analysis": analysis,
            "sources": sources,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _analyze_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model comparison results."""
        valid_results = {k: v for k, v in results.items() if "error" not in v}

        analysis = {
            "fastest_model": None,
            "slowest_model": None,
            "shortest_answer": None,
            "longest_answer": None,
            "cost_comparison": {},
        }

        if not valid_results:
            return analysis

        # Find fastest/slowest
        by_latency = sorted(valid_results.items(), key=lambda x: x[1]["latency_ms"])
        analysis["fastest_model"] = by_latency[0][0]
        analysis["slowest_model"] = by_latency[-1][0]

        # Find shortest/longest answers
        by_length = sorted(
            valid_results.items(), key=lambda x: len(x[1].get("answer", ""))
        )
        analysis["shortest_answer"] = by_length[0][0]
        analysis["longest_answer"] = by_length[-1][0]

        # Calculate costs
        for model, result in valid_results.items():
            if "usage" in result and model in MODEL_INFO:
                info = MODEL_INFO[model]
                usage = result["usage"]
                input_cost = (usage["prompt_tokens"] / 1_000_000) * info.get(
                    "input_cost_per_m", 0
                )
                output_cost = (usage["completion_tokens"] / 1_000_000) * info.get(
                    "output_cost_per_m", 0
                )
                analysis["cost_comparison"][model] = {
                    "input_cost": round(input_cost, 6),
                    "output_cost": round(output_cost, 6),
                    "total_cost": round(input_cost + output_cost, 6),
                }

        return analysis

    # =========================================================================
    # QUERY LOGGING
    # =========================================================================

    def log_query(
        self,
        query_data: Dict[str, Any],
        include_sources: bool = True,
    ) -> Optional[str]:
        """
        Store query and response in database for analytics.

        Args:
            query_data: Query result from self.query().
            include_sources: Whether to include source details. Defaults to True.

        Returns:
            Query log ID if successful, None otherwise.
        """
        try:
            supabase = _get_supabase_client()

            # Prepare data
            question = query_data.get("query", "")
            answer = query_data.get("answer", "")

            # Optional PII redaction
            if self.enable_pii_redaction:
                question = self._redact_pii(question)
                answer = self._redact_pii(answer)

            # Format sources for JSONB
            sources_json = None
            if include_sources and "sources" in query_data:
                sources_json = [
                    {
                        "document": src.get("document"),
                        "chunk": src.get("chunk_index"),
                        "similarity": src.get("similarity"),
                        "cited": src.get("was_cited", False),
                    }
                    for src in query_data.get("sources", [])
                ]

            # Create log entry
            timing = query_data.get("timing", {})
            log_entry = {
                "question": question,
                "answer": answer,
                "model": query_data.get("model", "unknown"),
                "sources": sources_json,
                "response_time": timing.get("total", 0),
                "complexity": query_data.get("complexity", "medium"),
                "fallback_used": query_data.get("fallback_used", False),
            }

            # Insert into database
            response = supabase.table("query_logs").insert(log_entry).execute()

            if response.data:
                return response.data[0].get("id")
            return None

        except Exception as e:
            # Log error but don't raise - logging shouldn't break queries
            print(f"Warning: Failed to log query: {e}")
            return None

    def _redact_pii(self, text: str) -> str:
        """Redact potential PII from text."""
        # Email addresses
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", text)
        # Phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", text)
        # SSN
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
        return text

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze query patterns and system performance.

        Args:
            start_date: Start of analysis period. Defaults to 7 days ago.
            end_date: End of analysis period. Defaults to now.
            limit: Maximum logs to analyze. Defaults to 100.

        Returns:
            Dictionary containing:
                - total_queries: Count of queries
                - date_range: Analysis period
                - performance: Response time metrics
                - models: Model usage statistics
                - popular_queries: Common question patterns
                - insights: Generated insights
        """
        try:
            supabase = _get_supabase_client()

            # Set date range
            if end_date is None:
                end_date = datetime.now(timezone.utc)
            if start_date is None:
                start_date = end_date - timedelta(days=7)

            # Fetch logs
            response = (
                supabase.table("query_logs")
                .select("*")
                .gte("created_at", start_date.isoformat())
                .lte("created_at", end_date.isoformat())
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            logs = response.data or []

            if not logs:
                return {
                    "total_queries": 0,
                    "date_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                    },
                    "performance": {},
                    "models": {},
                    "popular_queries": [],
                    "insights": ["No queries found in the specified date range."],
                }

            # Calculate performance metrics
            response_times = [log.get("response_time", 0) for log in logs if log.get("response_time")]
            performance = {}
            if response_times:
                sorted_times = sorted(response_times)
                performance = {
                    "avg_response_time": round(mean(response_times), 3),
                    "p50_response_time": round(sorted_times[len(sorted_times) // 2], 3),
                    "p95_response_time": round(
                        sorted_times[int(len(sorted_times) * 0.95)], 3
                    )
                    if len(sorted_times) > 1
                    else sorted_times[0],
                    "fastest_query": round(min(response_times), 3),
                    "slowest_query": round(max(response_times), 3),
                }

            # Model usage statistics
            model_stats = defaultdict(lambda: {"count": 0, "latencies": [], "fallback_count": 0})
            for log in logs:
                model = log.get("model", "unknown")
                model_stats[model]["count"] += 1
                if log.get("response_time"):
                    model_stats[model]["latencies"].append(log["response_time"])
                if log.get("fallback_used"):
                    model_stats[model]["fallback_count"] += 1

            models = {}
            for model, stats in model_stats.items():
                models[model] = {
                    "usage_count": stats["count"],
                    "usage_percentage": round((stats["count"] / len(logs)) * 100, 1),
                    "avg_latency": round(mean(stats["latencies"]), 3) if stats["latencies"] else 0,
                    "fallback_rate": round(
                        (stats["fallback_count"] / stats["count"]) * 100, 1
                    )
                    if stats["count"] > 0
                    else 0,
                }

            # Popular queries (simple word frequency)
            query_words = defaultdict(int)
            for log in logs:
                question = log.get("question", "").lower()
                for word in question.split():
                    if len(word) > 3:  # Skip short words
                        query_words[word] += 1

            popular_terms = sorted(query_words.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]

            # Generate insights
            insights = self._generate_insights(len(logs), performance, models, popular_terms)

            return {
                "total_queries": len(logs),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "performance": performance,
                "models": models,
                "popular_queries": [
                    {"term": term, "count": count} for term, count in popular_terms
                ],
                "insights": insights,
            }

        except Exception as e:
            return {
                "error": str(e),
                "total_queries": 0,
                "date_range": {},
                "performance": {},
                "models": {},
                "popular_queries": [],
                "insights": [f"Failed to fetch analytics: {e}"],
            }

    def _generate_insights(
        self,
        total_queries: int,
        performance: Dict[str, Any],
        models: Dict[str, Any],
        popular_terms: List[Tuple[str, int]],
    ) -> List[str]:
        """Generate human-readable insights from analytics."""
        insights = []

        # Query volume insight
        if total_queries > 0:
            insights.append(f"Processed {total_queries} queries in the analysis period.")

        # Performance insights
        if performance:
            avg = performance.get("avg_response_time", 0)
            p95 = performance.get("p95_response_time", 0)
            if avg < 2:
                insights.append(f"Excellent performance: average response time is {avg:.2f}s.")
            elif avg < 3:
                insights.append(f"Good performance: 95% of queries complete in under {p95:.2f}s.")
            else:
                insights.append(f"Performance attention needed: average response time is {avg:.2f}s.")

        # Model insights
        if models:
            most_used = max(models.items(), key=lambda x: x[1]["usage_count"])
            insights.append(
                f"Most used model: {most_used[0]} ({most_used[1]['usage_percentage']:.1f}% of queries)."
            )

            # Check for high fallback rates
            for model, stats in models.items():
                if stats.get("fallback_rate", 0) > 10:
                    insights.append(
                        f"High fallback rate for {model}: {stats['fallback_rate']:.1f}%. "
                        "Consider reviewing model availability."
                    )

        # Popular topic insights
        if popular_terms:
            top_terms = [term for term, _ in popular_terms[:3]]
            insights.append(f"Top query topics: {', '.join(top_terms)}.")

        return insights


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """Interactive CLI for production Q&A system."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="DocuMind Production Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.documind.rag.production_qa "What is the vacation policy?"
  python -m src.documind.rag.production_qa "Benefits overview" --model premium
  python -m src.documind.rag.production_qa "Compare policies" --compare
  python -m src.documind.rag.production_qa --analytics
  python -m src.documind.rag.production_qa --interactive
        """,
    )

    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument(
        "--model", "-m", default="default", help="Model to use (default, premium, budget)"
    )
    parser.add_argument(
        "--compare", "-c", action="store_true", help="Compare multiple models"
    )
    parser.add_argument(
        "--models", nargs="+", help="Models to compare (with --compare)"
    )
    parser.add_argument(
        "--analytics", "-a", action="store_true", help="Show query analytics"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive Q&A mode"
    )
    parser.add_argument(
        "--json", "-j", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Disable query logging"
    )

    args = parser.parse_args()

    # Initialize system
    qa = ProductionQA(enable_logging=not args.no_log)

    print("\n" + "=" * 70)
    print("  DocuMind Production Q&A System")
    print("=" * 70)

    try:
        # Analytics mode
        if args.analytics:
            print("\n  Fetching analytics...")
            analytics = qa.get_analytics()

            if args.json:
                print(json.dumps(analytics, indent=2))
            else:
                _display_analytics(analytics)
            return

        # Interactive mode
        if args.interactive:
            print("\n  Interactive Q&A Mode")
            print("  Type 'quit' or 'exit' to end, 'analytics' to see stats")
            print("-" * 70)

            while True:
                try:
                    question = input("\n  Question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not question:
                    continue
                if question.lower() in ["quit", "exit", "q"]:
                    break
                if question.lower() == "analytics":
                    analytics = qa.get_analytics()
                    _display_analytics(analytics)
                    continue

                print("\n  Thinking...")
                result = qa.query(question)
                _display_result(result)

            print("\n  Goodbye!")
            return

        # Require query for other modes
        if not args.query:
            parser.print_help()
            print("\n  Error: Please provide a query or use --interactive/--analytics")
            return

        # Compare mode
        if args.compare:
            print(f"\n  Query: {args.query}")
            print("  Comparing models...")

            models = args.models or ["default", "premium", "budget"]
            result = qa.compare_models(args.query, models=models)

            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                _display_comparison(result)
            return

        # Single query mode
        print(f"\n  Query: {args.query}")
        print(f"  Model: {args.model}")
        print("-" * 70)

        result = qa.query(args.query, model=args.model)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            _display_result(result)

    except ValueError as e:
        print(f"\n  Configuration Error: {e}")
        print("\n  Required environment variables:")
        print("    - OPENROUTER_API_KEY")
        print("    - OPENAI_API_KEY")
        print("    - SUPABASE_URL")
        print("    - SUPABASE_ANON_KEY")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise

    print("\n" + "=" * 70)


def _display_result(result: Dict[str, Any]) -> None:
    """Display formatted query result."""
    print(f"\n  Answer:\n")

    # Format answer with proper indentation
    answer = result.get("answer", "No answer generated")
    for line in answer.split("\n"):
        print(f"    {line}")

    # Display sources
    sources = result.get("sources", [])
    if sources:
        print(f"\n  {'=' * 66}")
        print(f"  Sources ({len(sources)}):")
        for src in sources:
            cited = "*" if src.get("was_cited") else " "
            print(f"\n    [{cited}] {src.get('document', 'Unknown')} (chunk {src.get('chunk_index', 0)})")
            print(f"        Similarity: {src.get('similarity', 0):.4f}")
            preview = src.get("preview", "")[:80]
            print(f"        Preview: {preview}...")

    # Display timing
    timing = result.get("timing", {})
    if timing:
        print(f"\n  {'=' * 66}")
        print(f"  Performance:")
        print(f"    Total: {timing.get('total', 0)*1000:.0f}ms")
        print(f"    Search: {timing.get('search', 0)*1000:.0f}ms")
        print(f"    Generation: {timing.get('generation', 0)*1000:.0f}ms")

    if result.get("fallback_used"):
        print(f"\n  Note: Used fallback model due to primary model failure")


def _display_comparison(result: Dict[str, Any]) -> None:
    """Display model comparison results."""
    print(f"\n  Sources ({len(result.get('sources', []))}):")
    for i, src in enumerate(result.get("sources", []), 1):
        print(f"    [{i}] {src.get('document')} (sim: {src.get('similarity', 0):.2f})")

    for model, model_result in result.get("results", {}).items():
        info = MODEL_INFO.get(model, {})
        name = info.get("name", model)

        print(f"\n  {'=' * 66}")
        print(f"  Model: {name}")
        print("-" * 70)

        if "error" in model_result:
            print(f"    ERROR: {model_result['error']}")
        else:
            print(f"    Latency: {model_result.get('latency_ms', 0)}ms")
            if "usage" in model_result:
                usage = model_result["usage"]
                print(f"    Tokens: {usage.get('total_tokens', 0)}")

            answer = model_result.get("answer", "")[:500]
            print(f"\n    Answer:\n")
            for line in answer.split("\n"):
                print(f"      {line}")

    # Analysis
    analysis = result.get("analysis", {})
    if analysis:
        print(f"\n  {'=' * 66}")
        print("  Analysis:")
        print(f"    Fastest: {analysis.get('fastest_model', 'N/A')}")
        print(f"    Slowest: {analysis.get('slowest_model', 'N/A')}")


def _display_analytics(analytics: Dict[str, Any]) -> None:
    """Display analytics results."""
    print(f"\n  Query Analytics")
    print(f"  {'-' * 64}")

    print(f"\n  Total Queries: {analytics.get('total_queries', 0)}")

    date_range = analytics.get("date_range", {})
    if date_range:
        print(f"  Period: {date_range.get('start', 'N/A')[:10]} to {date_range.get('end', 'N/A')[:10]}")

    # Performance
    perf = analytics.get("performance", {})
    if perf:
        print(f"\n  Performance:")
        print(f"    Average Response: {perf.get('avg_response_time', 0):.2f}s")
        print(f"    P95 Response: {perf.get('p95_response_time', 0):.2f}s")
        print(f"    Fastest: {perf.get('fastest_query', 0):.2f}s")
        print(f"    Slowest: {perf.get('slowest_query', 0):.2f}s")

    # Models
    models = analytics.get("models", {})
    if models:
        print(f"\n  Model Usage:")
        for model, stats in models.items():
            print(f"    {model}: {stats.get('usage_count', 0)} queries ({stats.get('usage_percentage', 0):.1f}%)")

    # Popular queries
    popular = analytics.get("popular_queries", [])
    if popular:
        print(f"\n  Popular Query Terms:")
        for item in popular[:5]:
            print(f"    - {item.get('term', '')}: {item.get('count', 0)} occurrences")

    # Insights
    insights = analytics.get("insights", [])
    if insights:
        print(f"\n  Insights:")
        for insight in insights:
            print(f"    * {insight}")


if __name__ == "__main__":
    main()
