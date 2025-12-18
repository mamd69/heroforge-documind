"""
DocuMind Q&A Pipeline Module

Provides question-answering capabilities using RAG (Retrieval-Augmented Generation)
with OpenRouter for LLM inference. Supports multiple models optimized for grounded
Q&A tasks with source citations.

Models selected based on research for RAG optimization:
- Low latency and high throughput
- Strong instruction following
- Minimal hallucination on grounded tasks
- Cost efficiency for production use
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .search import search_documents, get_query_embedding

# Load environment variables
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# =============================================================================
# MODEL CONFIGURATION - Research-optimized models for RAG Q&A tasks
# =============================================================================

MODELS: Dict[str, str] = {
    # Default - fastest & cheapest, optimized for high-throughput RAG
    "default": "google/gemini-2.5-flash-lite",

    # Premium quality - excellent instruction following, low hallucination
    "premium": "anthropic/claude-3.5-haiku",

    # Budget alternative - reliable baseline with good grounding
    "budget": "openai/gpt-4o-mini",

    # Open source option - competitive quality at very low cost
    "opensource": "deepseek/deepseek-chat",

    # High quality fallback - step up from lite when needed
    "quality": "google/gemini-2.5-flash",
}

# Model metadata for display and cost tracking
MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "google/gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash Lite",
        "input_cost_per_m": 0.10,
        "output_cost_per_m": 0.40,
        "context_window": 1050000,
        "strengths": ["Ultra-low latency", "Huge context", "Cost-efficient"],
    },
    "anthropic/claude-3.5-haiku": {
        "name": "Claude 3.5 Haiku",
        "input_cost_per_m": 0.80,
        "output_cost_per_m": 4.00,
        "context_window": 200000,
        "strengths": ["Instruction following", "Low hallucination", "Grounded generation"],
    },
    "openai/gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "input_cost_per_m": 0.15,
        "output_cost_per_m": 0.60,
        "context_window": 128000,
        "strengths": ["Reliable baseline", "Good grounding", "Production-ready"],
    },
    "deepseek/deepseek-chat": {
        "name": "DeepSeek V3",
        "input_cost_per_m": 0.28,
        "output_cost_per_m": 0.42,
        "context_window": 64000,
        "strengths": ["Best open-source", "Low cost", "Strong instruction following"],
    },
    "google/gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "input_cost_per_m": 0.30,
        "output_cost_per_m": 2.50,
        "context_window": 1050000,
        "strengths": ["Higher capability", "Good speed", "Large context"],
    },
}

# =============================================================================
# OPENROUTER CLIENT INITIALIZATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Lazy-initialized client
_openrouter_client: Optional[OpenAI] = None


def _get_openrouter_client() -> OpenAI:
    """
    Get or create OpenRouter client instance.

    Uses the OpenAI-compatible API with OpenRouter's base URL.

    Returns:
        OpenAI client configured for OpenRouter.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    global _openrouter_client
    if _openrouter_client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Get your API key at https://openrouter.ai/keys"
            )
        _openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
    return _openrouter_client


# =============================================================================
# CONTEXT ASSEMBLY
# =============================================================================

def assemble_context(
    documents: List[Dict[str, Any]],
    max_tokens: int = 3000
) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Creates a structured context with source citations that the LLM can
    reference in its answer. Documents are separated by dividers and include
    metadata for traceability.

    Args:
        documents: List of document dictionaries from search_documents().
            Each should contain: content, document_name, chunk_index, similarity
        max_tokens: Maximum estimated tokens for context. Uses ~4 chars per token.
            Defaults to 3000 tokens (~12000 characters).

    Returns:
        Formatted context string with source citations like:
        "[Source 1: employee_handbook.txt, chunk 2]"

    Example:
        >>> docs = search_documents("vacation policy", top_k=3)
        >>> context = assemble_context(docs, max_tokens=2000)
        >>> print(context[:200])
    """
    if not documents:
        return "No relevant documents found."

    max_chars = max_tokens * 4  # ~4 characters per token estimate
    context_parts: List[str] = []
    current_chars = 0

    for i, doc in enumerate(documents, 1):
        # Extract document info
        doc_name = doc.get("document_name", "Unknown")
        chunk_index = doc.get("chunk_index", 0)
        content = doc.get("content", "")
        similarity = doc.get("similarity", 0.0)

        # Build source citation header
        source_header = f"[Source {i}: {doc_name}, chunk {chunk_index}] (similarity: {similarity:.2f})"

        # Format document section
        doc_section = f"{source_header}\n{content}"
        section_chars = len(doc_section) + 5  # +5 for separator

        # Check if adding this would exceed limit
        if current_chars + section_chars > max_chars:
            # Try to include a truncated version
            remaining_chars = max_chars - current_chars - len(source_header) - 50
            if remaining_chars > 100:
                truncated_content = content[:remaining_chars] + "... [truncated]"
                doc_section = f"{source_header}\n{truncated_content}"
                context_parts.append(doc_section)
            break

        context_parts.append(doc_section)
        current_chars += section_chars

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_qa_prompt(query: str, context: str) -> str:
    """
    Create a RAG prompt with instructions for grounded Q&A.

    Builds a structured prompt that instructs the LLM to:
    - Answer using ONLY the provided context
    - Admit when information is not available
    - Include source references using [Source X] format
    - Be concise but comprehensive

    Args:
        query: The user's question.
        context: Assembled context from assemble_context().

    Returns:
        Complete prompt string ready for LLM inference.

    Example:
        >>> context = assemble_context(docs)
        >>> prompt = build_qa_prompt("What is the vacation policy?", context)
    """
    system_instructions = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. Answer using ONLY information from the CONTEXT below
2. If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."
3. ALWAYS cite your sources using [Source X] format when referencing information
4. Be concise but comprehensive - include all relevant details from the context
5. If multiple sources contain relevant information, synthesize them and cite all
6. Do not make up or infer information not explicitly stated in the context"""

    prompt = f"""{system_instructions}

CONTEXT:
{context}

---

QUESTION: {query}

ANSWER:"""

    return prompt


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def generate_answer(
    query: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    top_k: int = 5,
    context_max_tokens: int = 3000
) -> Dict[str, Any]:
    """
    Generate an answer to a query using RAG pipeline.

    Complete pipeline:
    1. Retrieve relevant documents using semantic search
    2. Assemble context with source citations
    3. Build grounded Q&A prompt
    4. Generate answer using specified LLM via OpenRouter
    5. Format response with sources and metadata

    Args:
        query: The user's question.
        model: OpenRouter model ID. Defaults to MODELS["default"]
            (google/gemini-2.5-flash-lite).
        temperature: Sampling temperature (0-1). Lower = more deterministic.
            Defaults to 0.1 for grounded answers.
        max_tokens: Maximum tokens in the response. Defaults to 500.
        top_k: Number of documents to retrieve. Defaults to 5.
        context_max_tokens: Maximum tokens for context. Defaults to 3000.

    Returns:
        Dictionary containing:
            - answer: The generated answer text
            - sources: List of source dictionaries with id, document, chunk,
                similarity, and preview
            - query: Original query
            - model: Model ID used
            - context_chunks: Number of context chunks used
            - timestamp: ISO format timestamp
            - usage: Token usage statistics (if available)

    Raises:
        ValueError: If API keys are not configured.
        Exception: If LLM inference fails.

    Example:
        >>> result = generate_answer("What is the vacation policy?")
        >>> print(result["answer"])
        >>> for source in result["sources"]:
        ...     print(f"  - {source['document']}: {source['preview'][:50]}...")
    """
    # Default model
    if model is None:
        model = MODELS["default"]

    # Resolve model alias if provided
    if model in MODELS:
        model = MODELS[model]

    # Step 1: Retrieve documents
    documents = search_documents(query, top_k=top_k, similarity_threshold=0.1)

    # Step 2: Assemble context
    context = assemble_context(documents, max_tokens=context_max_tokens)

    # Step 3: Build prompt
    prompt = build_qa_prompt(query, context)

    # Step 4: Generate answer via OpenRouter
    client = _get_openrouter_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,  # 60 second timeout
        )
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "auth" in error_msg.lower():
            raise ValueError(
                f"OpenRouter authentication failed. Please check your OPENROUTER_API_KEY. "
                f"Get a new key at https://openrouter.ai/keys\nError: {error_msg}"
            )
        raise

    answer = response.choices[0].message.content

    # Step 5: Format sources
    sources = []
    for doc in documents:
        content = doc.get("content", "")
        sources.append({
            "id": doc.get("id"),
            "document": doc.get("document_name", "Unknown"),
            "chunk": doc.get("chunk_index", 0),
            "similarity": round(doc.get("similarity", 0.0), 4),
            "preview": content[:200] + "..." if len(content) > 200 else content,
        })

    # Build response
    result = {
        "answer": answer,
        "sources": sources,
        "query": query,
        "model": model,
        "context_chunks": len(documents),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add usage if available
    if hasattr(response, "usage") and response.usage:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return result


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    query: str,
    models: Optional[List[str]] = None,
    temperature: float = 0.1,
    max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Compare responses from multiple models for the same query.

    Useful for A/B testing and evaluating model quality for specific
    use cases. Each model receives the same context and prompt.

    Args:
        query: The user's question.
        models: List of model IDs or aliases from MODELS dict.
            Defaults to ["default", "premium", "budget"].
        temperature: Sampling temperature. Defaults to 0.1.
        max_tokens: Maximum response tokens. Defaults to 500.

    Returns:
        Dictionary containing:
            - query: Original query
            - timestamp: ISO format timestamp
            - results: Dict mapping model ID to response dict with:
                - answer: Generated answer
                - usage: Token usage (if available)
                - error: Error message (if failed)
                - latency_ms: Response time in milliseconds
            - sources: Shared sources used for all models

    Example:
        >>> comparison = compare_models(
        ...     "What is the vacation policy?",
        ...     models=["default", "premium", "opensource"]
        ... )
        >>> for model, result in comparison["results"].items():
        ...     if "error" not in result:
        ...         print(f"{model}: {result['answer'][:100]}...")
    """
    import time

    # Default models to compare
    if models is None:
        models = ["default", "premium", "budget"]

    # Resolve model aliases
    resolved_models = []
    for m in models:
        if m in MODELS:
            resolved_models.append(MODELS[m])
        else:
            resolved_models.append(m)

    # Get documents once (shared context)
    documents = search_documents(query, top_k=5, similarity_threshold=0.1)
    context = assemble_context(documents, max_tokens=3000)
    prompt = build_qa_prompt(query, context)

    # Format shared sources
    sources = []
    for doc in documents:
        content = doc.get("content", "")
        sources.append({
            "id": doc.get("id"),
            "document": doc.get("document_name", "Unknown"),
            "chunk": doc.get("chunk_index", 0),
            "similarity": round(doc.get("similarity", 0.0), 4),
            "preview": content[:200] + "..." if len(content) > 200 else content,
        })

    # Query each model
    client = _get_openrouter_client()
    results: Dict[str, Dict[str, Any]] = {}

    for model in resolved_models:
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,  # 60 second timeout
            )

            latency_ms = int((time.time() - start_time) * 1000)

            result_entry: Dict[str, Any] = {
                "answer": response.choices[0].message.content,
                "latency_ms": latency_ms,
            }

            if hasattr(response, "usage") and response.usage:
                result_entry["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            results[model] = result_entry

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            results[model] = {
                "error": str(e),
                "latency_ms": latency_ms,
            }

    return {
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "sources": sources,
        "context_chunks": len(documents),
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DocuMind Q&A Pipeline - RAG-powered question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.documind.rag.qa_pipeline "What is the vacation policy?"
  python -m src.documind.rag.qa_pipeline "How do I request time off?" --model premium
  python -m src.documind.rag.qa_pipeline "Benefits overview" --compare
  python -m src.documind.rag.qa_pipeline --list-models
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to answer"
    )

    parser.add_argument(
        "--model", "-m",
        default="default",
        help=f"Model to use (default: default). Options: {', '.join(MODELS.keys())}"
    )

    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare responses from multiple models"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to compare (with --compare)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens in response (default: 500)"
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON response"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("\n" + "=" * 70)
        print("Available Models for RAG Q&A")
        print("=" * 70)

        for alias, model_id in MODELS.items():
            info = MODEL_INFO.get(model_id, {})
            name = info.get("name", model_id)
            input_cost = info.get("input_cost_per_m", "?")
            output_cost = info.get("output_cost_per_m", "?")
            strengths = info.get("strengths", [])

            print(f"\n  {alias.upper()} ({name})")
            print(f"    Model ID: {model_id}")
            print(f"    Cost: ${input_cost}/M input, ${output_cost}/M output")
            if strengths:
                print(f"    Strengths: {', '.join(strengths)}")

        print("\n" + "=" * 70)
        exit(0)

    # Require query for other modes
    if not args.query:
        parser.print_help()
        print("\nError: Please provide a query or use --list-models")
        exit(1)

    print("\n" + "=" * 70)
    print("DocuMind Q&A Pipeline")
    print("=" * 70)

    try:
        if args.compare:
            # Compare models mode
            print(f"\nQuery: {args.query}")
            print(f"Comparing models...")
            print("-" * 70)

            models_to_compare = args.models or ["default", "premium", "budget"]
            result = compare_models(
                args.query,
                models=models_to_compare,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                # Display sources
                print(f"\nSources ({len(result['sources'])} chunks):")
                for i, src in enumerate(result["sources"], 1):
                    print(f"  [{i}] {src['document']} (chunk {src['chunk']}, sim: {src['similarity']:.2f})")

                # Display each model's response
                for model_id, model_result in result["results"].items():
                    info = MODEL_INFO.get(model_id, {})
                    name = info.get("name", model_id)

                    print(f"\n{'=' * 70}")
                    print(f"Model: {name}")
                    print(f"ID: {model_id}")
                    print("-" * 70)

                    if "error" in model_result:
                        print(f"ERROR: {model_result['error']}")
                    else:
                        print(f"Latency: {model_result['latency_ms']}ms")
                        if "usage" in model_result:
                            usage = model_result["usage"]
                            print(f"Tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out")
                        print(f"\nAnswer:\n{model_result['answer']}")

        else:
            # Single model mode
            print(f"\nQuery: {args.query}")
            print(f"Model: {args.model}")
            print("-" * 70)

            result = generate_answer(
                args.query,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\nAnswer:\n{result['answer']}")

                print(f"\n{'=' * 70}")
                print("Sources:")
                for i, src in enumerate(result["sources"], 1):
                    print(f"\n  [{i}] {src['document']} (chunk {src['chunk']})")
                    print(f"      Similarity: {src['similarity']:.4f}")
                    print(f"      Preview: {src['preview'][:100]}...")

                if "usage" in result:
                    usage = result["usage"]
                    print(f"\n{'=' * 70}")
                    print(f"Token Usage: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")

        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70 + "\n")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nRequired environment variables:")
        print("  - OPENROUTER_API_KEY (get at https://openrouter.ai/keys)")
        print("  - OPENAI_API_KEY (for embeddings)")
        print("  - SUPABASE_URL")
        print("  - SUPABASE_ANON_KEY")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise
