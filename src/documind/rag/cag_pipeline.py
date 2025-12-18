"""
DocuMind CAG (Context-Augmented Generation) Pipeline Module

CAG is a simpler alternative to RAG that loads ALL documents directly from the
filesystem into the context window. No database, no embeddings, no retrieval.

Key differences from RAG:
- RAG: Query → Embed → Search DB → Retrieve chunks → Generate
- CAG: Load ALL files → Stuff into context → Generate

When to use CAG:
- Small document sets (< 100KB total)
- When you need ALL context available (no retrieval errors)
- Simpler deployment (no vector DB required)
- Development/testing scenarios

Limitations:
- Context window limits scalability
- Cost increases with document size
- No semantic filtering of irrelevant content
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# =============================================================================
# MODEL CONFIGURATION - Same defaults as RAG pipeline for consistency
# =============================================================================

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"

# Context window warning threshold (100KB)
CONTEXT_WARNING_THRESHOLD = 100 * 1024  # 100KB in bytes

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
    CAG only needs the LLM client - no database or embedding clients.

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
# DOCUMENT LOADING - Direct filesystem access (NO DATABASE)
# =============================================================================

def load_all_documents(docs_dir: str = "demo-docs") -> str:
    """
    Load ALL .txt files from a directory into a single context string.

    This is the core CAG function - it reads files directly from the filesystem
    without any database or embedding operations. All documents are concatenated
    into a single string that gets stuffed into the LLM context.

    Args:
        docs_dir: Directory path containing .txt files to load.
            Defaults to "demo-docs" (relative to current working directory).

    Returns:
        Concatenated string of all documents formatted as:
        "[Document: filename.txt]
        <file contents>

        ---

        [Document: another.txt]
        <file contents>"

    Warns:
        Prints warning if total content exceeds 100KB (context window concern).

    Example:
        >>> context = load_all_documents("demo-docs")
        >>> print(context[:200])
        [Document: employee_handbook.txt]
        Welcome to Acme Corp...

        >>> # Use with custom directory
        >>> context = load_all_documents("/path/to/my/docs")
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    if not docs_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {docs_dir}")

    # Find all .txt files
    txt_files = sorted(docs_path.glob("*.txt"))

    if not txt_files:
        return "No .txt documents found in the specified directory."

    # Load and format each document
    document_sections = []
    total_chars = 0

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            filename = file_path.name

            # Format document section
            section = f"[Document: {filename}]\n{content}"
            document_sections.append(section)
            total_chars += len(section)

        except Exception as e:
            # Log error but continue with other files
            print(f"Warning: Could not read {file_path.name}: {e}")

    # Join with separator
    full_context = "\n\n---\n\n".join(document_sections)

    # Warn if context is large
    total_bytes = len(full_context.encode("utf-8"))
    if total_bytes > CONTEXT_WARNING_THRESHOLD:
        print(
            f"⚠️  Warning: Total document size is {total_bytes / 1024:.1f}KB "
            f"(exceeds {CONTEXT_WARNING_THRESHOLD / 1024:.0f}KB threshold). "
            "This may approach context window limits for some models."
        )

    return full_context


# =============================================================================
# ANSWER GENERATION - Full context approach
# =============================================================================

def generate_answer_cag(
    query: str,
    docs_dir: str = "demo-docs",
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Generate an answer using CAG (Context-Augmented Generation).

    Unlike RAG which retrieves relevant chunks, CAG loads ALL documents
    into the context and lets the LLM find relevant information.

    Pipeline:
    1. Load ALL documents from docs_dir (no retrieval)
    2. Stuff full context into prompt
    3. Generate answer with instructions to cite sources
    4. Return answer with metadata

    Args:
        query: The user's question.
        docs_dir: Directory containing .txt documents to load.
            Defaults to "demo-docs".
        model: OpenRouter model ID. Defaults to google/gemini-2.5-flash-lite.
        temperature: Sampling temperature (0-1). Lower = more deterministic.
            Defaults to 0.1 for grounded answers.
        max_tokens: Maximum tokens in the response. Defaults to 500.

    Returns:
        Dictionary containing:
            - answer: The generated answer text
            - method: "CAG" (to distinguish from RAG responses)
            - docs_loaded: Number of documents loaded
            - total_chars: Total characters in context
            - query: Original query
            - model: Model ID used
            - timestamp: ISO format timestamp
            - usage: Token usage statistics (if available)

    Raises:
        ValueError: If OPENROUTER_API_KEY is not configured.
        FileNotFoundError: If docs_dir doesn't exist.
        Exception: If LLM inference fails.

    Example:
        >>> result = generate_answer_cag("What is the vacation policy?")
        >>> print(result["answer"])
        >>> print(f"Loaded {result['docs_loaded']} documents")
        >>> print(f"Method: {result['method']}")  # "CAG"

        >>> # Compare with RAG
        >>> from .qa_pipeline import generate_answer
        >>> rag_result = generate_answer("What is the vacation policy?")
        >>> cag_result = generate_answer_cag("What is the vacation policy?")
    """
    # Default model (same as RAG pipeline)
    if model is None:
        model = DEFAULT_MODEL

    # Step 1: Load ALL documents (this is the CAG difference - no retrieval!)
    full_context = load_all_documents(docs_dir)

    # Count documents loaded
    docs_loaded = full_context.count("[Document:")

    # Step 2: Build prompt with full context
    system_instructions = """You are a helpful assistant that answers questions based ONLY on the provided documents.

IMPORTANT RULES:
1. Answer using ONLY information from the DOCUMENTS below
2. If the answer is not in the documents, say "I don't have enough information in the provided documents to answer this question."
3. ALWAYS cite your sources by referencing the document name (e.g., "According to employee_handbook.txt...")
4. Be concise but comprehensive - include all relevant details
5. If multiple documents contain relevant information, synthesize them and cite all
6. Do not make up or infer information not explicitly stated in the documents"""

    prompt = f"""{system_instructions}

DOCUMENTS:
{full_context}

---

QUESTION: {query}

ANSWER:"""

    # Step 3: Generate answer via OpenRouter
    client = _get_openrouter_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60.0,
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

    # Step 4: Build response
    result = {
        "answer": answer,
        "method": "CAG",
        "docs_loaded": docs_loaded,
        "total_chars": len(full_context),
        "query": query,
        "model": model,
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
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="DocuMind CAG Pipeline - Context-Augmented Generation (no database)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CAG vs RAG:
  CAG loads ALL documents into context (simple, no DB, limited by context window)
  RAG retrieves relevant chunks via embeddings (scalable, requires vector DB)

Examples:
  python -m src.documind.rag.cag_pipeline "What is the vacation policy?"
  python -m src.documind.rag.cag_pipeline "Benefits overview" --docs-dir /path/to/docs
  python -m src.documind.rag.cag_pipeline "How many sick days?" --json
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to answer"
    )

    parser.add_argument(
        "--docs-dir", "-d",
        default="demo-docs",
        help="Directory containing .txt documents (default: demo-docs)"
    )

    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model to use (default: {DEFAULT_MODEL})"
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
        "--list-docs",
        action="store_true",
        help="List documents in directory and exit"
    )

    args = parser.parse_args()

    # List docs mode
    if args.list_docs:
        docs_path = Path(args.docs_dir)
        if not docs_path.exists():
            print(f"Error: Directory not found: {args.docs_dir}")
            exit(1)

        txt_files = sorted(docs_path.glob("*.txt"))
        print(f"\n{'=' * 60}")
        print(f"Documents in: {args.docs_dir}")
        print("=" * 60)

        if not txt_files:
            print("  No .txt files found")
        else:
            total_size = 0
            for f in txt_files:
                size = f.stat().st_size
                total_size += size
                print(f"  {f.name:<30} {size:>8,} bytes")
            print("-" * 60)
            print(f"  {'TOTAL':<30} {total_size:>8,} bytes ({total_size/1024:.1f}KB)")

        print("=" * 60 + "\n")
        exit(0)

    # Require query for answer mode
    if not args.query:
        parser.print_help()
        print("\nError: Please provide a query or use --list-docs")
        exit(1)

    print("\n" + "=" * 60)
    print("DocuMind CAG Pipeline (Context-Augmented Generation)")
    print("=" * 60)

    try:
        print(f"\nQuery: {args.query}")
        print(f"Docs Dir: {args.docs_dir}")
        print(f"Model: {args.model or DEFAULT_MODEL}")
        print("-" * 60)
        print("Loading ALL documents into context (no retrieval)...")

        result = generate_answer_cag(
            args.query,
            docs_dir=args.docs_dir,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nDocuments Loaded: {result['docs_loaded']}")
            print(f"Context Size: {result['total_chars']:,} chars ({result['total_chars']/1024:.1f}KB)")

            print(f"\n{'=' * 60}")
            print("Answer:")
            print("-" * 60)
            print(result["answer"])

            if "usage" in result:
                usage = result["usage"]
                print(f"\n{'=' * 60}")
                print(f"Token Usage: {usage['prompt_tokens']:,} prompt + "
                      f"{usage['completion_tokens']:,} completion = "
                      f"{usage['total_tokens']:,} total")

        print("\n" + "=" * 60)
        print("Done! (Method: CAG - full context, no retrieval)")
        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        print(f"\nFile Error: {e}")
        print(f"Make sure the docs directory exists: {args.docs_dir}")
        exit(1)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nRequired environment variable:")
        print("  - OPENROUTER_API_KEY (get at https://openrouter.ai/keys)")
        print("\nNote: CAG does NOT require Supabase or embedding credentials!")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise
