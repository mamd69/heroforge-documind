"""
DocuMind RAG vs CAG Comparison Tool

Compares Retrieval-Augmented Generation (RAG) vs Context-Augmented Generation (CAG)
approaches to demonstrate the trade-offs between the two methods.

RAG: Query → Embed → Search DB → Retrieve chunks → Generate
CAG: Load ALL files → Stuff into context → Generate

Run with: python -m src.documind.rag.compare
"""

import time
from typing import Dict, Any, Optional

from .qa_pipeline import generate_answer as rag_answer
from .cag_pipeline import generate_answer_cag as cag_answer


def compare_approaches(
    query: str,
    docs_dir: str = "demo-docs",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare RAG and CAG approaches for the same query.

    Executes both pipelines, measures latency, and provides a side-by-side
    comparison of results and performance characteristics.

    Args:
        query: The question to answer.
        docs_dir: Directory containing documents for CAG (default: demo-docs).
        model: Model to use for both approaches. Defaults to pipeline defaults.

    Returns:
        Dictionary with comparison results including both answers and metrics.
    """
    print("\n" + "=" * 80)
    print("  RAG vs CAG COMPARISON")
    print("=" * 80)
    print(f"\n  Query: \"{query}\"")
    print("-" * 80)

    results: Dict[str, Any] = {
        "query": query,
        "rag": {},
        "cag": {},
    }

    # ==========================================================================
    # TEST RAG (Retrieval-Augmented Generation)
    # ==========================================================================
    print("\n[1/2] Testing RAG (Retrieval-Augmented Generation)...")
    print("      Pipeline: Query → Embed → Search DB → Retrieve chunks → Generate")

    rag_start = time.perf_counter()
    try:
        rag_result = rag_answer(query, model=model)
        rag_latency = time.perf_counter() - rag_start

        results["rag"] = {
            "answer": rag_result["answer"],
            "latency_ms": int(rag_latency * 1000),
            "context_chunks": rag_result.get("context_chunks", 0),
            "sources": rag_result.get("sources", []),
            "model": rag_result.get("model", "unknown"),
            "usage": rag_result.get("usage", {}),
            "success": True,
        }
        print(f"      ✓ RAG completed in {results['rag']['latency_ms']}ms")
        print(f"      ✓ Retrieved {results['rag']['context_chunks']} chunks")

    except Exception as e:
        rag_latency = time.perf_counter() - rag_start
        results["rag"] = {
            "error": str(e),
            "latency_ms": int(rag_latency * 1000),
            "success": False,
        }
        print(f"      ✗ RAG failed: {e}")

    # ==========================================================================
    # TEST CAG (Context-Augmented Generation)
    # ==========================================================================
    print("\n[2/2] Testing CAG (Context-Augmented Generation)...")
    print("      Pipeline: Load ALL files → Stuff into context → Generate")

    cag_start = time.perf_counter()
    try:
        cag_result = cag_answer(query, docs_dir=docs_dir, model=model)
        cag_latency = time.perf_counter() - cag_start

        results["cag"] = {
            "answer": cag_result["answer"],
            "latency_ms": int(cag_latency * 1000),
            "docs_loaded": cag_result.get("docs_loaded", 0),
            "total_chars": cag_result.get("total_chars", 0),
            "model": cag_result.get("model", "unknown"),
            "usage": cag_result.get("usage", {}),
            "success": True,
        }
        print(f"      ✓ CAG completed in {results['cag']['latency_ms']}ms")
        print(f"      ✓ Loaded {results['cag']['docs_loaded']} documents ({results['cag']['total_chars']:,} chars)")

    except Exception as e:
        cag_latency = time.perf_counter() - cag_start
        results["cag"] = {
            "error": str(e),
            "latency_ms": int(cag_latency * 1000),
            "success": False,
        }
        print(f"      ✗ CAG failed: {e}")

    # ==========================================================================
    # DISPLAY RESULTS SIDE-BY-SIDE
    # ==========================================================================
    _display_comparison(results)

    return results


def _display_comparison(results: Dict[str, Any]) -> None:
    """Display formatted comparison results."""

    print("\n" + "=" * 80)
    print("  ANSWERS COMPARISON")
    print("=" * 80)

    # RAG Answer
    print("\n┌─ RAG ANSWER " + "─" * 66 + "┐")
    if results["rag"].get("success"):
        answer = results["rag"]["answer"]
        # Wrap long lines for display
        wrapped = _wrap_text(answer, width=76)
        for line in wrapped.split("\n"):
            print(f"│ {line:<76} │")
    else:
        error = results["rag"].get("error", "Unknown error")
        print(f"│ ERROR: {error:<68} │")
    print("└" + "─" * 78 + "┘")

    # CAG Answer
    print("\n┌─ CAG ANSWER " + "─" * 66 + "┐")
    if results["cag"].get("success"):
        answer = results["cag"]["answer"]
        wrapped = _wrap_text(answer, width=76)
        for line in wrapped.split("\n"):
            print(f"│ {line:<76} │")
    else:
        error = results["cag"].get("error", "Unknown error")
        print(f"│ ERROR: {error:<68} │")
    print("└" + "─" * 78 + "┘")

    # ==========================================================================
    # PERFORMANCE METRICS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PERFORMANCE METRICS")
    print("=" * 80)

    rag = results["rag"]
    cag = results["cag"]

    print("\n  ┌────────────────────┬───────────────────┬───────────────────┐")
    print("  │ Metric             │ RAG               │ CAG               │")
    print("  ├────────────────────┼───────────────────┼───────────────────┤")

    # Latency
    rag_lat = f"{rag.get('latency_ms', 'N/A')}ms" if rag.get("success") else "FAILED"
    cag_lat = f"{cag.get('latency_ms', 'N/A')}ms" if cag.get("success") else "FAILED"
    print(f"  │ Latency            │ {rag_lat:<17} │ {cag_lat:<17} │")

    # Context Size
    rag_ctx = f"{rag.get('context_chunks', 0)} chunks" if rag.get("success") else "N/A"
    cag_ctx = f"{cag.get('total_chars', 0):,} chars" if cag.get("success") else "N/A"
    print(f"  │ Context Size       │ {rag_ctx:<17} │ {cag_ctx:<17} │")

    # Token Usage (if available)
    rag_tokens = rag.get("usage", {}).get("total_tokens", "N/A")
    cag_tokens = cag.get("usage", {}).get("total_tokens", "N/A")
    rag_tok_str = f"{rag_tokens:,}" if isinstance(rag_tokens, int) else str(rag_tokens)
    cag_tok_str = f"{cag_tokens:,}" if isinstance(cag_tokens, int) else str(cag_tokens)
    print(f"  │ Total Tokens       │ {rag_tok_str:<17} │ {cag_tok_str:<17} │")

    # Database Required
    print(f"  │ Requires DB        │ {'Yes':<17} │ {'No':<17} │")

    # Scalability
    print(f"  │ Scalability        │ {'High':<17} │ {'Limited':<17} │")

    print("  └────────────────────┴───────────────────┴───────────────────┘")

    # ==========================================================================
    # APPROACH DIFFERENCES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  APPROACH DIFFERENCES")
    print("=" * 80)

    print("""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ RAG (Retrieval-Augmented Generation)                                        │
  │ ─────────────────────────────────────                                       │
  │ • Embeds query and searches vector database for relevant chunks             │
  │ • Only includes semantically similar content in context                     │
  │ • Scales to large document collections (millions of docs)                   │
  │ • Requires: Vector DB (Supabase), Embedding model (OpenAI)                  │
  │ • Best for: Large document sets, production deployments                     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ CAG (Context-Augmented Generation)                                          │
  │ ─────────────────────────────────────                                       │
  │ • Loads ALL documents directly into LLM context window                      │
  │ • No retrieval step - LLM sees everything                                   │
  │ • Limited by context window size (~100KB practical limit)                   │
  │ • Requires: Only LLM API (OpenRouter)                                       │
  │ • Best for: Small doc sets, development, guaranteed full context            │
  └─────────────────────────────────────────────────────────────────────────────┘
""")

    # Winner analysis
    if rag.get("success") and cag.get("success"):
        print("  ANALYSIS:")
        rag_ms = rag.get("latency_ms", 0)
        cag_ms = cag.get("latency_ms", 0)

        if rag_ms < cag_ms:
            diff = cag_ms - rag_ms
            print(f"  • RAG was {diff}ms faster ({rag_ms}ms vs {cag_ms}ms)")
        elif cag_ms < rag_ms:
            diff = rag_ms - cag_ms
            print(f"  • CAG was {diff}ms faster ({cag_ms}ms vs {rag_ms}ms)")
        else:
            print(f"  • Both approaches had equal latency ({rag_ms}ms)")

        rag_tok = rag.get("usage", {}).get("total_tokens", 0)
        cag_tok = cag.get("usage", {}).get("total_tokens", 0)
        if rag_tok and cag_tok:
            if rag_tok < cag_tok:
                print(f"  • RAG used fewer tokens ({rag_tok:,} vs {cag_tok:,}) = lower cost")
            elif cag_tok < rag_tok:
                print(f"  • CAG used fewer tokens ({cag_tok:,} vs {rag_tok:,}) = lower cost")

    print("\n" + "=" * 80)


def _wrap_text(text: str, width: int = 76) -> str:
    """Wrap text to specified width."""
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) <= width:
            lines.append(paragraph)
        else:
            words = paragraph.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= width:
                    current_line = f"{current_line} {word}".strip()
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
    return "\n".join(lines)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compare RAG vs CAG approaches for document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.documind.rag.compare "What is the vacation policy?"
  python -m src.documind.rag.compare "How many sick days?" --docs-dir demo-docs
  python -m src.documind.rag.compare "Benefits overview" --json
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Question to answer (if not provided, runs test questions)"
    )

    parser.add_argument(
        "--docs-dir", "-d",
        default="demo-docs",
        help="Directory containing .txt documents for CAG (default: demo-docs)"
    )

    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use for both approaches"
    )

    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON results"
    )

    args = parser.parse_args()

    # Test questions for demonstration
    TEST_QUESTIONS = [
        "What is the vacation policy?",
        "How many sick days do employees get?",
        "What are the health insurance options?",
    ]

    try:
        if args.query:
            # Single query mode
            results = compare_approaches(
                args.query,
                docs_dir=args.docs_dir,
                model=args.model,
            )
            if args.json:
                print(json.dumps(results, indent=2, default=str))
        else:
            # Demo mode with test questions
            print("\n" + "=" * 80)
            print("  RAG vs CAG COMPARISON DEMO")
            print("  Running test questions to demonstrate both approaches")
            print("=" * 80)

            all_results = []
            for i, question in enumerate(TEST_QUESTIONS, 1):
                print(f"\n{'#' * 80}")
                print(f"  TEST {i}/{len(TEST_QUESTIONS)}")
                print(f"{'#' * 80}")

                result = compare_approaches(
                    question,
                    docs_dir=args.docs_dir,
                    model=args.model,
                )
                all_results.append(result)

                if i < len(TEST_QUESTIONS):
                    print("\n  Continuing to next test question...")

            # Summary
            print("\n" + "=" * 80)
            print("  OVERALL SUMMARY")
            print("=" * 80)

            rag_wins = 0
            cag_wins = 0
            for r in all_results:
                rag_ms = r["rag"].get("latency_ms", float("inf"))
                cag_ms = r["cag"].get("latency_ms", float("inf"))
                if rag_ms < cag_ms:
                    rag_wins += 1
                elif cag_ms < rag_ms:
                    cag_wins += 1

            print(f"\n  Latency wins: RAG={rag_wins}, CAG={cag_wins}")
            print("\n  Both approaches generated answers for comparison.")
            print("  Review the answers above to assess quality differences.")

            if args.json:
                print("\n" + json.dumps(all_results, indent=2, default=str))

        print("\n" + "=" * 80)
        print("  Comparison complete!")
        print("=" * 80 + "\n")

    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nRequired environment variables:")
        print("  - OPENROUTER_API_KEY (for LLM inference)")
        print("  - OPENAI_API_KEY (for RAG embeddings)")
        print("  - SUPABASE_URL (for RAG vector search)")
        print("  - SUPABASE_ANON_KEY (for RAG vector search)")
        exit(1)
    except FileNotFoundError as e:
        print(f"\nFile Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise
