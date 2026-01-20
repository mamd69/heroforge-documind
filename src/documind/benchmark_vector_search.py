"""
Benchmark vector search performance - Before and After HNSW index
Uses existing match_documents RPC from S5

Run this script TWICE:
  1. BEFORE creating HNSW index (to get baseline)
  2. AFTER creating HNSW index (to see improvement)

Usage:
  python src/documind/benchmark_vector_search.py
"""
import os
import time
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

# Initialize clients
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list:
    """Generate embedding using OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def benchmark_search(query: str, query_embedding: list, num_runs: int = 10) -> dict:
    """Benchmark search performance for a single query"""
    # Warm-up run (not counted)
    supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "similarity_threshold": 0.3,
        "match_count": 10
    }).execute()

    # Benchmark multiple runs
    times = []
    result = None
    for _ in range(num_runs):
        start = time.time()
        result = supabase.rpc("match_documents", {
            "query_embedding": query_embedding,
            "similarity_threshold": 0.3,
            "match_count": 10
        }).execute()
        times.append(time.time() - start)

    return {
        "query": query,
        "avg_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "num_results": len(result.data) if result and result.data else 0,
        "results": result.data[:3] if result and result.data else []
    }


def check_hnsw_index() -> bool:
    """Check if HNSW index likely exists by checking query speed"""
    test_embedding = [0.0] * 1536

    times = []
    for _ in range(3):
        start = time.time()
        supabase.rpc("match_documents", {
            "query_embedding": test_embedding,
            "similarity_threshold": 0.0,
            "match_count": 1
        }).execute()
        times.append(time.time() - start)

    avg_ms = sum(times) / len(times) * 1000
    # Heuristic: with index, queries are typically < 50ms
    return avg_ms < 50


def main():
    print("=" * 70)
    print("🔬 Vector Search Performance Benchmark")
    print("=" * 70)

    # Check chunk count
    result = supabase.table("document_chunks").select("id", count="exact").limit(1).execute()
    chunk_count = result.count or 0
    print(f"\n📚 Document chunks in database: {chunk_count}")

    if chunk_count == 0:
        print("\n⚠️  No document chunks found! Upload documents first:")
        print("   python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/")
        return

    # Detect index status
    print("\n🔍 Checking index status...")
    has_index = check_hnsw_index()
    if has_index:
        print("   ✅ HNSW index appears to be active (fast queries detected)")
    else:
        print("   ⏳ No HNSW index detected (slower sequential scan)")

    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "How do I request time off?",
        "Tell me about employee benefits",
        "Remote work guidelines"
    ]

    print("\n📊 Generating embeddings for test queries...")
    embeddings = {}
    for query in test_queries:
        embeddings[query] = get_embedding(query)
        print(f"   ✓ '{query[:40]}...'")

    print(f"\n🔍 Running benchmark (10 runs per query)...\n")
    print("-" * 70)

    results = []
    for query in test_queries:
        result = benchmark_search(query, embeddings[query], num_runs=10)
        results.append(result)

        print(f"Query: '{query}'")
        print(f"   ⏱️  Avg: {result['avg_ms']:.2f}ms | Min: {result['min_ms']:.2f}ms | Max: {result['max_ms']:.2f}ms")

        # Show top result
        if result['results']:
            top = result['results'][0]
            content = top.get('content', top.get('chunk_text', ''))[:60]
            similarity = top.get('similarity', 0)
            print(f"   📄 Top result: [{similarity:.4f}] {content}...")
        print()

    # Summary
    avg_times = [r["avg_ms"] for r in results]
    overall_avg = sum(avg_times) / len(avg_times)
    qps = 1000.0 / overall_avg

    print("=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"   Average query time: {overall_avg:.2f} ms")
    print(f"   Queries per second: {qps:.1f}")
    print("=" * 70)

    # Guidance
    if has_index:
        print("\n✅ You're running WITH the HNSW index.")
        print("   Compare these results to your BEFORE numbers to see the improvement!")
    else:
        print("\n📝 SAVE THESE RESULTS! This is your BEFORE baseline.")
        print("\n   Next steps:")
        print("   1. Go to Supabase SQL Editor")
        print("   2. Run: CREATE INDEX document_chunks_embedding_hnsw_idx")
        print("          ON document_chunks USING hnsw (embedding vector_cosine_ops)")
        print("          WITH (m = 16, ef_construction = 64);")
        print("   3. Run this benchmark again to see the improvement!")


if __name__ == "__main__":
    main()
