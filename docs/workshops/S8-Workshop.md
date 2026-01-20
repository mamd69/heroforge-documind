# HeroForge.AI Course: AI-Powered Software Development
## Lesson 8 Workshop: Vector Databases Deep Dive - pgvector, HNSW & Hybrid Search

**Estimated Time:** 30-45 minutes (reduced from 60 mins - S5 setup not repeated)\
**Difficulty:** Intermediate-Advanced\
**Prerequisites:** Completed Sessions 1-7 (Document processing, RAG system, Supabase integration with pgvector)

---

### 🧭 Where to Run Code - Quick Reference

Throughout this workshop, code blocks are labeled with where to run them:

| Icon | Location | What to Run |
|------|----------|-------------|
| 🗄️ **Supabase SQL Editor** | [supabase.com/dashboard](https://supabase.com/dashboard) → Your Project → SQL Editor | SQL queries: `SELECT`, `CREATE INDEX`, `ALTER TABLE`, `CREATE FUNCTION` |
| 🖥️ **Terminal** | Claude Code terminal or your local terminal | Bash/Python commands: `python ...`, `pip install ...` |
| 📝 **Create file** | Your IDE (VS Code, etc.) or Claude Code | Python files to save in your project |
| 🤖 **Claude Code** | Chat with Claude in Claude Code | Prompts to ask Claude for help implementing features |

---

## Before You Begin: Plan Your Work!

> **📋 Reminder:** In Session 3, we learned about **Issue-Driven Development** - the practice of creating GitHub Issues *before* starting work. This ensures clear requirements, enables collaboration, and creates traceability between your code and original requirements.
>
> **Before diving into this workshop:**
> 1. Create a GitHub Issue for the features you'll build today
> 2. Reference that issue in your branch name (`issue-XX-feature-name`)
> 3. Include `Closes #XX` or `Relates to #XX` in your commit messages
>
> 👉 See [S3-Workshop: Planning Your Work with GitHub Issues](./S3-Workshop.md#planning-your-work-with-github-issues-5-minutes) for the full workflow.

---

## Workshop Objectives

By completing this workshop, you will:

**Review (completed in S5):**
- [x] Understand vector embeddings and semantic similarity
- [x] Implement dense embeddings with OpenAI API
- [x] Configure pgvector extension in Supabase

**New in this session:**
- [ ] Use sparse embeddings (BM25) for keyword matching
- [ ] Optimize vector search performance with HNSW indexing
- [ ] Build hybrid search combining semantic and keyword signals
- [ ] Apply vector search optimization to DocuMind

---

### ⚠️ Supabase Client Type Warning

When working with Supabase in Python, ensure you're using the correct client initialization:

**Standard Pattern (used in this workshop):**
```python
from supabase import create_client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
```

**Common Errors and Fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: 'Client' object has no attribute 'table'` | Wrong client type | Use `create_client()` from `supabase` package |
| `TypeError: Missing required argument 'supabase_key'` | Missing env vars | Check `.env` has `SUPABASE_URL` and `SUPABASE_KEY` |
| `Invalid API key` | Wrong key copied | Copy from Supabase Dashboard → Settings → API |

**Verify Your Setup:**

> 🖥️ **Run in: Terminal (Claude Code or local)**

```bash
# Test your Supabase connection before starting exercises
python -c "from supabase import create_client; import os; c = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')); print('✅ Supabase connected!')"
```
NOTE: If using free version, your Supabase project may be paused.  If so, restore in Supabase to continue.
---

## Module 1: Vector Embeddings Review (5 minutes)

> **📚 Review:** This module summarizes concepts from **Session 5**. If you need a refresher, see [S5-Workshop](./S5-Workshop.md).

### Quick Recap: What We Built in S5

In Session 5, we implemented:

1. **pgvector extension** - Enabled in Supabase with `CREATE EXTENSION vector`
2. **OpenAI embeddings** - Using `text-embedding-3-small` (1536 dimensions)
3. **document_chunks table** - With embedding column for vector storage
4. **match_documents RPC** - For similarity search

**Verify your S5 setup is working:**

> 🖥️ **Run in: Terminal (Claude Code or local)**

```bash
# Quick test - should return results if S5 was completed
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
result = client.table('document_chunks').select('id', count='exact').limit(1).execute()
print(f'✅ document_chunks table exists ({result.count} chunks)')

# Test pgvector
result = client.rpc('match_documents', {
    'query_embedding': [0.0] * 1536,
    'match_count': 1,
    'similarity_threshold': 0.0
}).execute()
print('✅ pgvector and match_documents RPC working')
"
```

### ⚠️ Troubleshooting: If Verification Fails

If you don't see two green checkmarks, use this guide to fix your setup:

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `relation "document_chunks" does not exist` | Table wasn't created in S5 | Run the S5 migration SQL in Supabase SQL Editor (see [S5-Workshop](./S5-Workshop.md#exercise-21-set-up-pgvector)) |
| `function match_documents(...) does not exist` | RPC function not created | Create the function from [S5-Workshop](./S5-Workshop.md#exercise-22-create-similarity-search-function) |
| `Project is paused` | Supabase free tier timeout | Go to [Supabase Dashboard](https://supabase.com/dashboard) → Select project → Click "Restore" |
| `Invalid API key` or `Unauthorized` | Wrong credentials in `.env` | Copy correct keys from Supabase Dashboard → Settings → API |
| `SUPABASE_URL` or `SUPABASE_ANON_KEY` is None | Missing `.env` file or vars | Create `.env` with `SUPABASE_URL=https://xxx.supabase.co` and `SUPABASE_ANON_KEY=eyJ...` |
| `0 chunks` in table | No documents uploaded | Run: `python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/` |
| Chunks exist but `embedding` is NULL | Embeddings not generated | Re-upload with embeddings: `python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/` (requires `OPENAI_API_KEY` in `.env`) |

**Quick S5 Setup Recovery:**

> 🖥️ **Run in: Terminal (Claude Code or local)**

```bash
# 1. Check your .env file exists and has required variables
cat .env | grep -E "SUPABASE|OPENAI"

# 2. If missing, create .env from template
cp .env.example .env
# Then edit .env with your actual keys

# 3. Upload sample documents (creates table + embeddings)
python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/

# 4. Re-run verification
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
result = client.table('document_chunks').select('id', count='exact').limit(1).execute()
print(f'✅ document_chunks: {result.count} chunks')
"
```

**Still stuck?** Complete the [S5-Workshop](./S5-Workshop.md) first, then return here.

---

### Key Concepts (Review)

| Concept | What It Does | Implemented In |
|---------|--------------|----------------|
| **Dense Embeddings** | Neural network vectors (1536 dims) capturing semantic meaning | `src/documind/rag/search.py` |
| **Cosine Similarity** | Measures angle between vectors (0-1, higher = more similar) | `match_documents` RPC |
| **Semantic Search** | Find documents by meaning, not just keywords | `search_documents()` |

**Dense vs Sparse Embeddings (NEW concept for this session):**

| Type | Example | Best For |
|------|---------|----------|
| **Dense** (S5) | OpenAI text-embedding-3-small | Semantic similarity, synonyms |
| **Sparse** (NEW) | BM25, TF-IDF | Exact keyword matching, rare terms |

> **This session's focus:** We'll add **sparse embeddings (BM25)** and combine them with dense embeddings for **hybrid search**.

---

## Module 2: HNSW Indexing for Performance (15 minutes)

> **📚 Review:** pgvector setup was completed in **Session 5**. This module focuses on **NEW content: HNSW indexing** for production performance.

### What You Already Have (from S5)

These were set up in Session 5 - verify they exist:

> 🗄️ **Run in: Supabase SQL Editor** (Dashboard → SQL Editor → New Query)

```sql
-- Verify pgvector extension (should already exist)
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Verify embedding column exists
SELECT column_name, udt_name FROM information_schema.columns
WHERE table_name = 'document_chunks' AND column_name = 'embedding';

-- Verify match_documents function exists
SELECT routine_name FROM information_schema.routines
WHERE routine_name = 'match_documents';
```

If any of these are missing, refer to [S5-Workshop](./S5-Workshop.md) to set them up.

### NEW: Indexing Strategies

Without an index, pgvector does a full table scan (slow). Indexes enable **approximate nearest neighbor (ANN)** search:

| Index Type | Speed | Recall | Memory | Best For |
|------------|-------|--------|--------|----------|
| **No Index** | Slow | 100% | Low | Small datasets (<10k vectors) |
| **IVFFlat** | Fast | 90-95% | Medium | Medium datasets (10k-1M vectors) |
| **HNSW** | Very Fast | 95-99% | High | Large datasets (1M+ vectors), production |

**Distance Metrics:**
- `<->` : L2 distance (Euclidean)
- `<#>` : Negative inner product (maximize dot product)
- `<=>` : Cosine distance (1 - cosine similarity)

---

### Exercise 2.1: Benchmark BEFORE Creating Index

**Task:** First, measure search performance WITHOUT an HNSW index to establish a baseline.

> ⚠️ **Important:** Run this benchmark BEFORE creating the HNSW index so you can see the performance improvement!

**Step 1: Verify you have data to benchmark**

> 🖥️ **Run in: Terminal**

```bash
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
result = client.table('document_chunks').select('id', count='exact').not_.is_('embedding', 'null').execute()
print(f'Chunks with embeddings: {result.count}')
if result.count == 0:
    print('⚠️  No embeddings found! Run the upload CLI first:')
    print('   python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/')
"
```

**Step 2: Verify NO HNSW index exists yet**

> 🗄️ **Run in: Supabase SQL Editor**

```sql
-- Check if HNSW index exists (should return empty if not created yet)
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks' AND indexdef LIKE '%hnsw%';

-- If this returns rows, drop the index first to see the "before" performance:
-- DROP INDEX IF EXISTS document_chunks_embedding_hnsw_idx;
```

**Step 3: Create the benchmark script**

> 📝 **Create file:** `src/documind/benchmark_vector_search.py`

```python
"""
Benchmark vector search performance - Before and After HNSW index
Uses existing match_documents RPC from S5

Run this script TWICE:
  1. BEFORE creating HNSW index (to get baseline)
  2. AFTER creating HNSW index (to see improvement)
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
    # Warm-up run
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
        "avg_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "results": result.data[:3] if result and result.data else []
    }


def main():
    print("=" * 70)
    print("🔬 Vector Search Performance Benchmark")
    print("=" * 70)

    # Check chunk count
    result = supabase.table("document_chunks").select("id", count="exact").limit(1).execute()
    print(f"\n📚 Document chunks: {result.count}")

    # Test queries
    test_queries = [
        "What is the vacation policy?",
        "How do I request time off?",
        "Tell me about employee benefits",
        "Remote work guidelines"
    ]

    print("\n📊 Generating embeddings...")
    embeddings = {q: get_embedding(q) for q in test_queries}

    print(f"\n🔍 Running benchmark (10 runs per query)...\n")

    results = []
    for query in test_queries:
        r = benchmark_search(query, embeddings[query])
        results.append(r)
        print(f"Query: '{query}'")
        print(f"   ⏱️  Avg: {r['avg_ms']:.2f}ms | Min: {r['min_ms']:.2f}ms | Max: {r['max_ms']:.2f}ms\n")

    # Summary
    overall_avg = sum(r["avg_ms"] for r in results) / len(results)
    print("=" * 70)
    print(f"📊 Average query time: {overall_avg:.2f} ms")
    print(f"📊 Queries per second: {1000/overall_avg:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

**Step 4: Run the benchmark (BEFORE creating index)**

> 🖥️ **Run in: Terminal**

```bash
python src/documind/benchmark_vector_search.py
```

**Expected Output (BEFORE HNSW index):**
```
======================================================================
🔬 Vector Search Performance Benchmark
======================================================================

📚 Document chunks: 45

📊 Generating embeddings...

🔍 Running benchmark (10 runs per query)...

Query: 'What is the vacation policy?'
   ⏱️  Avg: 127.45ms | Min: 115.23ms | Max: 142.67ms

Query: 'How do I request time off?'
   ⏱️  Avg: 134.82ms | Min: 121.34ms | Max: 148.91ms

[... similar for other queries ...]

======================================================================
📊 Average query time: 130.25 ms
📊 Queries per second: 7.7
======================================================================
```

> 📝 **Write down your results!** Note the "Average query time" - you'll compare this after creating the index.

**Example BEFORE Index Results (no HNSW):**
```
Average query time: 150-500ms (depending on dataset size)
Queries per second: 2-7
```

---

### Exercise 2.2: Create HNSW Index

**Task:** Now create the HNSW index to speed up vector search.

> 🗄️ **Run in: Supabase SQL Editor**

```sql
-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Parameters:
-- m = 16: Number of connections per layer (higher = better recall, more memory)
-- ef_construction = 64: Build quality (higher = better index, slower to build)

-- Verify index was created
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks' AND indexdef LIKE '%hnsw%';
```

**Expected output:**
```
indexname                              | indexdef
---------------------------------------|------------------------------------------
document_chunks_embedding_hnsw_idx     | CREATE INDEX document_chunks_embedding_hnsw_idx ON public.document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')
```

**Index Configuration Guidelines:**

| Parameter | Default | Increase For | Trade-off |
|-----------|---------|--------------|-----------|
| `m` | 16 | Better accuracy | More memory |
| `ef_construction` | 64 | Better index quality | Slower build |

---

### Exercise 2.3: Benchmark AFTER Index (Compare Results)

**Task:** Run the same benchmark again to see the performance improvement.

> 🖥️ **Run in: Terminal**

```bash
# Run the SAME benchmark script again
python src/documind/benchmark_vector_search.py
```

**Expected Output (AFTER HNSW index):**
```
======================================================================
🔬 Vector Search Performance Benchmark
======================================================================

📚 Document chunks: 45

📊 Generating embeddings...

🔍 Running benchmark (10 runs per query)...

Query: 'What is the vacation policy?'
   ⏱️  Avg: 18.23ms | Min: 15.45ms | Max: 22.67ms

Query: 'How do I request time off?'
   ⏱️  Avg: 19.82ms | Min: 16.34ms | Max: 24.91ms

[... similar for other queries ...]

======================================================================
📊 Average query time: 18.75 ms
📊 Queries per second: 53.3
======================================================================
```

### 📊 Compare Your Results

| Metric | Before (No Index) | After (HNSW) | Improvement |
|--------|-------------------|--------------|-------------|
| Avg Query Time | ___ ms | ___ ms | ___x faster |
| Queries/Second | ___ | ___ | ___x more |

> **Expected improvement:** 5-50x faster queries depending on dataset size. Small datasets (<100 chunks) may show less improvement since the overhead is minimal.

---

### Quiz 2:

**Question 1:** What is the purpose of creating indexes on vector columns?\
   a) Indexes make the database look more professional\
   b) Indexes are required for pgvector to work at all\
   c) Indexes slow down queries but save storage space\
   d) Indexes enable fast approximate nearest neighbor search, dramatically improving query performance

**Question 2:** What is the difference between HNSW and IVFFlat indexes?\
   a) HNSW only works with text, IVFFlat only works with numbers\
   b) HNSW is faster with better recall but uses more memory; IVFFlat is a good balance\
   c) They are identical, just different names\
   d) IVFFlat is always better than HNSW

**Question 3:** What does the match_threshold parameter do in the similarity search function?\
   a) It sets the maximum number of results to return\
   b) It determines which index to use\
   c) It filters out results with similarity below the threshold, returning only high-quality matches\
   d) It has no effect on the search

**Answers:**
1. **d)** Indexes enable fast approximate nearest neighbor search (vs slow exhaustive search)
2. **b)** HNSW: faster, better recall, more memory; IVFFlat: balanced performance
3. **c)** match_threshold filters results, keeping only those above the similarity threshold

---

## Module 3: Hybrid Search Implementation (15 minutes)

### Concept Review

**What is Hybrid Search?**

Hybrid search combines multiple search techniques to achieve better results than any single method:
- **Dense embeddings**: Capture semantic similarity
- **Sparse embeddings** (BM25): Capture exact keyword matches
- **Reranking**: Combine and optimize results

**Why Hybrid?**

Each method has strengths and weaknesses:

| Scenario | Dense Wins | Sparse Wins |
|----------|------------|-------------|
| "How do I reset password?" vs "Password recovery" | ✅ Semantic match | ❌ No keywords |
| "Python tutorial" vs "Python programming guide" | ✅ Similar meaning | ✅ Exact keyword |
| "GPT-4" vs "Large language model" | ❌ Different words | ✅ Exact term |
| "AI safety" vs document with rare term "GPT-4" | ❌ May miss rare term | ✅ Catches exact match |

**Hybrid = Best of Both Worlds**

**BM25 (Sparse Embeddings):**

BM25 is a ranking function that scores documents based on term frequency, document length, and term rarity (IDF).

```
BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

Where:
- D: Document
- Q: Query
- qi: Query term i
- f(qi, D): Frequency of qi in D
- IDF(qi): Inverse document frequency of qi
- k1: Term saturation parameter (typically 1.2-2.0)
- b: Length normalization (typically 0.75)
- |D|: Document length
- avgdl: Average document length
```

**Reranking Strategies:**

1. **Linear Combination**: `score = α * semantic + (1-α) * keyword`
2. **Reciprocal Rank Fusion (RRF)**: Combine rankings, not scores
3. **Learned Reranking**: Train ML model to optimize combination

---

### Exercise 3.1: Implement BM25 Keyword Search

**Task:** Add BM25-based keyword search to complement vector search.

**Instructions:**

**Step 1: Enable Full-Text Search in Supabase (3 mins)**

> 🗄️ **Run in: Supabase SQL Editor**

```sql
-- Increase memory limit for this session (required for large tables)
SET maintenance_work_mem = '256MB';

-- Add tsvector column for full-text search
-- Note: Column is named "content" in document_chunks table
ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS fts tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Create GIN index for fast text search
CREATE INDEX IF NOT EXISTS document_chunks_fts_idx
ON document_chunks
USING gin(fts);

-- Test full-text search
SELECT content, ts_rank(fts, query) AS rank
FROM document_chunks, to_tsquery('english', 'machine & learning') AS query
WHERE fts @@ query
ORDER BY rank DESC
LIMIT 5;
```

**Step 2: Create Keyword Search Function (5 mins)**

> 🗄️ **Run in: Supabase SQL Editor**

```sql
-- Function for BM25-style keyword search
-- Note: Returns "content" column (actual column name in document_chunks table)
CREATE OR REPLACE FUNCTION search_document_chunks_keyword(
  search_query text,
  match_count int DEFAULT 10
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  chunk_index int,
  content text,
  rank float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    document_chunks.id,
    document_chunks.document_id,
    document_chunks.chunk_index,
    document_chunks.content,
    ts_rank(document_chunks.fts, to_tsquery('english', search_query)) AS rank
  FROM document_chunks
  WHERE document_chunks.fts @@ to_tsquery('english', search_query)
  ORDER BY rank DESC
  LIMIT match_count;
END;
$$;

-- Test keyword search
SELECT * FROM search_document_chunks_keyword('machine & learning', 5);
```

**Step 3: Implement Hybrid Search in Python (7 mins)**

> 📝 **Create file:** `src/documind/hybrid_search.py` (use your IDE or Claude Code)

```python
"""
Hybrid Search: Combine semantic (vector) and keyword (BM25) search
"""
import os
from typing import List, Dict, Any
from supabase import create_client
from embeddings_client import EmbeddingsClient
from collections import defaultdict

class HybridSearcher:
    """Combines vector and keyword search with reranking"""

    def __init__(self, semantic_weight: float = 0.7):
        """
        Initialize hybrid searcher.

        Args:
            semantic_weight: Weight for semantic search (0-1)
                           1.0 = pure semantic, 0.0 = pure keyword
        """
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.embeddings_client = EmbeddingsClient()
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

    def search_semantic(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search using vector embeddings"""
        # Generate query embedding
        query_embedding = self.embeddings_client.generate_embedding(query)

        # Search with pgvector
        result = self.supabase.rpc("match_document_chunks", {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k
        }).execute()

        return [{
            "id": row["id"],
            "document_id": row["document_id"],
            "content": row["content"],
            "semantic_score": row["similarity"]
        } for row in result.data]

    def search_keyword(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Keyword search using full-text search (BM25-style)"""
        # Format query for PostgreSQL full-text search
        # Convert "machine learning" to "machine & learning"
        formatted_query = " & ".join(query.split())

        # Search with full-text search
        result = self.supabase.rpc("search_document_chunks_keyword", {
            "search_query": formatted_query,
            "match_count": top_k
        }).execute()

        return [{
            "id": row["id"],
            "document_id": row["document_id"],
            "content": row["content"],
            "keyword_score": row["rank"]
        } for row in result.data]

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
        "What is machine learning?",
        "Tell me about GPT-4",  # Rare term - keyword should help
        "How to build neural networks",
        "Cloud computing and DevOps"
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
```

> 🖥️ **Run in: Terminal**

```bash
# Run hybrid search test
python src/documind/hybrid_search.py
```

---

### Quiz 3:

**Question 1:** What is the main advantage of hybrid search over pure semantic search?\
   a) Hybrid search is always slower so results are more accurate\
   b) Hybrid search uses less memory\
   c) Hybrid search combines semantic similarity with exact keyword matching for better overall results\
   d) Hybrid search doesn't require any embeddings

**Question 2:** What does Reciprocal Rank Fusion (RRF) do?\
   a) Deletes results that don't match\
   b) Reverses the order of search results\
   c) Multiplies all scores together\
   d) Combines multiple search result rankings by considering position rather than raw scores

**Question 3:** When would pure keyword (BM25) search outperform pure semantic search?\
   a) When searching for synonyms or paraphrased concepts\
   b) When searching for rare or specific terms like "GPT-4" or exact product names\
   c) Keyword search is always better than semantic search\
   d) Only on Tuesdays

**Answers:**
1. **c)** Hybrid combines semantic similarity (synonyms, concepts) with exact keyword matching (rare terms, exact names)
2. **d)** RRF combines rankings by position (1/(k+rank)) rather than raw scores
3. **b)** Keyword excels at exact matches and rare terms (e.g., "GPT-4", product codes)

---

## Module 4: Challenge Project - Optimize DocuMind Search (15 minutes)

### Challenge Overview

Optimize DocuMind's search system with production-ready vector search and hybrid ranking.

**Your Mission:**
Build a production search system that:
1. Uses optimal vector indexes for fast queries
2. Implements hybrid search with tunable weights
3. Includes query performance monitoring
4. Provides search quality metrics

---

### Challenge Requirements

**Feature:** Production-Ready Search System

**What to Build:**

1. **Search API** with multiple search modes
   - Pure semantic search
   - Pure keyword search
   - Hybrid search (configurable weights)
   - Automatic mode selection based on query

2. **Performance Monitoring**
   - Query latency tracking
   - Result quality metrics
   - Index usage statistics

3. **Query Optimization**
   - Query expansion (add synonyms)
   - Query rewriting (fix typos)
   - Result diversification

4. **Search Quality Metrics**
   - Precision and recall
   - Mean Average Precision (MAP)
   - Normalized Discounted Cumulative Gain (NDCG)

---

### Starter Code

> 📝 **Create file:** `src/documind/search_api.py` (use your IDE or Claude Code)

```python
"""
Production Search API for DocuMind
"""
from typing import List, Dict, Any, Optional
from enum import Enum
import time
from dataclasses import dataclass, field
from hybrid_search import HybridSearcher

class SearchMode(Enum):
    """Search mode options"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    document_id: str
    chunk_text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchMetrics:
    """Search performance metrics"""
    query: str
    mode: SearchMode
    latency_ms: float
    num_results: int
    avg_score: float
    top_score: float

class SearchAPI:
    """
    Production search API with monitoring and optimization.
    """

    def __init__(self):
        self.searcher = HybridSearcher(semantic_weight=0.7)
        self.query_history: List[SearchMetrics] = []

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search documents with specified mode.

        TODO: Implement
        - Auto mode selection based on query characteristics
        - Query expansion/rewriting
        - Performance monitoring
        - Result diversification
        """
        # Your code here
        pass

    def auto_select_mode(self, query: str) -> SearchMode:
        """
        Automatically select best search mode based on query.

        TODO: Implement heuristics
        - Use KEYWORD if query contains rare terms, codes, or exact phrases
        - Use SEMANTIC for conceptual or paraphrased queries
        - Use HYBRID as default
        """
        # Your code here
        pass

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms.

        TODO: Implement query expansion
        - Add synonyms (e.g., "AI" → "AI, artificial intelligence, machine learning")
        - Expand acronyms
        - Add related terms
        """
        # Your code here
        pass

    def diversify_results(
        self,
        results: List[SearchResult],
        max_per_document: int = 2
    ) -> List[SearchResult]:
        """
        Diversify results to avoid returning too many chunks from same document.

        TODO: Implement result diversification
        - Limit number of chunks per document
        - Ensure diversity of sources
        """
        # Your code here
        pass

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report from query history.

        TODO: Implement
        - Average latency by mode
        - Query volume by mode
        - Score distribution
        - Trend analysis
        """
        # Your code here
        pass


# Test the search API
if __name__ == "__main__":
    api = SearchAPI()

    # TODO: Implement tests
    # 1. Test all search modes
    # 2. Test auto mode selection
    # 3. Test query expansion
    # 4. Test result diversification
    # 5. Generate performance report

    pass
```

---

### Your Task

**Step 1: Implement Core Features (10 mins)**

> 🤖 **Using Claude Code:** Ask Claude to help implement the TODO sections

```
Complete the implementation of src/documind/search_api.py.

Requirements:
- search(): Main search function with all modes
- auto_select_mode(): Heuristics for mode selection (check for rare terms, codes, quotes)
- expand_query(): Add synonyms and related terms
- diversify_results(): Limit chunks per document to max_per_document
- get_performance_report(): Aggregate metrics (avg latency, queries by mode, score stats)

Use HybridSearcher for actual search execution.
Track SearchMetrics for each query.
```

**Step 2: Test and Benchmark (5 mins)**

> 🖥️ **Run in: Terminal**

```bash
# Create test queries
python -c "
from src.documind.search_api import SearchAPI, SearchMode

api = SearchAPI()

test_cases = [
    ('What is machine learning?', SearchMode.AUTO),
    ('GPT-4', SearchMode.AUTO),
    ('\"artificial intelligence\"', SearchMode.AUTO),
    ('How to train neural networks', SearchMode.HYBRID),
]

for query, mode in test_cases:
    results = api.search(query, mode=mode, top_k=5)
    print(f'\nQuery: {query}')
    print(f'Mode: {mode}')
    print(f'Results: {len(results)}')
    for i, r in enumerate(results, 1):
        print(f'  {i}. [{r.score:.4f}] {r.chunk_text[:60]}...')

# Generate performance report
print('\n' + '='*70)
report = api.get_performance_report()
print('Performance Report:')
for key, value in report.items():
    print(f'  {key}: {value}')
"
```

---

### Success Criteria

Your implementation is complete when:

- [ ] `search()` supports all modes (semantic, keyword, hybrid, auto)
- [ ] `auto_select_mode()` selects keyword for quoted strings and rare terms
- [ ] `expand_query()` adds at least 2-3 synonyms/related terms
- [ ] `diversify_results()` limits chunks per document
- [ ] `get_performance_report()` shows avg latency, query counts, and score stats
- [ ] AUTO mode correctly selects KEYWORD for "GPT-4" or quoted strings
- [ ] AUTO mode selects SEMANTIC for conceptual queries
- [ ] Performance report shows meaningful metrics

---

## Answer Key

### Module 4 Challenge Solution

See complete implementation in appendix or request from instructor.

Key implementation points:

**Auto Mode Selection:**
```python
def auto_select_mode(self, query: str) -> SearchMode:
    # Check for quoted strings (exact match intent)
    if '"' in query or "'" in query:
        return SearchMode.KEYWORD

    # Check for rare terms (capitals, numbers, codes)
    has_capitals = any(word.isupper() or word[0].isupper() for word in query.split())
    has_numbers = any(char.isdigit() for char in query)
    has_codes = bool(re.search(r'[A-Z]{2,}', query))  # e.g., "GPT-4", "API"

    if has_capitals or has_numbers or has_codes:
        return SearchMode.KEYWORD

    # Default to hybrid for general queries
    return SearchMode.HYBRID
```

**Query Expansion:**
```python
def expand_query(self, query: str) -> str:
    synonyms = {
        "ai": "AI artificial intelligence machine learning",
        "ml": "ML machine learning",
        "password": "password credential authentication",
        "error": "error issue problem bug",
    }

    words = query.lower().split()
    expanded = []
    for word in words:
        expanded.append(word)
        if word in synonyms:
            expanded.extend(synonyms[word].split())

    return " ".join(expanded)
```

---

## Additional Challenges (Optional)

### Challenge 1: Query Analytics Dashboard
Build a dashboard that visualizes:
- Query volume over time
- Search mode usage
- Average latency by mode
- Score distributions
- Popular queries

### Challenge 2: A/B Testing Framework
Implement A/B testing to compare:
- Different semantic_weight values (0.5 vs 0.7)
- Linear vs RRF reranking
- With vs without query expansion

### Challenge 3: Learned Reranking
Train a neural reranker using:
- User click data as labels
- Features: semantic score, keyword score, document metadata
- Model: XGBoost or LightGBM

### Challenge 4: Multi-Modal Search
Extend to support:
- Image search (CLIP embeddings)
- Code search (CodeBERT embeddings)
- Table search (structured data matching)

---

## Key Takeaways

By completing this workshop, you've learned:

1. **Vector Embeddings**: How text becomes numerical representations for semantic search
2. **pgvector**: Production-ready vector storage and similarity search in PostgreSQL
3. **Indexing**: HNSW and IVFFlat indexes for fast approximate nearest neighbor search
4. **Hybrid Search**: Combining dense and sparse embeddings for optimal results
5. **Performance Optimization**: Benchmarking, monitoring, and tuning search systems

**The Hybrid Search Pattern:**
```
Semantic Search (meaning) + Keyword Search (exact) + Reranking (combine) = Optimal Search
```

---

## Next Session Preview

In **Session 9: Evaluation with RAGAS and TruLens**, we'll:
- Understand RAG evaluation metrics (faithfulness, relevance, precision, recall)
- Implement automated evaluation with RAGAS
- Set up real-time observability with TruLens
- Create quality dashboards and alerting
- Optimize DocuMind based on evaluation insights

**Preparation:**
1. Ensure your hybrid search is working correctly
2. Have several documents ingested with varied content
3. Create a set of test queries with expected answers
4. Install RAGAS and TruLens packages

See you in Session 9!

---

**Workshop Complete! 🎉**

You've built a production-ready vector search system with hybrid ranking! You're ready to evaluate and optimize your RAG pipeline with RAGAS and TruLens.
