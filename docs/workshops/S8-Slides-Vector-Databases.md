# Session 8: Vector Databases Deep Dive

## Session 8 Title

### HNSW Indexing & Hybrid Search for Production

**Duration**: 30-45 minutes | **Complexity**: 7/10

> **Note:** This session builds on Session 5 foundations. We assume pgvector, embeddings, and basic search are already working.

---

## What's Review vs What's NEW

### Review from Session 5 (5 min)
- pgvector extension enabled
- document_chunks table with embedding column
- OpenAI text-embedding-3-small (1536 dims)
- match_documents RPC function
- Basic similarity search

### NEW Today (35-40 min)
- **HNSW Indexing** - 50x faster similarity search
- **Sparse Embeddings (BM25)** - Keyword matching capabilities
- **Hybrid Search** - Dense + sparse with RRF for 89% accuracy
- **Production Optimization** - Batching, caching, tuning
- **Beyond Vectors** - When to consider graphs and GraphRAG

---

## The Journey So Far

### Session 5: Multi-Agent Systems
Basic pgvector storage, embedding generation, semantic search

### Session 6: RAG & CAG Fundamentals
Retrieval-Augmented Generation, Q&A systems, chunking strategies

### Session 7: Advanced Data Extraction
PDF/DOCX parsing, multi-format support, metadata preservation

### Session 8 (Today): Production Search
HNSW optimization, hybrid search, enterprise-ready configuration

---

# Module 1: S5 Review (5 minutes)

## Verify Your S5 Setup

Before we dive into new content, confirm your setup is working:

```bash
python -c "
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
result = client.table('document_chunks').select('id', count='exact').limit(1).execute()
print(f'✅ document_chunks table exists ({result.count} chunks)')

result = client.rpc('match_documents', {
    'query_embedding': [0.0] * 1536,
    'match_count': 1,
    'similarity_threshold': 0.0
}).execute()
print('✅ pgvector and match_documents RPC working')
"
```

**If you see two green checkmarks, you're ready for today's content.**

### ⚠️ If Verification Fails

| Error | Cause | Quick Fix |
|-------|-------|-----------|
| `relation "document_chunks" does not exist` | Table not created in S5 | Run S5 migrations or see S5-Workshop.md |
| `function match_documents does not exist` | RPC not created | Create the function from S5-Workshop.md |
| `Project is paused` | Free tier timeout | Restore project in Supabase Dashboard |
| `Invalid API key` | Wrong env vars | Check `.env` has correct `SUPABASE_URL` and `SUPABASE_ANON_KEY` |
| `0 chunks` returned | No data uploaded | Run: `python -m src.documind.cli.upload_cli --dir docs/workshops/S7-sample-docs/` |

**Need full S5 setup?** See [S5-Workshop.md](./S5-Workshop.md) for complete instructions.

---

## Quick Recap: What We Built in S5

### 1. pgvector Extension
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Document Chunks Table
```sql
CREATE TABLE document_chunks (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id),
  chunk_text TEXT NOT NULL,
  embedding vector(1536),  -- OpenAI dimensions
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3. match_documents RPC
Similarity search using cosine distance

### 4. Embedding Generation
OpenAI text-embedding-3-small API integration

**This is our foundation. Now we optimize it.**

---

# Module 2: HNSW Indexing (NEW)

## The Scale Problem

### Without an Index (Linear Scan)
- Compares query against EVERY vector
- O(n) complexity
- 100,000 vectors = 100,000 comparisons
- **Result: ~500ms per query**

### With HNSW Index (Graph Navigation)
- Navigates through connected graph layers
- O(log n) complexity
- 100,000 vectors = ~17 comparisons
- **Result: ~10ms per query**

**HNSW delivers 50x speedup at scale**

---

## HNSW Algorithm Explained

### Hierarchical Navigable Small World

Think of it like navigating a city:

**Top Layer:** Major highways connecting distant cities
**Middle Layers:** Main roads connecting neighborhoods
**Bottom Layer:** Local streets connecting every house

When you search:
1. Start at the top layer (coarse navigation)
2. Move down through layers (getting closer)
3. Find nearest neighbors at bottom layer

**You never check every house - you navigate intelligently**

---

## Creating HNSW Index

### The SQL

```sql
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Parameter Breakdown

**`m = 16`** (connections per node)
- Higher = better recall, more memory
- Range: 4-64, default 16 is sweet spot

**`ef_construction = 64`** (build quality)
- Higher = better index, slower to build
- Range: 4-512, default 64 is production-ready

**`vector_cosine_ops`** (distance metric)
- Best for text embeddings
- Measures angle between vectors

---

## Index Tuning Parameters

### Build-Time Parameters

```sql
CREATE INDEX ... WITH (
  m = 16,              -- Connections per layer
  ef_construction = 64 -- Build-time search depth
);
```

### Query-Time Parameter

```sql
SET hnsw.ef_search = 100;  -- Search depth at query time
```

| Parameter | Default | Increase For | Trade-off |
|-----------|---------|--------------|-----------|
| m | 16 | Better accuracy | More memory |
| ef_construction | 64 | Better index quality | Slower build |
| ef_search | 40 | Better recall | Slower queries |

**DocuMind recommendation:** m=16, ef_construction=64, ef_search=100

---

## Performance Comparison

### Benchmark Results (10,000 chunks)

| Configuration | Query Time | Accuracy |
|--------------|------------|----------|
| No index (scan) | 500ms | 100% |
| HNSW (default) | 10ms | 99.2% |
| HNSW (tuned) | 15ms | 99.8% |

### Run the Benchmark

```python
import time
from supabase import create_client

# Without HNSW (sequential scan)
start = time.time()
result = client.rpc('match_documents', {
    'query_embedding': query_embedding,
    'match_count': 5,
    'similarity_threshold': 0.7
}).execute()
print(f"Query time: {(time.time() - start)*1000:.1f}ms")
```

**HNSW: Essential for production at any scale > 1,000 vectors**

---

# Module 3: Hybrid Search (NEW)

## Why Hybrid Search?

### Vector Search Strengths
- "How do I reset my password?" matches "Password recovery guide"
- Understands concepts, synonyms, paraphrasing

### Vector Search Weaknesses
- "GPT-4" might not match documents containing exactly "GPT-4"
- Specific terms, codes, product names get lost

### Keyword Search (BM25) Strengths
- "GPT-4" finds documents containing "GPT-4" exactly
- Great for specific terms, codes, identifiers

### The Solution
**Hybrid search combines both for 89% accuracy (vs 82% semantic only)**

---

## Dense vs Sparse Embeddings

### Dense Embeddings (OpenAI)
- Every dimension has a value
- 1536 dimensions for text-embedding-3-small
- Captures semantic relationships
- **Best for:** Conceptual similarity

```
[0.23, -0.45, 0.67, 0.12, ..., -0.34]  (1536 values)
```

### Sparse Embeddings (BM25)
- Most dimensions are zero
- Vocabulary size (~50,000+ dimensions)
- Captures keyword importance
- **Best for:** Exact term matching

```
[0, 0, 2.3, 0, 0, 1.7, 0, ..., 0]  (mostly zeros)
```

---

## BM25 Explained

### Best Match 25 Algorithm
Scores documents by:

1. **Term Frequency (TF)** - How often does the search term appear?
2. **Inverse Document Frequency (IDF)** - How rare is this term across all documents?
3. **Length Normalization** - Adjusts for document length

### PostgreSQL Implementation (Full-Text Search)

```sql
-- Add FTS column
ALTER TABLE document_chunks
ADD COLUMN chunk_text_tsv tsvector;

-- Create GIN index for fast text search
CREATE INDEX document_chunks_fts_idx
ON document_chunks USING GIN (chunk_text_tsv);

-- Populate tsvector column
UPDATE document_chunks
SET chunk_text_tsv = to_tsvector('english', chunk_text);
```

---

## Setting Up Full-Text Search

### 1. Add tsvector Column

```sql
ALTER TABLE document_chunks
ADD COLUMN chunk_text_tsv tsvector;
```

### 2. Create GIN Index

```sql
CREATE INDEX document_chunks_fts_idx
ON document_chunks USING GIN (chunk_text_tsv);
```

### 3. Create Auto-Update Trigger

```sql
CREATE TRIGGER document_chunks_tsv_update
BEFORE INSERT OR UPDATE ON document_chunks
FOR EACH ROW
EXECUTE FUNCTION
  tsvector_update_trigger(chunk_text_tsv, 'pg_catalog.english', chunk_text);
```

### 4. Populate Existing Data

```sql
UPDATE document_chunks
SET chunk_text_tsv = to_tsvector('english', chunk_text);
```

---

## Keyword Search RPC Function

### Create the Function

```sql
CREATE OR REPLACE FUNCTION keyword_search_chunks(
  search_query TEXT,
  match_count INT DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  document_id UUID,
  chunk_text TEXT,
  rank REAL
)
LANGUAGE sql
AS $$
  SELECT
    id,
    document_id,
    chunk_text,
    ts_rank(chunk_text_tsv, query) AS rank
  FROM document_chunks, plainto_tsquery('english', search_query) AS query
  WHERE chunk_text_tsv @@ query
  ORDER BY rank DESC
  LIMIT match_count;
$$;
```

### Usage

```python
result = client.rpc('keyword_search_chunks', {
    'search_query': 'vacation policy',
    'match_count': 5
}).execute()
```

---

## Reciprocal Rank Fusion (RRF)

### The Problem
Dense and sparse scores are on different scales. How do we combine them?

### The Solution: Combine RANKS, Not Scores

```
RRF_score = 1/(k + rank_semantic) + 1/(k + rank_keyword)
```

Where `k = 60` (constant to smooth rankings)

### Example

| Document | Semantic Rank | Keyword Rank | RRF Score |
|----------|--------------|--------------|-----------|
| Doc A | 1 | 3 | 1/61 + 1/63 = 0.032 |
| Doc B | 2 | 1 | 1/62 + 1/61 = 0.032 |
| Doc C | 5 | 2 | 1/65 + 1/62 = 0.031 |

**RRF elegantly merges different ranking systems**

---

## RRF Implementation

### Python Code

```python
def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = 60
) -> list:
    """Combine results using RRF algorithm"""
    # Build rank maps
    dense_ranks = {r['id']: i + 1 for i, r in enumerate(dense_results)}
    sparse_ranks = {r['id']: i + 1 for i, r in enumerate(sparse_results)}

    # Get all unique chunk IDs
    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Calculate RRF scores
    rrf_scores = {}
    for chunk_id in all_ids:
        dense_rank = dense_ranks.get(chunk_id, 1000)
        sparse_rank = sparse_ranks.get(chunk_id, 1000)
        rrf_scores[chunk_id] = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))

    # Sort by RRF score descending
    return sorted(all_ids, key=lambda x: rrf_scores[x], reverse=True)
```

---

## Complete Hybrid Search

### Python Implementation

```python
class HybridSearch:
    def __init__(self, supabase, generator):
        self.supabase = supabase
        self.generator = generator

    def search(self, query: str, top_k: int = 5) -> list:
        # Get dense (semantic) results
        query_embedding = self.generator.generate_single(query)
        dense_results = self.supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_count': top_k * 2,
            'similarity_threshold': 0.5
        }).execute().data

        # Get sparse (keyword) results
        sparse_results = self.supabase.rpc('keyword_search_chunks', {
            'search_query': query,
            'match_count': top_k * 2
        }).execute().data

        # Combine with RRF
        combined = reciprocal_rank_fusion(dense_results, sparse_results)

        return combined[:top_k]
```

---

## Hybrid Search: When to Use What

### Use Dense (Semantic) Only When:
- Queries are conceptual questions
- Latency is critical (<10ms)
- No specific terms/codes to match

### Use Sparse (Keyword) Only When:
- Queries contain exact identifiers
- Product codes, error codes, names
- Boolean search needed (AND/OR)

### Use Hybrid When:
- Mixed query types expected
- Best accuracy required (89% vs 82%)
- Acceptable latency trade-off (~60ms)

**DocuMind uses Hybrid for production**

---

# Module 4: Production Optimization (NEW)

## Production Index Configuration

### For Large Scale (100K+ chunks)

```sql
-- Optimized HNSW for production
CREATE INDEX document_chunks_embedding_hnsw_prod_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (
  m = 24,               -- Increased connections
  ef_construction = 100 -- Higher build quality
);

-- Set query-time search depth
SET hnsw.ef_search = 200;
```

### Trade-offs
- Higher `m` = better accuracy, more memory (1.5-2x vector size)
- Higher `ef_construction` = slower index build, better quality
- Higher `ef_search` = slower queries, better recall

---

## Embedding Cost Reality Check

### OpenAI text-embedding-3-small Pricing
**$0.02 per 1 million tokens**

### Real-World Costs

| Scale | Tokens | One-Time Cost |
|-------|--------|---------------|
| 1,000 chunks | ~200K | $0.004 |
| 10,000 chunks | ~2M | $0.04 |
| 100,000 chunks | ~20M | $0.40 |

### Monthly Query Costs
- 1,000 queries/month: ~$0.005
- 10,000 queries/month: ~$0.05

**Total for DocuMind (~10K docs, 10K queries/month): < $0.10/month**

Embedding costs are negligible compared to LLM inference.

---

## Batching for Efficiency

### Bad: One API Call Per Chunk

```python
# 1,000 API calls for 1,000 chunks
for chunk in chunks:
    embedding = generate_single(chunk)  # Slow!
```

### Good: Batch Processing

```python
# 10 API calls for 1,000 chunks (batch size 100)
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    embeddings = generate_batch(batch)  # 100x fewer calls
```

### OpenAI Limits
- Max 100 texts per request
- 3,000 requests per minute

**Batching = 100x fewer API requests**

---

## Embedding Cache Strategy

### Cache by Text Hash

```python
import hashlib

class EmbeddingCache:
    def __init__(self, supabase):
        self.supabase = supabase

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get_or_generate(self, text: str, generator) -> list:
        key = self._hash(text)

        # Check cache
        result = self.supabase.table('embedding_cache').select(
            'embedding'
        ).eq('text_hash', key).single().execute()

        if result.data:
            return result.data['embedding']

        # Generate and cache
        embedding = generator.generate_single(text)
        self.supabase.table('embedding_cache').insert({
            'text_hash': key,
            'embedding': embedding
        }).execute()

        return embedding
```

**Typical cache hit rate: ~40% (saves API costs and latency)**

---

## Performance Monitoring

### Track Key Metrics

```python
class SearchMonitor:
    def log_search(self, query, results, query_time_ms, strategy):
        self.supabase.table('search_metrics').insert({
            'query': query,
            'result_count': len(results),
            'top_similarity': results[0]['similarity'] if results else 0,
            'query_time_ms': query_time_ms,
            'strategy': strategy,
            'timestamp': 'now()'
        }).execute()
```

### Metrics to Watch
- **Average query time** - Should be <100ms
- **Average similarity** - Should be >0.7 for good results
- **Zero-result rate** - High rate = indexing issues
- **Cache hit rate** - Target >30%

---

# Module 5: Beyond Vectors (NEW)

## When Vectors Aren't Enough

### Vector Search Excels At:
- "Find similar documents"
- Semantic similarity
- Single-hop retrieval

### Vector Search Struggles With:
- "Find documents by co-authors of this person"
- Multi-hop relationship queries
- "What's the citation chain from A to B?"

### Graph Databases (Neo4j) Excel At:
- Relationship traversal
- Multi-hop reasoning
- Explainable paths

---

## GraphRAG and Hybrid Approaches

### Microsoft GraphRAG
- Combines semantic similarity + relationship traversal
- 70% accuracy improvement on complex queries

### RuVector-Postgres
- Unified PostgreSQL extension
- pgvector + Apache AGE + Self-Learning GNN
- 16,400 queries/second, 61µs latency
- Adaptive learning with ReasoningBank

### When to Consider Graph-Vector Hybrids:
- Complex relationship queries
- Need both "what's similar" AND "how connected"
- Knowledge graph that evolves over time

**For DocuMind scale: pgvector is sufficient. Consider hybrid for enterprise.**

---

## Technology Decision Tree

```
Is exact keyword matching important?
├─ Yes
│  └─ Do you need semantic understanding too?
│     ├─ Yes → Hybrid Search (RRF)
│     └─ No → Sparse Search (BM25/FTS)
└─ No
   └─ Are relationship queries needed?
      ├─ Yes → Graph Database or GraphRAG
      └─ No
         └─ Is latency critical (<10ms)?
            ├─ Yes → Optimize HNSW (higher ef_search)
            └─ No → Dense Search (vector only)
```

**DocuMind choice: Hybrid Search (best accuracy for document Q&A)**

---

# Challenge Project

## Build Production Search API

Implement a complete search API with:

### Required Features
1. **All search modes** - semantic, keyword, hybrid, auto
2. **Auto mode selection** - Detect query type and choose best mode
3. **Query expansion** - Add synonyms for better recall
4. **Result diversification** - Avoid redundant results
5. **Performance monitoring** - Track metrics

### Auto Mode Detection Hints
- Quoted strings → keyword search
- Rare terms/codes → keyword search
- Question format → semantic search
- Mixed → hybrid search

**See S8-Workshop.md for starter code and full requirements.**

---

# Wrap-Up

## Key Takeaways

### HNSW Indexing
50x faster searches with graph-based navigation

### Sparse Embeddings (BM25)
Keyword matching via PostgreSQL Full-Text Search

### Hybrid Search
RRF combines dense + sparse for 89% accuracy

### Production Tuning
Batching, caching, index parameters for enterprise scale

### Beyond Vectors
Graph databases and GraphRAG for complex relationships

**Your DocuMind search is now enterprise-ready.**

---

## Session 9 Preview

### Evaluation with RAGAS and TruLens

Next session we'll measure:
- **Faithfulness** - Does the answer match retrieved content?
- **Relevance** - Did we retrieve the right documents?
- **Precision** - How many retrieved docs were actually relevant?
- **Recall** - Did we find all relevant documents?

**Come prepared with test queries and expected answers for your document set.**

---

## Additional Resources

### pgvector Documentation
github.com/pgvector/pgvector

### HNSW Paper
"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"

### OpenAI Embeddings Guide
platform.openai.com/docs/guides/embeddings

### Hybrid Search Research
Papers on combining dense and sparse retrieval

### RuVector-Postgres
github.com/ruvnet/ruvector (unified vector-graph extension)

---

## Session Complete

### Remember:
- **HNSW** = 50x faster approximate nearest neighbor search
- **BM25** = keyword matching with Full-Text Search
- **RRF** = elegant way to combine ranking signals
- **Hybrid** = best accuracy for production
- **Batch + Cache** = cost optimization

### Your S5 Foundation + Today's Optimization = Production-Ready Search

**See you in Session 9!**
