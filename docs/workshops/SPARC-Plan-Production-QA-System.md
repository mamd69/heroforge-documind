# SPARC Implementation Plan: Production-Ready Q&A System for DocuMind

## Executive Summary

This SPARC plan provides a comprehensive blueprint for building a production-ready Q&A system that enhances DocuMind with advanced RAG capabilities, multi-model support, citation tracking, query logging, and analytics. The implementation follows Test-Driven Development (TDD) principles and integrates seamlessly with existing codebase components.

**Target Deliverable**: `src/documind/rag/production_qa.py`

**Estimated Effort**: 8-12 hours for complete implementation with tests and documentation

---

## Phase 1: SPECIFICATION (Requirements Analysis)

### 1.1 Functional Requirements

#### FR-1: Enhanced Semantic Search
- **FR-1.1**: Semantic search with configurable relevance threshold (0.0-1.0)
- **FR-1.2**: Result re-ranking by relevance score and freshness
- **FR-1.3**: Automatic deduplication of similar chunks using content similarity
- **FR-1.4**: Support for hybrid search (semantic + keyword) with configurable weights
- **FR-1.5**: Result filtering by document source, date range, or metadata

#### FR-2: Multi-Model Q&A Support
- **FR-2.1**: Support for 3+ models via OpenRouter API
  - Claude 3.5 Haiku (premium)
  - GPT-4o Mini (budget)
  - Gemini 2.5 Flash Lite (default)
  - DeepSeek V3 (opensource)
- **FR-2.2**: Side-by-side model comparison mode
- **FR-2.3**: Automatic fallback mechanism if primary model fails
- **FR-2.4**: Model selection based on query complexity (simple/medium/complex)
- **FR-2.5**: Cost tracking per model and query

#### FR-3: Citation & Attribution System
- **FR-3.1**: Track which chunks contributed to answer generation
- **FR-3.2**: Provide document links/references in standardized format
- **FR-3.3**: Highlight relevant passages with character-level precision
- **FR-3.4**: Support multiple citation formats (inline, footnote, APA-style)
- **FR-3.5**: Confidence scores for each cited source

#### FR-4: Query Logging & Analytics
- **FR-4.1**: Store all queries with responses in Supabase database
- **FR-4.2**: Track response times (embedding, search, generation, total)
- **FR-4.3**: Record user feedback (thumbs up/down, ratings)
- **FR-4.4**: Query pattern analysis (popular queries, failure patterns)
- **FR-4.5**: Model performance metrics (accuracy, latency, cost per query)

#### FR-5: User Interface
- **FR-5.1**: Rich CLI interface with colored output and formatting
- **FR-5.2**: Interactive Q&A session mode with history
- **FR-5.3**: Display sources alongside answers with expandable details
- **FR-5.4**: Export results to JSON, Markdown, or HTML formats
- **FR-5.5**: (Optional) Simple web UI using Streamlit or Flask

### 1.2 Non-Functional Requirements

#### NFR-1: Performance
- **NFR-1.1**: Total query response time < 3 seconds (p95)
- **NFR-1.2**: Embedding generation < 500ms
- **NFR-1.3**: Database search < 1 second
- **NFR-1.4**: LLM generation < 2 seconds (with default model)
- **NFR-1.5**: Support concurrent queries (10+ simultaneous users)

#### NFR-2: Reliability
- **NFR-2.1**: 99.5% uptime for search functionality
- **NFR-2.2**: Graceful degradation when models are unavailable
- **NFR-2.3**: Automatic retry logic with exponential backoff
- **NFR-2.4**: Comprehensive error handling and logging
- **NFR-2.5**: Data consistency guarantees for query logs

#### NFR-3: Scalability
- **NFR-3.1**: Handle 1000+ documents in knowledge base
- **NFR-3.2**: Support 10,000+ queries/day
- **NFR-3.3**: Horizontal scaling capability for multiple instances
- **NFR-3.4**: Efficient memory usage (< 512MB per instance)
- **NFR-3.5**: Database query optimization for large log tables

#### NFR-4: Security & Privacy
- **NFR-4.1**: API key encryption and secure storage
- **NFR-4.2**: Query sanitization to prevent injection attacks
- **NFR-4.3**: Optional PII redaction in query logs
- **NFR-4.4**: Rate limiting to prevent abuse
- **NFR-4.5**: Audit trail for all operations

#### NFR-5: Maintainability
- **NFR-5.1**: Comprehensive unit test coverage (>80%)
- **NFR-5.2**: Integration tests for all major workflows
- **NFR-5.3**: Clear documentation with examples
- **NFR-5.4**: Type hints for all public methods
- **NFR-5.5**: Modular design for easy extension

### 1.3 Integration Requirements

#### IR-1: Existing Codebase Integration
- **IR-1.1**: Leverage `search.py` functions (search_documents, hybrid_search)
- **IR-1.2**: Extend `qa_pipeline.py` functionality without breaking changes
- **IR-1.3**: Maintain compatibility with existing `compare.py` utilities
- **IR-1.4**: Use consistent error handling patterns from current modules

#### IR-2: Database Schema
- **IR-2.1**: `query_logs` table with comprehensive tracking fields
- **IR-2.2**: Indexes for efficient query pattern analysis
- **IR-2.3**: Foreign key relationships for data integrity
- **IR-2.4**: Migration scripts for schema updates

#### IR-3: External Dependencies
- **IR-3.1**: Supabase for vector search and query logging
- **IR-3.2**: OpenAI for embeddings (text-embedding-3-small)
- **IR-3.3**: OpenRouter for multi-model LLM access
- **IR-3.4**: Rich library for CLI formatting
- **IR-3.5**: (Optional) Streamlit for web interface

### 1.4 Success Criteria

#### Acceptance Tests
1. **AT-1**: System answers 90%+ of domain questions correctly
2. **AT-2**: All answers include proper source citations
3. **AT-3**: Response time < 3 seconds for 95% of queries
4. **AT-4**: Model comparison shows measurable quality differences
5. **AT-5**: Query logs capture all required metadata accurately
6. **AT-6**: Analytics provide actionable insights on usage patterns
7. **AT-7**: CLI interface provides intuitive user experience
8. **AT-8**: Zero data loss in query logging
9. **AT-9**: Graceful fallback when primary model unavailable
10. **AT-10**: Documentation enables new users to use system in < 15 minutes

---

## Phase 2: PSEUDOCODE (Algorithm Design)

### 2.1 ProductionQA Class Architecture

```python
class ProductionQA:
    """
    Production-ready Q&A system with multi-model support, citations, and analytics.

    Attributes:
        supabase: Supabase client for database operations
        openai_client: OpenAI client for embeddings
        openrouter_client: OpenRouter client for LLM inference
        default_model: Default model for queries
        fallback_models: List of fallback models
        config: Configuration object
    """

    def __init__(
        self,
        default_model: str = "google/gemini-2.5-flash-lite",
        fallback_models: List[str] = None,
        enable_logging: bool = True,
        cache_embeddings: bool = True
    ):
        """Initialize ProductionQA with configuration."""
        # Algorithm:
        # 1. Validate environment variables (API keys)
        # 2. Initialize client connections (lazy loading)
        # 3. Load configuration from env or defaults
        # 4. Setup caching layer if enabled
        # 5. Verify database connectivity
        # 6. Initialize logging handlers
        pass
```

### 2.2 Enhanced Search Algorithm

```python
def enhanced_search(
    self,
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.5,
    rerank: bool = True,
    deduplicate: bool = True,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced semantic search with re-ranking and deduplication.

    Algorithm:
    1. INPUT: query string, parameters

    2. EMBEDDING GENERATION:
       embedding = get_query_embedding(query)
       cache_key = hash(query)
       IF cache_key in embedding_cache:
           embedding = embedding_cache[cache_key]
       ELSE:
           embedding = openai.embed(query)
           embedding_cache[cache_key] = embedding

    3. INITIAL RETRIEVAL:
       # Retrieve more than needed for re-ranking
       initial_k = top_k * 2 if rerank else top_k
       results = supabase.rpc(
           "match_documents",
           {
               "query_embedding": embedding,
               "match_count": initial_k,
               "similarity_threshold": similarity_threshold,
               "filters": filters  # Document type, date range, etc.
           }
       )

    4. DEDUPLICATION (if enabled):
       IF deduplicate:
           unique_results = []
           seen_content_hashes = set()

           FOR doc in results:
               content_hash = fuzzy_hash(doc.content, threshold=0.9)
               IF content_hash NOT IN seen_content_hashes:
                   unique_results.append(doc)
                   seen_content_hashes.add(content_hash)

           results = unique_results

    5. RE-RANKING (if enabled):
       IF rerank:
           # Combine multiple ranking signals
           FOR doc in results:
               base_score = doc.similarity

               # Boost by document quality indicators
               quality_boost = calculate_quality_score(doc.metadata)

               # Boost by recency (if timestamp available)
               recency_boost = calculate_recency_score(doc.metadata.timestamp)

               # Boost by user engagement (view count, feedback)
               engagement_boost = calculate_engagement_score(doc.id)

               # Combined score with weighted factors
               doc.final_score = (
                   base_score * 0.6 +
                   quality_boost * 0.2 +
                   recency_boost * 0.1 +
                   engagement_boost * 0.1
               )

           # Sort by final score
           results.sort(key=lambda x: x.final_score, reverse=True)
           results = results[:top_k]

    6. ENRICH RESULTS:
       FOR doc in results:
           doc.document_link = generate_document_link(doc.id)
           doc.highlighted_content = highlight_query_terms(doc.content, query)
           doc.citation_format = format_citation(doc, style="inline")

    7. OUTPUT: List of enhanced document results
    """
    pass
```

### 2.3 Multi-Model Query Algorithm

```python
def query(
    self,
    question: str,
    model: Optional[str] = None,
    enable_fallback: bool = True,
    include_sources: bool = True,
    log_query: bool = True
) -> Dict[str, Any]:
    """
    Complete RAG pipeline with multi-model support and fallback.

    Algorithm:
    1. INPUT VALIDATION:
       IF NOT question OR len(question.strip()) == 0:
           RAISE ValueError("Question cannot be empty")

    2. SELECT MODEL:
       target_model = model IF model ELSE self.default_model
       complexity = analyze_query_complexity(question)

       # Auto-select model based on complexity
       IF model is None:
           IF complexity == "simple":
               target_model = "google/gemini-2.5-flash-lite"
           ELIF complexity == "medium":
               target_model = "openai/gpt-4o-mini"
           ELSE:
               target_model = "anthropic/claude-3.5-haiku"

    3. START TIMING:
       start_time = time.perf_counter()
       timing = {"embedding": 0, "search": 0, "generation": 0}

    4. RETRIEVE CONTEXT:
       embed_start = time.perf_counter()
       documents = self.enhanced_search(
           query=question,
           top_k=10,
           similarity_threshold=0.5,
           rerank=True,
           deduplicate=True
       )
       timing["embedding"] = time.perf_counter() - embed_start

    5. ASSEMBLE CONTEXT WITH CITATIONS:
       context_parts = []
       citation_map = {}  # Maps source ID to citation number

       FOR idx, doc in enumerate(documents):
           citation_num = idx + 1
           citation_map[doc.id] = citation_num

           source_header = format_source_header(doc, citation_num)
           context_parts.append(f"{source_header}\n{doc.content}")

       context = "\n\n---\n\n".join(context_parts)

    6. BUILD GROUNDED PROMPT:
       prompt = build_production_prompt(question, context, citation_map)

    7. GENERATE ANSWER WITH FALLBACK:
       search_end = time.perf_counter()
       timing["search"] = search_end - embed_start - timing["embedding"]

       gen_start = time.perf_counter()
       answer = None
       model_used = None
       error = None

       # Try primary model
       TRY:
           answer = self._generate_with_model(target_model, prompt)
           model_used = target_model

       # Fallback logic
       EXCEPT Exception as e:
           error = str(e)

           IF enable_fallback AND self.fallback_models:
               FOR fallback_model in self.fallback_models:
                   TRY:
                       answer = self._generate_with_model(fallback_model, prompt)
                       model_used = fallback_model
                       BREAK
                   EXCEPT:
                       CONTINUE

           IF answer is None:
               RAISE Exception(f"All models failed. Last error: {error}")

       timing["generation"] = time.perf_counter() - gen_start
       timing["total"] = time.perf_counter() - start_time

    8. EXTRACT CITATIONS FROM ANSWER:
       # Parse answer for citation markers like [Source 1]
       cited_sources = extract_citation_references(answer, citation_map)

    9. FORMAT SOURCES:
       formatted_sources = []
       FOR doc in documents:
           IF doc.id in cited_sources:
               formatted_sources.append({
                   "id": doc.id,
                   "citation_number": citation_map[doc.id],
                   "document": doc.document_name,
                   "chunk_index": doc.chunk_index,
                   "similarity": doc.similarity,
                   "link": doc.document_link,
                   "preview": doc.content[:200] + "...",
                   "highlighted": doc.highlighted_content,
                   "was_cited": True
               })

    10. LOG QUERY (if enabled):
        IF log_query:
            log_entry = {
                "question": question,
                "answer": answer,
                "model": model_used,
                "sources": formatted_sources,
                "response_time": timing["total"],
                "timestamp": datetime.now(timezone.utc),
                "complexity": complexity,
                "fallback_used": model_used != target_model
            }

            self.log_query(log_entry)

    11. BUILD RESPONSE:
        response = {
            "answer": answer,
            "sources": formatted_sources,
            "model": model_used,
            "timing": timing,
            "complexity": complexity,
            "query": question,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    12. OUTPUT: Complete response dictionary
    """
    pass
```

### 2.4 Model Comparison Algorithm

```python
def compare_models(
    self,
    question: str,
    models: List[str] = None,
    parallel: bool = True,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Compare responses from multiple models side-by-side.

    Algorithm:
    1. INPUT VALIDATION:
       IF models is None:
           models = ["google/gemini-2.5-flash-lite",
                     "anthropic/claude-3.5-haiku",
                     "openai/gpt-4o-mini"]

    2. SHARED CONTEXT RETRIEVAL:
       # Retrieve context once to ensure fair comparison
       documents = self.enhanced_search(question)
       context = assemble_context(documents)
       prompt = build_production_prompt(question, context)

    3. GENERATE RESPONSES:
       results = {}

       IF parallel:
           # Use ThreadPoolExecutor for concurrent requests
           WITH ThreadPoolExecutor(max_workers=len(models)) AS executor:
               futures = {}

               FOR model in models:
                   future = executor.submit(
                       self._generate_with_timing,
                       model,
                       prompt
                   )
                   futures[future] = model

               FOR future in as_completed(futures, timeout=timeout):
                   model = futures[future]
                   TRY:
                       result = future.result()
                       results[model] = result
                   EXCEPT Exception as e:
                       results[model] = {"error": str(e)}

       ELSE:
           # Sequential execution
           FOR model in models:
               TRY:
                   result = self._generate_with_timing(model, prompt)
                   results[model] = result
               EXCEPT Exception as e:
                   results[model] = {"error": str(e)}

    4. ANALYZE RESULTS:
       analysis = {
           "fastest_model": None,
           "slowest_model": None,
           "most_detailed": None,
           "shortest_answer": None,
           "longest_answer": None,
           "cost_comparison": {}
       }

       # Find fastest/slowest
       valid_results = {k: v for k, v in results.items() if "error" not in v}
       IF valid_results:
           analysis["fastest_model"] = min(
               valid_results.items(),
               key=lambda x: x[1]["latency_ms"]
           )[0]

           analysis["slowest_model"] = max(
               valid_results.items(),
               key=lambda x: x[1]["latency_ms"]
           )[0]

           # Analyze answer lengths
           analysis["shortest_answer"] = min(
               valid_results.items(),
               key=lambda x: len(x[1]["answer"])
           )[0]

           analysis["longest_answer"] = max(
               valid_results.items(),
               key=lambda x: len(x[1]["answer"])
           )[0]

           # Calculate costs
           FOR model, result in valid_results.items():
               IF "usage" in result:
                   cost = calculate_model_cost(model, result["usage"])
                   analysis["cost_comparison"][model] = cost

    5. OUTPUT: Comparison results with analysis
       RETURN {
           "query": question,
           "models_compared": models,
           "results": results,
           "analysis": analysis,
           "sources": format_sources(documents),
           "timestamp": datetime.now(timezone.utc).isoformat()
       }
    """
    pass
```

### 2.5 Query Logging Algorithm

```python
def log_query(
    self,
    query_data: Dict[str, Any],
    include_sources: bool = True,
    redact_pii: bool = False
) -> str:
    """
    Store query and response in database for analytics.

    Algorithm:
    1. PREPARE DATA:
       IF redact_pii:
           query_data["question"] = redact_pii_data(query_data["question"])
           query_data["answer"] = redact_pii_data(query_data["answer"])

    2. FORMAT SOURCES FOR JSONB:
       sources_json = None
       IF include_sources AND "sources" in query_data:
           sources_json = [
               {
                   "id": src["id"],
                   "document": src["document"],
                   "chunk": src["chunk_index"],
                   "similarity": src["similarity"],
                   "cited": src.get("was_cited", False)
               }
               FOR src in query_data["sources"]
           ]

    3. INSERT INTO DATABASE:
       log_entry = {
           "question": query_data["question"],
           "answer": query_data["answer"],
           "model": query_data["model"],
           "sources": sources_json,
           "response_time": query_data["timing"]["total"],
           "embedding_time": query_data["timing"]["embedding"],
           "search_time": query_data["timing"]["search"],
           "generation_time": query_data["timing"]["generation"],
           "complexity": query_data.get("complexity", "medium"),
           "fallback_used": query_data.get("fallback_used", False),
           "created_at": query_data["timestamp"]
       }

       response = self.supabase.table("query_logs").insert(log_entry).execute()

       IF response.error:
           RAISE Exception(f"Failed to log query: {response.error}")

    4. OUTPUT: Query log ID
       RETURN response.data[0]["id"]
    """
    pass
```

### 2.6 Analytics Algorithm

```python
def get_analytics(
    self,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Analyze query patterns and system performance.

    Algorithm:
    1. SET DATE RANGE:
       IF start_date is None:
           start_date = datetime.now() - timedelta(days=7)
       IF end_date is None:
           end_date = datetime.now()

    2. FETCH QUERY LOGS:
       query = self.supabase.table("query_logs").select("*")
       query = query.gte("created_at", start_date.isoformat())
       query = query.lte("created_at", end_date.isoformat())
       query = query.order("created_at", desc=True)
       query = query.limit(limit)

       logs = query.execute().data

    3. CALCULATE METRICS:
       analytics = {
           "total_queries": len(logs),
           "date_range": {
               "start": start_date.isoformat(),
               "end": end_date.isoformat()
           },
           "performance": {},
           "models": {},
           "popular_queries": {},
           "failure_patterns": []
       }

       # Performance metrics
       response_times = [log["response_time"] for log in logs]
       analytics["performance"] = {
           "avg_response_time": mean(response_times),
           "p50_response_time": percentile(response_times, 50),
           "p95_response_time": percentile(response_times, 95),
           "p99_response_time": percentile(response_times, 99),
           "fastest_query": min(response_times),
           "slowest_query": max(response_times)
       }

       # Model usage statistics
       model_stats = defaultdict(lambda: {
           "count": 0,
           "avg_latency": [],
           "fallback_count": 0
       })

       FOR log in logs:
           model = log["model"]
           model_stats[model]["count"] += 1
           model_stats[model]["avg_latency"].append(log["response_time"])

           IF log.get("fallback_used", False):
               model_stats[model]["fallback_count"] += 1

       FOR model, stats in model_stats.items():
           analytics["models"][model] = {
               "usage_count": stats["count"],
               "usage_percentage": (stats["count"] / len(logs)) * 100,
               "avg_latency": mean(stats["avg_latency"]),
               "fallback_rate": (stats["fallback_count"] / stats["count"]) * 100
           }

       # Popular queries (group similar questions)
       query_clusters = cluster_similar_queries(
           [log["question"] for log in logs]
       )

       analytics["popular_queries"] = [
           {
               "query_pattern": cluster["pattern"],
               "count": cluster["count"],
               "example": cluster["example"]
           }
           FOR cluster in query_clusters[:10]
       ]

       # Failure patterns
       failures = [
           log for log in logs
           IF "error" in log.get("answer", "").lower()
           OR log.get("fallback_used", False)
       ]

       analytics["failure_patterns"] = analyze_failure_patterns(failures)

    4. GENERATE INSIGHTS:
       analytics["insights"] = generate_insights(analytics)
       # Example insights:
       # - "95% of queries complete in under 2.5s"
       # - "Claude 3.5 Haiku is most reliable with 0.5% fallback rate"
       # - "Vacation policy questions are most common (23% of queries)"

    5. OUTPUT: Analytics dictionary
       RETURN analytics
    """
    pass
```

### 2.7 CLI Interface Algorithm

```python
def main():
    """
    Interactive CLI for production Q&A system.

    Algorithm:
    1. PARSE ARGUMENTS:
       parser = create_argument_parser()
       args = parser.parse_args()

    2. INITIALIZE SYSTEM:
       qa = ProductionQA(
           default_model=args.model,
           enable_logging=not args.no_logging
       )

    3. HANDLE MODES:
       IF args.mode == "query":
           # Single query mode
           result = qa.query(args.question)
           display_result(result)

       ELIF args.mode == "compare":
           # Model comparison mode
           result = qa.compare_models(args.question, args.models)
           display_comparison(result)

       ELIF args.mode == "interactive":
           # Interactive session
           console = create_rich_console()
           history = []

           WHILE True:
               question = console.input("\n[bold cyan]Question:[/] ")

               IF question.lower() in ["exit", "quit", "q"]:
                   BREAK

               IF question.lower() == "analytics":
                   analytics = qa.get_analytics()
                   display_analytics(analytics)
                   CONTINUE

               WITH console.status("[bold green]Thinking..."):
                   result = qa.query(question)

               display_result(result)
               history.append((question, result))

       ELIF args.mode == "analytics":
           # Analytics mode
           analytics = qa.get_analytics(
               start_date=args.start_date,
               end_date=args.end_date
           )
           display_analytics(analytics)

    4. FORMAT OUTPUT:
       IF args.json:
           print(json.dumps(result, indent=2))
       ELIF args.markdown:
           print(format_as_markdown(result))
       ELSE:
           # Rich formatted output
           display_rich_output(result)
    """
    pass
```

---

## Phase 3: ARCHITECTURE (System Design)

### 3.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ProductionQA System                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌────────────────┐         ┌─────────────────┐
│  Search       │          │   Generation   │         │   Analytics     │
│  Component    │          │   Component    │         │   Component     │
├───────────────┤          ├────────────────┤         ├─────────────────┤
│ - enhanced_   │          │ - query()      │         │ - log_query()   │
│   search()    │          │ - compare_     │         │ - get_          │
│ - rerank()    │          │   models()     │         │   analytics()   │
│ - deduplicate│          │ - fallback_    │         │ - generate_     │
│ - highlight() │          │   logic()      │         │   insights()    │
└───────┬───────┘          └────────┬───────┘         └────────┬────────┘
        │                           │                          │
        │                           │                          │
        └───────────────────────────┼──────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌────────────────┐         ┌─────────────────┐
│  Supabase     │          │   OpenRouter   │         │   OpenAI        │
│  (Vector DB)  │          │   (Multi-LLM)  │         │   (Embeddings)  │
└───────────────┘          └────────────────┘         └─────────────────┘
```

### 3.2 Class Diagram

```python
┌────────────────────────────────────────────────────────────────┐
│                        ProductionQA                            │
├────────────────────────────────────────────────────────────────┤
│ - supabase: Client                                             │
│ - openai_client: OpenAI                                        │
│ - openrouter_client: OpenAI                                    │
│ - default_model: str                                           │
│ - fallback_models: List[str]                                   │
│ - config: Config                                               │
│ - embedding_cache: Dict[str, List[float]]                      │
├────────────────────────────────────────────────────────────────┤
│ + __init__(default_model, fallback_models, enable_logging)    │
│ + enhanced_search(query, top_k, threshold, rerank, filters)   │
│ + query(question, model, fallback, sources, logging)           │
│ + compare_models(question, models, parallel, timeout)          │
│ + log_query(query_data, include_sources, redact_pii)          │
│ + get_analytics(start_date, end_date, limit)                  │
│ + _generate_with_model(model, prompt)                          │
│ + _generate_with_timing(model, prompt)                         │
│ - _validate_environment()                                      │
│ - _initialize_clients()                                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      CitationManager                           │
├────────────────────────────────────────────────────────────────┤
│ - citation_map: Dict[str, int]                                 │
│ - citation_style: str                                          │
├────────────────────────────────────────────────────────────────┤
│ + format_source_header(doc, citation_num)                      │
│ + extract_citations(answer, citation_map)                      │
│ + format_citation(doc, style)                                  │
│ + highlight_query_terms(content, query)                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                     AnalyticsEngine                            │
├────────────────────────────────────────────────────────────────┤
│ - supabase: Client                                             │
│ - cache: Dict[str, Any]                                        │
├────────────────────────────────────────────────────────────────┤
│ + calculate_performance_metrics(logs)                          │
│ + analyze_model_usage(logs)                                    │
│ + cluster_similar_queries(queries)                             │
│ + analyze_failure_patterns(failures)                           │
│ + generate_insights(analytics)                                 │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Database Schema

```sql
-- Query logs table
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,

    -- Query information
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    query_hash VARCHAR(64),  -- For deduplication and clustering

    -- Model information
    model VARCHAR(100) NOT NULL,
    fallback_used BOOLEAN DEFAULT false,

    -- Sources (JSONB for flexibility)
    sources JSONB,

    -- Performance metrics
    response_time FLOAT NOT NULL,
    embedding_time FLOAT,
    search_time FLOAT,
    generation_time FLOAT,

    -- Metadata
    complexity VARCHAR(20),  -- simple, medium, complex
    user_id VARCHAR(100),    -- Optional user tracking
    session_id VARCHAR(100), -- Session grouping

    -- User feedback
    feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
    feedback_comment TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at DESC);
CREATE INDEX idx_query_logs_model ON query_logs(model);
CREATE INDEX idx_query_logs_query_hash ON query_logs(query_hash);
CREATE INDEX idx_query_logs_session_id ON query_logs(session_id);
CREATE INDEX idx_query_logs_response_time ON query_logs(response_time);

-- GIN index for JSONB sources
CREATE INDEX idx_query_logs_sources ON query_logs USING GIN (sources);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_query_logs_updated_at
    BEFORE UPDATE ON query_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### 3.4 Configuration Management

```python
# config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ProductionQAConfig:
    """Configuration for ProductionQA system."""

    # Model settings
    default_model: str = "google/gemini-2.5-flash-lite"
    fallback_models: List[str] = None
    model_timeout: float = 30.0

    # Search settings
    default_top_k: int = 10
    similarity_threshold: float = 0.5
    enable_reranking: bool = True
    enable_deduplication: bool = True

    # Performance settings
    max_context_tokens: int = 3000
    max_response_tokens: int = 500
    embedding_cache_size: int = 1000
    enable_parallel_models: bool = True

    # Logging settings
    enable_query_logging: bool = True
    enable_pii_redaction: bool = False
    log_source_details: bool = True

    # Citation settings
    citation_style: str = "inline"  # inline, footnote, apa
    highlight_sources: bool = True

    # Analytics settings
    analytics_retention_days: int = 90
    enable_query_clustering: bool = True

    # API settings
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None

    def __post_init__(self):
        """Initialize with defaults if not provided."""
        if self.fallback_models is None:
            self.fallback_models = [
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-haiku"
            ]
```

### 3.5 Error Handling Strategy

```python
class ProductionQAError(Exception):
    """Base exception for ProductionQA errors."""
    pass

class ModelInferenceError(ProductionQAError):
    """Error during LLM inference."""
    pass

class SearchError(ProductionQAError):
    """Error during document search."""
    pass

class LoggingError(ProductionQAError):
    """Error during query logging."""
    pass

class ConfigurationError(ProductionQAError):
    """Error in system configuration."""
    pass

# Error handling decorator
def handle_errors(fallback_value=None, log_error=True):
    """Decorator for consistent error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")

                if fallback_value is not None:
                    return fallback_value

                raise ProductionQAError(f"Operation failed: {e}") from e
        return wrapper
    return decorator
```

### 3.6 File Structure

```
src/documind/rag/
├── production_qa.py          # Main ProductionQA class
├── citation_manager.py       # Citation formatting and tracking
├── analytics_engine.py       # Analytics and insights generation
├── config.py                 # Configuration management
├── exceptions.py             # Custom exceptions
└── cli.py                    # CLI interface

tests/rag/
├── test_production_qa.py     # Unit tests for ProductionQA
├── test_citation_manager.py  # Citation system tests
├── test_analytics_engine.py  # Analytics tests
├── test_integration.py       # End-to-end integration tests
└── fixtures/                 # Test data and fixtures

docs/workshops/
└── production_qa_guide.md    # User guide and documentation
```

---

## Phase 4: REFINEMENT (TDD Implementation)

### 4.1 Test Suite Structure

#### 4.1.1 Unit Tests for Enhanced Search

```python
# tests/rag/test_production_qa.py

import pytest
from src.documind.rag.production_qa import ProductionQA

class TestEnhancedSearch:
    """Unit tests for enhanced search functionality."""

    @pytest.fixture
    def qa_system(self):
        """Create ProductionQA instance for testing."""
        return ProductionQA(enable_logging=False)

    def test_enhanced_search_basic(self, qa_system):
        """Test basic enhanced search returns results."""
        results = qa_system.enhanced_search(
            "vacation policy",
            top_k=5
        )

        assert len(results) <= 5
        assert all("similarity" in r for r in results)
        assert all("content" in r for r in results)

    def test_enhanced_search_threshold_filtering(self, qa_system):
        """Test similarity threshold filters low-quality results."""
        results = qa_system.enhanced_search(
            "vacation policy",
            top_k=10,
            similarity_threshold=0.7
        )

        assert all(r["similarity"] >= 0.7 for r in results)

    def test_enhanced_search_reranking(self, qa_system):
        """Test re-ranking improves result quality."""
        # Get results without re-ranking
        no_rerank = qa_system.enhanced_search(
            "vacation policy",
            top_k=5,
            rerank=False
        )

        # Get results with re-ranking
        with_rerank = qa_system.enhanced_search(
            "vacation policy",
            top_k=5,
            rerank=True
        )

        # Results should be sorted by final_score when reranked
        assert len(with_rerank) <= 5
        if len(with_rerank) > 1:
            scores = [r.get("final_score", r["similarity"]) for r in with_rerank]
            assert scores == sorted(scores, reverse=True)

    def test_enhanced_search_deduplication(self, qa_system):
        """Test deduplication removes similar chunks."""
        results_with_dups = qa_system.enhanced_search(
            "vacation policy",
            top_k=10,
            deduplicate=False
        )

        results_deduped = qa_system.enhanced_search(
            "vacation policy",
            top_k=10,
            deduplicate=True
        )

        # Deduplication might reduce result count
        assert len(results_deduped) <= len(results_with_dups)

    def test_enhanced_search_with_filters(self, qa_system):
        """Test filtering by document metadata."""
        results = qa_system.enhanced_search(
            "vacation policy",
            top_k=5,
            filters={"document_type": "hr_policy"}
        )

        # All results should match filter
        assert all(
            r.get("metadata", {}).get("document_type") == "hr_policy"
            for r in results
        )

    def test_enhanced_search_empty_query(self, qa_system):
        """Test error handling for empty queries."""
        with pytest.raises(ValueError, match="empty"):
            qa_system.enhanced_search("")

    def test_enhanced_search_citation_enrichment(self, qa_system):
        """Test that results include citation metadata."""
        results = qa_system.enhanced_search("vacation policy", top_k=3)

        for result in results:
            assert "document_link" in result
            assert "highlighted_content" in result
            assert "citation_format" in result
```

#### 4.1.2 Unit Tests for Query Method

```python
class TestQuery:
    """Unit tests for main query method."""

    @pytest.fixture
    def qa_system(self):
        """Create ProductionQA instance for testing."""
        return ProductionQA(enable_logging=False)

    def test_query_basic(self, qa_system):
        """Test basic query returns answer with sources."""
        result = qa_system.query("What is the vacation policy?")

        assert "answer" in result
        assert "sources" in result
        assert "model" in result
        assert "timing" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_with_specific_model(self, qa_system):
        """Test query with specific model selection."""
        result = qa_system.query(
            "What is the vacation policy?",
            model="anthropic/claude-3.5-haiku"
        )

        assert result["model"] == "anthropic/claude-3.5-haiku"

    def test_query_response_time(self, qa_system):
        """Test that query completes within performance target."""
        import time

        start = time.perf_counter()
        result = qa_system.query("What is the vacation policy?")
        elapsed = time.perf_counter() - start

        # Should complete within 3 seconds (NFR-1.1)
        assert elapsed < 3.0
        assert result["timing"]["total"] < 3.0

    def test_query_includes_citations(self, qa_system):
        """Test that answer includes proper citations."""
        result = qa_system.query("What is the vacation policy?")

        # Answer should reference sources
        assert "[Source" in result["answer"] or "source" in result["answer"].lower()

        # Sources should be provided
        assert len(result["sources"]) > 0

        # Sources should have required fields
        for source in result["sources"]:
            assert "citation_number" in source
            assert "document" in source
            assert "was_cited" in source

    def test_query_fallback_mechanism(self, qa_system, monkeypatch):
        """Test fallback to alternative model on failure."""
        # Mock primary model to fail
        original_generate = qa_system._generate_with_model

        def mock_generate(model, prompt):
            if model == qa_system.default_model:
                raise Exception("Primary model failed")
            return original_generate(model, prompt)

        monkeypatch.setattr(qa_system, "_generate_with_model", mock_generate)

        result = qa_system.query(
            "What is the vacation policy?",
            enable_fallback=True
        )

        # Should succeed with fallback model
        assert "answer" in result
        assert result["model"] != qa_system.default_model

    def test_query_without_fallback_fails(self, qa_system, monkeypatch):
        """Test that query fails without fallback when model errors."""
        def mock_generate(model, prompt):
            raise Exception("Model failed")

        monkeypatch.setattr(qa_system, "_generate_with_model", mock_generate)

        with pytest.raises(Exception, match="failed"):
            qa_system.query(
                "What is the vacation policy?",
                enable_fallback=False
            )

    def test_query_timing_breakdown(self, qa_system):
        """Test that timing metrics are accurate."""
        result = qa_system.query("What is the vacation policy?")

        timing = result["timing"]
        assert "embedding" in timing
        assert "search" in timing
        assert "generation" in timing
        assert "total" in timing

        # Total should be sum of parts (with small tolerance)
        total_calculated = (
            timing["embedding"] +
            timing["search"] +
            timing["generation"]
        )
        assert abs(timing["total"] - total_calculated) < 0.1
```

#### 4.1.3 Unit Tests for Model Comparison

```python
class TestModelComparison:
    """Unit tests for multi-model comparison."""

    @pytest.fixture
    def qa_system(self):
        """Create ProductionQA instance for testing."""
        return ProductionQA(enable_logging=False)

    def test_compare_models_basic(self, qa_system):
        """Test basic model comparison."""
        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=[
                "google/gemini-2.5-flash-lite",
                "openai/gpt-4o-mini"
            ]
        )

        assert "query" in result
        assert "models_compared" in result
        assert "results" in result
        assert "analysis" in result

        # Should have results for both models
        assert len(result["results"]) == 2

    def test_compare_models_parallel(self, qa_system):
        """Test parallel execution is faster than sequential."""
        models = [
            "google/gemini-2.5-flash-lite",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku"
        ]

        import time

        # Parallel execution
        start = time.perf_counter()
        parallel_result = qa_system.compare_models(
            "What is the vacation policy?",
            models=models,
            parallel=True
        )
        parallel_time = time.perf_counter() - start

        # Sequential execution
        start = time.perf_counter()
        sequential_result = qa_system.compare_models(
            "What is the vacation policy?",
            models=models,
            parallel=False
        )
        sequential_time = time.perf_counter() - start

        # Parallel should be faster (at least 1.5x)
        assert parallel_time < sequential_time * 0.7

    def test_compare_models_analysis(self, qa_system):
        """Test that comparison includes analysis."""
        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=[
                "google/gemini-2.5-flash-lite",
                "openai/gpt-4o-mini"
            ]
        )

        analysis = result["analysis"]

        # Should identify fastest/slowest
        assert "fastest_model" in analysis
        assert "slowest_model" in analysis

        # Should analyze answer characteristics
        assert "shortest_answer" in analysis
        assert "longest_answer" in analysis

        # Should provide cost comparison
        assert "cost_comparison" in analysis

    def test_compare_models_handles_failures(self, qa_system, monkeypatch):
        """Test graceful handling of model failures."""
        def mock_generate(model, prompt):
            if "gemini" in model:
                raise Exception("Model unavailable")
            return {"answer": "Test answer", "latency_ms": 100}

        monkeypatch.setattr(qa_system, "_generate_with_timing", mock_generate)

        result = qa_system.compare_models(
            "What is the vacation policy?",
            models=[
                "google/gemini-2.5-flash-lite",
                "openai/gpt-4o-mini"
            ]
        )

        # Should have error for failed model
        assert "error" in result["results"]["google/gemini-2.5-flash-lite"]

        # Should have success for working model
        assert "answer" in result["results"]["openai/gpt-4o-mini"]
```

#### 4.1.4 Unit Tests for Query Logging

```python
class TestQueryLogging:
    """Unit tests for query logging functionality."""

    @pytest.fixture
    def qa_system(self):
        """Create ProductionQA instance with logging enabled."""
        return ProductionQA(enable_logging=True)

    def test_log_query_stores_data(self, qa_system, db_connection):
        """Test that queries are logged to database."""
        # Perform a query
        result = qa_system.query("What is the vacation policy?")

        # Check database
        logs = db_connection.table("query_logs").select("*").order(
            "created_at", desc=True
        ).limit(1).execute()

        assert len(logs.data) > 0
        latest_log = logs.data[0]

        assert latest_log["question"] == "What is the vacation policy?"
        assert latest_log["answer"] == result["answer"]
        assert latest_log["model"] == result["model"]

    def test_log_query_includes_timing(self, qa_system, db_connection):
        """Test that timing metrics are logged."""
        result = qa_system.query("What is the vacation policy?")

        logs = db_connection.table("query_logs").select("*").order(
            "created_at", desc=True
        ).limit(1).execute()

        latest_log = logs.data[0]

        assert latest_log["response_time"] > 0
        assert latest_log["embedding_time"] is not None
        assert latest_log["search_time"] is not None
        assert latest_log["generation_time"] is not None

    def test_log_query_includes_sources(self, qa_system, db_connection):
        """Test that sources are logged as JSONB."""
        result = qa_system.query("What is the vacation policy?")

        logs = db_connection.table("query_logs").select("*").order(
            "created_at", desc=True
        ).limit(1).execute()

        latest_log = logs.data[0]

        assert latest_log["sources"] is not None
        assert isinstance(latest_log["sources"], list)
        assert len(latest_log["sources"]) > 0

        # Check source structure
        source = latest_log["sources"][0]
        assert "document" in source
        assert "similarity" in source

    def test_log_query_pii_redaction(self, qa_system, db_connection):
        """Test PII redaction in logged queries."""
        # Query with potential PII
        query_with_pii = "What is John Smith's vacation policy? Email: john@example.com"

        qa_system_redact = ProductionQA(
            enable_logging=True,
            config={"enable_pii_redaction": True}
        )

        result = qa_system_redact.query(query_with_pii)

        logs = db_connection.table("query_logs").select("*").order(
            "created_at", desc=True
        ).limit(1).execute()

        latest_log = logs.data[0]

        # PII should be redacted
        assert "john@example.com" not in latest_log["question"]
        assert "[REDACTED]" in latest_log["question"]

    def test_query_without_logging(self, qa_system):
        """Test that logging can be disabled per query."""
        result = qa_system.query(
            "What is the vacation policy?",
            log_query=False
        )

        # Query should succeed but not be logged
        assert "answer" in result
        # Verification would require checking DB didn't increase count
```

#### 4.1.5 Unit Tests for Analytics

```python
class TestAnalytics:
    """Unit tests for analytics functionality."""

    @pytest.fixture
    def qa_system_with_data(self, db_connection):
        """Create QA system with sample query logs."""
        qa = ProductionQA(enable_logging=True)

        # Generate sample queries
        test_queries = [
            "What is the vacation policy?",
            "How many sick days do I get?",
            "What are the health insurance options?",
            "What is the vacation policy?",  # Duplicate for clustering
        ]

        for query in test_queries:
            qa.query(query)

        return qa

    def test_get_analytics_basic(self, qa_system_with_data):
        """Test basic analytics retrieval."""
        analytics = qa_system_with_data.get_analytics()

        assert "total_queries" in analytics
        assert analytics["total_queries"] > 0

        assert "performance" in analytics
        assert "models" in analytics
        assert "popular_queries" in analytics

    def test_get_analytics_performance_metrics(self, qa_system_with_data):
        """Test performance metrics calculation."""
        analytics = qa_system_with_data.get_analytics()

        perf = analytics["performance"]

        assert "avg_response_time" in perf
        assert "p50_response_time" in perf
        assert "p95_response_time" in perf
        assert "p99_response_time" in perf
        assert "fastest_query" in perf
        assert "slowest_query" in perf

        # Sanity checks
        assert perf["fastest_query"] <= perf["avg_response_time"]
        assert perf["avg_response_time"] <= perf["p95_response_time"]
        assert perf["p95_response_time"] <= perf["slowest_query"]

    def test_get_analytics_model_usage(self, qa_system_with_data):
        """Test model usage statistics."""
        analytics = qa_system_with_data.get_analytics()

        models = analytics["models"]

        # Should have at least one model
        assert len(models) > 0

        for model, stats in models.items():
            assert "usage_count" in stats
            assert "usage_percentage" in stats
            assert "avg_latency" in stats
            assert "fallback_rate" in stats

            # Percentages should sum to ~100
            assert 0 <= stats["usage_percentage"] <= 100

    def test_get_analytics_popular_queries(self, qa_system_with_data):
        """Test popular query identification."""
        analytics = qa_system_with_data.get_analytics()

        popular = analytics["popular_queries"]

        # Should identify common patterns
        assert len(popular) > 0

        for query_cluster in popular:
            assert "query_pattern" in query_cluster
            assert "count" in query_cluster
            assert "example" in query_cluster
            assert query_cluster["count"] >= 1

    def test_get_analytics_date_filtering(self, qa_system_with_data):
        """Test analytics with date range filtering."""
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        analytics = qa_system_with_data.get_analytics(
            start_date=start_date,
            end_date=end_date
        )

        assert analytics["date_range"]["start"] == start_date.isoformat()
        assert analytics["date_range"]["end"] == end_date.isoformat()

    def test_get_analytics_insights(self, qa_system_with_data):
        """Test automatic insight generation."""
        analytics = qa_system_with_data.get_analytics()

        assert "insights" in analytics
        assert isinstance(analytics["insights"], list)
        assert len(analytics["insights"]) > 0

        # Insights should be human-readable strings
        for insight in analytics["insights"]:
            assert isinstance(insight, str)
            assert len(insight) > 0
```

#### 4.1.6 Integration Tests

```python
class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_rag_pipeline(self):
        """Test complete RAG pipeline from query to answer."""
        qa = ProductionQA()

        question = "What is the vacation policy for new employees?"

        # Execute complete pipeline
        result = qa.query(question)

        # Verify all components worked
        assert result["answer"]
        assert len(result["sources"]) > 0
        assert result["timing"]["total"] < 3.0

        # Verify citations
        assert any(
            src["was_cited"]
            for src in result["sources"]
        )

    def test_model_comparison_workflow(self):
        """Test complete model comparison workflow."""
        qa = ProductionQA()

        comparison = qa.compare_models(
            "What is the vacation policy?",
            models=[
                "google/gemini-2.5-flash-lite",
                "openai/gpt-4o-mini"
            ]
        )

        # All models should return results
        assert all(
            "answer" in result or "error" in result
            for result in comparison["results"].values()
        )

        # Analysis should identify differences
        assert comparison["analysis"]["fastest_model"]
        assert comparison["analysis"]["slowest_model"]

    def test_analytics_workflow(self):
        """Test complete analytics workflow."""
        qa = ProductionQA()

        # Generate some queries
        queries = [
            "What is the vacation policy?",
            "How many sick days?",
            "Health insurance options?"
        ]

        for q in queries:
            qa.query(q)

        # Get analytics
        analytics = qa.get_analytics()

        # Should have meaningful data
        assert analytics["total_queries"] >= len(queries)
        assert len(analytics["models"]) > 0
        assert len(analytics["popular_queries"]) > 0

    def test_error_recovery(self):
        """Test system recovers gracefully from errors."""
        qa = ProductionQA()

        # Test with invalid model (should fallback)
        result = qa.query(
            "What is the vacation policy?",
            model="invalid/model-name",
            enable_fallback=True
        )

        # Should succeed with fallback
        assert "answer" in result
        assert result["model"] != "invalid/model-name"
```

### 4.2 TDD Development Workflow

#### Step 1: Write Failing Tests
```bash
# Create test file
touch tests/rag/test_production_qa.py

# Write tests (they should fail initially)
pytest tests/rag/test_production_qa.py -v
# Expected: All tests fail (implementation doesn't exist yet)
```

#### Step 2: Implement Minimum Code to Pass
```bash
# Create implementation file
touch src/documind/rag/production_qa.py

# Implement ProductionQA class incrementally
# Run tests after each method implementation
pytest tests/rag/test_production_qa.py::TestEnhancedSearch -v
```

#### Step 3: Refactor
```bash
# After tests pass, refactor for clarity and performance
# Run tests to ensure refactoring didn't break anything
pytest tests/rag/test_production_qa.py -v --cov=src/documind/rag/production_qa
```

#### Step 4: Repeat for Each Component
```bash
# Enhanced Search → Query → Model Comparison → Logging → Analytics
# Build incrementally, ensuring each component is tested before moving on
```

### 4.3 Test Coverage Goals

```bash
# Target coverage metrics:
# - Overall coverage: >80%
# - Critical paths: >95% (query, enhanced_search, log_query)
# - Error handling: >90%

# Run coverage report
pytest tests/rag/ --cov=src/documind/rag/production_qa --cov-report=html
```

---

## Phase 5: COMPLETION (Integration & Validation)

### 5.1 Implementation Checklist

#### Core Features
- [ ] ProductionQA class with complete initialization
- [ ] Enhanced search with re-ranking and deduplication
- [ ] Multi-model query support with fallback
- [ ] Citation tracking and formatting
- [ ] Query logging to Supabase
- [ ] Analytics engine with insights
- [ ] CLI interface with rich formatting

#### Database Setup
- [ ] Create `query_logs` table with schema
- [ ] Add necessary indexes for performance
- [ ] Create triggers for auto-updating fields
- [ ] Test database connectivity and permissions

#### Testing
- [ ] All unit tests passing (>80% coverage)
- [ ] Integration tests passing
- [ ] Performance tests validate <3s response time
- [ ] Error handling tests pass
- [ ] Database tests verify logging accuracy

#### Documentation
- [ ] Inline code documentation (docstrings)
- [ ] User guide with examples
- [ ] API reference documentation
- [ ] Configuration guide
- [ ] Troubleshooting section

#### Performance Optimization
- [ ] Embedding caching implemented
- [ ] Database query optimization
- [ ] Parallel model execution
- [ ] Memory usage profiling
- [ ] Latency profiling and optimization

### 5.2 Integration Steps

#### Step 1: Database Schema Setup
```bash
# Apply database migration
python -m src.documind.rag.production_qa --setup-database

# Verify table creation
python -m src.documind.rag.production_qa --verify-database
```

#### Step 2: Environment Configuration
```bash
# Create .env file with required keys
cat > .env << EOF
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...
SUPABASE_URL=https://...
SUPABASE_ANON_KEY=eyJ...
EOF

# Verify configuration
python -m src.documind.rag.production_qa --check-config
```

#### Step 3: Run Integration Tests
```bash
# Run all integration tests
pytest tests/rag/test_integration.py -v

# Run performance benchmarks
pytest tests/rag/test_production_qa.py::TestPerformance -v
```

#### Step 4: CLI Testing
```bash
# Test basic query
python -m src.documind.rag.production_qa "What is the vacation policy?"

# Test model comparison
python -m src.documind.rag.production_qa "What is the vacation policy?" --compare

# Test analytics
python -m src.documind.rag.production_qa --analytics

# Test interactive mode
python -m src.documind.rag.production_qa --interactive
```

### 5.3 Validation & Acceptance

#### Performance Validation
```python
# tests/rag/test_performance.py

def test_response_time_p95():
    """Validate 95th percentile response time < 3s."""
    qa = ProductionQA()

    response_times = []
    for _ in range(100):
        start = time.perf_counter()
        qa.query("What is the vacation policy?")
        response_times.append(time.perf_counter() - start)

    p95 = np.percentile(response_times, 95)
    assert p95 < 3.0, f"P95 response time {p95:.2f}s exceeds 3s target"

def test_concurrent_queries():
    """Validate system handles 10+ concurrent queries."""
    qa = ProductionQA()

    def run_query():
        return qa.query("What is the vacation policy?")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_query) for _ in range(10)]
        results = [f.result() for f in futures]

    # All queries should succeed
    assert all("answer" in r for r in results)
```

#### Accuracy Validation
```python
def test_answer_accuracy():
    """Validate answers are accurate and include citations."""
    qa = ProductionQA()

    # Test questions with known answers
    test_cases = [
        {
            "question": "How many vacation days do new employees get?",
            "expected_keywords": ["15 days", "vacation", "new employees"],
            "must_cite": True
        },
        # Add more test cases...
    ]

    for case in test_cases:
        result = qa.query(case["question"])

        # Check answer contains expected keywords
        answer_lower = result["answer"].lower()
        for keyword in case["expected_keywords"]:
            assert keyword.lower() in answer_lower

        # Check citations are included
        if case["must_cite"]:
            assert "[Source" in result["answer"]
            assert any(src["was_cited"] for src in result["sources"])
```

### 5.4 Deployment Preparation

#### Documentation
```bash
# Generate API documentation
python -m pdoc src.documind.rag.production_qa --html --output-dir docs/api

# Create user guide
python scripts/generate_user_guide.py > docs/production_qa_guide.md
```

#### Performance Profiling
```bash
# Profile memory usage
python -m memory_profiler src/documind/rag/production_qa.py

# Profile CPU usage
python -m cProfile -o profile.stats src/documind/rag/production_qa.py
python -m pstats profile.stats
```

#### Final Validation
```bash
# Run complete test suite
pytest tests/rag/ -v --cov=src/documind/rag/production_qa --cov-report=html

# Run linting
ruff check src/documind/rag/production_qa.py
black src/documind/rag/production_qa.py --check

# Run type checking
mypy src/documind/rag/production_qa.py
```

---

## Appendix A: File Templates

### A.1 ProductionQA Class Skeleton

```python
"""
DocuMind Production-Ready Q&A System

Complete RAG implementation with multi-model support, citations, and analytics.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

from .search import search_documents, hybrid_search, get_query_embedding
from .qa_pipeline import MODELS, MODEL_INFO, assemble_context, build_qa_prompt
from .citation_manager import CitationManager
from .analytics_engine import AnalyticsEngine
from .exceptions import ProductionQAError, ModelInferenceError, SearchError


class ProductionQA:
    """
    Production-ready Q&A system with advanced RAG capabilities.

    Features:
    - Enhanced semantic search with re-ranking
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

    def __init__(
        self,
        default_model: str = "google/gemini-2.5-flash-lite",
        fallback_models: Optional[List[str]] = None,
        enable_logging: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ProductionQA system."""
        # TODO: Implement initialization
        pass

    def enhanced_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        rerank: bool = True,
        deduplicate: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced semantic search with re-ranking and deduplication."""
        # TODO: Implement enhanced search
        pass

    def query(
        self,
        question: str,
        model: Optional[str] = None,
        enable_fallback: bool = True,
        include_sources: bool = True,
        log_query: bool = True
    ) -> Dict[str, Any]:
        """Execute complete RAG pipeline."""
        # TODO: Implement query method
        pass

    def compare_models(
        self,
        question: str,
        models: Optional[List[str]] = None,
        parallel: bool = True,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Compare responses from multiple models."""
        # TODO: Implement model comparison
        pass

    def log_query(
        self,
        query_data: Dict[str, Any],
        include_sources: bool = True,
        redact_pii: bool = False
    ) -> str:
        """Store query and response in database."""
        # TODO: Implement query logging
        pass

    def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Analyze query patterns and performance."""
        # TODO: Implement analytics
        pass
```

### A.2 Database Migration Script

```python
"""
Database migration script for query_logs table.
"""

from supabase import create_client
import os

def create_query_logs_table(supabase):
    """Create query_logs table with all required fields."""

    sql = """
    -- Query logs table
    CREATE TABLE IF NOT EXISTS query_logs (
        id BIGSERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        query_hash VARCHAR(64),
        model VARCHAR(100) NOT NULL,
        fallback_used BOOLEAN DEFAULT false,
        sources JSONB,
        response_time FLOAT NOT NULL,
        embedding_time FLOAT,
        search_time FLOAT,
        generation_time FLOAT,
        complexity VARCHAR(20),
        user_id VARCHAR(100),
        session_id VARCHAR(100),
        feedback_rating INTEGER CHECK (feedback_rating BETWEEN 1 AND 5),
        feedback_comment TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_query_logs_created_at
        ON query_logs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_query_logs_model
        ON query_logs(model);
    CREATE INDEX IF NOT EXISTS idx_query_logs_query_hash
        ON query_logs(query_hash);
    CREATE INDEX IF NOT EXISTS idx_query_logs_sources
        ON query_logs USING GIN (sources);

    -- Trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER update_query_logs_updated_at
        BEFORE UPDATE ON query_logs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """

    # Execute migration
    supabase.rpc("exec_sql", {"sql": sql}).execute()
    print("✓ query_logs table created successfully")

if __name__ == "__main__":
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_ANON_KEY")
    )

    create_query_logs_table(supabase)
```

---

## Appendix B: GitHub Issue Template

```markdown
# Production-Ready Q&A System - Implementation

## Overview
Build a production-ready Q&A system for DocuMind with enhanced RAG capabilities, multi-model support, citation tracking, and comprehensive analytics.

## Objectives
- ✅ Enhanced semantic search with re-ranking and deduplication
- ✅ Multi-model Q&A with automatic fallback
- ✅ Citation system with source attribution
- ✅ Query logging for analytics
- ✅ Performance optimization (<3s response time)
- ✅ Rich CLI interface

## Implementation Plan
Following SPARC methodology (detailed plan in `docs/workshops/SPARC-Plan-Production-QA-System.md`):

### Phase 1: Specification ✓
- [x] Functional requirements defined
- [x] Non-functional requirements specified
- [x] Success criteria established

### Phase 2: Pseudocode ✓
- [x] Algorithm design for enhanced search
- [x] Query pipeline architecture
- [x] Model comparison logic
- [x] Analytics algorithms

### Phase 3: Architecture ✓
- [x] Component architecture designed
- [x] Database schema defined
- [x] Error handling strategy
- [x] File structure planned

### Phase 4: Refinement (TDD)
- [ ] Write unit tests for enhanced search
- [ ] Implement enhanced search
- [ ] Write unit tests for query method
- [ ] Implement query method
- [ ] Write unit tests for model comparison
- [ ] Implement model comparison
- [ ] Write unit tests for logging
- [ ] Implement query logging
- [ ] Write unit tests for analytics
- [ ] Implement analytics engine
- [ ] Integration tests
- [ ] CLI implementation

### Phase 5: Completion
- [ ] Database setup and migration
- [ ] Environment configuration
- [ ] Performance validation
- [ ] Accuracy validation
- [ ] Documentation
- [ ] Final review and merge

## Deliverables
1. `src/documind/rag/production_qa.py` - Main implementation
2. `tests/rag/test_production_qa.py` - Comprehensive test suite
3. Database migration for `query_logs` table
4. User documentation and examples

## Success Criteria
- [ ] Response time <3s for 95% of queries
- [ ] All answers include proper citations
- [ ] Query logging captures 100% of queries
- [ ] Analytics provide actionable insights
- [ ] Test coverage >80%
- [ ] CLI provides intuitive UX

## Timeline
Estimated: 8-12 hours

## Resources
- Full SPARC plan: `docs/workshops/SPARC-Plan-Production-QA-System.md`
- Existing modules: `search.py`, `qa_pipeline.py`, `compare.py`
- Database: Supabase with vector search
```

---

## Summary

This SPARC implementation plan provides a complete blueprint for building a production-ready Q&A system for DocuMind. The plan follows a systematic approach through all five phases of the SPARC methodology:

1. **Specification**: Comprehensive requirements analysis with functional, non-functional, and integration requirements
2. **Pseudocode**: Detailed algorithms for each major component
3. **Architecture**: System design, database schema, and component interactions
4. **Refinement**: TDD approach with extensive test coverage
5. **Completion**: Integration steps, validation procedures, and deployment preparation

The plan is ready to be executed by development teams and can be directly converted into a GitHub issue for tracking progress. All components integrate seamlessly with the existing DocuMind codebase while adding significant production-ready capabilities.
