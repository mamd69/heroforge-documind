"""
DocuMind RAG (Retrieval-Augmented Generation) Module

Provides document search, retrieval, and question-answering capabilities
for the DocuMind system.

Includes both RAG and CAG pipelines:
- RAG: Retrieval-Augmented Generation (semantic search + retrieval)
- CAG: Context-Augmented Generation (full context, no database)
"""

from .search import (
    get_query_embedding,
    search_documents,
    hybrid_search,
)

from .qa_pipeline import (
    assemble_context,
    build_qa_prompt,
    generate_answer,
    compare_models,
)

from .cag_pipeline import (
    load_all_documents,
    generate_answer_cag,
)

__all__ = [
    # Search functions (RAG)
    "get_query_embedding",
    "search_documents",
    "hybrid_search",
    # Q&A Pipeline functions (RAG)
    "assemble_context",
    "build_qa_prompt",
    "generate_answer",
    "compare_models",
    # CAG Pipeline functions (no database)
    "load_all_documents",
    "generate_answer_cag",
]
