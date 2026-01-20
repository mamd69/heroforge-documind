"""
DocuMind RAG (Retrieval-Augmented Generation) Module.

This module provides semantic search and document retrieval capabilities
for the DocuMind knowledge management system.
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
