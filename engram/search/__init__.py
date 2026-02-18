"""engram.search — FTS5 indexed search, ChromaDB semantic search, unified search, code-aware tokenization, and dual embeddings."""

from engram.search.indexed import IndexedSearch
from engram.search.unified import UnifiedSearch
from engram.search.tokenizer import (
    split_identifier,
    expand_text,
    expand_query,
    is_compound_identifier,
)
from engram.search.code_embeddings import (
    CodeEmbedder,
    is_code_content,
    build_code_embedding_func,
    CODE_TRACE_KINDS,
)

__all__ = [
    "IndexedSearch",
    "UnifiedSearch",
    "split_identifier",
    "expand_text",
    "expand_query",
    "is_compound_identifier",
    "CodeEmbedder",
    "is_code_content",
    "build_code_embedding_func",
    "CODE_TRACE_KINDS",
]
