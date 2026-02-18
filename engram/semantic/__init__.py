"""engram.semantic — File-backed semantic memory (YAML + Markdown)."""

from engram.semantic.store import SemanticStore
from engram.semantic.identity import IdentityResolver

__all__ = ["SemanticStore", "IdentityResolver"]
