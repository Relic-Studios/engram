"""engram.working — Token-budget-aware working memory and context assembly."""

from engram.working.context import ContextBuilder
from engram.working.allocator import knapsack_allocate, compress_text, fit_messages

__all__ = ["ContextBuilder", "knapsack_allocate", "compress_text", "fit_messages"]
