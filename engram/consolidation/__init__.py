"""
engram.consolidation — Memory pressure management, compaction, and
hierarchical consolidation.

Inspired by:
  - MemGPT's virtual-context paging and memory pressure warnings
  - NotebookLM's auto-generated Source Guides
  - Neuroscience: hippocampal replay during sleep
"""

from engram.consolidation.pressure import MemoryPressure, PressureState
from engram.consolidation.compactor import ConversationCompactor, CompactionResult
from engram.consolidation.consolidator import MemoryConsolidator

__all__ = [
    "MemoryPressure",
    "PressureState",
    "ConversationCompactor",
    "CompactionResult",
    "MemoryConsolidator",
]
