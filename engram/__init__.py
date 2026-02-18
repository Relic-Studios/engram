"""
Engram -- Four-layer memory system for persistent AI identity.

    from engram import MemorySystem

    memory = MemorySystem(data_dir="./data")
    context = memory.before(person="alice", message="hello", token_budget=6000)
    result = memory.after(person="alice", their_message="hello",
                          response="Hi Alice!", trace_ids=context.trace_ids)
"""

from engram.core.types import (
    Trace,
    Message,
    Signal,
    Context,
    AfterResult,
    MemoryStats,
)
from engram.core.config import Config
from engram.system import MemorySystem

__version__ = "0.2.0"

__all__ = [
    "MemorySystem",
    "Config",
    "Trace",
    "Message",
    "Signal",
    "Context",
    "AfterResult",
    "MemoryStats",
]
