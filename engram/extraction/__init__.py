"""
engram.extraction — AST-based code analysis, symbol extraction, and error fingerprinting.

Provides structural code analysis using tree-sitter (multi-language)
and Python's stdlib ast module (Python-specific deep analysis), plus
Sentry-style error fingerprinting for deduplication and recall.

Public API:
  extract_symbols()      — extract functions, classes, imports from code
  extract_complexity()   — cyclomatic complexity and nesting depth
  extract_dependencies() — import graph and call relationships
  detect_language()      — auto-detect language from code content
  SymbolIndex            — persistent symbol index for repo-map generation
  CodeAnalysis           — structured result from full code analysis
  compute_fingerprint()  — SHA-256 error fingerprint from message + trace
  analyze_error()        — full error analysis with fingerprint + components
"""

from engram.extraction.ast_engine import (
    extract_symbols,
    extract_complexity,
    extract_dependencies,
    detect_language,
    CodeAnalysis,
    Symbol,
    ImportInfo,
    ComplexityMetrics,
    analyze_code,
)
from engram.extraction.fingerprint import (
    analyze_error,
    compute_fingerprint,
    extract_exception_type,
    extract_frames,
    normalize_message,
)
from engram.extraction.symbol_index import SymbolIndex

__all__ = [
    # AST analysis
    "extract_symbols",
    "extract_complexity",
    "extract_dependencies",
    "detect_language",
    "analyze_code",
    "CodeAnalysis",
    "Symbol",
    "ImportInfo",
    "ComplexityMetrics",
    "SymbolIndex",
    # Error fingerprinting
    "compute_fingerprint",
    "analyze_error",
    "extract_exception_type",
    "extract_frames",
    "normalize_message",
]
