"""
engram.extraction — AST-based code analysis and symbol extraction.

Provides structural code analysis using tree-sitter (multi-language)
and Python's stdlib ast module (Python-specific deep analysis).

Public API:
  extract_symbols()   — extract functions, classes, imports from code
  extract_complexity() — cyclomatic complexity and nesting depth
  extract_dependencies() — import graph and call relationships
  detect_language()    — auto-detect language from code content
  SymbolIndex          — persistent symbol index for repo-map generation
  CodeAnalysis         — structured result from full code analysis
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
from engram.extraction.symbol_index import SymbolIndex

__all__ = [
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
]
