"""
engram.signal — Code Quality Signal measurement, extraction, and adaptation.

Public API:
  measure()           — measure code quality signal (regex + optional LLM)
  measure_regex()     — regex-only CQS measurement
  extract()           — LLM-based semantic extraction
  SignalTracker       — rolling window signal analytics
  ReinforcementEngine — Hebbian salience adjustment
  DecayEngine         — adaptive memory decay
"""

from engram.signal.measure import (
    measure,
    measure_regex,
    check_correctness,
    check_consistency,
    check_completeness,
    check_robustness,
    blend_signals,
    parse_llm_signal,
    SignalTracker,
)
from engram.signal.extract import extract, parse_extraction
from engram.signal.reinforcement import ReinforcementEngine
from engram.signal.decay import DecayEngine

__all__ = [
    "measure",
    "measure_regex",
    "check_correctness",
    "check_consistency",
    "check_completeness",
    "check_robustness",
    "blend_signals",
    "parse_llm_signal",
    "extract",
    "parse_extraction",
    "SignalTracker",
    "ReinforcementEngine",
    "DecayEngine",
]
