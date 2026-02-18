"""
engram.runtime.mind — Local inference stub.

Scaffold for future local model integration.
Currently a placeholder that raises NotImplementedError.
"""

from __future__ import annotations

from typing import Optional


class LocalMind:
    """
    Stub for local inference (e.g. llama.cpp, Ollama direct).

    In the future this would load a local model and provide
    ``generate(prompt, system) -> str``.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.loaded = False

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError(
            "LocalMind is a scaffold — local inference is not yet implemented. "
            "Use Config.get_llm_func() for remote LLM calls."
        )
