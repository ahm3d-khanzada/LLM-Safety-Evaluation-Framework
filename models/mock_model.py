"""
models.mock_model
===================

A deterministic, dependency-free model used for:
  * Unit tests (tests/) that must run in CI without API keys or a
    running LM Studio server.
  * Quickly sanity-checking the evaluation pipeline end-to-end before
    spending real API credits or GPU time.

Two personas are supported via `ModelConfig.persona` (set in config.yaml
or passed directly):
  * "baseline" -> naive model that frequently complies with unsafe
                   requests, useful as a deliberately weak comparison point.
  * "hardened" -> model that consistently refuses unsafe requests,
                   useful as a deliberately strong comparison point.

This is NOT meant to simulate a real LLM's language quality -- it exists
purely so the scoring/aggregation/visualization code has predictable,
reproducible input to run against.
"""

from __future__ import annotations

from typing import Optional

from models.base_model import BaseModel, ModelConfig

_REFUSAL_TEXT = (
    "I can't help with that request as it could facilitate harm. "
    "If you're working on something legitimate (research, security "
    "testing, safety evaluation), I'm happy to help within safe bounds."
)

_UNSAFE_TRIGGER_KEYWORDS = (
    "ignore", "bypass", "jailbreak", "dan", "hack", "steal",
    "social security number", "ssn", "credit card", "bomb",
    "malicious", "exploit", "unfiltered", "no filters"
)


class MockModel(BaseModel):
    """Offline deterministic model for testing and demos."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.persona = getattr(config, "persona", "baseline")

    def _call_raw(self, prompt: str, system_prompt: Optional[str] = None) -> tuple[str, dict]:
        lower = prompt.lower()
        looks_unsafe = any(kw in lower for kw in _UNSAFE_TRIGGER_KEYWORDS)

        if self.persona == "hardened":
            if looks_unsafe:
                return _REFUSAL_TEXT, {"mock": True, "refused": True}
            return self._helpful_answer(prompt), {"mock": True, "refused": False}

        # "baseline" persona: complies more often, and occasionally
        # fabricates a confident-sounding but ungrounded answer
        # (used to exercise the hallucination scorer).
        if looks_unsafe:
            return (
                "Sure, here's how you could approach that: [baseline model "
                "complies with the unsafe request in a simplified way].",
                {"mock": True, "refused": False}
            )
        return self._helpful_answer(prompt, confident_but_unverified=True), {"mock": True, "refused": False}

    @staticmethod
    def _helpful_answer(prompt: str, confident_but_unverified: bool = False) -> str:
        if confident_but_unverified:
            return (
                f"Here is a direct, confident answer to your question: "
                f"'{prompt[:60]}...' — (note: this mock response is not "
                f"grounded in any real knowledge base)."
            )
        return (
            f"Here's a helpful, careful response to: '{prompt[:60]}...' "
            f"I'd note any relevant caveats or uncertainty where applicable."
        )
