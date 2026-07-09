"""
models.base_model
==================

Defines the common interface every model backend (OpenAI, LM Studio /
local models, future backends) implements, plus a **LangGraph-powered**
robust invocation wrapper that adds retry-with-backoff and structured
error capture around every model call.

Design rationale
-----------------
Every safety-eval prompt eventually becomes a single call: "send this
prompt to this model, get text back". That call can fail for many
reasons (network blip, local server not warmed up, rate limiting,
empty/garbage completion). Rather than scattering try/except blocks
through the evaluation pipeline, we encode the retry policy *once* as
a small LangGraph state machine:

    call_model --> validate_response --(fail, retries left)--> call_model
                                      (else)--> (success OR retries exhausted) --> END

This keeps `evaluator/pipeline.py` (built later) simple: it just calls
`model.generate(prompt)` and always gets back a well-formed
`ModelResponse`, whether the underlying call succeeded, failed, or was
retried.

Subclasses only need to implement `_call_raw(prompt, system_prompt)`,
which does the actual LangChain invocation. Everything else (timing,
retries, logging, response normalization) is handled here.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Tuple

from langgraph.graph import StateGraph, END

# Suppress LangChain/LangGraph deprecation warnings that clutter logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
try:
    from langchain_core._api import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model under evaluation.

    Mirrors one entry under `models:` in config.yaml.
    """

    name: str
    display_name: str
    type: str  # "openai" | "lmstudio" | "mock"
    model_id: str = ""
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 2
    request_timeout: float = 60.0
    persona: Optional[str] = None  # used only by MockModel ("baseline" | "hardened")


@dataclass
class ModelResponse:
    """Normalized result of asking a model to respond to one prompt."""

    model_name: str
    prompt_id: str
    prompt_text: str
    response_text: str = ""
    success: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0
    attempts: int = 0
    raw_metadata: dict = field(default_factory=dict)


class _GraphState(TypedDict, total=False):
    """Internal state threaded through the LangGraph retry graph."""

    prompt: str
    system_prompt: Optional[str]
    response_text: str
    success: bool
    error: Optional[str]
    attempts: int
    max_retries: int
    raw_metadata: dict


class BaseModel(ABC):
    """Abstract base class for every model backend used in the framework.

    Concrete subclasses (e.g. `LangChainChatModel`) implement `_call_raw`.
    This base class provides `generate()`, the single method the rest of
    the framework calls, which wraps `_call_raw` in a LangGraph retry loop.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._graph = self._build_retry_graph()

    # ------------------------------------------------------------------
    # Subclass responsibility
    # ------------------------------------------------------------------
    @abstractmethod
    def _call_raw(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, dict]:
        """Perform the actual model call and return (text, metadata).

        Must raise an exception on failure (network error, API error,
        empty response, etc.) — the retry graph catches and handles it.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # LangGraph retry graph
    # ------------------------------------------------------------------
    def _build_retry_graph(self):
        """Builds a small LangGraph state machine implementing:

            call_model -> validate -> (retry via call_model) | END

        Using a graph (rather than a plain while-loop) makes the retry
        policy declarative and easy to extend later -- e.g. adding a
        "fallback_model" node if the primary model keeps failing.
        """

        def call_model_node(state: _GraphState) -> _GraphState:
            attempt = state.get("attempts", 0) + 1
            try:
                text, metadata = self._call_raw(state["prompt"], state.get("system_prompt"))
                if not text or not text.strip():
                    raise ValueError("Model returned an empty response.")
                return {**state, "response_text": text, "success": True,
                        "error": None, "attempts": attempt, "raw_metadata": metadata}
            except Exception as exc:  # noqa: BLE001 - we want to catch any backend error
                logger.warning(
                    "Model '%s' call failed on attempt %d: %s",
                    self.config.name, attempt, exc,
                )
                return {**state, "success": False, "error": str(exc), "attempts": attempt}

        def should_retry(state: _GraphState) -> str:
            if state.get("success"):
                return "end"
            if state.get("attempts", 0) < state.get("max_retries", 0) + 1:
                # brief backoff before retrying
                time.sleep(min(2 ** state.get("attempts", 0), 8))
                return "retry"
            return "end"

        graph = StateGraph(_GraphState)
        graph.add_node("call_model", call_model_node)
        graph.set_entry_point("call_model")
        graph.add_conditional_edges(
            "call_model", should_retry, {"retry": "call_model", "end": END}
        )
        return graph.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        prompt_id: str = "",
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """Send `prompt` to this model and return a normalized ModelResponse.

        Handles retries (via the LangGraph state machine above) and
        always returns a well-formed ModelResponse, even on failure.
        """
        start = time.perf_counter()
        initial_state: _GraphState = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "attempts": 0,
            "max_retries": self.config.max_retries,
        }
        final_state = self._graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            model_name=self.config.name,
            prompt_id=prompt_id,
            prompt_text=prompt,
            response_text=final_state.get("response_text", ""),
            success=final_state.get("success", False),
            error=final_state.get("error"),
            latency_ms=round(latency_ms, 2),
            attempts=final_state.get("attempts", 0),
            raw_metadata=final_state.get("raw_metadata", {}),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.config.name!r} type={self.config.type!r}>"
