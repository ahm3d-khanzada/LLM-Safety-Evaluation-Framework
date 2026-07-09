"""
models.anthropic_model
========================

Concrete `BaseModel` implementation backed by LangChain's `ChatAnthropic`.
Used primarily for Claude models (e.g., Claude 3.5 Sonnet as a judge).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from models.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)


class AnthropicChatModel(BaseModel):
    """BaseModel implementation using LangChain's ChatAnthropic client."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._llm = self._build_llm(config)

    def _build_llm(self, config: ModelConfig) -> ChatAnthropic:
        if ChatAnthropic is None:
            raise ImportError(
                "langchain-anthropic is not installed. Please install it to "
                "use Claude models: pip install langchain-anthropic"
            )
        api_key = self._resolve_api_key(config)

        kwargs = dict(
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
            anthropic_api_key=api_key,
        )

        logger.info(
            "Initializing ChatAnthropic for model '%s' (model_id=%s)",
            config.name, config.model_id
        )
        return ChatAnthropic(**kwargs)

    def _resolve_api_key(self, config: ModelConfig) -> str:
        if not config.api_key_env:
            raise ValueError(
                f"Model '{config.name}' has type='{config.type}' but no "
                f"'api_key_env' set in config.yaml."
            )
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set."
            )
        return api_key

    def _call_raw(self, prompt: str, system_prompt: Optional[str] = None) -> tuple[str, dict]:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = self._llm.invoke(messages)
        text = result.content if hasattr(result, "content") else str(result)
        
        metadata = getattr(result, "response_metadata", {}).copy()
        if hasattr(result, "additional_kwargs"):
            metadata.update(result.additional_kwargs)
            
        if not text or not text.strip():
            raise ValueError(f"Empty response from Anthropic model_id='{self.config.model_id}'")
            
        return text, metadata
