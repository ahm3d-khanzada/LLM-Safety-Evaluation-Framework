"""
models.langchain_model
========================

Concrete `BaseModel` implementation backed by LangChain's `ChatOpenAI`.

Why one class covers both OpenAI *and* LM Studio
-------------------------------------------------
LM Studio's local server implements the OpenAI REST API spec
(`/v1/chat/completions`) on `http://localhost:1234/v1` by default. This
means the *exact same* `ChatOpenAI` client works for:

  * Real OpenAI models   -> base_url = "https://api.openai.com/v1" (default),
                             api_key = your real OPENAI_API_KEY.
  * Local LM Studio model -> base_url = "http://localhost:1234/v1",
                              api_key = any non-empty placeholder string
                              (LM Studio does not validate it).

So instead of writing two separate backends, `LangChainChatModel` just
reads `base_url` / `api_key_env` from config.yaml and instantiates
`ChatOpenAI` accordingly. Fewer moving parts, less duplicated logic to
keep in sync.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models.base_model import BaseModel, ModelConfig

logger = logging.getLogger(__name__)

# For local LM Studio servers, we default to an empty string if no key 
# is provided. This often allows the server to skip authentication checks
# if they are not required, avoiding "malformed token" errors.
_LOCAL_DUMMY_API_KEY = ""


class LangChainChatModel(BaseModel):
    """BaseModel implementation using LangChain's ChatOpenAI client.

    Works for:
        - type: "openai"   -> hosted OpenAI API
        - type: "lmstudio" -> local LM Studio server (OpenAI-compatible)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._llm = self._build_llm(config)

    def _build_llm(self, config: ModelConfig) -> ChatOpenAI:
        api_key = self._resolve_api_key(config)

        kwargs = dict(
            model=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
            api_key=api_key,
        )
        # Only pass base_url when explicitly set (None lets ChatOpenAI use
        # the real OpenAI endpoint by default).
        if config.base_url:
            kwargs["base_url"] = config.base_url

        logger.info(
            "Initializing ChatOpenAI for model '%s' (type=%s, model_id=%s, base_url=%s)",
            config.name, config.type, config.model_id, config.base_url or "default (OpenAI)",
        )
        return ChatOpenAI(**kwargs)

    def _resolve_api_key(self, config: ModelConfig) -> str:
        """Resolve the API key from an environment variable.

        For local LM Studio servers, no real key is needed, so we fall
        back to a placeholder if `api_key_env` is unset or the env var
        is empty -- this avoids forcing users to set a fake env var just
        to run a local model.
        """
        if config.type == "lmstudio":
            if config.api_key_env and os.environ.get(config.api_key_env):
                return os.environ[config.api_key_env]
            return _LOCAL_DUMMY_API_KEY

        # For real hosted APIs (OpenAI, etc.) the key is mandatory.
        if not config.api_key_env:
            raise ValueError(
                f"Model '{config.name}' has type='{config.type}' but no "
                f"'api_key_env' set in config.yaml."
            )
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{config.api_key_env}' is not set. "
                f"Export it before running, e.g.:\n"
                f"    export {config.api_key_env}=sk-...\n"
                f"(required for model '{config.name}')"
            )
        return api_key

    def _call_raw(self, prompt: str, system_prompt: Optional[str] = None) -> tuple[str, dict]:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        result = self._llm.invoke(messages)
        text = result.content if hasattr(result, "content") else str(result)
        
        # Capture metadata for AI safety analysis (e.g. reasoning tokens, finish reason)
        metadata = getattr(result, "response_metadata", {}).copy()
        if hasattr(result, "additional_kwargs"):
            metadata.update(result.additional_kwargs)
            
        # Ensure common keys are present for downstream evaluators
        if "finish_reason" not in metadata and hasattr(result, "response_metadata"):
            metadata["finish_reason"] = result.response_metadata.get("finish_reason")

        if not text or not text.strip():
            # If we have reasoning content but no visible content, it's still
            # technically a failure to provide a response, but we record the reasoning.
            err_msg = (
                f"Empty response from model_id='{self.config.model_id}' at "
                f"base_url='{self.config.base_url or 'https://api.openai.com/v1'}'. "
            )
            if metadata.get("reasoning_content"):
                err_msg += " Model provided reasoning_content but no visible answer."
            
            raise ValueError(err_msg)
            
        return text, metadata
