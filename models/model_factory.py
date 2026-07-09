"""
models.model_factory
======================

Reads a model entry from config.yaml and returns the correct `BaseModel`
subclass instance. This is the *only* place that needs to know about
every backend type, keeping the rest of the framework (pipeline,
scorers, notebook) backend-agnostic -- they only ever see `BaseModel`.

Adding a new backend later (e.g. a raw Hugging Face Transformers
pipeline, or Ollama) means: (1) add a subclass of BaseModel, and
(2) add one `elif` branch here. Nothing else changes.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from models.base_model import BaseModel, ModelConfig
from models.langchain_model import LangChainChatModel
from models.anthropic_model import AnthropicChatModel
from models.mock_model import MockModel

logger = logging.getLogger(__name__)

_SUPPORTED_TYPES = {"openai", "lmstudio", "anthropic", "mock"}


def build_model(model_cfg: Dict) -> BaseModel:
    """Build a single BaseModel instance from a config.yaml `models:` entry.

    Parameters
    ----------
    model_cfg:
        A dict matching one item under `models:` in config.yaml, e.g.:
            {"name": "gpt-4o-mini", "display_name": "GPT-4o-mini",
             "type": "openai", "model_id": "gpt-4o-mini",
             "api_key_env": "OPENAI_API_KEY", "temperature": 0.0,
             "max_tokens": 512}

    Returns
    -------
    BaseModel
        A ready-to-use model wrapper (`.generate(prompt)` works).

    Raises
    ------
    ValueError
        If `type` is not one of the supported backends, or required
        fields are missing.
    """
    model_type = model_cfg.get("type")
    if model_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported model type '{model_type}' for model "
            f"'{model_cfg.get('name')}'. Supported types: {sorted(_SUPPORTED_TYPES)}"
        )

    config = ModelConfig(
        name=model_cfg["name"],
        display_name=model_cfg.get("display_name", model_cfg["name"]),
        type=model_type,
        model_id=model_cfg.get("model_id", ""),
        api_key_env=model_cfg.get("api_key_env"),
        base_url=model_cfg.get("base_url"),
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 512),
        max_retries=model_cfg.get("max_retries", 2),
        request_timeout=model_cfg.get("request_timeout", 60.0),
        persona=model_cfg.get("persona"),
    )

    if model_type in ("openai", "lmstudio"):
        logger.info("Building LangChainChatModel for '%s' (%s)", config.name, model_type)
        return LangChainChatModel(config)

    if model_type == "anthropic":
        logger.info("Building AnthropicChatModel for '%s'", config.name)
        return AnthropicChatModel(config)

    if model_type == "mock":
        logger.info("Building MockModel for '%s' (persona=%s)", config.name, config.persona)
        return MockModel(config)

    # Unreachable given the check above, but keeps mypy/readers happy.
    raise ValueError(f"Unhandled model type: {model_type}")


def build_models_from_config(full_config: Dict) -> List[BaseModel]:
    """Build every model listed under `models:` in the loaded config.yaml dict.

    Skips (with a warning) any model entry that fails to build -- e.g. a
    missing API key for one model shouldn't prevent evaluating the others.
    """
    models: List[BaseModel] = []
    for model_cfg in full_config.get("models", []):
        try:
            models.append(build_model(model_cfg))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Skipping model '%s' -- failed to initialize: %s",
                model_cfg.get("name", "<unnamed>"), exc,
            )
    if not models:
        raise RuntimeError(
            "No models could be initialized. Check config.yaml and your "
            "environment variables (API keys) / LM Studio server status."
        )
    return models
