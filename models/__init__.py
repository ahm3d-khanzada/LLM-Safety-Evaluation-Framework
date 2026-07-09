"""
models package
===============

Model interface layer for the LLM Safety Evaluation Framework.

Provides a common `BaseModel` interface so any LLM backend -- hosted
OpenAI API, a local LM Studio server, or a deterministic offline
`MockModel` -- can be evaluated through the exact same pipeline.

Public API:
    - BaseModel, ModelConfig, ModelResponse   (base_model.py)
    - LangChainChatModel                       (langchain_model.py — OpenAI & LM Studio)
    - MockModel                                (mock_model.py — offline testing)
    - build_model, build_models_from_config    (model_factory.py)
"""

from models.base_model import BaseModel, ModelConfig, ModelResponse
from models.langchain_model import LangChainChatModel
from models.mock_model import MockModel
from models.model_factory import build_model, build_models_from_config

__all__ = [
    "BaseModel",
    "ModelConfig",
    "ModelResponse",
    "LangChainChatModel",
    "MockModel",
    "build_model",
    "build_models_from_config",
]
