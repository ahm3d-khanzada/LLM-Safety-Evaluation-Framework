"""
utils.config_loader
=====================

Tiny helper to load and validate config.yaml from anywhere in the
project (scripts, notebooks, tests) without duplicating path logic.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict

import yaml

# Suppress LangChain/LangGraph deprecation warnings that clutter logs during import
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
try:
    # This specific warning comes from langgraph.cache.base
    from langchain_core._api import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover - dotenv is an optional convenience
    _DOTENV_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_file(env_path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Load environment variables from a local .env file, if present.

    This is optional -- you can also just `export OPENAI_API_KEY=...` in
    your shell / ~/.bashrc. The .env approach is convenient if you prefer
    keeping secrets in a single gitignored file instead of shell config.
    Requires `pip install python-dotenv` (already in requirements.txt).
    """
    env_path = Path(env_path)
    if not _DOTENV_AVAILABLE:
        logging.getLogger(__name__).debug(
            "python-dotenv not installed; skipping .env loading (fine if you "
            "export env vars via your shell instead)."
        )
        return
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logging.getLogger(__name__).info("Loaded environment variables from %s", env_path)


def load_config(config_path: str | Path = PROJECT_ROOT / "config.yaml") -> Dict:
    """Load config.yaml into a plain dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict) -> None:
    """Configure root logging based on the `logging:` section of config.yaml."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = []
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    if log_cfg.get("console", True):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        handlers.append(console)

    if log_cfg.get("file", True):
        log_file = PROJECT_ROOT / config.get("paths", {}).get("log_file", "logs/eval_run.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
