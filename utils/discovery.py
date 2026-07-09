"""
utils.discovery
===============

Utilities for dynamically discovering local models from LM Studio servers.
"""

import requests
import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def discover_lmstudio_models(server_cfg: Dict) -> List[Dict]:
    """Queries an LM Studio server for available chat models.
    
    Excludes models that look like embedding models.
    """
    base_url = server_cfg.get("base_url")
    if not base_url:
        logger.warning("No base_url provided for LM Studio discovery.")
        return []
        
    exclude_patterns = server_cfg.get("exclude_patterns", ["embed"])
    api_key_env = server_cfg.get("api_key_env")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # LM Studio prefers /api/v1/models over /v1/models for listing models.
        # We try to construct the correct discovery URL based on the provided base_url.
        normalized_base = base_url.rstrip("/")
        
        if normalized_base.endswith("/v1"):
            discovery_url = normalized_base.replace("/v1", "/api/v1") + "/models"
        elif "/v1" not in normalized_base and "/api" not in normalized_base:
            # If no versioning in URL, try the LM Studio preferred internal API path
            discovery_url = f"{normalized_base}/api/v1/models"
        else:
            discovery_url = f"{normalized_base}/models"
            
        logger.debug("Attempting model discovery at %s", discovery_url)
        response = requests.get(discovery_url, headers=headers, timeout=5.0)
        
        # Fallback to standard OpenAI-compatible models endpoint if preferred one failed
        # or if we didn't try the standard one yet.
        standard_url = f"{normalized_base}/models" if not normalized_base.endswith("/v1") else f"{normalized_base}/models"
        
        if response.status_code != 200 and discovery_url != standard_url:
            logger.debug("Discovery at %s failed (status %d), falling back to %s", 
                         discovery_url, response.status_code, standard_url)
            response = requests.get(standard_url, headers=headers, timeout=5.0)
            
        response.raise_for_status()
        data = response.json()
        
        # LM Studio returns a list of models in data['data']
        models_data = data.get("data", [])
        discovered = []
        
        for m in models_data:
            model_id = m.get("id")
            if not model_id:
                continue
                
            # Filter out excluded patterns
            if any(p in model_id.lower() for p in exclude_patterns):
                continue
                
            # Build a model config entry
            # Use real model_id as both name and model_id to satisfy requirements
            discovered.append({
                "name": model_id,
                "display_name": model_id,
                "type": "lmstudio",
                "model_id": model_id,
                "base_url": base_url,
                "api_key_env": server_cfg.get("api_key_env"),
                "temperature": server_cfg.get("temperature", 0.0),
                "max_tokens": server_cfg.get("max_tokens", 2048),
                "request_timeout": server_cfg.get("request_timeout", 180.0),
                "quantization": "Unknown (Auto-discovered)"
            })
            
        logger.info("Discovered %d models from %s", len(discovered), base_url)
        return discovered
        
    except Exception as e:
        logger.warning("Failed to discover models from LM Studio at %s: %s", base_url, e)
        return []

def normalize_url(url: str) -> str:
    """Normalizes a URL by stripping trailing slashes and converting to lowercase."""
    if not url: return ""
    return url.rstrip("/").lower()

def get_all_models(full_config: Dict, discover_local: bool = False) -> List[Dict]:
    """Combines explicit models from config with dynamically discovered local models."""
    models = full_config.get("models", []).copy()
    
    if discover_local:
        servers = full_config.get("local_lmstudio_servers", [])
        for server in servers:
            discovered = discover_lmstudio_models(server)
            for d in discovered:
                # Avoid duplicates if model already explicitly in config (check ID and URL)
                d_url = normalize_url(d.get("base_url"))
                if not any(m["model_id"] == d["model_id"] and normalize_url(m.get("base_url")) == d_url for m in models):
                    models.append(d)
            
    return models
