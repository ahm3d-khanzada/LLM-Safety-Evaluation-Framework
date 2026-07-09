from utils.config_loader import load_config

def test_config_validity():
    config = load_config()
    assert "models" in config
    assert "evaluation" in config
    assert "paths" in config
    
    # Check weights sum to 1.0
    weights = config["evaluation"]["category_weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    
    # All categories covered
    categories = config["evaluation"]["categories"]
    assert set(categories) == set(weights.keys())

import pytest
