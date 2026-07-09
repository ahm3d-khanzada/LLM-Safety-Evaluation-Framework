import pandas as pd
import pytest
from evaluator.aggregator import ResultsAggregator

def test_overall_safety_score_math():
    agg = ResultsAggregator()
    # Mock some data
    data = [
        {"model_name": "m1", "category": "jailbreak", "jailbreak_resistance": 1.0, "success": True},
        {"model_name": "m1", "category": "harmful_requests", "safe_unsafe": 1.0, "success": True},
        {"model_name": "m1", "category": "prompt_injection", "prompt_injection_resistance": 0.0, "success": True},
    ]
    df = pd.DataFrame(data)
    
    # We only have 3 categories out of 10. The others will default to 0.
    # Weights: jailbreak=0.14, harmful=0.14, prompt_injection=0.12
    # Expected: (1.0*0.14 + 1.0*0.14 + 0.0*0.12) * 100 = 28.0
    
    results = agg.aggregate(df)
    overall = results["model_overall"]
    m1_score = overall[overall["model_name"] == "m1"]["overall_safety_score"].values[0]
    
    assert m1_score == pytest.approx(28.0)
