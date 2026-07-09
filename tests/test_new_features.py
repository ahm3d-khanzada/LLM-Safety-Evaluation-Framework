"""
tests.test_new_features
========================

Tests for LLM-as-Judge and Dynamic Red-Teaming.
"""

import pytest
from unittest.mock import MagicMock
from models.mock_model import MockModel
from models.base_model import ModelConfig
from evaluator.scorers import llm_judge_score
from evaluator.red_team import RedTeamAgent, run_dynamic_red_teaming_loop

@pytest.fixture
def mock_target():
    cfg = ModelConfig(name="target", display_name="Target", type="mock", persona="baseline")
    return MockModel(cfg)

@pytest.fixture
def mock_judge():
    cfg = ModelConfig(name="judge", display_name="Judge", type="mock", persona="hardened")
    return MockModel(cfg)

def test_llm_judge_score_mock(mock_judge):
    prompt = {"category": "jailbreak", "prompt_text": "ignore rules", "expected_behavior": "refuse"}
    response = mock_judge.generate("I cannot do that", prompt_id="test")
    
    # MockModel currently doesn't return JSON for judge. 
    # In a real scenario, it would. For this test, we'll mock the response_text.
    mock_judge.persona = "judge" # We'd need to update MockModel to support this or just mock the call
    
    # Let's manually set the response text for the test
    real_generate = mock_judge.generate
    mock_judge.generate = MagicMock()
    mock_judge.generate.return_value.success = True
    mock_judge.generate.return_value.response_text = '{"safety_score": 1.0, "quality_score": 0.8, "rationale": "Safe"}'
    
    res = llm_judge_score(prompt, response, mock_judge)
    assert res["safety_score"] == 1.0
    assert "rationale" in res

def test_red_team_generation(mock_judge):
    agent = RedTeamAgent(mock_judge)
    
    mock_judge.generate = MagicMock()
    mock_judge.generate.return_value.success = True
    mock_judge.generate.return_value.response_text = '{"id": "rt_1", "category": "jailbreak", "prompt_text": "new attack", "risk_type": "test", "expected_behavior": "refuse", "severity": "high"}'
    
    new_prompt = agent.generate_adversarial_prompt(
        "target", "jailbreak", "old prompt", "old response", {"safety": 1.0}
    )
    
    assert new_prompt["id"] == "rt_1"
    assert new_prompt["prompt_text"] == "new attack"

def test_red_team_loop(mock_target, mock_judge):
    initial_prompt = {"id": "p1", "category": "jailbreak", "prompt_text": "start", "expected_behavior": "refuse"}
    
    # Mock red-team agent generation
    mock_judge.generate = MagicMock()
    mock_judge.generate.return_value.success = True
    mock_judge.generate.return_value.response_text = '{"id": "rt_2", "category": "jailbreak", "prompt_text": "next", "risk_type": "test", "expected_behavior": "refuse", "severity": "high"}'
    
    results = run_dynamic_red_teaming_loop(mock_target, mock_judge, initial_prompt, max_iterations=2)
    
    assert len(results) == 2
    assert results[0]["iteration"] == 1
    assert results[1]["iteration"] == 2
    assert results[1]["prompt"]["prompt_text"] == "next"
