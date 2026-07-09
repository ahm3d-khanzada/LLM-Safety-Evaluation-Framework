"""
evaluator
=========

Core evaluation logic: pipeline orchestration, scoring heuristics,
aggregation, and visualization.
"""

__all__ = [
    "EvaluationPipeline",
    "ResultsAggregator",
    "run_visualizations",
    "plot_overall_safety_score",
    "plot_hallucination_rate",
    "plot_injection_success_rate",
    "plot_jailbreak_success_rate",
    "plot_latency_comparison",
    "plot_category_heatmap",
    "RedTeamAgent",
    "run_dynamic_red_teaming_loop",
    "EvalDB",
    "SCORER_REGISTRY",
    "score_safe_unsafe",
    "score_refusal_quality",
    "score_helpfulness",
    "score_truthfulness",
    "score_prompt_injection",
    "score_jailbreak",
    "score_privacy",
    "score_bias",
    "score_instruction_following",
    "llm_judge_score",
]

def __getattr__(name):
    """Lazy-load modules to avoid RuntimeWarning when running scripts via -m."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "EvaluationPipeline":
        from evaluator.pipeline import EvaluationPipeline
        return EvaluationPipeline
    if name == "ResultsAggregator":
        from evaluator.aggregator import ResultsAggregator
        return ResultsAggregator
    if name == "run_visualizations" or name.startswith("plot_"):
        from evaluator.visualize import run_visualizations, plot_overall_safety_score, plot_hallucination_rate, plot_injection_success_rate, plot_jailbreak_success_rate, plot_latency_comparison, plot_category_heatmap
        return locals()[name]
    if name == "RedTeamAgent" or name == "run_dynamic_red_teaming_loop":
        from evaluator.red_team import RedTeamAgent, run_dynamic_red_teaming_loop
        return locals()[name]
    if name == "EvalDB":
        from evaluator.db import EvalDB
        return EvalDB
    
    # All other exports are from scorers.py
    import evaluator.scorers as scorers
    return getattr(scorers, name)
