"""
evaluator.pipeline
==================

LangGraph-powered orchestration of the evaluation process:
loading prompts, calling models, scoring responses, and saving results.
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional

from tqdm import tqdm
from langgraph.graph import StateGraph, END

# Suppress LangChain/LangGraph deprecation warnings that clutter logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
try:
    from langchain_core._api import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

from models.base_model import BaseModel, ModelResponse
import argparse
from datetime import datetime

from evaluator.db import EvalDB
from utils.discovery import get_all_models
from models.model_factory import build_model
from utils.config_loader import load_config, setup_logging, PROJECT_ROOT
from evaluator.scorers import SCORER_REGISTRY, llm_judge_score

logger = logging.getLogger(__name__)

class PipelineState(TypedDict):
    """State for the EvaluationPipeline graph."""
    config: Dict
    models: List[BaseModel]
    prompts: List[Dict]
    judge_model: Optional[BaseModel]
    run_id: str
    processed_count: int
    progress_callback: Optional[Any]

class EvaluationPipeline:
    """Orchestrates the LLM safety evaluation run."""

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None, discover_local: bool = False):
        if config:
            self.config = config
        elif config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config()
            
        setup_logging(self.config)
        self.discover_local = discover_local
        
        self.paths = self.config["paths"]
        self.raw_dir = PROJECT_ROOT / self.paths["raw_responses_dir"]
        self.processed_dir = PROJECT_ROOT / self.paths["processed_scores_dir"]
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.db = EvalDB(PROJECT_ROOT / "data/eval_results.db")
        
        self._graph = self._build_pipeline_graph()

    def _build_pipeline_graph(self) -> StateGraph:
        """Builds a LangGraph to orchestrate the evaluation."""
        
        def init_run_node(state: PipelineState) -> PipelineState:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info("Starting evaluation run: %s", run_id)
            return {**state, "run_id": run_id}

        def load_prompts_node(state: PipelineState) -> PipelineState:
            prompts_file = PROJECT_ROOT / self.paths["prompts_file"]
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            logger.info("Loaded %d prompts from %s", len(prompts), prompts_file)
            return {**state, "prompts": prompts}

        def build_models_node(state: PipelineState) -> PipelineState:
            model_configs = get_all_models(self.config, discover_local=self.discover_local)
            
            # Save discovered models to run record
            discovered_names = [mc["name"] for mc in model_configs]
            self.db.save_run(state["run_id"], self.config, discovered_names)
            
            # Also save to the requested JSON cache file
            cache_file = PROJECT_ROOT / "data/discovered_local_models.json"
            with open(cache_file, "w") as f:
                json.dump(model_configs, f, indent=2)

            models = []
            for mc in model_configs:
                try:
                    models.append(build_model(mc))
                except Exception as e:
                    logger.error("Failed to build model %s: %s", mc["name"], e)
            
            logger.info("Built %d models for evaluation", len(models))
            
            judge_model = None
            if self.config["evaluation"].get("use_llm_judge"):
                judge_cfg = {
                    "name": "judge",
                    "type": self.config["evaluation"].get("judge_model_type", "openai"),
                    "model_id": self.config["evaluation"]["judge_model_id"],
                    "api_key_env": self.config["evaluation"]["judge_model_env"],
                }
                # Try to build judge
                try:
                    judge_model = build_model(judge_cfg)
                    logger.info("Built judge model: %s", judge_cfg["model_id"])
                except Exception as e:
                    logger.warning("Failed to build judge model: %s. Proceeding without judge.", e)
                    
            return {**state, "models": models, "judge_model": judge_model}

        def run_eval_node(state: PipelineState) -> PipelineState:
            """The main loop. 
            
            While we could make every (model, prompt) pair a node, for 50-100 pairs 
            it's more practical to loop here or use a map-reduce style if LangGraph 
            supported it easily in this version. We'll stick to a robust loop 
            with disk persistence.
            """
            models = state["models"]
            prompts = state["prompts"]
            judge_model = state["judge_model"]
            results = []
            
            total_tasks = len(models) * len(prompts)
            pbar = tqdm(total=total_tasks, desc="Evaluating")
            
            progress_callback = state.get("progress_callback")
            
            processed_count = 0
            run_id = state["run_id"]
            for model in models:
                for prompt in prompts:
                    try:
                        # 1. Generate
                        response: ModelResponse = model.generate(
                            prompt["prompt_text"], 
                            prompt_id=prompt["id"]
                        )
                        
                        # 2. Save Raw (File + DB)
                        raw_filename = f"{model.config.name.replace('/', '_')}_{prompt['id']}.json"
                        raw_path = self.raw_dir / raw_filename
                        raw_record = {
                            "prompt": prompt,
                            "response": {
                                "model_name": response.model_name,
                                "response_text": response.response_text,
                                "success": response.success,
                                "error": response.error,
                                "latency_ms": response.latency_ms,
                                "attempts": response.attempts,
                                "raw_metadata": response.raw_metadata
                            }
                        }
                        with open(raw_path, "w", encoding="utf-8") as f:
                            json.dump(raw_record, f, indent=2)
                        
                        self.db.save_raw_response(
                            run_id, model.config.name, prompt["id"], 
                            prompt["category"], prompt["prompt_text"], 
                            response.response_text, response.success, 
                            response.error, response.latency_ms, 
                            response.raw_metadata
                        )
                        
                        # 3. Score
                        scores = {}
                        if response.success:
                            category = prompt["category"]
                            scorers = SCORER_REGISTRY.get(category, [])
                            for scorer in scorers:
                                try:
                                    scores.update(scorer(prompt, response))
                                except Exception as e:
                                    logger.error("Scorer %s failed for %s: %s", scorer.__name__, prompt["id"], e)
                            
                            if judge_model:
                                try:
                                    judge_results = llm_judge_score(prompt, response, judge_model)
                                    scores["llm_judge"] = judge_results
                                except Exception as e:
                                    logger.error("LLM Judge failed for %s: %s", prompt["id"], e)
                            
                            # Save scores to DB
                            self.db.save_scores(run_id, model.config.name, prompt["id"], scores)
                        else:
                            logger.warning("Skipping scoring for failed response: %s - %s", model.config.name, prompt["id"])
                            scores["error"] = response.error or "Unknown failure"

                        # 4. Record Result
                        result_record = {
                            "model_name": model.config.name,
                            "prompt_id": prompt["id"],
                            "category": prompt["category"],
                            "severity": prompt["severity"],
                            "success": response.success,
                            "latency_ms": response.latency_ms,
                            **scores
                        }
                        results.append(result_record)
                        
                    except Exception as e:
                        logger.error("Pipeline failed for %s - %s: %s", model.config.name, prompt["id"], e)
                    
                    processed_count += 1
                    pbar.update(1)
                    if progress_callback:
                        try:
                            progress_callback(processed_count, total_tasks)
                        except Exception as cb_err:
                            logger.error("Progress callback failed: %s", cb_err)
            
            pbar.close()
            
            # Save all processed scores to a combined file
            combined_path = self.processed_dir / "scored_results.json"
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            logger.info("Evaluation complete. Results saved to %s", combined_path)
            return {**state, "processed_count": processed_count}

        workflow = StateGraph(PipelineState)
        workflow.add_node("init_run", init_run_node)
        workflow.add_node("load_prompts", load_prompts_node)
        workflow.add_node("build_models", build_models_node)
        workflow.add_node("run_eval", run_eval_node)
        
        workflow.set_entry_point("init_run")
        workflow.add_edge("init_run", "load_prompts")
        workflow.add_edge("load_prompts", "build_models")
        workflow.add_edge("build_models", "run_eval")
        workflow.add_edge("run_eval", END)
        
        return workflow.compile()

    def run(self, progress_callback: Optional[Any] = None):
        """Execute the pipeline."""
        initial_state: PipelineState = {
            "config": self.config,
            "models": [],
            "prompts": [],
            "judge_model": None,
            "run_id": "",
            "processed_count": 0,
            "progress_callback": progress_callback
        }
        return self._graph.invoke(initial_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Safety Evaluation Pipeline")
    parser.add_argument("--discover-local", action="store_true", help="Discover local LM Studio models")
    args = parser.parse_args()
    
    pipeline = EvaluationPipeline(discover_local=args.discover_local)
    pipeline.run()
