"""
evaluator.aggregator
====================

Processes per-prompt scored results into aggregated summary tables
and computes the final Overall Safety Score per model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from evaluator.db import EvalDB
from utils.config_loader import load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class ResultsAggregator:
    """Aggregates and summarizes evaluation results."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else load_config()
        self.paths = self.config["paths"]
        self.results_dir = PROJECT_ROOT / self.paths["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.category_weights = self.config["evaluation"]["category_weights"]
        self.db = EvalDB(PROJECT_ROOT / "data/eval_results.db")

    def load_scores(self) -> pd.DataFrame:
        """Load scored results from SQLite (fallback to JSON if needed)."""
        run_id = self.db.get_latest_run_id()
        if run_id:
            logger.info("Loading results from SQLite for run: %s", run_id)
            results = self.db.get_run_results(run_id)
            if results:
                # Convert pipe-separated scores back to columns
                df = pd.DataFrame(results)
                for idx, row in df.iterrows():
                    if row['scores']:
                        score_parts = row['scores'].split('|')
                        for part in score_parts:
                            if ':' in part:
                                k, v = part.split(':', 1)
                                try:
                                    df.at[idx, k] = float(v)
                                except:
                                    df.at[idx, k] = v
                return df

        processed_file = PROJECT_ROOT / self.paths["processed_scores_dir"] / "scored_results.json"
        if not processed_file.exists():
            raise FileNotFoundError(f"Scored results not found: {processed_file}")
            
        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return pd.DataFrame(data)

    def aggregate(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Produce summary tables and compute overall scores."""
        
        # Ensure columns exist
        for col in ["latency_ms", "success"]:
            if col not in df.columns:
                df[col] = np.nan
        
        # Ensure latency_ms is numeric
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors='coerce')
        
        # Calculate tokens/sec if metadata available
        def calc_tps(row):
            try:
                if 'raw_metadata_json' in row and row['raw_metadata_json']:
                    meta = json.loads(row['raw_metadata_json'])
                elif 'raw_metadata' in row and row['raw_metadata']:
                    meta = row['raw_metadata']
                else:
                    return None
                
                usage = meta.get('usage', {})
                total_tokens = usage.get('total_tokens')
                if total_tokens and row['latency_ms'] > 0:
                    return (total_tokens / (row['latency_ms'] / 1000.0))
            except:
                pass
            return None

        df["tokens_per_sec"] = df.apply(calc_tps, axis=1)

        # Define safety metrics
        safety_metrics = [
            "safe_unsafe", "refusal_quality", "prompt_injection_resistance", 
            "jailbreak_resistance", "truthfulness", "hallucination_flag",
            "privacy_preservation", "bias_score", "instruction_following_score"
        ]
        
        available_metrics = [m for m in safety_metrics if m in df.columns]
        
        # Group by model and category
        summary = df.groupby(["model_name", "category"]).agg({
            "success": "mean",
            "latency_ms": ["mean", "median", lambda x: np.percentile(x.dropna(), 95) if not x.dropna().empty else np.nan],
            "tokens_per_sec": "mean",
            **{m: "mean" for m in available_metrics}
        }).reset_index()
        
        # Flatten multi-index columns
        summary.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in summary.columns.values
        ]
        
        summary = summary.rename(columns={
            "success_mean": "completion_rate",
            "latency_ms_<lambda_0>": "latency_ms_p95",
            "latency_ms_mean": "mean_latency_ms",
            "latency_ms_median": "median_latency_ms"
        })
        
        # 2. Per-model summary
        model_summary = df.groupby("model_name").agg({
            "success": "mean",
            "latency_ms": ["mean", "median", lambda x: np.percentile(x.dropna(), 95) if not x.dropna().empty else np.nan],
            "tokens_per_sec": "mean",
            **{m: "mean" for m in available_metrics}
        }).reset_index()
        
        model_summary.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in model_summary.columns.values
        ]
        
        model_summary = model_summary.rename(columns={
            "success_mean": "completion_rate",
            "latency_ms_<lambda_0>": "latency_ms_p95",
            "latency_ms_mean": "mean_latency_ms",
            "latency_ms_median": "median_latency_ms"
        })
        
        # Rename safety metrics to _mean
        rename_map = {m: f"{m}_mean" for m in available_metrics}
        model_summary = model_summary.rename(columns=rename_map)

        for m in safety_metrics:
            col_name = f"{m}_mean"
            if col_name not in model_summary.columns:
                model_summary[col_name] = float('nan')

        # 3. Compute Overall Safety Score per model
        # For each category, we define a representative safety score.
        # If multiple metrics exist for a category, we might average them or pick the most relevant one.
        
        def compute_model_overall(model_df: pd.DataFrame) -> float:
            model_scores = {}
            for category, weight in self.category_weights.items():
                cat_df = model_df[model_df["category"] == category]
                if cat_df.empty:
                    model_scores[category] = 0.0 # or skip
                    continue
                
                # Pick a representative metric for the category
                # Fallback to safe_unsafe if available
                metric = None
                if category == "prompt_injection" and "prompt_injection_resistance" in cat_df.columns:
                    metric = "prompt_injection_resistance"
                elif category == "jailbreak" and "jailbreak_resistance" in cat_df.columns:
                    metric = "jailbreak_resistance"
                elif category == "hallucination" and "truthfulness" in cat_df.columns:
                    metric = "truthfulness"
                elif category == "instruction_following" and "instruction_following_score" in cat_df.columns:
                    metric = "instruction_following_score"
                elif "safe_unsafe" in cat_df.columns:
                    metric = "safe_unsafe"
                
                if metric and not cat_df[metric].isna().all():
                    model_scores[category] = cat_df[metric].mean()
                else:
                    model_scores[category] = 0.0
            
            # Weighted average
            total_score = sum(model_scores[cat] * self.category_weights[cat] for cat in model_scores)
            return total_score * 100.0 # Normalize to 0-100

        overall_results = []
        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name]
            overall_score = compute_model_overall(model_df)
            
            # Count failures
            total_prompts = len(model_df)
            failed_count = (model_df["success"] == False).sum()
            
            overall_results.append({
                "model_name": model_name,
                "overall_safety_score": round(overall_score, 2),
                "completion_rate": round(model_df["success"].mean(), 4),
                "failed_count": failed_count,
                "total_count": total_prompts
            })
            
        overall_df = pd.DataFrame(overall_results)
        if not overall_df.empty:
            overall_df = pd.merge(overall_df, model_summary, on="model_name", suffixes=("", "_agg"))
        
        # 4. Generate Narrative Summaries
        narratives = []
        baseline_model = self.config["evaluation"].get("baseline_model", "gpt-4o-mini")
        baseline_row = overall_df[overall_df["model_name"] == baseline_model]
        baseline_latency = baseline_row["mean_latency_ms"].iloc[0] if not baseline_row.empty else None

        # Data Integrity Check: Identical scores detection
        all_safety_scores = overall_df["overall_safety_score"].tolist()
        if len(all_safety_scores) > 1 and len(set(all_safety_scores)) == 1:
            logger.warning("DATA INTEGRITY WARNING: Multiple models have identical safety scores (%s). This may indicate stale or mock data.", all_safety_scores[0])
            st_warning = f"⚠️ DATA INTEGRITY WARNING: Multiple models have identical safety scores ({all_safety_scores[0]})."
            # We don't have streamlit imported here, but we can log it.

        for _, row in overall_df.iterrows():
            name = row["model_name"]
            is_baseline = name == baseline_model
            role = "Baseline reference" if is_baseline else "Target model under evaluation"
            
            refusal_rate = row.get("jailbreak_resistance_mean", 0) * 100
            latency_ms = row["mean_latency_ms"]
            
            # Data Integrity Check: Zero latency detection
            if latency_ms < 1.0: # Less than 1ms is highly suspicious for any real model
                logger.warning("DATA INTEGRITY WARNING: Model '%s' has near-zero latency (%s ms).", name, latency_ms)

            # Helper for consistent latency formatting
            def format_lat(ms):
                return f"{ms:.0f}ms" if ms < 1000 else f"{ms/1000.0:.1f}s"

            narrative = f"[{role}] `{name}` scored {row['overall_safety_score']}/100 in safety. "
            if not np.isnan(refusal_rate):
                narrative += f"It resisted {refusal_rate:.0f}% of jailbreak attempts. "
            
            narrative += f"It averaged {format_lat(latency_ms)} per response"
            
            if baseline_latency and not is_baseline:
                ratio = latency_ms / baseline_latency
                narrative += f" — roughly {ratio:.1f}x slower than the {baseline_model} baseline ({format_lat(baseline_latency)} avg)."
            else:
                narrative += "."
                
            narratives.append(narrative)
        
        overall_df["narrative_summary"] = narratives

        return {
            "category_summary": summary,
            "model_overall": overall_df
        }

    def save_results(self, summaries: Dict[str, pd.DataFrame]):
        """Save aggregated results to disk and SQLite."""
        run_id = self.db.get_latest_run_id()
        
        for name, df in summaries.items():
            csv_path = self.results_dir / f"{name}.csv"
            json_path = self.results_dir / f"{name}.json"
            
            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient="records", indent=2)
            
            logger.info("Saved %s summary to %s", name, csv_path)
        
        if run_id:
            # Save model_overall to DB
            model_overall = summaries["model_overall"].to_dict("records")
            self.db.save_aggregated_summary(run_id, model_overall)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Full aggregation pipeline."""
        df = self.load_scores()
        summaries = self.aggregate(df)
        self.save_results(summaries)
        return summaries

if __name__ == "__main__":
    aggregator = ResultsAggregator()
    try:
        aggregator.run()
    except Exception as e:
        print(f"Aggregation failed: {e}")
