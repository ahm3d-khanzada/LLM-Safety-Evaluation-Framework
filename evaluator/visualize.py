"""
evaluator.visualize
===================

Generates Matplotlib charts for visualizing evaluation results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from utils.config_loader import load_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

def plot_overall_safety_score(df: pd.DataFrame, output_path: Optional[Path] = None, ax=None):
    """Bar chart of overall safety scores."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(df["model_name"], df["overall_safety_score"], color="skyblue")
    ax.set_title("Overall Safety Score by Model (Higher is Better)")
    ax.set_ylabel("Score (0-100)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 105)
    for i, v in enumerate(df["overall_safety_score"]):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", fontweight="bold")
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_hallucination_rate(df: pd.DataFrame, output_path: Optional[Path] = None, ax=None):
    """Plot hallucination rate (inverse of truthfulness)."""
    plot_metric_by_model(
        df, "truthfulness_mean", "Hallucination Rate (Lower is Better)", 
        "Rate", output_path, inverse=True, ax=ax
    )

def plot_injection_success_rate(df: pd.DataFrame, output_path: Optional[Path] = None, ax=None):
    """Plot injection success rate."""
    plot_metric_by_model(
        df, "prompt_injection_resistance_mean", "Prompt Injection Success Rate (Lower is Better)", 
        "Success Rate", output_path, inverse=True, ax=ax
    )

def plot_jailbreak_success_rate(df: pd.DataFrame, output_path: Optional[Path] = None, ax=None):
    """Plot jailbreak success rate."""
    plot_metric_by_model(
        df, "jailbreak_resistance_mean", "Jailbreak Success Rate (Lower is Better)", 
        "Success Rate", output_path, inverse=True, ax=ax
    )

def plot_latency_comparison(df: pd.DataFrame, output_path: Optional[Path] = None, ax=None, baseline_name: Optional[str] = None):
    """Bar chart of mean latency with error bars."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # We use mean_latency_ms
    df = df.dropna(subset=["mean_latency_ms"]).sort_values("mean_latency_ms")
    
    ax.bar(df["model_name"], df["mean_latency_ms"] / 1000.0, color="lightgreen")
    ax.set_title("Mean Response Latency (Lower is Faster)")
    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Model")
    
    # Add baseline comparison line if baseline_name is present
    if baseline_name and baseline_name in df["model_name"].values:
        baseline = df[df["model_name"] == baseline_name]
        baseline_val = baseline["mean_latency_ms"].iloc[0] / 1000.0
        ax.axhline(y=baseline_val, color='r', linestyle='--', label=f'Baseline: {baseline_name} ({baseline_val:.2f}s)')
        ax.legend()
    elif not df.empty:
        # Fallback to gpt-4o-mini if it exists and no baseline_name provided
        baseline = df[df["model_name"] == "gpt-4o-mini"]
        if not baseline.empty:
            baseline_val = baseline["mean_latency_ms"].iloc[0] / 1000.0
            ax.axhline(y=baseline_val, color='r', linestyle='--', label=f'OpenAI Baseline ({baseline_val:.2f}s)')
            ax.legend()

    for i, v in enumerate(df["mean_latency_ms"]):
        ax.text(i, (v / 1000.0) + 0.1, f"{v/1000.0:.2f}s", ha="center")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_metric_by_model(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Optional[Path] = None, inverse: bool = False, ax=None):
    """Plot a specific metric across models."""
    if metric not in summary_df.columns:
        logger.warning("Metric %s not found in results", metric)
        return

    # Filter rows where the metric exists
    df = summary_df.dropna(subset=[metric]).copy()
    if df.empty:
        logger.warning("No data for metric %s", metric)
        return
        
    if inverse:
        df[metric] = 1.0 - df[metric]
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df.plot(kind="bar", x="model_name", y=metric, color="salmon", ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.1)
    # Rotation of xticks can be handled outside or here
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def plot_category_heatmap(summary_df: pd.DataFrame, output_path: Path):
    """Heatmap of models vs categories for a representative safety metric."""
    # Pivot to get models on one axis, categories on another
    # Use safe_unsafe as the representative metric
    if "safe_unsafe" not in summary_df.columns:
        logger.warning("safe_unsafe metric not found for heatmap")
        return
        
    pivot_df = summary_df.pivot(index="model_name", columns="category", values="safe_unsafe")
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_df, cmap="RdYlGn", vmin=0, vmax=1)
    
    # Show all ticks and label them with the respective list entries
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            plt.text(j, i, f"{val:.2f}" if not pd.isna(val) else "N/A",
                     ha="center", va="center", color="black")

    plt.title("Safety Score Heatmap (safe_unsafe metric)")
    plt.colorbar(im, label="Safety Score")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_visualizations(config_path: Optional[str] = None):
    """Load results and generate all charts."""
    config = load_config(config_path) if config_path else load_config()
    paths = config["paths"]
    results_dir = PROJECT_ROOT / paths["results_dir"]
    figures_dir = PROJECT_ROOT / paths["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        overall_df = pd.read_csv(results_dir / "model_overall.csv")
        summary_df = pd.read_csv(results_dir / "category_summary.csv")
    except FileNotFoundError as e:
        logger.error("Could not load result files for visualization: %s", e)
        return

    # Generate charts
    plot_overall_safety_score(overall_df, figures_dir / "overall_safety.png")
    
    # Use overall_df for model-level comparisons as it has aggregated means
    plot_metric_by_model(
        overall_df, "truthfulness_mean", "Hallucination Rate (Lower is Better)", 
        "Rate", figures_dir / "hallucination_rate.png", inverse=True
    )
    
    plot_metric_by_model(
        overall_df, "prompt_injection_resistance_mean", "Prompt Injection Success Rate (Lower is Better)", 
        "Success Rate", figures_dir / "prompt_injection_success.png", inverse=True
    )
    
    plot_metric_by_model(
        overall_df, "jailbreak_resistance_mean", "Jailbreak Success Rate (Lower is Better)", 
        "Success Rate", figures_dir / "jailbreak_success.png", inverse=True
    )
    
    plot_metric_by_model(
        overall_df, "bias_score_mean", "Bias Score (Higher is Better)", 
        "Score", figures_dir / "bias_score.png"
    )
    
    plot_metric_by_model(
        overall_df, "privacy_preservation_mean", "Privacy Preservation Score (Higher is Better)", 
        "Score", figures_dir / "privacy_preservation.png"
    )
    
    plot_category_heatmap(summary_df, figures_dir / "category_heatmap.png")
    
    plot_latency_comparison(overall_df, figures_dir / "latency_comparison.png")
    
    logger.info("All visualizations generated in %s", figures_dir)

if __name__ == "__main__":
    run_visualizations()
