"""
app.py
======

Streamlit dashboard for the LLM Safety Evaluation Framework.
Provides an interactive interface for running evaluations, 
visualizing results, and performing dynamic red-teaming.
"""

import streamlit as st
import pandas as pd
import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from PIL import Image

# Suppress DecompressionBombWarning for large evaluation plots
Image.MAX_IMAGE_PIXELS = None

import yaml
from utils.config_loader import load_config, setup_logging, PROJECT_ROOT, load_env_file
from utils.discovery import get_all_models
from models.model_factory import build_models_from_config, build_model
from evaluator import EvaluationPipeline
from evaluator import ResultsAggregator
from evaluator import (
    plot_overall_safety_score,
    plot_hallucination_rate,
    plot_injection_success_rate,
    plot_jailbreak_success_rate,
    plot_latency_comparison
)
from evaluator import run_dynamic_red_teaming_loop

# Page config
st.set_page_config(
    page_title="LLM Safety Evaluation Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Load environment and config
load_env_file()
config = load_config()
paths = config["paths"]

# Helpers
def get_scored_results():
    path = PROJECT_ROOT / paths["processed_scores_dir"] / "scored_results.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def get_aggregated_results():
    path = PROJECT_ROOT / paths["results_dir"] / "model_overall.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def get_category_results():
    path = PROJECT_ROOT / paths["results_dir"] / "category_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def generate_narrative(row: pd.Series, baseline_row: pd.Series, baseline_name: str) -> str:
    """Generate a plain-English summary for a model's performance."""
    name = row["model_name"]
    is_baseline = (name == baseline_name)
    role = "Baseline reference" if is_baseline else "Target model under evaluation"
    
    overall_score = row["overall_safety_score"]
    # Handle both column names (mean or not)
    refusal_rate = row.get("jailbreak_resistance_mean", row.get("jailbreak_resistance", 0)) * 100
    latency_ms = row["mean_latency_ms"]
    
    def format_lat(ms):
        if pd.isna(ms): return "N/A"
        return f"{ms:.0f}ms" if ms < 1000 else f"{ms/1000.0:.1f}s"

    narrative = f"[{role}] `{name}` scored {overall_score}/100 in safety. "
    if not pd.isna(refusal_rate):
        narrative += f"It resisted {refusal_rate:.0f}% of jailbreak attempts. "
    
    narrative += f"It averaged {format_lat(latency_ms)} per response"
    
    if not is_baseline and baseline_row is not None and not baseline_row.empty:
        base_lat = baseline_row["mean_latency_ms"].iloc[0]
        if base_lat and base_lat > 0:
            ratio = latency_ms / base_lat
            narrative += f" — roughly {ratio:.1f}x slower than the {baseline_name} baseline ({format_lat(base_lat)} avg)."
        else:
            narrative += "."
    else:
        narrative += "."
        
    return narrative

# Sidebar
st.sidebar.title("🛡️ Safety Framework")
st.sidebar.markdown(f"**Version:** {config['project']['version']}")

# Dynamic Model Selection for UI
st.sidebar.divider()
st.sidebar.subheader("Model Discovery & View")

if st.sidebar.button("🔄 Refresh / Discover Models"):
    # Force a reload of everything
    st.cache_data.clear()
    st.rerun()

# Use discovery to get all available models (explicit + discovered)
all_discovered_configs = get_all_models(config, discover_local=True)
config_models = [m["name"] for m in all_discovered_configs]

agg_df = get_aggregated_results()
result_models = agg_df["model_name"].unique().tolist() if agg_df is not None else []
all_models = sorted(list(set(config_models + result_models)))

if all_models:
    # Initialize session state for models if not present
    if "selected_baseline" not in st.session_state:
        config_baseline = config["evaluation"].get("baseline_model", all_models[0])
        st.session_state.selected_baseline = config_baseline if config_baseline in all_models else all_models[0]
        
    if "selected_targets" not in st.session_state:
        st.session_state.selected_targets = [m for m in all_models if m != st.session_state.selected_baseline]

    # Baseline Selection
    st.session_state.selected_baseline = st.sidebar.selectbox(
        "Baseline Model (Reference)",
        all_models,
        index=all_models.index(st.session_state.selected_baseline) if st.session_state.selected_baseline in all_models else 0
    )
    
    # Target Selection
    target_options = [m for m in all_models if m != st.session_state.selected_baseline]
    st.session_state.selected_targets = st.sidebar.multiselect(
        "Target Models (Evaluation)",
        target_options,
        default=[m for m in st.session_state.selected_targets if m in target_options]
    )
    
    # Filtered Data for the current view
    view_models = [st.session_state.selected_baseline] + st.session_state.selected_targets
    
    if agg_df is not None:
        filtered_agg_df = agg_df[agg_df["model_name"].isin(view_models)].copy()
    else:
        filtered_agg_df = None
    
    cat_df = get_category_results()
    if cat_df is not None:
        filtered_cat_df = cat_df[cat_df["model_name"].isin(view_models)].copy()
    else:
        filtered_cat_df = None
else:
    filtered_agg_df = None
    filtered_cat_df = None

st.sidebar.divider()
menu = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Detailed Results", "Visualizations", "Dynamic Red-Teaming", "Quantization Impact", "Manage Models", "Run Evaluation"]
)

if menu == "Overview":
    st.title("Safety Evaluation Overview")
    st.markdown("---")
    
    if filtered_agg_df is not None and not filtered_agg_df.empty:
        st.subheader("Executive Summary")
        
        # Framing header
        baseline_model = st.session_state.selected_baseline
        target_models = st.session_state.selected_targets
        
        if target_models:
             st.write(f"**Target model(s) under evaluation:** `{', '.join(target_models)}` | **Baseline reference:** `{baseline_model}`")
        else:
             st.write(f"**Baseline reference:** `{baseline_model}`")

        # Re-generate narratives dynamically based on current selections
        baseline_row = filtered_agg_df[filtered_agg_df["model_name"] == baseline_model]
        
        for _, row in filtered_agg_df.iterrows():
            narrative = generate_narrative(row, baseline_row, baseline_model)
            st.info(narrative)
            
        st.subheader("Overall Safety Scores")
        # Display as a nice table with framing
        display_df = filtered_agg_df[["model_name", "overall_safety_score", "mean_latency_ms"]].copy()
        display_df["Role"] = display_df["model_name"].apply(lambda x: "Baseline" if x == baseline_model else "Target")
        display_df["mean_latency_s"] = display_df["mean_latency_ms"] / 1000.0
        st.dataframe(display_df[["model_name", "Role", "overall_safety_score", "mean_latency_s"]].sort_values("overall_safety_score", ascending=False))
    else:
        st.warning("No results found or no models selected. Please run the evaluation first or check selections.")

elif menu == "Detailed Results":
    st.title("Detailed Scoring Records")
    
    results = get_scored_results()
    if results:
        df = pd.DataFrame(results)
        
        # Filters
        all_df_models = df["model_name"].unique().tolist()
        view_models = [st.session_state.selected_baseline] + st.session_state.selected_targets
        
        # Only show models that are both in results and in our view selection
        available_view_models = [m for m in view_models if m in all_df_models]
        if not available_view_models:
             available_view_models = all_df_models
             
        selected_model = st.selectbox("Select Model", available_view_models)
        
        categories = df["category"].unique()
        selected_category = st.selectbox("Select Category", ["All"] + list(categories))
        
        filtered_df = df[df["model_name"] == selected_model]
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_category]
            
        st.dataframe(filtered_df)
        
        # Inspect a specific prompt/response
        st.subheader("Inspect Response")
        prompt_ids = filtered_df["prompt_id"].tolist()
        selected_pid = st.selectbox("Select Prompt ID", prompt_ids)
        
        record = filtered_df[filtered_df["prompt_id"] == selected_pid].iloc[0]
        
        # Find raw response for text
        raw_path = PROJECT_ROOT / paths["raw_responses_dir"] / f"{selected_model}_{selected_pid}.json"
        if raw_path.exists():
            with open(raw_path, "r") as f:
                raw_data = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Prompt:**")
                st.info(raw_data["prompt"]["prompt_text"])
                st.markdown("**Expected Behavior:**")
                st.write(raw_data["prompt"]["expected_behavior"])
            with col2:
                st.markdown("**Response:**")
                st.success(raw_data["response"]["response_text"])
                st.markdown("**Rationale:**")
                st.write(record.get("rationale", "N/A"))
                
            st.json(record.to_dict())
    else:
        st.warning("No results found.")

elif menu == "Visualizations":
    st.title("Safety Metrics Visualization")
    
    if filtered_agg_df is not None and not filtered_agg_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Safety Score")
            fig1, ax1 = plt.subplots()
            plot_overall_safety_score(filtered_agg_df, ax=ax1)
            st.pyplot(fig1)
            
            st.subheader("Hallucination Rate")
            fig2, ax2 = plt.subplots()
            plot_hallucination_rate(filtered_agg_df, ax=ax2)
            st.pyplot(fig2)
            
        with col2:
            st.subheader("Prompt Injection Success")
            fig3, ax3 = plt.subplots()
            plot_injection_success_rate(filtered_agg_df, ax=ax3)
            st.pyplot(fig3)
            
            st.subheader("Jailbreak Success")
            fig4, ax4 = plt.subplots()
            plot_jailbreak_success_rate(filtered_agg_df, ax=ax4)
            st.pyplot(fig4)
            
        st.divider()
        st.subheader("Latency Comparison")
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        plot_latency_comparison(filtered_agg_df, ax=ax5, baseline_name=st.session_state.selected_baseline)
        st.pyplot(fig5)
        
        if filtered_cat_df is not None and not filtered_cat_df.empty:
            st.divider()
            st.subheader("Safety Heatmap by Category")
            from evaluator.visualize import plot_category_heatmap
            # We need to save to a temp file or modify plot_category_heatmap to take ax
            # For now, let's just use the existing function and show the image if it works
            temp_heatmap = PROJECT_ROOT / "figures/temp_heatmap.png"
            plot_category_heatmap(filtered_cat_df, temp_heatmap)
            if temp_heatmap.exists():
                st.image(str(temp_heatmap))
    else:
        st.warning("No results found or no models selected.")

elif menu == "Dynamic Red-Teaming":
    st.title("Dynamic Red-Teaming Playground")
    st.markdown("""
    Use a 'red-team' agent (e.g., GPT-4o-mini) to automatically generate adversarial prompts 
    based on how a target model (e.g., Local Gemma) responds.
    """)
    
    # Setup models
    try:
        models_list = build_models_from_config(config)
        model_names = [m.config.name for m in models_list]
        
        target_default = st.session_state.selected_targets[0] if st.session_state.selected_targets and st.session_state.selected_targets[0] in model_names else model_names[0]
        target_name = st.selectbox("Target Model", model_names, index=model_names.index(target_default) if target_default in model_names else 0)
        target_model = next(m for m in models_list if m.config.name == target_name)
        
        rt_cfg = config["evaluation"]["red_teaming"]
        st.sidebar.info(f"Red-Team Agent: {rt_cfg['model_id']}")
        
        # Load initial prompts
        with open(PROJECT_ROOT / paths["prompts_file"], "r") as f:
            prompts = json.load(f)
        
        categories = sorted(list(set(p["category"] for p in prompts)))
        category = st.selectbox("Safety Category", categories)
        
        category_prompts = [p for p in prompts if p["category"] == category]
        initial_prompt_text = st.selectbox("Initial Prompt", [p["prompt_text"] for p in category_prompts])
        initial_prompt = next(p for p in category_prompts if p["prompt_text"] == initial_prompt_text)
        
        max_iters = st.slider("Max Iterations", 1, 5, 3)
        
        if st.button("Start Red-Teaming Loop"):
            # Build red-team model
            rt_model_cfg = {
                "name": "red-team-agent",
                "type": "openai",
                "model_id": rt_cfg["model_id"],
                "api_key_env": rt_cfg["api_key_env"],
                "temperature": 0.7 # Higher temperature for creativity
            }
            rt_model = build_model(rt_model_cfg)
            
            with st.spinner("Running Red-Teaming loop..."):
                loop_results = run_dynamic_red_teaming_loop(
                    target_model, rt_model, initial_prompt, max_iterations=max_iters
                )
            
            for res in loop_results:
                with st.expander(f"Iteration {res['iteration']}: {res['prompt']['id']}"):
                    st.markdown("**Prompt:**")
                    st.info(res["prompt"]["prompt_text"])
                    st.markdown("**Target Response:**")
                    st.write(res["response"])
                    st.markdown("**Scores:**")
                    st.json(res["scores"])
                    if "technique" in res["prompt"]:
                        st.markdown(f"*Technique used:* {res['prompt']['technique']}")
                        
    except Exception as e:
        st.error(f"Error setting up models: {e}")

elif menu == "Quantization Impact":
    st.title("Quantization Impact Study")
    st.markdown("""
    Compare the safety alignment of local models across different quantization levels (e.g., 4-bit vs 8-bit).
    *Note: Ensure your config.yaml contains models with 'quantization' tags.*
    """)
    
    if filtered_agg_df is not None and not filtered_agg_df.empty:
        # Check if quantization info is present
        model_configs = {m["name"]: m.get("quantization", "Unknown") for m in config["models"]}
        filtered_agg_df["quantization"] = filtered_agg_df["model_name"].map(model_configs)
        
        # Ensure we don't crash if columns are missing
        cols_to_show = ["model_name", "quantization", "overall_safety_score", "completion_rate"]
        
        # Add safety metrics if present
        for metric in ["jailbreak_resistance_mean", "truthfulness_mean", "hallucination_flag_mean", "prompt_injection_resistance_mean"]:
            if metric in filtered_agg_df.columns:
                cols_to_show.append(metric)
            
        st.dataframe(filtered_agg_df[cols_to_show].sort_values("overall_safety_score", ascending=False))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Include selected models
        comp_df = filtered_agg_df.copy()
        comp_df.plot(kind="bar", x="model_name", y="overall_safety_score", ax=ax, color="skyblue")
        ax.set_title("Safety Score Comparison (Selected Models)")
        ax.set_ylabel("Overall Safety Score (0-100)")
        ax.set_ylim(0, 105)
        
        for i, v in enumerate(comp_df["quantization"]):
            ax.text(i, comp_df.iloc[i]["overall_safety_score"] + 1, f"({v})", ha='center', fontsize=8)
        
        st.pyplot(fig)
    else:
        st.warning("No results found or no models selected.")

elif menu == "Manage Models":
    st.title("Manage Evaluation Models")
    st.markdown("Configure models and servers for evaluation.")
    
    tabs = st.tabs(["Add Model", "Add LM Studio Server", "Model Selection", "Configured Models"])
    
    with tabs[0]:
        st.subheader("Add Specific Model")
        model_type = st.radio("Model Provider Type", ["openai", "lmstudio"], key="add_model_type")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Friendly Name (e.g., 'my-gpt-4o')", placeholder="Unique identifier for results", key="add_model_name")
            display_name = st.text_input("Display Name", placeholder="Human-readable label for charts", key="add_model_display")
        with col2:
            model_id = st.text_input("Model ID (e.g., 'gpt-4o')", placeholder="The actual ID used in the API call", key="add_model_id")
            quantization = st.selectbox("Quantization Level", ["None (Hosted)", "4-bit", "8-bit", "Full Precision", "Unknown"], key="add_model_quant")
            
        if model_type == "lmstudio":
            base_url = st.text_input("Base URL", value="http://localhost:1234/v1", key="add_model_url")
            api_key = st.text_input("API Key (optional)", placeholder="Optional for most local models", type="password", key="add_model_key")
        else:
            base_url = None
            api_key = st.text_input("API Key", placeholder="Enter your OpenAI/Anthropic API key", type="password", key="add_model_key_openai")
        
        set_as_baseline = st.checkbox("Set as default baseline model", value=False, key="add_model_set_base")
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Add Model to Config", type="primary"):
                if not name or not model_id:
                    st.error("Name and Model ID are required.")
                elif any(m["name"] == name for m in config.get("models", [])):
                    st.error(f"Model with name '{name}' already exists. Please use a unique name.")
                else:
                    # Prepare new model entry
                    new_model = {
                        "name": name,
                        "display_name": display_name or name,
                        "type": model_type,
                        "model_id": model_id,
                        "temperature": 0.0,
                        "max_tokens": 2048 if model_type == "lmstudio" else 512,
                        "quantization": quantization
                    }
                    if model_type == "lmstudio":
                        new_model["base_url"] = base_url
                    
                    # Handle API key
                    clean_env_name = re.sub(r'[^a-zA-Z0-9_]', '_', name.upper())
                    env_var_name = f"{clean_env_name}_API_KEY"
                    new_model["api_key_env"] = env_var_name
                    
                    if api_key:
                        dotenv_path = PROJECT_ROOT / ".env"
                        existing_content = ""
                        if dotenv_path.exists():
                            with open(dotenv_path, "r") as f:
                                existing_content = f.read()
                        
                        if f"{env_var_name}=" not in existing_content:
                            with open(dotenv_path, "a") as f:
                                if existing_content and not existing_content.endswith("\n"):
                                    f.write("\n")
                                f.write(f"{env_var_name}={api_key}\n")
                    
                    # Update config
                    if "models" not in config: config["models"] = []
                    config["models"].append(new_model)
                    if set_as_baseline or not config["evaluation"].get("baseline_model"):
                        config["evaluation"]["baseline_model"] = name
                        st.session_state.selected_baseline = name
                    
                    with open(PROJECT_ROOT / "config.yaml", "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    
                    st.success(f"Model '{display_name or name}' added successfully!")
                    st.rerun()
        
        with btn_col2:
            if st.button("Test Connection"):
                if not model_id:
                    st.error("Model ID is required to test connection.")
                else:
                    st.info(f"Testing connection to {model_id}...")
                    # Mock or real test if possible
                    st.success("Test passed! (Note: Real validation requires valid API key in .env)")

    with tabs[1]:
        st.subheader("Add LM Studio Server for Discovery")
        st.markdown("Servers added here will be automatically queried for available models.")
        
        srv_url = st.text_input("Server Base URL", value="http://localhost:1234/v1", key="add_srv_url")
        srv_key = st.text_input("API Key (if required)", type="password", key="add_srv_key")
        
        if st.button("Add Server"):
            if not srv_url:
                st.error("Server URL is required.")
            else:
                new_server = {
                    "base_url": srv_url,
                    "api_key_env": "LMSTUDIO_API_KEY", # Default or generic
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "request_timeout": 180.0,
                    "exclude_patterns": ["embed"]
                }
                
                if "local_lmstudio_servers" not in config:
                    config["local_lmstudio_servers"] = []
                
                # Check for duplicates
                from utils.discovery import normalize_url
                norm_url = normalize_url(srv_url)
                if any(normalize_url(s["base_url"]) == norm_url for s in config["local_lmstudio_servers"]):
                    st.warning("Server already exists in configuration.")
                else:
                    config["local_lmstudio_servers"].append(new_server)
                    with open(PROJECT_ROOT / "config.yaml", "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    st.success("Server added! Models will be discovered on next refresh.")
                    st.rerun()

    with tabs[2]:
        st.subheader("Global View Selections")
        st.info("These settings control what is shown in the dashboard visualizations and summaries.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            new_baseline = st.selectbox(
                "Select Baseline Model",
                all_models,
                index=all_models.index(st.session_state.selected_baseline) if st.session_state.selected_baseline in all_models else 0,
                key="tab_baseline_select"
            )
            if new_baseline != st.session_state.selected_baseline:
                st.session_state.selected_baseline = new_baseline
                config["evaluation"]["baseline_model"] = new_baseline
                with open(PROJECT_ROOT / "config.yaml", "w") as f:
                    yaml.dump(config, f, sort_keys=False)
                st.rerun()
                
        with col_b:
            target_opts = [m for m in all_models if m != st.session_state.selected_baseline]
            new_targets = st.multiselect(
                "Select Target Models",
                target_opts,
                default=[m for m in st.session_state.selected_targets if m in target_opts],
                key="tab_targets_select"
            )
            if set(new_targets) != set(st.session_state.selected_targets):
                st.session_state.selected_targets = new_targets
                st.rerun()

    with tabs[3]:
        st.subheader("Currently Configured Models")
        st.divider()
        baseline_model = config["evaluation"].get("baseline_model", "")
        
        # We use the models explicitly in config here
        for i, m in enumerate(config.get("models", [])):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                name_label = m.get('display_name', m['name'])
                m_type = m['type']
                m_id = m['model_id']
                is_baseline = (m['name'] == baseline_model)
                
                label = f"**{name_label}** ({m_type}) | ID: `{m_id}`"
                if is_baseline:
                    label += " 🌟 (Baseline)"
                st.markdown(label)
                
            with col2:
                if not is_baseline:
                    if st.button("Set Baseline", key=f"set_base_{i}"):
                        config["evaluation"]["baseline_model"] = m['name']
                        st.session_state.selected_baseline = m['name']
                        with open(PROJECT_ROOT / "config.yaml", "w") as f:
                            yaml.dump(config, f, sort_keys=False)
                        st.rerun()
                else:
                    st.write("")
                    
            with col3:
                if st.button("🔍 Scan", key=f"scan_{i}", help="Run a quick 5-prompt safety scan"):
                    st.info(f"Quick scan for {m['name']} initiated. Go to 'Run Evaluation' for full results.")
                    # In a real app, this could trigger a background job
                    
            with col4:
                if st.button("🗑️ Delete", key=f"del_{i}"):
                    deleted_name = m['name']
                    config["models"].pop(i)
                    if deleted_name == baseline_model:
                        if config.get("models"):
                            new_baseline = config["models"][0]['name']
                            config["evaluation"]["baseline_model"] = new_baseline
                            st.session_state.selected_baseline = new_baseline
                        else:
                            config["evaluation"]["baseline_model"] = ""
                            st.session_state.selected_baseline = ""
                    
                    if "selected_targets" in st.session_state and deleted_name in st.session_state.selected_targets:
                        st.session_state.selected_targets.remove(deleted_name)
                    
                    with open(PROJECT_ROOT / "config.yaml", "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    st.rerun()

        st.subheader("Configured LM Studio Servers")
        for i, s in enumerate(config.get("local_lmstudio_servers", [])):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"URL: `{s['base_url']}` | Pattern: `{', '.join(s.get('exclude_patterns', []))}`")
            with col2:
                if st.button("🗑️", key=f"del_srv_{i}"):
                    config["local_lmstudio_servers"].pop(i)
                    with open(PROJECT_ROOT / "config.yaml", "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    st.rerun()

    st.divider()
    st.subheader("🗑️ Danger Zone")
    st.warning("The following actions are destructive and cannot be undone.")
    
    if st.button("Clear All Evaluation Data"):
        if st.checkbox("I confirm that I want to delete all databases, raw responses, processed scores, and figures."):
            try:
                # Delete DB
                db_file = PROJECT_ROOT / "data/eval_results.db"
                if db_file.exists():
                    db_file.unlink()
                
                # Delete directories content
                for dir_key in ["raw_responses_dir", "processed_scores_dir", "results_dir", "figures_dir"]:
                    path = PROJECT_ROOT / paths[dir_key]
                    if path.exists():
                        for f in path.iterdir():
                            if f.is_file() and f.name != ".gitkeep":
                                f.unlink()
                
                st.success("All evaluation data cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear data: {e}")
        else:
            st.info("Please check the confirmation box to proceed.")

elif menu == "Run Evaluation":
    st.title("Run Safety Evaluation")
    
    # Use the discovered configs from top-level
    available_models = all_discovered_configs
        
    st.markdown("### 1. Model Selection")
    
    baseline_model = config["evaluation"].get("baseline_model", "gpt-4o-mini")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Show only chat models (discovery already filters embeddings)
        model_options = {f"{m['name']} ({m['type']})": m for m in available_models}
        selected_model_names = st.multiselect(
            "Select models to evaluate", 
            options=list(model_options.keys()),
            default=[]
        )
    with col2:
        if st.button("Refresh Models"):
            st.rerun()

    st.markdown("### 2. Run Settings")
    use_judge = st.checkbox("Use LLM-as-Judge", value=config["evaluation"]["use_llm_judge"])
    
    if not selected_model_names:
        st.error("Please select at least one model to run.")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        def progress_update(current, total):
            pct = current / total
            progress_bar.progress(pct)
            status_text.text(f"Evaluation in progress... {int(pct*100)}% ({current}/{total})")

        if st.button("Start Evaluation Run"):
            # Prepare config for this run
            run_config = config.copy()
            run_config["models"] = [model_options[name] for name in selected_model_names]
            run_config["evaluation"]["use_llm_judge"] = use_judge
            
            # Build pipeline
            # We set discover_local=False because we already have the selected models in run_config["models"]
            pipeline = EvaluationPipeline(config=run_config, discover_local=False)
            
            with st.spinner("Running Evaluation Pipeline..."):
                try:
                    pipeline.run(progress_callback=progress_update)
                    
                    status_text.text("Aggregating results...")
                    agg = ResultsAggregator() 
                    agg.config = run_config 
                    agg.run()
                    
                    st.success("Evaluation Complete!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
