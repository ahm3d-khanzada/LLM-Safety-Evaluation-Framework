<<<<<<< HEAD
# LLM-Safety-Evaluation-Framework
=======
# LLM Safety Evaluation Framework

A portfolio-quality framework for evaluating the safety and alignment of Large Language Models (LLMs) across multiple risk dimensions. This project supports both hosted models (OpenAI) and local models (via LM Studio), using a robust LangGraph-powered evaluation pipeline.

## Features

- **Multi-Category Evaluation**: 50+ adversarial prompts across 10 categories (Jailbreak, Prompt Injection, Hallucination, PII Leaking, Bias, etc.).
- **Robust Pipeline**: LangGraph-powered orchestration with automated retries, exponential backoff, and graceful error handling.
- **Automated Scoring**: Heuristic and regex-based scorers for transparent, reproducible, and cost-effective grading.
- **Local & Hosted Support**: Seamless switching between OpenAI API and local LM Studio models.
- **Dynamic Discovery**: Automatically discover every model loaded on your local LM Studio server.
- **Structured Analytics**: SQLite-backed result storage for historical comparison and latency tracking.
- **Interactive Dashboard**: Streamlit-based UI for running evaluations, visualizing results, and dynamic red-teaming.
 - **LLM-as-Judge**: Nuanced safety grading using advanced models (e.g., GPT-4o, Claude 3.5 Sonnet).
- **Dynamic Red-Teaming**: Automated adversarial prompt generation based on target model failures.
- **Quantization Impact Analysis**: Compare safety alignment across different model quantization levels.

## Project Structure

```
llm-safety-eval-framework/
├── config.yaml               # Central configuration (paths, models, weights)
├── prompts/
│   └── eval_prompts.json     # 50 evaluation prompts with expected behaviors
├── evaluator/
│   ├── scorers.py            # Heuristic scoring logic
│   ├── pipeline.py           # LangGraph evaluation orchestrator
│   ├── aggregator.py         # Results summarization logic
│   └── visualize.py          # Matplotlib chart generation
├── models/                   # Model backend wrappers (Base, LangChain, Mock)
├── notebooks/
│   └── run_evaluation.ipynb  # End-to-end walkthrough
├── tests/                    # Pytest suite for scorers, pipeline, and config
├── data/                     # Raw, processed, and aggregated results
├── figures/                  # Generated visualization PNGs
└── reports/                  # Technical report and documentation
```

## Setup

1. **Clone and Install**:
   ```bash
   git clone <repo-url>
   cd llm-safety-eval-framework
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env:
   # OPENAI_API_KEY=sk-...
   # LMSTUDIO_API_KEY=lm-studio
   ```

3. **Configure Models**:
   Edit `config.yaml` to select models for evaluation. Ensure LM Studio is running if using local models.

## Usage

### Run Full Evaluation
```bash
python -m evaluator.pipeline --discover-local  # Discover local models first
python -m evaluator.aggregator
python -m evaluator.visualize
```

### Interactive Dashboard
Launch the Streamlit dashboard for a user-friendly interface:
```bash
streamlit run app.py
```
From the dashboard, you can:
- **Manage Models**: Add new OpenAI or LM Studio models dynamically without editing config files.
- **Run Evaluations**: Execute the full pipeline and monitor progress in real-time.
- **Visualize Results**: Explore safety metrics, quantization impact, and detailed scoring records.
- **Dynamic Red-Teaming**: Play with the automated adversarial prompt generator.

### Command Line
```bash
pytest tests/
```

## Scoring Methodology

The framework uses weighted averages across 10 categories to compute an **Overall Safety Score** (0-100). Safety metrics are prioritized for adversarial categories, while quality and helpfulness are considered for benign tasks.

## Results & Reporting

Aggregated results are saved to `data/results/`, and visualizations to `figures/`. A comprehensive technical analysis can be found in `reports/technical_report.md`.

## Technical Documentation
For a deep dive into the system design, security measures, and evaluation methodology, see:
- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md): Architecture and security details.
- [technical_report.md](./reports/technical_report.md): Detailed analysis of quantization impact and safety alignment results.

## Future Improvements
- Support for multi-turn adversarial dialogues.
- Integration with more local model providers (e.g., Ollama, vLLM).

---
**Note**: This project is intended for research and educational purposes in AI safety. Use adversarial prompts responsibly.
>>>>>>> 51cf6ae (Full evaluation framework)
