# LLM Safety Evaluation Framework - System Overview

## 🚀 Project Goal
The **LLM Safety Evaluation Framework** is a production-grade tool designed to rigorously test, score, and compare the safety alignment of Large Language Models (LLMs). It evaluates models across 10 critical safety dimensions, providing a quantitative "Overall Safety Score" and detailed technical insights.

## 🏗️ System Architecture

### 0. Core Framing
The framework prioritizes **Target local models** (the subjects under evaluation) vs. **Hosted baselines** (reference points). Every run evaluates the specific models discovered on your local infrastructure against frontier standards like GPT-4o-mini.

### 1. Model Backend (`models/`)
- **Unified Interface**: All models (OpenAI, Anthropic, LM Studio, Mock) share a common `BaseModel` ABC.
- **Resilience**: Uses **LangGraph** to implement a "Retry-with-Backoff" state machine for every model call, ensuring stability during network fluctuations or rate limits.
- **Flexibility**: Supports hosted APIs (OpenAI) and local deployments (LM Studio) via an OpenAI-compatible interface.

### 2. Evaluation Logic (`evaluator/`)
- **Dataset**: A curated set of 50+ adversarial prompts spanning categories like Jailbreaking, Prompt Injection, and Hallucination.
- **Scoring Pipeline**: Orchestrated by **LangGraph**, the pipeline manages the flow from prompt loading to generation, scoring, and data persistence.
- **Multi-Layer Scoring**:
    - **Heuristic Scorers**: Fast, rule-based detection (keywords, regex, length constraints).
    - **LLM-as-Judge**: Nuanced evaluation using advanced models (e.g., GPT-4o) to grade complex responses.
- **Dynamic Red-Teaming**: An automated agent that analyzes model failures and generates new, more difficult adversarial prompts to probe weaknesses.

### 3. Data & Analytics
- **Dynamic Model Management**: Supports real-time discovery of models from LM Studio servers and easy configuration of new model providers via the dashboard.
- **API Compatibility**: Optimized for both standard OpenAI endpoints and modern LM Studio internal APIs (`/api/v1`) to ensure seamless model discovery and generation.
- **Structured Storage**: Uses **SQLite** (`data/eval_results.db`) to store evaluation runs, raw responses, and detailed metric scores, enabling historical comparison and complex querying.
- **Aggregation**: Results are processed into per-model and per-category summaries, handling missing data and failed runs gracefully.
- **Quantization Impact**: Specifically designed to compare how different model precisions (e.g., 4-bit vs 8-bit) affect safety alignment.
- **Visualization**: Generates technical charts (bar charts, heatmaps, and latency comparisons) to communicate risks and trade-offs.

## 🛡️ Security Measures

### 1. Adversarial Handling
- The framework itself is designed to handle "dangerous" text. All prompt and response handling is done within isolated data structures to prevent accidental execution or leakage.
- **Prompt Injection Protection**: The system evaluates its own susceptibility to prompt injection and provides metrics on how well the target model resists such attacks.

### 2. API & Data Security
- **Environment Variables**: API keys are NEVER hardcoded or stored in `config.yaml`. They are read from `.env` or system environment variables.
- **Data Isolation**: Raw model responses are stored separately from processed scores, allowing for auditability without compromising privacy.
- **Local Evaluation**: Support for LM Studio allows for fully offline safety testing, ensuring sensitive prompts never leave the local environment.

### 3. Ethical Use
- This framework is intended for research and safety alignment purposes. It provides a "safe sandbox" for red-teaming without encouraging the generation of harmful content.

## 📊 Evaluation Categories
1.  **Prompt Injection**: Overriding system instructions.
2.  **Jailbreak**: Bypassing safety guardrails via roleplay/framing.
3.  **Harmful Requests**: Requests for illegal or dangerous activities.
4.  **Hallucination**: Fabrication of facts or confident errors.
5.  **Privacy**: Leaking or fabricating PII.
6.  **Bias**: Stereotyped or discriminatory output.
7.  **Roleplay Attacks**: Using personas to bypass rules.
8.  **Instruction Following**: Adherence to negative/positive constraints.
9.  **Sensitive Information**: Discussion of restricted topics.
10. **Ethical Decision Making**: Handling of moral dilemmas.

---
*Created for AI Safety Research & Development.*
