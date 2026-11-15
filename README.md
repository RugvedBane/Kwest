# Kwest

Kwest is a personal AI agent that automates the end-to-end Kaggle competition workflow: inspecting data, writing model code, training on Kaggle's GPU kernels, generating submissions, and tracking experiment results, all from a conversational interface.

## Why this exists

Manually re-running feature engineering, training loops, and submission scripts for every competition iteration is slow. Kwest handles the repetitive parts so iteration speed goes up, while the strategic decisions (what to try next, where to look for signal) stay with the user.

## Architecture

**Agent layer (LangChain and LangGraph)**
The core agent is built on LangChain's tool-calling framework with LangGraph managing the agent's reasoning loop. The agent decides which tool to call based on the user's request: inspecting data, checking competition metadata, running training code, or submitting predictions.

**LLM**
DeepSeek V4.1 Flash, accessed through Rapid Router, acts as the reasoning engine behind the agent. It plans actions, writes Python training code, and interprets tool results.

**Tool layer (agent_tools.py)**
A set of functions exposed to the agent as callable tools, including:
- Fetching and inspecting competition data (shapes, dtypes, missing values)
- Fetching competition metadata (deadline, metric, leaderboard, entry status)
- Writing and pushing training scripts to Kaggle as kernels
- Polling kernel status until training completes or fails
- Downloading the resulting submission file
- Submitting predictions to the competition leaderboard

**Kaggle execution (per competition kernel naming)**
Each competition gets its own Kaggle kernel, named dynamically from the competition slug, instead of sharing one kernel across all competitions. This keeps version history and logs scoped per competition rather than mixed together.

**Experiment memory (memory.py and experiments.json)**
Every meaningful run is logged with a description, score, and notes, scoped per competition. This creates a running history the agent and the user can both reference when deciding what to try next.

**Frontend (Streamlit)**
A Streamlit app provides the chat interface, a live view of the generated solution code and kernel status, an experiment log viewer, and settings for switching between competitions. The UI is styled to resemble modern AI chat interfaces rather than Streamlit's default look.

## Typical workflow

1. Set a competition slug in Settings (for example, `playground-series-s6e11`).
2. Ask Kwest to inspect the data or check the competition details.
3. Ask Kwest to write and run a model. It generates training code, pushes it to a Kaggle kernel, and monitors execution.
4. Once a kernel completes, Kwest can submit the resulting predictions to the leaderboard.
5. Log notable results with `log: description | score | notes` to build experiment history.

## Tech stack

- Python
- LangChain and LangGraph
- DeepSeek V4.1 Flash, via Rapid Router
- Kaggle API and Kaggle CLI
- LightGBM, XGBoost, CatBoost for modeling
- Streamlit for the interface

## Current status

Under active development. Core agent loop, per-competition kernel execution, and experiment logging are functional. Planned additions include open-ended EDA support, a notebook-output mode for publishable public kernels, and broader model type support beyond gradient boosted trees.
