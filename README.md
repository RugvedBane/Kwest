
# Kwest

**A personal AI agent for automating Kaggle competition workflows.**

Kwest is a tool-using AI agent that automates key stages of the Kaggle competition workflow, from inspecting datasets and generating machine learning code to executing training on Kaggle kernels, submitting predictions, and tracking experiments.

Built with Python, LangChain, LangGraph, and DeepSeek, Kwest provides a conversational interface for managing competition experiments without repeatedly handling the same setup and execution steps manually.

---

## Why Kwest?

Iterating on Kaggle competitions involves repeatedly inspecting datasets, writing training scripts, launching kernels, checking execution results, and submitting predictions.

Kwest brings these operations into a single agent-driven workflow.

- **Automated execution:** Generates training code and runs it on Kaggle kernels.
- **Tool-based interaction:** Uses dedicated tools to inspect data, retrieve competition information, analyze features, execute code, and submit predictions.
- **Experiment tracking:** Maintains a persistent record of experiment descriptions, scores, and observations.
- **Competition isolation:** Maintains separate Kaggle kernels and solution files for each competition.
- **Human-in-the-loop workflow:** Keeps the user involved in reviewing generated code, evaluating results, and deciding what to experiment with next.

The goal is to reduce repetitive work and make competition iteration more efficient while keeping modeling decisions under the user's control.

---

## Architecture

Kwest follows a tool-calling agent architecture. The LLM interprets user requests, selects the appropriate tools, and uses their results to formulate responses.

```mermaid

flowchart TD
    U(["User"]) <--> UI["Streamlit Frontend"]

    subgraph AGENT["Agent Layer"]
        A["LangChain + LangGraph<br/>ReAct Agent"]
        LLM["DeepSeek V4.1 Flash<br/>via Rapid Router"]
        MEM["Experiment Memory<br/>memory.py"]
        DB[("experiments.json")]

        A <--> LLM
        A <--> MEM
        MEM <--> DB
    end

    UI <--> A

    subgraph TOOLS["Tool Layer · agent_tools.py"]
        T1["Data Inspection & Feature Analysis"]
        T2["Competition Info, Leaderboard<br/>& Submission History"]
        T3["Current Solution Retrieval"]
        T4["Kaggle Kernel Execution"]
        T5["Prediction Submission"]
    end

    A <--> TOOLS

    subgraph KAGGLE["Kaggle Infrastructure"]
        DATA[("Competition Data")]
        KERNEL["Per-Competition Kernel"]
        OUTPUT["Kernel Output<br/>submission.csv"]
        BOARD["Competition Leaderboard"]
    end

    T1 <--> DATA
    T2 <--> BOARD
    T3 <--> KERNEL
    T4 <--> KERNEL
    KERNEL --> OUTPUT
    OUTPUT --> T5
    T5 --> BOARD
```

### Core Components

#### 1. Agent and Reasoning Layer

- Built using LangChain and LangGraph's `create_react_agent`.
- Uses DeepSeek V4.1 Flash through Rapid Router as the LLM.
- Interprets natural-language requests and selects tools based on their descriptions and system instructions.
- Uses controlled tool execution to avoid unnecessary parallel calls and repeated kernel submissions.
- Maintains a limited conversation history to provide context across interactions.

#### 2. Tool Layer

Implemented in `agent_tools.py`, these tools give the agent access to Kaggle competition operations and local experiment data.

| Tool | Responsibility |
|---|---|
| `inspect_competition_data` | Downloads and profiles competition datasets, including shapes, data types, and missing values. |
| `analyze_feature_importance` | Performs local LightGBM-based feature analysis using gain importance, correlation, mutual information, and categorical target rates. |
| `get_competition_info_tool` | Retrieves competition metadata and related information. |
| `get_leaderboard_tool` | Fetches competition leaderboard information. |
| `get_submissions_tool` | Retrieves submission history and scores. |
| `get_current_solution` | Reads the existing solution before modifying it. |
| `run_on_kaggle` | Writes the training script, pushes a Kaggle kernel, monitors execution, and retrieves output logs. |
| `submit_to_competition` | Submits the generated prediction file to the competition. |

#### 3. Kaggle Execution Layer

- Uses the Kaggle CLI through Python's `subprocess` module.
- Generates and pushes Python training scripts as Kaggle kernels.
- Monitors kernel execution and retrieves relevant logs.
- Uses a separate kernel for each competition, with dynamically generated kernel identifiers.
- Stores competition-specific solution files and kernel metadata locally.
- Retrieves the generated `submission.csv` for submission.

#### 4. Experiment Memory

Implemented through `memory.py` and `experiments.json`.

- Stores experiment descriptions, scores, and notes.
- Keeps experiment records scoped to individual competitions.
- Loads relevant experiment history into the agent's context.
- Allows the user to log results directly through chat.

Example command:

```text
log: LightGBM with interaction features | 0.94706 | Improved public leaderboard score
```

#### 5. Streamlit Frontend

The Streamlit interface provides a centralized workspace for interacting with Kwest.

- **Chat:** Conversational interface for requesting competition operations.
- **Kwest's Code:** Displays the generated solution code and kernel information.
- **Experiments:** Displays logged experiments, scores, and observations.
- **Settings:** Manages the selected competition.
- **About:** Provides project and workflow information.

The interface uses a custom green visual theme aligned with Kwest's branding.

---

## Typical Workflow

1. **Configure a competition:** Enter a Kaggle competition slug in Settings.
2. **Inspect the data:** Ask Kwest to download and analyze the competition dataset.
3. **Analyze features:** Use feature importance analysis to identify potentially useful predictors.
4. **Generate and execute code:** Ask Kwest to develop a model. The agent generates a training script, pushes it to a Kaggle kernel, and monitors execution.
5. **Review results:** Inspect the generated solution, kernel status, and execution logs.
6. **Submit predictions:** Ask Kwest to submit the generated `submission.csv`.
7. **Track experiments:** Log important runs and compare results over time.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Agent Framework | LangChain, LangGraph |
| LLM | DeepSeek V4.1 Flash |
| LLM Provider | Rapid Router |
| Tool Execution | LangChain Tools, Python subprocess |
| Kaggle Integration | Kaggle API, Kaggle CLI |
| Machine Learning | LightGBM, XGBoost, CatBoost |
| Data Analysis | Pandas, NumPy, scikit-learn |
| Frontend | Streamlit |
| Experiment Storage | JSON |
| Configuration | python-dotenv |

---

## Project Structure

```text
Kwest/
│
├── .streamlit/                 # Streamlit configuration and secrets
│
├── solutions/
│   └── <competition-slug>/
│       ├── solution.py         # Competition training script
│       ├── kernel-metadata.json
│       └── output/             # Generated kernel outputs
│
├── .gitignore
├── Kwest.py                    # Agent construction and system instructions
├── README.md
├── agent_tools.py              # Kaggle and ML tools
├── experiments.json            # Persistent experiment history
├── main.py                     # Kaggle CLI operations
├── memory.py                   # Experiment memory management
├── requirements.txt
└── streamlit_app.py            # Streamlit frontend
```

Competition datasets and generated training artifacts are maintained locally and excluded from version control.

---

## Engineering Decisions

A few implementation decisions were made to improve reliability and keep the architecture manageable.

- **Dedicated competition kernels:** Each competition gets its own kernel instead of sharing a single kernel across unrelated competitions.
- **On-demand competition context:** Competition metadata, leaderboard information, and submission history are retrieved through tools rather than loading everything into the agent's initial context.
- **Controlled tool execution:** Parallel tool calls are disabled to reduce conflicting or duplicated execution requests.
- **Bounded execution history:** Conversation history and returned kernel logs are limited to prevent unnecessary context growth.
- **Error handling:** The agent is instructed to report execution failures and timeouts rather than automatically retrying kernel runs.
- **Existing solution awareness:** The agent can retrieve the current solution before making modifications.
- **Persistent experiment tracking:** Experiment history is stored locally and remains available across sessions.

---

## Current Status

**Functional prototype under active development.**

Implemented and tested:

- LangChain and LangGraph agent with tool calling.
- Eight competition-related tools.
- Competition dataset inspection and feature importance analysis.
- Per-competition Kaggle kernel creation and execution.
- Kernel status monitoring and output retrieval.
- Automated prediction submission.
- Persistent experiment logging and retrieval.
- Streamlit-based conversational frontend.

### End-to-End Results

Kwest has successfully completed the following workflow on the Kaggle Playground Series S6E11 competition:

`Dataset Inspection → Model Training → Prediction Generation → Kaggle Submission`

The project achieved a public leaderboard score of **0.94706**, improving on the previous manually achieved score of **0.94579**.

This is an individual competition result, not a claim of consistent leaderboard performance across competitions.

---

## Future Improvements

The current priority is improving the reliability of existing functionality rather than continuously expanding the toolset.

Planned areas of improvement:

- More reliable code generation and modification.
- Better validation of tool outputs and execution results.
- Improved conversation and experiment context management.
- More robust kernel monitoring and failure handling.
- Error analysis using out-of-fold predictions.
- Deployment through Streamlit Community Cloud.
- Automated code-quality checks through GitHub Actions.

---

## About

Kwest is a personal project built to explore practical applications of agentic AI, tool calling, LLM-driven code generation, and automated machine learning workflows.

It is designed around a real Kaggle workflow, with the aim of making experimentation more efficient while retaining user oversight over the modeling process.
