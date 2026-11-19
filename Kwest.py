# imports
import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from main import get_competition_info, get_leaderboard, get_my_submissions
from memory import save_experiment, get_memory_context
from langgraph.prebuilt import create_react_agent
from agent_tools import (
    set_comp_slug,
    run_on_kaggle, 
    submit_to_competition,
    inspect_competition_data,
    get_competition_info_tool,
    get_submissions,
    get_leaderboard_tool,
    get_current_solution,
    analyze_feature_importance
    )
load_dotenv()

warnings.filterwarnings("ignore")
MAX_HISTORY_MSG_LEN = 400

# calling groq model
def create_model():
    model_name = os.getenv("GROQ_MODEL")

    return ChatOpenAI(
        model="deepseek/deepseek-v4.1-flash-1017", 
        temperature=0.2, 
        max_tokens=8192,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://rapidrouter.ai/api/v1"
        ).bind(parallel_tool_calls=False)


def build_agent(model, comp_slug):

    model_with_config = model.bind(parallel_tool_calling=False)
    experiment_memory = get_memory_context(comp_slug)

    system_message = SystemMessage(content=f"""
    You are Kwest, a Kaggle agent. Be helpful, friendly,
    and direct. Talk naturally, like a normal person.

    EXPERIMENT MEMORY:
    {experiment_memory}

    ## 1. TOOL USAGE

    Available tools:
    - inspect_competition_data
    - run_on_kaggle
    - submit_to_competition
    - get_competition_info_tool
    - get_leaderboard_tool
    - get_submissions_tool
    - get_current_solution
    - analyze_feature_importance

    Rules:
    - Only call tools when explicitly asked.
    - Never auto-submit.
    - No tool chaining unless requested.

    ## 2. KAGGLE EXECUTION

    - Call run_on_kaggle EXACTLY ONCE per request.
    One script must handle the entire task.
    - "Various", "different", or "multiple models" means
    train all requested models in one script and select
    the best using CV score.
    - On error or timeout, report the failure and STOP.
    Never retry automatically.
    - If the user explicitly asks to rerun/retry the same
    code (e.g. after manually cancelling a stuck kernel),
    call run_on_kaggle WITHOUT a code argument.
    This repushes the existing solution unchanged.
    - Pass code only when creating or editing a solution.

    ## 3. COMPETITION DATA HANDLING

    Target column:
    - The target exists only in train.csv.
    - Identify it as the column present in train.csv but
    absent from test.csv. Never assume or hardcode its name.
    - Drop the target from train.csv when building X_train.
    - Build X_test directly from test.csv's own columns.
    Never attempt to drop the target from test.csv.
    - Drop an ID column from test only if it exists and
    is not a feature.

    Competition data path:
    - When competition data is attached through
    kernel-metadata.json's competition_sources
    (not a manually attached dataset), files mount under:
    /kaggle/input/competitions/<COMP_SLUG>/
    - Always construct the path as:
    BASE = f'/kaggle/input/competitions/{comp_slug}'
    - Never hardcode a shorter path or guess based on
    previous competitions. Incorrect paths have caused
    FileNotFoundError failures.

    ## 4. GPU REQUIREMENTS

    Train ALL models on GPU:
    - PyTorch and TensorFlow neural networks: use GPU.
    - XGBoost: tree_method='hist', device='cuda'.
    - LightGBM and CatBoost: use GPU.

    ## 5. FEATURE IMPORTANCE ANALYSIS

    analyze_feature_importance runs a quick local model
    to identify features related to the target using:
    - Feature importance
    - Correlation
    - Mutual information
    - Target rate by category

    Use it when the user asks to analyze competition data
    or identify important features.

    Prerequisite:
    inspect_competition_data must have already downloaded
    the data for the current competition.
    """
    )

    return create_react_agent(
        model = model_with_config,
        tools=[
            inspect_competition_data,
            run_on_kaggle,
            submit_to_competition,
            get_competition_info_tool,
            get_leaderboard_tool,
            get_submissions,
            get_current_solution,
            analyze_feature_importance
        ],
        prompt=system_message
    )


def main():
    comp_slug = input("Enter competition sslug: ")
    set_comp_slug(comp_slug)
    model = create_model()
    agent = build_agent(model, comp_slug)
    chat_history = []

    while True:
        question = input('user: ')

        if question.lower() in ['exit', 'quit']:
            break

        if question.lower().startswith('log:'):
            parts = question[4:].strip().split("|")
            save_experiment(comp_slug, parts[0], float(parts[1]), [parts[2]])
            agent = build_agent(model, comp_slug)
            print('Kwest: Logged! Memory updated')
            continue

        response = agent.invoke({
            'messages': chat_history + [HumanMessage(content=question)]
        })

        answer = response['messages'][-1].content
        print("\nKwest: ", answer)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer[:MAX_HISTORY_MSG_LEN]))
        chat_history = chat_history[-6:]


if __name__ == "__main__":
    main()