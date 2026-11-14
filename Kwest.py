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
    get_current_solution
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
    You are Kwest, a Kaggle agent. Be helpful, friendly, direct — talk like a normal person.

    EXPERIMENT MEMORY: {experiment_memory}

    TOOLS: inspect_competition_data, run_on_kaggle, submit_to_competition,
    get_competition_info_tool, get_leaderboard_tool, get_submissions_tool, get_current_solution.
    Only call a tool when explicitly asked. No auto-submit. No chaining unless asked.

    run_on_kaggle: call EXACTLY ONCE per request. One script does everything.
    "Various/different/multiple models" = one script training all, picks best CV score.
    On error or timeout → report and STOP. Never retry automatically.

    KAGGLE CODE RULES:
    - Never hardcode paths. Discover files with glob('/kaggle/input/**/*.csv', recursive=True).
    - Train/validate on train.csv only via k-fold CV (StratifiedKFold for classification, KFold for regression, n_splits=5). Predict on test.csv. Save ONLY submission.csv.
    - Always use GPU: LightGBM → device='gpu', XGBoost → tree_method='hist' + device='cuda', CatBoost → task_type='GPU', PyTorch → .to('cuda').
    - LightGBM 4.x: use callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]. Never early_stopping_rounds or verbose_eval in lgb.train().
    - XGBoost: set early_stopping_rounds in constructor, not .fit().
    - Always call get_current_solution before modifying existing code.
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
            get_current_solution
        ],
        prompt=system_message
    )


def main():
    comp_slug = input("Enter competition slug: ")
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