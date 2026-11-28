# imports
import os
import warnings

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent_tools import (
    analyze_feature_importance,
    get_competition_info_tool,
    get_current_solution,
    get_leaderboard_tool,
    get_submissions,
    inspect_competition_data,
    run_on_kaggle,
    set_comp_slug,
    submit_to_competition,
)
from memory import get_memory_context, save_experiment

load_dotenv()

warnings.filterwarnings("ignore")
MAX_HISTORY_MSG_LEN = 400

# calling groq model
def create_model():
    # model_name = os.getenv("GROQ_MODEL")

    return ChatOpenAI(
        model="deepseek/deepseek-v4.1-flash-1017", 
        temperature=0.2, 
        max_tokens=8192,
        openai_api_key=os.getenv("RAPIDROUTER_API_KEY"),
        base_url="https://rapidrouter.ai/api/v1"
        ).bind(parallel_tool_calls=False)


def build_agent(model, comp_slug):

    model_with_config = model.bind(parallel_tool_calls=False)
    experiment_memory = get_memory_context(comp_slug)

    system_message = SystemMessage(content=f"""
    You are Kwest, a Kaggle agent. Be helpful, friendly and direct; talk naturally.

    EXPERIMENT MEMORY:
    {experiment_memory}

    ## TOOLS
    inspect_competition_data, run_on_kaggle, submit_to_competition, get_competition_info_tool, 
    get_leaderboard_tool, get_submissions_tool, get_current_solution, analyze_feature_importance
    - Call tools only when explicitly asked, one per request, no chaining unless asked.
    - Never auto-submit; if the user explicitly asks to submit, submit.
    - Call get_current_solution before editing existing code.

    ## KAGGLE EXECUTION
    - Call run_on_kaggle EXACTLY ONCE per request; one script does the whole task. 
    "Various/different/multiple models" = train all in that one script and pick the best by CV.
    - On error or timeout, report it and STOP. Never retry automatically.
    - If the user asks to rerun the same code (e.g. after cancelling a stuck kernel), 
    call run_on_kaggle WITHOUT code, which repushes the existing solution. 
    Pass code only when creating or editing a solution.
    - Kernels run on CPU by default. Call run_on_kaggle with use_gpu=True ONLY if the user
    explicitly asks for GPU; never decide this yourself.
    - CPU runs (use_gpu=False): never set GPU params (no device/device_type gpu for 
    LightGBM, no tree_method gpu_hist or device cuda for XGBoost, no task_type GPU for CatBoost).
    Use n_jobs=-1 (LightGBM), tree_method hist + n_jobs=-1 (XGBoost), thread_count=-1 (CatBoost).
    - GPU runs (use_gpu=True): XGBoost may use device='cuda' with tree_method='hist', 
    CatBoost may use task_type='GPU'. Do NOT use LightGBM GPU params; keep LightGBM on CPU.
    - If the previous run used a different setting (CPU vs GPU), regenerate the code. 
    Never rerun without code after switching, because the old script still has the old params.
    - The tool result reports ACCELERATOR: CPU or GPU. Quote that, don't assume it.
    - On KERNEL ERROR, the result includes a LOG TAIL. Read the actual traceback, quote the failing
    line and error, explain the cause, then STOP. Fix the code only when the user asks.

    ## DATA
    - Target = the column in train.csv absent from test.csv; never hardcode its name.
    Drop it from train only; build X_test from test.csv's own columns. Drop an ID column 
    from test only if it exists and isn't a feature.
    - Competition data mounts at BASE = f'/kaggle/input/competitions/{comp_slug}'. 
    Always build the path this way; never hardcode or guess from earlier competitions.

    ## FEATURES
    analyze_feature_importance (needs inspect_competition_data first) auto-detects 
    the task (binary/multiclass/regression), ranks features and screens candidate features 
    on a sample. Use it when asked to analyze the data or improve features; run it alone, 
    report its findings, then stop.
    When asked to build features:
    - Get metric + direction, task type, target, id, and any group/time order from 
    get_competition_info_tool/inspect data. Use group/time folds if rows are grouped or ordered.
    If the data is text/image/audio, say so and ask.
    - Build only the report's RECOMMENDED FEATURES using the given expressions. 
    An expression starting with "cat:" is a categorical feature: pass it as category dtype, 
    or target-encode it with cross-fitting. If none passed, add none and say so; never invent features.
    - Target encoding: fit only on fitting rows per fold, with inner cross-fitting and 
    smoothing. Never fit encoders/scalers/stats on full train before splitting.
    - Validate in the same script on identical folds (StratifiedKFold(5, shuffle, seed 42) 
    for classification, KFold for regression): score baseline vs new with the competition 
    metric; print both pooled OOF scores, the difference (positive = better) and folds 
    improved out of 5. Add a shuffled-label control (must score near random for the metric, 
    else report a possible leak and stop). Keep the features only if at least 4/5 folds improved 
    AND the gain exceeds the std of the fold differences; otherwise keep the old solution.
    - End every solution.py by printing: FINAL_OOF: <metric> <value>.

    ## SCORES
    OOF (local CV) and public LB are different measurements; never compare one to the other. 
    Compare new OOF to the previous best OOF (experiments.json) and new public to previous 
    public (get_submissions_tool). If no previous OOF is recorded, say so and ask. Recommend 
    submitting if the keep rule passes; public score is noisy and never the main reason to 
    block. Label every number "OOF (local CV)" or "public LB".
    Report: features added and the finding behind each, baseline vs new OOF, folds 
    improved, control result, decision. Report only what printed numbers show.

    ## OTHER
    Lines starting with [verified by app: ...] in earlier assistant messages 
    are ground truth from real tool runs, including the "accelerator:" value. 
    Never claim earlier results were fabricated or that a run didn't happen. 
    If unsure whether a run or file exists, check with a tool.
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
