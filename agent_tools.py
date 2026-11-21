import os
import json
import time
import subprocess
import tempfile
import shutil
import pandas as pd
import zipfile
from langchain_core.tools import tool
from main import get_my_submissions, get_competition_info, get_leaderboard, get_my_submissions
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

COMP_SLUG = ""
LAST_OUTPUT_FOLDER = ""
DATA_FOLDER = ""

def set_comp_slug(slug: str):
    global COMP_SLUG
    COMP_SLUG = slug


@tool
def run_on_kaggle(code: str = None) -> str:
    """
    Pushes code to Kaggle, waits for actual kernel completion,
    downloads fresh outputs and verifies submission.csv.

    Pass code normally to push a new/edited solution.
    Omit code (call with no argument) to re-push the existing
    solution.py unchanged — use this after manually cancelling
    a stuck/queued kernel on kaggle.com and wanting to just retry
    the same run.
    """

    global LAST_OUTPUT_FOLDER

    LAST_OUTPUT_FOLDER = ""

    kernel_slug = f"Kwest-{COMP_SLUG}"
    kernel_ref = f"rugvedbane/{kernel_slug}"

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pre = subprocess.run(
    ["kaggle", "kernels", "status", kernel_ref],
    capture_output=True, text=True, encoding="utf-8",
    errors="replace", timeout=120
    )
    pre_text = (pre.stdout + pre.stderr).upper()
    if pre.returncode == 0 and any(s in pre_text for s in ("QUEUED", "RUNNING", "CANCELLING")):
        return (
            "Previous version of this kernel is still queued/running/cancelling.\n"
            "Stop that session on kaggle.com first, then rerun."
        )

    folder = os.path.join(base_dir, "solutions", COMP_SLUG)
    output_folder = os.path.join(folder, "output")  

    os.makedirs(folder, exist_ok=True)

    # Remove previous execution output
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Save generated solution
    code_path = os.path.join(folder, "solution.py")

    if code is not None:
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        if not os.path.isfile(code_path):
            return "No existing solution.py found to rerun — provide code."
        print("No code provided — re-pushing existing solution.py as-is.")

    # Kernel metadata
    
    metadata = {
        "id": f"rugvedbane/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "solution.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMP_SLUG],
        "kernel_sources": []
    }

    meta_path = os.path.join(folder, "kernel-metadata.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Push kernel
    push_result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", folder],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180
    )

    if push_result.returncode != 0:
        return (
            f"KERNEL PUSH FAILED\n"
            f"{push_result.stderr}\n{push_result.stdout}"
        )

    print("Kernel pushed successfully. Monitoring execution...")

    print("Kernel pushed. Waiting 30s for status to refresh...")
    time.sleep(30)

    start_time = time.time()
    timeout = 2700
    poll_interval = 15
    queue_limit = 150  
    terminal_status = None
    running_seen = False

    while time.time() - start_time < timeout:
        status_result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60
        )
        if status_result.returncode != 0:
            return f"Unable to retrieve Kaggle kernel status.\n{status_result.stderr}"

        status_text = (status_result.stdout + "\n" + status_result.stderr).strip()
        status_upper = status_text.upper()
        elapsed_now = int(time.time() - start_time)

        status_line = next((l for l in status_text.splitlines() if "has status" in l), status_text)
        print(f"[{elapsed_now}s] {status_line}") 

        if "RUNNING" in status_upper:
            running_seen = True

        if "COMPLETE" in status_upper:
            terminal_status = "COMPLETE"; break
        if "ERROR" in status_upper:
            terminal_status = "ERROR"; break
        if "CANCEL" in status_upper:
            terminal_status = "CANCELLED"; break

        if not running_seen and elapsed_now > queue_limit:
            return (
                f"KERNEL STUCK IN QUEUE for {elapsed_now}s and never started.\n"
                "Cancel it on kaggle.com (stop the session), then ask me to "
                "rerun with no code to re-push the same solution."
            )

        time.sleep(poll_interval)

    elapsed = round(time.time() - start_time, 1)

    if terminal_status is None:
        return (
            f"KERNEL TIMEOUT after {elapsed} seconds.\n"
            "Execution status was never confirmed as complete.\n"
            "Check Kaggle manually before retrying."
        )

    if terminal_status != "COMPLETE":
        return (
            f"KERNEL {terminal_status}.\n"
            f"Execution status: {status_text}\n"
            "No submission will be attempted."
        )

    print(f"Kernel completed successfully in {elapsed} seconds.")

    # Download fresh output only after completion
    output_result = subprocess.run(
        [
            "kaggle", "kernels", "output",
            kernel_ref,
            "-p", output_folder
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180
    )

    if output_result.returncode != 0:
        return (
            "KERNEL COMPLETED, BUT OUTPUT DOWNLOAD FAILED.\n"
            f"{output_result.stderr}"
        )

    files = os.listdir(output_folder)

    if not files:
        return (
            "KERNEL COMPLETED, BUT NO OUTPUT FILES WERE DOWNLOADED.\n"
            f"Download message: {output_result.stdout}"
        )

    LAST_OUTPUT_FOLDER = output_folder

    submission_path = os.path.join(
        output_folder, "submission.csv"
    )

    if os.path.isfile(submission_path):
        size = os.path.getsize(submission_path)

        if size == 0:
            submission_status = "submission.csv exists but is EMPTY."
        else:
            submission_status = (
                f"submission.csv FOUND. Size: {size:,} bytes."
            )
    else:
        submission_status = "submission.csv NOT FOUND."

    output_text = []

    for filename in files:
        filepath = os.path.join(output_folder, filename)

        if not os.path.isfile(filepath):
            continue

        size = os.path.getsize(filepath)

        if filename.lower().endswith(".csv"):
            output_text.append(
                f"{filename}: downloaded, {size:,} bytes."
            )
            continue

        if filename.lower().endswith(
            (".txt", ".log", ".json", ".out")
        ):
            with open(
                filepath, "r",
                encoding="utf-8",
                errors="replace"
            ) as f:
                content = f.read()

            output_text.append(
                f"\n--- {filename} ---\n{content}"
            )

    return (
        f"KERNEL STATUS: COMPLETE\n"
        f"EXECUTION TIME: {elapsed} seconds\n"
        f"{submission_status}\n\n"
        f"DOWNLOADED FILES:\n"
        + "\n".join(files)
        + "\n\nOUTPUT DETAILS:\n"
        + "\n".join(output_text)
    )

@tool
def submit_to_competition(message: str) -> str:
    """
    Validates submission.csv against the competition's sample_submission.csv
    and submits it to the current Kaggle competition after user approval.
    """
    folder = os.path.dirname(os.path.abspath(__file__))

    output_folder = os.path.join(
        folder, "solutions", COMP_SLUG, "output"
    )

    submission_path = os.path.join(
        output_folder, "submission.csv"
    )

    sample_path = os.path.join(
        folder, "comp_data", COMP_SLUG, "sample_submission.csv"
    )

    if not os.path.isfile(submission_path):
        return f"SUBMISSION FAILED: submission.csv not found at {submission_path}"

    if not os.path.isfile(sample_path):
        return f"SUBMISSION FAILED: sample_submission.csv not found at {sample_path}"

    if os.path.getsize(submission_path) == 0:
        return "SUBMISSION FAILED: submission.csv is empty."

    try:
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_path)
    except Exception as e:
        return f"SUBMISSION FAILED: Unable to read CSV files.\n{e}"

    if list(submission.columns) != list(sample.columns):
        return (
            "SUBMISSION FAILED: Column mismatch.\n"
            f"Expected: {list(sample.columns)}\n"
            f"Found: {list(submission.columns)}"
        )

    if len(submission) != len(sample):
        return (
            "SUBMISSION FAILED: Row count mismatch.\n"
            f"Expected: {len(sample):,}\n"
            f"Found: {len(submission):,}"
        )

    if submission.isna().any().any():
        return "SUBMISSION FAILED: Missing values detected."

    if submission.empty:
        return "SUBMISSION FAILED: No predictions found."

    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "submit",
                "-c", COMP_SLUG,
                "-f", submission_path,
                "-m", message
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180
        )
    except subprocess.TimeoutExpired:
        return (
            "SUBMISSION STATUS UNKNOWN: Upload timed out. "
            "Check Kaggle submissions before retrying."
        )

    if result.returncode != 0:
        return (
            f"SUBMISSION FAILED.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return (
        f"SUBMISSION UPLOAD ACCEPTED.\n"
        f"Competition: {COMP_SLUG}\n"
        f"Rows: {len(submission):,}\n"
        f"Message: {message}\n\n"
        f"Kaggle response:\n{result.stdout}\n"
        "Check Kaggle submissions for evaluation status."
    )


@tool
def inspect_competition_data() -> str:
    """
    Downloads competition data, inspects train.csv and test.csv,
    and returns a full data profile including features, dtypes,
    missing values, stats and sample rows for the LLM to use.
    """

    global DATA_FOLDER

    DATA_FOLDER = os.path.join(os.getcwd(), "comp_data", COMP_SLUG)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # now download the data
    download_result = subprocess.run(
        ['kaggle', 'competitions', 'download',
         '-c', COMP_SLUG,
         '-p', DATA_FOLDER],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if download_result.returncode != 0:
        return f"Download Failed: {download_result}"

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('zip'):
            zip_path = os.path.join(DATA_FOLDER, filename)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_FOLDER)

    profile = ""

    for csv_name in ['train.csv', 'test.csv']:
        csv_path = os.path.join(DATA_FOLDER, csv_name)

        if not os.path.exists(csv_path):
            profile += f"\n{csv_name}: not found\n"
            continue

        df = pd.read_csv(csv_path)

        profile += f"\n{'='*20} {csv_name} {'='*20}\n"
        profile += f"Shape: {df.shape}\n"
        profile += f"\nDtypes:\n{df.dtypes.to_string()}\n"
        profile += f"\nMissing Values:\n{df.isnull().sum().to_string()}\n"
        profile += f"\nBasic Stats:\n{df.describe().to_string()}\n"
        profile += f"\nFirst 3 rows:\n{df.head(3).to_string()}\n"

        result = profile


    return result[:3000] if result else "No CSV files found in the competition data"


@tool 
def get_submissions() -> str:
    """
    Fetches your latest submissions for the current competition
    including their public leaderboard scores.
    """
    return get_my_submissions(COMP_SLUG)


@tool
def get_competition_info_tool() -> str:
    """
    Fetches details about the current competition —
    description, evaluation metric, deadline, and rules.
    """
    return get_competition_info(COMP_SLUG)


@tool
def get_leaderboard_tool() -> str:
    """
    Fetches the current public leaderboard for the competition.
    """
    return get_leaderboard(COMP_SLUG)

@tool
def get_current_solution() -> str:
    """
    Reads the current solution.py for this competition.
    Call this before making changes to existing code.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    solution_path = os.path.join(base_dir, "solutions", COMP_SLUG, "solution.py")

    if not os.path.exists(solution_path):
        return "No solution file exists for this competition yet."

    with open(solution_path, "r", encoding='utf-8') as f:
        return f.read()

def detect_target_column() -> str:
    """
    Auto-detects the target column: the column present in train.csv but
    absent from test.csv. Falls back to sample_submission.csv's non-id
    column if train/test diff doesn't resolve to exactly one column.
    Internal helper — not exposed to the agent as a tool.
    """
    train_path = os.path.join(DATA_FOLDER, "train.csv")
    test_path = os.path.join(DATA_FOLDER, "test.csv")
    sample_path = os.path.join(DATA_FOLDER, "sample_submission.csv")

    train_cols = set(pd.read_csv(train_path, nrows=0).columns)
    diff = set()

    if os.path.exists(test_path):
        test_cols = set(pd.read_csv(test_path, nrows=0).columns)
        diff = train_cols - test_cols
        if len(diff) == 1:
            return diff.pop()

    if os.path.exists(sample_path):
        sample_cols = list(pd.read_csv(sample_path, nrows=0).columns)
        candidates = [c for c in sample_cols if c in train_cols and c.lower() != "id"]
        if len(candidates) == 1:
            return candidates[0]

    raise ValueError(
        "Could not auto-detect target column — train/test diff and "
        f"sample_submission.csv didn't agree. Diff candidates: {diff}"
    )

@tool
def analyze_feature_importance() -> str:
    """
    Runs a quick analysis of which features actually relate to the target:
    LightGBM feature importance, numeric correlation with target, mutual
    information (captures nonlinear relationships), and target rate by
    category for top categorical columns. Meant for guiding feature
    engineering / ablation decisions, not for the final model.
    Call inspect_competition_data first if comp_data hasn't been downloaded yet.
    """

    csv_path = os.path.join(DATA_FOLDER, "train.csv")

    if not os.path.exists(csv_path):
        return "train.csv not found locally — run inspect_competition_data first."

    try:
        target_col = detect_target_column()
    except ValueError as e:
        return str(e)

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        return f"Detected target '{target_col}' not actually in train.csv columns: {list(df.columns)}"

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    drop_cols = [target_col]
    if 'id' in X.columns:
        X = X.drop(columns=['id'])

    X_enc = X.copy()
    for col in cat_cols:
        X_enc[col] = X_enc[col].astype('category').cat.codes

    sample_size = min(150_000, len(X_enc))
    sample_idx = X_enc.sample(sample_size, random_state=42).index
    X_sample, y_sample = X_enc.loc[sample_idx], y.loc[sample_idx]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=63,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    importances = pd.Series(
        model.booster_.feature_importance(importance_type='gain'),
        index=X_enc.columns
    ).sort_values(ascending=False)

    result = f"QUICK MODEL VALID AUC: {roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]):.5f}\n"
    result += "(sampled model for ranking only — not representative of final score)\n\n"

    result += "=== FEATURE IMPORTANCE (gain, top 20) ===\n"
    result += importances.head(20).to_string() + "\n\n"

    if num_cols:
        corr = df[num_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        result += "=== NUMERIC CORRELATION WITH TARGET (top 15 by |corr|) ===\n"
        result += corr.head(15).to_string() + "\n\n"

    mi = mutual_info_classif(X_sample, y_sample, discrete_features=[c in cat_cols for c in X_enc.columns], random_state=42)
    mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)
    result += "=== MUTUAL INFORMATION WITH TARGET (top 15) ===\n"
    result += mi_series.head(15).to_string() + "\n\n"

    if cat_cols:
        result += "=== TARGET RATE BY CATEGORY (top 5 categorical cols by MI) ===\n"
        top_cats = [c for c in mi_series.index if c in cat_cols][:5]
        for col in top_cats:
            rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            result += f"\n{col}:\n{rates.head(10).to_string()}\n"

    return result[:8000]