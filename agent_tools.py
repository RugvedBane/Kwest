import json
import os
import shutil
import subprocess
import time
import zipfile
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from main import get_competition_info, get_leaderboard, get_my_submissions

COMP_SLUG = ""
LAST_OUTPUT_FOLDER = ""
DATA_FOLDER = ""

def set_comp_slug(slug: str):
    global COMP_SLUG
    COMP_SLUG = slug

def get_kaggle_username() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.join(os.path.expanduser("~"), ".kaggle"))
    path = os.path.join(cfg_dir, "kaggle.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["username"]
    raise RuntimeError("Kaggle username not found. Set KAGGLE_USERNAME or add ~/.kaggle/kaggle.json.")

KAGGLE_USER = get_kaggle_username()

@tool
def run_on_kaggle(code: str = None, use_gpu: bool = False) -> str:
    """
    Pushes code to Kaggle, waits for actual kernel completion,
    downloads fresh outputs and verifies submission.csv.

    Pass code normally to push a new/edited solution.
    Omit code (call with no argument) to re-push the existing
    solution.py unchanged — use this after manually cancelling
    a stuck/queued kernel on kaggle.com and wanting to just retry
    the same run.
    
    use_gpu: False (default) runs on CPU. Set True ONLY if the user
    explicitly asks for GPU. When True, the script must use GPU params
    (XGB device='cuda'); when False it must contain none.
    """

    global LAST_OUTPUT_FOLDER

    LAST_OUTPUT_FOLDER = ""
    accel = "GPU (T4)" if use_gpu else "CPU"

    kernel_slug = f"Kwest-{COMP_SLUG}"
    kernel_ref = f"{KAGGLE_USER}/{kernel_slug}"

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
        "id": f"{kernel_ref}",
        "title": kernel_slug,
        "code_file": "solution.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
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
            ["kaggle", "kernels", "status", kernel_ref],
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
                f"KERNEL STUCK IN QUEUE for {elapsed_now}s on {accel} and never started.\n"
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
        log_tail = ""
        try:
            subprocess.run(
                ["kaggle", "kernels", "output", kernel_ref, "-p", output_folder],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=180
            )
            for fn in os.listdir(output_folder):
                fp = os.path.join(output_folder, fn)
                if os.path.isfile(fp) and fn.lower().endswith((".log", ".txt", ".out")):
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        log_tail += f"\n--- {fn} (last 3000 chars) ---\n" + f.read()[-8000:]
        except Exception as e:
            log_tail = f"(could not fetch log: {e})"

        return (
            f"KERNEL {terminal_status}.\n"
            f"ACCELERATOR: {accel}\n"
            f"Execution status: {status_text}\n"
            f"LOG TAIL:{log_tail or ' (no log available)'}\n"
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
        f"ACCELERATOR: {accel}\n"
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

MAX_REPORT_CHARS = 9000
METRIC_NAMES = {"binary": "AUC", "multiclass": "log-loss", "regression": "RMSE"}


def _prepare_target(raw, task="auto"):
    """Encode the target once so every section works for any task.
    Returns (task, y, Y, k, classes). y = int labels (classification) or floats (regression).
    Y = (n, m) float matrix used to measure how a category level differs from average."""
    is_num = pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw)
    if task == "auto":
        if is_num and (raw.nunique() > 20 or (raw.dropna() % 1 != 0).any()):
            task = "regression"
        else:
            task = "binary" if raw.nunique() == 2 else "multiclass"
    if task == "regression":
        y = raw.to_numpy(dtype=float)
        Y = ((y - y.mean()) / (y.std() + 1e-9)).reshape(-1, 1)
        return "regression", y, Y, 1, None
    le = LabelEncoder()
    y = le.fit_transform(raw.astype(str))
    k = len(le.classes_)
    Y = np.eye(k)[y] if k > 2 else y.reshape(-1, 1).astype(float)
    return ("binary" if k == 2 else "multiclass"), y, Y, k, list(le.classes_)


def _score(task, y, pred, k):
    """Higher is always better: AUC, -log-loss, -RMSE."""
    if task == "binary":
        return roc_auc_score(y, pred[:, 0])
    if task == "multiclass":
        return -log_loss(y, pred, labels=list(range(k)))
    return -mean_squared_error(y, pred[:, 0]) ** 0.5


def _fit_predict(task, Xtr, ytr, Xva, seed=0):
    kw = dict(n_estimators=100, learning_rate=0.1, num_leaves=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbosity=-1,
              random_state=seed)
    if task == "regression":
        return lgb.LGBMRegressor(**kw).fit(Xtr, ytr).predict(Xva).reshape(-1, 1)
    p = lgb.LGBMClassifier(**kw).fit(Xtr, ytr).predict_proba(Xva)
    return p[:, 1:] if task == "binary" else p


def _digit_candidates(X, floats):
    """Generic 'how the number is written' candidates for float columns:
    cents remainder, tenths digit, leading digit, integer part (low-cardinality only), round-number flag."""
    out = []
    for c in floats:
        av = X[c].astype(float).abs()
        tot = (av * 100).round()
        cents, tenths, intpart = tot % 100, (tot // 10) % 10, tot // 100
        pos = av.where(av > 0)
        lead = np.floor(pos / 10 ** np.floor(np.log10(pos) + 1e-12))
        t = f"(df['{c}'].abs() * 100).round()"
        out.append((f"cat: ({t} % 100).astype(int)  [cents of {c}]", cents.astype("category")))
        out.append((f"cat: (({t} // 10) % 10).astype(int)  [first decimal of {c}]", tenths.astype("category")))
        out.append((f"cat: first significant digit of abs({c})  [leading digit of {c}]", lead.astype("category")))
        out.append((f"({t} % 100 == 0).astype(int)  [round number flag for {c}]", (cents == 0).astype(int)))
        if 2 <= intpart.nunique() <= 100:
            out.append((f"cat: ({t} // 100).astype(int)  [integer part of {c}]", intpart.astype("category")))
    return out


def _scan_candidate_features(X, task, y, Y, k, ranked_feats, max_feats=20):
    """Screen candidate features one by one, then add them greedily until gains stop
    beating the scan's own noise. X must be reset-indexed (text columns as category dtype)."""
    if task == "multiclass" and np.bincount(y).min() < 10:
        return "CANDIDATE SCAN skipped: a class has fewer than 10 rows in the sample."
    splitter = KFold if task == "regression" else StratifiedKFold
    folds = list(splitter(3, shuffle=True, random_state=0).split(X, y))

    def score(F, seed=0):
        oof = np.zeros((len(F), k if task == "multiclass" else 1))
        for tr, va in folds:
            oof[va] = _fit_predict(task, F.iloc[tr], y[tr], F.iloc[va], seed)
        return _score(task, y, oof, k)

    d = lambda c: f"df['{c}']"
    ranked = [f for f in ranked_feats if f in X.columns]
    top = ranked[:8]
    floats = [f for f in ranked if pd.api.types.is_float_dtype(X[f]) and X[f].nunique() > 50][:4]
    nums = [f for f in top if pd.api.types.is_numeric_dtype(X[f]) and X[f].nunique() > 2][:4]
    cats = [f for f in top if str(X[f].dtype) == "category"][:2]
    cand = []  # (expression, values)

    for a, b in combinations(nums, 2):
        cand += [(f"{d(a)} * {d(b)}", X[a] * X[b]),
                 (f"{d(a)} / ({d(b)}.abs() + 1e-6)", X[a] / (X[b].abs() + 1e-6)),
                 (f"{d(a)} - {d(b)}", X[a] - X[b])]

    for c in cats:
        s = X[c].astype(str)
        # levels whose target behaviour differs most from average (any task type)
        g = pd.DataFrame(Y).groupby(s.values)
        dev = ((g.mean() - Y.mean(0)).abs() / (Y.std(0) + 1e-9)).max(axis=1)
        dev = dev[g.size() >= 0.01 * len(X)].sort_values(ascending=False)
        for lvl in list(dev.index[:2]):
            flag = (s == lvl).astype(int)
            cand.append((f"({d(c)}.astype(str) == '{lvl}').astype(int)", flag))
            for n in nums[:2]:
                cand.append((f"{d(n)} * ({d(c)}.astype(str) == '{lvl}').astype(int)", X[n] * flag))
        if X[c].nunique() > 10:
            freq = s.map(s.value_counts(normalize=True))
            cand.append((f"{d(c)}.astype(str).map(train['{c}'].astype(str).value_counts(normalize=True)).fillna(0)", freq))

    bools = [c for c in X.columns if X[c].nunique() == 2 and pd.api.types.is_numeric_dtype(X[c])]
    if len(bools) >= 3:
        cand.append((f"df[{bools}].astype(int).sum(axis=1)", X[bools].sum(axis=1)))

    cand = cand[:40] + _digit_candidates(X, floats)
    prepared = []
    for e, v in cand[:60]:
        if str(getattr(v, "dtype", "")) == "category":
            prepared.append((e, v))
        else:
            a = np.asarray(v, dtype=float)
            prepared.append((e, np.where(np.isfinite(a), a, np.nan)))
    cand = prepared

    base = score(X)
    noise = float(np.std([base, score(X, seed=1), score(X, seed=2)]))   # run-to-run noise of this scan
    gate = max(3 * noise, 3e-3 * abs(base))            # a feature must beat the noise AND a minimum lift
    singles = sorted(((score(X.assign(cand_=v)) - base, e, v) for e, v in cand),
                     key=lambda t: t[0], reverse=True)

    # Greedy forward selection: stop at max_feats, at lift <= 0, or after 3 failures
    F, cur, chosen, fails = X.copy(), base, [], 0
    for lift, e, v in singles:
        if lift <= 0 or len(chosen) >= max_feats or fails >= 3:
            break
        name = f"cand_{len(chosen)}"
        s2 = score(F.assign(**{name: v}))
        if s2 - cur >= gate:
            F[name] = v
            chosen.append((s2 - cur, e))
            cur = s2
        else:
            fails += 1

    out = [f"RECOMMENDED FEATURES (scan on {len(X):,} rows, 3-fold, {METRIC_NAMES[task]} proxy, "
           f"baseline {base:.5f}, gate {gate:.5f}, higher=better)"]
    out += [f"{g:+.5f} | {e}" for g, e in chosen] or ["none passed the gate: add no features"]
    out.append(f"total gain {cur - base:+.5f} with {len(chosen)} features")
    return "\n".join(out)

@tool
def analyze_feature_importance(task: str = "auto") -> str:
    """
    Quick analysis of which features relate to the target, for ANY task type
    (binary, multiclass, regression; text labels like Yes/No are fine): LightGBM
    importance, correlation, mutual information, target behaviour by category, and a
    screen of candidate engineered features that returns a RECOMMENDED FEATURES list.
    Uses AUC / log-loss / RMSE as proxy metrics, not the competition metric.
    Call inspect_competition_data first if comp_data hasn't been downloaded yet.
    Leave task='auto' unless auto-detection is clearly wrong
    (use 'regression', 'binary' or 'multiclass').
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
        return f"Detected target '{target_col}' not in train.csv columns: {list(df.columns)}"
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    try:
        task, y_all, Y_all, k, classes = _prepare_target(df[target_col], task)
    except Exception as e:
        return f"Could not encode target '{target_col}': {e}"
    if task != "regression" and k < 2:
        return "Target has a single class; nothing to analyze."

    X = df.drop(columns=[target_col])
    dropped = []
    for c in list(X.columns):
        s = X[c]
        is_text = not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        id_like = c.lower() == "id" or (
            s.nunique() == len(X) and (pd.api.types.is_integer_dtype(s) or is_text))
        if id_like or (is_text and s.nunique() > 500):
            dropped.append(c)
            X = X.drop(columns=[c])
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    idx = np.random.RandomState(42).permutation(len(X))[:min(150_000, len(X))]
    Xs, ys, Ys = X.iloc[idx].reset_index(drop=True), y_all[idx], Y_all[idx]

    strat = ys if task != "regression" and np.bincount(ys).min() >= 2 else None
    Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=0.2, stratify=strat, random_state=42)
    kw = dict(n_estimators=200, num_leaves=63, learning_rate=0.05, random_state=42, verbosity=-1)
    model = (lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier)(**kw)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(20, verbose=False)])
    importances = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                            index=Xs.columns).sort_values(ascending=False)
    if task == "regression":
        pv = model.predict(Xva).reshape(-1, 1)
    elif task == "binary":
        pv = model.predict_proba(Xva)[:, 1:]
    else:
        pv = model.predict_proba(Xva)
    valid = _score(task, yva, pv, k)
    valid = valid if task == "binary" else -valid

    head = f"TASK: {task} | rows: {len(df):,} | features: {X.shape[1]}"
    if task == "binary":
        head += f" | positive class: {classes[1]}"
    if task == "multiclass":
        head += f" | classes: {k}"
    if dropped:
        head += f"\nignored columns (id-like or high-cardinality text): {dropped}"
    head += (f"\nQUICK MODEL VALID {METRIC_NAMES[task]}: {valid:.5f} "
             "(sampled model for ranking only, proxy for the competition metric)\n")

    n_scan = 30_000 if (task == "multiclass" and k > 5) else 60_000
    try:
        scan = _scan_candidate_features(Xs.iloc[:n_scan].reset_index(drop=True), task,
                                        ys[:n_scan], Ys[:n_scan], k,
                                        importances.head(8).index.tolist())
    except Exception as e:
        scan = f"CANDIDATE SCAN failed: {e}"

    body = "\n=== FEATURE IMPORTANCE (gain, top 20) ===\n" + importances.head(20).to_string() + "\n"

    if num_cols and task != "multiclass":
        corr = (Xs[num_cols].replace([np.inf, -np.inf], np.nan)
                .corrwith(pd.Series(ys.astype(float))).dropna()
                .sort_values(key=abs, ascending=False))
        body += "\n=== NUMERIC CORRELATION WITH TARGET (top 15 by |corr|) ===\n" + corr.head(15).to_string() + "\n"

    Xmi = Xs.copy()
    for c in cat_cols:
        Xmi[c] = Xmi[c].cat.codes
    for c in num_cols:
        Xmi[c] = Xmi[c].replace([np.inf, -np.inf], np.nan)
        Xmi[c] = Xmi[c].fillna(Xmi[c].median())
    Xmi = Xmi.fillna(0)
    m = min(50_000, len(Xmi))
    mi_fn = mutual_info_regression if task == "regression" else mutual_info_classif
    mi = mi_fn(Xmi.iloc[:m], ys[:m], discrete_features=[c in cat_cols for c in Xmi.columns],
               random_state=42)
    mi_series = pd.Series(mi, index=Xmi.columns).sort_values(ascending=False)
    body += "\n=== MUTUAL INFORMATION WITH TARGET (top 15) ===\n" + mi_series.head(15).to_string() + "\n"

    top_cats = [c for c in mi_series.index if c in cat_cols and X[c].nunique() <= 200][:5]
    if top_cats:
        what = {"binary": "rate of positive class", "regression": "mean target",
                "multiclass": "most common class and its share"}[task]
        body += f"\n=== TARGET BY CATEGORY: {what} (levels with >=30 rows) ===\n"
    for col in top_cats:
        g = X[col].astype(str).values
        if task == "multiclass":
            ct = pd.crosstab(g, y_all)
            n = ct.sum(axis=1)
            share = ct.div(n, axis=0)
            r = pd.DataFrame({"class": [classes[i] for i in share.values.argmax(axis=1)],
                              "share": share.max(axis=1).round(3), "n": n}, index=ct.index)
            r = r[r["n"] >= 30].sort_values("share", ascending=False)
        else:
            r = pd.DataFrame({"y": y_all, "g": g}).groupby("g")["y"].agg(["mean", "count"])
            r = r[r["count"] >= 30].sort_values("mean", ascending=False)
        body += f"\n{col}:\n{r.head(8).to_string()}\n"

    return (head + "\n" + scan + "\n" + body)[:MAX_REPORT_CHARS]