import os
import json
import time
import subprocess
import tempfile
import pandas as pd
import zipfile
from langchain_core.tools import tool
from main import get_my_submissions, get_competition_info, get_leaderboard, get_my_submissions

COMP_SLUG = ""
LAST_OUTPUT_FOLDER = ""
DATA_FOLDER = ""

def set_comp_slug(slug: str):
    global COMP_SLUG
    COMP_SLUG = slug

@tool
def run_on_kaggle(code: str) -> str:
    """
    Takes Python code as a string, pushes it to a Kaggle kernel 
    for the current competition, waits for it to finish, 
    and returns the output.
    """

    max_attempt = 40
    attempt = 0

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, 'solutions', COMP_SLUG)
    os.makedirs(folder, exist_ok=True)

    code_path = os.path.join(folder, 'solution.py')
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    metadata = {
        "id": "rugvedbane/kwest-kernel",
        "title": "Kwest Kernel",
        "code_file": "solution.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMP_SLUG],
        "kernel_sources": []
    }

    meta_path = os.path.join(folder, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # push to kaggle
    push_result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', folder],
        capture_output = True,
        text = True,
        encoding = 'utf-8',
        errors = 'replace'
    )

    if push_result.returncode != 0:
        return f"push failed: {push_result.stderr}"

    kernel_slug = "rugvedbane/kwest-kernel"

    print('Kernel Pushed. Waiting for 90s for kernel to start...')
    time.sleep(90)

    while attempt < max_attempt:
        time.sleep(30)
        attempt += 1

        output_folder = os.path.join(folder, "output")
        os.makedirs(output_folder, exist_ok=True)

        subprocess.run(
            ['kaggle', 'kernels', 'output', kernel_slug, '-p', output_folder],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        files = os.listdir(output_folder)
        if files:
            break
        print(f"Not ready yet, attempt {attempt}/{max_attempt}...")

    if attempt >= max_attempt:
        return "Kernel timed out after 10 minutes. Check Kaggle manually."
    
    global LAST_OUTPUT_FOLDER 
    LAST_OUTPUT_FOLDER  = output_folder

    output_text = ""
    for filename in os.listdir(output_folder):
        filepath = os.path.join(output_folder, filename)

        if not os.path.isfile(filepath):
            continue

        if filename.endswith(".csv"):
            output_text += f"\n--- {filename} --- (saved to disk, not shown)\n"

        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            content = f.read()
            output_text += f'\n--- {filename} ---\n'
            output_text += content[:3000]
    
                
    return output_text if output_text else "kernel completed but no output found"


@tool
def submit_to_competition(message: str) -> str:
    """
    Submits submission.csv from the last kernel output to the current competition.
    message: short description of this submission.
    """
    submission_path = os.path.join(LAST_OUTPUT_FOLDER, 'submission.csv')

    if not os.path.exists(submission_path):
        return f"No submission.csv found in last kernel output folder: {LAST_OUTPUT_FOLDER}"
    
    result = subprocess.run(
        ['kaggle', 'competitions', 'submit', 
         '-c', COMP_SLUG, 
         '-f', submission_path, 
         '-m', message],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode != 0:
        return f"submission failed {result.stderr}"

    return f"submitted sucessfully: {result.stdout}"


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
    base_dir = os.path.dirname(__file__)
    solution_path = os.path.join(base_dir, 'solutions', COMP_SLUG, "solutions.py")

    if not os.path.exists(solution_path):
        return "No solution file exists for this competition yet."

    with open(solution_path, "r", encoding='utf-8') as f:
        return f.read()

