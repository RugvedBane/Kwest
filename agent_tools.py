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

COMP_SLUG = ""
LAST_OUTPUT_FOLDER = ""
DATA_FOLDER = ""

def set_comp_slug(slug: str):
    global COMP_SLUG
    COMP_SLUG = slug

@tool
def run_on_kaggle(code: str) -> str:
    """
    Pushes code to Kaggle, waits for actual kernel completion,
    downloads fresh outputs and verifies submission.csv.
    """

    global LAST_OUTPUT_FOLDER

    LAST_OUTPUT_FOLDER = ""

    kernel_slug = f"Kwest-{COMP_SLUG}"
    base_dir = os.path.dirname(os.path.abspath(__file__))

    folder = os.path.join(base_dir, "solutions", COMP_SLUG)
    output_folder = os.path.join(folder, "output")  

    os.makedirs(folder, exist_ok=True)

    # Remove previous execution output
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Save generated solution
    code_path = os.path.join(folder, "solution.py")

    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Kernel metadata
    
    metadata = {
        "id": f"rugvedbane/{kernel_slug}",
        "title": kernel_slug,
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

    # Wait for actual execution status
    start_time = time.time()
    timeout = 2700
    poll_interval = 15

    terminal_status = None
    last_status = ""

    while time.time() - start_time < timeout:

        status_result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60
        )

        if status_result.returncode != 0:
            return (
                "Unable to retrieve Kaggle kernel status.\n"
                f"{status_result.stderr}"
            )

        status_text = (
            status_result.stdout + "\n" + status_result.stderr
        ).strip()

        if status_text != last_status:
            print("Kernel status:", status_text)
            last_status = status_text

        status_upper = status_text.upper()

        if "COMPLETE" in status_upper:
            terminal_status = "COMPLETE"
            break

        if "ERROR" in status_upper:
            terminal_status = "ERROR"
            break

        if "CANCEL" in status_upper:
            terminal_status = "CANCELLED"
            break

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
            kernel_slug,
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

    # Verify submission
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

    # Return useful logs without loading entire CSV files
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    solution_path = os.path.join(base_dir, "solutions", COMP_SLUG, "solution.py")

    if not os.path.exists(solution_path):
        return "No solution file exists for this competition yet."

    with open(solution_path, "r", encoding='utf-8') as f:
        return f.read()
