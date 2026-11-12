import os
import json
import time
import subprocess
import tempfile
from langchain_core.tools import tool

COMP_SLUG = ""

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

    folder = tempfile.mkdtemp()

    # following part writes the code
    code_path = os.path.join(folder, 'solution.py')
    with open(code_path, "w") as f:
        f.write(code)

    metadata = {
        "id": f"rugvedbane/kwest-kernel",
        "title": "Kwest Kernel",
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
    with open(meta_path, "w") as f:
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

    print('kernel pushed. Polling status...')

    kernel_slug = "rugvedbane/kwest-kernel"

    while True:
        time.sleep(10)

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
        print("Not ready yet, waiting...")

    output_text = ""
    for filename in os.listdir(output_folder):
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "r", encoding='utf-8', errors='replace') as f:
            output_text += f'\n--- {filename} ---\n'
            output_text += f.read()
    
                
    return output_text if output_text else "kernel completed but no output found"


@tool
def submit_to_competition(submission_file_path: str, message: str) -> str:
    """
    Submits a CSV file to the current Kaggle competition.
    submission_file_path: full path to the submission.csv file.
    message: short description of this submission.
    """

    result = subprocess.run(
        ['kaggle', 'competitions', 'submit', 
         '-c', COMP_SLUG, 
         '-f', submission_file_path, 
         '-m', message],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode != 0:
        return f"submission failed {result.stderr}"

    return f"submitted sucessfully: {result.stdout}"


set_comp_slug("playground-series-s6e11")

result = submit_to_competition.invoke({
    "submission_file_path": r"C:\Users\rugve\Kwest\sample_submission.csv",
    "message": "Kwest submission tool test"
})

print(result)