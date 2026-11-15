import os
import json
from datetime import datetime

def save_experiment(comp_slug, description, score, notes):

    EXPERIMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "experiments.json"
    )

    try:
        with open(EXPERIMENTS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data=[]
    
    experiment = {
        'comp_slug': comp_slug,
        'description': description,
        'score': score,
        'notes': notes
    }

    data.append(experiment)

    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def get_memory_context(comp_slug):
    try:
        with open("EXPERIMENTS_FILE", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):  # json.JSONDecodeError handles empty files and correpted files.
        data = []

    filtered = [exp for exp in data if exp['comp_slug'] == comp_slug]
    last_5 = filtered[-5:]
    
    list_of_string = []
    for exp in last_5:
        list_of_string.append(f"""
        
        description: {exp['description']},
        score: {exp['score']},
        notes: {exp['notes']}

        """
    )
    return "\n".join(list_of_string) if list_of_string else "No experiments logged yet."
        
