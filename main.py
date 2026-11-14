import subprocess
import os


def run_kaggle(args):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        ['kaggle'] + args,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env
    )
    # strip the warning line
    lines = [l for l in result.stdout.splitlines() if not l.startswith('Warning')]

    return '\n'.join(lines)


def get_competition_info(comp_slug):
    return run_kaggle(['competitions', 'list', '--search', comp_slug, '--csv'])


def get_leaderboard(comp_slug):
    output = run_kaggle([
        'competitions', 'leaderboard', comp_slug, '--show', '--csv'
    ])

    return output

def get_my_submissions(comp_slug):
    return run_kaggle(['competitions', 'submissions', comp_slug, '--csv'])

if __name__ == "__main__":
    comp = input("Enter competition slug: ")

    print("=== COMPETITION INFO ===")
    print(get_competition_info(comp))

    print("=== LEADERBOARD ===")
    print(get_leaderboard(comp))

    print("=== MY SUBMISSIONS ===")
    print(get_my_submissions(comp))