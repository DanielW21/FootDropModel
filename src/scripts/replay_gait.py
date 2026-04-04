import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.animate import run_animation_from_csv

def get_latest_csv():
    """Finds the most recent gait CSV in the output folder."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_path = os.path.join(project_root, "output", "sim_data", "*.csv")
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getctime)

if __name__ == "__main__":
    # Manual path choice
    # csv_path = os.path.join(os.path.dirname(__file__), "sample_gait.csv")
    
    csv_path = get_latest_csv()
    run_animation_from_csv(csv_path, interval=15)
