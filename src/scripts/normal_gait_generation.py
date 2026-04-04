import sys
import os

# Add src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gait_engine import GaitEngine
from utils.figures import run_figures, run_joint_trajectories
from utils.animate import run_animation


if __name__ == "__main__":
    engine = GaitEngine()
    engine.save_to_csv(samples = 100)
    
    run_figures()
    run_animation()
    run_joint_trajectories()