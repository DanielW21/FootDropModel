import numpy as np
import os
import sys
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sims.muscle_sim import run_integrated_sim, load_config, get_latest_baseline, animate_gait

def view_best_result(best_nodes):
    config = load_config()

    baseline_path = get_latest_baseline()
    if not baseline_path:
        print("Error: No baseline CSV found.")
        return

    df_base = pd.read_csv(baseline_path)
    phases = df_base['phase'].values

    win_start = config['optimization'].get('swing_start', 0.55)
    win_end = config['optimization'].get('swing_end', 0.9)

    node_times = np.linspace(win_start, win_end, len(best_nodes))
    fes_interpolator = interp1d(node_times, best_nodes, kind='cubic', fill_value="extrapolate")

    u_ta_traj = []
    for t in phases:
        if win_start <= t <= win_end:
            val = np.clip(float(fes_interpolator(t)), 0.01, 1.0)
            u_ta_traj.append(val)
        else:
            u_ta_traj.append(0.01)

    u_ta_traj = np.array(u_ta_traj)

    print(f"Running Simulation with {len(best_nodes)} nodes...")
    df, sol, foot_len = run_integrated_sim(config, u_ta_traj)

    print("Launching Animation Window...")
    animate_gait(df, sol, foot_len, interval=50)

if __name__ == "__main__":
    default_best_nodes = np.array([0.303, 0.29, 0.61, 0.606, 0.596, 0.577, 0.3, 0.44, 0.416, 0.432])
    view_best_result(default_best_nodes)