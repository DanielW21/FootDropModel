import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sims.muscle_sim import run_integrated_sim, load_config

def extract_coords(df, sol, foot_len):
    """
    Manually calculates X,Y coordinates based on muscle_sim.py logic.
    This bypasses naming conflicts by using the physics defined in the sim.
    """
    toe_x, toe_y = [], []
    ankle_x_vals, ankle_y_vals = [], []
    
    for i in range(len(df)):
        row = df.iloc[i]
        # Physics logic from muscle_sim.py animate_gait function
        theta = sol.y[0][i] 
        phi = row['hip_q'] - row['knee_q']
        psi = phi + theta
        
        ax, ay = row['ankle_x'], row['ankle_y']
        tx = ax + foot_len * np.cos(psi)
        ty = ay + foot_len * np.sin(psi)
        
        toe_x.append(tx)
        toe_y.append(ty)
        ankle_x_vals.append(ax)
        ankle_y_vals.append(ay)
        
    return np.array(toe_x), np.array(toe_y)

def run_metrics_comparison(best_nodes):
    config = load_config()
    
    # 2. Define FES Trajectories
    phases = np.linspace(0, 1, 100)
    win_start, win_end = 0.50, 0.90 # Consistent with muscle_sim.py swing window
    
    node_times = np.linspace(win_start, win_end, len(best_nodes))
    fes_interp = interp1d(node_times, best_nodes, kind='cubic', fill_value="extrapolate")
    
    u_opt = np.array([np.clip(float(fes_interp(t)), 0.01, 1.0) if win_start <= t <= win_end else 0.01 for t in phases])
    u_none = np.array([0 for _ in phases])

    # 3. Run Simulations
    print("Simulating Baseline...")
    df_no, sol_no, flen = run_integrated_sim(config, u_none)
    
    print("Simulating Optimized FES...")
    df_opt, sol_opt, _ = run_integrated_sim(config, u_opt)

    # 4. Extract Trajectories
    no_tx, no_ty = extract_coords(df_no, sol_no, flen)
    opt_tx, opt_ty = extract_coords(df_opt, sol_opt, flen)

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0, color='black', lw=2, label='Floor')
    
    # Plot Baseline (Red Dashed)
    ax.plot(no_tx, no_ty, 'r--', lw=2, alpha=0.6, label='Baseline (Drop Foot/Toe Drag)')
    
    # Plot Optimized (Blue Solid)
    ax.plot(opt_tx, opt_ty, 'b-', lw=3, label='Optimized FES (Clearance Achieved)')

    # Aesthetics
    ax.set_aspect('equal')
    ax.set_title("Gait Success Metric: Toe Trajectory (X, Y)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(-0.4, 0.4)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()

    plt.savefig("fes_success_metrics.png", dpi=300)
    print("Graph saved: fes_success_metrics.png")
    plt.show()

if __name__ == "__main__":
    best_nodes = np.array([0.27, 0.107, 0.399, 0.454, 0.528, 0.39, 0.0, 0.0, 0.0, 0.0])
    run_metrics_comparison(best_nodes)
    
    # Results of node opt:
    # Optimizing Trajectory: 869sim [21:50,  1.51s/sim, Loss=6.9589]  
    # Final Loss: 4.4589
    # Optimal Nodes: [0.429 0.428 0.429 0.429 0.43  0.429 0.432 0.438 0.438 0.439]