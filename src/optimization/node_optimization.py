import numpy as np
import os
import sys
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sims.muscle_sim import run_integrated_sim, load_config, get_latest_baseline

pbar = None

def generate_u_ta_trajectory(amp_nodes, config, phases):
    opt_cfg = config.get('optimization', {})
    win_start = opt_cfg.get('swing_start', 0.55)
    win_end = opt_cfg.get('swing_end', 0.9)
    
    node_times = np.linspace(win_start, win_end, len(amp_nodes))
    
    fes_curve = interp1d(node_times, amp_nodes, kind='cubic', fill_value="extrapolate")
    
    u_ta_array = []
    for t in phases:
        if win_start <= t <= win_end:
            val = np.clip(float(fes_curve(t)), 0.01, 1.0)
            u_ta_array.append(val)
        else:
            u_ta_array.append(0.01)
            
    return np.array(u_ta_array)

def objective(amp_nodes, config, phases):
    """
    Loss function: Pre-calculates trajectory, runs sim, and returns total cost.
    """
    global pbar
    opt_cfg = config.get('optimization', {})
    
    u_ta_traj = generate_u_ta_trajectory(amp_nodes, config, phases)
    
    df, sol, L_FOOT = run_integrated_sim(config, u_ta_traj)
    
    t, theta, a_ta = sol.t, sol.y[0], sol.y[2]
    
    effort = np.trapezoid(a_ta**2, t) * opt_cfg.get('effort_weight', 1.0)

    penalty = 0
    shank_angle = df['hip_q'].values - df['knee_q'].values
    ay_pos = df['ankle_y'].values
    
    win_start, win_end = opt_cfg.get('swing_start', 0.55), opt_cfg.get('swing_end', 0.9)
    clearance_k = opt_cfg.get('clearance_weight', 100000.0)

    for i in range(len(t)):
        if win_start < t[i] < win_end:
            psi = shank_angle[i] + theta[i]
            y_toe = ay_pos[i] + L_FOOT * np.sin(psi)
            if y_toe < 0.02: # clearnance plus rounding buffer
                penalty += (0 - y_toe)**2 * clearance_k
    
    total_loss = effort + penalty

    if pbar is not None:
        pbar.update(1)
        pbar.set_postfix({"Loss": f"{total_loss:.4f}"})
                
    return total_loss

if __name__ == "__main__":
    config = load_config()
    node_cfg = config.get('node_optimization', {})
    opt_cfg = config.get('optimization', {}) 

    df_base = pd.read_csv(get_latest_baseline())
    phases = df_base['phase'].values
    
    n_nodes = node_cfg.get('num_nodes', 7)
    x0 = np.full(n_nodes, node_cfg.get('initial_guess', 0.44))
    bounds = [(0.0, 1.0) for _ in range(n_nodes)]
    
    print(f"\nSTARTING NODE OPTIMIZATION ({n_nodes} Nodes)")
    pbar = tqdm(desc="Optimizing Trajectory", unit="sim")
    
    res = minimize(
        objective, 
        x0, 
        args=(config, phases), 
        method='L-BFGS-B', 
        bounds=bounds,
        tol=1e-3, 
        options={
            'maxiter': 25, 
            'ftol': 1e-4,   
            'gtol': 1e-2,   
            'eps': 1e-2,
            'maxcor': 20   
        }
    )
    
    pbar.close()

    print(f"Final Loss: {res.fun:.4f}")
    print(f"Optimal Nodes: {np.round(res.x, 3)}")
        
