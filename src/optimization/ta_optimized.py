import numpy as np
import os
import sys
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sims.muscle_sim import run_integrated_sim, load_config

pbar = None

def objective(params, config):
    """
    Loss function: Balance metabolic effort vs. toe clearance.
    """
    global pbar
    amp = params[0]
    opt_cfg = config.get('optimization', {})
    
    df, sol, L_FOOT = run_integrated_sim(config, u_max_ta=amp)
    
    t = sol.t
    theta = sol.y[0]
    a_ta = sol.y[2]
    
    effort = np.trapezoid(a_ta**2, t) * opt_cfg.get('effort_weight', 1.0)
    
    penalty = 0
    shank_angle = df['hip_q'].values - df['knee_q'].values
    ay_pos = df['ankle_y'].values
    
    win_start = opt_cfg.get('swing_start', 0.5)
    win_end = opt_cfg.get('swing_end', 0.9)
    clearance_k = opt_cfg.get('clearance_weight', 100000.0)

    for i in range(len(t)):
        if win_start < t[i] < win_end:
            psi = shank_angle[i] + theta[i]
            y_toe = ay_pos[i] + L_FOOT * np.sin(psi)
            
            if y_toe < 0:
                penalty += (0 - y_toe)**2 * clearance_k
    
    total_loss = effort + penalty

    if pbar is not None:
        pbar.update(1)
        pbar.set_postfix({"Amp": f"{amp:.3f}", "Loss": f"{total_loss:.4f}"})
                
    return total_loss

if __name__ == "__main__":
    config = load_config()
    
    x0 = [0.5]
    bounds = [(0.1, 1.0)]
    
    print("-" * 30)
    print("STARTING FES OPTIMIZATION")
    print("-" * 30)
    
    pbar = tqdm(total=None, desc="Evaluating Profiles", unit="sim")

    res = minimize(
        objective, 
        x0, 
        args=(config,), 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 50}
    )
    
    pbar.close() 

    if res.success:
        print("\n" + "=" * 30)
        print("OPTIMIZATION SUCCESSFUL")
        print(f"Target Amplitude: {res.x[0]:.4f}")
        print("=" * 30)
    else:
        print(f"\nOptimization failed: {res.message}")