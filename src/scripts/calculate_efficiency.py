import numpy as np
import matplotlib.pyplot as plt
import os
from sims.muscle_sim import run_integrated_sim, load_config, get_latest_baseline
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simpson

def calculate_and_plot_fes_efficiency(best_nodes, flat_val=0.55):
    config = load_config()
    win_start, win_end = 0.55, 0.90
    
    df_base = pd.read_csv(get_latest_baseline())
    phases = df_base['phase'].values
    node_times = np.linspace(win_start, win_end, len(best_nodes))
    fes_interp = interp1d(node_times, best_nodes, kind='cubic', fill_value="extrapolate")
    
    u_opt = np.array([np.clip(float(fes_interp(t)), 0.01, 1.0) if win_start <= t <= win_end else 0.01 for t in phases])


    print("Simulating Nodal FES...")
    _, sol_opt, _ = run_integrated_sim(config, u_opt)
    
    print("Simulating Flat FES...")
    _, sol_flat, _ = run_integrated_sim(config, flat_val)

    a_opt = sol_opt.y[2]
    a_flat = sol_flat.y[2]
    t = sol_opt.t

    effort_opt = simpson(a_opt**2, t)
    effort_flat = simpson(a_flat**2, t)
    
    efficiency_gain = ((effort_flat - effort_opt) / effort_flat) * 100

    print("\n" + "="*30)
    print("METABOLIC EFFICIENCY RESULTS")
    print("="*30)
    print(f"Effort (Flat {flat_val}): {effort_flat:.6f}")
    print(f"Effort (Nodal Opt): {effort_opt:.6f}")
    print(f"Net Energy Saving:  {efficiency_gain:.2f}%")
    print("="*30 + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(t, a_flat, 'k--', lw=2, alpha=0.6, label=f'Flat Baseline (Effort: {effort_flat:.3f})')
    plt.plot(t, a_opt, 'b-', lw=2.5, label=f'Nodal Optimized (Effort: {effort_opt:.3f})')
    
    plt.fill_between(t, a_opt, alpha=0.1, color='blue')
    plt.axvspan(win_start, win_end, color='gray', alpha=0.05, label='Swing Window')
    
    plt.title(f"TA Activation State ($a_{{TA}}$) Comparison\nEnergy Reduction: {efficiency_gain:.2f}%", fontweight='bold')
    plt.xlabel("Gait Cycle (Normalized Time)")
    plt.ylabel("Activation Level ($a$)")
    plt.ylim(-0.05, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/activation_efficiency_comparison.png", dpi=300)
    plt.show()

    return efficiency_gain

if __name__ == "__main__":
    nodes = np.array([0.303, 0.29, 0.61, 0.606, 0.596, 0.577, 0.3, 0.44, 0.416, 0.432])
    calculate_and_plot_fes_efficiency(nodes)