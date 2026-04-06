import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os, glob, yaml, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muscles.millard_model import MillardMuscle

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_latest_baseline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_path = os.path.join(project_root, "output", "sim_data", "trajectory_baseline_*.csv")
    files = glob.glob(search_path)
    return max(files, key=os.path.getctime) if files else None

def run_integrated_sim(config, u_ta_input=1.0):
    anthro = config['anthropometrics']
    L_FOOT = anthro['foot']
    FOOT_MASS = anthro['foot_weight']
    
    I_COM = (1/12) * FOOT_MASS * (L_FOOT**2 + 0.09**2) 
    I_FOOT = I_COM + FOOT_MASS * (L_FOOT/2.0)**2

    ta_muscle = MillardMuscle('TA', config)
    sol_muscle = MillardMuscle('Soleus', config)

    csv_path = get_latest_baseline()
    if not csv_path:
        raise FileNotFoundError("No trajectory baseline found.")
    
    df = pd.read_csv(csv_path)
    phases = df['phase'].values
    dt = np.mean(np.diff(phases))
    
    ax_pos, ay_pos = df['ankle_x'].values, df['ankle_y'].values
    shank_angle = df['hip_q'].values - df['knee_q'].values
    
    vx, vy = np.gradient(ax_pos, dt), np.gradient(ay_pos, dt)
    accel_x, accel_y = np.gradient(vx, dt), np.gradient(vy, dt)

    if isinstance(u_ta_input, (float, int, np.float64)):
        u_ta_traj = np.full(len(phases), float(u_ta_input))
        for i, t in enumerate(phases):
            if not (0.55 < t < 0.90):
                u_ta_traj[i] = 0.01
    else:
        u_ta_traj = u_ta_input

    def system_dynamics(t, state):
        theta, omega, a_ta, a_sol = state
        
        idx = min(int(t / dt), len(phases) - 1)
        phi = shank_angle[idx]
        ax, ay = accel_x[idx], accel_y[idx]
        psi = phi + theta 
        
        u_ta = u_ta_traj[idx]
        u_sol = 0.01
        
        da_ta = ta_muscle.get_activation_derivative(u_ta, a_ta)
        da_sol = sol_muscle.get_activation_derivative(u_sol, a_sol)
        
        tau_g = -FOOT_MASS * 9.81 * (L_FOOT/2.0) * np.cos(psi)
        tau_in = (FOOT_MASS * ax * (L_FOOT/2.0) * np.sin(psi)) - (FOOT_MASS * ay * (L_FOOT/2.0) * np.cos(psi))
        tau_m = ta_muscle.get_torque(theta, omega, a_ta) + sol_muscle.get_torque(theta, omega, a_sol)
        
        tau_limit = 0
        limit_plantar = np.radians(config['limits']['foot'][0])
        limit_dorsi = np.radians(config['limits']['foot'][1])
        
        y_toe = ay + L_FOOT * np.sin(psi)
        is_swing = (0.50 < t < 0.90)

        if not is_swing:
            target_theta = -phi 
            k_floor, d_floor = 200000.0, 5000.0    
            if y_toe < 0 or theta < target_theta:
                tau_limit = k_floor * (target_theta - theta) - d_floor * omega
            
        if theta < limit_plantar:
            tau_limit += 25000.0 * (limit_plantar - theta) - 100.0 * omega
        elif theta > limit_dorsi:
            tau_limit += 25000.0 * (limit_dorsi - theta) - 100.0 * omega

        alpha = (tau_g + tau_in + tau_m + tau_limit - 0.8 * omega) / I_FOOT
        return [omega, alpha, da_ta, da_sol]

    init_state = [np.radians(config['state_variables']['ankle_θ']), 0, 0.01, 0.01]
    sol = solve_ivp(
        system_dynamics, [0, 1.0], init_state, 
        t_eval=phases, method='Radau', 
        max_step=0.0005, rtol=1e-8, atol=1e-10
    )
    return df, sol, L_FOOT

def animate_gait(df, sol, foot_len, interval=50):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax_top.set_aspect('equal')
    ax_top.set_xlim(-1.5, 1.5)
    ax_top.set_ylim(-0.3, 1.5)
    ax_top.axhline(0, color='black', lw=2)

    trail_line, = ax_top.plot([], [], 'b:', lw=1.5, alpha=0.6, label='Toe Trajectory')
    leg_line, = ax_top.plot([], [], 'o-', lw=6, color='blue', mfc='white')
    
    act_line, = ax_bot.plot(sol.t, sol.y[2], 'b-', label='TA Activation')
    marker = ax_bot.axvline(0, color='k', ls=':')
    ax_bot.set_ylim(-0.05, 1.1)

    trail_x, trail_y = [], []

    def update(frame):
        row = df.iloc[frame]
        theta, t_curr = sol.y[0][frame], sol.t[frame]
        
        phi = row['hip_q'] - row['knee_q']
        psi = phi + theta
        ax, ay = row['ankle_x'], row['ankle_y']
        tx = ax + foot_len * np.cos(psi)
        ty = ay + foot_len * np.sin(psi)
        
        trail_x.append(tx)
        trail_y.append(ty)
        
        trail_line.set_data(trail_x, trail_y)
        leg_line.set_data([row['hip_x'], row['knee_x'], ax, tx], 
                          [row['hip_y'], row['knee_y'], ay, ty])
        
        marker.set_xdata([t_curr])
        
        is_swing = 0.55 < t_curr < 0.90
        leg_line.set_color('red' if ty < -1e-2 and is_swing else 'blue')
        
        return leg_line, marker, trail_line

    ani = FuncAnimation(fig, update, frames=len(df), interval=interval, blit=True)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    
if __name__ == "__main__":
    config = load_config()
    u_strength = 0.55
    df, sol, foot_len = run_integrated_sim(config, u_ta_input=u_strength)
    animate_gait(df, sol, foot_len)