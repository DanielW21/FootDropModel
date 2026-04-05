"""Simulates ankle dorsiflexion using coupled muscle activation and joint dynamics.
It compares a stimulated tibialis anterior response against passive motion under joint limits and floor contact constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import os
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muscles.millard_model import MillardMuscle

def load_config():
    """Loads the central configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
anthro = config['anthropometrics']
d_params = config['dynamics_params']
lim_cfg = config['limits']

FOOT_MASS = anthro['foot_weight']  #
L_FOOT = anthro['foot']            #
W_FOOT = anthro.get('foot_width', 0.09) #
L_SHANK = anthro['shank']          #
G = d_params.get('gravity', 9.81)  #
D_COM = L_FOOT / 2.0               #

I_COM = (1/12) * FOOT_MASS * (L_FOOT**2 + W_FOOT**2) #
I_FOOT = I_COM + FOOT_MASS * D_COM**2               #

LIMITS = np.radians(lim_cfg['foot'])   # Anatomical stops
K_LIMIT = lim_cfg['k_limit']           #
D_LIMIT = 2 * np.sqrt(K_LIMIT * I_FOOT) * lim_cfg['d_limit'] #
JOINT_DAMPING = d_params['joint_damping'] #

FES_START = 0.8
FES_END = 1.8

ta_muscle = MillardMuscle('TA', config)
sol_muscle = MillardMuscle('Soleus', config)

def dynamics(t, state, u_max_ta):
    theta, omega, a_ta, a_sol = state
    
    u_ta = u_max_ta if FES_START < t < FES_END else 0.01 
    u_sol = 0.01
    
    da_ta = ta_muscle.get_activation_derivative(u_ta, a_ta)
    da_sol = sol_muscle.get_activation_derivative(u_sol, a_sol)
    
    tau_g = -FOOT_MASS * G * D_COM * np.cos(theta)
    
    tau_ta = ta_muscle.get_torque(theta, omega, a_ta)
    tau_sol = sol_muscle.get_torque(theta, omega, a_sol)
    tau_m = tau_ta + tau_sol
    
    tau_limit = 0
    
    if t < FES_START:
        floor_angle = 0.0
        if theta < floor_angle:
            tau_limit = K_LIMIT * (floor_angle - theta) - D_LIMIT * omega
            
    if theta < LIMITS[0]:
        tau_limit = K_LIMIT * (LIMITS[0] - theta) - D_LIMIT * omega
    elif theta > LIMITS[1]:
        tau_limit = K_LIMIT * (LIMITS[1] - theta) - D_LIMIT * omega
    
    alpha_accel = (tau_g + tau_m + tau_limit + JOINT_DAMPING * omega) / I_FOOT
    
    return [omega, alpha_accel, da_ta, da_sol]

t_eval = np.linspace(0, 2.5, 120)
init_state = [
    np.radians(config['state_variables']['ankle_θ']),
    np.radians(config['state_variables']['ankle_θ_dot']),
    config['state_variables']['activation_TA'],
    config['state_variables']['activation_SOL']
]

sol_h = solve_ivp(dynamics, [0, 2.5], init_state, t_eval=t_eval, args=(1.0,), method='Radau')
sol_l = solve_ivp(dynamics, [0, 2.5], init_state, t_eval=t_eval, args=(0.4,), method='Radau')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.set_xlim(-0.6, 0.6); ax1.set_ylim(-0.8, 0.2); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.2)

floor_line, = ax1.plot([-0.6, 0.6], [-L_SHANK, -L_SHANK], 'k-', lw=2, label='Floor')

line_h, = ax1.plot([], [], 'b-', lw=6, label='High Act (1.0)')
line_l, = ax1.plot([], [], 'r--', lw=4, label='Low Act (0.4)')
ax1.legend(loc='upper right')

ax2.plot(sol_h.t, sol_h.y[2], 'b-', alpha=0.6, label='TA Act (High)')
ax2.plot(sol_l.t, sol_l.y[2], 'r--', alpha=0.6, label='TA Act (Low)')
ax2.set_ylabel('Activation'); ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right')
marker = ax2.axvline(0, color='k', ls=':')

def update(frame):
    t = sol_h.t[frame]
    th_h, th_l = sol_h.y[0][frame], sol_l.y[0][frame]
    x0, y0 = 0, -L_SHANK
    
    line_h.set_data([x0, x0 + L_FOOT * np.cos(th_h)], [y0, y0 + L_FOOT * np.sin(th_h)])
    line_l.set_data([x0, x0 + L_FOOT * np.cos(th_l)], [y0, y0 + L_FOOT * np.sin(th_l)])
    
    floor_line.set_alpha(1.0 if t < FES_START else 0.1)
    
    marker.set_xdata([t])
    return line_h, line_l, marker, floor_line

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=True)
plt.tight_layout()
plt.show()