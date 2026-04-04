import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from .gait_engine import GaitEngine

def setup_plot(title_text=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.0, 1.5); ax.set_ylim(-0.2, 1.4)
    ax.axhline(0, color='black', lw=2)
    if title_text: ax.set_title(title_text, pad=20, fontweight='bold')
    
    info = ax.text(0.98, 0.98, '', transform=ax.transAxes, ha='right', va='top', 
                   family='monospace', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    l_r, = ax.plot([], [], 'o-', lw=6, color='blue', mfc='white', zorder=10)
    l_l, = ax.plot([], [], 'o-', lw=4, color='red', alpha=0.15)
    ref_h, = ax.plot([], [], 'k--', lw=1, alpha=0.4)
    ref_k, = ax.plot([], [], 'k--', lw=1, alpha=0.4)
    ref_a, = ax.plot([], [], 'k--', lw=1, alpha=0.6)
    
    t_h = ax.text(0, 0, '', color='blue', fontweight='bold')
    t_k = ax.text(0, 0, '', color='blue', fontweight='bold')
    t_a = ax.text(0, 0, '', color='darkblue', fontweight='bold')

    return fig, info, l_r, l_l, ref_h, ref_k, ref_a, t_h, t_k, t_a

def update_kinematics(engine, frame_data, objects):
    """The 'Brain' of the animation—shared by both modes."""
    info_box, l_r, l_l, ref_h, ref_k, ref_a, t_h, t_k, t_a = objects
    phase, angles_r, angles_l = frame_data
    
    rc = engine.get_leg_coords(angles_r)
    lc = engine.get_leg_coords(angles_l)
    
    l_r.set_data(rc[0::2], rc[1::2])
    l_l.set_data(lc[0::2], lc[1::2])

    ref_h.set_data([0, 0], [engine.params['hip_height'], engine.params['hip_height'] - 0.3])

    th_v = np.array([rc[2]-rc[0], rc[3]-rc[1]])
    th_v = (th_v / np.linalg.norm(th_v)) * 0.3
    ref_k.set_data([rc[2], rc[2]+th_v[0]], [rc[3], rc[3]+th_v[1]])

    sh_v = np.array([rc[4]-rc[2], rc[5]-rc[3]])
    sh_mag = np.linalg.norm(sh_v)
    perp_v = (np.array([-sh_v[1], sh_v[0]]) / sh_mag) * 0.25
    ref_a.set_data([rc[4], rc[4]+perp_v[0]], [rc[5], rc[5]+perp_v[1]])


    foot_v = np.array([rc[6]-rc[4], rc[7]-rc[5]])
    rel_deg = np.degrees(np.arctan2(foot_v[1], foot_v[0]) - np.arctan2(sh_v[1], sh_v[0]) - (np.pi/2))
    rel_deg = (rel_deg + 180) % 360 - 180

    t_h.set_position((rc[0]+0.05, rc[1]-0.1)); t_h.set_text(f"H: {np.degrees(angles_r[0]):.0f}°")
    t_k.set_position((rc[2]+0.05, rc[3]-0.1)); t_k.set_text(f"K: {np.degrees(angles_r[1]):.0f}°")
    t_a.set_position((rc[4]+0.08, rc[5]+0.05)); t_a.set_text(f"A: {rel_deg:.0f}°")
    
    main, sub = engine.get_clinical_labels(phase)
    info_box.set_text(f"{main:^25}\n{sub:^25}\nPHASE: {phase*100:>6.1f}%")

def run_animation_from_csv(csv_path, interval = 25):
    engine = GaitEngine()
    df = pd.read_csv(csv_path)
    fig, *objs = setup_plot(f"Replay: {os.path.basename(csv_path)}")

    def update(f):
        row = df.iloc[f % len(df)]
        opp = df.iloc[(f + len(df)//2) % len(df)]
        update_kinematics(engine, (row['phase'], row[['hip_q','knee_q','ankle_q']].values, 
                             opp[['hip_q','knee_q','ankle_q']].values), objs)

    ani = FuncAnimation(fig, update, frames=len(df), interval=interval)
    plt.show()

def run_animation():
    engine = GaitEngine()
    fig, *objs = setup_plot("Live Gait Simulation")

    def update(f):
        p = (f / 100) % 1.0
        update_kinematics(engine, (p, engine.interpolate_pose(p), engine.interpolate_pose(p + 0.5)), objs)

    ani = FuncAnimation(fig, update, frames=100, interval=25)
    plt.show()