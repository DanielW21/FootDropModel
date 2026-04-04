import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from .gait_engine import GaitEngine

def _render_single_pose(ax, engine, kf, phase_offset=0.5):
    """Internal helper to render a single pose with clinical labeling."""
    phase = kf[3]
    angles_r = kf[:3]
    rc = engine.get_leg_coords(angles_r)
    lc = engine.get_leg_coords(engine.interpolate_pose(phase + phase_offset))
    
    ax.axhline(0, color='black', lw=1.5)
    ax.plot(lc[0::2], lc[1::2], 'o-', lw=3, color='red', alpha=0.15)
    ax.plot(rc[0::2], rc[1::2], 'o-', lw=6, color='blue', mfc='white', zorder=10)
    
    ax.plot([0, 0], [engine.params['hip_height'], engine.params['hip_height'] - 0.3], 'k--', lw=1, alpha=0.4)
    
    th_v = np.array([rc[2]-rc[0], rc[3]-rc[1]])
    th_v = (th_v / np.linalg.norm(th_v)) * 0.25
    ax.plot([rc[2], rc[2]+th_v[0]], [rc[3], rc[3]+th_v[1]], 'k--', lw=1, alpha=0.4)
    
    sh_v = np.array([rc[4]-rc[2], rc[5]-rc[3]])
    sh_mag = np.linalg.norm(sh_v)
    perp_v = (np.array([-sh_v[1], sh_v[0]]) / sh_mag) * 0.25
    ax.plot([rc[4], rc[4]+perp_v[0]], [rc[5], rc[5]+perp_v[1]], 'k--', lw=1, alpha=0.6)

    foot_v = np.array([rc[6]-rc[4], rc[7]-rc[5]])
    rel_deg = np.degrees(np.arctan2(foot_v[1], foot_v[0]) - np.arctan2(sh_v[1], sh_v[0]) - (np.pi/2))
    rel_deg = (rel_deg + 180) % 360 - 180

    ax.text(rc[0]+0.05, rc[1]-0.1, f"H: {np.degrees(angles_r[0]):.0f}°", color='blue', fontweight='bold', fontsize=9)
    ax.text(rc[2]+0.05, rc[3]-0.1, f"K: {np.degrees(angles_r[1]):.0f}°", color='blue', fontweight='bold', fontsize=9)
    ax.text(rc[4]+0.08, rc[5]+0.05, f"A: {rel_deg:.0f}°", color='darkblue', fontweight='bold', fontsize=9)
    
    ax.set_aspect('equal')
    ax.set_xlim(-0.8, 1.2)
    ax.set_ylim(-0.15, 1.3)

def run_figures():
    engine = GaitEngine()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(engine.root, "gait_keyframes", ts)
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(engine.root, "configs/config.yaml"), 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    indices = config.get('labelled_keyframes', list(range(len(engine.keyframes))))

    for i in indices:
        if i >= len(engine.keyframes): continue
        kf = engine.keyframes[i]
        fig, ax = plt.subplots(figsize=(10, 7))
        _render_single_pose(ax, engine, kf)
        
        main, sub = engine.get_clinical_labels(kf[3])
        info_text = f"{main}\n{sub}\nPHASE: {kf[3]*100:.1f}%"
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, ha='right', va='top', 
                family='monospace', fontweight='bold', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        ax.set_title(f"Gait Keyframe {i} | {main}", fontweight='bold')
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Height (m)")
        
        plt.savefig(os.path.join(folder, f"keyframe_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    run_summary_figure(engine, folder, indices)

def run_summary_figure(engine, folder, indices):
    kfs = [engine.keyframes[i] for i in indices if i < len(engine.keyframes)]
    
    cols = 4
    rows = (len(kfs) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < len(kfs):
            _render_single_pose(ax, engine, kfs[i])
            main, _ = engine.get_clinical_labels(kfs[i][3])
            ax.set_title(f"{kfs[i][3]*100:.0f}% | {main}", fontsize=10, fontweight='bold', pad=5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.axis('off')

    plt.suptitle("Selected Gait Cycle Keyframes", fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.88)
    plt.savefig(os.path.join(folder, "gait_cycle_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_joint_trajectories():
    engine = GaitEngine()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(engine.root, "gait_trajectories", ts)
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(engine.root, "configs/config.yaml"), 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    indices = config.get('labelled_keyframes', list(range(len(engine.keyframes))))

    phases = np.linspace(0, 1, 100)
    data = {"knee": [], "ankle": [], "toe": []}
    ankle_angles = []

    for p in phases:
        angles = engine.interpolate_pose(p)
        rc = engine.get_leg_coords(angles)
        data["knee"].append((rc[2], rc[3]))
        data["ankle"].append((rc[4], rc[5]))
        data["toe"].append((rc[6], rc[7]))
        
        foot_v = np.array([rc[6]-rc[4], rc[7]-rc[5]])
        sh_v = np.array([rc[4]-rc[2], rc[5]-rc[3]])
        rel_deg = np.degrees(np.arctan2(foot_v[1], foot_v[0]) - np.arctan2(sh_v[1], sh_v[0]) - (np.pi/2))
        rel_deg = (rel_deg + 180) % 360 - 180
        ankle_angles.append(rel_deg)

    def save_path_plot(joint_name, coords, color, ylims):
        x, y = zip(*coords)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_aspect('equal')
        ax.axhline(0, color='black', lw=2, zorder=1)
        ax.plot(x, y, color=color, lw=3, label=f'{joint_name.capitalize()} Path')
        
        for i in indices:
            if i >= len(engine.keyframes): continue
            kf = engine.keyframes[i]
            rc = engine.get_leg_coords(kf[:3])
            idx = {"knee": 2, "ankle": 4, "toe": 6}[joint_name]
            ax.scatter(rc[idx], rc[idx+1], color=color, s=40, edgecolors='black', zorder=5)
            ax.text(rc[idx], rc[idx+1] + 0.05, f"{kf[3]*100:.0f}%", fontsize=8, ha='center')

        ax.set_title(f"{joint_name.capitalize()} Trajectory", fontweight='bold')
        ax.set_xlim(-0.8, 1.2)
        ax.set_ylim(ylims[0], ylims[1])
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.savefig(os.path.join(folder, f"{joint_name}_trajectory.png"), dpi=300)
        plt.close()

    def save_angle_plot(angles, phases, color):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.plot(phases * 100, angles, color=color, lw=3, label='Ankle Angle (A)')
        
        for i in indices:
            if i >= len(engine.keyframes): continue
            p = engine.keyframes[i][3]
            kf_angle = np.interp(p, phases, angles)
            ax.scatter(p * 100, kf_angle, color=color, s=50, edgecolors='black', zorder=5)
            ax.text(p * 100, kf_angle + 2, f"{p*100:.0f}%", fontsize=8, ha='center')

        ax.set_title("Ankle Angle", fontweight='bold')
        ax.set_ylim(-60, 60)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.savefig(os.path.join(folder, "ankle_angle_graph.png"), dpi=300)
        plt.close()

    save_path_plot("knee", data["knee"], "darkred", [-0.1, 1.0])
    save_path_plot("ankle", data["ankle"], "darkblue", [-0.1, 0.6])
    save_path_plot("toe", data["toe"], "blue", [-0.1, 0.6])
    save_angle_plot(np.array(ankle_angles), phases, "darkblue")