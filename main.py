import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# Modify these & keyframes to accurately model the gait cycle 
# Anthropometric data is 50%tile of male from sample population
THIGH, SHANK, FOOT = 0.6164, 0.5588, 0.241
HIP_HEIGHT = 1.0892 

keyframes = np.array([
    [ 0.40, 0.00,  0.00, 0.00],
    [ 0.20, 0.25, -0.10, 0.12],
    [ 0.00, 0.05,  0.00, 0.30],
    [-0.25, 0.05,  0.00, 0.50],
    [-0.30, 0.30, -0.40, 0.60],
    [ 0.10, 1.40,  -0.40, 0.75],
    [ 0.55, 0.80,  0.40, 0.88],
    [ 0.45, 0.10,  0.00, 1.00]
])

def get_clinical_labels(phase):
    p = phase * 100
    if p < 60:
        main = "STANCE PHASE"
        if p < 10:   sub = "HS (Heel Strike)"
        elif p < 20: sub = "CTO (Contralat. Toe Off)"
        elif p < 45: sub = "Mid Stance"
        else:        sub = "CHS (Contralat. Heel Strike)"
    else:
        main = "SWING PHASE"
        if p < 75:   sub = "Initial Swing"
        elif p < 88: sub = "Mid Swing"
        else:        sub = "Terminal Swing / HS"
    return main, sub

def interpolate_pose(phase):
    phase = phase % 1.0
    times = keyframes[:, 3]
    idx = np.searchsorted(times, phase) - 1
    idx = max(0, min(idx, len(keyframes) - 2))
    t0, t1 = times[idx], times[idx+1]
    frac = (phase - t0) / (t1 - t0)
    return keyframes[idx, :3] * (1 - frac) + keyframes[idx+1, :3] * frac

def get_leg_coords(angles):
    h_ang, k_ang, a_ang = angles
    hip = np.array([0, HIP_HEIGHT])
    knee = hip + np.array([THIGH * np.sin(h_ang), -THIGH * np.cos(h_ang)])
    ankle = knee + np.array([SHANK * np.sin(h_ang - k_ang), -SHANK * np.cos(h_ang - k_ang)])
    toe = ankle + np.array([FOOT * np.cos(a_ang), FOOT * np.sin(a_ang)])
    return [hip[0], hip[1], knee[0], knee[1], ankle[0], max(0, ankle[1]), toe[0], max(0, toe[1])]

# KEYFRAME SAVE FILES
output_folder = "gait_keyframes"
os.makedirs(output_folder, exist_ok=True)

for i, kf in enumerate(keyframes):
    fig_kf, ax_kf = plt.subplots(figsize=(10, 7))
    ax_kf.set_aspect('equal')
    ax_kf.set_xlim(-1.0, 1.5)
    ax_kf.set_ylim(-0.2, 1.4)
    ax_kf.axhline(0, color='black', lw=2)
    
    phase = kf[3]
    main, sub = get_clinical_labels(phase)
    rc = get_leg_coords(kf[:3])
    lc = get_leg_coords(interpolate_pose((phase + 0.5) % 1.0))
    degs = np.degrees(kf[:3])

    # Plot legs
    ax_kf.plot(lc[0::2], lc[1::2], 'o-', lw=4, ms=6, color='red', alpha=0.15)
    ax_kf.plot(rc[0::2], rc[1::2], 'o-', lw=6, ms=8, color='blue', mfc='white')
    
    # Dotted Refs
    ax_kf.plot([rc[0], rc[0]], [rc[1], rc[1]-0.3], 'k--', lw=1, alpha=0.5)
    ax_kf.plot([rc[0], rc[2], rc[2] + (rc[2]-rc[0])*0.5], [rc[1], rc[3], rc[3] + (rc[3]-rc[1])*0.5], 'k--', lw=1, alpha=0.5)
    
    # Angle Labels
    ax_kf.text(rc[0]+0.05, rc[1]-0.05, f"θ: {degs[0]:.0f}°", color='blue', fontweight='bold')
    ax_kf.text(rc[2]+0.05, rc[3]-0.05, f"θ: {degs[1]:.0f}°", color='blue', fontweight='bold')
    
    label = f"KF {i} | {phase*100:.1f}%\n{main}\n{sub}"
    ax_kf.text(0.95, 0.05, label, transform=ax_kf.transAxes, family='monospace', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(output_folder, f"keyframe_{i}.png"), dpi=150)
    plt.close(fig_kf)

# SIMULATION
frames_per_cycle = 100
all_data = []
for f in range(frames_per_cycle):
    phase = f / frames_per_cycle
    angles = interpolate_pose(phase)
    coords = get_leg_coords(angles)
    main, sub = get_clinical_labels(phase)
    all_data.append({'coords': coords, 'angles': angles, 'main': main, 'sub': sub})

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.5)
ax.set_ylim(-0.2, 1.4)
ax.axhline(0, color='black', lw=2)

line_l, = ax.plot([], [], 'o-', lw=4, ms=6, color='red', alpha=0.2)
line_r, = ax.plot([], [], 'o-', lw=6, ms=8, color='blue', mfc='white', zorder=10)
ref_hip, = ax.plot([], [], 'k--', lw=1, alpha=0.5)
ref_knee, = ax.plot([], [], 'k--', lw=1, alpha=0.5)
txt_hip = ax.text(0, 0, '', color='blue', fontweight='bold')
txt_knee = ax.text(0, 0, '', color='blue', fontweight='bold')
info_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, family='monospace', va='top', ha='right', bbox=dict(facecolor='white', alpha=0.7))

def update(frame):
    r, l = all_data[frame], all_data[(frame + 50) % 100]
    rc = r['coords']
    line_r.set_data(rc[0::2], rc[1::2])
    line_l.set_data(l['coords'][0::2], l['coords'][1::2])
    
    ref_hip.set_data([rc[0], rc[0]], [rc[1], rc[1]-0.3])
    ref_knee.set_data([rc[0], rc[2], rc[2] + (rc[2]-rc[0])*0.5], [rc[1], rc[3], rc[3] + (rc[3]-rc[1])*0.5])

    degs = np.degrees(r['angles'])
    txt_hip.set_position((rc[0]+0.05, rc[1]-0.05))
    txt_hip.set_text(f"θ: {degs[0]:.0f}°")
    txt_knee.set_position((rc[2]+0.05, rc[3]-0.05))
    txt_knee.set_text(f"θ: {degs[1]:.0f}°")

    info_text.set_text(f"{r['main']}\n{r['sub']}\nPHASE: {(frame/100)*100:4.1f}%")
    return line_r, line_l, ref_hip, ref_knee, txt_hip, txt_knee, info_text

ani = FuncAnimation(fig, update, frames=frames_per_cycle, interval=25, blit=False) # Modify interval to change speed of animation
plt.show()