import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os

# ANTROPOMETRIC CONSTANTS
THIGH, SHANK, FOOT = 0.6164, 0.5588, 0.241
HIP_HEIGHT = 1.0892 

# REFINED KEYFRAME DEFINITIONS
# [Hip, Knee, Ankle, Target Phase (% of gait cycle)]
keyframes = np.array([
    [ 0.40, 0.00,  0.00, 0.00], # KF 0: HS
    [ 0.20, 0.25, -0.10, 0.12], # KF 1: CTO
    [ 0.00, 0.05,  0.00, 0.30], # KF 2: Mid Stance
    [-0.25, 0.05,  0.00, 0.50], # KF 3: CHS 
    [-0.30, 0.30, -0.40, 0.60], # KF 4: TO
    [ 0.10, 1.40,  -0.40, 0.75], # KF 5: Initial Swing
    [ 0.55, 0.80,  0.10, 0.88], # KF 6: Mid Swing 
    [ 0.45, 0.10,  0.00, 1.00]  # KF 7: HS
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

output_folder = "gait_keyframes"
os.makedirs(output_folder, exist_ok=True)

for i, kf in enumerate(keyframes):
    fig_kf, ax_kf = plt.subplots(figsize=(10, 7))
    ax_kf.set_aspect('equal')
    ax_kf.set_xlim(-1.0, 1.5)
    ax_kf.set_ylim(-0.2, 1.4)
    ax_kf.grid(True, which='major', color='#CCCCCC', linestyle='--')
    ax_kf.axhline(0, color='black', lw=2)
    ax_kf.fill_between([-1.5, 2], -0.2, 0, color='#f0f0f0', alpha=0.5)

    phase = kf[3]
    main, sub = get_clinical_labels(phase)
    r_pts = get_leg_coords(kf[:3])
    l_pts = get_leg_coords(interpolate_pose((phase + 0.5) % 1.0))

    ax_kf.plot(l_pts[0::2], l_pts[1::2], 'o-', lw=6, ms=8, color='red', alpha=0.3)
    ax_kf.plot(r_pts[0::2], r_pts[1::2], 'o-', lw=6, ms=8, color='blue', mfc='white')
    
    label = f"KF {i} | {phase*100:.1f}%\n{main}\n{sub}"
    ax_kf.text(0.95, 0.95, label, transform=ax_kf.transAxes, family='monospace', va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(output_folder, f"keyframe_{i}.png"), dpi=150)
    plt.close(fig_kf)

frames_per_cycle = 100
all_data = []
with open('gait_regeneration_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame', 'phase', 'main_phase', 'sub_phase', 'hx', 'hy', 'kx', 'ky', 'ax', 'ay', 'tx', 'ty'])
    for f in range(frames_per_cycle):
        phase = f / frames_per_cycle
        main, sub = get_clinical_labels(phase)
        coords = get_leg_coords(interpolate_pose(phase))
        writer.writerow([f, round(phase, 3), main, sub] + [round(c, 4) for c in coords])
        all_data.append({'coords': coords, 'main': main, 'sub': sub})

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_xlim(-1.0, 1.5)
ax.set_ylim(-0.2, 1.4)
ax.axhline(0, color='black', lw=2)
line_r, = ax.plot([], [], 'o-', lw=6, ms=8, color='blue', mfc='white')
line_l, = ax.plot([], [], 'o-', lw=6, ms=8, color='red', alpha=0.3)
info_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, family='monospace', va='top', ha='right')

def update(frame):
    r, l = all_data[frame], all_data[(frame + 50) % 100]
    line_r.set_data(r['coords'][0::2], r['coords'][1::2])
    line_l.set_data(l['coords'][0::2], l['coords'][1::2])
    info_text.set_text(f"{r['main']}\n{r['sub']}\nPHASE: {(frame/100)*100:4.1f}%")
    return line_r, line_l, info_text

ani = FuncAnimation(fig, update, frames=frames_per_cycle, interval=5, blit=False)
plt.show()