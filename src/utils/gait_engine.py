import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

class GaitEngine:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(self.root, "configs/config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
        
        self.params = config['anthropometrics']
        self.keyframes = np.array(config['keyframes'])

    def get_clinical_labels(self, phase):
        p = phase * 100
        if p < 60:
            main = "STANCE PHASE"
            if p < 10:   sub = "HS (Heel Strike)"
            elif p < 20: sub = "Controlat. Foot Off (Foot Flat)"
            elif p < 45: sub = "Mid Stance"
            else:        sub = "Controlat. Foot Strike (Heel Off)"
        else:
            main = "SWING PHASE"
            if p < 75:    sub = "Initial Swing"
            elif p < 88:  sub = "Mid Swing"
            elif p < 100: sub = "Terminal Swing"
            else:         sub = "HS (Heel Strike)"
            
        return main, sub

    def interpolate_pose(self, phase):
        phase %= 1.0
        times = self.keyframes[:, 3]
        idx = np.searchsorted(times, phase) - 1
        idx = max(0, min(idx, len(self.keyframes) - 2))
        t0, t1 = times[idx], times[idx+1]
        frac = (phase - t0) / (t1 - t0)
        return self.keyframes[idx, :3] * (1 - frac) + self.keyframes[idx+1, :3] * frac

    def get_leg_coords(self, angles):
        h_ang, k_ang, a_ang = angles
        hip = np.array([0, self.params['hip_height']])
        knee = hip + np.array([self.params['thigh'] * np.sin(h_ang), -self.params['thigh'] * np.cos(h_ang)])
        ankle = knee + np.array([self.params['shank'] * np.sin(h_ang - k_ang), -self.params['shank'] * np.cos(h_ang - k_ang)])
        toe = ankle + np.array([self.params['foot'] * np.cos(a_ang), self.params['foot'] * np.sin(a_ang)])
        return [hip[0], hip[1], knee[0], knee[1], ankle[0], max(0, ankle[1]), toe[0], max(0, toe[1])]

    def save_to_csv(self, samples=500):
        output_dir = os.path.join(self.root, "output", "sim_data")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        data_log = []
        for f in range(samples):
            phase = f / samples
            angles = self.interpolate_pose(phase)
            c = self.get_leg_coords(angles)
            data_log.append({
                'phase': phase, 'hip_q': angles[0], 'knee_q': angles[1], 'ankle_q': angles[2],
                'toe_x': c[6], 'toe_y': c[7]
            })
        
        df = pd.DataFrame(data_log)
        path = os.path.join(output_dir, f"gait_ref_{ts}.csv")
        df.to_csv(path, index=False)
        return path