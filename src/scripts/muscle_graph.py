import numpy as np
import matplotlib.pyplot as plt

BETA = 0.1

def plot_muscle_curves():
    l_norm = np.linspace(0.4, 1.6, 500)
    v_norm = np.linspace(-1.0, 1.0, 500)

    f_al = np.exp(-((l_norm - 1)**2) / 0.45) 

    f_pe = np.array([(np.exp(5.0 * (ln - 1.0)) - 1.0) / (np.exp(5.0) - 1.0) if ln > 1.0 else 0 for ln in l_norm])

    f_v = []
    for vn in v_norm:
        if vn <= 0:
            f_v.append((1 + vn) / (1 - vn / 0.25))
        else:
            f_v.append((1.8 + 0.8 * (1 + vn) / (1 + vn / 0.25)) / 2.6)
    f_v = np.array(f_v)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(l_norm, f_al, 'b-', lw=2, label='Active (f_al)')
    ax1.plot(l_norm, f_pe, 'r--', lw=2, label='Passive (f_pe)')
    ax1.plot(l_norm, f_al + f_pe, 'k:', lw=1.5, label='Total (a=1.0, v=0)')
    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Force-Length Relationship')
    ax1.set_xlabel('Normalized Length (l/l_opt)')
    ax1.set_ylabel('Normalized Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(v_norm, f_v, 'g-', lw=2, label='Muscle Multiplier (f_v)')
    ax2.plot(v_norm, f_v + BETA * v_norm, 'm--', lw=1.5, label='f_v + Beta*v (Improved)')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Force-Velocity Relationship')
    ax2.set_xlabel('Normalized Velocity (v/v_max)')
    ax2.set_ylabel('Force Multiplier')
    ax2.annotate('Shortening\n(Concentric)', xy=(-0.5, 0.5), ha='center')
    ax2.annotate('Lengthening\n(Eccentric)', xy=(0.5, 1.3), ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_muscle_curves()