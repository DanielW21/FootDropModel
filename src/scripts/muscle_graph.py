import numpy as np
import matplotlib.pyplot as plt

BETA = 0.1
BASE_WIDTH = 0.45
TILT = 0.35 
ECC_MAX = 1.8
K1, K2 = 0.25, 1.0 
PSHAPE = 5.0
PSTRAIN = 0.75

def plot_muscle_curves():
    l_norm = np.linspace(0.4, 1.8, 500)
    v_norm = np.linspace(-1.0, 1.0, 500)

    f_al_tilt = np.exp(-((l_norm - 1)**2) / (BASE_WIDTH + TILT * (l_norm - 1)))
    f_pe = np.where(l_norm > 1.0, (np.exp(PSHAPE/PSTRAIN * (l_norm - 1.0)) - 1.0) / (np.exp(PSHAPE) - 1.0), 0)

    def calc_fv(K):
        v_crit = (ECC_MAX - 1) / (1 + 1/K)
        return np.where(v_norm <= 0, 
                        (1 + v_norm) / (1 - v_norm / K), 
                        (ECC_MAX * v_norm + v_crit) / (v_norm + v_crit))
    
    f_v1 = calc_fv(K1)
    f_v2 = calc_fv(K2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(l_norm, f_al_tilt, 'b-', lw=2.5, label='Active Force, ($f^{L}$)')
    ax1.plot(l_norm, f_pe, 'r--', lw=2, label='Passive Force ($f^{PE}$)')
    ax1.plot(l_norm, f_al_tilt + f_pe, 'k-', lw=1.5, label='Total Force Sum')
    
    ax1.set_title('A. force-length ($f^L$ & $f^{PE}$)', loc='left', pad=15)
    ax1.set_xlabel('Normalized Length ($\ell/\ell^M_o$)')
    ax1.set_ylabel('Normalized Force ($f/f^M_o$)')
    
    ax1.set_xticks([0.52, 1.0, 1.39, 1.62, 1.8])
    ax1.set_yticks([0, 1.0])
    ax1.set_ylim(-0.05, 1.2) 
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False)

    ax2.plot(v_norm, f_v1, 'g-', lw=2.5, label=f'$f^V$ (K={K1})')
    ax2.plot(v_norm, f_v1 + BETA * v_norm, 'g--', alpha=0.3)
    
    ax2.plot(v_norm, f_v2, 'b-', lw=2.5, label=f'$f^V$ (K={K2})')
    ax2.plot(v_norm, f_v2 + BETA * v_norm, 'b--', alpha=0.3)

    ax2.set_title('B. force-velocity ($f^V$)', loc='left', pad=15)
    ax2.set_xlabel('Normalized Velocity ($v/v^M_{max}$)')
    ax2.set_ylabel('Force Multiplier ($f/f^M_o$)')
    
    ax2.axhline(1.0, color='gray', lw=0.8, alpha=0.5)
    ax2.axvline(0.0, color='gray', lw=0.8, alpha=0.5)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([0, 1.0, ECC_MAX])
    ax2.set_ylim(-0.05, ECC_MAX + 0.2)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_muscle_curves()