import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.simulation import run_simulation

def main():
    params = dict(
        v_ref=config.V_REF, tau_c=config.TAU_C, s0=config.S0,
        kappa=config.KAPPA, kappa0=config.KAPPA0, K=config.K_GAIN,
        delta_min=config.DELTA_MIN, gamma_cbf=config.GAMMA_CBF,
        alpha_cbf=config.ALPHA_CBF, c_bar_d=3.0 # Severe Storm
    )

    print("Simulating cases under severe disturbance")
    rng = np.random.default_rng(42)
    s, st_uncon, _, _ = run_simulation(config.N_MAX, config.L, config.DS, params, 0.0, "crash", rng, use_cbf=False, u_clip=None)
    
    rng = np.random.default_rng(42)
    _, st_sat, _, h_sat = run_simulation(config.N_MAX, config.L, config.DS, params, 0.0, "crash", rng, use_cbf=False, u_clip=config.U_MAX)
    
    rng = np.random.default_rng(42)
    _, st_cbf, _, h_cbf = run_simulation(config.N_MAX, config.L, config.DS, params, 0.0, "crash", rng, use_cbf=True, u_clip=config.U_MAX)

    gap_uncon = np.diff(st_uncon[:, :config.N_MAX + 1], axis=1) - config.TAU_C
    gap_sat = np.diff(st_sat[:, :config.N_MAX + 1], axis=1) - config.TAU_C
    gap_cbf = np.diff(st_cbf[:, :config.N_MAX + 1], axis=1) - config.TAU_C

    # Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    axes[0].plot(s, gap_uncon)
    axes[0].axhline(config.DELTA_MIN, color="red", linestyle="--")
    axes[0].set_title("1. Unconstrained")
    axes[0].set_ylabel(r"$\Delta_i(s)$ (s)")

    axes[1].plot(s, gap_sat)
    axes[1].axhline(config.DELTA_MIN, color="red", linestyle="--")
    axes[1].set_title("2. Saturated Actuation (No CBF)")

    axes[2].plot(s, gap_cbf)
    axes[2].axhline(config.DELTA_MIN, color="red", linestyle="--", label=r"Safety Limit $\Delta^{min}$")
    axes[2].set_title("3. Saturated Actuation w/ CBF")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig("fig3_cbf_comparison.png", dpi=150)

    # Barrier Plot
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for i in range(1, config.N_MAX + 1):
        ax.plot(s[:-1], h_sat[:, i], linestyle=":", label=f"Veh {i} (No CBF)")
        ax.plot(s[:-1], h_cbf[:, i], label=f"Veh {i} (WITH CBF)")
    ax.axhline(0, color="red", linestyle="--", label="Safe Boundary")
    ax.set_title("Forward Invariance of Safe Set")
    ax.set_ylabel(r"$h_i(s)$")
    ax.set_xlabel("Distance (m)")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig("fig4_barrier_invariance.png", dpi=150)
    print("Saved figures.")

if __name__ == "__main__":
    main()