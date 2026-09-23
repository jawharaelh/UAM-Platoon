import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.theory import compute_theorem_constants, epsilon_fixed_point
from src.simulation import run_simulation

def main():
    b_base = (np.sqrt(2) + config.KAPPA0 * (config.N_MAX - 1)) / config.KAPPA
    c_bar_rho_base = 0.5 / (b_base * config.V_REF)
    v_bar = config.V_REF / (1 - b_base * config.V_REF * c_bar_rho_base)
    theo_bound = (v_bar**3) * c_bar_rho_base * (config.K_GAIN + 2 * b_base / config.KAPPA)

    alphas = np.linspace(0.05, 1.0, 25)
    eps_list, cap_list, max_u_list = [], [], []

    print("Running alpha sweep...")
    for alpha in alphas:
        k_a = alpha * config.KAPPA
        K_a = config.K_GAIN / (alpha**2)
        c_rho_a = (alpha**2) * c_bar_rho_base

        consts = compute_theorem_constants(k_a, config.KAPPA0, K_a, config.N_MAX, config.V_REF, c_rho_a)
        eps, _, _ = epsilon_fixed_point(consts, config.N_MAX, config.L, config.C_BAR_D)
        
        eps_list.append(eps)
        if eps is not None:
            cap_list.append(1.0 / (config.TAU_C + eps))
            params = dict(v_ref=config.V_REF, tau_c=config.TAU_C, s0=config.S0,
                          kappa=k_a, kappa0=config.KAPPA0, K=K_a, c_bar_d=config.C_BAR_D)
            rng = np.random.default_rng(42)
            _, _, u_hist, _ = run_simulation(config.N_MAX, config.L, config.DS, params, eps, "random", rng)
            max_u_list.append(np.max(np.abs(u_hist)))
        else:
            cap_list.append(np.nan)
            max_u_list.append(np.nan)

    # Figure 1
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(alphas, eps_list, color="tab:blue", marker="o")
    ax1.set_xlabel(r"Scaling Factor $\alpha$")
    ax1.set_ylabel(r"Delay $\epsilon$", color="tab:blue")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(alphas, cap_list, color="tab:green", marker="s")
    ax2.axhline(1.0 / config.TAU_C, color="black", linestyle="--")
    ax2.set_ylabel(r"Throughput (veh/s)", color="tab:green")
    fig1.suptitle(r"Figure 1: Throughput Recovery as $\alpha$ converges")
    fig1.tight_layout()
    fig1.savefig("fig1_throughput_recovery.png", dpi=150)

    # Figure 2
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, max_u_list, color="tab:orange", marker="o", label="Simulated max $|u|$")
    ax.axhline(theo_bound, color="red", linestyle="--", label="Theoretical Bound")
    ax.set_xlabel(r"Scaling Factor $\alpha$")
    ax.set_ylabel(r"Acceleration $|u|$")
    ax.invert_xaxis()
    ax.set_title(r"Figure 2: Bounded Control Effort")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig("fig2_control_bounded.png", dpi=150)
    print("Saved figures.")

if __name__ == "__main__":
    main()