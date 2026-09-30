import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.theory import compute_theorem_constants, epsilon_fixed_point
from src.simulation import run_simulation

def main():

    b_base = (np.sqrt(2) + config.KAPPA0 * (config.N_MAX - 1)) / config.KAPPA
    c_bar_rho_base = config.C_BAR_RHO
    
    # Bounded maximum velocity
    v_bar = config.V_REF / (1 - b_base * config.V_REF * c_bar_rho_base)
    theo_bound = (v_bar**3) * c_bar_rho_base * (config.K_GAIN + 2 * b_base / config.KAPPA)

    alphas = np.linspace(0.2, 1.0, 12)
    
    eps_list = []
    cap_theo_list = []
    cap_sim_list = []
    max_u_list = []

    print("Running alpha scaling sweep (Unconstrained actuation)")
    for alpha in alphas:
        k_a = alpha * config.KAPPA
        K_a = config.K_GAIN / (alpha**2)
        c_rho_a = (alpha**2) * c_bar_rho_base

        consts = compute_theorem_constants(
            kappa=k_a, 
            kappa0=config.KAPPA0, 
            K=K_a, 
            N_max=config.N_MAX, 
            v_ref=config.V_REF, 
            c_bar_rho=c_rho_a,
            q=config.Q_EXP,
            omega=config.OMEGA
        )
        

        eps, _, _ = epsilon_fixed_point(consts, config.N_MAX, config.L, config.C_BAR_D)
        
        eps_list.append(eps)
        if eps is not None:
            cap_theo = 1.0 / (config.TAU_C + eps)
            cap_theo_list.append(cap_theo)
            
            params = dict(
                v_ref=config.V_REF, 
                tau_c=config.TAU_C, 
                s0=config.S0,
                kappa=k_a, 
                kappa0=config.KAPPA0, 
                K=K_a, 
                c_bar_d=config.C_BAR_D
            )
            rng = np.random.default_rng(42)
            
            ds_stable = min(0.05, 1.2 / (k_a * K_a))
            

            L_sim = 200.0
            
            s_vals, states, u_hist, _ = run_simulation(
                N=config.N_MAX, 
                l=L_sim, 
                ds=ds_stable,
                params=params, 
                eps=eps, 
                mode="random", 
                rng=rng, 
                use_cbf=False, 
                u_clip=None #Unconstrained actuation
            )
            
            # Record maximum unconstrained control effort
            sim_u = float(np.max(np.abs(u_hist)))
            max_u_list.append(sim_u)

            # Compute exit headway and empirical throughput
            t_exit = states[-1, :config.N_MAX + 1]
            empirical_headways = np.diff(t_exit)
            max_exit_gap = float(np.max(empirical_headways))
            sim_cap = 1.0 / max_exit_gap
            cap_sim_list.append(sim_cap)

            print(f"  alpha = {alpha:.2f} | eps = {eps:.4f}s | max |u| = {sim_u:.2f} m/s^2 | exit throughput = {sim_cap:.4f} veh/s")
        else:
            cap_theo_list.append(np.nan)
            cap_sim_list.append(np.nan)
            max_u_list.append(np.nan)


    fig1, ax1 = plt.subplots(figsize=(8, 4.8))
    
    line1 = ax1.plot(alphas, eps_list, color="tab:blue", marker="o", label=r"Theoretical Buffer $\epsilon$")
    ax1.set_xlabel(r"Scaling Factor $\alpha$")
    ax1.set_ylabel(r"Takeoff Headway Buffer $\epsilon$ [s]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    line2 = ax2.plot(alphas, cap_theo_list, color="tab:green", marker="s", label=r"Guaranteed Throughput $1/(\tau_c + \epsilon)$")
    line3 = ax2.plot(alphas, cap_sim_list, color="tab:purple", marker="^", linestyle="--", label=r"Simulated Exit Throughput")
    line4 = ax2.axhline(1.0 / config.TAU_C, color="black", linestyle=":", label=r"Nominal Capacity $1/\tau_c$")
    ax2.set_ylabel(r"Throughput [veh/s]", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines = line1 + line2 + line3 + [line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", framealpha=0.9)

    fig1.suptitle(r"Figure 1: Throughput Recovery via $\alpha$-Gain Scaling", fontsize=12)
    fig1.tight_layout()
    fig1.savefig("fig1_throughput_recovery.png", dpi=300)


    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, max_u_list, color="tab:orange", marker="o", label=r"Simulated $\max_{i, s} |u_i(s)|$")
    ax.axhline(theo_bound, color="red", linestyle="--", label=r"Theoretical Bound $\bar{v}^3 \bar{c}_\rho (K + 2b/\kappa)$")
    ax.set_xlabel(r"Scaling Factor $\alpha$ (Decreasing $\rightarrow$)")
    ax.set_ylabel(r"Max Acceleration $|u|$ [m/s$^2$]")
    ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title(r"Figure 2: Bounded Control Effort Under decreading $\alpha$", fontsize=12)
    ax.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig("fig2_control_bounded.png", dpi=300)

    print("Saved 'fig1_throughput_recovery.png' and 'fig2_control_bounded.png'.")

if __name__ == "__main__":
    main()