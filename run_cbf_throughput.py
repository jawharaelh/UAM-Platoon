import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.simulation import run_simulation

def main():

    # Disturbance magnitude
    disturbances = np.linspace(0.0, 3.0, 25)

    effective_throughput_cbf = []
    effective_throughput_nocbf = []
    cbf_intervention_pct = []
    min_barrier_cbf = []
    min_barrier_nocbf = []

    params = dict(
        v_ref=config.V_REF, tau_c=config.TAU_C, s0=config.S0,
        kappa=config.KAPPA, kappa0=config.KAPPA0, K=config.K_GAIN,
        delta_min=config.DELTA_MIN, gamma_cbf=config.GAMMA_CBF,
        alpha_cbf=config.ALPHA_CBF
    )

    for d_val in disturbances:
        params["c_bar_d"] = d_val

        # Saturated w/ CBF
        rng = np.random.default_rng(42)
        s, st_cbf, u_cbf, h_cbf = run_simulation(
            config.N_MAX, config.L, config.DS, params, 0.0, "crash", rng,
            use_cbf=True, u_clip=config.U_MAX
        )
        
        # Measure minimum barrier value
        min_barrier_cbf.append(np.min(h_cbf[:, 1:]))
        
        # Exit time headway between consecutive vehicles at conflict point
        exit_times_cbf = st_cbf[-1, :config.N_MAX + 1]
        exit_headways_cbf = np.diff(exit_times_cbf)
        
        # Worst-case inter-exit interval determines the bottleneck throughput
        max_exit_gap_cbf = np.max(exit_headways_cbf)
        # Calculate maximum spacing compression along the corridor
       # deal with NaN
        gaps_cbf = np.diff(st_cbf[:, :config.N_MAX + 1], axis=1) - config.TAU_C
        
        if not np.isnan(gaps_cbf).any():
            worst_compression = max(0.0, -np.min(gaps_cbf))
            tau_safe = config.TAU_C + worst_compression
            effective_throughput_cbf.append(1.0 / tau_safe)
        else:
            effective_throughput_cbf.append(np.nan)

        # Saturated w/o CBF
        rng = np.random.default_rng(42)
        _, st_nocbf, _, h_nocbf = run_simulation(
            config.N_MAX, config.L, config.DS, params, 0.0, "crash", rng,
            use_cbf=False, u_clip=config.U_MAX
        )
        min_barrier_nocbf.append(np.min(h_nocbf[:, 1:]))
        
        exit_times_nocbf = st_nocbf[-1, :config.N_MAX + 1]
        exit_headways_nocbf = np.diff(exit_times_nocbf)
        max_exit_gap_nocbf = np.max(exit_headways_nocbf)
        effective_throughput_nocbf.append(1.0 / max_exit_gap_nocbf)

    # Plotting trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Safety Guarantee
    ax1.plot(disturbances, min_barrier_cbf, color="tab:green", marker="o", linewidth=2, label="With CBF")
    ax1.plot(disturbances, min_barrier_nocbf, color="tab:red", linestyle="--", marker="x", linewidth=2, label="Without CBF")
    ax1.axhline(0, color="black", linestyle=":", label="Safety Boundary")
    ax1.set_xlabel(r"Disturbance Magnitude ($m/s^2$)")
    ax1.set_ylabel(r"MinimumSafety Margin")
    ax1.set_title("Barrier Preservation")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Throughput Degradation
    ax2.plot(disturbances, effective_throughput_cbf, color="tab:blue", marker="s", linewidth=2, label="Guaranteed Safe Throughput (CBF)")
    ax2.axhline(1.0 / config.TAU_C, color="black", linestyle="--", label=r"Nominal Throughput ($1/\tau_c$)")
    ax2.set_xlabel(r"Disturbance Severity / Wind Gust ($m/s^2$)")
    ax2.set_ylabel(r"Effective Exit Throughput (veh/s)")
    ax2.set_title("Throughput Degradation")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Safety-Throughput Trade-off", fontsize=14)
    fig.tight_layout()
    fig.savefig("cbf_throughput_tradeoff.png", dpi=150)
    print("Saved cbf_throughput_tradeoff.png successfully!")

if __name__ == "__main__":
    main()