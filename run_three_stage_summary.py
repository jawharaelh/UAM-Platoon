"""
Three-stage comparison
  1. Unconstrained: plain string-stability controller, no actuator limit, no CBF
  2. Saturated Actuation (No CBF): same controller, hard-clipped at the actuator limit U_MAX, no proactive safety logic
  3. Saturated Actuation w/ CBF: actuator-clipped & CBF safety filter active

plotsthe minimum takeoff-delay epsilon, the smallest barrier value h_i
reaches along the trajectory, and the resulting throughput 1/(tau_c+epsilon).

The three stages only diverge once the disturbance is large enough to
make saturation/CBF intervention happen at all

epsilon is measured EMPIRICALLY rather than from the theorem's closed-form fixed point: it is
defined (following sec:take-off-epsilon) as the asymptotic gap-deviation floor |Delta_i(s)|, so
we run each stage once with zero initial delay and read off the worst |Delta_i| over the last
10% of the corridor
"""

import numpy as np
import matplotlib.pyplot as plt

from src import config
from src.simulation import run_simulation

C_BAR_D = 3.0
TAIL_FRACTION = 0.1


def empirical_epsilon(s_vals, states, N, tau_c):
    t = states[:-1, :N + 1]
    tail_start = int(len(s_vals) * (1 - TAIL_FRACTION))
    Delta = np.diff(t[tail_start:], axis=1) - tau_c
    return float(np.max(np.abs(Delta)))


def main():
    base_params = dict(v_ref=config.V_REF, tau_c=config.TAU_C, s0=config.S0,
                        kappa=config.KAPPA, kappa0=config.KAPPA0, K=config.K_GAIN,
                        delta_min=config.DELTA_MIN, gamma_cbf=config.GAMMA_CBF,
                        alpha_cbf=config.ALPHA_CBF, c_bar_d=C_BAR_D)

    stage_defs = [
        ("1. Unconstrained", dict(use_cbf=False, u_clip=None)),
        ("2. Saturated Actuation (No CBF)", dict(use_cbf=False, u_clip=config.U_MAX)),
        ("3. Saturated Actuation w/ CBF", dict(use_cbf=True, u_clip=config.U_MAX)),
    ]

    results = {}
    for label, sim_kwargs in stage_defs:
        rng = np.random.default_rng(42)
        s_vals, states, u_hist, h_hist = run_simulation(config.N_MAX, config.L, config.DS, base_params, 0.0, "crash", rng, **sim_kwargs)
        eps = empirical_epsilon(s_vals, states, config.N_MAX, config.TAU_C)
        h_min = float(np.min(h_hist[:, 1:]))
        throughput = 1.0 / (config.TAU_C + eps)
        max_u = float(np.max(np.abs(u_hist)))

        results[label] = dict(epsilon=eps, h_min=h_min, throughput=throughput, max_u=max_u)

    print(f"\n{'stage':<36}{'epsilon':>12}{'h_min':>12}{'throughput':>12}{'max|u|':>12}")
    for label, r in results.items():
        print(f"{label:<36}{r['epsilon']:>12.4g}{r['h_min']:>12.4g}{r['throughput']:>12.4g}{r['max_u']:>12.4g}")

    labels = list(results.keys())
    short_labels = [l.split(". ", 1)[1] for l in labels]
    eps_vals = [results[l]["epsilon"] for l in labels]
    h_vals = [results[l]["h_min"] for l in labels]
    tp_vals = [results[l]["throughput"] for l in labels]
    u_vals = [results[l]["max_u"] for l in labels]

    def label_bars(ax, values, fmt="{:.3g}"):
        ylo, yhi = ax.get_ylim()
        pad = 0.03 * (yhi - ylo)
        for i, val in enumerate(values):
            offset = pad if val >= 0 else -pad
            va = "bottom" if val >= 0 else "top"
            ax.text(i, val + offset, fmt.format(val), ha="center", va=va, fontsize=9)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    axes[0].bar(short_labels, eps_vals, color="tab:blue")
    axes[0].set_title(r"Minimum $\epsilon$ (empirical gap-deviation floor)")
    axes[0].tick_params(axis="x", rotation=20)
    label_bars(axes[0], eps_vals)

    axes[1].bar(short_labels, h_vals, color="tab:orange")
    axes[1].axhline(0, color="red", linestyle="--", label="safety boundary")
    axes[1].set_title(r"Smallest $h$")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)
    label_bars(axes[1], h_vals)

    axes[2].bar(short_labels, tp_vals, color="tab:green")
    axes[2].axhline(1.0 / config.TAU_C, color="black", linestyle=":", label=r"nominal $1/\tau_c$")
    axes[2].set_title("Throughput")
    axes[2].legend()
    axes[2].tick_params(axis="x", rotation=20)
    label_bars(axes[2], tp_vals)

    axes[3].bar(short_labels, u_vals, color="tab:purple")
    axes[3].axhline(config.U_MAX, color="red", linestyle="--", linewidth=1.5, label=f"Engine Limit ({config.U_MAX})")
    axes[3].set_title(r"Peak Control Effort $\max |u_i|$")
    axes[3].legend(loc="upper right")
    axes[3].tick_params(axis="x", rotation=20)
    label_bars(axes[3], u_vals)

    fig.suptitle(f"Three-stage comparison under disturbance")
    fig.tight_layout()
    fig.savefig("fig5_three_stage_summary.png", dpi=150)


if __name__ == "__main__":
    main()
