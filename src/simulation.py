import numpy as np
from src import config
from src.controllers import compute_nominal_control, compute_cbf_safe_control

def sample_disturbance(N, c_bar_d, mode, rng):
    if mode == "random":
        return rng.uniform(-c_bar_d, c_bar_d, size=N + 1)
    elif mode == "worst_case":
        return np.full(N + 1, c_bar_d)
    elif mode == "crash":
        d = np.full(N + 1, c_bar_d)
        d[0] = -c_bar_d  # Leader headwind, followers tailwind
        return d
    raise ValueError(f"invalid")

def dynamics(s, state, N, d_vec, params, use_cbf=False, u_clip=None):
    t = state[:N + 1]
    v = state[N + 1:]
    v_ref = params.get("v_ref", config.V_REF)
    tau_c = params.get("tau_c", config.TAU_C)
    s0 = params.get("s0", config.S0)

    # Safe CBF parameter
    delta_min = params.get("delta_min", config.DELTA_MIN)
    gamma_cbf = params.get("gamma_cbf", config.GAMMA_CBF)
    alpha_cbf = params.get("alpha_cbf", config.ALPHA_CBF)
    c_bar_d = params.get("c_bar_d", config.C_BAR_D)

    e = 1.0 / v - 1.0 / v_ref
    Delta = np.zeros(N + 1)
    Delta[0] = t[0] - (s - s0) / v_ref
    for i in range(1, N + 1):
        Delta[i] = t[i] - t[i - 1] - tau_c

    u_nom = compute_nominal_control(s, t, v, e, Delta, s0, v_ref, params["kappa"], params["kappa0"], params["K"])
    u_applied = np.zeros(N + 1)
    h_vals = np.zeros(N + 1)

    # Leader
    u_applied[0] = np.clip(u_nom[0], -u_clip, u_clip) if u_clip is not None else u_nom[0]

    # Followers
    for i in range(1, N + 1):
        if use_cbf:
            u_safe, h_i = compute_cbf_safe_control(
                i, v, e, u_nom[i], u_applied[i - 1], d_vec[i - 1],
                Delta[i], delta_min, gamma_cbf, alpha_cbf, c_bar_d
            )
            h_vals[i] = h_i
            u_applied[i] = np.clip(u_nom[i], -u_clip, min(u_clip, u_safe)) if u_clip is not None else min(u_nom[i], u_safe)
        else:
            h_vals[i] = (Delta[i] - delta_min) + gamma_cbf * (e[i] - e[i - 1])
            u_applied[i] = np.clip(u_nom[i], -u_clip, u_clip) if u_clip is not None else u_nom[i]

    dt = 1.0 / v
    dv = (u_applied + d_vec) / v
    return np.concatenate([dt, dv]), u_applied, h_vals

def rk4_step(s, state, ds, N, d_vec, params, use_cbf, u_clip):
    k1, u_app, h_val = dynamics(s, state, N, d_vec, params, use_cbf, u_clip)
    k2, _, _ = dynamics(s + ds / 2, state + ds / 2 * k1, N, d_vec, params, use_cbf, u_clip)
    k3, _, _ = dynamics(s + ds / 2, state + ds / 2 * k2, N, d_vec, params, use_cbf, u_clip)
    k4, _, _ = dynamics(s + ds, state + ds * k3, N, d_vec, params, use_cbf, u_clip)
    return state + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4), u_app, h_val

def run_simulation(N, l, ds, params, eps, mode, rng, use_cbf=False, u_clip=None):
    t0 = np.zeros(N + 1)
    t0[0] = eps
    for i in range(1, N + 1):
        t0[i] = t0[i - 1] + params["tau_c"] + eps
    v0 = np.full(N + 1, params["v_ref"])

    n_steps = int(np.ceil(l / ds))
    s_vals = params["s0"] + np.arange(n_steps + 1) * ds
    states = np.zeros((n_steps + 1, 2 * (N + 1)))
    u_hist = np.zeros((n_steps, N + 1))
    h_hist = np.zeros((n_steps, N + 1))
    states[0] = np.concatenate([t0, v0])

    for k in range(n_steps):
        d_vec = sample_disturbance(N, params["c_bar_d"], mode, rng)
        states[k + 1], u_hist[k], h_hist[k] = rk4_step(s_vals[k], states[k], ds, N, d_vec, params, use_cbf, u_clip)

    return s_vals, states, u_hist, h_hist