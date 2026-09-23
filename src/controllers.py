import numpy as np

def compute_nominal_control(s, t, v, e, Delta, s0, v_ref, kappa, kappa0, K):
    N = len(t) - 1
    u_nom = np.zeros(N + 1)
    
    # Leader
    delta0 = Delta[0] + kappa * e[0]
    u_nom[0] = (v[0] ** 3 / kappa) * (e[0] + kappa * K * delta0)
    
    # Followers
    for i in range(1, N + 1):
        Delta_i0 = t[i] - t[0] - i * (t[1] - t[0] - Delta[1])  # i * tau_c
        delta_i = (1 - kappa0) * Delta[i] + kappa0 * Delta_i0 + kappa * e[i]
        u_nom[i] = (v[i] ** 3 / kappa) * (e[i] - (1 - kappa0) * e[i - 1] - kappa0 * e[0] + kappa * K * delta_i)
        
    return u_nom

def compute_cbf_safe_control(i, v, e, u_nom, u_prev, d_prev, Delta_i, delta_min, gamma, alpha, c_bar_d):
    h_i = (Delta_i - delta_min) + gamma * (e[i] - e[i - 1])
    term1 = (v[i] ** 3 / gamma) * (alpha * h_i + (e[i] - e[i - 1]))
    term2 = (v[i] ** 3 / v[i - 1] ** 3) * (u_prev + d_prev)
    u_safe = term1 + term2 - c_bar_d
    return u_safe, h_i