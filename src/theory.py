import numpy as np

def compute_theorem_constants(kappa, kappa0, K, N_max, v_ref, c_bar_rho, q=1.0, omega=None):
    b = (np.sqrt(2) + kappa0 * (N_max - 1)) / kappa
    if omega is None:
        target_pow = 1.0 - kappa0 / 2.0
        omega = target_pow ** (1.0 / q)
    else:
        assert 1 - kappa0 < omega ** q < 1, (
            f"Invalid choice: requires 1 - kappa0 ({1 - kappa0}) < omega^q ({omega**q}) < 1"
        )

    lam = 0.5 * min(kappa * K, 1.0 / kappa)
    C = 1.0 / (1 - lam * kappa)
    Cx = Cy = C + 1
    Ax = Cx + Cy / kappa0
    Bq = Cy * (Cx + 2 * Cy / kappa0) / ((1 - omega) ** q * (omega ** q - (1 - kappa0)))

    v_min = v_ref / (1 + b * v_ref * c_bar_rho)
    L = kappa / (v_min ** 3)
    sigma_Delta = L / (kappa * K) * (1 + 4 / kappa0)

    return dict(kappa=kappa, kappa0=kappa0, K=K, b=b, q=q, omega=omega, lam=lam, C=C,
                Cx=Cx, Cy=Cy, Ax=Ax, Bq=Bq, v_min=v_min, L=L, sigma_Delta=sigma_Delta)

def h_q(r, consts):
    lam, q = consts["lam"], consts["q"]
    r = np.asarray(r, dtype=float)
    small_r = np.exp(-lam * r / 2)
    with np.errstate(divide="ignore"):
        large_r = (2 * q / (np.e * lam * np.where(r > 0, r, 1.0))) ** q
    return np.where(lam * r <= 2 * q, small_r, large_r)

def beta_Delta(r, consts):
    lam, Bq, kappa = consts["lam"], consts["Bq"], consts["kappa"]
    return 2 * np.exp(-lam * r) + Bq * (np.exp(-r / (2 * kappa)) + h_q(r, consts))

def epsilon_fixed_point(consts, N_max, l, c_bar_d):
    m = (N_max - 1) * consts["kappa0"] + 2
    beta_l = beta_Delta(l, consts)
    denom = 1 - m * beta_l
    if denom <= 0:
        return None, m, beta_l
    return consts["sigma_Delta"] * c_bar_d / denom, m, beta_l

def check_admissibility(consts, c_bar, c_bar_d, c_bar_rho, eps=None):
    """
    Verifies the domain-of-attraction conditions from Theorem 1 and Section VII.
    """
    b = consts["b"]
    v_ref = consts.get("v_ref", config.V_REF)
    kappa = consts["kappa"]
    kappa0 = consts["kappa0"]
    K = consts["K"]
    L = consts["L"]
    Ax = consts["Ax"]

    c_rho_max = 1.0 / (b * v_ref)
    cond1 = c_bar_rho < c_rho_max
    lhs_23 = Ax * c_bar + (L / (kappa * K)) * (2.0 + 2.0 / kappa0) * c_bar_d
    cond2 = lhs_23 < c_bar_rho

    cond3 = True
    if eps is not None:
        m = (config.N_MAX - 1) * kappa0 + 2
        cond3 = (m * eps < c_bar)

    return {
        "c_rho_valid": cond1,
        "c_rho_max": c_rho_max,
        "eq23_valid": cond2,
        "eq23_lhs": lhs_23,
        "initial_eps_valid": cond3,
        "all_valid": cond1 and cond2 and cond3
    }