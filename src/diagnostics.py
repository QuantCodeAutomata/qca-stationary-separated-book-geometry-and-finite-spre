"""Diagnostic checks for the order-book PDE model.

Verifies analytical predictions against numerical results:
  - Exponential inward profile fit
  - Threshold sensitivity identity d_s / d_ln(eps_i) = ell_nu_i
  - Weak-overlap regime diagnostics
  - Symmetry error |m - x_phi| under mirrored conditions
"""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import curve_fit


def fit_exponential_profile(
    x_inner: np.ndarray,
    rho_inner: np.ndarray,
    x_edge: float,
    side: str,
) -> dict:
    """Fit rho inside source-free interval to exponential decay.

    Bid model: rho_B(x) ~ rho_q * exp(-(x - q_b) / ell)  for x > q_b
    Ask model: rho_A(x) ~ rho_q * exp(-(q_a - x) / ell)  for x < q_a

    Parameters
    ----------
    x_inner: spatial grid points inside the source-free interval (near the edge)
    rho_inner: density values at x_inner
    x_edge: placement edge location (q_b for bid, q_a for ask)
    side: 'bid' or 'ask'

    Returns
    -------
    dict with keys: rho_q (fit amplitude), ell_fit (fit decay length),
                    slope_fit (= 1/ell_fit), r_squared (semilog linearity)
    """
    eps_safe = 1e-30
    log_rho = np.log(np.maximum(rho_inner, eps_safe))

    if side == "bid":
        d = x_inner - x_edge
    elif side == "ask":
        d = x_edge - x_inner
    else:
        raise ValueError(f"side must be 'bid' or 'ask', got '{side}'")

    mask = d > 0
    if mask.sum() < 3:
        return {"rho_q": float("nan"), "ell_fit": float("nan"), "slope_fit": float("nan"), "r_squared": float("nan")}

    d_fit = d[mask]
    log_rho_fit = log_rho[mask]

    # Linear fit: log_rho = log(rho_q) - d / ell
    coeffs = np.polyfit(d_fit, log_rho_fit, 1)
    slope_fit = coeffs[0]   # should be -1/ell (negative)
    log_rho_q = coeffs[1]

    rho_q = math.exp(log_rho_q)
    ell_fit = -1.0 / slope_fit if abs(slope_fit) > 1e-15 else float("nan")

    # R^2 for semilog linearity
    log_pred = log_rho_q + slope_fit * d_fit
    ss_res = np.sum((log_rho_fit - log_pred) ** 2)
    ss_tot = np.sum((log_rho_fit - np.mean(log_rho_fit)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

    return {
        "rho_q": rho_q,
        "ell_fit": ell_fit,
        "slope_fit": slope_fit,
        "r_squared": r_squared,
    }


def threshold_sensitivity_fd(
    eps_i: float,
    spread_func,
    delta_eps_rel: float = 0.01,
) -> float:
    """Finite-difference estimate of ds / d_ln(eps_i).

    Uses central differences:
      ds/d_ln(eps) = (s(eps+) - s(eps-)) / (ln(eps+) - ln(eps-))

    where eps+= eps_i * (1 + delta_eps_rel), eps- = eps_i * (1 - delta_eps_rel).

    Parameters
    ----------
    eps_i: base threshold value
    spread_func: callable(eps) -> float, the spread as function of threshold
    delta_eps_rel: relative perturbation in eps_i

    Returns
    -------
    float: finite-difference sensitivity ds / d_ln(eps_i)
    """
    eps_plus = eps_i * (1.0 + delta_eps_rel)
    eps_minus = eps_i * (1.0 - delta_eps_rel)

    s_plus = spread_func(eps_plus)
    s_minus = spread_func(eps_minus)

    d_ln_eps = math.log(eps_plus) - math.log(eps_minus)
    return (s_plus - s_minus) / d_ln_eps


def weak_overlap_ratios(
    ell_nu_A: float,
    ell_nu_B: float,
    spread: float,
    kappa: float,
    nu_A: float,
    nu_B: float,
    rho_A_at_pb: float,
    rho_B_at_pa: float,
) -> dict:
    """Compute weak-overlap diagnostics for the penetration approximation.

    The approximation s ~ Sigma - d_A* - d_B* is valid when:
      (1) ell_nu_i / s << 1   (penetration length small relative to spread)
      (2) kappa * rho_opposite(p_i) / nu_i << 1  (reaction small relative to cancellation)

    Parameters
    ----------
    ell_nu_A, ell_nu_B: cancellation lengths
    spread: observed spread
    kappa: reaction coefficient
    nu_A, nu_B: cancellation rates
    rho_A_at_pb: ask density evaluated at bid quote p_b
    rho_B_at_pa: bid density evaluated at ask quote p_a

    Returns
    -------
    dict with 'ratio_ell_A', 'ratio_ell_B', 'ratio_react_A', 'ratio_react_B'
    """
    if spread < 1e-15:
        return {
            "ratio_ell_A": float("inf"),
            "ratio_ell_B": float("inf"),
            "ratio_react_A": float("nan"),
            "ratio_react_B": float("nan"),
        }

    return {
        "ratio_ell_A": ell_nu_A / spread,
        "ratio_ell_B": ell_nu_B / spread,
        "ratio_react_A": kappa * rho_A_at_pb / nu_A if nu_A > 0 else float("nan"),
        "ratio_react_B": kappa * rho_B_at_pa / nu_B if nu_B > 0 else float("nan"),
    }


def symmetry_error(midpoint: float, x_phi: float) -> float:
    """Compute |m - x_phi| to test symmetric branch identity.

    Under symmetric mirrored conditions with equal thresholds:
      m = x_phi   (midpoint coincides with reaction price)

    Parameters
    ----------
    midpoint: observed midpoint m = (p_a + p_b) / 2
    x_phi: reaction price (zero of phi = rho_B - rho_A)

    Returns
    -------
    float: absolute deviation |m - x_phi|
    """
    return abs(midpoint - x_phi)
