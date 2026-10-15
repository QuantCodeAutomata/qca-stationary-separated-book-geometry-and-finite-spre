"""Observables extracted from the reaction-diffusion order book fields.

Provides quote extraction, spread computation, reaction price, and penetration
depth metrics following the paper's methodology.
"""

from __future__ import annotations

import math
import numpy as np

from .params import GridParams, ModelParams


def interp1d_linear(x: np.ndarray, f: np.ndarray, xi: float) -> float:
    """Linear interpolation of f at point xi within the grid x.

    Parameters
    ----------
    x: monotone grid points
    f: function values at x
    xi: query point (must be within x range)

    Returns
    -------
    Interpolated value (float). Returns NaN if xi is outside x range.
    """
    if xi < x[0] or xi > x[-1]:
        return float("nan")
    idx = int(np.searchsorted(x, xi, side="right")) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    t = (xi - x[idx]) / (x[idx + 1] - x[idx])
    return float(f[idx] + t * (f[idx + 1] - f[idx]))


def compute_derived_fields(
    rho_A: np.ndarray, rho_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute total density rho and imbalance phi from side densities.

    On the equal-coefficient branch:
      rho = rho_A + rho_B  (total standing density)
      phi = rho_B - rho_A  (signed imbalance field)

    Parameters
    ----------
    rho_A: ask-side density field
    rho_B: bid-side density field

    Returns
    -------
    (rho, phi)
    """
    rho = rho_A + rho_B
    phi = rho_B - rho_A
    return rho, phi


def find_threshold_crossing(
    x: np.ndarray,
    f: np.ndarray,
    threshold: float,
    side: str,
) -> float:
    """Find where f crosses threshold using linear interpolation.

    Parameters
    ----------
    x: spatial grid (increasing)
    f: function values
    threshold: crossing level
    side: 'bid' -> highest x where f(x) >= threshold (rightmost crossing)
          'ask' -> lowest x where f(x) >= threshold (leftmost crossing)

    Returns
    -------
    float: interpolated crossing location, or NaN if not found
    """
    g = f - threshold
    sign_changes = np.where(np.diff(np.sign(g)))[0]

    if len(sign_changes) == 0:
        return float("nan")

    if side == "bid":
        idx = sign_changes[-1]  # rightmost crossing
    elif side == "ask":
        idx = sign_changes[0]   # leftmost crossing
    else:
        raise ValueError(f"side must be 'bid' or 'ask', got '{side}'")

    # Linear interpolation between grid points idx and idx+1
    g0, g1 = g[idx], g[idx + 1]
    if abs(g1 - g0) < 1e-30:
        return float(x[idx])
    t = -g0 / (g1 - g0)
    return float(x[idx] + t * (x[idx + 1] - x[idx]))


def extract_quotes(
    x: np.ndarray,
    rho_A: np.ndarray,
    rho_B: np.ndarray,
    eps_A: float,
    eps_B: float,
) -> tuple[float, float]:
    """Extract observed bid and ask quotes from density thresholds.

    Bid quote p_b: highest x s.t. rho_B(x) = eps_B
    Ask quote p_a: lowest  x s.t. rho_A(x) = eps_A

    Parameters
    ----------
    x: spatial grid
    rho_A: ask-side density
    rho_B: bid-side density
    eps_A: ask threshold
    eps_B: bid threshold

    Returns
    -------
    (p_b, p_a)
    """
    p_b = find_threshold_crossing(x, rho_B, eps_B, side="bid")
    p_a = find_threshold_crossing(x, rho_A, eps_A, side="ask")
    return p_b, p_a


def extract_reaction_price(x: np.ndarray, phi: np.ndarray) -> float:
    """Extract the reaction price x_phi as the zero crossing of phi = rho_B - rho_A.

    Parameters
    ----------
    x: spatial grid
    phi: imbalance field rho_B - rho_A

    Returns
    -------
    float: x_phi where phi changes sign, NaN if not found
    """
    sign_changes = np.where(np.diff(np.sign(phi)))[0]
    if len(sign_changes) == 0:
        return float("nan")

    # Take the crossing nearest to the center
    mid_idx = len(x) // 2
    idx = sign_changes[np.argmin(np.abs(sign_changes - mid_idx))]
    phi0, phi1 = phi[idx], phi[idx + 1]
    if abs(phi1 - phi0) < 1e-30:
        return float(x[idx])
    t = -phi0 / (phi1 - phi0)
    return float(x[idx] + t * (x[idx + 1] - x[idx]))


def compute_spread_metrics(
    x: np.ndarray,
    rho_A: np.ndarray,
    rho_B: np.ndarray,
    mp: ModelParams,
    gp: GridParams,
) -> dict:
    """Compute comprehensive spread and penetration metrics.

    Implements the paper's weak-overlap penetration formula:
      s_pred = Sigma - d_A* - d_B*
    where d_i* = ell_nu_i * ln(rho_i^q / eps_i).

    Parameters
    ----------
    x: spatial grid
    rho_A: ask-side density field
    rho_B: bid-side density field
    mp: model parameters
    gp: grid parameters (contains q_b, q_a, eps_A, eps_B)

    Returns
    -------
    dict with keys: p_b, p_a, spread, midpoint, x_phi,
                    rho_B_q, rho_A_q, ell_nu_B, ell_nu_A,
                    d_B_star, d_A_star, spread_pred, spread_error
    """
    rho, phi = compute_derived_fields(rho_A, rho_B)

    p_b, p_a = extract_quotes(x, rho_A, rho_B, mp.eps_A, mp.eps_B)
    x_phi = extract_reaction_price(x, phi)

    spread = p_a - p_b if not (math.isnan(p_a) or math.isnan(p_b)) else float("nan")
    midpoint = (p_a + p_b) / 2.0 if not (math.isnan(p_a) or math.isnan(p_b)) else float("nan")

    # Edge densities at placement boundaries
    rho_B_q = interp1d_linear(x, rho_B, gp.q_b)
    rho_A_q = interp1d_linear(x, rho_A, gp.q_a)

    ell_nu_B = mp.ell_nu_B
    ell_nu_A = mp.ell_nu_A

    # Penetration depths
    if not math.isnan(rho_B_q) and rho_B_q > mp.eps_B:
        d_B_star = ell_nu_B * math.log(rho_B_q / mp.eps_B)
    else:
        d_B_star = float("nan")

    if not math.isnan(rho_A_q) and rho_A_q > mp.eps_A:
        d_A_star = ell_nu_A * math.log(rho_A_q / mp.eps_A)
    else:
        d_A_star = float("nan")

    if not (math.isnan(d_A_star) or math.isnan(d_B_star)):
        spread_pred = gp.Sigma - d_A_star - d_B_star
    else:
        spread_pred = float("nan")

    if not (math.isnan(spread) or math.isnan(spread_pred)):
        spread_error = abs(spread - spread_pred)
    else:
        spread_error = float("nan")

    return {
        "p_b": p_b,
        "p_a": p_a,
        "spread": spread,
        "midpoint": midpoint,
        "x_phi": x_phi,
        "rho_B_q": rho_B_q,
        "rho_A_q": rho_A_q,
        "ell_nu_B": ell_nu_B,
        "ell_nu_A": ell_nu_A,
        "d_B_star": d_B_star,
        "d_A_star": d_A_star,
        "spread_pred": spread_pred,
        "spread_error": spread_error,
    }
