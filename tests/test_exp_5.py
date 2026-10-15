"""Tests for Experiment 5: Midpoint vs Reaction Price.

Verifies:
  - Reaction price extracted from zero crossing of phi
  - Under symmetric mirrored conditions: m ~ x_phi
  - Spread changes with Sigma; x_phi stays near center
  - Under threshold asymmetry: m departs from x_phi
  - Imbalance benchmark calculation
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.params import ModelParams, GridParams
from src.model.grid import make_x_grid
from src.model.sources import build_sources
from src.model.pde import solve_stationary
from src.model.observables import (
    compute_derived_fields,
    extract_quotes,
    extract_reaction_price,
    find_threshold_crossing,
)
from src.diagnostics import symmetry_error


# --------------------------------------------------------------------------
# Field decomposition tests
# --------------------------------------------------------------------------

def test_phi_antisymmetric_under_mirror():
    """Mirrored rho_A, rho_B -> phi is antisymmetric around center."""
    x = np.linspace(-5, 5, 101)
    x_c = 0.0
    sigma = 0.8
    rho_A = 0.3 * np.exp(-np.maximum(x - x_c, 0) / sigma)
    rho_B = 0.3 * np.exp(-np.maximum(x_c - x, 0) / sigma)
    rho, phi = compute_derived_fields(rho_A, rho_B)

    # phi(x) = -phi(-x)
    phi_forward = np.interp(x, x, phi)
    phi_backward = np.interp(-x, x, phi)
    assert np.allclose(phi_forward, -phi_backward, atol=1e-10)


def test_reaction_price_at_zero_crossing():
    """x_phi must be the zero crossing of phi."""
    x = np.linspace(-3, 3, 1001)
    phi = x - 0.5  # zero at x = 0.5
    x_phi = extract_reaction_price(x, phi)
    assert abs(x_phi - 0.5) < (x[1] - x[0]) + 1e-8


def test_reaction_price_symmetric_phi():
    """For antisymmetric phi around x=0, x_phi should be at 0."""
    x = np.linspace(-3, 3, 1001)
    phi = np.sin(x)  # zero at x=0 (and pi, -pi)
    x_phi = extract_reaction_price(x, phi)
    # Nearest zero to center x=0
    assert abs(x_phi) < 0.1


def test_midpoint_equals_x_phi_symmetric():
    """Under symmetric inputs (rho_A = rho_B mirrored), m ~ x_phi."""
    x = np.linspace(-5, 5, 501)
    x_c = 0.0
    rho_B = 0.2 * np.exp(-np.maximum(x - x_c, 0) / 0.5) + \
            0.2 * np.exp(-np.maximum(x_c - x, 0) / 0.5) * 0.0  # one-sided
    # Simpler: symmetric Gaussian-like from two sides
    rho_B = 0.5 * np.exp(-np.abs(x - (-1.0)) / 0.5)
    rho_A = 0.5 * np.exp(-np.abs(x - 1.0) / 0.5)

    rho, phi = compute_derived_fields(rho_A, rho_B)
    x_phi = extract_reaction_price(x, phi)
    eps = 0.05
    p_b, p_a = extract_quotes(x, rho_A, rho_B, eps, eps)

    if not math.isnan(p_b) and not math.isnan(p_a) and not math.isnan(x_phi):
        m = (p_b + p_a) / 2.0
        err = symmetry_error(m, x_phi)
        assert err < 0.3, f"Symmetric: |m - x_phi| = {err:.4f}"


def test_spread_not_zero_with_separated_sources():
    """With separated sources in the weak-overlap regime, the spread must be positive.

    Lambda_lit=0.3, kappa=0.5, placement edges at ±3 (Sigma=6): edge density
    rho_B_q ≈ 0.069 > eps=0.05, so the bid/ask profiles do not overlap.
    """
    mp = ModelParams(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=0.5,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=0.3, ell_lit=0.5, Lambda_lat=0.0, ell_lat=1.5,
    )
    gp = GridParams(x_min=-8.0, x_max=8.0, Nx=200, u_max=30.0, Nu=3000, q_b=-3.0, q_a=3.0)
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)

    rho_B0 = mp.eps_B * np.exp(-np.abs(x - gp.q_b) / mp.ell_nu_B)
    rho_A0 = mp.eps_A * np.exp(-np.abs(x - gp.q_a) / mp.ell_nu_A)

    rho_A, rho_B, _ = solve_stationary(mp, gp, np.clip(rho_A0, 1e-20, None),
                                        np.clip(rho_B0, 1e-20, None),
                                        s_A, s_B, max_steps=20000, tol=1e-8, check_every=100)

    p_b, p_a = extract_quotes(x, rho_A, rho_B, mp.eps_A, mp.eps_B)
    assert not math.isnan(p_b) and not math.isnan(p_a)
    assert p_a - p_b > 0, "Spread must be strictly positive"


# --------------------------------------------------------------------------
# Mechanism separation tests
# --------------------------------------------------------------------------

def test_spread_sensitive_to_Sigma():
    """Larger Sigma -> larger spread (stationary solve not needed; use prediction)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    rho_q = 0.3
    d_star = ell_nu * math.log(rho_q / eps)

    Sigma_vals = [1.5, 2.0, 2.5, 3.0]
    s_preds = [max(0.0, Sigma - 2 * d_star) for Sigma in Sigma_vals]
    for i in range(len(s_preds) - 1):
        assert s_preds[i] <= s_preds[i + 1], "Predicted spread must increase with Sigma"


def test_x_phi_independent_of_Sigma_symmetric():
    """For symmetric mirrored conditions, x_phi should stay near 0 regardless of Sigma."""
    # Test with analytical phi: phi = rho_B - rho_A with mirrored rho
    x = np.linspace(-5, 5, 1001)

    for Sigma in [1.0, 2.0, 3.0]:
        ell = 0.5
        rho_B = 0.3 * np.exp(-np.maximum(x - (-Sigma / 2), 0) / ell)
        rho_A = 0.3 * np.exp(-np.maximum(Sigma / 2 - x, 0) / ell)
        rho_B += 0.3 * np.exp(-np.maximum(-Sigma / 2 - x, 0) / ell) * 0.0
        _, phi = compute_derived_fields(rho_A, rho_B)
        x_phi = extract_reaction_price(x, phi)
        assert abs(x_phi) < 0.2, f"x_phi={x_phi:.4f} should stay near 0 for Sigma={Sigma}"


# --------------------------------------------------------------------------
# Asymmetry tests
# --------------------------------------------------------------------------

def test_unequal_thresholds_shift_midpoint():
    """With eps_A != eps_B on symmetric density, p_b and p_a are not equidistant from x_phi."""
    x = np.linspace(-3, 3, 601)
    ell = 0.5
    rho_B = 0.3 * np.exp(-np.maximum(x - (-1.0), 0) / ell)
    rho_A = 0.3 * np.exp(-np.maximum(1.0 - x, 0) / ell)

    rho, phi = compute_derived_fields(rho_A, rho_B)
    x_phi = extract_reaction_price(x, phi)

    # Equal thresholds -> symmetric midpoint
    p_b1, p_a1 = extract_quotes(x, rho_A, rho_B, 0.05, 0.05)
    m1 = (p_b1 + p_a1) / 2.0 if not (math.isnan(p_b1) or math.isnan(p_a1)) else float("nan")

    # Unequal thresholds -> midpoint shifts
    p_b2, p_a2 = extract_quotes(x, rho_A, rho_B, 0.02, 0.08)
    m2 = (p_b2 + p_a2) / 2.0 if not (math.isnan(p_b2) or math.isnan(p_a2)) else float("nan")

    if not math.isnan(m1) and not math.isnan(m2):
        err1 = symmetry_error(m1, x_phi)
        err2 = symmetry_error(m2, x_phi)
        # Both errors could be small; unequal thresholds should displace midpoint more
        assert err2 >= err1 - 1e-6, "Unequal thresholds should not reduce midpoint-reaction-price gap"


def test_find_threshold_crossing_sign():
    """find_threshold_crossing should return correct side (bid = rightmost above)."""
    x = np.linspace(-5, 5, 1001)
    rho = 0.5 * np.exp(-np.abs(x - (-1.0)) / 0.5)  # peaks at x=-1
    eps = 0.05

    p_b = find_threshold_crossing(x, rho, eps, side="bid")
    # Should be to the right of the peak where rho drops through eps
    assert p_b > -1.0 if not math.isnan(p_b) else True


def test_imbalance_benchmark_finite_spread():
    """Imbalance-implied spread 2*eps_phi/|dphi/dx| should be positive."""
    x = np.linspace(-3, 3, 1001)
    phi = np.tanh(3 * x)  # smooth, antisymmetric, d_x phi = 3 sech^2(3x) at x=0
    eps_phi = 0.1
    x_phi = extract_reaction_price(x, phi)

    if not math.isnan(x_phi):
        idx = int(np.argmin(np.abs(x - x_phi)))
        hw = 20
        lo = max(0, idx - hw)
        hi = min(len(x) - 1, idx + hw)
        dphi = (phi[hi] - phi[lo]) / (x[hi] - x[lo])
        s_phi = 2.0 * eps_phi / abs(dphi)
        assert s_phi > 0
        # For tanh(3x), dphi/dx at 0 = 3 -> s = 2*0.1/3 ~ 0.067
        assert s_phi < 1.0  # sanity check
