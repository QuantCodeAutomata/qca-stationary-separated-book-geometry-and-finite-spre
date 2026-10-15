"""Tests for Experiment 1: Stationary Separated-Book Geometry and Finite Spread.

Verifies:
  - Grid construction
  - Source support enforcement
  - PDE solver stability and convergence
  - Penetration depth formula
  - Spread prediction formula
  - Weak-overlap diagnostics
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.params import ModelParams, GridParams
from src.model.grid import make_x_grid, make_u_grid
from src.model.sources import build_sources, lit_source, latent_source, external_source
from src.model.pde import solve_stationary, step_cn, make_cn_matrix
from src.model.observables import (
    compute_derived_fields,
    extract_quotes,
    extract_reaction_price,
    compute_spread_metrics,
    interp1d_linear,
)
from src.diagnostics import fit_exponential_profile, symmetry_error


# --------------------------------------------------------------------------
# Grid tests
# --------------------------------------------------------------------------

def test_x_grid_shape():
    gp = GridParams(Nx=200)
    x = make_x_grid(gp)
    assert x.shape == (200,)
    assert x[0] == pytest.approx(gp.x_min)
    assert x[-1] == pytest.approx(gp.x_max)


def test_u_grid_shape():
    gp = GridParams(Nu=100, u_max=10.0)
    u = make_u_grid(gp)
    assert u.shape == (101,)
    assert u[0] == pytest.approx(0.0)
    assert u[-1] == pytest.approx(10.0)


def test_grid_spacing():
    gp = GridParams(Nx=100, x_min=-5.0, x_max=5.0)
    x = make_x_grid(gp)
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0], rtol=1e-10)


# --------------------------------------------------------------------------
# Source tests
# --------------------------------------------------------------------------

def test_lit_source_positive():
    d = np.linspace(0, 5, 100)
    s = lit_source(d, Lambda_lit=1.0, ell_lit=0.5)
    assert np.all(s >= 0)


def test_latent_source_saturates():
    d = np.array([0.0, 1.0, 10.0, 100.0])
    s = latent_source(d, Lambda_lat=2.0, ell_lat=1.0)
    assert s[0] == pytest.approx(0.0, abs=1e-12)
    assert s[-1] == pytest.approx(2.0, rel=1e-4)


def test_source_free_interval():
    """Sources must be exactly zero inside the withheld interval."""
    gp = GridParams(q_b=-1.0, q_a=1.0, Nx=200)
    mp = ModelParams(Lambda_lit=1.0, ell_lit=0.5, Lambda_lat=0.3, ell_lat=1.5)
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)

    inner = (x > gp.q_b) & (x < gp.q_a)
    assert np.all(s_B[inner] == 0.0), "Bid source must be zero inside withheld interval"
    assert np.all(s_A[inner] == 0.0), "Ask source must be zero inside withheld interval"


def test_source_side_support():
    """Bid source only for x<=q_b; ask source only for x>=q_a."""
    gp = GridParams(q_b=-1.0, q_a=1.0, Nx=200)
    mp = ModelParams(Lambda_lit=1.0, ell_lit=0.5, Lambda_lat=0.3, ell_lat=1.5)
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)

    ask_zone = x > gp.q_b
    bid_zone = x < gp.q_a
    assert np.all(s_B[ask_zone & (x > gp.q_a)] == 0.0)
    assert np.all(s_A[bid_zone & (x < gp.q_b)] == 0.0)


# --------------------------------------------------------------------------
# PDE / observables tests
# --------------------------------------------------------------------------

def test_derived_fields_symmetric():
    """Under symmetric rho_A = rho_B, phi = 0 and rho = 2*rho_A."""
    rho_A = np.array([0.1, 0.2, 0.15])
    rho_B = rho_A.copy()
    rho, phi = compute_derived_fields(rho_A, rho_B)
    assert np.allclose(phi, 0.0)
    assert np.allclose(rho, 2 * rho_A)


def test_derived_fields_asymmetric():
    rho_A = np.array([0.1, 0.2])
    rho_B = np.array([0.3, 0.1])
    rho, phi = compute_derived_fields(rho_A, rho_B)
    assert np.allclose(rho, [0.4, 0.3])
    assert np.allclose(phi, [0.2, -0.1])


def test_interpolation_exact_grid_point():
    x = np.linspace(0, 10, 11)
    f = x ** 2
    val = interp1d_linear(x, f, 5.0)
    assert val == pytest.approx(25.0, rel=1e-12)


def test_interpolation_between_points():
    x = np.array([0.0, 1.0, 2.0])
    f = np.array([0.0, 1.0, 4.0])
    val = interp1d_linear(x, f, 1.5)
    assert val == pytest.approx(2.5, rel=1e-10)


def test_extract_quotes_simple():
    """Bid quote: rightmost x where rho_B >= eps; Ask: leftmost where rho_A >= eps."""
    x = np.linspace(-4, 4, 801)
    # Placement edges at ±2; profiles decay inward with length 0.7.
    # Penetration: ell*ln(0.3/0.05)=0.7*ln(6)≈1.25 → p_b≈-0.75, p_a≈0.75.
    rho_B = 0.3 * np.exp(-np.maximum(x - (-2.0), 0) / 0.7)
    rho_A = 0.3 * np.exp(-np.maximum(2.0 - x, 0) / 0.7)
    eps = 0.05

    p_b, p_a = extract_quotes(x, rho_A, rho_B, eps, eps)
    assert not math.isnan(p_b)
    assert not math.isnan(p_a)
    assert p_b < p_a, "Bid quote should be below ask quote"


def test_penetration_depth_formula():
    """d* = ell_nu * ln(rho_q / eps) -> for rho_q/eps = e, d* = ell_nu."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    rho_q = eps * math.e  # rho_q / eps = e -> d* = ell_nu * 1 = ell_nu

    d_star = ell_nu * math.log(rho_q / eps)
    assert d_star == pytest.approx(ell_nu, rel=1e-12)


def test_spread_prediction_formula():
    """s_pred = Sigma - d_A* - d_B* should be positive when Sigma > d_A* + d_B*."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    # Choose rho_q = eps * e^1 -> d* = ell_nu exactly; Sigma=3 gives s_pred=3-2*ell_nu>0.
    rho_q = eps * math.e

    d_star = ell_nu * math.log(rho_q / eps)  # = ell_nu
    assert d_star == pytest.approx(ell_nu, rel=1e-10)
    Sigma = 3.0
    s_pred = Sigma - 2 * d_star
    assert s_pred > 0, "Spread prediction should be positive in weak-overlap regime"


def test_exponential_profile_fit():
    """Fit should recover ell from synthetic exponential data."""
    ell_true = 0.5
    x_edge = -1.0
    d = np.linspace(0, 1.5, 30)
    x_inner = x_edge + d
    rho_q = 0.3
    rho_inner = rho_q * np.exp(-d / ell_true)

    result = fit_exponential_profile(x_inner, rho_inner, x_edge, side="bid")
    assert abs(result["ell_fit"] - ell_true) < 0.05 * ell_true
    assert result["r_squared"] > 0.99


def test_symmetry_error_zero_under_exact_symmetry():
    assert symmetry_error(0.0, 0.0) == pytest.approx(0.0)
    assert symmetry_error(2.5, 2.5) == pytest.approx(0.0)


def test_symmetry_error_nonzero_under_asymmetry():
    assert symmetry_error(0.1, 0.0) == pytest.approx(0.1)


def test_spread_nonneg_and_finite():
    """Spread from a convergent stationary solve should be positive and finite.

    With Lambda_lit=0.3 and placement edges at ±3 (Sigma=6), the edge density
    rho_B_q ≈ 0.069 > eps=0.05, giving d* ≈ 0.22 and s_pred ≈ 5.55 > 0.
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
    rho_A0 = np.clip(rho_A0, 1e-20, None)
    rho_B0 = np.clip(rho_B0, 1e-20, None)

    rho_A, rho_B, steps = solve_stationary(
        mp, gp, rho_A0, rho_B0, s_A, s_B, max_steps=20000, tol=1e-8, check_every=100
    )
    metrics = compute_spread_metrics(x, rho_A, rho_B, mp, gp)

    assert not math.isnan(metrics["spread"]), "Spread should not be NaN"
    assert metrics["spread"] > 0, "Spread should be positive"
    assert metrics["spread"] < gp.Sigma, "Spread should be less than Sigma"


def test_densities_nonneg_after_solve():
    """PDE solver must keep densities non-negative."""
    mp = ModelParams()
    gp = GridParams(x_min=-6.0, x_max=6.0, Nx=150, u_max=20.0, Nu=2000)
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)
    rho_A0 = np.ones(gp.Nx) * 0.1
    rho_B0 = np.ones(gp.Nx) * 0.1

    rho_A, rho_B, _ = solve_stationary(mp, gp, rho_A0, rho_B0, s_A, s_B, max_steps=500)
    assert np.all(rho_A >= 0), "Ask density must remain non-negative"
    assert np.all(rho_B >= 0), "Bid density must remain non-negative"


def test_sigma_property():
    gp = GridParams(q_b=-1.5, q_a=2.0)
    assert gp.Sigma == pytest.approx(3.5)


def test_ell_nu_property():
    mp = ModelParams(D_A=1.0, nu_A=4.0, D_B=2.0, nu_B=2.0)
    assert mp.ell_nu_A == pytest.approx(0.5)
    assert mp.ell_nu_B == pytest.approx(1.0)
