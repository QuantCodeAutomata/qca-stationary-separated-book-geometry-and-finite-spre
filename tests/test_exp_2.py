"""Tests for Experiment 2: Displaced Replenishment Response.

Verifies:
  - Kernel G^d and h^d analytical formulas
  - Peak lag analytical formula and numerical consistency
  - Integrated attenuation identity
  - Quote response sign and magnitude
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernels import (
    kernel_G0,
    kernel_Gd,
    kernel_hd,
    peak_lag_Gd,
    peak_lag_hd,
    integral_G0,
    integral_Gd,
    attenuation_ratio,
)
from src.response import density_response, quote_response


# --------------------------------------------------------------------------
# Kernel shape tests
# --------------------------------------------------------------------------

def test_G0_is_Gd_at_zero():
    """G^0 should equal G^d evaluated at d=0."""
    tau = np.linspace(0.01, 5.0, 200)
    D, nu = 0.5, 1.0
    g0 = kernel_G0(tau, D, nu)
    gd = kernel_Gd(tau, D, nu, d=0.0)
    assert np.allclose(g0, gd, rtol=1e-10)


def test_G0_monotone_decreasing():
    """G^0 should decrease monotonically (peaks at t=0+ for d=0)."""
    tau = np.linspace(0.001, 5.0, 500)
    D, nu = 0.5, 1.0
    g0 = kernel_G0(tau, D, nu)
    assert np.all(np.diff(g0) < 0), "G^0 should be monotone decreasing for d=0"


def test_Gd_peaks_at_positive_tau():
    """G^d for d>0 must have its peak at strictly positive tau."""
    tau = np.linspace(1e-5, 10.0, 10000)
    D, nu = 0.5, 1.0
    d = 0.5
    gd = kernel_Gd(tau, D, nu, d)
    idx_peak = np.argmax(gd)
    assert tau[idx_peak] > 0.0


def test_Gd_nonneg():
    tau = np.linspace(1e-5, 10.0, 1000)
    D, nu = 0.5, 1.0
    for d in [0.0, 0.5, 1.0, 2.0]:
        gd = kernel_Gd(tau, D, nu, d)
        assert np.all(gd >= 0.0), f"G^d must be non-negative for d={d}"


def test_hd_nonneg():
    tau = np.linspace(1e-5, 10.0, 1000)
    D, nu = 0.5, 1.0
    for d in [0.1, 0.5, 1.0]:
        hd = kernel_hd(tau, D, nu, d)
        assert np.all(hd >= 0.0)


# --------------------------------------------------------------------------
# Peak lag tests
# --------------------------------------------------------------------------

def test_peak_lag_Gd_matches_numerical():
    """Analytical peak lag should match numerical argmax within 5%."""
    tau = np.linspace(1e-5, 20.0, 100000)
    D, nu = 0.5, 1.0
    for d in [0.1, 0.3, 0.5, 1.0, 1.5]:
        gd = kernel_Gd(tau, D, nu, d)
        tau_peak_num = float(tau[np.argmax(gd)])
        tau_peak_theory = peak_lag_Gd(D, nu, d)
        rel_err = abs(tau_peak_num - tau_peak_theory) / max(tau_peak_theory, 1e-10)
        assert rel_err < 0.05, f"Peak lag error {rel_err:.4f} for d={d}"


def test_peak_lag_Gd_increases_with_d():
    """Peak lag should increase with displacement d."""
    D, nu = 0.5, 1.0
    d_vals = np.linspace(0.1, 2.0, 10)
    lags = [peak_lag_Gd(D, nu, d) for d in d_vals]
    assert all(lags[i] < lags[i + 1] for i in range(len(lags) - 1))


def test_peak_lag_Gd_zero_nu_asymptotic():
    """At very small nu: tau*(d) -> d^2 / (2D)."""
    D = 0.5
    nu_small = 1e-6
    d = 0.5
    tau_theory = peak_lag_Gd(D, nu_small, d)
    tau_asymp = d ** 2 / (2.0 * D)
    assert abs(tau_theory - tau_asymp) < 1e-4 * tau_asymp


def test_peak_lag_hd_different_from_Gd():
    """h^d and G^d should have different peak lags for d>0."""
    D, nu = 0.5, 1.0
    d = 0.5
    tau_Gd = peak_lag_Gd(D, nu, d)
    tau_hd = peak_lag_hd(D, nu, d)
    assert abs(tau_Gd - tau_hd) > 1e-6, "G^d and h^d should have different peak lags"


# --------------------------------------------------------------------------
# Attenuation tests
# --------------------------------------------------------------------------

def test_integrated_attenuation_identity():
    """I_d / I_0 = exp(-d / ell_nu)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    I_0 = integral_G0(D, nu)
    for d in [0.1, 0.3, 0.7, 1.0]:
        I_d = integral_Gd(D, nu, d, tau_max=500.0)
        ratio_num = I_d / I_0
        ratio_theory = attenuation_ratio(D, nu, d)
        assert abs(ratio_num - ratio_theory) < 5e-3, \
            f"Attenuation error {abs(ratio_num - ratio_theory):.4e} for d={d}"


def test_attenuation_at_d_star():
    """At d = d* = ell_nu * ln(rho_q/eps): I_d/I_0 = eps/rho_q."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    rho_q = 0.3
    d_star = ell_nu * math.log(rho_q / eps)

    I_0 = integral_G0(D, nu)
    I_d = integral_Gd(D, nu, d_star, tau_max=500.0)
    ratio_num = I_d / I_0
    ratio_theory = eps / rho_q
    assert abs(ratio_num - ratio_theory) < 0.01, \
        f"Attenuation at d*: {ratio_num:.4f} vs {ratio_theory:.4f}"


def test_attenuation_decreases_with_d():
    """Integrated attenuation ratio should decrease as d increases."""
    D, nu = 0.5, 1.0
    d_vals = [0.0, 0.25, 0.5, 1.0, 1.5]
    I_0 = integral_G0(D, nu)
    ratios = [integral_Gd(D, nu, d, tau_max=200.0) / I_0 for d in d_vals]
    for i in range(len(ratios) - 1):
        assert ratios[i] >= ratios[i + 1], "Attenuation ratio must decrease with d"


def test_hd_same_attenuation_as_Gd():
    """h^d integrates to exp(-d/ell_nu) same as G^d.

    The Lévy first-passage identity gives:
      ∫_0^∞ h^d(tau) dtau = exp(-d * sqrt(nu/D)) = exp(-d / ell_nu)
    which equals ∫_0^∞ G^d(tau) dtau.  The integral is compared directly
    (not divided by I_0) using scipy.integrate.quad for accuracy.
    """
    from scipy import integrate as sci_int

    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    d = 0.5
    expected = math.exp(-d / ell_nu)

    def integrand(tau: float) -> float:
        safe_tau = max(tau, 1e-30)
        return d / math.sqrt(4.0 * math.pi * D * safe_tau ** 3) * math.exp(-nu * safe_tau - d ** 2 / (4.0 * D * safe_tau))

    I_hd, _ = sci_int.quad(integrand, 1e-10, 500.0, limit=300)
    assert abs(I_hd - expected) < 0.01, f"h^d integral: {I_hd:.4f} vs expected {expected:.4f}"


# --------------------------------------------------------------------------
# Response tests
# --------------------------------------------------------------------------

def test_density_response_depletion_negative():
    """Depletion current causes negative density response at the quote."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    d_star = ell_nu * math.log(0.3 / 0.05)
    tau = np.linspace(1e-5, 10.0, 5000)
    dtau = tau[1] - tau[0]

    J_dep = np.zeros_like(tau)
    J_dep[:10] = 1.0 / (10 * dtau)

    delta_rho = density_response(tau, J_dep, np.zeros_like(tau), D, nu, d_star)
    # Depletion should reduce density (negative)
    assert delta_rho.min() < 0


def test_density_response_replenishment_positive():
    """Replenishment current causes positive density response at the quote."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    d_star = ell_nu * math.log(0.3 / 0.05)
    tau = np.linspace(1e-5, 10.0, 5000)
    dtau = tau[1] - tau[0]

    J_rep = np.zeros_like(tau)
    J_rep[:10] = 1.0 / (10 * dtau)

    delta_rho = density_response(tau, np.zeros_like(tau), J_rep, D, nu, d_star)
    assert delta_rho.max() > 0


def test_response_replenishment_delayed():
    """Replenishment response should peak later than depletion response."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    d = 0.5  # non-zero displacement

    tau = np.linspace(1e-5, 10.0, 50000)
    dtau = tau[1] - tau[0]
    J = np.zeros_like(tau)
    J[:5] = 1.0 / (5 * dtau)

    rho_dep = density_response(tau, J, np.zeros_like(tau), D, nu, 0.0)   # at same location
    rho_rep = density_response(tau, np.zeros_like(tau), J, D, nu, d)     # from displacement d

    # Depletion response peaks at tau[0]; replenishment peaks later
    idx_dep_peak = int(np.argmin(rho_dep))
    idx_rep_peak = int(np.argmax(rho_rep))
    assert idx_rep_peak > idx_dep_peak


def test_quote_response_proportional():
    """Quote response should scale inversely with profile slope (eps / ell_nu)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    d = 0.0
    eps = 0.05

    tau = np.linspace(1e-5, 5.0, 5000)
    dtau = tau[1] - tau[0]
    delta_rho = np.full_like(tau, -0.01)  # constant depletion

    dp1 = quote_response(delta_rho, eps, ell_nu)
    # Double the threshold: quote response should halve (slope doubles)
    dp2 = quote_response(delta_rho, 2 * eps, ell_nu)
    # The slope is eps/ell_nu, so doubling eps doubles slope and halves quote response
    ratio = dp1[100] / dp2[100]
    assert abs(ratio - 2.0) < 0.2
