"""Tests for Experiment 3: Finite-Window Quote Capacity.

Verifies:
  - Exact integral I(T) = [2*sqrt(D*nu)]^{-1} erf(sqrt(nu*T))
  - Capacity formula L(T) = ell_0 * T / I(T)
  - Short-window asymptotic scaling ~sqrt(T)
  - Capacity scaling exponent approaches 0.5 as T->0
"""

import sys
import os
import math
import numpy as np
import pytest
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capacity import (
    integral_G0_exact,
    integral_G0_numerical,
    quote_capacity_exact,
    quote_capacity_asymptotic,
    scaling_exponent,
)


# --------------------------------------------------------------------------
# Integral I(T) tests
# --------------------------------------------------------------------------

def test_integral_G0_exact_formula():
    """I(T) = [2 sqrt(D nu)]^{-1} erf(sqrt(nu T))."""
    D, nu = 0.5, 1.0
    for T in [0.01, 0.1, 1.0, 10.0]:
        I = integral_G0_exact(D, nu, T)
        I_ref = erf(math.sqrt(nu * T)) / (2.0 * math.sqrt(D * nu))
        assert abs(I - I_ref) < 1e-12, f"Exact formula mismatch at T={T}"


def test_integral_G0_zero_at_T0():
    """I(T) -> 0 as T -> 0 since erf(0) = 0."""
    D, nu = 0.5, 1.0
    I = integral_G0_exact(D, nu, 1e-10)
    assert I < 1e-5


def test_integral_G0_numerical_vs_exact():
    """Numerical quadrature of G^0 should match exact erf formula."""
    D, nu = 0.5, 1.0
    for T in [0.1, 1.0, 5.0]:
        I_exact = integral_G0_exact(D, nu, T)
        I_num = integral_G0_numerical(D, nu, T)
        rel_err = abs(I_exact - I_num) / max(I_exact, 1e-30)
        assert rel_err < 2e-3, f"Numerical vs exact error {rel_err:.4e} at T={T}"


def test_integral_G0_monotone_increasing():
    """I(T) must increase with T (integral of non-negative kernel)."""
    D, nu = 0.5, 1.0
    T_vals = np.linspace(0.01, 20.0, 50)
    I_vals = [integral_G0_exact(D, nu, T) for T in T_vals]
    assert all(I_vals[i] <= I_vals[i + 1] for i in range(len(I_vals) - 1))


def test_integral_G0_limit():
    """I(inf) = [2 sqrt(D nu)]^{-1} since erf(inf) = 1."""
    D, nu = 0.5, 1.0
    I_large = integral_G0_exact(D, nu, 1000.0)
    I_theory = 1.0 / (2.0 * math.sqrt(D * nu))
    assert abs(I_large - I_theory) < 1e-6 * I_theory


# --------------------------------------------------------------------------
# Capacity formula tests
# --------------------------------------------------------------------------

def test_capacity_positive():
    """Quote capacity must be positive for T > 0."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    for T in [0.01, 0.1, 1.0, 10.0]:
        L = quote_capacity_exact(D, nu, eps, ell_nu, T)
        assert L > 0, f"Capacity must be positive at T={T}"


def test_capacity_formula_identity():
    """L(T) = ell_0 * T / I(T)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    ell_0 = eps / ell_nu

    for T in [0.1, 1.0, 5.0]:
        L = quote_capacity_exact(D, nu, eps, ell_nu, T)
        I_T = integral_G0_exact(D, nu, T)
        L_ref = ell_0 * T / I_T
        assert abs(L - L_ref) < 1e-10 * L_ref


def test_capacity_short_window_scaling():
    """L(T) ~ ell_0 * sqrt(pi * D * T) for nu*T << 1."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    ell_0 = eps / ell_nu

    T_small = 0.001  # nu*T = 0.001 << 1
    L_exact = quote_capacity_exact(D, nu, eps, ell_nu, T_small)
    L_lead = ell_0 * math.sqrt(math.pi * D * T_small)
    rel_err = abs(L_exact - L_lead) / max(abs(L_lead), 1e-30)
    assert rel_err < 0.01, f"Short-window leading order error: {rel_err:.4f}"


def test_capacity_asymptotic_corrected():
    """Corrected asymptotic L(T) ~ ell_0*sqrt(pi*D*T)*(1 + nu*T/3) for small nu*T."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    ell_0 = eps / ell_nu

    for T in [0.01, 0.05]:
        L_exact = quote_capacity_exact(D, nu, eps, ell_nu, T)
        asym = quote_capacity_asymptotic(D, nu, eps, ell_nu, T)
        L_corr = asym["L_corrected"]
        rel_err_corr = abs(L_corr - L_exact) / max(abs(L_exact), 1e-30)
        # Corrected should be better than leading
        L_lead = asym["L_lead"]
        rel_err_lead = abs(L_lead - L_exact) / max(abs(L_exact), 1e-30)
        assert rel_err_corr <= rel_err_lead + 1e-8, \
            f"Corrected should not be worse than leading at T={T}"


def test_capacity_increases_with_T():
    """Quote capacity should increase with T (more time = more capacity)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    T_vals = np.linspace(0.01, 20.0, 50)
    L_vals = [quote_capacity_exact(D, nu, eps, ell_nu, T) for T in T_vals]
    assert all(L_vals[i] <= L_vals[i + 1] for i in range(len(L_vals) - 1))


def test_capacity_eps_dependence():
    """Capacity should increase with eps (higher threshold = more capacity)."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    T = 1.0
    eps_vals = [0.01, 0.05, 0.1, 0.2]
    L_vals = [quote_capacity_exact(D, nu, eps, ell_nu, T) for eps in eps_vals]
    for i in range(len(L_vals) - 1):
        assert L_vals[i] < L_vals[i + 1]


# --------------------------------------------------------------------------
# Scaling exponent tests
# --------------------------------------------------------------------------

def test_scaling_exponent_short_window():
    """Scaling exponent d log L / d log T -> 0.5 as T -> 0."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    T_vals = np.logspace(-3, -1, 50)
    L_vals = np.array([quote_capacity_exact(D, nu, eps, ell_nu, T) for T in T_vals])
    exps = scaling_exponent(T_vals, L_vals)
    short_window_exp = np.mean(exps[T_vals < 0.01])
    assert abs(short_window_exp - 0.5) < 0.05, \
        f"Short-window scaling exponent {short_window_exp:.4f} (expect ~0.5)"


def test_scaling_exponent_range():
    """Scaling exponents should remain in (0, 1) for T > 0."""
    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    eps = 0.05
    T_vals = np.logspace(-3, 1.5, 100)
    L_vals = np.array([quote_capacity_exact(D, nu, eps, ell_nu, T) for T in T_vals])
    exps = scaling_exponent(T_vals, L_vals)
    valid = np.isfinite(exps)
    assert np.all(exps[valid] > 0.0)
    assert np.all(exps[valid] < 1.1)
