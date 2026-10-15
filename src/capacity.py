"""Finite-window quote capacity formulas.

Implements the exact and asymptotic formulas for liquidity capacity over an
operational-time window T:

  I(T) = integral_0^T G^0(tau) dtau = [2*sqrt(D*nu)]^{-1} * erf(sqrt(nu*T))
  L(T) = ell_0 * T / I(T)
       ~ ell_0 * sqrt(pi*D*T) * [1 + nu*T/3]   for nu*T << 1

where ell_0 = |d_x rho(p)| = eps / ell_nu is the local quote slope.

The sqrt(T) short-window scaling confirms that the instantaneous capacity
coefficient vanishes as T -> 0, so liquidity capacity is inherently finite-window.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import integrate
from scipy.special import erf

from .kernels import kernel_G0


def integral_G0_exact(D: float, nu: float, T: float) -> float:
    """Exact finite-window integral of same-location kernel G^0.

    I(T) = [2 * sqrt(D * nu)]^{-1} * erf(sqrt(nu * T))

    Derived from the closed-form antiderivative of the Gaussian heat kernel.

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    T: operational-time window (> 0)

    Returns
    -------
    float
    """
    return erf(math.sqrt(nu * T)) / (2.0 * math.sqrt(D * nu))


def integral_G0_numerical(D: float, nu: float, T: float) -> float:
    """Numerically integrate G^0 over [eps, T] using adaptive quadrature.

    Used to validate the exact formula. Integrates from a small lower bound
    eps > 0 to avoid the integrable tau^{-1/2} singularity at tau=0.

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    T: operational-time window (> 0)

    Returns
    -------
    float
    """
    def integrand(tau: float) -> float:
        return float(kernel_G0(np.array([max(tau, 1e-30)]), D, nu)[0])

    result, _ = integrate.quad(integrand, 1e-12, T, limit=200)
    return result


def quote_capacity_exact(
    D: float,
    nu: float,
    eps_i: float,
    ell_nu_i: float,
    T: float,
) -> float:
    """Exact finite-window quote capacity.

    L(T) = ell_0 * T / I(T)
         = 2 * ell_0 * sqrt(D * nu) * T / erf(sqrt(nu * T))

    where ell_0 = eps_i / ell_nu_i is the local quote slope.

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    eps_i: density threshold
    ell_nu_i: cancellation length sqrt(D/nu)
    T: operational-time window (> 0)

    Returns
    -------
    float
    """
    ell_0 = eps_i / ell_nu_i
    I_T = integral_G0_exact(D, nu, T)
    if I_T < 1e-30:
        return float("nan")
    return ell_0 * T / I_T


def quote_capacity_asymptotic(
    D: float,
    nu: float,
    eps_i: float,
    ell_nu_i: float,
    T: float,
) -> dict:
    """Short-window asymptotic quote capacity (nu*T << 1).

    Leading-order:   L_lead(T) = ell_0 * sqrt(pi * D * T)
                                = eps_i * sqrt(pi * nu * T)  [using ell_0 = eps/ell_nu]

    First correction: L_asym(T) = ell_0 * sqrt(pi * D * T) * (1 + nu * T / 3)

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    eps_i: density threshold
    ell_nu_i: cancellation length sqrt(D/nu)
    T: operational-time window (> 0)

    Returns
    -------
    dict with keys 'L_lead' and 'L_corrected'
    """
    ell_0 = eps_i / ell_nu_i
    L_lead = ell_0 * math.sqrt(math.pi * D * T)
    L_corrected = L_lead * (1.0 + nu * T / 3.0)
    return {"L_lead": L_lead, "L_corrected": L_corrected}


def scaling_exponent(T_grid: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Estimate local log-log scaling exponent d log L / d log T.

    Uses centered finite differences on the log-log data. Endpoints use
    forward/backward differences.

    Parameters
    ----------
    T_grid: operational-time window values (positive, monotone increasing)
    capacity: L(T) values

    Returns
    -------
    np.ndarray: local scaling exponent at each T point
    """
    log_T = np.log(T_grid)
    log_L = np.log(np.maximum(capacity, 1e-30))

    exponents = np.gradient(log_L, log_T)
    return exponents
