"""Green function kernels for displaced replenishment response.

Implements:
  G^0(tau): same-location density-response kernel
  G^d(tau): displaced density-response kernel
  h^d(tau): first-passage kernel (distinct shape, same integrated attenuation)

All kernels derived from the diffusion-cancellation Green's function on R^1.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import integrate
from scipy.special import erf


def kernel_G0(tau: np.ndarray, D: float, nu: float) -> np.ndarray:
    """Same-location (d=0) density-response kernel.

    G^0(tau) = exp(-nu * tau) / sqrt(4 * pi * D * tau)

    Parameters
    ----------
    tau: operational-time lags (> 0)
    D: diffusivity
    nu: cancellation rate

    Returns
    -------
    np.ndarray of same shape as tau
    """
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.maximum(tau, 1e-15)
    return np.exp(-nu * tau) / np.sqrt(4.0 * math.pi * D * safe_tau)


def kernel_Gd(tau: np.ndarray, D: float, nu: float, d: float) -> np.ndarray:
    """Displaced density-response kernel.

    G^d(tau) = exp(-nu * tau) / sqrt(4 * pi * D * tau) * exp(-d^2 / (4 * D * tau))

    For d=0, reduces to G^0. The Gaussian factor exp(-d^2/(4Dtau)) provides
    short-time regularization when d > 0.

    Parameters
    ----------
    tau: operational-time lags (> 0)
    D: diffusivity
    nu: cancellation rate
    d: displacement distance >= 0

    Returns
    -------
    np.ndarray of same shape as tau
    """
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.maximum(tau, 1e-15)
    return (
        np.exp(-nu * tau - d ** 2 / (4.0 * D * safe_tau))
        / np.sqrt(4.0 * math.pi * D * safe_tau)
    )


def kernel_hd(tau: np.ndarray, D: float, nu: float, d: float) -> np.ndarray:
    """First-passage kernel at distance d.

    h^d(tau) = (d / sqrt(4 * pi * D * tau^3)) * exp(-nu * tau - d^2 / (4 * D * tau))

    Has the same integrated attenuation exp(-d/ell_nu) as G^d but a different
    shape and a different peak location. This distinction is explicit in the paper.

    Parameters
    ----------
    tau: operational-time lags (> 0)
    D: diffusivity
    nu: cancellation rate
    d: displacement distance > 0

    Returns
    -------
    np.ndarray of same shape as tau
    """
    tau = np.asarray(tau, dtype=float)
    safe_tau = np.maximum(tau, 1e-15)
    return (
        d / np.sqrt(4.0 * math.pi * D * safe_tau ** 3)
        * np.exp(-nu * tau - d ** 2 / (4.0 * D * safe_tau))
    )


def peak_lag_Gd(D: float, nu: float, d: float) -> float:
    """Analytical peak lag of G^d.

    tau*(d) = (sqrt(1 + 4*nu*d^2/D) - 1) / (4*nu)

    For nu -> 0 asymptotic: tau* -> d^2 / (2*D).

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    d: displacement

    Returns
    -------
    float: peak lag tau*(d)
    """
    if nu < 1e-12:
        return d ** 2 / (2.0 * D)
    return (math.sqrt(1.0 + 4.0 * nu * d ** 2 / D) - 1.0) / (4.0 * nu)


def peak_lag_hd(D: float, nu: float, d: float) -> float:
    """Analytical peak lag of first-passage kernel h^d.

    tau_FP*(d) = (sqrt(9 + 4*nu*d^2/D) - 3) / (4*nu)

    For nu -> 0 asymptotic: tau_FP* -> d^2 / (6*D).

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    d: displacement

    Returns
    -------
    float: first-passage peak lag
    """
    if nu < 1e-12:
        return d ** 2 / (6.0 * D)
    return (math.sqrt(9.0 + 4.0 * nu * d ** 2 / D) - 3.0) / (4.0 * nu)


def integral_G0(D: float, nu: float, tau_max: float | None = None) -> float:
    """Integrate G^0 from 0 to tau_max (or infinity).

    Exact formulas:
      integral_0^inf G^0 = 1 / (2 * sqrt(D * nu))
      integral_0^T   G^0 = [2 * sqrt(D * nu)]^{-1} * erf(sqrt(nu * T))

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    tau_max: upper integration limit (None -> infinity)

    Returns
    -------
    float
    """
    if tau_max is None:
        return 1.0 / (2.0 * math.sqrt(D * nu))
    return erf(math.sqrt(nu * tau_max)) / (2.0 * math.sqrt(D * nu))


def integral_Gd(D: float, nu: float, d: float, tau_max: float = 1000.0) -> float:
    """Numerically integrate G^d from a small positive value to tau_max.

    Uses scipy.integrate.quad for accuracy. The Gaussian factor regularizes
    the tau -> 0 singularity when d > 0.

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    d: displacement distance
    tau_max: upper integration limit (chosen large relative to 1/nu and d^2/D)

    Returns
    -------
    float: numerical integral
    """
    def integrand(tau: float) -> float:
        safe_tau = max(tau, 1e-30)
        return (
            math.exp(-nu * tau - d ** 2 / (4.0 * D * safe_tau))
            / math.sqrt(4.0 * math.pi * D * safe_tau)
        )

    eps_lower = 1e-10
    result, _ = integrate.quad(integrand, eps_lower, tau_max, limit=200)
    return result


def attenuation_ratio(D: float, nu: float, d: float) -> float:
    """Theoretical integrated attenuation: integral_Gd / integral_G0 = exp(-d/ell_nu).

    ell_nu = sqrt(D / nu)

    Parameters
    ----------
    D: diffusivity
    nu: cancellation rate
    d: displacement distance

    Returns
    -------
    float: exp(-d * sqrt(nu / D))
    """
    ell_nu = math.sqrt(D / nu)
    return math.exp(-d / ell_nu)
