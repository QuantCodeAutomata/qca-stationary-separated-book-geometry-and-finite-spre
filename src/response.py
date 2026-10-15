"""Linear response computations for quote-level density perturbations.

Implements convolution-based density and quote response:
  delta_rho(p, u) = -conv(G^0, J^-)(u) + conv(G^d, J^+)(u)

and the quote displacement linearization:
  delta_p(u) = -delta_rho(p, u) / (eps_i / ell_nu_i)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .kernels import kernel_G0, kernel_Gd


def convolution_trapz(tau: np.ndarray, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Causal numerical convolution using the trapezoidal rule.

    result[n] = sum_{k=0}^{n} f[n-k] * g[k] * dtau

    Uses FFT convolution for efficiency and truncates to causal length.

    Parameters
    ----------
    tau: uniformly spaced time grid of length N
    f: first function sampled on tau
    g: second function (causal impulse / kernel) sampled on tau

    Returns
    -------
    np.ndarray of shape (N,): causal convolution evaluated at each tau point
    """
    dtau = tau[1] - tau[0]
    full_conv = fftconvolve(f, g, mode="full")
    return full_conv[: len(tau)] * dtau


def density_response(
    tau_grid: np.ndarray,
    J_minus: np.ndarray,
    J_plus: np.ndarray,
    D: float,
    nu: float,
    d: float,
) -> np.ndarray:
    """Compute quote-level density response to depletion and replenishment flows.

    delta_rho(p, u) = -integral_0^u G^0(u-v) J^-(v) dv
                     + integral_0^u G^d(u-v) J^+(v) dv

    Depletion J^- depletes density at the quote via G^0 (same-location).
    Replenishment J^+ arrives from displacement d via G^d.

    Parameters
    ----------
    tau_grid: uniformly spaced operational-time grid (N points)
    J_minus: depletion current evaluated on tau_grid
    J_plus: replenishment current evaluated on tau_grid
    D: diffusivity
    nu: cancellation rate
    d: displacement from placement edge to quote

    Returns
    -------
    np.ndarray of shape (N,): delta_rho at the quote over time
    """
    G0 = kernel_G0(tau_grid, D, nu)
    Gd = kernel_Gd(tau_grid, D, nu, d)

    depletion_term = convolution_trapz(tau_grid, J_minus, G0)
    replenishment_term = convolution_trapz(tau_grid, J_plus, Gd)

    return -depletion_term + replenishment_term


def quote_response(
    delta_rho: np.ndarray,
    eps_i: float,
    ell_nu_i: float,
) -> np.ndarray:
    """Convert density response to quote displacement.

    On the exponential profile, the local quote slope is:
      |d_x rho_i(p_i)| = eps_i / ell_nu_i

    Quote displacement:
      delta_p(u) = -delta_rho(p_i, u) / (eps_i / ell_nu_i)

    Parameters
    ----------
    delta_rho: density response at the quote over time
    eps_i: density threshold
    ell_nu_i: cancellation length sqrt(D/nu)

    Returns
    -------
    np.ndarray: quote displacement over time
    """
    local_slope = eps_i / ell_nu_i
    return -delta_rho / local_slope
