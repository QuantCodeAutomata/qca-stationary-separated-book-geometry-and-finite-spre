"""External source specifications for the reaction-diffusion order book model.

Sources represent the flow of new limit orders into the book from lit (visible)
and latent (hidden) order pools, assembled as:
  s_i^ext(d) = s_i^lit(d) + s_i^lat(d)
where d is the distance from the placement edge.

The source-free interval (q_b, q_a) has no source activity.
"""

from __future__ import annotations

import numpy as np

from .params import GridParams, ModelParams


def lit_source(d: np.ndarray, Lambda_lit: float, ell_lit: float) -> np.ndarray:
    """Lit (visible) source term: exponential decay from placement edge.

    s^lit(d) = Lambda_lit * exp(-d / ell_lit)

    Parameters
    ----------
    d: distance from placement edge (non-negative)
    Lambda_lit: lit source amplitude
    ell_lit: lit source decay length scale

    Returns
    -------
    np.ndarray of same shape as d
    """
    return Lambda_lit * np.exp(-d / ell_lit)


def latent_source(d: np.ndarray, Lambda_lat: float, ell_lat: float) -> np.ndarray:
    """Latent (hidden) source term: saturating form from placement edge.

    s^lat(d) = Lambda_lat * (1 - exp(-d / ell_lat))

    Parameters
    ----------
    d: distance from placement edge (non-negative)
    Lambda_lat: latent source amplitude
    ell_lat: latent source saturation length scale

    Returns
    -------
    np.ndarray of same shape as d
    """
    return Lambda_lat * (1.0 - np.exp(-d / ell_lat))


def external_source(d: np.ndarray, mp: ModelParams) -> np.ndarray:
    """Total external source: sum of lit and latent contributions.

    s^ext(d) = s^lit(d) + s^lat(d)

    Parameters
    ----------
    d: distance from placement edge (non-negative)
    mp: model parameters

    Returns
    -------
    np.ndarray of same shape as d
    """
    return lit_source(d, mp.Lambda_lit, mp.ell_lit) + latent_source(d, mp.Lambda_lat, mp.ell_lat)


def build_sources(
    x: np.ndarray,
    gp: GridParams,
    mp: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bid and ask source arrays on the spatial grid.

    Source support rules (paper's separated-book geometry):
    - Bid source s_B is nonzero only where x <= q_b (d_B = q_b - x >= 0)
    - Ask source s_A is nonzero only where x >= q_a (d_A = x - q_a >= 0)
    - Source-free interval (q_b, q_a): both sources are exactly zero

    Parameters
    ----------
    x: spatial grid of shape (Nx,)
    gp: grid parameters with q_b and q_a
    mp: model parameters for source amplitudes

    Returns
    -------
    (s_B, s_A): each np.ndarray of shape (Nx,)
    """
    s_B = np.zeros_like(x)
    s_A = np.zeros_like(x)

    bid_mask = x <= gp.q_b
    ask_mask = x >= gp.q_a

    d_B = gp.q_b - x[bid_mask]   # distance from q_b for bid support
    d_A = x[ask_mask] - gp.q_a   # distance from q_a for ask support

    s_B[bid_mask] = external_source(d_B, mp)
    s_A[ask_mask] = external_source(d_A, mp)

    return s_B, s_A
