"""Grid construction for the reaction-diffusion order book model."""

from __future__ import annotations

import numpy as np

from .params import GridParams


def make_x_grid(gp: GridParams) -> np.ndarray:
    """Return the linearly spaced log-price grid.

    Parameters
    ----------
    gp:
        Grid parameters specifying x_min, x_max, and Nx.

    Returns
    -------
    np.ndarray of shape (Nx,)
    """
    return np.linspace(gp.x_min, gp.x_max, gp.Nx)


def make_u_grid(gp: GridParams) -> np.ndarray:
    """Return the linearly spaced operational-time grid.

    Parameters
    ----------
    gp:
        Grid parameters specifying u_max and Nu.

    Returns
    -------
    np.ndarray of shape (Nu + 1,) spanning [0, u_max]
    """
    return np.linspace(0.0, gp.u_max, gp.Nu + 1)
