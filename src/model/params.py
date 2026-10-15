"""Parameter definitions for the two-field reaction-diffusion order book model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ModelParams:
    """Physical model parameters for the reaction-diffusion order book.

    Notation follows the paper's two-field (bid/ask) decomposition.
    """

    D_A: float = 0.5    # Ask-side diffusivity
    D_B: float = 0.5    # Bid-side diffusivity
    nu_A: float = 1.0   # Ask-side cancellation rate (Greek nu)
    nu_B: float = 1.0   # Bid-side cancellation rate
    kappa: float = 2.0  # Reaction coefficient kappa
    eps_A: float = 0.05  # Ask-side density threshold epsilon_A
    eps_B: float = 0.05  # Bid-side density threshold epsilon_B
    Lambda_lit: float = 1.0   # Lit source amplitude
    ell_lit: float = 0.5      # Lit source length scale
    Lambda_lat: float = 0.5   # Latent source amplitude
    ell_lat: float = 1.0      # Latent source length scale

    @property
    def ell_nu_A(self) -> float:
        """Cancellation (diffusion) length for ask side: sqrt(D_A / nu_A)."""
        return math.sqrt(self.D_A / self.nu_A)

    @property
    def ell_nu_B(self) -> float:
        """Cancellation (diffusion) length for bid side: sqrt(D_B / nu_B)."""
        return math.sqrt(self.D_B / self.nu_B)


@dataclass
class GridParams:
    """Numerical grid and geometry parameters.

    Stores log-price grid bounds, time horizon, placement edges, and tolerances.
    """

    x_min: float = -10.0   # Left bound of log-price domain
    x_max: float = 10.0    # Right bound of log-price domain
    Nx: int = 400           # Number of spatial grid points
    u_max: float = 50.0    # Maximum operational time
    Nu: int = 5000          # Number of time steps
    tol: float = 1e-8       # Convergence tolerance for stationarity solve
    q_b: float = -1.0       # Bid-side placement edge
    q_a: float = 1.0        # Ask-side placement edge

    @property
    def Sigma(self) -> float:
        """Source-free interval width: Sigma = q_a - q_b."""
        return self.q_a - self.q_b

    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return (self.x_max - self.x_min) / (self.Nx - 1)

    @property
    def du(self) -> float:
        """Time step size."""
        return self.u_max / self.Nu
