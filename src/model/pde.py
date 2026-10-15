"""Finite-difference PDE solver for the two-field reaction-diffusion order book.

Solves the coupled system:
  d_u rho_A = D_A * d_xx rho_A - nu_A * rho_A - kappa * rho_A * rho_B + s_A
  d_u rho_B = D_B * d_xx rho_B - nu_B * rho_B - kappa * rho_A * rho_B + s_B

Uses Crank-Nicolson for the linear (diffusion + cancellation) terms and explicit
treatment for the nonlinear reaction term -kappa * rho_A * rho_B.

Boundary conditions: Neumann (zero-flux) at x_min and x_max.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .params import GridParams, ModelParams


def make_cn_matrix(
    N: int,
    D: float,
    nu: float,
    dx: float,
    du: float,
) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Build Crank-Nicolson LHS and RHS matrices for diffusion-cancellation.

    The spatial operator is: L = D/dx^2 * tridiag(1, -2, 1) - nu * I
    (second-order finite difference with Neumann BC applied separately)

    CN scheme:
      (I - du/2 * L) rho^{n+1} = (I + du/2 * L) rho^n + du * source

    Parameters
    ----------
    N: number of spatial grid points
    D: diffusivity coefficient
    nu: cancellation rate
    dx: spatial grid spacing
    du: time step

    Returns
    -------
    (A_lhs, A_rhs): scipy CSC sparse matrices of shape (N, N)
    """
    r = D * du / (2.0 * dx ** 2)
    c = nu * du / 2.0

    main_diag = np.full(N, -2.0 * r - c)
    off_diag = np.full(N - 1, r)

    L = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N), format='csc')

    I = sp.eye(N, format='csc')
    A_lhs = I - L  # I - du/2 * L (with r and c already scaled by du/2)
    A_rhs = I + L  # I + du/2 * L

    return A_lhs, A_rhs


def _apply_neumann_bc(
    A_lhs: sp.csc_matrix,
    A_rhs: sp.csc_matrix,
) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Apply Neumann (zero-flux) boundary conditions in-place by modifying rows 0 and -1.

    Enforces rho[0] = rho[1] and rho[-1] = rho[-2] via one-sided BC rows.
    """
    N = A_lhs.shape[0]

    # Convert to LIL for efficient row editing
    L = A_lhs.tolil()
    R = A_rhs.tolil()

    # BC at x_min: rho[0] - rho[1] = 0  (zero flux)
    L[0, :] = 0.0
    R[0, :] = 0.0
    L[0, 0] = 1.0
    L[0, 1] = -1.0

    # BC at x_max: rho[-1] - rho[-2] = 0
    L[N - 1, :] = 0.0
    R[N - 1, :] = 0.0
    L[N - 1, N - 1] = 1.0
    L[N - 1, N - 2] = -1.0

    return L.tocsc(), R.tocsc()


def step_cn(
    rho_A: np.ndarray,
    rho_B: np.ndarray,
    s_A: np.ndarray,
    s_B: np.ndarray,
    A_lhs_A: sp.csc_matrix,
    A_rhs_A: sp.csc_matrix,
    A_lhs_B: sp.csc_matrix,
    A_rhs_B: sp.csc_matrix,
    mp: ModelParams,
    gp: GridParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Crank-Nicolson time step.

    Reaction term -kappa * rho_A * rho_B is treated explicitly on the old step.

    Parameters
    ----------
    rho_A, rho_B: current density fields of shape (Nx,)
    s_A, s_B: source terms of shape (Nx,)
    A_lhs_A, A_rhs_A: CN matrices for ask side
    A_lhs_B, A_rhs_B: CN matrices for bid side
    mp: model parameters
    gp: grid parameters

    Returns
    -------
    (rho_A_new, rho_B_new): updated density fields, clipped to >= 0
    """
    du = gp.du
    reaction = mp.kappa * rho_A * rho_B  # explicit nonlinear reaction

    rhs_A = A_rhs_A @ rho_A + du * (s_A - reaction)
    rhs_B = A_rhs_B @ rho_B + du * (s_B - reaction)

    # Zero-flux BC: RHS boundary entries must be 0 (ghost cells)
    rhs_A[0] = 0.0
    rhs_A[-1] = 0.0
    rhs_B[0] = 0.0
    rhs_B[-1] = 0.0

    rho_A_new = spla.spsolve(A_lhs_A, rhs_A)
    rho_B_new = spla.spsolve(A_lhs_B, rhs_B)

    return np.clip(rho_A_new, 0.0, None), np.clip(rho_B_new, 0.0, None)


def solve_stationary(
    mp: ModelParams,
    gp: GridParams,
    rho_A0: np.ndarray,
    rho_B0: np.ndarray,
    s_A: np.ndarray,
    s_B: np.ndarray,
    max_steps: int = 20000,
    tol: float = 1e-8,
    check_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Evolve PDE to stationarity or until max_steps.

    Convergence criterion: max_i ||rho_i^{n+1} - rho_i^n||_inf < tol

    Parameters
    ----------
    mp: model parameters
    gp: grid parameters
    rho_A0, rho_B0: initial density fields
    s_A, s_B: (time-independent) source terms
    max_steps: maximum number of time steps
    tol: convergence tolerance (sup-norm)
    check_every: check convergence every this many steps

    Returns
    -------
    (rho_A, rho_B, steps_taken)
    """
    A_lhs_A, A_rhs_A = make_cn_matrix(gp.Nx, mp.D_A, mp.nu_A, gp.dx, gp.du)
    A_lhs_B, A_rhs_B = make_cn_matrix(gp.Nx, mp.D_B, mp.nu_B, gp.dx, gp.du)

    A_lhs_A, A_rhs_A = _apply_neumann_bc(A_lhs_A, A_rhs_A)
    A_lhs_B, A_rhs_B = _apply_neumann_bc(A_lhs_B, A_rhs_B)

    rho_A = rho_A0.copy()
    rho_B = rho_B0.copy()

    for step in range(1, max_steps + 1):
        rho_A_new, rho_B_new = step_cn(
            rho_A, rho_B, s_A, s_B,
            A_lhs_A, A_rhs_A, A_lhs_B, A_rhs_B,
            mp, gp,
        )

        if step % check_every == 0:
            err = max(
                np.max(np.abs(rho_A_new - rho_A)),
                np.max(np.abs(rho_B_new - rho_B)),
            )
            if err < tol:
                return rho_A_new, rho_B_new, step

        rho_A = rho_A_new
        rho_B = rho_B_new

    return rho_A, rho_B, max_steps
