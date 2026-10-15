"""Completion-memory mechanism for pending replenishment state variables.

Implements the standoff rule:
  Sigma = Sigma_0 + chi_s * Q_Sigma

where Q_Sigma = Q_A + Q_B is total pending replenishment volume and
Q_Delta = Q_A - Q_B is the signed imbalance.

The exact completion law is not specified in the paper's main text, so this
module implements a generic exponential-decay completion memory and an
exogenous scenario generator. This is labeled a mechanism demonstration.

State-update chronology (mandatory per paper):
  1. Update pending states Q_A, Q_B from past executions / completion law
  2. Compute Q_Sigma, Q_Delta
  3. Update Sigma = Sigma_0 + chi_s * Q_Sigma
  4. Update placement edges q_b, q_a
  5. Build sources
  6. Update densities
  7. Read quotes
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StandoffParams:
    """Parameters governing the completion-memory standoff mechanism."""

    Sigma_0: float           # Baseline source-withheld interval width
    chi_s: float             # Sensitivity of Sigma to total pending volume Q_Sigma (must be >= 0)
    x_center: float = 0.0   # Placement center (midpoint of q_b and q_a)

    def __post_init__(self) -> None:
        if self.chi_s < 0:
            raise ValueError(f"chi_s must be >= 0 (got {self.chi_s})")


@dataclass
class CompletionMemory:
    """Pending replenishment volume state for both sides.

    Attributes
    ----------
    Q_A: pending ask-side replenishment
    Q_B: pending bid-side replenishment
    """

    Q_A: float = 0.0
    Q_B: float = 0.0

    @property
    def Q_Sigma(self) -> float:
        """Total pending replenishment volume Q_Sigma = Q_A + Q_B."""
        return self.Q_A + self.Q_B

    @property
    def Q_Delta(self) -> float:
        """Signed pending imbalance Q_Delta = Q_A - Q_B."""
        return self.Q_A - self.Q_B


def compute_standoff(mem: CompletionMemory, sp: StandoffParams) -> float:
    """Compute current standoff Sigma = Sigma_0 + chi_s * Q_Sigma.

    Parameters
    ----------
    mem: current completion memory state
    sp: standoff parameters

    Returns
    -------
    float: current standoff Sigma
    """
    return sp.Sigma_0 + sp.chi_s * mem.Q_Sigma


def compute_placement_edges(mem: CompletionMemory, sp: StandoffParams) -> tuple[float, float]:
    """Compute placement edges from current standoff and center.

    Symmetric placement:
      q_b = x_center - Sigma / 2
      q_a = x_center + Sigma / 2

    Parameters
    ----------
    mem: current completion memory state
    sp: standoff parameters

    Returns
    -------
    (q_b, q_a): bid and ask placement edges
    """
    Sigma = compute_standoff(mem, sp)
    half = Sigma / 2.0
    return sp.x_center - half, sp.x_center + half


class ExponentialDecayCompletion:
    """Generic exponential-decay completion mechanism.

    Executions (depletions) on side i generate pending replenishment Q_i
    that decays at rate mu_comp. This models a finite-memory completion law.

    Mechanism mode: used when the exact paper supplement law is unavailable.

    Parameters
    ----------
    mu_A, mu_B: completion decay rates for ask and bid sides
    """

    def __init__(self, mu_A: float, mu_B: float) -> None:
        self.mu_A = mu_A
        self.mu_B = mu_B

    def update(
        self,
        mem: CompletionMemory,
        exec_A: float,
        exec_B: float,
        du: float,
    ) -> CompletionMemory:
        """Update pending states one step.

        Q_i^{n+1} = Q_i^n * exp(-mu_i * du) + exec_i

        Parameters
        ----------
        mem: current memory state
        exec_A: executed volume on ask side this step
        exec_B: executed volume on bid side this step
        du: time step

        Returns
        -------
        Updated CompletionMemory
        """
        new_Q_A = (mem.Q_A + exec_A) * np.exp(-self.mu_A * du)
        new_Q_B = (mem.Q_B + exec_B) * np.exp(-self.mu_B * du)
        return CompletionMemory(Q_A=new_Q_A, Q_B=new_Q_B)


class ExogenousScenario:
    """Provides exogenous Q_A(u), Q_B(u) time series for mechanism testing.

    Three built-in scenarios:
      'immediate': Q_A = Q_B = 0 at all times -> Sigma = Sigma_0
      'symmetric': Q_A = Q_B = Q(u), Q_Sigma = 2*Q(u), Q_Delta = 0
      'asymmetric': Q_A != Q_B, both Q_Sigma and Q_Delta are nonzero
    """

    def __init__(
        self,
        u_grid: np.ndarray,
        scenario: str = "symmetric",
        Q_max: float = 1.0,
        tau_rise: float = 1.0,
        tau_fall: float = 2.0,
    ) -> None:
        self.u_grid = u_grid
        self.scenario = scenario
        self.Q_max = Q_max
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall

    def _Q_pulse(self, shift: float = 0.0) -> np.ndarray:
        """Triangular pulse: rises then decays to zero."""
        u = self.u_grid
        u0 = u.min() + shift
        Q = np.where(
            u < u0 + self.tau_rise,
            self.Q_max * (u - u0) / self.tau_rise,
            self.Q_max * np.exp(-(u - (u0 + self.tau_rise)) / self.tau_fall),
        )
        return np.maximum(Q, 0.0)

    def get_Q_A(self) -> np.ndarray:
        if self.scenario == "immediate":
            return np.zeros_like(self.u_grid)
        elif self.scenario == "symmetric":
            return self._Q_pulse()
        elif self.scenario == "asymmetric":
            return self._Q_pulse() * 1.5
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

    def get_Q_B(self) -> np.ndarray:
        if self.scenario == "immediate":
            return np.zeros_like(self.u_grid)
        elif self.scenario == "symmetric":
            return self._Q_pulse()
        elif self.scenario == "asymmetric":
            return self._Q_pulse() * 0.5
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
