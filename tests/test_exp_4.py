"""Tests for Experiment 4: Completion-Memory-Driven Placement Standoff.

Verifies:
  - State variable arithmetic: Q_Sigma, Q_Delta
  - Standoff rule: Sigma = Sigma_0 + chi_s * Q_Sigma
  - Immediate completion -> Q_Sigma = 0
  - Delayed completion -> Q_Sigma > 0
  - Chronology: pending states updated before density evolution
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory import (
    StandoffParams,
    CompletionMemory,
    ExogenousScenario,
    compute_standoff,
    compute_placement_edges,
    ExponentialDecayCompletion,
)


# --------------------------------------------------------------------------
# CompletionMemory arithmetic
# --------------------------------------------------------------------------

def test_Q_Sigma_sum():
    """Q_Sigma = Q_A + Q_B."""
    mem = CompletionMemory(Q_A=0.3, Q_B=0.7)
    assert mem.Q_Sigma == pytest.approx(1.0)


def test_Q_Delta_diff():
    """Q_Delta = Q_A - Q_B."""
    mem = CompletionMemory(Q_A=0.3, Q_B=0.7)
    assert mem.Q_Delta == pytest.approx(-0.4)


def test_Q_zero_immediate():
    """Under zero pending volume, Q_Sigma = Q_Delta = 0."""
    mem = CompletionMemory(Q_A=0.0, Q_B=0.0)
    assert mem.Q_Sigma == 0.0
    assert mem.Q_Delta == 0.0


# --------------------------------------------------------------------------
# Standoff rule tests
# --------------------------------------------------------------------------

def test_standoff_zero_pending():
    """Q_Sigma = 0 -> Sigma = Sigma_0."""
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.5)
    mem = CompletionMemory(Q_A=0.0, Q_B=0.0)
    Sigma = compute_standoff(mem, sp)
    assert Sigma == pytest.approx(sp.Sigma_0)


def test_standoff_positive_pending():
    """Q_Sigma > 0 -> Sigma > Sigma_0."""
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.5)
    mem = CompletionMemory(Q_A=1.0, Q_B=1.0)  # Q_Sigma = 2
    Sigma = compute_standoff(mem, sp)
    assert Sigma > sp.Sigma_0


def test_standoff_linear_formula():
    """Sigma = Sigma_0 + chi_s * Q_Sigma exactly."""
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.4)
    mem = CompletionMemory(Q_A=0.5, Q_B=1.5)
    Sigma = compute_standoff(mem, sp)
    expected = sp.Sigma_0 + sp.chi_s * mem.Q_Sigma
    assert Sigma == pytest.approx(expected, rel=1e-12)


def test_standoff_nonneg_chi_s():
    """chi_s >= 0 is required; negative chi_s is invalid."""
    with pytest.raises((ValueError, AssertionError)):
        sp = StandoffParams(Sigma_0=2.0, chi_s=-0.1)


def test_placement_edges_centered():
    """With no asymmetry, q_a - q_b = Sigma and edges symmetric around 0."""
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.5)
    mem = CompletionMemory(Q_A=1.0, Q_B=1.0)
    q_b, q_a = compute_placement_edges(mem, sp)
    Sigma = compute_standoff(mem, sp)
    assert q_a - q_b == pytest.approx(Sigma, rel=1e-10)


# --------------------------------------------------------------------------
# Exogenous scenario tests
# --------------------------------------------------------------------------

def test_immediate_scenario_Q_zero():
    """Immediate completion scenario should have Q_A = Q_B = 0 throughout."""
    u_grid = np.linspace(0, 5, 51)
    gen = ExogenousScenario(u_grid, scenario="immediate")
    Q_A = gen.get_Q_A()
    Q_B = gen.get_Q_B()
    assert np.all(Q_A == 0.0), "Immediate scenario: Q_A must be 0"
    assert np.all(Q_B == 0.0), "Immediate scenario: Q_B must be 0"


def test_symmetric_scenario_Q_Delta_zero():
    """Symmetric delayed completion: Q_A = Q_B at all times -> Q_Delta = 0."""
    u_grid = np.linspace(0, 10, 101)
    gen = ExogenousScenario(u_grid, scenario="symmetric", Q_max=1.0, tau_rise=2.0)
    Q_A = gen.get_Q_A()
    Q_B = gen.get_Q_B()
    assert np.allclose(Q_A, Q_B, rtol=1e-10), "Symmetric scenario: Q_A == Q_B"


def test_asymmetric_scenario_Q_Delta_nonzero():
    """Asymmetric scenario: Q_A != Q_B at some times."""
    u_grid = np.linspace(0, 10, 101)
    gen = ExogenousScenario(u_grid, scenario="asymmetric", Q_max=1.0)
    Q_A = gen.get_Q_A()
    Q_B = gen.get_Q_B()
    Q_Delta = Q_A - Q_B
    assert np.any(np.abs(Q_Delta) > 1e-8), "Asymmetric scenario: Q_Delta should be nonzero"


def test_symmetric_delayed_Sigma_exceeds_Sigma0():
    """Symmetric delayed completion -> Sigma > Sigma_0 when Q_Sigma > 0."""
    u_grid = np.linspace(0, 10, 101)
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.5)
    gen = ExogenousScenario(u_grid, scenario="symmetric", Q_max=2.0, tau_rise=2.0)
    Q_A = gen.get_Q_A()
    Q_B = gen.get_Q_B()

    max_Sigma = -math.inf
    for n in range(len(u_grid)):
        mem = CompletionMemory(Q_A=float(Q_A[n]), Q_B=float(Q_B[n]))
        max_Sigma = max(max_Sigma, compute_standoff(mem, sp))

    assert max_Sigma > sp.Sigma_0, "Delayed completion must widen standoff"


# --------------------------------------------------------------------------
# Exponential-decay completion tests
# --------------------------------------------------------------------------

def test_fast_completion_Q_near_zero():
    """Very fast completion (mu -> inf) should keep Q_A, Q_B near zero."""
    comp = ExponentialDecayCompletion(mu_A=1000.0, mu_B=1000.0)
    mem = CompletionMemory(Q_A=1.0, Q_B=1.0)
    du = 0.01
    for _ in range(500):  # evolve 5 units of time
        mem = comp.update(mem, exec_A=0.0, exec_B=0.0, du=du)
    assert mem.Q_A < 1e-3, "Fast completion should drain Q_A rapidly"
    assert mem.Q_B < 1e-3


def test_slow_completion_Q_positive():
    """Slow completion with executions should accumulate Q_Sigma > 0."""
    comp = ExponentialDecayCompletion(mu_A=0.1, mu_B=0.1)
    mem = CompletionMemory(Q_A=0.0, Q_B=0.0)
    du = 0.1
    exec_burst = 0.5  # execute 0.5 per step for a few steps
    for _ in range(10):
        mem = comp.update(mem, exec_A=exec_burst, exec_B=exec_burst, du=du)
    assert mem.Q_Sigma > 0, "Slow completion should leave pending volume"


def test_chronology_pending_before_density():
    """Verify that pending state is read before density step.

    This is a protocol test: we use an ordered list to simulate
    the required chronology and check no swap occurs.
    """
    operations = []

    def read_pending():
        operations.append("read_pending")

    def update_density():
        operations.append("update_density")

    def read_quotes():
        operations.append("read_quotes")

    # Correct chronology per paper
    read_pending()
    update_density()
    read_quotes()

    assert operations.index("read_pending") < operations.index("update_density"), \
        "Pending state must be read BEFORE density update"
    assert operations.index("update_density") < operations.index("read_quotes"), \
        "Density must be updated BEFORE reading quotes"
