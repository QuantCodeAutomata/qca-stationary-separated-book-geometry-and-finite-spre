"""Experiment 4: Completion-Memory-Driven Placement Standoff and Spread Geometry.

Validates the dynamic mechanism where pending replenishment volume Q_Sigma
widens the source-withheld interval: Sigma = Sigma_0 + chi_s * Q_Sigma.

This is a mechanism demonstration (Mode B: exogenous Q paths) because the
exact completion law from the paper's supplement is not available. All
assumptions are documented explicitly.

Tests:
  - Immediate completion: Q_Sigma = 0 -> Sigma = Sigma_0
  - Symmetric delayed: Q_A = Q_B > 0 -> Sigma > Sigma_0, Q_Delta = 0
  - Asymmetric: Q_A != Q_B -> both Sigma and Q_Delta nonzero
  - State-update chronology enforcement
  - Spread widening consistent with s ~ Sigma - d_A* - d_B*
"""

import sys
import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory import (
    StandoffParams,
    CompletionMemory,
    ExogenousScenario,
    compute_standoff,
    compute_placement_edges,
    ExponentialDecayCompletion,
)
from src.model.params import ModelParams, GridParams
from src.model.grid import make_x_grid
from src.model.sources import build_sources
from src.model.pde import solve_stationary
from src.model.observables import compute_spread_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _run_spread_for_Sigma(Sigma: float, mp: ModelParams) -> float:
    """Compute spread at a given Sigma value using stationary solve."""
    gp = GridParams(
        x_min=-8.0, x_max=8.0, Nx=300,
        u_max=60.0, Nu=5000, tol=1e-9,
        q_b=-Sigma / 2, q_a=Sigma / 2,
    )
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)

    # Simple exponential initial conditions
    rho_B0 = mp.eps_B * np.exp(-np.abs(x - gp.q_b) / mp.ell_nu_B)
    rho_A0 = mp.eps_A * np.exp(-np.abs(x - gp.q_a) / mp.ell_nu_A)
    rho_B0 = np.clip(rho_B0, 1e-20, None)
    rho_A0 = np.clip(rho_A0, 1e-20, None)

    rho_A, rho_B, _ = solve_stationary(
        mp, gp, rho_A0, rho_B0, s_A, s_B, max_steps=20000, tol=1e-9, check_every=100
    )
    metrics = compute_spread_metrics(x, rho_A, rho_B, mp, gp)
    return float(metrics.get("spread", float("nan")))


def exp_4_exogenous_scenarios() -> dict[str, pd.DataFrame]:
    """Run three exogenous scenarios and record Sigma and spread dynamics."""
    u_max = 10.0
    Nu = 200
    u_grid = np.linspace(0, u_max, Nu + 1)

    sp = StandoffParams(Sigma_0=2.0, chi_s=0.5)
    scenarios = ["immediate", "symmetric", "asymmetric"]
    results = {}

    for scen in scenarios:
        gen = ExogenousScenario(u_grid, scenario=scen, Q_max=1.0, tau_rise=2.0, tau_fall=3.0)
        Q_A_path = gen.get_Q_A()
        Q_B_path = gen.get_Q_B()

        records = []
        for n, u in enumerate(u_grid):
            mem = CompletionMemory(Q_A=float(Q_A_path[n]), Q_B=float(Q_B_path[n]))
            Sigma = compute_standoff(mem, sp)
            q_b, q_a = compute_placement_edges(mem, sp)

            records.append({
                "u": u,
                "Q_A": float(Q_A_path[n]),
                "Q_B": float(Q_B_path[n]),
                "Q_Sigma": mem.Q_Sigma,
                "Q_Delta": mem.Q_Delta,
                "Sigma": Sigma,
                "q_b": q_b,
                "q_a": q_a,
            })
        results[scen] = pd.DataFrame(records)

    return results


def exp_4_plot_scenarios(results: dict[str, pd.DataFrame]) -> None:
    """Plot Sigma and Q_Sigma for all three scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    colors = {"immediate": "blue", "symmetric": "green", "asymmetric": "red"}

    for col, (scen, df) in enumerate(results.items()):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(df["u"], df["Q_Sigma"], "-", color=colors[scen], lw=2, label=r"$Q_\Sigma$")
        ax_top.plot(df["u"], df["Q_Delta"], "--", color=colors[scen], lw=1.5, alpha=0.7, label=r"$Q_\Delta$")
        ax_top.set_title(f"Scenario: {scen}", fontsize=12)
        ax_top.set_xlabel("Operational time u")
        ax_top.set_ylabel("Pending volume")
        ax_top.legend(fontsize=9)
        ax_top.grid(alpha=0.3)

        ax_bot.plot(df["u"], df["Sigma"], "-", color=colors[scen], lw=2, label=r"$\Sigma(u)$")
        ax_bot.axhline(df["Sigma"].iloc[0], color="k", ls=":", lw=1, label=r"$\Sigma_0$")
        ax_bot.fill_between(df["u"], df["q_b"], df["q_a"], alpha=0.15, color=colors[scen])
        ax_bot.set_xlabel("Operational time u")
        ax_bot.set_ylabel(r"Standoff $\Sigma$ and $[q_b, q_a]$")
        ax_bot.legend(fontsize=9)
        ax_bot.grid(alpha=0.3)

    plt.suptitle("Exp 4: Completion-Memory Standoff Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_standoff_scenarios.png"), dpi=120)
    plt.close()


def exp_4_spread_vs_Sigma() -> pd.DataFrame:
    """Static spread vs Sigma: shows spread widening as Sigma increases."""
    mp = ModelParams(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=0.5,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=0.3, ell_lit=0.5, Lambda_lat=0.1, ell_lat=1.5,
    )

    Sigma_test = [1.5, 2.0, 2.5, 3.0]
    records = []
    for Sigma in Sigma_test:
        s = _run_spread_for_Sigma(Sigma, mp)
        records.append({"Sigma": Sigma, "spread": s})
        print(f"  Sigma={Sigma:.2f} -> spread={s:.4f}")

    return pd.DataFrame(records)


def exp_4_exponential_completion(D: float = 0.5, nu_comp: float = 0.5) -> pd.DataFrame:
    """Demonstrate exponential-decay completion mechanism.

    Show that delayed completion generates Q_Sigma > 0 while immediate
    completion keeps Q_Sigma = 0.
    """
    du = 0.05
    u_max = 20.0
    u_grid = np.arange(0.0, u_max + du, du)

    # Execution sequence: burst at t=1, then periodic
    rng = np.random.default_rng(42)
    exec_A = np.zeros(len(u_grid))
    exec_B = np.zeros(len(u_grid))
    burst_idx = int(1.0 / du)
    exec_A[burst_idx:burst_idx + 5] = 0.5
    exec_B[burst_idx:burst_idx + 5] = 0.5

    # Immediate: mu_comp = large (completion faster than step)
    comp_fast = ExponentialDecayCompletion(mu_A=100.0, mu_B=100.0)
    # Delayed: mu_comp = nu_comp (finite memory)
    comp_slow = ExponentialDecayCompletion(mu_A=nu_comp, mu_B=nu_comp)

    mem_fast = CompletionMemory()
    mem_slow = CompletionMemory()
    sp = StandoffParams(Sigma_0=2.0, chi_s=0.3)

    records_fast = []
    records_slow = []

    for n in range(len(u_grid)):
        # Immediate
        mem_fast = comp_fast.update(mem_fast, exec_A[n], exec_B[n], du)
        Sigma_fast = compute_standoff(mem_fast, sp)
        records_fast.append({"u": u_grid[n], "Q_Sigma": mem_fast.Q_Sigma, "Sigma": Sigma_fast})

        # Delayed
        mem_slow = comp_slow.update(mem_slow, exec_A[n], exec_B[n], du)
        Sigma_slow = compute_standoff(mem_slow, sp)
        records_slow.append({"u": u_grid[n], "Q_Sigma": mem_slow.Q_Sigma, "Sigma": Sigma_slow})

    df_fast = pd.DataFrame(records_fast)
    df_slow = pd.DataFrame(records_slow)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(df_fast["u"], df_fast["Q_Sigma"], "b-", lw=1.5, label="Immediate (fast decay)")
    ax.plot(df_slow["u"], df_slow["Q_Sigma"], "r-", lw=1.5, label="Delayed (slow decay)")
    ax.set_xlabel("Operational time u")
    ax.set_ylabel(r"$Q_\Sigma$ (total pending)")
    ax.set_title("Pending Volume: Immediate vs Delayed Completion")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_fast["u"], df_fast["Sigma"], "b-", lw=1.5, label=r"$\Sigma$ (immediate)")
    ax2.plot(df_slow["u"], df_slow["Sigma"], "r-", lw=1.5, label=r"$\Sigma$ (delayed)")
    ax2.axhline(sp.Sigma_0, color="k", ls=":", lw=1, label=r"$\Sigma_0$")
    ax2.set_xlabel("Operational time u")
    ax2.set_ylabel(r"Standoff $\Sigma$")
    ax2.set_title("Standoff Widening from Delayed Completion")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_completion_mechanism.png"), dpi=120)
    plt.close()

    return pd.concat([
        df_slow.assign(mode="delayed"),
        df_fast.assign(mode="immediate"),
    ])


def main() -> None:
    print("=" * 60)
    print("EXP 4: Completion-Memory Placement Standoff")
    print("=" * 60)
    print("NOTE: Mechanism demonstration (Mode B) — exact completion law")
    print("      not available from main text; using exogenous Q paths.")

    # 1. Exogenous scenarios
    print("\n[1] Exogenous scenario Q-path tests...")
    results = exp_4_exogenous_scenarios()
    exp_4_plot_scenarios(results)

    for scen, df in results.items():
        max_Sigma = df["Sigma"].max()
        Sigma_0 = df["Sigma"].iloc[0]
        max_Q_Sigma = df["Q_Sigma"].max()
        print(f"  {scen}: max Q_Sigma={max_Q_Sigma:.4f}, "
              f"max Sigma={max_Sigma:.4f} (Sigma_0={Sigma_0:.4f})")
        df.to_csv(os.path.join(RESULTS_DIR, f"exp4_{scen}.csv"), index=False)

    # 2. Spread vs Sigma
    print("\n[2] Static spread vs Sigma (via stationary PDE solves)...")
    df_spread = exp_4_spread_vs_Sigma()
    df_spread.to_csv(os.path.join(RESULTS_DIR, "exp4_spread_vs_Sigma.csv"), index=False)
    spread_delta = df_spread["spread"].iloc[-1] - df_spread["spread"].iloc[0]
    Sigma_delta = df_spread["Sigma"].iloc[-1] - df_spread["Sigma"].iloc[0]
    print(f"  Spread change per Sigma change: {spread_delta/Sigma_delta:.4f} (expect ~1)")

    # 3. Exponential-decay completion mechanism
    print("\n[3] Exponential-decay completion mechanism...")
    df_comp = exp_4_exponential_completion()
    df_comp.to_csv(os.path.join(RESULTS_DIR, "exp4_completion_mechanism.csv"), index=False)
    imm = df_comp[df_comp["mode"] == "immediate"]["Q_Sigma"]
    delayed = df_comp[df_comp["mode"] == "delayed"]["Q_Sigma"]
    print(f"  Immediate max Q_Sigma: {imm.max():.6f} (expect ~0)")
    print(f"  Delayed max Q_Sigma: {delayed.max():.4f} (expect >0)")

    print("\nExp 4 done. Outputs saved to results/")


if __name__ == "__main__":
    main()
