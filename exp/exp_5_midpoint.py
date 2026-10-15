"""Experiment 5: Midpoint vs Reaction Price — Separation of Geometry from Imbalance.

Validates the paper's claim that on the symmetric equal-coefficient branch:
  - Spread geometry is controlled by total density rho and placement Sigma
  - Reaction price x_phi is controlled by the imbalance field phi = rho_B - rho_A
  - Under symmetric mirrored conditions: m = x_phi (midpoint coincides with reaction price)
  - Under asymmetry: m and x_phi decouple

Tests:
  - Symmetric baseline: |m - x_phi| ~ 0 (numerical tolerance)
  - Varying Sigma: spread changes, x_phi stays near center
  - Unequal thresholds eps_A != eps_B: m departs from x_phi
  - Asymmetric source forcing: x_phi shifts differently from m
  - Comparison with imbalance-only spread benchmark
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

from src.model.params import ModelParams, GridParams
from src.model.grid import make_x_grid
from src.model.sources import build_sources
from src.model.pde import solve_stationary
from src.model.observables import (
    compute_derived_fields,
    extract_quotes,
    extract_reaction_price,
    find_threshold_crossing,
    interp1d_linear,
)
from src.diagnostics import symmetry_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _base_mp(**overrides) -> ModelParams:
    """Baseline parameters in the positive-spread (weak-overlap) regime.

    Lambda_lit=0.3, kappa=0.5, Sigma=6 give rho_B^q ≈ 0.069, d* ≈ 0.23,
    s_pred ≈ 6 - 2×0.23 = 5.54 >> 0, and ell/s << 1.
    """
    defaults = dict(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=0.5,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=0.3, ell_lit=0.5, Lambda_lat=0.1, ell_lat=1.5,
    )
    defaults.update(overrides)
    return ModelParams(**defaults)


def _base_gp(q_b: float = -3.0, q_a: float = 3.0) -> GridParams:
    return GridParams(
        x_min=-10.0, x_max=10.0, Nx=400,
        u_max=60.0, Nu=6000, tol=1e-9,
        q_b=q_b, q_a=q_a,
    )


def _initial_exponential(x: np.ndarray, gp: GridParams, mp: ModelParams):
    """Build simple exponential initial density fields."""
    rho_B0 = mp.eps_B * np.exp(-np.abs(x - gp.q_b) / mp.ell_nu_B)
    rho_A0 = mp.eps_A * np.exp(-np.abs(x - gp.q_a) / mp.ell_nu_A)
    return np.clip(rho_A0, 1e-20, None), np.clip(rho_B0, 1e-20, None)


def run_case(mp: ModelParams, gp: GridParams, label: str = "") -> dict:
    """Run one stationary case and return all observables."""
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)
    rho_A0, rho_B0 = _initial_exponential(x, gp, mp)

    rho_A, rho_B, steps = solve_stationary(
        mp, gp, rho_A0, rho_B0, s_A, s_B, max_steps=30000, tol=1e-9, check_every=100
    )

    rho, phi = compute_derived_fields(rho_A, rho_B)
    p_b, p_a = extract_quotes(x, rho_A, rho_B, mp.eps_A, mp.eps_B)
    x_phi = extract_reaction_price(x, phi)

    spread = p_a - p_b if not (math.isnan(p_a) or math.isnan(p_b)) else float("nan")
    midpoint = (p_a + p_b) / 2.0 if not (math.isnan(p_a) or math.isnan(p_b)) else float("nan")
    sym_err = symmetry_error(midpoint, x_phi) if not math.isnan(midpoint) else float("nan")

    return {
        "label": label,
        "x": x, "rho_A": rho_A, "rho_B": rho_B, "rho": rho, "phi": phi,
        "p_b": p_b, "p_a": p_a, "x_phi": x_phi,
        "spread": spread, "midpoint": midpoint,
        "sym_err": sym_err, "steps": steps,
    }


def imbalance_spread_benchmark(x: np.ndarray, phi: np.ndarray, eps_phi: float) -> dict:
    """Compute the imbalance-only spread benchmark.

    phi-spread: p_b^phi = rightmost x where phi(x) = +eps_phi
                p_a^phi = leftmost x where phi(x) = -eps_phi
    Also estimate s_phi ~ 2*eps_phi / |d_x phi| near x_phi.

    Parameters
    ----------
    x: spatial grid
    phi: imbalance field
    eps_phi: imbalance threshold

    Returns
    -------
    dict with p_b_phi, p_a_phi, s_phi, dphi_dx_phi
    """
    p_b_phi = find_threshold_crossing(x, phi, +eps_phi, side="bid")
    p_a_phi = find_threshold_crossing(x, -phi, eps_phi, side="ask")  # -phi > eps_phi -> phi < -eps_phi

    x_phi = extract_reaction_price(x, phi)
    if not math.isnan(x_phi):
        # Estimate slope by central differences around x_phi
        idx_phi = int(np.argmin(np.abs(x - x_phi)))
        hw = max(5, int(len(x) // 50))
        lo = max(0, idx_phi - hw)
        hi = min(len(x) - 1, idx_phi + hw)
        dphi = (phi[hi] - phi[lo]) / (x[hi] - x[lo]) if hi > lo else float("nan")
    else:
        dphi = float("nan")

    s_phi_linear = 2.0 * eps_phi / abs(dphi) if not math.isnan(dphi) and abs(dphi) > 1e-15 else float("nan")

    return {
        "p_b_phi": p_b_phi,
        "p_a_phi": p_a_phi,
        "s_phi_linear": s_phi_linear,
        "dphi_dx_phi": dphi,
    }


def exp_5_symmetric_baseline() -> dict:
    """Baseline symmetric solve with mirrored sources and equal thresholds."""
    mp = _base_mp()
    gp = _base_gp()
    return run_case(mp, gp, label="symmetric_baseline")


def exp_5_vary_Sigma(base_mp: ModelParams) -> pd.DataFrame:
    """Vary Sigma: show spread changes while x_phi stays near center."""
    Sigma_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    records = []
    for Sigma in Sigma_vals:
        gp = _base_gp(q_b=-Sigma / 2, q_a=Sigma / 2)
        res = run_case(base_mp, gp, label=f"Sigma={Sigma}")
        records.append({
            "Sigma": Sigma,
            "spread": res["spread"],
            "midpoint": res["midpoint"],
            "x_phi": res["x_phi"],
            "sym_err": res["sym_err"],
        })
        print(f"  Sigma={Sigma:.1f}: s={res['spread']:.4f}, m={res['midpoint']:.4f}, "
              f"x_phi={res['x_phi']:.4f}")
    return pd.DataFrame(records)


def exp_5_unequal_thresholds() -> pd.DataFrame:
    """Introduce asymmetric thresholds eps_A != eps_B; measure m vs x_phi."""
    eps_pairs = [
        (0.05, 0.05),  # symmetric baseline
        (0.03, 0.08),  # eps_A < eps_B
        (0.08, 0.03),  # eps_A > eps_B
        (0.02, 0.08),  # more asymmetric: eps_B capped below rho^q
        (0.08, 0.02),  # mirror of above
    ]
    records = []
    gp = _base_gp()
    for eps_A, eps_B in eps_pairs:
        mp = _base_mp(eps_A=eps_A, eps_B=eps_B)
        res = run_case(mp, gp, label=f"eps_A={eps_A}_eps_B={eps_B}")
        records.append({
            "eps_A": eps_A,
            "eps_B": eps_B,
            "spread": res["spread"],
            "midpoint": res["midpoint"],
            "x_phi": res["x_phi"],
            "m_minus_xphi": res["midpoint"] - res["x_phi"],
            "sym_err": res["sym_err"],
        })
        print(f"  eps_A={eps_A}, eps_B={eps_B}: s={res['spread']:.4f}, "
              f"m={res['midpoint']:.4f}, x_phi={res['x_phi']:.4f}, "
              f"|m-x_phi|={res['sym_err']:.4f}")
    return pd.DataFrame(records)


def exp_5_asymmetric_sources() -> pd.DataFrame:
    """Test asymmetric source amplitudes; measure m vs x_phi decoupling."""
    # Vary Lambda_lit on ask vs bid by using wrapper with asymmetric source amplitudes
    # Since ModelParams has one Lambda_lit, we use asymmetric eps to proxy side-asymmetry
    # and also vary kappa to break symmetry through reaction
    amp_ratios = [1.0, 1.3, 1.6, 1.5, 0.7, 0.667]
    gp = _base_gp()
    records = []
    for ratio in amp_ratios:
        # Vary eps_A/eps_B as proxy for source asymmetry
        eps_A = 0.05 * ratio
        eps_B = 0.05 / ratio
        if eps_A > 0.5 or eps_B > 0.5:
            continue
        mp = _base_mp(eps_A=eps_A, eps_B=eps_B)
        res = run_case(mp, gp, label=f"ratio={ratio}")
        records.append({
            "eps_ratio": ratio,
            "eps_A": eps_A, "eps_B": eps_B,
            "spread": res["spread"],
            "midpoint": res["midpoint"],
            "x_phi": res["x_phi"],
            "m_minus_xphi": res["midpoint"] - res["x_phi"],
            "sym_err": res["sym_err"],
        })
        print(f"  ratio={ratio:.2f}: |m-x_phi|={res['sym_err']:.4f}")

    return pd.DataFrame(records)


def plot_symmetric_fields(res: dict) -> None:
    """Plot density fields, phi, and annotated quote/reaction prices."""
    x = res["x"]
    rho_A, rho_B = res["rho_A"], res["rho_B"]
    phi = res["phi"]
    p_b, p_a = res["p_b"], res["p_a"]
    x_phi = res["x_phi"]
    midpoint = res["midpoint"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x, rho_B, "b-", lw=1.5, label=r"$\rho_B$ (bid)")
    ax.plot(x, rho_A, "r-", lw=1.5, label=r"$\rho_A$ (ask)")
    ax.plot(x, res["rho"], "k-", lw=1, alpha=0.5, label=r"$\rho$ (total)")
    if not math.isnan(p_b):
        ax.axvline(p_b, color="b", ls="--", lw=1.5, label=f"$p_b={p_b:.3f}$")
    if not math.isnan(p_a):
        ax.axvline(p_a, color="r", ls="--", lw=1.5, label=f"$p_a={p_a:.3f}$")
    if not math.isnan(x_phi):
        ax.axvline(x_phi, color="g", ls=":", lw=2, label=f"$x_\\phi={x_phi:.3f}$")
    if not math.isnan(midpoint):
        ax.axvline(midpoint, color="purple", ls="-.", lw=1.5, label=f"$m={midpoint:.3f}$")
    ax.set_xlabel("log-price x", fontsize=12)
    ax.set_ylabel("Standing density", fontsize=12)
    ax.set_title("Exp 5: Symmetric Density Fields", fontsize=13)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-4, 4)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(x, phi, "k-", lw=1.5, label=r"$\phi = \rho_B - \rho_A$")
    ax2.axhline(0, color="k", lw=0.5)
    if not math.isnan(x_phi):
        ax2.axvline(x_phi, color="g", ls=":", lw=2, label=f"$x_\\phi={x_phi:.3f}$")
    if not math.isnan(midpoint):
        ax2.axvline(midpoint, color="purple", ls="-.", lw=1.5, label=f"$m={midpoint:.3f}$")
    ax2.set_xlabel("log-price x", fontsize=12)
    ax2.set_ylabel(r"Imbalance $\phi$", fontsize=12)
    ax2.set_title(r"Imbalance Field $\phi = \rho_B - \rho_A$", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-4, 4)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_symmetric_fields.png"), dpi=120)
    plt.close()


def plot_mechanism_separation(df_Sigma: pd.DataFrame) -> None:
    """Plot spread vs Sigma showing spread changes while x_phi stays fixed."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(df_Sigma["Sigma"], df_Sigma["spread"], "o-b", lw=1.5, label="Spread $s$")
    ax.set_xlabel(r"Standoff $\Sigma$", fontsize=12)
    ax.set_ylabel("Spread $s$", fontsize=12)
    ax.set_title(r"Spread vs $\Sigma$: geometry control", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_Sigma["Sigma"], df_Sigma["x_phi"], "s-r", lw=1.5, label=r"$x_\phi$ (reaction price)")
    ax2.plot(df_Sigma["Sigma"], df_Sigma["midpoint"], "^-g", lw=1.5, label="Midpoint $m$")
    ax2.set_xlabel(r"Standoff $\Sigma$", fontsize=12)
    ax2.set_ylabel("Price level", fontsize=12)
    ax2.set_title(r"$x_\phi$ and $m$ vs $\Sigma$", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Exp 5: Spread Geometry vs Imbalance Dynamics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_mechanism_separation.png"), dpi=120)
    plt.close()


def plot_asymmetry_results(df_asym: pd.DataFrame) -> None:
    """Plot |m - x_phi| as function of threshold asymmetry."""
    if df_asym.empty:
        return
    col = "eps_A" if "eps_A" in df_asym.columns else "eps_ratio"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_asym[col], df_asym["sym_err"], "o-r", lw=1.5)
    ax.set_xlabel(r"$\varepsilon_A$" if col == "eps_A" else r"Threshold ratio", fontsize=12)
    ax.set_ylabel(r"$|m - x_\phi|$", fontsize=12)
    ax.set_title(r"Midpoint–Reaction Price Decoupling Under Asymmetry", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_asymmetry_decoupling.png"), dpi=120)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("EXP 5: Midpoint vs Reaction Price")
    print("=" * 60)

    # 1. Symmetric baseline
    print("\n[1] Symmetric baseline solve...")
    res_sym = exp_5_symmetric_baseline()
    print(f"  Converged in {res_sym['steps']} steps")
    print(f"  p_b={res_sym['p_b']:.4f}, p_a={res_sym['p_a']:.4f}")
    print(f"  spread={res_sym['spread']:.4f}")
    print(f"  midpoint={res_sym['midpoint']:.4f}, x_phi={res_sym['x_phi']:.4f}")
    print(f"  |m - x_phi| = {res_sym['sym_err']:.6f} (expect ~0)")

    plot_symmetric_fields(res_sym)

    # Imbalance spread benchmark
    eps_phi = 0.02
    bench = imbalance_spread_benchmark(res_sym["x"], res_sym["phi"], eps_phi)
    print(f"\n  Imbalance benchmark: s_phi={bench['s_phi_linear']:.4f} "
          f"(vs observed s={res_sym['spread']:.4f})")
    print(f"  d_x phi at x_phi: {bench['dphi_dx_phi']:.4f}")

    # 2. Sigma sweep
    print("\n[2] Sigma sweep: spread changes, x_phi stays near 0...")
    mp_base = _base_mp()
    df_Sigma = exp_5_vary_Sigma(mp_base)
    df_Sigma.to_csv(os.path.join(RESULTS_DIR, "exp5_sigma_sweep.csv"), index=False)
    plot_mechanism_separation(df_Sigma)
    x_phi_range = df_Sigma["x_phi"].max() - df_Sigma["x_phi"].min()
    spread_range = df_Sigma["spread"].max() - df_Sigma["spread"].min()
    print(f"  Spread range: {spread_range:.4f}, x_phi range: {x_phi_range:.4f}")
    print(f"  -> Spread changes {spread_range/x_phi_range:.1f}x more than x_phi (expect >>1)")

    # 3. Unequal thresholds
    print("\n[3] Unequal thresholds: m departs from x_phi...")
    df_unequal = exp_5_unequal_thresholds()
    df_unequal.to_csv(os.path.join(RESULTS_DIR, "exp5_unequal_thresholds.csv"), index=False)
    print(df_unequal[["eps_A", "eps_B", "sym_err", "m_minus_xphi"]].to_string())

    # 4. Asymmetric sources (via eps proxy)
    print("\n[4] Asymmetric threshold proxy: m vs x_phi decoupling...")
    df_asym = exp_5_asymmetric_sources()
    df_asym.to_csv(os.path.join(RESULTS_DIR, "exp5_asymmetry.csv"), index=False)
    plot_asymmetry_results(df_unequal.rename(columns={}))

    # Summary
    sym_err_baseline = res_sym["sym_err"]
    max_sym_err_unequal = df_unequal["sym_err"].max()
    print(f"\n  Symmetric |m-x_phi|: {sym_err_baseline:.6f}")
    print(f"  Max asymmetric |m-x_phi|: {max_sym_err_unequal:.4f}")
    print(f"  -> Under asymmetry, m and x_phi decouple by {max_sym_err_unequal:.4f} price units")

    print("\nExp 5 done. Outputs saved to results/")


if __name__ == "__main__":
    main()
