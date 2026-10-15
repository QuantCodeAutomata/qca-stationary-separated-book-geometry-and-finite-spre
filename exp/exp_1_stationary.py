"""Experiment 1: Stationary Separated-Book Geometry and Finite Spread.

Validates the core stationary claim: a finite observed bid-ask spread emerges
from the two-field reaction-diffusion model when bid and ask standing densities
are separated and replenishment is withheld from the interval (q_b, q_a).

Tests:
  - Spread formula: s ~ Sigma - d_A* - d_B*
  - Inward exponential profile with decay length ell_nu
  - Parameter sweeps over Sigma, eps_i, D_i, nu_i, kappa
  - Threshold sensitivity: ds/d_ln(eps_i) ~ ell_nu_i
  - Weak-overlap regime diagnostics
"""

import sys
import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.params import ModelParams, GridParams
from src.model.grid import make_x_grid, make_u_grid
from src.model.sources import build_sources
from src.model.pde import solve_stationary
from src.model.observables import (
    compute_derived_fields,
    extract_quotes,
    extract_reaction_price,
    compute_spread_metrics,
    interp1d_linear,
)
from src.diagnostics import (
    fit_exponential_profile,
    threshold_sensitivity_fd,
    weak_overlap_ratios,
    symmetry_error,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _initial_densities(x: np.ndarray, gp: GridParams, mp: ModelParams) -> tuple[np.ndarray, np.ndarray]:
    """Build mirrored initial density fields as exponential profiles from edges."""
    x_c = (gp.q_b + gp.q_a) / 2.0
    rho_B0 = np.where(
        x <= gp.q_b,
        mp.eps_B * np.exp(-(gp.q_b - x) / mp.ell_nu_B),
        mp.eps_B * np.exp(-(x - gp.q_b) / mp.ell_nu_B),
    )
    rho_A0 = np.where(
        x >= gp.q_a,
        mp.eps_A * np.exp(-(x - gp.q_a) / mp.ell_nu_A),
        mp.eps_A * np.exp(-(gp.q_a - x) / mp.ell_nu_A),
    )
    rho_B0 = np.clip(rho_B0, 1e-20, None)
    rho_A0 = np.clip(rho_A0, 1e-20, None)
    return rho_A0, rho_B0


def run_stationary_solve(mp: ModelParams, gp: GridParams) -> dict:
    """Run one stationary solve and return all observables."""
    x = make_x_grid(gp)
    s_B, s_A = build_sources(x, gp, mp)
    rho_A0, rho_B0 = _initial_densities(x, gp, mp)

    rho_A, rho_B, steps = solve_stationary(
        mp, gp, rho_A0, rho_B0, s_A, s_B, max_steps=30000, tol=1e-9, check_every=100
    )

    metrics = compute_spread_metrics(x, rho_A, rho_B, mp, gp)
    metrics["steps"] = steps
    metrics["x"] = x
    metrics["rho_A"] = rho_A
    metrics["rho_B"] = rho_B
    # Store placement edges for downstream profile fitting
    metrics["q_b"] = gp.q_b
    metrics["q_a"] = gp.q_a

    # Derived fields
    rho, phi = compute_derived_fields(rho_A, rho_B)
    metrics["rho"] = rho
    metrics["phi"] = phi

    # Weak-overlap diagnostics
    p_b = metrics["p_b"]
    p_a = metrics["p_a"]
    if not (math.isnan(p_b) or math.isnan(p_a)) and p_a > p_b:
        rho_A_at_pb = interp1d_linear(x, rho_A, p_b)
        rho_B_at_pa = interp1d_linear(x, rho_B, p_a)
        metrics["weak_overlap"] = weak_overlap_ratios(
            mp.ell_nu_A, mp.ell_nu_B,
            metrics["spread"],
            mp.kappa, mp.nu_A, mp.nu_B,
            rho_A_at_pb, rho_B_at_pa,
        )

    return metrics


def exp_1_baseline() -> dict:
    """Baseline symmetric solve with Sigma=4 (placement edges ±2).

    With Lambda_lit=1.0 and kappa=2, rho_B^q ≈ 0.23, d* ≈ 1.08, and
    s_pred = 4.0 - 2×1.08 = 1.84 > 0  (positive-spread regime).
    """
    mp = ModelParams(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=2.0,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=1.0, ell_lit=0.5, Lambda_lat=0.3, ell_lat=1.5,
    )
    gp = GridParams(
        x_min=-8.0, x_max=8.0, Nx=400,
        u_max=60.0, Nu=6000, tol=1e-9,
        q_b=-2.0, q_a=2.0,
    )
    return run_stationary_solve(mp, gp)


def exp_1_plot_profiles(result: dict, label: str = "baseline") -> None:
    """Plot density profiles and semilog profile inside the withheld interval."""
    x = result["x"]
    rho_A = result["rho_A"]
    rho_B = result["rho_B"]
    p_b = result["p_b"]
    p_a = result["p_a"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x, rho_B, "b-", lw=1.5, label=r"$\rho_B$ (bid)")
    ax.plot(x, rho_A, "r-", lw=1.5, label=r"$\rho_A$ (ask)")
    if not math.isnan(p_b):
        ax.axvline(p_b, color="b", ls="--", lw=1, label=f"$p_b={p_b:.3f}$")
    if not math.isnan(p_a):
        ax.axvline(p_a, color="r", ls="--", lw=1, label=f"$p_a={p_a:.3f}$")
    x_phi = result.get("x_phi", float("nan"))
    if not math.isnan(x_phi):
        ax.axvline(x_phi, color="k", ls=":", lw=1, label=f"$x_\\phi={x_phi:.3f}$")
    ax.set_xlabel("log-price x")
    ax.set_ylabel("standing density")
    ax.set_title(f"Exp 1 Density Profiles ({label})")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Semilog inside withheld interval — show inward exponential penetration
    ax2 = axes[1]
    q_b_val = result.get("q_b", -1.0)
    q_a_val = result.get("q_a", 1.0)
    xlim_lo = q_b_val - 0.5
    xlim_hi = q_a_val + 0.5

    plot_mask = (x >= xlim_lo) & (x <= xlim_hi)
    rho_B_plot = np.where(rho_B[plot_mask] > 0, rho_B[plot_mask], np.nan)
    rho_A_plot = np.where(rho_A[plot_mask] > 0, rho_A[plot_mask], np.nan)
    ax2.semilogy(x[plot_mask], rho_B_plot, "b-", lw=1.5, label=r"$\rho_B$ (bid)")
    ax2.semilogy(x[plot_mask], rho_A_plot, "r-", lw=1.5, label=r"$\rho_A$ (ask)")
    ax2.axvline(q_b_val, color="b", ls=":", lw=1, label=f"$q_b={q_b_val:.2f}$")
    ax2.axvline(q_a_val, color="r", ls=":", lw=1, label=f"$q_a={q_a_val:.2f}$")
    if not math.isnan(p_b):
        ax2.axvline(p_b, color="b", ls="--", lw=1, label=f"$p_b={p_b:.3f}$")
    if not math.isnan(p_a):
        ax2.axvline(p_a, color="r", ls="--", lw=1, label=f"$p_a={p_a:.3f}$")
    ax2.set_xlabel("log-price x")
    ax2.set_ylabel("log standing density")
    ax2.set_title("Semilog Profile (withheld interval penetration)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(xlim_lo, xlim_hi)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"exp1_profiles_{label}.png"), dpi=120)
    plt.close()


def exp_1_sweep_Sigma() -> pd.DataFrame:
    """Sweep over source-free interval width Sigma and measure spread.

    Values below ~2.2 will show overlapping quotes (negative spread).
    The formula s≈Σ-d_A*-d_B* is only meaningful for positive-spread rows.
    """
    Sigma_vals = np.linspace(1.0, 6.0, 16)
    records = []
    base_mp = ModelParams(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=2.0,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=1.0, ell_lit=0.5, Lambda_lat=0.3, ell_lat=1.5,
    )

    for Sigma in Sigma_vals:
        gp = GridParams(
            x_min=-8.0, x_max=8.0, Nx=400,
            u_max=60.0, Nu=6000, tol=1e-9,
            q_b=-Sigma / 2, q_a=Sigma / 2,
        )
        res = run_stationary_solve(base_mp, gp)
        records.append({
            "Sigma": Sigma,
            "spread": res["spread"],
            "spread_pred": res["spread_pred"],
            "spread_error": res["spread_error"],
            "d_A_star": res["d_A_star"],
            "d_B_star": res["d_B_star"],
            "steps": res["steps"],
        })
        print(f"  Sigma={Sigma:.2f}: s={res['spread']:.4f}, s_pred={res['spread_pred']:.4f}, err={res['spread_error']:.4f}")

    return pd.DataFrame(records)


def exp_1_sweep_eps() -> pd.DataFrame:
    """Sweep over threshold eps and measure spread; verify ds/d_ln(eps) ~ ell_nu."""
    eps_vals = np.array([0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25])
    base_mp = ModelParams(
        D_A=0.5, D_B=0.5, nu_A=1.0, nu_B=1.0, kappa=2.0,
        eps_A=0.05, eps_B=0.05,
        Lambda_lit=1.0, ell_lit=0.5, Lambda_lat=0.3, ell_lat=1.5,
    )
    gp = GridParams(x_min=-8.0, x_max=8.0, Nx=400, u_max=60.0, Nu=6000,
                    q_b=-1.0, q_a=1.0)
    ell_nu_theory = base_mp.ell_nu_A  # symmetric

    records = []
    spreads_for_sens = {}
    for eps in eps_vals:
        mp_i = ModelParams(
            D_A=base_mp.D_A, D_B=base_mp.D_B,
            nu_A=base_mp.nu_A, nu_B=base_mp.nu_B,
            kappa=base_mp.kappa,
            eps_A=float(eps), eps_B=float(eps),
            Lambda_lit=base_mp.Lambda_lit, ell_lit=base_mp.ell_lit,
            Lambda_lat=base_mp.Lambda_lat, ell_lat=base_mp.ell_lat,
        )
        res = run_stationary_solve(mp_i, gp)
        records.append({
            "eps": eps,
            "spread": res["spread"],
            "spread_pred": res["spread_pred"],
            "d_A_star": res["d_A_star"],
            "d_B_star": res["d_B_star"],
        })
        spreads_for_sens[float(eps)] = res["spread"]

    # Compute sensitivity via central differences in log-eps space
    eps_arr = np.array([r["eps"] for r in records])
    s_arr = np.array([r["spread"] for r in records if not math.isnan(r["spread"])])

    if len(s_arr) >= 3:
        sens = np.gradient(s_arr, np.log(eps_arr[:len(s_arr)]))
        for i, rec in enumerate(records[:len(s_arr)]):
            rec["ds_dlneps"] = float(sens[i])
            rec["ell_nu_theory"] = ell_nu_theory

    df = pd.DataFrame(records)
    print(f"\n  Theoretical ell_nu = {ell_nu_theory:.4f}")
    print(df[["eps", "spread", "ds_dlneps", "ell_nu_theory"]].to_string() if "ds_dlneps" in df.columns else df.to_string())
    return df


def exp_1_exponential_profile(result: dict) -> dict:
    """Fit exponential profiles inside the source-free (withheld) interval.

    Bid profile: rho_B(x) ~ rho_B^q * exp(-(x - q_b) / ell_nu_B) for x in [q_b, p_b].
    Ask profile: rho_A(x) ~ rho_A^q * exp(-(q_a - x) / ell_nu_A) for x in [p_a, q_a].
    """
    x = result["x"]
    rho_A = result["rho_A"]
    rho_B = result["rho_B"]
    p_b = result["p_b"]
    p_a = result["p_a"]
    q_b = result.get("q_b", -1.0)
    q_a = result.get("q_a", 1.0)

    if math.isnan(p_b) or math.isnan(p_a) or p_a <= p_b:
        return {}

    # Bid penetration: from placement edge q_b inward to just past quote p_b
    inner_mask_B = (x >= q_b - 0.05) & (x <= p_b + 0.1)
    fit_B = fit_exponential_profile(
        x[inner_mask_B], rho_B[inner_mask_B], x_edge=q_b, side="bid"
    )

    # Ask penetration: from quote p_a inward to placement edge q_a
    inner_mask_A = (x >= p_a - 0.1) & (x <= q_a + 0.05)
    fit_A = fit_exponential_profile(
        x[inner_mask_A], rho_A[inner_mask_A], x_edge=q_a, side="ask"
    )

    return {"bid_fit": fit_B, "ask_fit": fit_A}


def plot_sigma_sweep(df_sigma: pd.DataFrame) -> None:
    """Plot spread vs Sigma: measured vs predicted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_sigma["Sigma"], df_sigma["spread"], "o-b", lw=1.5, label="Measured spread $s$")
    ax.plot(df_sigma["Sigma"], df_sigma["spread_pred"], "s--r", lw=1.5, label=r"Predicted $\Sigma - d_A^* - d_B^*$")
    # Theory: if d_i* constant, slope should be 1
    s0 = df_sigma["spread_pred"].iloc[0]
    Sigma0 = df_sigma["Sigma"].iloc[0]
    ax.plot(df_sigma["Sigma"], df_sigma["Sigma"] - (Sigma0 - s0), ":k", alpha=0.6, label="Slope-1 reference")
    ax.set_xlabel(r"Source-free interval $\Sigma = q_a - q_b$", fontsize=12)
    ax.set_ylabel("Spread $s$", fontsize=12)
    ax.set_title("Exp 1: Spread vs $\\Sigma$ (parameter sweep)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_sigma_sweep.png"), dpi=120)
    plt.close()


def plot_eps_sweep(df_eps: pd.DataFrame) -> None:
    """Plot spread vs eps and sensitivity."""
    if "ds_dlneps" not in df_eps.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogx(df_eps["eps"], df_eps["spread"], "o-b", lw=1.5, label="Measured spread")
    ax.semilogx(df_eps["eps"], df_eps["spread_pred"], "s--r", lw=1.5, label="Predicted spread")
    ax.set_xlabel(r"Threshold $\varepsilon$", fontsize=12)
    ax.set_ylabel("Spread $s$", fontsize=12)
    ax.set_title(r"Spread vs $\varepsilon$", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    valid = df_eps["ds_dlneps"].notna()
    ax2.plot(df_eps.loc[valid, "eps"], df_eps.loc[valid, "ds_dlneps"], "o-g", lw=1.5,
             label=r"$ds/d\ln\varepsilon$ (FD)")
    ax2.axhline(df_eps["ell_nu_theory"].iloc[0], color="r", ls="--",
                label=r"Theory $\ell_\nu$")
    ax2.set_xlabel(r"Threshold $\varepsilon$", fontsize=12)
    ax2.set_ylabel(r"$ds / d\ln\varepsilon$", fontsize=12)
    ax2.set_title(r"Threshold Sensitivity Identity", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_eps_sweep.png"), dpi=120)
    plt.close()


def main() -> None:
    print("=" * 60)
    print("EXP 1: Stationary Separated-Book Geometry")
    print("=" * 60)

    # 1. Baseline
    print("\n[1] Baseline symmetric solve...")
    res = exp_1_baseline()
    print(f"  Converged in {res['steps']} steps")
    print(f"  p_b={res['p_b']:.4f}, p_a={res['p_a']:.4f}")
    print(f"  spread={res['spread']:.4f}, spread_pred={res['spread_pred']:.4f}")
    print(f"  spread_error={res['spread_error']:.4f}")
    print(f"  x_phi={res['x_phi']:.4f}, midpoint={res['midpoint']:.4f}")
    print(f"  |m - x_phi| = {symmetry_error(res['midpoint'], res['x_phi']):.4f}")
    print(f"  d_B*={res['d_B_star']:.4f}, d_A*={res['d_A_star']:.4f}")
    if "weak_overlap" in res:
        wo = res["weak_overlap"]
        print(f"  Weak-overlap ratios: ell_A/s={wo['ratio_ell_A']:.3f}, "
              f"ell_B/s={wo['ratio_ell_B']:.3f}")

    exp_1_plot_profiles(res, label="baseline")

    # Exponential profile fit
    fit_result = exp_1_exponential_profile(res)
    if fit_result:
        bid_fit = fit_result["bid_fit"]
        ask_fit = fit_result["ask_fit"]
        mp_base = ModelParams()
        print(f"\n  Bid profile fit: ell_fit={bid_fit['ell_fit']:.4f}, "
              f"theory ell_nu_B={mp_base.ell_nu_B:.4f}, R^2={bid_fit['r_squared']:.4f}")
        print(f"  Ask profile fit: ell_fit={ask_fit['ell_fit']:.4f}, "
              f"theory ell_nu_A={mp_base.ell_nu_A:.4f}, R^2={ask_fit['r_squared']:.4f}")

    # 2. Sigma sweep
    print("\n[2] Sigma sweep...")
    df_sigma = exp_1_sweep_Sigma()
    df_sigma.to_csv(os.path.join(RESULTS_DIR, "exp1_sigma_sweep.csv"), index=False)
    plot_sigma_sweep(df_sigma)
    pos_mask = df_sigma["spread"] > 0
    if pos_mask.sum() > 0:
        rel_err = (df_sigma.loc[pos_mask, "spread_error"].abs().mean()
                   / df_sigma.loc[pos_mask, "spread"].mean())
        print(f"  Mean |spread error| / mean spread (positive-spread rows) = {rel_err:.4f}")
    else:
        print("  No positive-spread rows in Sigma sweep; widen range or reduce sources.")

    # 3. Epsilon sweep and sensitivity
    print("\n[3] Epsilon threshold sweep...")
    df_eps = exp_1_sweep_eps()
    df_eps.to_csv(os.path.join(RESULTS_DIR, "exp1_eps_sweep.csv"), index=False)
    plot_eps_sweep(df_eps)

    # Save baseline metrics as JSON
    metrics_save = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    if "weak_overlap" in metrics_save:
        metrics_save["weak_overlap"] = dict(metrics_save["weak_overlap"])
    with open(os.path.join(RESULTS_DIR, "exp1_baseline_metrics.json"), "w") as f:
        json.dump(metrics_save, f, indent=2, default=str)

    print("\nExp 1 done. Outputs saved to results/")


if __name__ == "__main__":
    main()
