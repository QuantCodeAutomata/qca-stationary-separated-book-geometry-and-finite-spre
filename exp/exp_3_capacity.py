"""Experiment 3: Finite-Window Quote Capacity in Operational Time.

Validates the exact and asymptotic finite-window capacity formulas:
  I(T) = [2*sqrt(D*nu)]^{-1} * erf(sqrt(nu*T))
  L(T) = ell_0 * T / I(T) ~ ell_0 * sqrt(pi*D*T) * (1 + nu*T/3) for nu*T << 1

Demonstrates that capacity scales as sqrt(T) and therefore has no positive
instantaneous coefficient at T -> 0.
"""

import sys
import os
import math

import numpy as np
import pandas as pd
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capacity import (
    integral_G0_exact,
    integral_G0_numerical,
    quote_capacity_exact,
    quote_capacity_asymptotic,
    scaling_exponent,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def exp_3_integral_validation(D: float = 0.5, nu: float = 1.0) -> pd.DataFrame:
    """Validate I(T) exact vs numerical quadrature."""
    T_vals = np.logspace(-3, 2, 50)
    records = []
    for T in T_vals:
        I_exact = integral_G0_exact(D, nu, float(T))
        I_num = integral_G0_numerical(D, nu, float(T))
        rel_err = abs(I_exact - I_num) / max(abs(I_exact), 1e-30)
        records.append({
            "T": float(T),
            "nu_T": float(nu * T),
            "I_exact": I_exact,
            "I_numerical": I_num,
            "rel_error": rel_err,
        })

    df = pd.DataFrame(records)
    print(f"  Max relative error (exact vs numerical): {df['rel_error'].max():.2e}")
    return df


def exp_3_capacity_curves(D: float = 0.5, nu: float = 1.0) -> pd.DataFrame:
    """Compute and compare exact vs asymptotic capacity over a wide T range."""
    ell_nu = math.sqrt(D / nu)
    eps_i = 0.05

    T_vals = np.logspace(-3, 1.5, 100)
    records = []
    for T in T_vals:
        T = float(T)
        L_exact = quote_capacity_exact(D, nu, eps_i, ell_nu, T)
        asym = quote_capacity_asymptotic(D, nu, eps_i, ell_nu, T)
        L_lead = asym["L_lead"]
        L_corr = asym["L_corrected"]

        I_T = integral_G0_exact(D, nu, T)
        # Alternative exact: eps_i * sqrt(pi * nu * T) [leading]
        L_alt_lead = eps_i * math.sqrt(math.pi * nu * T)

        records.append({
            "T": T,
            "nu_T": nu * T,
            "L_exact": L_exact,
            "L_lead": L_lead,
            "L_corrected": L_corr,
            "L_alt_lead": L_alt_lead,
            "rel_err_lead": abs(L_lead - L_exact) / max(abs(L_exact), 1e-30),
            "rel_err_corr": abs(L_corr - L_exact) / max(abs(L_exact), 1e-30),
        })

    return pd.DataFrame(records)


def exp_3_scaling_exponent(df_cap: pd.DataFrame) -> pd.DataFrame:
    """Compute local scaling exponent d log L / d log T."""
    T = df_cap["T"].values
    L = df_cap["L_exact"].values
    exps = scaling_exponent(T, L)
    df_cap = df_cap.copy()
    df_cap["scaling_exponent"] = exps
    return df_cap


def plot_capacity(df: pd.DataFrame, D: float, nu: float) -> None:
    """Plot capacity curves and scaling exponent."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.loglog(df["T"], df["L_exact"], "b-", lw=2, label=r"$\mathcal{L}^{exact}(T)$")
    ax.loglog(df["T"], df["L_lead"], "r--", lw=1.5, label=r"Leading $\propto \sqrt{T}$")
    ax.loglog(df["T"], df["L_corrected"], "g:", lw=1.5, label=r"Corrected (+$\nu T/3$)")
    ax.set_xlabel(r"Operational window $T$", fontsize=12)
    ax.set_ylabel(r"Quote capacity $\mathcal{L}(T)$", fontsize=12)
    ax.set_title("Finite-Window Quote Capacity", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["nu_T"], df["rel_err_lead"], "r--", lw=1.5, label="Leading-order error")
    ax2.plot(df["nu_T"], df["rel_err_corr"], "g:", lw=1.5, label="Corrected error")
    ax2.set_xlabel(r"$\nu T$", fontsize=12)
    ax2.set_ylabel("Relative approximation error", fontsize=12)
    ax2.set_title("Asymptotic Approximation Error", fontsize=13)
    ax2.set_xlim([0, 3])
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.semilogx(df["T"], df["scaling_exponent"], "b-", lw=2)
    ax3.axhline(0.5, color="r", ls="--", lw=1.5, label=r"Theoretical slope 1/2")
    ax3.set_xlabel(r"Operational window $T$", fontsize=12)
    ax3.set_ylabel(r"$d\log\mathcal{L}/d\log T$", fontsize=12)
    ax3.set_title("Local Scaling Exponent", fontsize=13)
    ax3.legend(fontsize=9)
    ax3.set_ylim([0.3, 1.1])
    ax3.grid(alpha=0.3)

    plt.suptitle(f"Exp 3: Finite-Window Capacity (D={D}, $\\nu$={nu})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_capacity.png"), dpi=120)
    plt.close()


def exp_3_parameter_sweep() -> pd.DataFrame:
    """Sweep D and nu; report capacity at several T values."""
    eps_i = 0.05
    configs = [
        (0.5, 1.0),
        (1.0, 1.0),
        (0.5, 2.0),
        (0.25, 0.5),
        (1.0, 0.5),
    ]
    T_test = [0.01, 0.1, 1.0, 10.0]
    records = []
    for D, nu in configs:
        ell_nu = math.sqrt(D / nu)
        for T in T_test:
            L_exact = quote_capacity_exact(D, nu, eps_i, ell_nu, T)
            L_lead = eps_i * math.sqrt(math.pi * nu * T)
            records.append({
                "D": D, "nu": nu, "ell_nu": ell_nu, "T": T,
                "L_exact": L_exact, "L_lead": L_lead,
                "rel_err": abs(L_lead - L_exact) / max(abs(L_exact), 1e-30),
            })
    return pd.DataFrame(records)


def main() -> None:
    print("=" * 60)
    print("EXP 3: Finite-Window Quote Capacity")
    print("=" * 60)

    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    print(f"  D={D}, nu={nu}, ell_nu={ell_nu:.4f}")

    # 1. Validate integral
    print("\n[1] Validating I(T) exact vs numerical...")
    df_int = exp_3_integral_validation(D, nu)
    df_int.to_csv(os.path.join(RESULTS_DIR, "exp3_integral_validation.csv"), index=False)

    # 2. Capacity curves
    print("\n[2] Computing capacity curves...")
    df_cap = exp_3_capacity_curves(D, nu)
    df_cap = exp_3_scaling_exponent(df_cap)

    # Check short-window scaling exponent
    short_mask = df_cap["nu_T"] < 0.1
    if short_mask.sum() > 0:
        avg_exp = df_cap.loc[short_mask, "scaling_exponent"].mean()
        print(f"  Average scaling exponent at nu*T<0.1: {avg_exp:.4f} (expect ~0.5)")

    # Check approximation quality
    small_mask = df_cap["nu_T"] < 0.3
    if small_mask.sum() > 0:
        max_err_lead = df_cap.loc[small_mask, "rel_err_lead"].max()
        max_err_corr = df_cap.loc[small_mask, "rel_err_corr"].max()
        print(f"  Max relative error (leading) at nu*T<0.3: {max_err_lead:.2e}")
        print(f"  Max relative error (corrected) at nu*T<0.3: {max_err_corr:.2e}")

    df_cap.to_csv(os.path.join(RESULTS_DIR, "exp3_capacity_curves.csv"), index=False)
    plot_capacity(df_cap, D, nu)

    # 3. Parameter sweep
    print("\n[3] Parameter sweep over D, nu...")
    df_sweep = exp_3_parameter_sweep()
    df_sweep.to_csv(os.path.join(RESULTS_DIR, "exp3_param_sweep.csv"), index=False)
    print(df_sweep.to_string())

    # Verify: at T -> 0, d log L / d log T -> 0.5
    T_tiny = df_cap[df_cap["T"] < 0.01]["scaling_exponent"]
    if len(T_tiny) > 0:
        print(f"\n  Scaling exponent at T<0.01: mean={T_tiny.mean():.4f} (expect 0.5)")

    print("\nExp 3 done. Outputs saved to results/")


if __name__ == "__main__":
    main()
