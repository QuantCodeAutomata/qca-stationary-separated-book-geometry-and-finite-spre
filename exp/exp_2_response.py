"""Experiment 2: Delayed and Attenuated Displaced Replenishment Response.

Validates the local linear-response theory for quote-level density response
when replenishment is delivered from a displaced placement edge.

Tests:
  - Kernel G^d shape and peak lag: tau*(d) = (sqrt(1 + 4*nu*d^2/D) - 1) / (4*nu)
  - Integrated attenuation: I_d / I_0 = exp(-d / ell_nu)
  - For d = d*: I_d / I_0 = eps_i / rho_i^q
  - Quote displacement via linear response convolution
  - Distinction between G^d (density-response) and h^d (first-passage)
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

from src.kernels import (
    kernel_G0,
    kernel_Gd,
    kernel_hd,
    peak_lag_Gd,
    peak_lag_hd,
    integral_G0,
    integral_Gd,
    attenuation_ratio,
)
from src.response import density_response, quote_response

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def exp_2_kernel_shapes(D: float = 0.5, nu: float = 1.0) -> None:
    """Plot G^0, G^d, and h^d for several displacement values d."""
    d_vals = [0.0, 0.2, 0.5, 1.0]
    tau = np.linspace(1e-4, 5.0, 5000)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title(r"Density-response kernel $G^d(\tau)$", fontsize=13)
    for d in d_vals:
        k = kernel_Gd(tau, D, nu, d)
        lbl = f"d={d:.1f}" if d > 0 else r"$G^0$ (d=0)"
        ax.plot(tau, k, lw=1.5, label=lbl)
    ax.set_xlabel(r"Operational lag $\tau$", fontsize=12)
    ax.set_ylabel("Kernel value", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, 4])

    ax2 = axes[1]
    ax2.set_title(r"First-passage kernel $h^d(\tau)$ vs $G^d(\tau)$ at $d=0.5$", fontsize=13)
    d_fp = 0.5
    ax2.plot(tau, kernel_Gd(tau, D, nu, d_fp), "b-", lw=1.5, label=r"$G^d$ (density-response)")
    ax2.plot(tau, kernel_hd(tau, D, nu, d_fp), "r--", lw=1.5, label=r"$h^d$ (first-passage)")
    tau_Gd_peak = peak_lag_Gd(D, nu, d_fp)
    tau_hd_peak = peak_lag_hd(D, nu, d_fp)
    ax2.axvline(tau_Gd_peak, color="b", ls=":", lw=1, label=f"$G^d$ peak={tau_Gd_peak:.3f}")
    ax2.axvline(tau_hd_peak, color="r", ls=":", lw=1, label=f"$h^d$ peak={tau_hd_peak:.3f}")
    ax2.set_xlabel(r"Operational lag $\tau$", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 4])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_kernel_shapes.png"), dpi=120)
    plt.close()


def exp_2_peak_lag_sweep(D: float = 0.5, nu: float = 1.0) -> pd.DataFrame:
    """Sweep over d and compare numerical vs analytical peak lags."""
    d_vals = np.linspace(0.01, 2.0, 30)
    tau = np.linspace(1e-5, 20.0 / nu, 10000)

    records = []
    for d in d_vals:
        k = kernel_Gd(tau, D, nu, d)
        tau_peak_num = float(tau[np.argmax(k)])
        tau_peak_theory = peak_lag_Gd(D, nu, float(d))

        k_hd = kernel_hd(tau, D, nu, d)
        tau_peak_hd_num = float(tau[np.argmax(k_hd)])
        tau_peak_hd_theory = peak_lag_hd(D, nu, float(d))

        # nu -> 0 asymptotic
        tau_peak_zero_nu = d ** 2 / (2.0 * D)

        records.append({
            "d": float(d),
            "tau_peak_Gd_num": tau_peak_num,
            "tau_peak_Gd_theory": tau_peak_theory,
            "tau_peak_Gd_err": abs(tau_peak_num - tau_peak_theory),
            "tau_peak_hd_num": tau_peak_hd_num,
            "tau_peak_hd_theory": tau_peak_hd_theory,
            "tau_peak_zero_nu": tau_peak_zero_nu,
        })

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df["d"], df["tau_peak_Gd_num"], "o", ms=4, label="Numerical peak (G^d)")
    ax.plot(df["d"], df["tau_peak_Gd_theory"], "-r", lw=1.5, label="Theory peak (G^d)")
    ax.plot(df["d"], df["tau_peak_hd_theory"], "--g", lw=1.5, label="Theory peak (h^d)")
    ax.set_xlabel("Displacement d", fontsize=12)
    ax.set_ylabel(r"Peak lag $\tau^*(d)$", fontsize=12)
    ax.set_title("Peak Lag: Numerical vs Theory", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(df["d"], df["tau_peak_Gd_err"], "o-", ms=4)
    ax2.set_xlabel("Displacement d", fontsize=12)
    ax2.set_ylabel("Absolute Error |num - theory|", fontsize=12)
    ax2.set_title("Peak Lag Error", fontsize=13)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_peak_lag.png"), dpi=120)
    plt.close()

    return df


def exp_2_attenuation(D: float = 0.5, nu: float = 1.0) -> pd.DataFrame:
    """Verify integrated attenuation: I_d / I_0 = exp(-d / ell_nu)."""
    ell_nu = math.sqrt(D / nu)
    d_vals = np.linspace(0.0, 3.0 * ell_nu, 20)
    tau_max = max(100.0 / nu, 100.0 * (d_vals[-1] ** 2 / D))
    I_0 = integral_G0(D, nu)  # infinite

    records = []
    for d in d_vals:
        I_d = integral_Gd(D, nu, float(d), tau_max=tau_max)
        ratio_num = I_d / I_0
        ratio_theory = attenuation_ratio(D, nu, float(d))
        records.append({
            "d": float(d),
            "I_d": I_d,
            "I_d_over_I_0_num": ratio_num,
            "I_d_over_I_0_theory": ratio_theory,
            "error": abs(ratio_num - ratio_theory),
        })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(df["d"] / ell_nu, df["I_d_over_I_0_num"], "o", ms=5, label="Numerical $I_d/I_0$")
    ax.semilogy(df["d"] / ell_nu, df["I_d_over_I_0_theory"], "-r", lw=2,
                label=r"Theory: $\exp(-d/\ell_\nu)$")
    ax.set_xlabel(r"$d / \ell_\nu$", fontsize=12)
    ax.set_ylabel(r"$I_d / I_0$", fontsize=12)
    ax.set_title("Integrated Attenuation: Numerical vs Theory", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_attenuation.png"), dpi=120)
    plt.close()

    return df


def exp_2_quote_response(
    D: float = 0.5,
    nu: float = 1.0,
    eps_i: float = 0.05,
    rho_i_q: float = 0.3,
) -> pd.DataFrame:
    """Test linear response to three forcing scenarios."""
    ell_nu = math.sqrt(D / nu)
    d_star = ell_nu * math.log(rho_i_q / eps_i)

    tau_max = 100.0 * (1.0 / nu + d_star ** 2 / D)
    tau = np.linspace(0, tau_max, 5001)[1:]
    dtau = tau[1] - tau[0]

    # Impulse at t=0: approximate delta function as narrow pulse
    pulse_width = 5  # grid points
    J0 = 1.0  # impulse size

    J_impulse = np.zeros_like(tau)
    J_impulse[:pulse_width] = J0 / (pulse_width * dtau)  # normalized

    # Case a: pure depletion impulse
    delta_rho_a = density_response(tau, J_impulse, np.zeros_like(tau), D, nu, d_star)
    delta_p_a = quote_response(delta_rho_a, eps_i, ell_nu)

    # Case b: pure replenishment impulse at displacement d*
    delta_rho_b = density_response(tau, np.zeros_like(tau), J_impulse, D, nu, d_star)
    delta_p_b = quote_response(delta_rho_b, eps_i, ell_nu)

    # Case c: matched depletion + replenishment
    delta_rho_c = density_response(tau, J_impulse, J_impulse, D, nu, d_star)
    delta_p_c = quote_response(delta_rho_c, eps_i, ell_nu)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(tau, delta_rho_a, "b-", lw=1.5, label="(a) Pure depletion")
    ax.plot(tau, delta_rho_b, "r--", lw=1.5, label="(b) Displaced replenishment")
    ax.plot(tau, delta_rho_c, "g:", lw=2, label="(c) Matched depletion + replenishment")
    ax.set_xlabel(r"Operational time $u$", fontsize=12)
    ax.set_ylabel(r"$\delta\rho(p_i, u)$", fontsize=12)
    ax.set_title("Quote-level Density Response", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    ax2 = axes[1]
    ax2.plot(tau, delta_p_a, "b-", lw=1.5, label="(a) Depletion -> quote moves out")
    ax2.plot(tau, delta_p_b, "r--", lw=1.5, label="(b) Replenishment -> attenuated & delayed")
    ax2.plot(tau, delta_p_c, "g:", lw=2, label="(c) Matched: partial cancellation")
    ax2.set_xlabel(r"Operational time $u$", fontsize=12)
    ax2.set_ylabel(r"$\delta p_i(u)$", fontsize=12)
    ax2.set_title("Quote Displacement Response", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_quote_response.png"), dpi=120)
    plt.close()

    # Integrated attenuation identity: use exact kernel integrals to avoid
    # discretisation artifacts from the G^0 ~ tau^{-1/2} singularity.
    I_0_num = integral_G0(D, nu, tau_max)
    I_d_num = integral_Gd(D, nu, d_star, tau_max=tau_max)
    atten_num = I_d_num / I_0_num if I_0_num > 0 else float("nan")
    atten_theory = math.exp(-d_star / ell_nu)
    atten_theory_alt = eps_i / rho_i_q  # at d = d*

    records = [
        {"case": "(a) depletion", "peak_delta_rho": float(delta_rho_a.min()), "integrated": float(I_0_num)},
        {"case": "(b) replenishment", "peak_delta_rho": float(delta_rho_b.max()), "integrated": float(I_d_num)},
        {"case": "attenuation_ratio_num", "value": float(atten_num)},
        {"case": "attenuation_exp(-d*/ell)", "value": float(atten_theory)},
        {"case": "attenuation_eps/rho_q", "value": float(atten_theory_alt)},
    ]
    return pd.DataFrame(records)


def main() -> None:
    print("=" * 60)
    print("EXP 2: Displaced Replenishment Response")
    print("=" * 60)

    D, nu = 0.5, 1.0
    ell_nu = math.sqrt(D / nu)
    print(f"  D={D}, nu={nu}, ell_nu={ell_nu:.4f}")

    # 1. Kernel shapes
    print("\n[1] Plotting kernel shapes...")
    exp_2_kernel_shapes(D, nu)

    # 2. Peak lag sweep
    print("\n[2] Peak lag sweep...")
    df_peaks = exp_2_peak_lag_sweep(D, nu)
    df_peaks.to_csv(os.path.join(RESULTS_DIR, "exp2_peak_lag.csv"), index=False)
    max_err = df_peaks["tau_peak_Gd_err"].max()
    print(f"  Max peak lag error: {max_err:.4e}")

    # 3. Attenuation
    print("\n[3] Integrated attenuation validation...")
    df_atten = exp_2_attenuation(D, nu)
    df_atten.to_csv(os.path.join(RESULTS_DIR, "exp2_attenuation.csv"), index=False)
    max_atten_err = df_atten["error"].max()
    print(f"  Max attenuation error: {max_atten_err:.4e}")
    print(df_atten[["d", "I_d_over_I_0_num", "I_d_over_I_0_theory", "error"]].to_string())

    # 4. Quote response
    print("\n[4] Linear quote response test...")
    rho_i_q = 0.3
    eps_i = 0.05
    d_star = ell_nu * math.log(rho_i_q / eps_i)
    print(f"  d* = ell_nu * ln(rho_q/eps) = {d_star:.4f}, eps/rho_q = {eps_i/rho_i_q:.4f}")

    df_resp = exp_2_quote_response(D, nu, eps_i, rho_i_q)
    df_resp.to_csv(os.path.join(RESULTS_DIR, "exp2_quote_response.csv"), index=False)
    print(df_resp.to_string())

    print("\nExp 2 done. Outputs saved to results/")


if __name__ == "__main__":
    main()
