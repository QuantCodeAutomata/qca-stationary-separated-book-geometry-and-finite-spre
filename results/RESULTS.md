# Experiment Results — Stationary Separated-Book Geometry & Finite Spread

> All numbers obtained from synthetic PDE / Green-function runs (no empirical market data).
> Equal-coefficient branch: D_A = D_B = D, ν_A = ν_B = ν unless stated.
> Test suite: **76 / 76 tests pass**.

---

## Experiment 1 — Stationary Separated-Book Geometry

**Parameters (baseline)**

| Symbol | Value | Meaning |
|--------|-------|---------|
| D_A = D_B | 0.5 | diffusivity |
| ν_A = ν_B | 1.0 | cancellation rate |
| κ | 2.0 | reaction coefficient |
| ε_A = ε_B | 0.1 | density thresholds |
| Λ_lit | 1.0 | external lit-source amplitude |
| q_b / q_a | −2.0 / +2.0 | placement edges, Σ = 4 |
| ℓ_ν | √(D/ν) = 0.7071 | cancellation length |

**Baseline result (single symmetric solve)**

| Metric | Value |
|--------|-------|
| p_b | −0.8630 |
| p_a | +0.8630 |
| Measured spread s | 1.7259 |
| Formula spread s_pred = Σ − d_A* − d_B* | 1.7168 |
| Absolute spread error | **0.0091** |
| x_φ (reaction price) | 0.0000 |
| Midpoint m | 0.0000 |
| \|m − x_φ\| | 0.0000 |
| d_B* = d_A* | 1.1416 |
| ℓ_ν / s (weak-overlap ratio) | **0.410** |
| Bid profile fit ℓ_fit | 0.7043 (theory 0.7071) |
| Ask profile fit ℓ_fit | 0.7043 (theory 0.7071) |
| Profile R² | 1.0000 |

**Interpretation** — Measured spread matches formula `s ≈ Σ − d_A* − d_B*` with 0.5 % relative error.
Inward density profiles inside the withheld interval [q_b, q_a] decay exponentially with the theoretically predicted scale ℓ_ν = 0.707. Midpoint and reaction price coincide to machine precision (symmetric branch).

**Σ sweep (positive-spread rows, 10 of 16 cases)**

| Σ | s | s_pred | error |
|---|---|--------|-------|
| 2.33 | 0.1127 | 0.0296 | 0.083 |
| 3.00 | 0.6768 | 0.6382 | 0.039 |
| 4.00 | 1.7259 | 1.7168 | 0.009 |
| 5.00 | 2.6441 | 2.6422 | 0.002 |
| 6.00 | 3.7260 | 3.7253 | 0.001 |

Mean \|error\| / mean spread (positive-spread rows): **0.0103**. Spread grows approximately one-for-one with Σ.

**ε threshold sweep** (simultaneous change ε_A = ε_B = ε, same Σ = 2 grid)

| ε | spread | ∂s/∂(ln ε) | theory ℓ_ν |
|---|--------|-------------|------------|
| 0.04 | −0.44 | 1.257 | 0.707 |
| 0.12 | +1.00 | 1.349 | 0.707 |
| 0.25 | +2.01 | 1.376 | 0.707 |

Note: changing both ε_A and ε_B simultaneously gives ∂s/∂(ln ε) ≈ 2 × ℓ_ν = 1.41, which is consistent with the per-side identity `∂s/∂(ln ε_i) = ℓ_{ν,i}` (here both sides change in tandem, approximately doubling the sensitivity).

**Output files**

- `results/exp1_profiles_baseline.png` — density profiles with exponential fits
- `results/exp1_sigma_sweep.png` — measured vs predicted spread over Σ
- `results/exp1_eps_sweep.png` — spread and threshold sensitivity vs ε
- `results/exp1_sigma_sweep.csv`, `results/exp1_eps_sweep.csv`, `results/exp1_baseline_metrics.json`

---

## Experiment 2 — Displaced Replenishment Response Kernel

**Parameters**: D = 0.5, ν = 1.0, ℓ_ν = 0.7071.

**Kernel peak-lag validation**

Formula: `τ*(d) = (√(1 + 4ν d²/D) − 1) / (4ν)`.

- Grid: 20 d-values from 0 to 3 × ℓ_ν; numerical peak found via `np.argmax(G^d)`.
- **Max peak-lag error across all d**: 1.91 × 10⁻³ (< 0.2 %).
- ν → 0 asymptotic `τ*(d) → d²/(2D)` confirmed at ν = 0.01.

**Integrated attenuation I_d / I_0 = exp(−d/ℓ_ν)**

Max absolute error across 20 d-values: **2.08 × 10⁻⁸** (machine precision for the erf formula).

**Impulse response test** (d* = ℓ_ν ln(ρ^q/ε) = 1.267, ε/ρ^q = 0.1667)

| Case | Peak δρ | Integrated |
|------|---------|------------|
| (a) depletion at quote (G⁰) | −0.7192 | 0.7071 |
| (b) replenishment at q_i (G^d) | +0.0730 | 0.1179 |

Attenuation triple-check (all three routes agree to machine precision):

| Route | Value |
|-------|-------|
| I_d / I_0 (numerical) | 0.166667 |
| exp(−d*/ℓ_ν) (analytical) | 0.166667 |
| ε/ρ^q (direct ratio) | 0.166667 |

**Interpretation** — Displaced replenishment arrives later and weaker than same-location depletion; integrated attenuation matches e^(−d*/ℓ_ν) = ε/ρ^q exactly as predicted.

**Output files**

- `results/exp2_kernel_shapes.png` — G⁰ vs G^d shapes
- `results/exp2_peak_lag.png` — numerical vs analytical peak lag
- `results/exp2_attenuation.png` — attenuation ratio vs d
- `results/exp2_quote_response.png` — density and quote response to impulses
- `results/exp2_peak_lag.csv`, `results/exp2_attenuation.csv`, `results/exp2_quote_response.csv`

---

## Experiment 3 — Finite-Window Quote Capacity

**Parameters**: D = 0.5, ν = 1.0, ε = 0.1, ℓ_ν = 0.7071.

**Exact integral I(T) = [2√(Dν)]⁻¹ erf(√(νT))**

Max relative error (numerical quadrature vs closed-form erf): **3.02 × 10⁻⁸**.

**Capacity curve ℒ(T) = ℓ_ν⁰ · T / I(T)**

| T | ℒ_exact | ℒ_asymptotic | rel error |
|---|---------|--------------|-----------|
| 0.01 | 0.00889 | 0.00886 | 0.33 % |
| 0.10 | 0.02896 | 0.02803 | 3.24 % |
| 1.00 | 0.11867 | 0.08862 | 25.3 % |
| 10.0 | 1.00001 | 0.28025 | 71.9 % |

**Short-window scaling exponent** d log ℒ / d log T

- At ν T < 0.1: mean exponent = **0.507** (theory → 0.5) ✓
- At T < 0.01: mean exponent = **0.501** ✓

**Corrected asymptotic error** at ν T < 0.3: max relative error = **7.02 × 10⁻⁴**.

**Interpretation** — Capacity scales as √T for short windows, confirming that instantaneous liquidity coefficient at T → 0 is zero. Exact formula matches erf integral to numerical precision.

**Output files**

- `results/exp3_capacity.png` — capacity curves (exact, asymptotic, log-log)
- `results/exp3_capacity_curves.csv`, `results/exp3_integral_validation.csv`, `results/exp3_param_sweep.csv`

---

## Experiment 4 — Completion-Memory Placement Standoff

**Note**: Main-text excerpt does not specify the completion law; experiment runs in **Mode B** (exogenous Q-paths as mechanism demonstration). Σ_0 = 2.0, χ_s = 0.5 (sensitivity).

**Exogenous scenario results**

| Scenario | max Q_Σ | max Σ | Q_Δ = 0? |
|----------|---------|--------|----------|
| Immediate completion | **0.000** | **2.000** (= Σ_0) | yes |
| Symmetric delayed | 2.000 | 3.000 | yes |
| Asymmetric delayed | 2.000 | 3.000 | no |

**Static spread vs Σ (stationary PDE, weak-source regime)**

| Σ | Spread |
|---|--------|
| 1.50 | 0.875 |
| 2.00 | 1.438 |
| 2.50 | 1.798 |
| 3.00 | 2.364 |

Spread change per unit Σ change: **0.993** (theory → 1.0) ✓

**Exponential-decay completion mechanism**

- Fast completion (μ = 100, "immediate"): max Q_Σ = **0.007** ≈ 0 ✓
- Slow completion (μ = 0.1, "delayed"): max Q_Σ = **4.64** >> 0 ✓

**Interpretation** — Pending replenishment widens the source-free interval Σ = Σ_0 + χ_s Q_Σ even when signed imbalance Q_Δ = 0, demonstrating that total pending volume — not just directional imbalance — can widen observed spread.

**Output files**

- `results/exp4_standoff_scenarios.png` — Q_Σ and Σ time series under three scenarios
- `results/exp4_completion_mechanism.png` — fast vs slow completion comparison
- `results/exp4_immediate.csv`, `results/exp4_symmetric.csv`, `results/exp4_asymmetric.csv`,
  `results/exp4_spread_vs_Sigma.csv`, `results/exp4_completion_mechanism.csv`

---

## Experiment 5 — Midpoint vs Reaction Price

**Parameters (baseline)**: D = 0.5, ν = 1.0, κ = 0.5, Λ_lit = 0.3, ε_A = ε_B = 0.05, q_b = −3.0, q_a = +3.0 (Σ = 6).

**Symmetric baseline**

| Metric | Value |
|--------|-------|
| p_b / p_a | −2.6991 / +2.6991 |
| Spread s | 5.3982 |
| Midpoint m | −0.0000 |
| Reaction price x_φ | −0.0000 |
| \|m − x_φ\| | **0.0000** (machine precision) ✓ |
| Imbalance benchmark s_φ | 12.1898 (2.26× larger than s) |

**Σ sweep** (x_φ pinned to 0 under symmetry)

| Σ | Spread s | x_φ | m |
|---|---------|-----|---|
| 1.0 | 0.390 | 0.000 | 0.000 |
| 3.0 | 2.385 | 0.000 | 0.000 |
| 6.0 | 5.398 | 0.000 | 0.000 |

x_φ range = 0.000 while spread range = 5.008; spread changes **>> 1×** more than x_φ, confirming geometrical separation of mechanisms.

**Unequal-threshold asymmetry** (Σ = 6, mirrored sources)

| ε_A | ε_B | s | m | x_φ | \|m − x_φ\| |
|-----|-----|---|---|-----|------------|
| 0.05 | 0.05 | 5.398 | 0.000 | 0.000 | 0.000 |
| 0.03 | 0.08 | 5.370 | −0.347 | 0.000 | **0.347** |
| 0.08 | 0.03 | 5.370 | +0.347 | 0.000 | **0.347** |
| 0.02 | 0.08 | 5.083 | −0.490 | 0.000 | **0.490** |
| 0.08 | 0.02 | 5.083 | +0.490 | 0.000 | **0.490** |

**Asymmetric sources** (max \|m − x_φ\| = **0.490**)

Under ratio = 1.6, the midpoint departs from the reaction price by 0.333.

**Interpretation** — m = x_φ holds exactly under symmetric mirrored conditions; the equality breaks as soon as thresholds or source amplitudes are made asymmetric. Spread is controlled by total-density placement geometry (Σ sweep changes s strongly, x_φ not at all), while reaction-price movement is tied to the imbalance field φ = ρ_B − ρ_A. The imbalance-only benchmark overestimates observed spread by a factor ~2.26, confirming that the separated two-field mechanism, not imbalance thresholding, governs finite spread.

**Output files**

- `results/exp5_symmetric_fields.png` — symmetric baseline density snapshot
- `results/exp5_mechanism_separation.png` — spread and x_φ vs Σ
- `results/exp5_asymmetry_decoupling.png` — unequal-threshold asymmetry
- `results/exp5_sigma_sweep.csv`, `results/exp5_unequal_thresholds.csv`, `results/exp5_asymmetry.csv`

---

## Summary Table

| Experiment | Key Claim | Status | Key Number |
|-----------|-----------|--------|------------|
| 1 | s ≈ Σ − d_A* − d_B* | ✅ | mean error / mean s = **1.03 %** |
| 1 | Inward exp. profile, slope = 1/ℓ_ν | ✅ | R² = 1.000, ℓ_fit = 0.704 (theory 0.707) |
| 2 | Peak lag τ*(d) formula | ✅ | max error = **1.91 × 10⁻³** |
| 2 | Attenuation I_d/I_0 = e^(−d/ℓ_ν) | ✅ | max error = **2.08 × 10⁻⁸** |
| 2 | Attenuation = ε/ρ^q at d = d* | ✅ | triple match = 0.1667 |
| 3 | I(T) = erf formula | ✅ | max rel. error = **3.02 × 10⁻⁸** |
| 3 | ℒ(T) ∝ √T for short windows | ✅ | scaling exponent = **0.501** |
| 4 | Immediate completion → Q_Σ = 0 | ✅ | max Q_Σ = **0.007** |
| 4 | Delayed completion → Σ > Σ_0 | ✅ | Q_Σ = 4.64 |
| 4 | Spread ∝ Σ at rate ~1 | ✅ | dS/dΣ = **0.993** |
| 5 | m = x_φ under symmetry | ✅ | \|m − x_φ\| = **0.000** |
| 5 | Spread changes with Σ, x_φ stays fixed | ✅ | ratio = **> 10⁷** |
| 5 | Asymmetry decouples m from x_φ | ✅ | max \|m − x_φ\| = **0.490** |

**All 76 pytest tests pass.**
