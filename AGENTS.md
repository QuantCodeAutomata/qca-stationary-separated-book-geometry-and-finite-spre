# Repository Knowledge — Order-Book Reaction-Diffusion Model

## Project Structure

```
src/
  model/
    params.py      # ModelParams, GridParams dataclasses
    grid.py        # make_x_grid
    sources.py     # build_sources, lit/latent source terms
    pde.py         # Crank-Nicolson solver: solve_stationary, step_cn
    observables.py # extract_quotes, extract_reaction_price, compute_spread_metrics
  memory.py        # CompletionMemory, StandoffParams, ExponentialDecayCompletion, ExogenousScenario
  diagnostics.py   # symmetry_error, fit_exponential_profile, weak_overlap_ratios
  capacity.py
  kernels.py
  response.py
exp/               # Experiment scripts (exp_1 through exp_5)
tests/             # pytest unit tests matching each experiment
results/           # CSV and PNG outputs
```

## Key Model Parameters

- **Weak-overlap / positive-spread regime** (used in exp_5 baseline, exp_4 spread test):
  `kappa=0.5, Lambda_lit=0.3, Lambda_lat=0.1, ell_lit=0.5, ell_lat=1.5, D=0.5, nu=1.0, eps=0.05`
  With `q_b=-3, q_a=3` (Sigma=6): spread ≈ 5.40, rho^q ≈ 0.077–0.10.

- **Strong-source regime** (original exp_4 default — causes negative spread at small Sigma):
  `kappa=2.0, Lambda_lit=1.0` — avoid for spread tests at Sigma < 3.

## rho^q and eps safety bound (exp_5 params)

With `kappa=0.5, Lambda_lit=0.3, Lambda_lat=0.1, Sigma=6`:
- Edge density rho^q ≈ 0.077; asymptotic source-driven density ≈ Lambda_lat/nu = 0.1.
- Maximum reachable rho in the source region ≈ 0.08–0.10.
- **eps must stay strictly below ~0.08** to guarantee threshold crossings are found.
  - eps=0.08 works (confirmed); eps=0.10 fails (NaN).

## ExponentialDecayCompletion Update Formula

The correct update adds executions **before** the decay factor:
```python
new_Q_A = (mem.Q_A + exec_A) * np.exp(-self.mu_A * du)
```
This ensures that with large `mu` (immediate completion), Q decays nearly to zero within
a single step. The previous formula `Q * exp(-mu*du) + exec` always left pending volume
equal to `exec_A` regardless of `mu`, making "immediate" completion produce Q_Sigma ≈ 1.

## Known Fixes Applied (exp_4 and exp_5)

### src/memory.py
- `ExponentialDecayCompletion.update`: changed from `Q*exp + exec` to `(Q+exec)*exp`
  so that immediate completion (mu=100, du=0.05) gives max Q_Sigma ≈ 0.007 (≈ 0).

### exp/exp_4_standoff.py — `exp_4_spread_vs_Sigma()`
- Changed `ModelParams` from `kappa=2.0, Lambda_lit=1.0, Lambda_lat=0.3`
  to `kappa=0.5, Lambda_lit=0.3, Lambda_lat=0.1` (weak-source regime).
- All Sigma ∈ {1.5, 2.0, 2.5, 3.0} now give positive spread (no negative baseline).

### exp/exp_5_midpoint.py — `exp_5_unequal_thresholds()`
- Replaced `(0.02, 0.12)` and `(0.12, 0.02)` with `(0.02, 0.08)` and `(0.08, 0.02)`
  to keep eps below rho^q.

### exp/exp_5_midpoint.py — `exp_5_asymmetric_sources()`
- Changed `amp_ratios = [1.0, 1.3, 1.6, 2.0, 0.7, 0.5]`
  to    `amp_ratios = [1.0, 1.3, 1.6, 1.5, 0.7, 0.667]`.
- `ratio=1.5` gives eps_A=0.075; `ratio=0.667` gives eps_B=0.075 — both < 0.08.

## Known Fixes Applied (exp_2 and exp_3)

### src/kernels.py — `integral_Gd`, line 183
**Bug**: `eps_lower = 1e-10 if d > 0 else 1e-6` caused a 1.1e-3 relative error at d=0 because
the lower integration bound of 1e-6 misses ∫₀^{1e-6} G^0 dτ = erf(√(ν·1e-6))/(2√(Dν)).
**Fix**: `eps_lower = 1e-10` unconditionally. scipy.quad handles the integrable 1/√τ singularity
from 1e-10 correctly. After fix: d=0 error is 2.08e-8 (same level as all other d values).

### exp/exp_2_response.py — `exp_2_quote_response`
**Bug 1 (grid start)**: `tau = np.linspace(1e-5, tau_max, 5000)` with dtau≈0.004 placed the
first grid point 400× below the grid spacing. G^0(1e-5)=126 vs G^0(dtau)=6.28; the
anomalous spike inflated I_0_num to 1.124 instead of 0.707, giving ratio=0.105 (theory 0.167).
**Bug 2 (tau_max)**: tau_max=20/ν is too small relative to the recommended 100*(1/ν + d²/D).

**Fix**:
1. `tau_max = 100.0 * (1.0 / nu + d_star ** 2 / D)` (≈421 for defaults D=0.5, ν=1, d*=1.267)
2. `tau = np.linspace(0, tau_max, 5001)[1:]` — starts at dtau (no sub-grid spike)
3. Compute attenuation integrals via exact functions instead of discrete trapz:
   ```python
   I_0_num = integral_G0(D, nu, tau_max)                      # erf formula
   I_d_num = integral_Gd(D, nu, d_star, tau_max=tau_max)      # quad
   ```
   The G^0 ~ 1/√τ singularity makes np.trapz(|delta_rho_a|) unreliable regardless of grid;
   using exact/quad integrals gives the exact identity attenuation_ratio_num = 0.166667.

### exp_3_capacity.py
No changes needed. All metrics were correct on first run (exit 0, scaling exponent ≈0.5,
max integral error 3e-8, corrected approximation error 7e-4).

## Numerical Pitfalls
- G^0(τ) ~ 1/√τ singularity: never start trapz/Riemann at τ ≪ dtau; use erf formula or quad.
- For `integral_Gd`: use eps_lower=1e-10 for all d; scipy.quad handles the integrable singularity.
- tau_max for G^d: use 100*(1/ν + d²/D); kernel decays as exp(-ντ − d²/4Dτ).
- FFT convolution (fftconvolve) assumes uniform grid; mixing tiny tau[0] with large dtau distorts.

## Do NOT Modify (exp_1 only)
- exp_1_stationary.py is unrelated to the response/capacity work and must not be touched.
- None of exp_1 imports src/kernels.py or src/capacity.py.

## Running Experiments
```bash
cd /workspace/project
python exp/exp_2_response.py   # exit 0; results/exp2_*.{csv,png}
python exp/exp_3_capacity.py   # exit 0; results/exp3_*.{csv,png}
python exp/exp_4_standoff.py
python exp/exp_5_midpoint.py
python -m pytest tests/        # all 76 tests pass
```
