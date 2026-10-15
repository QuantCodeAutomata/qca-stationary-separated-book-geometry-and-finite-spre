# Stationary Separated-Book Geometry and Finite Spread

A Python research repository implementing a **two-field reaction–diffusion order-book model** that
generates a finite bid–ask spread from separated bid and ask standing-density fields with a
source-free (withheld) interval around the reaction boundary.

All experiments are purely synthetic — no empirical market data are required.

---

## Scientific Background

The model evolves two density fields ρ_A(x, u) and ρ_B(x, u) on a log-price grid in
**operational time** u according to:

```
∂_u ρ_i = D_i ∂_{xx} ρ_i − ν_i ρ_i − κ ρ_A ρ_B + s_i(x, u)
```

where:
- D_i — diffusivity of side i ∈ {A, B}
- ν_i — cancellation rate
- κ — cross-reaction (execution) coefficient
- s_i — external source, withheld from the interval (q_b, q_a)

Observed quotes p_b, p_a are density-level crossings ρ_i(p_i) = ε_i.
The **spread formula** on the quasistationary weak-overlap branch is:

```
s ≈ Σ − d_A* − d_B*
```

where Σ = q_a − q_b (placement standoff) and d_i* = ℓ_{ν,i} ln(ρ_i^q / ε_i) are penetration depths.

---

## Repository Structure

```
.
├── data/                        # (empty — all inputs are synthetic)
├── src/
│   ├── model/
│   │   ├── params.py            # ModelParams dataclass
│   │   ├── grid.py              # price and operational-time grids
│   │   ├── sources.py           # lit, latent, and external sources
│   │   ├── pde.py               # Crank–Nicolson PDE time-stepper
│   │   └── observables.py       # quote extraction, spread, x_φ, penetration depths
│   ├── kernels.py               # G_i^d density-response and h_i^d first-passage kernels
│   ├── response.py              # convolution-based linear density / quote response
│   ├── capacity.py              # finite-window quote capacity formulas
│   ├── memory.py                # completion-memory pending-volume state
│   └── diagnostics.py           # peak lag, attenuation, and regime-check utilities
├── exp/
│   ├── exp_1_stationary.py      # Exp 1: stationary geometry, spread formula validation
│   ├── exp_2_response.py        # Exp 2: displaced kernel, peak lag, attenuation
│   ├── exp_3_capacity.py        # Exp 3: finite-window capacity and √T scaling
│   ├── exp_4_standoff.py        # Exp 4: completion-memory placement standoff
│   └── exp_5_midpoint.py        # Exp 5: midpoint vs reaction price separation
├── tests/
│   ├── test_exp_1.py
│   ├── test_exp_2.py
│   ├── test_exp_3.py
│   ├── test_exp_4.py
│   └── test_exp_5.py
├── results/
│   ├── RESULTS.md               # Summary of all experiment findings
│   └── *.png / *.csv / *.json  # Figures and data tables
├── AGENTS.md
├── README.md
└── .gitignore
```

---

## Experiments

| ID | Title | Key Result |
|----|-------|------------|
| exp_1 | Stationary separated-book geometry | s ≈ Σ − d_A* − d_B*, mean error 1.0 % |
| exp_2 | Displaced replenishment response | Peak lag matches τ*(d) to < 0.2 %; attenuation error < 10⁻⁷ |
| exp_3 | Finite-window quote capacity | ℒ(T) ∝ √T; exact erf formula matches quadrature to 10⁻⁸ |
| exp_4 | Completion-memory standoff | Q_Σ widens Σ by χ_s Q_Σ; spread rate ≈ 1.0 per unit Σ |
| exp_5 | Midpoint vs reaction price | m = x_φ under symmetry; decouples up to 0.49 under asymmetry |

---

## Running the Experiments

```bash
# Run all experiments
python exp/exp_1_stationary.py
python exp/exp_2_response.py
python exp/exp_3_capacity.py
python exp/exp_4_standoff.py
python exp/exp_5_midpoint.py

# Run test suite (76 tests)
pytest tests/ -v
```

Results (figures + CSVs) are written to `results/`.

---

## Dependencies

Core libraries (all pre-installed in the target environment):

| Package | Role |
|---------|------|
| numpy | array operations, PDE stepping |
| scipy | sparse linear algebra, erf, integration |
| pandas | tabular results |
| matplotlib | visualisations |

No external market-data libraries are used — all inputs are synthetic.

---

## Key Formulas

### Spread formula (Eq. 7 of the paper)
```
s ≈ Σ − d_A* − d_B*,   d_i* = ℓ_{ν,i} ln(ρ_i^q / ε_i)
```

### Density-response kernel
```
G_i^d(τ) = exp(−ν_i τ) / √(4π D_i τ) × exp(−d² / (4 D_i τ))
```

### Finite-window capacity
```
ℒ_i(T) = ℓ_i⁰ · T / I_i(T),   I_i(T) = [2√(D_i ν_i)]⁻¹ erf(√(ν_i T))
```

### Placement standoff
```
Σ = Σ_0 + χ_s Q_Σ,   Q_Σ = Q_A + Q_B
```
