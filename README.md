# M5: Particle-Based Quantum Simulation via Stochastic Mechanics

**M5 (Method 5)** is a gridless, derivative-free quantum trajectory algorithm rooted in Nelson's stochastic mechanics, Holland's bi-Hamilton–Jacobi (bi-HJ) theory, and Schrödinger bridge theory.

## Core Innovation: ψ-KDE

The central technical contribution is the **ψ-KDE** (psi-KDE) estimator — a coherent phase-weighted kernel density estimator that reconstructs the wavefunction directly as:

$$\hat{\psi} = \frac{j_h}{\sqrt{n_h}}$$

where $j_h$ is the complex "coherent average" of neighbouring phase factors $e^{iS_k/\hbar}$ and $n_h$ is the standard density KDE. This construction:

- Eliminates separate estimation of density $\rho$ and phase $S$
- Handles interference nodes correctly via destructive interference in $j_h$
- Enforces the Wallstrom quantisation condition automatically

## Algorithm Overview

Each particle carries position $X$ and phase $S$. The dynamics proceed through:

1. **Phase gradient velocity:** $v = (\hbar/m)\,\mathrm{Im}(j'/j)$ — computed directly from neighbours' phases, no grid or finite differences
2. **Classical advection:** $X_{\mathrm{class}} = X + v \cdot dt$
3. **Stochastic diffusion (STEER):** Gauss–Hermite quadrature probes weighted by $\sqrt{\rho}$ implement the forward osmotic drift, maintaining quantum equilibrium $\rho = |\psi|^2$
4. **Quantum potential (WEIGH):** The same GH candidate cloud simultaneously extracts the quantum potential $Q$ via the mean-weight ratio — no density differentiation required
5. **Phase update:** $S \to S - (V + Q + \tfrac{1}{2}mv^2)\,dt$

The algorithm is fully gridless and derivative-free: the only spatial information comes from inter-particle phase comparisons.

## Theoretical Foundations

| Framework | Role in M5 |
|---|---|
| **Nelson's stochastic mechanics** | Provides the forward/backward SDE structure and osmotic velocity |
| **Holland's bi-HJ theory** | Decomposes the Schrödinger equation into coupled forward/backward Hamilton–Jacobi equations for $\sigma_\pm = S \pm (\hbar/2)\ln\rho$ |
| **Schrödinger bridge theory** | Connects the forward–backward coupling to optimal transport and the FBSDE framework |
| **Madelung hydrodynamics** | Supplies the continuity + quantum HJ equation pair underlying the particle dynamics |

## Key Design Principles

- **Physically motivated over ad hoc:** Numerical patches (e.g., clipping the quantum potential) are rejected in favour of reformulations with physical justification (e.g., the $Q_{\log}$ formulation working in log-density space)
- **Validation discipline:** All algorithmic changes are validated against FFT split-operator Schrödinger reference solutions
- **Single candidate cloud:** The same GH probe serves STEER (first derivative → position), WEIGH-forward (second derivative → $Q$), and WEIGH-backward (osmotic divergence → $\tilde{Q}$) simultaneously

## Repository Structure

```
docs/           — Algorithm documents and mathematical analysis
  algorithm/    — Core algorithm specification
  supplement/   — Extended analysis (Wasserstein dynamics, ontology, etc.)
  analysis/     — Mathematical derivations and proofs
src/            — Python implementation
  m5_gridless_opt.py      — Reference gridless swarmalator implementation
  m5psi_kde_catstate.py   — Cat state collision test case
  m5psi_kde_trajectories.py — Phase-coloured trajectory visualisation
  method5_2d_ho.py        — 2D harmonic oscillator
  method5_nd.py           — N-dimensional generalisation
```

## Test Cases (in order of difficulty)

1. **Free Gaussian wave packet** — single packet propagation
2. **Harmonic oscillator ground state** — stationary state stability
3. **Cat state collision** — counter-propagating Gaussians with interference

## Requirements

- Python 3.8+
- NumPy
- Optional: CuPy (GPU acceleration via `cupy-cuda12x`)

## References

- Nelson, E. (1966). Derivation of the Schrödinger Equation from Newtonian Mechanics. *Phys. Rev.* 150, 1079.
- Nelson, E. (2012). Review of stochastic mechanics. *J. Phys.: Conf. Ser.* 361, 012011.
- Holland, P. (2021). Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory. arXiv:2111.09235.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). Stochastic Control Liaisons. *SIAM Review* 63, 249–313.
- Hackebill, A. & Poirier, B. (2026). The Problem of Sparse Ontology for Hydrodynamic Formulations of Quantum Mechanics. arXiv:2602.21106.
- Wallstrom, T.C. (1994). Inequivalence between the Schrödinger equation and the Madelung hydrodynamic equations. *Phys. Rev. A* 49, 1613.

## Affiliation

This project is developed by Bill Page ([NewSynthesis](https://newsynthesis.org)) with contributions from David and Claude (Anthropic).

## License

TBD
