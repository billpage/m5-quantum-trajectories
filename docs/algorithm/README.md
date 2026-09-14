# docs/algorithm

Core algorithm specifications for the M5 project: the gridless
ψ-KDE quantum-trajectory framework built on Nelson's stochastic mechanics
and Holland's bi-Hamilton–Jacobi theory.

These two documents describe the same algorithm from two angles — a
grid-based field estimator and a fully gridless (swarmalator)
reformulation — rather than a dependency chain; either can be read first.
Extended derivations and context belong in `docs/supplement/` and
`docs/analysis/`.

## Contents

- **[`NelsonMechanics_SchrodingerBridge_Algorithm.md`](NelsonMechanics_SchrodingerBridge_Algorithm.md)** —
  The current M5 algorithm: a derivative-free particle method for quantum
  dynamics grounded in Nelson's stochastic mechanics and the
  Schrödinger-bridge FBSDE framework. Each particle carries a position X
  and action phase S; three readouts from a shared Gauss–Hermite
  candidate cloud implement the full Holland bi-Hamilton–Jacobi structure
  with no spatial differentiation of the density. Four advances over the
  original formulation: fixed GH quadrature nodes (deterministic
  STEER/WEIGH with controlled polynomial error), a log-density mean
  weight (machine-precision osmotic-velocity divergence), mirror
  particles (2–4× better boundary-tail Q estimates), and backward-channel
  readouts (Q̃ and the full bi-HJ coupling from the same forward ensemble,
  no second species or backward simulation needed). See the companion
  supplement for the Hackebill–Poirier classification and
  Wasserstein-dynamics context.
- **[`NelsonMechanics_SchrodingerBridge_Swarmalator.md`](NelsonMechanics_SchrodingerBridge_Swarmalator.md)** —
  A gridless reformulation of M5: each particle senses its local phase
  environment via a coherent kernel average of its neighbours' phases,
  computing velocity directly as vᵢ = (ℏ/m) Im(j'ᵢ/jᵢ) with no spatial
  grid, binning, interpolation, or finite differences. All M5 theorems,
  readouts, and backward-channel diagnostics carry over unchanged from
  the companion grid-based algorithm document; only the field-estimation
  method differs. Named for its kinship with the swarmalator literature
  (O'Keeffe, Hong & Strogatz 2017), where the quantum action S plays the
  role of the internal phase oscillator and the coherent kernel average
  is the coupling function.
