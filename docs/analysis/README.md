# docs/analysis

Extended mathematical derivations, comparative analyses, and reference
material for the M5 project.

These are free-standing topic notes rather than a strict dependency
chain — kernel theory, a Q&A discussion, an FBSDE synthesis, comparative
work against Holland/Nelson and against Yang & Han — so this index is an
unordered list, not a numbered ladder. Several notes do build on each
other in places (noted below where relevant), and this list is written
so it can become a numbered ladder later if the dependency structure
firms up enough to be worth stating that way.

## Contents

- **[`FBSDE_SchrodingerBridge_Nelson_Holland.md`](FBSDE_SchrodingerBridge_Nelson_Holland.md)** —
  A synthesis connecting three formalisms: forward-backward stochastic
  differential equations (FBSDEs, Pardoux–Peng / Ma–Protter–Yong),
  Schrödinger's bridge problem (the most-likely evolution conditional on
  a rare large-deviation event, entropic optimal transport), and the
  Nelson–Holland stochastic-mechanics picture of quantum dynamics. Traces
  how M5's forward-evolving ensemble with GH-quadrature readouts realizes
  the FBSDE/bridge structure without ever solving a backward equation
  directly.
- **[`Holland_Nelson_FokkerPlanck_Analysis.md`](Holland_Nelson_FokkerPlanck_Analysis.md)** —
  A comparative analysis of Holland's bi-Hamilton–Jacobi decomposition
  and Nelson's stochastic mechanics, both starting from the same
  Schrödinger equation. Derives and compares the forward/backward
  Fokker–Planck pair (Holland's equation 4.17), the backward Kolmogorov
  equation, and the coupled SDE system each framework implies,
  establishing the precise correspondence between Holland's σ± action
  functions and Nelson's forward/backward drift velocities.
- **[`Method5_QA_Discussion.md`](Method5_QA_Discussion.md)** — A Q&A
  discussion building intuition for why selecting Brownian candidates
  proportional to √ρ is sufficient to reproduce full quantum dynamics.
  Moves from Method 5's time-reversal symmetry, through its connection to
  the Sinkhorn/IPFP algorithm for Schrödinger bridges, to its
  interpretation as Newton's second law on Wasserstein space.
- **[`Method5_v3_Analysis.md`](Method5_v3_Analysis.md)** — *Historical.*
  Method 5 v3, a self-consistent particle method for the full Holland
  bi-Hamilton–Jacobi system: two coupled real Hamilton–Jacobi equations
  for the forward/backward action functions σ±. Extends the v2
  dual-readout framework (√ρ-selection, mean-weight quantum potential)
  with a backward weight 1/√ρ probing the anti-diffusive sector, a
  log-density mean-weight for machine-precision osmotic-velocity
  divergence, and mirror particles for KDE boundary correction. The
  log-density approach is identified as numerically superior to direct
  1/√ρ evaluation, avoiding exponential ill-conditioning. Superseded by
  [`../algorithm/NelsonMechanics_SchrodingerBridge_Algorithm.md`](../algorithm/NelsonMechanics_SchrodingerBridge_Algorithm.md)
  §7.3; kept for the record of what v3 established and why it changed.
- **[`complex_trajectories_analysis.md`](complex_trajectories_analysis.md)** —
  Examines Yang & Han's complex-valued Hamilton–Jacobi mechanics (Found.
  Phys. 50, 2020) — writing ψ = e^{iW/ℏ} with W complex rather than the
  standard polar decomposition — against Holland's dismissal of the
  approach, and identifies structural parallels with M5: the imaginary
  part of the complex velocity matches the Nelson osmotic velocity that
  M5 extracts via √ρ-selection, and Yang–Han's two point sets
  (intersections vs. projections) parallel M5's forward/backward
  dual-readout architecture. Closes with what each framework gets that
  the other doesn't.
- **[`kernel_expressions.md`](kernel_expressions.md)** — Reference for
  the ψ-KDE swarmalator's kernel functions: closed-form expressions and
  first derivatives for the Gaussian, quintic B-spline, and compact
  rational kernels, the resulting ψ-KDE kernel sums (density, coherent
  complex current, and their derivatives), the derived fields
  (reconstructed wavefunction, current velocity, GH-WEIGH quantum
  potential), the Poirier C-derivative connection, and a K″/K structure
  comparison across all three kernels.
- **[`kernel_probe_theory.md`](kernel_probe_theory.md)** — Develops the
  theory of kernel and probe selection for the gridless swarmalator
  algorithm (§7 of the companion Swarmalator document): the compact
  rational kernel used in ψ-KDE field estimates, and Gauss–Jacobi probe
  nodes as an alternative to the original Gaussian-kernel /
  Gauss–Hermite-probe combination. Builds on
  [`kernel_expressions.md`](kernel_expressions.md).
