# Quantum Swarmalator Algorithm: Gridless Coherent Particle Dynamics

## A ψ-KDE Alternative to the Grid-Based Method 5

---

### Abstract

We present a gridless particle algorithm for quantum dynamics in which each particle determines its velocity by sensing its local phase environment through a coherent kernel average of its neighbours' phases. The algorithm — a "quantum swarmalator" — replaces the grid-based field estimation pipeline of the Method 5 (M5) algorithm with direct particle-to-particle kernel sums, eliminating all spatial grids, binning, interpolation, and finite differences. The velocity of particle i is:

    vᵢ = (ℏ/m) Im(j'ᵢ / jᵢ)

where jᵢ = Σⱼ K_h(Xᵢ−Xⱼ) exp(iSⱼ/ℏ) is the coherent average of neighbouring phase factors and j'ᵢ = Σⱼ K'_h(Xᵢ−Xⱼ) exp(iSⱼ/ℏ) is its derivative-kernel counterpart. The selection weights for the osmotic drift and the quantum potential are computed from the same coherent framework via the ψ-KDE estimator ψ̂ = j/√n evaluated at candidate and GH probe positions.

The "swarmalator" designation reflects the algorithm's conceptual kinship with the swarmalator literature (O'Keeffe, Hong & Strogatz, 2017), where active agents couple their spatial motion to an internal phase oscillator. Here, the "internal phase" is the quantum action S, the "coupling function" is the coherent kernel average, and the emergent collective behaviour reproduces the Schrödinger equation.

This document parallels the structure of the companion grid-based analysis (*NelsonMechanics_SchrodingerBridge_Algorithm.md*). All theorems, readouts, and backward-channel diagnostics carry over; the difference is purely in how the underlying field estimates are computed.

---

## 1. Motivation: From Grid to Swarm

### 1.1 The Grid Bottleneck in M5

The M5 algorithm (all versions) estimates fields on a spatial grid:

1. **Density:** histogram or CIC deposit of particle positions → Gaussian smooth → ρ(x) on grid
2. **Phase/velocity:** bin particle actions by position → smooth → S(x) on grid → finite-difference ∂ₓS/m
3. **Selection weights:** interpolate √ρ from grid to candidate positions
4. **Quantum potential:** finite differences of √ρ on grid, or GH WEIGH with grid-interpolated √ρ

Each step introduces discretization error. The grid spacing dx limits spatial resolution. Interpolation from grid to particle positions smears sub-grid structure. Finite differences amplify noise. The ψ-KDE variant (NelsonMechanics_SchrodingerBridge_Algorithm.md §7.2 step 1b) improves steps 1–2 by smoothing in ψ-space, but still deposits onto and reads from a grid.

### 1.2 The Swarmalator Alternative

In swarmalator models (O'Keeffe, Hong & Strogatz, 2017), active agents carry an internal phase θᵢ and couple their spatial velocity to the phases of their neighbours:

    ẋᵢ = F(xᵢ, θᵢ, {xⱼ, θⱼ})

The coupling function F determines collective behaviour: synchronisation, clustering, phase waves, and chimera states emerge from different choices of F.

The quantum swarmalator postulate is:

> **The velocity of particle i is determined by the coherent average of its neighbours' phases:**
>
>     vᵢ = (ℏ/m) ∂ₓ arg[Σⱼ Kᵢⱼ e^{iθⱼ}]
>
> where θⱼ = Sⱼ/ℏ is the quantum phase and Kᵢⱼ = K_h(Xᵢ − Xⱼ) is a spatial kernel.

This is a **coherent** average: complex phasors are summed before extracting directional information. A classical swarmalator would use the **incoherent** average v ∝ Σⱼ Kᵢⱼ ∂ₓθⱼ — averaging individual phase gradients. The coherent sum introduces interference: contributions from particles with opposing phases cancel, suppressing both velocity and effective density in regions of phase disorder. This is what makes the coupling quantum rather than classical.

### 1.3 What Changes, What Stays the Same

| Aspect | Grid-based M5 | Quantum Swarmalator |
|--------|---------------|---------------------|
| Per-particle state | (X, S) | (X, S) — identical |
| Velocity source | Grid field ∂ₓS/m interpolated to particles | Direct coherent kernel sum Im(j'/j) |
| Density/√ρ source | Grid KDE, interpolated | Pointwise ψ-KDE kernel sum |
| √ρ selection | Candidates weighted by grid-interpolated √ρ | Candidates weighted by ψ-KDE √ρ |
| Quantum potential | GH WEIGH with grid-interpolated √ρ | GH WEIGH with ψ-KDE √ρ |
| Backward channel | GH log-density with grid-interpolated ln ρ | GH log-density with ψ-KDE ln ρ |
| Spatial grid | Required | Eliminated |
| Bandwidth parameters | σ_kde (grid cells) + σ_gh | h (physical units) + σ_gh |
| Candidate cloud | K GH nodes (shared STEER + WEIGH) | Same — evaluated by ψ-KDE |
| Computational cost | O(Np + Nx log Nx) per step | O(Np · K_nbr) per step |

The dynamical structure — STEER, WEIGH, backward channel, Holland bi-HJ — is identical. The swarmalator is a different *implementation* of the same mathematical algorithm, not a different algorithm.

---

## 2. The ψ-KDE Estimator: Coherent Kernel Density Estimation

### 2.1 Two Kernel Sums

Let K_h(u) = (2πh²)^{-1/2} exp(−u²/2h²) be a Gaussian kernel with bandwidth h (in physical units). For an ensemble {Xⱼ, Sⱼ}, j = 1,...,Np, define:

**Density KDE:**

    n_h(x) = (1/Np) Σⱼ K_h(x − Xⱼ)

This estimates the particle density ρ(x) = |ψ(x)|².

**Coherent average (phase-weighted KDE):**

    j_h(x) = (1/Np) Σⱼ K_h(x − Xⱼ) · exp(iSⱼ/ℏ)

This is the kernel-weighted sum of complex phasors. Each particle's contribution is "coloured" by its phase factor. In the large-sample limit, j_h(x) → ρ(x) · E[e^{iS/ℏ} | X = x].

The terminology "coherent average" is standard in optics and NMR: summing complex amplitudes preserves interference (constructive and destructive), while summing magnitudes (incoherent average) destroys it.

### 2.2 The ψ-KDE Estimator

**Definition:**

    ψ̂(x) = j_h(x) / √n_h(x)

**Theorem (Consistency).** If particles are distributed as ρ = |ψ|² and carry the deterministic phase S(Xⱼ), then as Np → ∞ with h → 0 at the standard KDE rate:

    ψ̂(x) → √ρ(x) · e^{iS(x)/ℏ} = ψ(x)

*Proof.* n_h → ρ and j_h → ρ · e^{iS/ℏ}, so j_h/√n_h → ρ · e^{iS/ℏ} / √ρ = √ρ · e^{iS/ℏ} = ψ. ∎

The division by √n removes one power of √ρ from the particle density (which samples |ψ|², not |ψ|), yielding the wave function amplitude √ρ rather than the probability density ρ.

**Normalization convention.** The definitions in §2.1 include the (1/Np) factor, so n_h → ρ as a proper probability density (∫ρ dx = 1). In the reference implementation `m5_gridless_opt.py`, the 1/Np is omitted from the kernel sums: n = Σⱼ K_h(·) = Np · n_h. Since all quantities used in the algorithm — the velocity v = Im(j'/j), the selection probabilities ∝ √ρ, the WEIGH ratio M₊ — involve ratios where Np cancels, this convention has no effect on the dynamics. However, the absolute values of √ρ from the code are √Np times larger than the theoretical √ρ. If absolute ψ̂ values are needed (e.g. for comparison with an exact wave function), divide by Np: n_h = n/Np, j_h = j/Np.

Unlike the grid-based CIC + `gaussian_filter1d` pipeline (`NelsonMechanics_SchrodingerBridge_Algorithm.md` §7.2 step 1b), where the discrete convolution introduces a √dx scale factor requiring an explicit normalization step every time step, the gridless kernel sums use a properly normalized Gaussian kernel K_h with ∫K_h du = 1. The gridless ψ-KDE converges to ψ directly — no normalization step is needed.

### 2.3 Derived Fields

From ψ̂ all physical fields follow:

    √ρ(x) = |ψ̂(x)| = |j_h(x)| / √n_h(x)

    ρ(x) = |ψ̂(x)|² = |j_h(x)|² / n_h(x)

    S(x) = ℏ · arg(j_h(x))      [phase, up to branch cuts]

### 2.4 Behaviour at Interference Nodes

Near a node where ψ(x₀) = 0, particles from two branches carry opposing phases. Both contribute to n (positive density), but their phasors cancel in j:

    j_h(x₀) → ρ_L e^{iS_L/ℏ} + ρ_R e^{iS_R/ℏ} → 0    (destructive interference)

Therefore |ψ̂(x₀)| = |j(x₀)|/√n(x₀) → 0, correctly producing a node. The density n remains positive (particles are present), but the coherent field j vanishes. This is the fundamental advantage over incoherent estimation.

---

## 3. Velocity from the Coherent Coupling

### 3.1 The Derivative Kernel

Define K'_h(u) = −(u/h²) K_h(u), the spatial derivative of the Gaussian kernel. The derivative coherent sum is:

    j'_h(x) = (1/Np) Σⱼ K'_h(x − Xⱼ) · exp(iSⱼ/ℏ)

### 3.2 Velocity as Im(j'/j)

**Theorem (Swarmalator velocity).** The current velocity v = ∂ₓS/m satisfies:

    v(x) = (ℏ/m) Im(j'_h(x) / j_h(x))

in the limit where the kernel bandwidth h resolves the phase structure.

*Proof.* Write ψ̂ = j/√n. Then:

    ∂ₓ ln ψ̂ = ∂ₓ ln j − ½ ∂ₓ ln n

Since n is real, Im(∂ₓ ln n) = 0. Therefore:

    Im(∂ₓ ln ψ̂) = Im(∂ₓ ln j) = Im(j'/j)

But Im(∂ₓ ln ψ̂) = Im(∂ₓ(ln √ρ + iS/ℏ)) = ∂ₓS/ℏ. Multiplying by ℏ/m gives v. ∎

**Key property:** The density n has dropped out entirely. The velocity depends only on the coherent sum j and its derivative j'. This is why v is well-defined even at nodes — it requires only j'/j, and while j → 0 at a node, j' remains finite (ψ passes through zero with finite slope), so the ratio has a well-defined (though possibly large) limit.

### 3.3 Osmotic Velocity as Re(j'/j)

The same ratio gives the osmotic velocity:

    u(x) = ν Re(j'_h(x) / j_h(x))

where ν = ℏ/(2m). This is because Re(∂ₓ ln ψ̂) = ∂ₓ ln |ψ̂| = ½ ∂ₓ ln ρ = u/ν.

Thus the complex ratio j'/j encodes both components of the Nelson drift:

    b = v + u = (ℏ/m) Im(j'/j) + ν Re(j'/j)

or equivalently:

    b = Re((ℏ/m − iν) · j'/j)

The swarmalator coupling function Im(j'/j) extracts the current velocity; the √ρ-selection mechanism handles the osmotic velocity independently (and more robustly) through the STEER operation.

### 3.4 Physical Interpretation

Each particle senses its local phase environment through the kernel-weighted coherent sum j. Regions where nearby particles have aligned phases produce large |j| — the particle is in a coherent flow and its velocity is well-determined. Regions where phases oppose produce small |j| — the particle is near a node where the velocity concept breaks down. The √ρ selection simultaneously suppresses particles from visiting such regions, providing a natural partnership: the coherent coupling sets the velocity where it's well-defined, and the selection keeps particles away from where it isn't.

This is structurally analogous to the Kuramoto model's mean-field coupling r e^{iΨ} = (1/N) Σⱼ e^{iθⱼ}, where r measures coherence and Ψ gives the mean phase. Here j plays the role of the local order parameter, and |j|/n is the local coherence measure.

### 3.5 Coherence Threshold

When |jᵢ| is very small relative to nᵢ (low local coherence), the ratio j'/j is dominated by noise. A coherence threshold should be applied:

    if |jᵢ| < ε · nᵢ:   set vᵢ = 0

where ε ~ 10⁻⁶ to 10⁻¹⁰ is a relative coherence floor. This is sufficient to suppress velocity in truly incoherent regions (extreme tails, near-exact nodal cancellation) without prematurely zeroing velocity at particles experiencing real phase variation near interference fringes. A value as large as ε ~ 10⁻² would aggressively suppress velocity on the flanks of interference patterns where the coherence dip is physical. Particles below the threshold are in regions where the velocity is physically irrelevant — the √ρ selection will move them away.

---

## 4. √ρ Selection (STEER) — Gridless

### 4.1 The Selection Mechanism

The STEER operation selects among the shared candidate cloud (the same GH nodes or random antithetic candidates used for WEIGH), weighted by √ρ. The only difference from the grid-based M5 is how √ρ is evaluated at the candidate positions.

**Grid-based M5:** Interpolate √ρ from a pre-smoothed grid field.

**Swarmalator:** Evaluate the ψ-KDE estimator directly at each candidate position x':

    n(x') = (1/Np) Σⱼ K_h(x' − Xⱼ)
    j(x') = (1/Np) Σⱼ K_h(x' − Xⱼ) · e^{iSⱼ/ℏ}
    √ρ(x') = |j(x')| / √n(x')

This is a direct particle-to-particle computation with no grid intermediary. The same n and j values also yield ln ρ = 2 ln|j| − ln n, needed for the backward channel (§6), at no additional cost.

### 4.2 Theorem 1 (Osmotic Drift) — Unchanged

The √ρ-weighted selection induces the osmotic drift u = ν ∂ₓ ln ρ in the small-dt limit, regardless of how √ρ is computed. The proof (M5 §3) depends only on the importance-sampling structure of the selection, not on the density estimator.

### 4.3 Coherent √ρ vs Incoherent √ρ

An important distinction: the ψ-KDE computes √ρ = |j|/√n, which can be less than √(density KDE) = √n when phases are partially incoherent. In the large-Np limit with correct phases, |j|/√n → √ρ exactly. But with accumulated phase noise, the coherent estimate "sees" the decoherence and gives a reduced amplitude — a form of self-diagnosis. Particles with very noisy phases contribute less to the coherent field, which is physically appropriate.

---

## 5. Quantum Potential (WEIGH) — Gridless

### 5.1 GH Quadrature with ψ-KDE

The WEIGH readout uses the same candidate cloud as STEER (§4). It extracts Q from the mean-to-departure weight ratio of the √ρ values already computed at the candidate positions:

    M₊ = [Σₖ wₖ · √ρ(xₖ)] / [√ρ(X_class) · Σₖ wₖ]
    Q = −(ℏ²/(m · σ_gh²)) · (M₊ − 1)

With GH nodes, wₖ are the GH quadrature weights. With random candidates, wₖ = 1 (uniform). The √ρ(xₖ) values are already available from step 4 of the algorithm — no additional kernel sums are needed.

### 5.2 Theorem 2 (Quantum Potential) — Unchanged

The mean-weight formula Q · dt = ℏ(1 − M₊) + O(σ_gh²) holds identically. The proof depends on the Gaussian mean-value property applied to √ρ, regardless of how √ρ is computed.

### 5.3 Consistency Advantage

In the grid-based M5, the √ρ used for STEER and WEIGH comes from the same grid field, but interpolation introduces position-dependent error. In the swarmalator, STEER and WEIGH operate on exactly the same √ρ values — computed once at the shared candidate positions in step 4, then used for selection in step 5 and for the mean-weight ratio in step 6. There is no interpolation step and no separate computation: the three readouts (STEER, WEIGH-fwd, WEIGH-bwd) are three views of a single set of ψ-KDE evaluations.

---

## 6. Backward Channel — Gridless

### 6.1 Log-Density Readout

The backward channel requires ln ρ at the candidate positions. Since step 4 of the algorithm computes both n(x) and j(x) at each candidate, ln ρ is immediately available:

    ln ρ(x) = 2 ln|j_h(x)| − ln n_h(x)

No additional kernel sums are required — the backward channel comes free from the same ψ-KDE evaluation used for STEER and WEIGH.

**Design note:** This uses the ψ-KDE density ρ̂ = |ψ̂|² = |j_h|²/n_h, consistent with the grid-based M5 (NelsonMechanics_SchrodingerBridge_Algorithm.md step 1b). An alternative would be ln n_h (the standard density KDE, ignoring phase information), which is smoother when phases are noisy but misses nodal structure. For single-branch states (Gaussians, coherent states), both give the same result. For multi-branch states with nodes, the ψ-KDE density is more physically correct — at nodes, |j| → 0 correctly drives ln ρ → −∞, reflecting the true ρ → 0. With accumulated phase noise, the coherence factor |j/n| < 1 causes systematic underestimation of ρ (§4.3 above); this is a self-diagnostic feature rather than a defect.

The osmotic divergence is:

    u' = ν · 2 · [⟨ln ρ⟩_GH − ln ρ(X_class)] / σ_gh²

And the backward quantum potential:

    Q̃ = Q + ℏu'

### 6.2 Holland bi-HJ — Unchanged

The complete Holland structure carries over without modification:

    Q + Q̃ = −mu²      (osmotic kinetic energy)
    Q − Q̃ = −ℏu'      (inter-congruence coupling)
    σ± = S ± (ℏ/2) ln ρ

The swarmalator simply provides a different (gridless) implementation of the same readouts.

---

## 7. Complete Algorithm

### 7.1 Per-Particle State

Each particle carries (Xₖ, Sₖ) — position and accumulated action phase. No per-particle amplitude, no grid arrays.

### 7.1a Ensemble Initialization

Given ψ₀(x) (analytically or on a grid), the initial ensemble {Xₖ, Sₖ} is constructed as:

1. **Positions** — deterministic CDF quantiles (preferred):

       X_k = CDF⁻¹((k − ½) / Np),   k = 1,...,Np

   where CDF(x) = ∫_{−∞}^x |ψ₀(x')|² dx'. This achieves O(Np⁻²) variance scaling in the ψ-KDE reconstruction, a 3–5× improvement over random CDF-inversion sampling. For stochastic initialization, use systematic sampling (single random offset u₀ ~ U(0, 1/Np), then u_k = u₀ + k/Np).

2. **Phases** — local evaluation at each particle position:

   If ψ₀ is available analytically:

       S_k = ℏ · arg(ψ₀(X_k))

   If ψ₀ is only on a grid, interpolate Re(ψ₀) and Im(ψ₀) separately to X_k, then take arg of the interpolated complex value. Do **not** use `np.unwrap(np.angle(ψ₀))` on the grid — this propagates floating-point noise from low-density tails into the bulk phase. No unwrapping is needed: particles sit where |ψ| is large, and the kernel sums use only cos(S/ℏ) and sin(S/ℏ), which are 2π-periodic.

3. **Optional refinement** — for initial states with interference structure (nodes, dense fringes), L-BFGS-B minimization of ‖ψ̂ − ψ₀‖² starting from CDF quantile positions can remove 40–68% of the remaining ψ-KDE variance. The optimizer naturally finds node-avoiding, curvature-adapted placement. For smooth initial states (single Gaussians, well-separated packets) the improvement is marginal (~15%). For stochastic initialization, **systematic sampling** (single random offset u₀ ~ U(0, 1/Np), then u_k = u₀ + k/Np) provides nearly deterministic-quality results while preserving stochastic character.

### 7.2 Pseudocode

```
INPUT: Ensemble {X_k, S_k}, k = 1,...,Np
       Kernel bandwidth h
       GH parameters: K_gh nodes, σ_gh scale
       Stochastic candidates: K_steer per particle

PARAMETERS: σ_noise = √(ℏ/m),  ν = ℏ/(2m)

═══════════ COHERENT VELOCITY ══════════════════════════

1. COMPUTE VELOCITY via swarmalator coherent coupling

   For each particle k:
     j_k  = (1/Np) Σⱼ K_h(X_k − X_j) · exp(iS_j/ℏ)
     j'_k = (1/Np) Σⱼ K'_h(X_k − X_j) · exp(iS_j/ℏ)

     if |j_k| > ε · n_k:
       v_k = (ℏ/m) · Im(j'_k / j_k)
     else:
       v_k = 0    [incoherent region]

   [With spatial hashing, cost is O(Np · K_nbr) where K_nbr
    is the number of neighbours within ~4h of each particle.]

═══════════ CLASSICAL ADVECTION ════════════════════════

2. CLASSICAL STEP
   X_class_k = X_k + v_k · dt

═══════════ CANDIDATE CLOUD (shared for STEER + WEIGH) ════

3. GENERATE CANDIDATES
   [Preferred: deterministic GH nodes, K = 4–8]
   For j = 1,...,K:
     x_j = X_class_k + √2 · σ_gh · ξ_j    (fixed GH nodes)

   [Alternative: random antithetic candidates, K = 64–128]
   For j = 1,...,K/2:
     ξ_j ~ N(0,1)
     x_{2j-1} = X_class_k + σ_noise · √dt · ξ_j
     x_{2j}   = X_class_k − σ_noise · √dt · ξ_j

4. EVALUATE √ρ AND ln ρ AT CANDIDATES via ψ-KDE
   For each candidate x_j:
     n(x_j)  = (1/Np) Σ_l K_h(x_j − X_l)
     j(x_j)  = (1/Np) Σ_l K_h(x_j − X_l) · exp(iS_l/ℏ)
     √ρ(x_j) = |j(x_j)| / √n(x_j)
     ln ρ(x_j) = 2 ln|j(x_j)| − ln n(x_j)

   Also evaluate at departure point X_class_k:
     √ρ₀ = √ρ(X_class_k),  ln ρ₀ = ln ρ(X_class_k)

═══════════ STEER (position update, 1st-order readout) ════

5. SELECT
   [GH: weight by GH node weight × √ρ]
   p_j = w_j · √ρ(x_j) / Σ w_l · √ρ(x_l)

   [Random: weight by √ρ alone]
   p_j = √ρ(x_j) / Σ √ρ(x_l)

   Draw x'* ~ Categorical({p_j})
   Set X_k ← x'*

═══════════ WEIGH (quantum potential, 2nd-order readout) ═══

6. FORWARD WEIGH
   M₊ = [Σ_j w_j · √ρ(x_j)] / [√ρ₀ · Σ w_j]
   Q_k = −(ℏ²/(m · σ_gh²)) · (M₊ − 1)

═══════════ BACKWARD CHANNEL ══════════════════════════

7. LOG-DENSITY WEIGH
   L_k = [Σ_j w_j · ln ρ(x_j)] / [Σ w_j]
   u'_k = ν · 2 · [L_k − ln ρ₀] / σ_gh²

8. BACKWARD POTENTIAL
    Q̃_k = Q_k + ℏ · u'_k

═══════════ PHASE UPDATE ══════════════════════════════

9. UPDATE ACTION
    S_k ← S_k + [½m v_k² − V(X_k) − Q_k] · dt
    Reduce: S_k ← S_k mod (2πℏ)
```

### 7.3 Comparison with Grid-Based M5

| Step | Grid-based M5 | Swarmalator |
|------|---------------|-------------|
| 1 (velocity) | CIC → smooth → ∂ₓS/m → interpolate | Direct Im(j'/j) at particle |
| 2 (advection) | Same | Same |
| 3 (candidates) | GH nodes or random antithetic | Same — shared for STEER + WEIGH |
| 4 (√ρ at candidates) | Interpolate from grid | ψ-KDE kernel sum at each candidate |
| 5 (STEER select) | Same | Same |
| 6 (WEIGH Q) | Mean weight with grid-interpolated √ρ | Mean weight with ψ-KDE √ρ |
| 7–8 (backward) | GH ln ρ with grid-interpolated values | GH ln ρ with ψ-KDE values |
| 9 (phase) | Same | Same |

---

## 8. Computational Cost

### 8.1 Naïve Cost

The dominant cost is evaluating the ψ-KDE kernel sums. Each evaluation at a single point x requires summing over all source particles: O(Np).

- Step 1 (velocity): 2 kernel sums per particle → O(Np²)
- Step 4 (√ρ + ln ρ at K shared candidates): 2 kernel sums × K candidates × Np particles → O(Np² · K)
- Steps 5–8 (STEER, WEIGH, backward): pointwise arithmetic on already-computed values → O(Np · K)

Total: O(Np² · K) per time step, where K = K_gh (typically 4–8) for the GH variant.

### 8.2 With Compact Kernel Support

For a Gaussian kernel truncated at |u| < 4h, only particles within radius 4h contribute. With spatial hashing (cell size = 4h), each kernel sum costs O(K_nbr) where K_nbr is the average number of neighbours within the kernel radius.

For a 1D distribution with characteristic width σ_dist and Np particles:

    K_nbr ≈ Np · 8h / (xR − xL)

For Np = 3000, h = 0.25, domain width 30: K_nbr ≈ 200. Cost per step: O(Np · K_nbr · K_steer) ≈ O(Np · K_nbr · K) ≈ 10⁷ operations.

### 8.3 Comparison with Grid-Based

| Operation | Grid M5 | Swarmalator |
|-----------|---------|-------------|
| Density estimation | O(Np + Nx log Nx) | — (folded into √ρ evaluation) |
| Velocity | O(Nx) | O(Np · K_nbr) |
| √ρ at K candidates | O(Np · K) [grid interp] | O(Np · K · K_nbr) |
| STEER + WEIGH + backward | O(Np · K) | O(Np · K) [reuses step above] |

The swarmalator is more expensive per step (by a factor ~K_nbr) but eliminates grid artifacts. For Np < ~5000 in 1D, the cost is manageable. In higher dimensions, the swarmalator scales as O(Np · K_nbr) where K_nbr is bounded by the kernel support volume, while grid methods scale as O(Np + Ng^d) where Ng^d grows exponentially. The swarmalator becomes favourable for d ≥ 3.

### 8.4 Hybrid Strategy

A practical compromise: use the swarmalator coherent coupling for velocity (O(Np²), no grid needed), but evaluate √ρ for candidates and GH probes using the grid-based ψ-KDE (CIC + FFT smooth + interpolate). This captures the main benefit — better velocity from direct coherent coupling — while keeping the selection step at grid cost.

---

## 9. Single Bandwidth Parameter

### 9.1 The Bandwidth h

The entire estimation is governed by a single physical bandwidth h. This replaces the grid-based M5's multiple parameters (σ_kde in grid cells, σ_S for phase smoothing, σ_Q for Q smoothing).

The kernel bandwidth h sets:
- The resolution of the density estimate (Δx ~ h)
- The coherence length for the velocity coupling
- The effective neighbourhood radius for each particle's "phase sensing"

### 9.2 Optimal Bandwidth

Standard KDE theory gives h_opt ~ Np^{−1/5} · σ_feature. For Np = 3000 and σ_feature ~ 1:

    h_opt ≈ 3000^{−0.2} ≈ 0.20

The interference fringe spacing πℏ/p₀ sets an upper bound: h must be smaller than this to resolve fringes. For the cat state (p₀ = 3): λ_fringe ≈ 1.05, so h < 0.5 is safe.

### 9.3 Relationship to σ_gh

The GH WEIGH scale σ_gh controls the probe cloud width for the quantum potential readout. It is independent of h and has the same role as in the grid-based M5: σ_gh ~ 0.1–0.3 balances bias (O(σ_gh²)) against sensitivity.

---

## 10. Coherent vs Incoherent: The Fundamental Distinction

### 10.1 Two Ways to Estimate Velocity

**Incoherent (grid M5):** Bin phases Sⱼ by position → average in each bin → smooth → finite-difference gradient:

    v_incoh(x) = (1/m) ∂ₓ [smooth(bin_average(Sⱼ))]

This averages S values first, then differentiates. Near a node, particles from two branches with S differing by ~πℏ are averaged to give a meaningless intermediate value. The velocity estimate is corrupted.

**Coherent (swarmalator):** Sum complex phasors first, then extract the phase gradient:

    v_coh(x) = (ℏ/m) Im(j'/j) = (ℏ/m) ∂ₓ arg(j)

The complex sum j automatically performs destructive interference at nodes. The phase gradient of j is the gradient of the resultant phasor, which correctly reflects the dominant flow direction even when two branches overlap.

### 10.2 The Coherent Average as a Quantum Postulate

From the swarmalator perspective, the distinction between classical and quantum coupling is precisely the distinction between incoherent and coherent averaging:

| | Classical swarmalator | Quantum swarmalator |
|---|---|---|
| Coupling | v ∝ Σⱼ Kᵢⱼ ∂ₓθⱼ | v ∝ ∂ₓ arg[Σⱼ Kᵢⱼ e^{iθⱼ}] |
| Averaging | Incoherent (average gradients) | Coherent (gradient of average phasor) |
| At nodes | Velocity ~ average of opposing flows | Velocity from resultant phasor (may vanish) |
| Interference | Cannot produce nodes | Produces nodes via destructive interference |

The coherent averaging is what makes the coupling quantum. It is a single postulate that simultaneously produces interference patterns, nodal structure, and the quantum velocity field.

### 10.3 Relationship to Wallstrom's Objection

The coherent average j_h automatically enforces integer winding numbers. When two branches interfere destructively, j passes smoothly through zero in the complex plane. On a closed loop around a node, arg(j) advances by exactly 2π — the quantisation condition ∮ ∇S · dl = 2πnℏ is built into the phasor structure of the coherent sum. This is the ψ-KDE's resolution of the Wallstrom objection: the quantisation is not an additional postulate but an automatic consequence of the coherent averaging of discrete phasors. Whether this constitutes a genuine resolution or merely shifts the postulate to the requirement that particles carry coherent phases remains an open philosophical question.

---

## 11. Connection to Active Matter

### 11.1 The te Vrugt et al. Mapping

Te Vrugt et al. (2023) derived a formal mapping between the microscopic Langevin equations of inertial active Brownian particles and the Madelung form of the Schrödinger equation. The correspondence identifies:

- Active particle density ρ ↔ quantum probability density |ψ|²
- Velocity field v ↔ phase gradient ∇S/m
- Gradient penalty terms (−κ∇²ρ + λ(∇ρ)²) ↔ quantum potential

The swarmalator algorithm operationalises this mapping at the particle level. Each active particle:

1. Senses its local density environment (through the kernel sum n)
2. Senses its local phase environment (through the coherent sum j)
3. Moves in response to the gradient of the coherent field (v from Im(j'/j))
4. Is subject to stochastic exploration (Brownian candidates)
5. Is constrained by the local density structure (√ρ selection)
6. Updates its internal phase according to a Hamilton-Jacobi equation

### 11.2 Key Disanalogies

Despite the structural parallel, fundamental differences remain:

1. **Dissipation:** Active matter has friction (γ > 0); quantum mechanics does not. The mapping requires γ → 0 with carefully tuned activity.
2. **Entanglement:** Active matter particles are individually identifiable; quantum particles in higher dimensions have configuration-space wavefunctions that cannot be factored into single-particle contributions.
3. **Bi-HJ structure:** The Holland two-fluid framework (forward/backward congruences) has no classical active-matter analogue. The anti-diffusive sector is intrinsically quantum.
4. **Ontology:** In active matter, ρ is the physical number density of actual particles. In quantum mechanics, ρ = |ψ|² is a probability density. The swarmalator particles are computational samplers, not ontological primitives.

---

## 12. Numerical Validation

### 12.1 Test Cases

The gridless swarmalator has been validated on three standard test cases:

**A. Free Gaussian** (V = 0, p₀ = 1.5): Tests velocity tracking and wave packet spreading. With Np = 500, K_steer = 8, h = 0.40, T = 1.0: L² error = 0.063. The swarmalator velocity v_i ≈ 1.5 in the bulk, with scatter ~±0.2 from finite-Np noise.

**B. Cat state collision** (V = 0, two counter-propagating Gaussians): Tests interference node handling. With Np = 600, K_steer = 8, h = 0.35, T = 1.0: L² error = 0.259. Interference fringes are visible in the density. The velocity correctly shows opposite signs for the two branches with rapid variation near x = 0.

**C. HO ground state** (V = ½ω²x², stationary state): Tests long-time stability. With Np = 500, K_steer = 8, h = 0.30, T = 1.0: L² error = 0.127. The density slowly broadens — the familiar ground-state instability, here limited by cumulative phase errors in S rather than by grid artifacts.

These results are proof-of-concept with very small particle counts. Larger Np (3000–8000) should yield accuracy comparable to or better than the grid-based M5.

### 12.2 Velocity Quality

The swarmalator velocity v = Im(j'/j) shows markedly different error characteristics from the grid-based velocity ∂ₓS/m:

- **Bulk:** Both methods produce velocities close to the exact value. The swarmalator has stochastic scatter ~1/√K_nbr from finite particle count; the grid method has systematic bias from binning and interpolation.
- **Leading edge / tails:** The grid method shows wild oscillations from sparse bins and interpolation artifacts. The swarmalator velocity gracefully degrades (scatter increases as K_nbr decreases), but without systematic bias.
- **Nodes:** The grid method averages opposing S values → meaningless velocity. The swarmalator's |j| → 0 signals incoherence → velocity set to zero → selection handles dynamics.

---

## 13. Higher Dimensions

### 13.1 Natural Generalisation

The swarmalator coupling generalises to d dimensions without modification:

    j_i = Σⱼ K_h(‖Xᵢ − Xⱼ‖) · exp(iSⱼ/ℏ)
    ∇j_i = Σⱼ ∇K_h(‖Xᵢ − Xⱼ‖) · exp(iSⱼ/ℏ)
    v_i = (ℏ/m) Im(∇j_i / j_i)

The gradient of the Gaussian kernel in d dimensions is:

    ∇K_h(r) = −(r/h²) K_h(r) · r̂

where r = Xᵢ − Xⱼ. The kernel sums remain particle-to-particle with no grid.

### 13.2 Dimensional Scaling

Grid-based methods require O(Ng^d) grid points, which becomes prohibitive for d ≥ 3. The swarmalator cost is O(Np · K_nbr) where K_nbr grows as h^d · Np / V (V = domain volume). By choosing h to keep K_nbr ≈ 100–500, the cost grows linearly in Np regardless of dimension.

This makes the swarmalator the natural choice for multi-particle quantum dynamics in configuration space.

---

## 14. Summary

The quantum swarmalator is a gridless reformulation of Method 5 in which:

1. **Velocity** comes from the coherent coupling v = (ℏ/m) Im(j'/j) — a direct particle-to-particle computation using the Gaussian kernel and its derivative, with no grid, no binning, and no finite differences.

2. **Selection weights** √ρ = |j|/√n are evaluated at candidate positions by the ψ-KDE kernel sum, again with no grid intermediary.

3. **Quantum potential** Q comes from GH WEIGH with √ρ evaluated by ψ-KDE at the probe points.

4. **Backward channel** (u', Q̃, Holland bi-HJ) is computed from ln ρ at GH probes, where ρ = |j|²/n from the same kernel sums.

The single bandwidth parameter h controls all smoothing. The coherent averaging naturally produces interference, nodes, and phase quantisation. The physical picture is of active particles embedded in a phase bath, responding to the local coherence direction — a quantum Kuramoto coupling that, when combined with stochastic exploration and √ρ selection, reproduces the Schrödinger equation.

---

## 15. References

### Swarmalator models

- O'Keeffe, K.P., Hong, H. & Strogatz, S.H. (2017). Oscillators that sync and swarm. *Nature Commun.* 8, 1504.
- O'Keeffe, K.P., Evers, J.H.M. & Kolokolnikov, T. (2022). Ring states in swarmalator systems. *Phys. Rev. Res.* 4, 023123.

### Active matter and quantum analogies

- te Vrugt, M., Frohoff-Hülsmann, T., Heifetz, E., Thiele, U. & Wittkowski, R. (2023). From a microscopic inertial active matter model to the Schrödinger equation. *Nature Commun.* 14, 1302.
- Heifetz, E. & Cohen, E. (2015). Toward a thermo-hydrodynamic like description of Schrödinger equation via the Madelung formulation and Fisher information. *Found. Phys.* 45, 1514–1525.

### Quantum mechanics and stochastic mechanics

- Nelson, E. (1966). Derivation of the Schrödinger Equation from Newtonian Mechanics. *Phys. Rev.* 150, 1079.
- Nelson, E. (2012). Review of stochastic mechanics. *J. Phys.: Conf. Ser.* 361, 012011.
- Holland, P. (2021). Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory. arXiv:2111.09235.
- Wallstrom, T.C. (1994). Inequivalence between the Schrödinger equation and the Madelung hydrodynamic equations. *Phys. Rev. A* 49, 1613.

### Schrödinger bridges and optimal transport

- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). Stochastic Control Liaisons. *SIAM Review* 63, 249–313.
- Conforti, G. & Pavon, M. (2017). Extremal flows on Wasserstein space. arXiv:1712.02257.

### Kernel density estimation and SPH

- Nadaraya, E.A. (1964). On estimating regression. *Theory Prob. Appl.* 9, 141–142.
- Watson, G.S. (1964). Smooth regression analysis. *Sankhyā A* 26, 359–372.
- Monaghan, J.J. (2005). Smoothed particle hydrodynamics. *Rep. Prog. Phys.* 68, 1703.
- Abramson, I.S. (1982). On bandwidth variation in kernel estimates. *Ann. Statist.* 10, 1217–1223.

### Quantum trajectory methods

- Wyatt, R.E. (2005). *Quantum Dynamics with Trajectories.* Springer.
- Hackebill, A. & Poirier, B. (2026). On Hydrodynamic Formulations of Quantum Mechanics and the Problem of Sparse Ontology. arXiv:2602.21106.

---

### Project Knowledge Cross-References

- `NelsonMechanics_SchrodingerBridge_Algorithm.md` — Grid-based M5: core algorithm, theorems, GH quadrature, backward channel
- `NelsonMechanics_SchrodingerBridge_Supplement.md` — Contextual analysis: Hackebill–Poirier, Wasserstein dynamics, bridge interpretation
- `NelsonMechanics_SchrodingerBridge_Algorithm.md` §7.2 step 1b — grid-based ψ-KDE implementation (CIC + Gaussian convolution)
- `Method5_QA_Discussion.md` — Time symmetry, local Sinkhorn, Fisher information
- `project_summary.md` — te Vrugt et al. active matter analysis
- `complex_trajectories_analysis.md` — Yang & Han complex trajectory connection
- `m5_gridless_opt.py` — Reference implementation of the gridless swarmalator algorithm
