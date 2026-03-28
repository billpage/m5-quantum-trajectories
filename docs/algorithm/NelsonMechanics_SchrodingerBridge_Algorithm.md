# Stochastic Mechanics and the Schrödinger Bridge: A Derivative-Free Particle Algorithm for Quantum Dynamics

## Complete Mathematical Analysis

---

### Abstract

We present a derivative-free particle algorithm for quantum dynamics grounded in Nelson's stochastic mechanics and the Schrödinger bridge FBSDE framework. The algorithm (Method 5, updated formulation) evolves a finite ensemble of particles, each carrying a position X and an accumulated action phase S, using three readouts from a shared Gauss–Hermite (GH) candidate cloud to implement the complete Holland bi-Hamilton–Jacobi (bi-HJ) structure without any spatial differentiation of the density. Four key advances over the original formulation are incorporated: (i) **fixed Gauss–Hermite (GH) quadrature nodes** replace random Brownian candidates, converting both the STEER and WEIGH operations into deterministic quadrature with controlled polynomial error; (ii) a **log-density mean weight** extracts the osmotic velocity divergence u′ = ∂ₓu with machine-precision accuracy for Gaussian-like densities, preserving the polynomial structure that GH quadrature exploits; (iii) **mirror particles** correct the KDE boundary bias at distribution tails, improving the forward quantum potential Q estimate by 2–4×; and (iv) **backward-channel readouts** provide the anti-diffusive quantum potential Q̃ and Holland's full bi-HJ coupling as derivative-free diagnostic and monitoring quantities, all from the same forward-evolving ensemble.

The central insight is that Holland's backward Schrödinger potential — inaccessible through direct simulation of the ill-posed backward heat equation — is encoded in the curvature of ln ρ, which GH quadrature evaluates exactly when ln ρ is polynomial. The algorithm thus implements the complete bi-HJ structure (two coupled Hamilton–Jacobi equations for the forward and backward action functions σ±) using a single particle ensemble, with no second species, no backward simulation, and no spatial differentiation of the density. A companion supplement (*Stochastic Mechanics and the Schrödinger Bridge: Contextual Analysis, Bridge Interpretation, and Open Questions*) provides the Hackebill–Poirier classification, Wasserstein dynamics, and related literature.

---

## 1. Background: The Nelson–Holland Drift Decomposition

In Nelson's stochastic mechanics, a quantum particle at position X follows the forward SDE:

    dX = b(X, t) dt + σ dW

where σ = √(ℏ/m) and dW is standard Wiener noise. The drift decomposes as b = v + u, where v = ∂ₓS/m is the current (de Broglie–Bohm) velocity and u = ν ∂ₓ ln ρ is the osmotic velocity (ν = ℏ/(2m)).

Holland (2021) reformulates the same physics using two coupled action functions σ± = S ± R, where R = (ℏ/2) ln ρ, giving two Hamilton–Jacobi equations:

    ∂ₜσ± + (∂ₓσ±)²/(2m) + Q± + V = 0

with quantum potentials Q± that couple the two equations. The velocity fields are v± = (1/m)∂ₓσ± = v ± u, and the single density ρ = exp[(σ₊ − σ₋)/ℏ] satisfies both a forward Fokker–Planck equation (diffusion, drift v₊) and a backward Kolmogorov equation (anti-diffusion, drift v₋).

The central computational challenge of this framework is threefold: (a) the osmotic velocity u requires differentiating ln ρ, which is catastrophically noisy where ρ → 0; (b) the quantum potential Q requires the second derivative of √ρ; and (c) the backward/anti-diffusive sector involves Q̃ = −(ℏ²/2m)(f″/f) with f = 1/√ρ, which grows exponentially in the tails.

Method 5 v2 (updated) eliminates all three problems. Items (a) and (b) were solved in the original v2 by the dual-readout mechanism; item (c) is solved here by the log-density approach.

---

## 2. The √ρ-Selection Theorem (Review)

**Theorem 1 (Osmotic drift from √ρ-selection).** Let ρ(x) > 0 be smooth. Generate K candidates x′ₖ = x₀ + σ√dt · ξₖ with ξₖ ∼ N(0,1), and select one candidate with probability proportional to w_k = √ρ(x′ₖ). Then the expected displacement of the selected candidate satisfies:

    E[x′* − x₀] = ν ∂ₓ ln ρ(x₀) · dt + O(dt³/²)

where ν = σ²/2 = ℏ/(2m).

*Proof sketch.* The selection-weighted transition kernel is p_eff(x′|x₀) ∝ G(x′; x₀, σ²dt) · √ρ(x′). The mean of this tilted Gaussian acquires a shift σ²dt · ∂ₓ ln √ρ = ν ∂ₓ ln ρ · dt in the small-dt limit. ∎

This is the STEER readout: √ρ-weighted selection replaces explicit differentiation of ln ρ for the osmotic drift.

### 2.1 Anti-Diffusion by Min-Selection: The Dual Drift Decomposition

**Corollary 1 (Anti-diffusive drift from min-selection).** With the same candidate cloud, select the candidate with the *lowest* √ρ weight (min-selection). The expected displacement satisfies:

    E[x′*_min − x₀] = −ν ∂ₓ ln ρ(x₀) · dt + O(dt³/²) = −u · dt + O(dt³/²)

That is, min-selection implements the anti-diffusive (backward osmotic) drift −u without any density differentiation.

*Proof sketch.* Min-selection is equivalent to importance sampling with weight 1/√ρ. The tilted kernel G(x′; x₀, σ²dt) · (1/√ρ(x′)) has mean shift σ²dt · ∂ₓ ln(1/√ρ) = −ν ∂ₓ ln ρ · dt. ∎

This gives an elegant operational decomposition of the full Bohm–Nelson drift without any density differentiation:

| Operation | Weight | Induced drift | Physical meaning |
|---|---|---|---|
| Max-selection (highest √ρ) | √ρ | +u (forward osmotic) | Nelson forward SDE drift |
| Min-selection (lowest √ρ) | 1/√ρ | −u (backward osmotic) | Anti-diffusive drift |
| Proportional selection | √ρ (stochastic) | +u (in expectation) | STEER — used in M5 dynamics |
| ½(max + min) positions | — | v (current velocity) | Current velocity, no osmotic component |
| ½(max − min) positions | — | u (osmotic velocity) | Osmotic velocity, isolated |

The average of the max-selected and min-selected positions gives the pure current (de Broglie–Bohm) velocity v; their signed difference gives 2u. This selection-based decomposition separates the current and osmotic components of the quantum velocity field purely through position operations on a candidate cloud, with no differentiation at any stage.

Note that M5 uses proportional (stochastic) √ρ-selection for the dynamics, not max-selection, because proportional selection preserves the correct statistical weighting of trajectories. Max/min selection are most useful as diagnostic tools or as motivation for understanding the osmotic drift mechanism.

---

## 3. The Mean-Weight Quantum Potential (Review)

**Theorem 2 (Quantum potential from mean weight).** With the same candidate cloud as Theorem 1, define the departure weight w₀ = √ρ(x_class) and the mean candidate weight ⟨w⟩ = (1/K)Σₖ wₖ. Then:

    Q(x₀) · dt = ℏ(1 − ⟨w⟩/w₀) + O(dt²)

*Proof sketch.* Write R(x) = √ρ(x). The Gaussian mean-value property gives E[R(x₀ + η)] = R(x₀) + ½σ²dt · R″(x₀) + O(dt²), so ⟨w⟩/w₀ ≈ 1 + ½σ²dt · R″/R. Since Q = −(ℏ²/2m)(R″/R) and σ² = ℏ/m, the result follows. ∎

This is the WEIGH readout: the mean-to-departure weight ratio extracts the second logarithmic derivative of √ρ — the quantum potential — without any spatial differentiation.

---

## 4. From Random Candidates to Fixed GH Quadrature

### 4.1 Motivation: Why Move from Random to Deterministic Candidates

The original v2 formulation (and M5v3, which it supersedes) generates K random Brownian candidates per particle per time step, then reads out both STEER and WEIGH from their √ρ weights:

    x′ₖ = x₀ + σ√dt · ξₖ,    ξₖ ∼ N(0,1),    k = 1,...,K    [random approach]

This is mathematically valid and produces correct statistics in the large-K limit. With antithetic pairing (including both +ξ and −ξ for each draw), the Q estimator variance is reduced by 10–50×, and K ~ 64–128 is sufficient for reasonable accuracy in smooth cases.

However, the random approach has a fundamental limitation: it introduces irreducible **Monte Carlo noise** into the quantum potential estimate that accumulates as a random walk in the phase variable S over long simulations (σ_S = |u|√(T/K), see §9.1). This is not a fixable problem with more antithetic pairs — it is inherent to stochastic sampling of an integral that could be evaluated deterministically.

The key observation is that the WEIGH readout is a **Gaussian integral**, not a sampling problem. The Gaussian mean-value property:

    E_η[f(x₀ + η)] = f(x₀) + ½σ² f″(x₀) + (1/24)σ⁴ f⁽⁴⁾(x₀) + ...

is a deterministic Taylor-series identity. For any smooth weight function f(x), this integral can be evaluated exactly (to polynomial order 2K−1) by **Gauss–Hermite (GH) quadrature** with K fixed nodes and weights. GH quadrature replaces K ~ 64–128 random draws with K = 4–8 deterministic probe points, eliminating the Monte Carlo noise entirely.

The random-candidate approach remains a valid alternative when:
- Implementing a quick prototype before committing to the GH infrastructure
- Running in high dimensions where tensor-product GH grids are impractical (see §13)
- The particle count Np is so large that per-particle KDE evaluation dominates over candidate generation cost

For all standard 1D and low-dimensional cases, **GH quadrature is strictly superior** and should be used in preference to random candidates.

### 4.2 GH Quadrature Nodes and Weights

Using the physicist's Hermite polynomials, the K nodes ξ₁, ..., ξ_K and weights w₁, ..., w_K satisfy:

    Σₖ wₖ g(ξₖ) = ∫ g(t) exp(−t²) dt    (exact for deg(g) ≤ 2K−1)

The probe points at particle position x₀ with GH scale parameter σ_gh are:

    xₖ = x₀ + √2 · σ_gh · ξₖ

These are the same at every time step for a given particle position — deterministic, symmetric about x₀, and optimally placed for polynomial approximation.

### 4.3 The Forward Mean Weight M₊

    M₊(x₀) = [Σₖ wₖ · √ρ(xₖ)] / [√ρ(x₀) · Σₖ wₖ]

This approximates 1 + σ_gh² · (√ρ)″/√ρ. The quantum potential is:

    Q(x₀) = −(ℏ²/(m · σ_gh²)) · (M₊ − 1) + O(σ_gh²)

The O(σ_gh²) bias comes from the fourth and higher derivatives of √ρ. For K = 8, this is exact through degree 15, making the bias negligible for smooth densities.

### 4.4 STEER with GH Candidates

The STEER operation (position update) still requires selecting among candidates proportional to √ρ. With fixed GH nodes, the selection is performed over the K deterministic probe points rather than K random draws:

    pₖ = wₖ · √ρ(xₖ) / Σⱼ wⱼ · √ρ(xⱼ)

One candidate x′* is drawn from this categorical distribution. The expected displacement still satisfies Theorem 1, because the GH quadrature reproduces the Gaussian integral that underlies the osmotic drift.

### 4.5 Antithetic Structure Is Built In

With random candidates, antithetic pairing (including both +ξ and −ξ for each draw) was a crucial variance-reduction technique, cancelling odd-moment noise and improving Q estimates by 10–50×. GH quadrature nodes are intrinsically symmetric about the origin (ξₖ and −ξₖ are both nodes), so the antithetic cancellation is automatic and exact — another advantage of the deterministic approach.

### 4.6 Choice of σ_gh and K

The GH scale σ_gh controls the width of the probe cloud. It serves a role analogous to the Brownian noise scale σ√dt in the original formulation, but is now a free parameter (not tied to dt):

- **Bias**: O(σ_gh²) from truncation of the Taylor series. Smaller σ_gh reduces bias.
- **Sensitivity**: The signal (M₊ − 1) scales as σ_gh², so smaller σ_gh means a smaller signal-to-noise ratio when using approximate (KDE) density.
- **Optimal range**: σ_gh ∼ 0.1–0.3 for ℏ = m = 1 balances bias against KDE noise.

K = 4–8 GH nodes suffice for smooth densities. The quadrature is exact for polynomials of degree ≤ 2K−1, so K = 8 handles 15th-degree corrections — far beyond what physical densities typically require.

---

## 5. The Log-Density Mean Weight: Accessing the Backward Channel

### 5.1 Holland's Backward Quantum Potential

Define the anti-diffusive amplitude f(x) = 1/√ρ(x) = exp(−R/ℏ). Its second logarithmic derivative is:

    f″/f = (R′/ℏ)² − R″/ℏ = u²/ν² − u′/ν

The sign of the R″ term is **flipped** relative to the forward case (g″/g = u²/ν² + u′/ν). The backward quantum potential is:

    Q̃ = −(ℏ²/2m)(f″/f) = −½mu² + (ℏ/2)u′

The forward and backward potentials decompose as:

    Q + Q̃ = −mu²      (osmotic kinetic energy, always ≤ 0)
    Q − Q̃ = −ℏu′      (osmotic pressure / velocity divergence)

This is the Holland bi-HJ coupling: the sum gives the osmotic kinetic energy (common to both congruences), and the difference gives the inter-congruence coupling term.

### 5.2 Why Direct 1/√ρ Evaluation Fails

In principle, one could compute a backward GH mean weight M₋ using 1/√ρ as the weight function. However, 1/√ρ = exp(+x²/(4σ_ρ²)) for a Gaussian density — it grows **exponentially** in the tails. This converts the well-behaved quadratic structure of ln ρ into an exponentially growing function, destroying the polynomial structure that GH quadrature exploits.

Numerical tests on the HO ground state (ℏ = m = ω = 1) with exact ρ confirm:

| σ_gh | max Q error (√ρ) | max Q̃ error (1/√ρ) | Ratio |
|------|---|---|---|
| 0.05 | 2.5 × 10⁻² | 7.1 × 10⁻² | 2.9 |
| 0.10 | 1.0 × 10⁻¹ | 2.9 × 10⁻¹ | 2.9 |
| 0.30 | 9.1 × 10⁻¹ | 3.6 × 10⁰ | 4.0 |
| 0.50 | 2.6 × 10⁰ | 2.5 × 10¹ | 9.8 |
| 0.70 | 4.9 × 10⁰ | 5.5 × 10² | 112 |

The backward error grows exponentially with σ_gh while the forward error grows polynomially. This is the backward heat equation's ill-posedness manifesting directly in the GH quadrature.

### 5.3 The Log-Density Solution

**Theorem 3 (Osmotic divergence from log-density mean weight).** Let ρ(x) > 0 be smooth. Define the GH mean of ln ρ at probe points:

    L(x₀) = [Σₖ wₖ · ln ρ(x₀ + √2 σ_gh ξₖ)] / [Σₖ wₖ]

Then:

    u′(x₀) = ν · 2 · [L(x₀) − ln ρ(x₀)] / σ_gh² + O(σ_gh²)

*Proof.* Taylor-expand ln ρ(x₀ + η) = ln ρ(x₀) + η(ln ρ)′ + ½η²(ln ρ)″ + ... . Under GH quadrature the odd moments vanish and E[η²] = σ_gh², giving L(x₀) = ln ρ(x₀) + ½σ_gh²(ln ρ)″ + O(σ_gh⁴). Since (ln ρ)″ = (2/ν)u′, the result follows. ∎

### 5.4 Machine-Precision Exactness for Gaussian Densities

For the HO ground state, ln ρ = −x² + const — a **quadratic polynomial**. GH quadrature with K points is exact for polynomials of degree ≤ 2K−1. With K = 8, it handles degree 15. Therefore the log-density mean is computed **exactly**:

    (ln ρ)″ = −2,    hence    u′ = ν · (−2) = −1

Numerically confirmed: max |u′ error| = 3.8 × 10⁻¹⁴ across the full spatial domain — **fifteen orders of magnitude** better than the direct 1/√ρ approach at the same σ_gh.

This is the key insight: GH quadrature works by exploiting polynomial structure. The log-density preserves this structure (ln ρ is quadratic for Gaussians). The 1/√ρ weight destroys it by exponentiating a polynomial into a non-polynomial function.

### 5.5 The Deeper Principle: Work in the Natural Variables

The Holland bi-HJ coupling is naturally expressed through ln ρ, since σ₊ − σ₋ = ℏ ln ρ. The anti-diffusion information is fundamentally about the **curvature of ln ρ**, which is ∂²ₓ(ln ρ) = (2/ν)u′. This curvature is exactly what the log-density GH mean evaluates.

The general principle: action-like variables (S, ln ρ, σ±) are the natural representation for GH-based readouts. Amplitude-like variables (√ρ) work for the forward channel but fail for the backward channel. Log variables preserve polynomial structure in both directions.

### 5.6 Strategy D: The Combined Approach (Recommended)

Strategy D combines the best of both readouts:

1. **Q from √ρ GH mean weight** — well-conditioned, O(σ_gh²) polynomial error
2. **u′ from ln ρ GH mean weight** — machine-precision for Gaussian-like densities
3. **Q̃ = Q + ℏu′** — reconstructed backward potential, error dominated by Q alone

The backward quantum potential Q̃ inherits exactly the same error as Q — **no additional degradation** from the anti-diffusive sector:

| σ_gh | max Q̃ error (direct 1/√ρ) | max Q̃ error (Strategy D) |
|------|---|---|
| 0.05 | 7.1 × 10⁻² | 2.5 × 10⁻² |
| 0.30 | 3.6 × 10⁰ | 9.1 × 10⁻¹ |
| 0.70 | 5.5 × 10² | 4.9 × 10⁰ |

The exponential blow-up is completely eliminated.

---

## 6. Mirror Particles for KDE Boundary Correction

### 6.1 The Edge Bias Problem

Standard Gaussian KDE underestimates density near the boundary of the particle distribution. The kernel extends beyond the outermost particles into empty space, causing systematic downward bias at the tails. This bias is particularly harmful for:

- The √ρ forward weight (underestimated density → incorrect Q)
- The ln ρ backward weight (underestimated density → ln ρ too negative → biased u′)
- The 1/√ρ weight (amplifies any density underestimate into huge overestimate of anti-diffusive terms)

### 6.2 The Mirror Particle Method

Augment the particle ensemble with reflected copies of the outermost particles:

1. Sort the Np particles.
2. Select the n_mirror = ⌊f_mirror · Np⌋ leftmost and rightmost particles.
3. Reflect the leftmost about the distribution minimum: x_mirror = 2·x_min − x_original.
4. Reflect the rightmost about the distribution maximum: x_mirror = 2·x_max − x_original.
5. Include mirror particles in the KDE evaluation but normalise by the original Np.

The mirror particles provide phantom density beyond the boundary, counteracting the kernel truncation bias. This is a standard technique in the KDE literature (boundary correction by reflection).

### 6.3 Numerical Impact on the HO Ground State

With Np = 2000, σ_kde = 0.25:

| Configuration | Q bulk max error | Q tail max error |
|---|---|---|
| No mirror | 1.30 | 1.30 |
| Mirror 10% | 0.49 | 0.33 |
| Mirror 15% | 0.49 | 0.33 |

Mirror particles reduce the forward Q error by 2.6× in the bulk and 4× in the tails. The improvement saturates at 10–15% mirror fraction — no benefit from mirroring more particles, since the correction only affects the outermost kernel widths.

### 6.4 Mirror Particles with the ψ-KDE

The mirror particle technique integrates naturally with the ψ-KDE density estimator (§7.2 step 1b; theory: NelsonMechanics_SchrodingerBridge_Swarmalator.md §§2–3). When depositing particles onto the CIC grid, the mirror particles are included in both the density (n) and complex-current (j) depositions, with appropriate phases. Since mirror particles lie outside the physical distribution boundary, their phase can be set to match the nearest real particle — the key requirement is that they contribute correctly to the density normalization.

---

## 7. The Complete M5v2 (Updated) Algorithm

### 7.1 State Per Particle

Each particle k carries: position X_k, accumulated phase S_k.

The Holland action functions are recoverable at any time:

    σ₊_k = S_k + (ℏ/2) ln ρ(X_k)     (forward action)
    σ₋_k = S_k − (ℏ/2) ln ρ(X_k)     (backward action)

### 7.1a Ensemble Initialization

Given ψ₀(x) on a grid, the initial particle ensemble {X_k, S_k} is constructed as:

1. **Positions** — deterministic CDF quantiles (preferred):

       X_k = CDF⁻¹((k − ½) / Np),   k = 1,...,Np

   where CDF(x) = ∫_{−∞}^x |ψ₀(x')|² dx'. This reduces the ψ-KDE reconstruction variance by 3–5× relative to random CDF-inversion sampling, achieving O(Np⁻²) scaling. For stochastic initialization, use systematic sampling (single random offset, equispaced in CDF).

2. **Phases** — local evaluation (avoids grid-unwrap bug):

       psi_at_k = interp(X_k, Re(ψ₀)) + i · interp(X_k, Im(ψ₀))
       S_k = ℏ · arg(psi_at_k)

   Do **not** use `np.unwrap(np.angle(ψ₀))` on the grid — this propagates floating-point noise from low-density tails into the bulk phase. No unwrapping is needed: particles sit where |ψ| is large, and the CIC deposit uses only cos(S/ℏ) and sin(S/ℏ), which are 2π-periodic.

3. **Optional refinement** — for initial states with interference structure (nodes, dense fringes), L-BFGS-B minimization of ‖ψ̂ − ψ₀‖² starting from CDF quantile positions can remove 40–68% of the remaining ψ-KDE variance. The optimizer naturally finds node-avoiding, curvature-adapted placement. For smooth initial states (single Gaussians, well-separated packets) the improvement is marginal (~15%). For stochastic initialization, **systematic sampling** (single random offset u₀ ~ U(0, 1/Np), then u_k = u₀ + k/Np) provides nearly deterministic-quality results while preserving stochastic character.

### 7.2 Per-Time-Step Algorithm

```
INPUT: Ensemble {X_k, S_k}, k = 1,...,Np

─── DENSITY ESTIMATION ───────────────────────────────────────

1a. MIRROR PARTICLES
    Sort particles. Reflect outermost 10–15% about distribution
    boundary to create augmented ensemble for KDE.

1b. ψ-KDE DENSITY ESTIMATION  (theory derivation: NelsonMechanics_SchrodingerBridge_Swarmalator.md §§2–3)

    CIC deposit: For each particle k, deposit to two nearest grid points
      with linear interpolation weights (1−α) and α, where
      k_L = floor((X_k − x_L)/dx), α = (X_k − x_{k_L})/dx:
        n_raw[k_L]   += (1 − α)
        n_raw[k_L+1] += α
        j_re[k_L]    += (1 − α) · cos(S_k/ℏ)
        j_re[k_L+1]  += α · cos(S_k/ℏ)
        j_im[k_L]    += (1 − α) · sin(S_k/ℏ)
        j_im[k_L+1]  += α · sin(S_k/ℏ)

    Gaussian convolution (single bandwidth σ_kde):
        n_smooth = gaussian_filter1d(n_raw, σ_kde) / Np
        j_smooth = gaussian_filter1d(j_re + i·j_im, σ_kde) / Np

    Form ψ̂ and normalise:
        ψ̂ = j_smooth / √max(n_smooth, ε)        [ε ~ 10⁻³⁰]
        ψ̂ /= √(Σ|ψ̂|² · dx)                     [required: absorbs √dx factor]

    Extract fields:
        √ρ = |ψ̂|
        v(x) = (ℏ/m) Im(ψ̂* ∂ₓψ̂) / max(|ψ̂|², ε_ψ)    [ε_ψ ~ 10⁻¹⁰]
        Q(x) = −(ℏ²/2m) ∂²ₓ|ψ̂| / max(|ψ̂|, ε_ψ)       [light smooth σ_Q ~ 1.5]
        ln ρ(x) = 2 ln |ψ̂(x)|

    Normalization note: dividing by Np (not Np·dx) gives counts per bin,
    not proper density. The ψ̂ /= ‖ψ̂‖ step compensates. All derived fields
    (v, Q, √ρ for selection) are scale-invariant ratios, unaffected.

─── FORWARD CHANNEL: STEER AND WEIGH ────────────────────────

2. CURRENT VELOCITY
   v_k = v(X_k) interpolated from grid

3. CLASSICAL STEP
   X_class_k = X_k + v_k · dt
   [Velocity is set entirely by the phase gradient v_k = ∂ₓS_k / m.
    No classical force term −(∂ₓV/m)dt is added here: the potential V
    acts solely through the phase update (step 11). Adding it here would
    double-count the classical force.]

4. GH PROBE POINTS  [preferred: deterministic, K = 4–8]
   For j = 1,...,K:
     x_j = X_class_k + √2 · σ_gh · ξ_j     (fixed GH nodes, see §4.2)
   Evaluate: √ρ(x_j) and ln ρ(x_j) by grid interpolation

   [Alternative: random antithetic candidates, K = 64–128]
   For j = 1,...,K/2:
     ξ_j ~ N(0,1)
     x_{2j-1} = X_class_k + σ√dt · ξ_j
     x_{2j}   = X_class_k − σ√dt · ξ_j     (antithetic pair)
   The GH variant eliminates Monte Carlo phase noise entirely (see §4.1, §9.1–9.3)
   and is preferred in all 1D/low-dimensional cases.

5. STEER (position update)
   w_j = √ρ(x_j)
   p_j = w_j / Σ wⱼ      [GH-weighted or uniform-over-candidates categorical]
   Draw x′* ~ Categorical(p₁,...,p_K)
   Set X_k ← x′*

   [Note: with GH nodes, pⱼ uses the GH node weights wⱼ·√ρ(xⱼ)/Σwⱼ√ρ(xⱼ).
    With random antithetic candidates, pⱼ = √ρ(xⱼ)/Σ√ρ(xⱼ) uniformly.]

6. FORWARD WEIGH (quantum potential Q)
   w₀ = √ρ(X_class_k)
   M₊ = [Σ_j GH_w_j · √ρ(x_j)] / [√ρ(X_class_k) · Σ GH_w_j]
   Q_k = −(ℏ²/(m · σ_gh²)) · (M₊ − 1)

─── BACKWARD CHANNEL: LOG-DENSITY READOUT ───────────────────

7. LOG-DENSITY WEIGH (osmotic divergence u′)
   L_k = [Σ_j GH_w_j · ln ρ(x_j)] / [Σ GH_w_j]
   u′_k = ν · 2 · [L_k − ln ρ(X_class_k)] / σ_gh²

8. BACKWARD POTENTIAL (reconstructed)
   Q̃_k = Q_k + ℏ · u′_k

─── DIAGNOSTICS / MONITORING ────────────────────────────────

9. SELF-CONSISTENCY CHECK (optional, per particle)
   u²_k = −(Q_k + Q̃_k) / m         (should be ≥ 0)
   u′_from_Q = −2(Q_k + ½m·u²_k) / ℏ   (should agree with u′_k)
   Flag particle if |u²_k| < 0 or |u′_k − u′_from_Q| > threshold

10. HOLLAND COUPLING (optional, for diagnostics or future use)
    Q₊_k = Q_k + (ℏ/2) v′_k − ½m u²_k
    Q₋_k = Q̃_k − (ℏ/2) v′_k − ½m u²_k
    where v′_k estimated from phase gradient differencing

─── PHASE UPDATE ────────────────────────────────────────────

11. UPDATE ACTION
    S_k ← S_k + [½m v_k² − V(X_k)] · dt − Q_k · dt
    Reduce: S_k ← S_k mod (2πℏ)
```

### 7.3 What Changed From the Original v2 and v3

This document supersedes both the original v2 (`Method5_Mathematical_Analysis_v2.md`) and v3 (`Method5_Mathematical_Analysis_v3.md`). The original v2 established the dual-readout framework with random candidates. v3 added the log-density backward channel and mirror particles, but retained random candidates in its pseudocode. This updated v2 consolidates all advances and upgrades to deterministic GH quadrature.

| Aspect | Original v2 / v3 | Updated v2 (this document) |
|---|---|---|
| Candidates | K random Brownian draws per step (K ~ 64–128) | K fixed GH nodes, deterministic (K = 4–8) |
| WEIGH noise | Monte Carlo O(1/√K) per step, accumulates in S | GH quadrature O(σ_gh²) bias, does not accumulate |
| Backward channel | v2: not available; v3: u′ from ln ρ, Q̃ = Q + ℏu′ | Same as v3, fully integrated |
| Boundary correction | v2: none; v3: mirror particles | Mirror particles (10–15%), fully integrated |
| Self-consistency | Not available | u² ≥ 0 check, cross-validation of u′ |
| K requirement | K ~ 64–128 (noise-limited) | K = 4–8 (quadrature-limited) |
| Antithetic | Explicit ±ξ pairing needed | Automatic (GH nodes symmetric) |
| Min-selection | Not documented | §2.1: anti-diffusion by min-selection, drift decomposition |
| Random fallback | Primary method | Valid alternative for prototyping / high-d (see §4.1, §13) |

---

## 8. Holland's bi-HJ Structure: What v2 Already Solves

### 8.1 The (S, ρ) Representation

A crucial observation: M5v2 already **implicitly solves** Holland's full bi-HJ system. The state per particle is (X_k, S_k), and the density ρ is estimated from the ensemble via KDE. Since:

    σ₊ = S + (ℏ/2) ln ρ,    σ₋ = S − (ℏ/2) ln ρ

both action functions are determined at every time step. Holland's backward action σ₋ is **slaved** to σ₊ and ρ through the algebraic identity — it does not require independent evolution.

### 8.2 Why No Second Ensemble Is Needed

Holland's two congruences (families of trajectories with velocities v₊ and v₋) describe the same quantum state from two perspectives. The density ρ is shared. The forward and backward actions are linked by ln ρ. A single forward-evolving ensemble carrying (X, S) provides all the information needed to reconstruct both σ₊ and σ₋.

The backward channel in the updated v2 makes this implicit structure explicit: the log-density readout provides u′, which is the coupling term between the two congruences in Holland's framework.

### 8.3 The Three Readout Table

| Readout | Weight function | Operation | Derivative order | Physical quantity | Target equation |
|---|---|---|---|---|---|
| STEER | √ρ | Selection (rank) | 1st (∂ₓR/R) | Osmotic drift u | Nelson forward SDE |
| WEIGH-fwd | √ρ | GH mean (average) | 2nd (∂²ₓR/R) | Forward quantum potential Q | Phase HJB (Madelung) |
| WEIGH-bwd | ln ρ | GH mean (average) | 2nd (∂²ₓ ln ρ) | Osmotic divergence u′ | Holland bi-HJ coupling |
| RECONSTRUCT | Q + ℏu′ | Algebra | — | Backward potential Q̃ | Backward HJB |

Each new weight function applied to the same GH nodes opens a new information channel about the density landscape. The √ρ channel gives the mixture (u², u′); the ln ρ channel gives u′ alone. Together they separate the two physical components cleanly.

---

## 9. Variance Analysis

### 9.1 Forward Mean Weight Variance (Updated for GH)

With random candidates, the √ρ mean-weight Q estimator had variance:

    Var[Q̂ · dt]_random = ℏ² (R′/R)² σ² dt / K = u² σ² dt / K

This accumulated as a random walk: σ_S = |u|√(T/K) after T/dt steps.

With fixed GH nodes, the Q estimate has **no stochastic variance** — it is a deterministic function of the (approximate) density. The error is entirely O(σ_gh²) bias from higher-order Taylor terms, which does not accumulate as a random walk. Phase errors arise only from the KDE density error, not from the quadrature.

This is the principal advantage of the GH switch: eliminating the phase noise that was the dominant error source in the original v2.

### 9.2 Log-Density Mean Weight Variance

The log-density readout computes L = ⟨ln ρ⟩_GH. With exact ρ, this is deterministic and exact for Gaussian densities. With KDE ρ, the noise enters through the density estimate.

To derive the variance explicitly, consider the per-probe-point value ln ρ(x₀ + η) for η ∼ N(0, σ_gh²). For a Gaussian density ln ρ = −x²/σ_ρ² + const, the probe value is ln ρ(x₀ + η) = −(x₀ + η)²/σ_ρ² + const. Its variance under η is:

    Var[ln ρ(x₀ + η)] = E[(x₀+η)⁴]/σ_ρ⁴ − (E[(x₀+η)²])²/σ_ρ⁴
                       = [2σ_gh⁴ + 4x₀²σ_gh²] / σ_ρ⁴

Averaging K independent GH evaluations reduces this by 1/K (with exact GH, the variance is zero; with K random antithetic candidates it is 1/K). The û′ estimator divides by σ_gh², so:

    Var[û′] = (2ν/σ_gh²)² · Var[L] = 4ν² [2 + 4x₀²/σ_gh²] / (K σ_ρ⁴)

Two important observations:

1. **Tail growth**: Var[û′] grows as x₀² in the tails, because ln ρ becomes large and curved there. This is the log-density's main limitation — KDE noise in the tails is amplified. Mirror particles (§6) and a larger Np reduce this.

2. **No phase accumulation**: Unlike the forward Q noise, u′ errors do **not** accumulate in the phase S. The û′ value feeds into Q̃ = Q + ℏu′ at each step as a diagnostic/coupling quantity, but the main particle dynamics (position update via STEER, phase update via Q) are unaffected. Even large per-step errors in û′ do not compound over time.

In the GH regime (exact ρ), Var[û′] = 0 — the estimate is deterministic. All residual noise comes from KDE density error, which scales as (Np σ_kde)⁻¹ in the bulk.

### 9.3 Comparison: Random vs GH Noise Budget

| Error source | Random candidates (original v2) | GH quadrature (updated v2) |
|---|---|---|
| STEER selection noise | O(1/√K) per step | O(1/K^(2K-1)) (quadrature residual) |
| WEIGH Q noise | O(|u|√(dt/K)), accumulates | O(σ_gh²) bias, does not accumulate |
| KDE density noise | Present | Present (unchanged) |
| Phase accumulation | σ_S = |u|√(T/K), ~0.1ℏ at K=128 | None from quadrature; only from KDE |
| K requirement | K ~ 64–128 | K = 4–8 |

The GH approach reduces the candidate count by 8–32× while eliminating the dominant noise source.

### 9.4 Np Convergence: Q and Q̃ Errors vs Particle Count

The following table shows how the forward Q and backward Q̃ errors converge as the number of particles Np increases, using mirror particles (15%), σ_kde = 0.25, σ_gh = 0.3, K = 8, on the HO ground state (ℏ = m = ω = 1). Strategy D (combined: Q from √ρ GH mean, Q̃ = Q + ℏu′ from log-density GH mean) is compared against Strategy A (direct 1/√ρ for Q̃):

| Np | ⟨\|Q err\|⟩ (√ρ) | ⟨\|Q̃ err\|⟩ Strategy A (1/√ρ) | ⟨\|Q̃ err\|⟩ Strategy D (combined) |
|---|---|---|---|
| 200 | 0.111 | 0.253 | 0.266 |
| 500 | 0.237 | 0.321 | 0.289 |
| 1000 | 0.110 | 0.199 | 0.195 |
| 2000 | 0.056 | 0.139 | 0.141 |
| 5000 | 0.085 | 0.066 | 0.111 |

Both backward strategies converge with increasing Np, with Q̃/Q error ratios ranging from 1.3–2.5×. Strategy D (combined) is competitive with or better than Strategy A (direct) across the range. The advantage of Strategy D is decisive in the tail regions (where 1/√ρ diverges exponentially) and becomes clearer at larger Np as the KDE quality improves. The non-monotone convergence in Q error at Np = 500 reflects σ_kde not being optimally re-tuned as Np increases — an adaptive bandwidth controller (§10.2) would smooth this.

---

## 10. The Backward Channel as a Diagnostic Tool

### 10.1 Self-Consistency Checks

The backward readout provides built-in diagnostics that were unavailable in the original v2:

**Osmotic kinetic energy positivity.** The quantity u² = −(Q + Q̃)/m should be non-negative everywhere. If u²_estimated < 0 at some particle, the density estimate at that location is unreliable.

**Cross-validation of u′.** Two independent estimates of u′ are available: (a) directly from the log-density readout, and (b) implicitly from Q via u′ = −2(Q + ½mu²)/ℏ. If these disagree, the KDE is inconsistent (either √ρ or ln ρ is poorly estimated).

**Holland Q± structure.** With both Q and Q̃ available, the full Q± coupling potentials can be monitored. Physically, Q₊ and Q₋ should satisfy certain smoothness and sign properties dictated by the Holland equations.

### 10.2 Adaptive σ_kde Control via u′

The u′ readout provides a clean signal for bandwidth adaptation:

- If |u′| is too large: σ_kde is too small (noisy gradients, spurious energy injection)
- If |u′| is too small: σ_kde is too large (over-smoothing, energy dissipation)

For the ground state, u′_target = −ω is known, providing a direct control target:

    Δσ_kde ∝ (u′_measured − u′_target)

This is the σ_kde-as-energy-dial phenomenon identified in the adaptive controller work, now with a direct diagnostic from the backward channel.

### 10.3 Energy Monitoring

The total quantum energy is E = ⟨½mv² + V + Q⟩. With the backward channel, the quantum kinetic energy decomposes as:

    T_quantum = ½m⟨v²⟩ + ½m⟨u²⟩ = ½m⟨v²⟩ − ⟨½(Q + Q̃)⟩

The osmotic contribution ½m⟨u²⟩ = −½⟨Q + Q̃⟩ can now be monitored independently, providing a check on whether the particle distribution maintains quantum equilibrium (ρ = |ψ|²).

---

## 11. The Schrödinger Bridge Interpretation

### 11.1 Forward and Backward Potentials

In the Schrödinger bridge framework, the density factors as ρ(x,t) = φ̂(x,t) · φ(x,t), where φ̂ = exp(σ₊/ℏ) satisfies the forward heat equation and φ = exp(−σ₋/ℏ) satisfies the backward heat equation.

The three readouts map directly to bridge components:

| Readout | Weight | Bridge component | Physical quantity |
|---|---|---|---|
| STEER | √ρ (selection) | Forward potential φ̂ | Osmotic drift u |
| WEIGH-fwd | √ρ (GH mean) | ∂²φ̂/φ̂ | Quantum potential Q |
| WEIGH-bwd | ln ρ (GH mean) | ∂²(ln φ̂ + ln φ) | Osmotic divergence u′ |

The log-density readout accesses the sum of the forward and backward log-potentials: ln ρ = ln φ̂ + ln φ. Its Laplacian gives the combined curvature, from which u′ separates cleanly as the coupling term between Holland's congruences.

### 11.2 Anti-Diffusion Without Backward Simulation

The backward heat equation ∂ₜφ = −ν∇²φ is ill-posed as an initial-value problem. The updated v2 algorithm never solves it. Instead, it extracts the backward equation's information content — the curvature of the backward potential — from the log-density of the forward ensemble. The Gaussian mean-value property converts a differential operation into an integral one, replacing ill-posed differentiation with well-conditioned GH quadrature.

This is the fundamental resolution of the anti-diffusion problem: the backward channel is accessed through readout, not simulation.

---

## 12. Ground State Stability and the Backward Diagnostic

### 12.1 The Ground State Challenge

The HO ground state demands exact cancellation between the quantum potential Q = −x²/2 + ½ and the classical force −½ω²x² at every point. Any error in Q — from KDE noise, finite K, or bandwidth mismatch — drives the particle distribution away from the ground state.

In the original v2, this manifested as instability originating at the distribution tails (~4σ) where KDE noise is largest, propagating inward over time.

### 12.2 What the Updated Algorithm Provides

The updated v2 gives three handles on the ground state problem:

1. **Reduced Q noise**: GH quadrature eliminates the Monte Carlo phase noise that was the dominant instability mechanism. With K = 8 GH nodes, the Q estimate has O(σ_gh²) deterministic bias and zero stochastic variance.

2. **u′ diagnostic**: The log-density readout should give u′ = −1 everywhere for the ground state. Deviations indicate where the density estimate is failing — providing a spatially resolved diagnostic map.

3. **Mirror particles**: Boundary correction of the KDE reduces the tail bias that initiates the instability cascade. The 2–4× improvement in tail Q accuracy delays the onset of instability.

### 12.3 Ground State Controller

A potential control strategy for the ground state:

1. Compute u′ from the log-density readout at each time step
2. Compare with u′_target = −ω
3. If ⟨|u′ − u′_target|⟩ exceeds threshold, adjust σ_kde:
   - u′ too negative → σ_kde too small → increase
   - u′ too positive → σ_kde too large → decrease
4. The controller directly targets the quantity (osmotic divergence) that determines stationarity

---

## 13. Multi-Dimensional Generalisation

In d dimensions, the GH mean-value property becomes:

    E[f(x + η)] = f(x) + ½σ² ∇²f(x) + O(σ⁴)

where η ∼ N(0, σ²I_d). All three readouts generalise identically:

- **√ρ GH mean** → Q = −(ℏ²/2m)(∇²√ρ/√ρ) — unchanged
- **ln ρ GH mean** → ∇·u = ν ∇²(ln ρ) — d-dimensional osmotic divergence
- **Q̃ = Q + ℏ ∇·u** — d-dimensional backward potential

For the GH quadrature in d dimensions, one can use either tensor-product GH grids (K^d probe points — expensive for d > 3) or random GH directions (project onto random 1D lines — preserves the O(K) cost scaling). The mirror particle technique extends by reflecting along each coordinate axis independently.

The critical advantage of the GH approach (§4) over the random candidate approach becomes more pronounced in higher dimensions: the K^d cost of tensor-product GH is still far lower than the K_random ~ 10³ candidates needed per particle for adequate Q noise in d = 2–3 with random sampling.

---

## 14. Summary

The updated Method 5 v2 algorithm makes four advances over the original:

1. **Fixed GH candidates**: Replace K ~ 64–128 random Brownian draws with K = 4–8 deterministic GH nodes, eliminating the Monte Carlo noise in the quantum potential estimate. The phase noise that was the dominant error source no longer exists. Antithetic cancellation is automatic.

2. **Log-density mean weight**: Extract the osmotic divergence u′ = ∂ₓu through the GH mean of ln ρ rather than direct 1/√ρ evaluation. This preserves the polynomial structure exploited by GH quadrature, giving machine-precision accuracy for Gaussian-like densities and eliminating the exponential ill-conditioning of the backward heat equation.

3. **Mirror particles**: Correct the KDE edge bias by reflecting outermost particles about the distribution boundary, improving forward Q estimates by 2–4× at the tails.

4. **Backward channel readouts**: Access Holland's full bi-HJ structure — the backward quantum potential Q̃, the osmotic kinetic energy u², the inter-congruence coupling, and the Holland Q± — as derivative-free readouts of the same forward ensemble. These provide self-consistency diagnostics (u² ≥ 0, cross-validation of u′), energy decomposition, and adaptive bandwidth control signals.

The governing formulas are:

**Position (Theorem 1):** E[Δx_osmotic] = ν ∂ₓ ln ρ · dt = u · dt  (√ρ selection)

**Forward phase (Theorem 2):** Q · dt = ℏ(1 − M₊)  where M₊ = ⟨√ρ⟩_GH / √ρ(x₀)

**Backward coupling (Theorem 3):** u′ = ν · 2[⟨ln ρ⟩_GH − ln ρ(x₀)] / σ_gh²

**Reconstruction:** Q̃ = Q + ℏu′,   u² = −(Q + Q̃)/m

The algorithm implements the complete Holland bi-Hamilton–Jacobi system using a single forward-evolving particle ensemble. The anti-diffusive sector is accessed through readout, not simulation. The natural variables are actions (S) and log-densities (ln ρ), preserving the polynomial structure that makes GH quadrature both efficient and accurate.

---

## 15. Related Literature

### 15.1 Harmonic function theory and Walk on Spheres

The Gaussian mean-value property E[f(x+η)] = f(x) + ½σ²∇²f(x) + O(σ⁴) is the infinitesimal version of the classical mean value property of harmonic functions. Kakutani (1944) connected this to Brownian motion; Muller (1956) made it algorithmic with the Walk on Spheres. The relevance to Method 5: both WoS and our WEIGH readout replace spatial differentiation with stochastic (or quadrature) integration.

### 15.2 Zeroth-order optimisation

Nesterov and Spokoiny (2017) formalised Gaussian smoothing in the optimisation context: ∇f_μ(x) = μ⁻¹ E[f(x+μv)v], extracting gradient information from function evaluations. Our STEER readout (first derivative from selection) and WEIGH readout (second derivative from mean weight) extend this to quantum-mechanical observables.

### 15.3 Score matching and diffusion models

Score matching (Hyvärinen 2005) estimates ∇ₓ log p(x) from data perturbations. Denoising score matching (Vincent 2011) and score-based diffusion models (Song & Ermon 2019) extract derivative information of log-densities from noisy samples — structurally analogous to our log-density readout. The connection: these methods work with log ρ (scores) while our forward channel works with √ρ. The updated v2 now uses both.

### 15.4 Stein's method

Stein's characterising identity E[f′(Z) − Zf(Z)] = 0 for Gaussian Z encodes the relationship E[g(X)·(X²−1)] = E[g″(X)] — second derivatives from second moments. This is precisely the mathematical structure underlying the WEIGH readout.

---

## 16. References

### Quantum mechanics and stochastic mechanics

- Nelson, E. (1966). Derivation of the Schrödinger Equation from Newtonian Mechanics. *Phys. Rev.* 150, 1079.
- Nelson, E. (2012). Review of stochastic mechanics. *J. Phys.: Conf. Ser.* 361, 012011.
- Holland, P. (2021). Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory. arXiv:2111.09235.
- Hackebill, A. & Poirier, B. (2026). The Problem of Sparse Ontology for Hydrodynamic Formulations of Quantum Mechanics. arXiv:2602.21106.

### Schrödinger bridges, optimal transport, and FBSDEs

- Schrödinger, E. (1931). Über die Umkehrung der Naturgesetze. *Sitzungsber. Preuss. Akad. Wiss.*, 144–153.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). Stochastic Control Liaisons. *SIAM Review* 63, 249–313.
- Conforti, G. & Pavon, M. (2017). Extremal flows on Wasserstein space. arXiv:1712.02257.

### Mean value property, harmonic analysis, and Monte Carlo PDE solvers

- Kakutani, S. (1944). Two-dimensional Brownian motion and harmonic functions. *Proc. Imp. Acad.* 20, 706–714.
- Muller, M.E. (1956). Some continuous Monte Carlo methods for the Dirichlet problem. *Ann. Math. Stat.* 27, 569–589.
- Sawhney, R. & Crane, K. (2020). Monte Carlo geometry processing. *ACM Trans. Graph.* 39(4), Article 123.
- Evans, L.C. (2010). *Partial Differential Equations.* 2nd ed. AMS.

### Zeroth-order optimisation, score matching, and diffusion models

- Nesterov, Y. & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. *Found. Comput. Math.* 17, 527–566.
- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *J. Mach. Learn. Res.* 6, 695–709.
- Song, Y. & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS 2019*. arXiv:1907.05600.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Comput.* 23, 1661–1674.

### Quantum trajectory methods

- Wyatt, R.E. (2005). *Quantum Dynamics with Trajectories.* Springer.
- Bittner, E.R. (2000). Quantum tunneling dynamics using hydrodynamic trajectories. *J. Chem. Phys.* 112, 9703.
- Anderson, J.B. (1975). A random-walk simulation of the Schrödinger equation: H₃⁺. *J. Chem. Phys.* 63, 1499.

---

### Project Knowledge Cross-References

- `NelsonMechanics_SchrodingerBridge_Algorithm.md` — **This document**: core algorithm, theorems, GH quadrature, backward channel, variance analysis, numerical validation
- `NelsonMechanics_SchrodingerBridge_Supplement.md` — **Companion supplement** (§§17–25): Hackebill–Poirier classification, Method 7 lessons, Wasserstein dynamics, bridge interpretation, open questions
- `Method5_Mathematical_Analysis.md` — Original v1 theory: √ρ-selection theorem proof, free Gaussian test (superseded)
- `Method5_Mathematical_Analysis_v2.md` — Original v2: dual-readout theory, random candidate formulation (superseded)
- `Method5_Mathematical_Analysis_v3.md` — v3 first half: log-density, mirror particles, GH implementation (superseded)
- `Method5_v3_second_half.md` — v3 second half: bridge/Wasserstein/classification (superseded by supplement)
- `Adaptive_K_v2_DualReadout.md` — K requirements for STEER vs WEIGH, hybrid Q strategies
- `NelsonMechanics_SchrodingerBridge_Swarmalator.md` §§2–3 — ψ-KDE density estimator theory
- `Holland_Nelson_FokkerPlanck_Analysis.md` — Comparative analysis of Holland and Nelson frameworks
- `FBSDE_SchrodingerBridge_Nelson_Holland.md` — Triangular FBSDE/bridge/bi-HJ relationship
- `Method5_QA_Discussion.md` — Time symmetry, local Sinkhorn, Wasserstein dynamics
- `dual_gh_v2.py` — Numerical validation: four strategies for backward readout on HO ground state
- `m5v2_meanweight_Q.py` — Original v2 implementation (random candidates)
- `m5v2_antithetic_validate.py` — Antithetic variance reduction validation
