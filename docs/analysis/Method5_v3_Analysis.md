# Method 5 v3: Dual-Weight Particle Algorithm for Holland's Bi-Hamilton–Jacobi Equations

## Complete Mathematical Analysis

---

**Status: historical.** This note records the v3 dual-weight algorithm (backward weight, log-density mean-weight, mirror particles) as it stood before the deterministic Gauss–Hermite consolidation. It is superseded by [`../algorithm/NelsonMechanics_SchrodingerBridge_Algorithm.md`](../algorithm/NelsonMechanics_SchrodingerBridge_Algorithm.md) §7.3, which carries the current algorithm forward. Kept for the record of what v3 established and why it changed, not as a current spec.

### Abstract

We present Method 5 v3, a self-consistent particle method for solving the full Holland bi-Hamilton–Jacobi (bi-HJ) system, which decomposes the Schrödinger equation into two coupled real Hamilton–Jacobi equations for the forward and backward action functions σ±. The algorithm extends the v2 dual-readout framework (√ρ-selection for osmotic drift, mean weight for quantum potential) with three new ingredients: (i) a **backward weight** 1/√ρ that probes the anti-diffusive (backward Schrödinger potential) sector, (ii) a **log-density mean-weight** that extracts the osmotic velocity divergence u' = ∂ₓu with machine-precision accuracy for Gaussian-like densities, and (iii) **mirror particles** for boundary correction of KDE density estimates. The log-density approach is identified as the numerically superior strategy for extracting backward information, avoiding the exponential ill-conditioning inherent in direct 1/√ρ evaluation. Together, these mechanisms provide particle-level access to both Holland quantum potentials Q± and all components of the inter-congruence coupling, using a single forward-evolving ensemble with no second or third particle set required.

We prove the key theorems, analyse convergence and variance, present numerical validation on the harmonic oscillator ground state, and connect the full structure to the Schrödinger bridge FBSDE framework.

---

## 1. Background and Motivation

### 1.1 From v2 to v3: What Was Missing

Method 5 v2 (Method5_Mathematical_Analysis_v2.md) established the dual-readout framework:

- **STEER** (selection by √ρ): implements the forward osmotic drift u = ν ∂ₓ ln ρ without differentiating ρ.
- **WEIGH** (mean weight of √ρ): implements the standard Madelung quantum potential Q = −(ℏ²/2m)(∂²ₓ√ρ/√ρ) without differentiating √ρ.

Both readouts extract information from a single Gaussian candidate cloud via the Gaussian mean-value property. The connection to Holland's bi-HJ theory was noted in v2 §10: the steer implements the forward velocity v₊ = v + u, while the weigh provides the quantum potential coupling.

However, v2 accessed only one half of Holland's structure — the **forward** half. The backward Schrödinger potential, the anti-diffusive quantum potential Q̃, and the explicit separation of the osmotic kinetic energy from the osmotic pressure term were not available. Method 5 v3 completes this picture by introducing weight functions that probe the backward sector.

### 1.2 Holland's Bi-Hamilton–Jacobi System

Holland (2021) decomposes the wavefunction ψ = √ρ · exp(iS/ℏ) using two real action functions:

    σ± = S ± R,    where R = (ℏ/2) ln ρ

The Schrödinger equation becomes two coupled Hamilton–Jacobi equations:

    ∂ₜσ₊ + (∂ₓσ₊)²/(2m) + Q₊ + V = 0
    ∂ₜσ₋ + (∂ₓσ₋)²/(2m) + Q₋ + V = 0

where the quantum potentials Q± couple the two equations:

    Q± = ±(ℏ/2m) ∂²ₓσ∓ − (1/4m)[∂ₓ(σ₊ − σ₋)]²

(Note: the ± sign corrects a sign error in Holland's published paper, where ∓ was printed.)

The velocity fields are v± = (1/m)∂ₓσ± = v ± u, where v = (1/m)∂ₓS is the current velocity and u = ν ∂ₓ ln ρ is the osmotic velocity. The single density ρ = exp[(σ₊ − σ₋)/ℏ] satisfies both a forward Fokker–Planck equation (with drift v₊ and +ν∇²ρ source) and a backward Kolmogorov equation (with drift v₋ and −ν∇²ρ source).

### 1.3 The Central Question

Holland's backward equation involves **anti-diffusion**: the source term −ν∇²ρ. This is numerically catastrophic to simulate directly as an SDE — it corresponds to a backward heat equation, which is ill-posed. The central question motivating v3 is:

**Can the anti-diffusive (backward) information be extracted from a forward-evolving particle ensemble, without simulating anti-diffusion?**

The answer is yes, through appropriate weight functions applied to the same Gauss–Hermite (GH) quadrature nodes used for the forward readouts.

### 1.4 Holland Has Only ONE Density

A crucial simplification: Holland's two congruences q±(q±0, t) have their own trajectory densities (Jacobian inverses J±⁻¹), but these do NOT reproduce ρ. Holland is emphatic (§4.2 of his paper) that the peaks of ρ occur where v₊ = v₋ (osmotic velocity vanishes), not where either congruence bunches. The individual congruence densities relate to velocity divergences and are derived quantities.

There is only one physical density ρ, and it is fully determined by the action difference:

    ρ = exp[(σ₊ − σ₋)/ℏ]

Given S and ρ, both action functions are recovered:

    σ₊ = S + (ℏ/2) ln ρ
    σ₋ = S − (ℏ/2) ln ρ

Therefore **M5v3 requires only one ensemble** — the same forward-evolving particles carrying (X, S) as in v2. The backward information comes from different weight functions applied to the density estimated from these particles, not from a separate ensemble.

---

## 2. Three Weight Functions and Their Physical Content

### 2.1 The Forward Weight: √ρ (Review from v2)

Define g(x) = √ρ(x) = exp(R/ℏ) where R = (ℏ/2) ln ρ. The GH mean weight is:

    M₊ = E_GH[g(x + √2 σ ξ)] / g(x) ≈ 1 + σ² g''(x)/g(x)

The second logarithmic derivative is:

    g''/g = (R'/ℏ)² + R''/ℏ = u²/ν² + u'/ν

where u = ν ∂ₓ ln ρ = R'/m is the osmotic velocity and u' = ∂ₓu. This gives the standard quantum potential:

    Q = −(ℏ²/2m)(g''/g) = −½mu² − (ℏ/2)u'

Extracting Q from the GH mean weight:

    Q = −(ℏ²/(mσ²))(M₊ − 1) + O(σ²)

This is the v2 WEIGH readout (Theorem 2 of v2).

### 2.2 The Backward Weight: 1/√ρ

Define f(x) = 1/√ρ(x) = exp(−R/ℏ). Its second logarithmic derivative is:

    f''/f = (R'/ℏ)² − R''/ℏ = u²/ν² − u'/ν

The sign of the R'' term flips compared to √ρ. Define the anti-diffusive quantum potential:

    Q̃ = −(ℏ²/2m)(f''/f) = −½mu² + (ℏ/2)u'

The GH mean weight with 1/√ρ weights gives:

    M₋ = E_GH[f(x + √2 σ ξ)] / f(x) ≈ 1 + σ² f''(x)/f(x)

    Q̃ = −(ℏ²/(mσ²))(M₋ − 1)

### 2.3 The Sum and Difference Decomposition

From the forward and backward weights:

    Q + Q̃  = −mu²           (osmotic kinetic energy)
    Q − Q̃  = −ℏu'           (osmotic pressure / velocity divergence)

These separate cleanly into the two physical components of the quantum potential:

    Q  = −½mu² − (ℏ/2)u'    (forward, standard Madelung)
    Q̃  = −½mu² + (ℏ/2)u'    (backward, anti-diffusive)

**The dual mean-weight readout provides both components of the Holland coupling independently, without ever differentiating the density.**

### 2.4 Connection to Holland's Quantum Potentials Q±

Holland's Q± involve both the density-derived terms (u² and u') and the phase-derived term (v' = ∂ₓv). Expressing everything in terms of Q, Q̃, and v':

    Q₊ = Q + (ℏ/2)v' − ½mu²
    Q₋ = Q̃ − (ℏ/2)v' − ½mu²

The v' term comes from differencing the particle-carried S values — the same spatial differencing already used for the current velocity v in v2. So all ingredients for Q± are available.

### 2.5 Anti-Diffusion as Selection

There is an elegant operational interpretation for the STEER version:

- **Max-selection** (pick candidate with highest √ρ): drift → +u (forward, diffusion)
- **Min-selection** (pick candidate with lowest √ρ): drift → −u (backward, anti-diffusion)

The average of max-selected and min-selected positions gives pure current velocity v; their difference gives 2u. This provides a selection-based decomposition of the full drift into current and osmotic components without any density differentiation.

---

## 3. The Ill-Posedness Problem and the Log-Density Solution

### 3.1 Why 1/√ρ Fails in Practice

The 1/√ρ weight = exp(+x²/(4σ_ρ²)) for a Gaussian density grows exponentially in the tails. This converts the well-behaved quadratic structure of ln ρ into an exponentially growing function, destroying the polynomial structure that GH quadrature exploits.

Numerical tests on the harmonic oscillator ground state (ℏ = m = ω = 1) confirm:

**σ_gh sweep with exact ρ, K_gh = 8:**

| σ_gh | max |Q err| (√ρ) | max |Q̃ err| (1/√ρ) | Ratio |
|------|---------------------|----------------------|-------|
| 0.05 | 2.5 × 10⁻² | 7.1 × 10⁻² | 2.9 |
| 0.10 | 1.0 × 10⁻¹ | 2.9 × 10⁻¹ | 2.9 |
| 0.30 | 9.1 × 10⁻¹ | 3.6 × 10⁰ | 4.0 |
| 0.50 | 2.6 × 10⁰ | 2.5 × 10¹ | 9.8 |
| 0.70 | 4.9 × 10⁰ | 5.5 × 10² | 112 |

The backward error grows exponentially with σ_gh while the forward error grows polynomially. At σ_gh = 0.7, the 1/√ρ estimate is 100× worse. This is the backward heat equation's ill-posedness manifesting directly in the GH quadrature.

### 3.2 The Log-Density Mean Weight (Strategy B)

Instead of evaluating 1/√ρ at GH probe points, evaluate **ln ρ** directly. The GH mean of ln ρ gives:

    ⟨ln ρ(x + √2 σξ)⟩_GH ≈ ln ρ(x) + σ² · ∂²ₓ(ln ρ)/2

Since ∂²ₓ(ln ρ) = 2u'/ν (where u = ν ∂ₓ ln ρ), we extract:

    u' = ν · 2 · [⟨ln ρ⟩_GH − ln ρ(x)] / σ²

**Theorem 3 (Osmotic divergence from log-density mean weight).** Let ρ(x) > 0 be smooth and let ξ₁, ..., ξ_K be GH quadrature nodes with weights w₁, ..., w_K. Define:

    L(x₀) = Σ_k w_k ln ρ(x₀ + √2 σ ξ_k) / Σ_k w_k

Then:

    ∂ₓu(x₀) = ν · [L(x₀) − ln ρ(x₀)] · 2/σ² + O(σ²)

*Proof.* Taylor expand ln ρ(x₀ + η) = ln ρ(x₀) + η(ln ρ)' + ½η²(ln ρ)'' + ⅙η³(ln ρ)''' + ... Under GH quadrature with variance σ², the odd moments vanish and E[η²] = σ², giving L(x₀) = ln ρ(x₀) + ½σ²(ln ρ)'' + O(σ⁴). Since (ln ρ)'' = 2u'/ν, the result follows. ∎

### 3.3 Why Log-Density Is Machine-Precision Exact for Gaussians

For the HO ground state, ln ρ = −x² + const — a **quadratic**. GH quadrature with K points is exact for polynomials of degree ≤ 2K − 1. With K = 8, it is exact for degree 15. Therefore the log-density mean value is computed **exactly**, giving (ln ρ)'' = −2, hence u' = ν(−2) = −1 to machine precision.

The numerical test confirms: max |u' error| = 3.8 × 10⁻¹⁴ at σ_gh = 0.3 — fifteen orders of magnitude better than the 1/√ρ approach at the same σ_gh.

### 3.4 Strategy D: The Combined Approach (Recommended)

**Strategy D** combines the strengths of both readouts:

1. **Q from √ρ mean weight** (well-conditioned, O(σ²) error)
2. **u' from log-density mean weight** (machine-precision for Gaussian-like densities)
3. **Q̃ = Q + ℏu'** (reconstructed, with error dominated by the √ρ error alone)

This completely eliminates the exponential instability of direct 1/√ρ evaluation. The Q̃ error equals the Q error — no additional degradation from the backward piece.

**σ_gh sweep, Strategy D vs Strategy A (exact ρ):**

| σ_gh | max |Q̃ err| (A: 1/√ρ) | max |Q̃ err| (D: combined) |
|------|---------------------|-------------------------|
| 0.05 | 7.1 × 10⁻² | 2.5 × 10⁻² |
| 0.10 | 2.9 × 10⁻¹ | 1.0 × 10⁻¹ |
| 0.30 | 3.6 × 10⁰ | 9.1 × 10⁻¹ |
| 0.50 | 2.5 × 10¹ | 2.6 × 10⁰ |
| 0.70 | 5.5 × 10² | 4.9 × 10⁰ |

Strategy D matches the forward Q error exactly, eliminating the exponential blow-up.

### 3.5 The Deeper Insight: Work in the Natural Variables

The Holland bi-HJ coupling is most naturally expressed through **ln ρ** (the action difference σ₊ − σ₋ = ℏ ln ρ), not through powers of ρ. Working in the log representation preserves the polynomial/smooth structure that GH quadrature exploits. This is consistent with the earlier finding that the (X, S) representation is superior to (X, p) — the natural variables are actions and log-densities, not exponentials of them.

The anti-diffusion information is fundamentally about the **curvature of ln ρ**, which is ∂²ₓ(ln ρ) = (2/ν)u'. The 1/√ρ weight converts this into exponential growth, destroying the polynomial structure. The log approach preserves it.

---

## 4. Mirror Particles for Boundary Correction

### 4.1 The Edge Bias Problem

Standard KDE with a Gaussian kernel underestimates density near the boundary of the particle distribution. For particles sampled from a distribution with finite support or rapid tail decay (like the HO ground state), the kernel extends beyond the outermost particles into empty space, causing a systematic downward bias.

This bias is amplified by the 1/√ρ weight (which magnifies small-density errors) and by the log weight (where ln ρ → −∞ as ρ → 0). Both backward-information channels are particularly sensitive to tail accuracy.

### 4.2 The Mirror Particle Method

Augment the particle ensemble with **reflected copies** of the outermost particles:

1. Sort the Np particles.
2. Select the n_mirror = ⌊f_mirror · Np⌋ leftmost and rightmost particles.
3. Reflect the leftmost particles about the distribution minimum: x_mirror = 2·x_min − x_original.
4. Reflect the rightmost particles about the distribution maximum: x_mirror = 2·x_max − x_original.
5. Include the mirror particles in the KDE evaluation but normalise by the original Np.

The mirror particles provide "phantom density" beyond the boundary, correcting the kernel truncation bias. This is a standard technique in the KDE literature (boundary correction by reflection).

### 4.3 Numerical Impact

On the HO ground state with Np = 2000, σ_kde = 0.25:

| Configuration | Q bulk max err | Q tail max err |
|---|---|---|
| No mirror | 1.30 | 1.30 |
| Mirror 10% | 0.49 | 0.33 |
| Mirror 15% | 0.49 | 0.33 |

Mirror particles reduce the forward Q error by 2.6× at the distribution edges and 4× in the tails. The improvement saturates at ~10–15% mirror fraction.

For the backward channel (log-density u'), mirror particles provide modest improvement. The KDE quality is the real bottleneck — the GH quadrature is exact for exact ρ, so all error comes from the density estimate.

---

## 5. The M5v3 Algorithm

### 5.1 State Per Particle

Each particle k carries position X_k and phase S_k. The action functions are:

    σ₊_k = S_k + (ℏ/2) ln ρ(X_k)     (forward, from KDE)
    σ₋_k = S_k − (ℏ/2) ln ρ(X_k)     (backward, from KDE)

### 5.2 Algorithm (Per Time Step)

```
INPUT: Particle ensemble {X_k, S_k}, k = 1,...,Np

1. DENSITY ESTIMATION
   a. KDE from particle positions → ρ(x)
   b. Augment with mirror particles for boundary correction
   c. Precompute √ρ and ln ρ on evaluation grid or at particle positions

2. CURRENT VELOCITY
   v_k = (1/m) ∂_x S evaluated at X_k (from neighbouring particles' S values
   or grid interpolation)

3. CLASSICAL STEP
   X_class_k = X_k + v_k · dt − (1/m) ∂_x V(X_k) · dt

4. GENERATE K CANDIDATES (with antithetic pairs)
   For j = 1,...,K/2:
     ξ_j ~ N(0,1)
     x'_{2j-1} = X_class_k + σ√dt · ξ_j
     x'_{2j}   = X_class_k − σ√dt · ξ_j

5. EVALUATE √ρ WEIGHTS
   w_j = √ρ(x'_j)  for each candidate
   w₀  = √ρ(X_class_k)

6. STEER (position update — forward selection)
   p_j = w_j / Σ w_j
   Draw x'* ~ Categorical(p₁,...,p_K)
   Set X_k ← x'*

7. FORWARD WEIGH (quantum potential Q)
   ⟨w⟩ = (1/K) Σ_j w_j
   Q_k · dt = ℏ (1 − ⟨w⟩ / w₀)

8. LOG-DENSITY WEIGH (osmotic divergence u')
   ⟨ln ρ⟩ = (1/K) Σ_j ln ρ(x'_j)     [or GH quadrature]
   u'_k = ν · 2 · [⟨ln ρ⟩ − ln ρ(X_class_k)] / (σ² dt)

9. BACKWARD POTENTIAL (reconstructed)
   Q̃_k = Q_k + ℏ · u'_k

10. HOLLAND COUPLING (optional diagnostic / future use)
    Q₊_k = Q_k + (ℏ/2) v'_k − ½m u²_k
    Q₋_k = Q̃_k − (ℏ/2) v'_k − ½m u²_k
    where u²_k = −(Q_k + Q̃_k)/m,  v'_k from S gradient differencing

11. UPDATE ACTION
    S_k ← S_k + [½m v_k² − V(X_k)] · dt − Q_k · dt

OUTPUT: Updated ensemble {X_k, S_k}
```

### 5.3 What v3 Adds to v2

The core dynamics (steps 1–7, 11) are identical to v2. The new ingredients are:

- **Step 8 (log-density weigh):** Extracts u' = ∂ₓu from the same candidate cloud, using ln ρ instead of √ρ. Cost: one log evaluation per candidate (trivial, since ρ is already computed).

- **Step 9 (backward reconstruction):** Combines Q and u' to give Q̃ without direct 1/√ρ evaluation. Zero additional cost.

- **Step 10 (Holland coupling):** Provides the full Q± for diagnostic purposes or for future algorithms that evolve σ± directly.

- **Mirror particles (step 1b):** Augments the KDE for boundary correction. Cost: O(n_mirror) additional particles in density evaluation.

### 5.4 Why No Second Ensemble Is Needed

The reason is physical: Holland's two congruences describe the **same quantum state** from two perspectives (forward and backward in the Schrödinger bridge sense). The density ρ is shared. The two action fields σ± are not independent — they are linked by ρ.

In the (S, ρ) representation:
- S = (σ₊ + σ₋)/2 → tracked per particle
- ρ = exp[(σ₊ − σ₋)/ℏ] → estimated from ensemble via KDE

Given S and ρ, both σ₊ and σ₋ are fully determined. **M5v2 already implicitly solves Holland's bi-HJ system.** M5v3 makes the backward sector explicit through the log-density readout.

---

## 6. The Schrödinger Bridge Interpretation

### 6.1 Forward and Backward Schrödinger Potentials

In the Schrödinger bridge framework, the density factors as ρ(x,t) = φ̂(x,t) · φ(x,t), where:

- φ̂ = exp(σ₊/ℏ) is the forward Schrödinger potential, satisfying a forward heat equation
- φ = exp(−σ₋/ℏ) is the backward Schrödinger potential, satisfying a backward heat equation

The bridge measure on paths is:

    dP = φ̂(x₀, 0) · φ(x_T, T) · dP_Wiener

The IPFP (Sinkhorn) algorithm computes the bridge by alternating forward and backward reweightings.

### 6.2 The Three Readouts as Bridge Components

| Readout | Weight function | Bridge component | Physical quantity |
|---------|----------------|------------------|-------------------|
| STEER | √ρ (selection) | Forward potential φ̂ | Osmotic drift u |
| WEIGH-forward | √ρ (mean weight) | ∂²φ̂/φ̂ | Quantum potential Q |
| WEIGH-backward | ln ρ (log mean) | ∂²(ln ρ) = ∂²(ln φ̂ + ln φ) | Osmotic divergence u' |

The log-density readout accesses the **sum** of the forward and backward log-potentials: ln ρ = ln φ̂ + ln φ. Its Laplacian gives the combined curvature, which separates cleanly into u' (the coupling term between Holland's congruences).

### 6.3 Anti-Diffusion Without Backward Simulation

The backward heat equation ∂ₜφ = −ν∇²φ is ill-posed as an initial-value problem. But its **information content** — the curvature of the backward potential — is encoded in the log-density. The v3 algorithm extracts this content through a forward-time GH quadrature of ln ρ, never solving the backward equation directly.

This is analogous to how the v2 WEIGH readout extracts the Laplacian of √ρ without differentiating: in both cases, the Gaussian mean-value property converts a differential operation into an integral one, replacing ill-posed differentiation with well-conditioned averaging.

The anti-diffusive Fokker–Planck equation ∂ₜρ + ∂ₓ(ρv₋) = −ν∂²ρ is never solved. Its content is captured by the log-density readout of the forward ensemble.

---

## 7. Variance Analysis

### 7.1 Forward Mean Weight (Review)

From v2 §7.2, the variance of the √ρ mean-weight Q estimator is:

    Var[Q̂ · dt] = ℏ² (R'/R)² σ² dt / K = u² σ² dt / K

The Q noise enters the phase update and accumulates as a random walk: σ_S = |u|√(T/K) after T/dt steps.

### 7.2 Log-Density Mean Weight

The log-density readout computes L = ⟨ln ρ⟩_GH. The variance of L is:

    Var[L] = Var[ln ρ(x + η)] / K

For a Gaussian density ln ρ = −x² + const, the probe value ln ρ(x + η) = −(x + η)² + const. Its variance under η ~ N(0, σ²dt) is:

    Var[−(x+η)²] = E[(x+η)⁴] − (E[(x+η)²])² = 2(σ²dt)² + 4x²(σ²dt)

So:

    Var[L] = [2σ⁴dt² + 4x²σ²dt] / K

The u' estimator involves dividing by σ²dt:

    Var[û'] = (ν²/σ⁴dt²) · 4 · Var[L] = 4ν²[2 + 4x²/(σ²dt)] / K

For finite dt, this has a term proportional to x²/(σ²dt·K), which can be large in the tails. However, the factor of 4ν² = ℏ²/m² is small in typical units, and the log-density estimate benefits from the same antithetic variance reduction as the √ρ weight (odd moments cancel exactly).

### 7.3 Comparison: √ρ vs ln ρ Noise

| Quantity | √ρ noise scaling | ln ρ noise scaling |
|----------|-----------------|-------------------|
| Per-step noise | |u|·σ√(dt/K) | (ν/σ)·√(8x²/K + ...) |
| Accumulation | Random walk in S | Not accumulated (u' enters Q̃, not S directly) |
| Tail behavior | Bounded (√ρ → 0 damps noise) | Grows as |x| (ln ρ diverges) |

The key insight: u' from the log readout does **not** accumulate in the phase. It is used to compute Q̃ at each step, which is a diagnostic/coupling quantity, not an input to the main dynamics (which uses Q from the √ρ weight). So even if the log-density noise is larger per step, it does not compound over time.

---

## 8. Gauss–Hermite Quadrature Implementation

### 8.1 GH Nodes and Weights

The GH mean weight uses physicist's Hermite polynomials. For K quadrature points:

    ξ₁, ..., ξ_K and w₁, ..., w_K from roots_hermite(K)

The probe points at particle position x with scale σ_gh are:

    x_k = x + √2 · σ_gh · ξ_k

### 8.2 Forward Mean Weight M₊

    M₊(x) = [Σ_k w_k · √ρ(x_k)] / [√ρ(x) · Σ_k w_k]

    Q(x) = −(ℏ²/(m·σ_gh²)) · (M₊ − 1)

### 8.3 Log-Density Mean

    L(x) = [Σ_k w_k · ln ρ(x_k)] / [Σ_k w_k]

    u'(x) = ν · 2 · [L(x) − ln ρ(x)] / σ_gh²

### 8.4 Combined Backward Potential

    Q̃(x) = Q(x) + ℏ · u'(x)

### 8.5 Choice of σ_gh and K

For the forward weight, the optimal σ_gh balances bias (O(σ²)) against noise (O(1/√K)). From v2, σ_gh ~ σ_rho/2 is a good starting point.

For the log-density weight, the bias is also O(σ²) but the coefficient is different: it depends on ∂⁴(ln ρ) rather than ∂⁴(√ρ). For Gaussian-like densities, ∂⁴(ln ρ) = 0 (ln ρ is exactly quadratic), so the bias vanishes identically and σ_gh can be chosen purely for noise reduction.

K = 4–8 suffices for both readouts. The GH quadrature is exact for polynomials of degree ≤ 2K−1, so K = 4 handles quartic ln ρ exactly, and K = 8 handles degree-15 corrections. Increasing K beyond 8 provides diminishing returns for smooth densities.

---

## 9. Numerical Validation: Harmonic Oscillator Ground State

### 9.1 Test Setup

The HO ground state (ℏ = m = ω = 1) provides the critical test case because it demands exact balance between the quantum and classical potentials:

    ρ(x) = (1/√π) exp(−x²),    σ_ρ = 1/√2

    Q(x)  = −x²/2 + 1/2        (forward quantum potential)
    Q̃(x) = −x²/2 − 1/2        (backward quantum potential)
    u(x)  = −x                  (osmotic velocity)
    u'    = −1                  (constant, osmotic divergence)

### 9.2 Results with Exact ρ

Using exact ρ (isolating GH quadrature error), with σ_gh = 0.3, K = 8:

| Quantity | Strategy A (direct) | Strategy B (log) | Strategy D (combined) |
|----------|----|----|-----|
| max Q error | 9.1 × 10⁻¹ | — | 9.1 × 10⁻¹ |
| max Q̃ error | 3.6 × 10⁰ | — | 9.1 × 10⁻¹ |
| max u' error | 4.7 × 10⁰ | 3.8 × 10⁻¹⁴ | 3.8 × 10⁻¹⁴ |

Strategy D reduces the Q̃ error by 4× compared to Strategy A, and recovers u' to machine precision.

### 9.3 Results with KDE ρ (Np = 2000)

With particle-based KDE (σ_kde = 0.25, mirror = 15%):

| Quantity | Strategy A | Strategy D |
|----------|-----------|-----------|
| Q bulk mean error | 0.065 | 0.065 |
| Q̃ bulk mean error | 0.093 | 0.115 |
| u' bulk mean | −0.928 | −0.882 |

The KDE quality now dominates — both strategies give comparable errors. The advantage of Strategy D becomes decisive only when the density estimate improves (larger Np, adaptive bandwidth, ψ-KDE).

### 9.4 Np Convergence

With mirror = 15%, σ_kde = 0.25, σ_gh = 0.3, K = 8:

| Np | ⟨|Q err|⟩ (√ρ) | ⟨|Q̃ err|⟩ (1/√ρ) | ⟨|Q̃ err|⟩ (combined) |
|---|---|---|---|
| 200 | 0.111 | 0.253 | 0.266 |
| 500 | 0.237 | 0.321 | 0.289 |
| 1000 | 0.110 | 0.199 | 0.195 |
| 2000 | 0.056 | 0.139 | 0.141 |
| 5000 | 0.085 | 0.066 | 0.111 |

Both backward strategies converge, with the ratio Q̃/Q error ranging from 1.3 to 2.5. Strategy D becomes competitive with Strategy A at large Np and may win when combined with the ψ-KDE density estimator.

---

## 10. Connections to the v2 Framework

### 10.1 Backward Compatibility

M5v3 is a strict extension of v2. Steps 1–7 and 11 are unchanged. The log-density readout (step 8) and backward reconstruction (step 9) are optional additions that provide new information without modifying the core dynamics.

Any v2 implementation can be upgraded to v3 by adding the log-density evaluation at the GH probe points (one extra line of code per probe point) and computing Q̃ = Q + ℏu'.

### 10.2 The Dual Readout Table (Updated)

| Readout | Weight function | Operation | Derivative order | Physical quantity | Target equation |
|---------|----------------|-----------|-----------------|-------------------|-----------------|
| STEER | √ρ | Selection (rank) | 1st (∂ₓR/R) | Osmotic drift u | Nelson forward SDE |
| WEIGH-fwd | √ρ | Mean (average) | 2nd (∂²ₓR/R) | Forward Q potential | Phase HJB (Madelung) |
| WEIGH-bwd | ln ρ | Mean (average) | 2nd (∂²ₓ ln ρ) | Osmotic divergence u' | Holland bi-HJ coupling |
| RECONSTRUCT | Q + ℏu' | Algebra | — | Backward Q̃ potential | Backward HJB |

### 10.3 Information-Theoretic Perspective (Extended)

The candidate cloud at each particle position samples the local density landscape through a Gaussian window. The v3 readouts extract:

| Derivative order | Physical quantity | v2 readout | v3 addition |
|---|---|---|---|
| 0th | √ρ(x₀) itself | w₀ (departure weight) | ln ρ(x₀) |
| 1st | ∂ₓ ln ρ → u | Selection (STEER) | Min-selection → −u |
| 2nd | ∂²ₓ√ρ/√ρ → Q | Mean √ρ weight (WEIGH) | Mean ln ρ → u' → Q̃ |

The pattern: **each new weight function applied to the same GH nodes opens a new information channel** about the density landscape. The √ρ channel gives (u², u') mixed; the ln ρ channel gives u' directly. Together they separate the two components cleanly.

---

## 11. Implications for Ground State Stability

### 11.1 The Ground State Challenge

The HO ground state has been the most persistent challenge for M5v2: the adaptive σ_kde controller hits ceilings and shows residual instability. The diagnostic tool (gs_ho_diagnostic.py, historical, not in repository) was designed to identify whether spurious phase gradients or incorrect initialization cause the instability.

### 11.2 What v3 Provides

The separate u' readout gives a new diagnostic: if u'_estimated ≠ −ω (the exact value for the HO ground state), the density estimate is unreliable. Specifically:

- If u' is too negative (|u'| > ω): the density is too peaked (σ_kde too small, noisy gradients inject energy)
- If u' is not negative enough (|u'| < ω): the density is too broad (σ_kde too large, dissipating energy)

This is precisely the σ_kde-as-energy-dial phenomenon identified in the adaptive controller work, but now diagnosed independently through u' rather than through the energy time series.

### 11.3 Potential Use as a Controller Signal

The u' error could serve as an alternative or supplementary signal for the adaptive bandwidth controller:

    σ_kde adjustment ∝ (u'_measured − u'_target)

For the ground state, u'_target = −ω is known. More generally, u' can be cross-checked against the value implied by the forward Q: since Q = −½mu² − (ℏ/2)u', one can extract u'_from_Q = −2(Q + ½mu²)/ℏ. If the log-density u' and the Q-implied u' agree, the density estimate is self-consistent.

---

## 12. Multi-Dimensional Generalisation

In d dimensions, the GH mean-value property becomes:

    E[f(x + η)] = f(x) + ½σ² ∇²f(x) + O(σ⁴)

where η ~ N(0, σ² I_d) and ∇² is the d-dimensional Laplacian.

All three readouts generalise identically:

- **√ρ mean weight** → Q = −(ℏ²/2m)(∇²√ρ/√ρ) — unchanged from v2
- **ln ρ mean weight** → ∇·u = ν ∇²(ln ρ) — the d-dimensional osmotic divergence
- **Q̃ = Q + ℏ ∇·u** — the d-dimensional backward potential

The mirror particle technique extends by reflecting about the convex hull of the particle distribution (or, more practically, by reflecting along each coordinate axis independently).

---

## 13. Summary

Method 5 v3 extends the v2 dual-readout framework to explicitly access Holland's bi-Hamilton–Jacobi structure:

1. **√ρ selection** → forward osmotic drift u (unchanged from v2)
2. **√ρ mean weight** → forward quantum potential Q (unchanged from v2)
3. **ln ρ mean weight** → osmotic divergence u' (NEW in v3)
4. **Algebraic reconstruction** → backward quantum potential Q̃ = Q + ℏu' (NEW in v3)
5. **Mirror particles** → boundary-corrected KDE (NEW in v3)

The key finding is that the anti-diffusive (backward) information is best extracted through the **log-density** channel rather than the **1/√ρ** channel, because:

- ln ρ preserves the polynomial structure exploited by GH quadrature
- 1/√ρ converts this into exponential growth, causing exponential error amplification
- For Gaussian-like densities, the log-density readout is machine-precision exact
- The combined Strategy D gives Q̃ error equal to Q error, with no backward-specific degradation

The anti-diffusive Fokker–Planck equation is never solved directly. Its information content is extracted from the forward ensemble through the log-density weight — the backward heat equation's content captured by a forward-time integral operation.

**M5v3 provides a complete particle-level implementation of Holland's bi-HJ system using a single forward-evolving ensemble, with all inter-congruence coupling terms available as derivative-free readouts of the particle density.**

---

## 14. References

### Quantum mechanics and stochastic mechanics

- Nelson, E. (1966). Derivation of the Schrödinger Equation from Newtonian Mechanics. *Phys. Rev.* 150, 1079.
- Nelson, E. (2012). Review of stochastic mechanics. *J. Phys.: Conf. Ser.* 361, 012011.
- Holland, P. (2021). Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory. arXiv:2111.09235.
- Hackebill, A. & Poirier, B. (2026). The Problem of Sparse Ontology for Hydrodynamic Formulations of Quantum Mechanics. arXiv:2602.21106.

### Schrödinger bridges, optimal transport, and FBSDEs

- Schrödinger, E. (1931). Über die Umkehrung der Naturgesetze. *Sitzungsber. Preuss. Akad. Wiss.*, 144–153.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). Stochastic Control Liaisons. *SIAM Review* 63, 249–313.
- Conforti, G. & Pavon, M. (2017). Extremal flows on Wasserstein space. arXiv:1712.02257.
- Pavon, M. (1995). Hamilton's principle in stochastic mechanics. *J. Math. Phys.* 36, 6774–6800.
- Zambrini, J.-C. (1986). Stochastic mechanics according to E. Schrödinger. *Phys. Rev. A* 33, 1532.

### Mean value property and Monte Carlo PDE solvers

- Kakutani, S. (1944). Two-dimensional Brownian motion and harmonic functions. *Proc. Imp. Acad.* 20, 706–714.
- Sawhney, R. & Crane, K. (2020). Monte Carlo geometry processing. *ACM Trans. Graph.* 39(4), Article 123.
- Evans, L.C. (2010). *Partial Differential Equations.* 2nd ed.

### Quantum trajectory methods

- Wyatt, R.E. (2005). *Quantum Dynamics with Trajectories: Introduction to Quantum Hydrodynamics.* Springer.
- Bittner, E.R. (2000). Quantum tunneling dynamics using hydrodynamic trajectories. *J. Chem. Phys.* 112, 9703.
- Anderson, J.B. (1975). A random-walk simulation of the Schrödinger equation: H₃⁺. *J. Chem. Phys.* 63, 1499.

### Zeroth-order optimisation and score matching

- Nesterov, Y. & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. *Found. Comput. Math.* 17, 527–566.
- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *J. Mach. Learn. Res.* 6, 695–709.
- Song, Y. & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS 2019*. arXiv:1907.05600.

## 15. Context: Trajectory Methods for Quantum Hydrodynamics

### 15.1 The Hackebill–Poirier Classification

Hackebill & Poirier (2026) provide a systematic taxonomy of hydrodynamic formulations of quantum mechanics that usefully situates Method 5 v3. Their classification rests on whether trajectories are defined by the **current velocity** v = ∂ₓS/m (the Bohmian guidance equation) or by some other velocity field, and on whether the ontology is **dense** (one trajectory through every point in space at every time, as in de Broglie–Bohm) or **sparse** (a finite ensemble, as in all practical computations).

The key distinction is between what they call:

- **CHV (current-velocity hydrodynamic) methods:** Trajectories follow v = ∇S/m. These include standard de Broglie–Bohm theory and the numerical quantum trajectory methods of Wyatt (2005), Bittner (2000), and Lopreore & Wyatt (1999). In 1D, trajectories never cross (topological constraint). The density evolves passively — ρ is the Jacobian inverse of the trajectory map, not an independently specified field.

- **DHV (drift-velocity hydrodynamic) methods:** Trajectories follow the **full Nelson drift** b = v + u, which includes the osmotic component. These are the stochastic trajectory methods — Nelson's original formulation, and by extension, all FBSDE-based approaches including Method 5.

- **Bipolar ansatz methods:** The wavefunction is decomposed as ψ = ψ₊ + ψ₋ (a sum of "plus" and "minus" components, each with its own density and phase). Each component's trajectories are CHV with respect to their own phase, but the two components couple through cross-terms in the quantum potential. This connects directly to Holland's two-congruence picture.

### 15.2 Where Method 5 v3 Sits

Method 5 v3 is a **sparse DHV method** in the Hackebill–Poirier taxonomy, but one that accesses information from **both** the forward (dissipative) and backward (anti-dissipative) sectors of Holland's bi-HJ system. This is a combination that does not appear in their classification, because they do not consider the possibility of extracting the backward sector's content without actually simulating backward trajectories.

The key structural relationships are:

The **current velocity** v = ∂ₓS/m is obtained from the particle-carried phases {Sᵢ} by local interpolation — the same as in any CHV method. This is the T-odd (time-reversal-antisymmetric) component of the drift.

The **osmotic velocity** u = ν ∂ₓ ln ρ is obtained from the √ρ-weighted selection mechanism without differentiating the density — this is the M5 innovation over standard DHV methods that require explicit computation of ∇ ln ρ.

The **osmotic divergence** u' = ∂ₓu is obtained from the ln ρ mean-weight readout — the M5v3 innovation that accesses Holland's backward sector.

The **quantum potentials** Q and Q̃ are assembled from the GH mean weights — no spatial differentiation of any field is required at any stage.

The result is a sparse DHV method that solves the full Holland bi-HJ system using only pointwise density evaluations and GH quadrature averages, with the only remaining spatial derivative being ∂ₓS for the current velocity (which is well-behaved everywhere).

### 15.3 The Sparse Ontology Question

Hackebill & Poirier emphasise the "sparse ontology" problem: in a finite-ensemble computation, the trajectories cannot faithfully represent the full continuum dynamics. Regions with few particles have noisy field estimates; density nodes are never exactly resolved; caustics and trajectory crossing cannot occur in a 1D CHV method but can (and do) occur in DHV methods.

Method 5 addresses this differently from standard approaches:

In Wyatt-style CHV methods, node-crossing is forbidden by the 1D topology, but the quantum potential Q = −(ℏ²/2m)(∂²ₓ√ρ/√ρ) must be computed from the local trajectory density, which diverges as trajectories approach a node from either side. The "node problem" (Bittner 2000, Kendrick 2003, Trahan & Wyatt 2003) has historically been the Achilles' heel of CHV trajectory methods.

In Method 5, the node problem is circumvented rather than solved. The √ρ selection weight vanishes at nodes, so particles are never placed there — the singular osmotic velocity that would be required to push particles through a node simply never needs to be computed. The quantum potential Q is obtained from the GH mean weight of √ρ, which is well-conditioned everywhere because it involves only function evaluation (not differentiation) of a smooth function that happens to vanish at nodes.

The sparse-ontology limitation remains — the density is always the particle histogram, and the velocity field is always reconstructed from finite samples — but the specific instabilities that plague both CHV and explicit-drift DHV methods at nodes are eliminated by construction.

### 15.4 Connection to the Bipolar Ansatz

Holland's bi-HJ decomposition ψ = √ρ · exp(iS/ℏ) with σ± = S ± (ℏ/2) ln ρ is related to but distinct from the bipolar ansatz ψ = ψ₊ + ψ₋. In the bipolar case, ψ± are independently propagated wavefunctions with their own densities |ψ±|² and phases S±, coupled through cross-quantum potentials. In Holland's case, σ± are action functions derived from the same (ρ, S) pair — they are not independent dynamical variables.

This distinction is crucial for M5v3: because σ± = S ± R are algebraically determined by (S, ρ), a single ensemble carrying (X, S) suffices. The bipolar ansatz requires two independent ensembles (or equivalently, a doubled state space), while Holland's decomposition requires only a single ensemble with one additional readout (the log-density mean weight) to access the backward sector.

The Method 7 implementation (§16) explored the bipolar-like approach of maintaining two separate ensembles and found that the anti-dissipative ("−") ensemble develops caustics and requires periodic resampling. The M5v3 approach avoids this entirely by staying within a single forward-evolving ensemble.

---

## 16. Lessons from the Two-Species Method 7

### 16.1 The Dissipative/Anti-Dissipative Asymmetry

The Method 7 prototype (developed in parallel with v3) implemented Holland's two-congruence structure literally: "+" particles propagated stochastically with √ρ selection (drift v₊ = v + u), while "−" particles propagated deterministically with drift v₋ = v − u.

The key finding was a fundamental **algorithmic asymmetry** between the two channels:

The **forward (dissipative) channel** is well-conditioned. The √ρ selection biases candidates toward higher density, implementing the osmotic drift u = ν ∂ₓ ln ρ that drives particles toward density peaks. Diffusion spreads, selection corrects — a stable feedback loop.

The **backward (anti-dissipative) channel** cannot use the same mechanism. The target drift v₋ = v − u would require selection proportional to ρ⁻¹/², which biases candidates toward lower density and diverges catastrophically at nodes. This is the backward heat equation's ill-posedness manifesting in the selection framework.

The mathematical reason: in Method 5's "classical push + Brownian diffusion + weighted selection" structure, the selection weight w(x) generates an osmotic drift σ² · ∂ₓ ln w(x). To get drift −u, one needs ∂ₓ ln w = −½ ∂ₓ ln ρ, which gives w = ρ⁻¹/² — exponentially growing in the tails and divergent at nodes.

This is not a limitation of the specific implementation but a structural fact: **anti-diffusion cannot be implemented by importance-sampling over Brownian candidates, because diffusion only knows how to spread.** The anti-dissipative process concentrates rather than spreads, and concentration requires a fundamentally different algorithmic mechanism.

### 16.2 The Inescapable Conclusion and the v3 Resolution

The Method 7 experience established that:

1. Maintaining separate "−" particles is fragile — the anti-dissipative flow develops caustics (trajectory crossing and density singularities) that require periodic resampling.

2. The phase-based density diagnostic ρ = exp[(σ₊ − σ₋)/ℏ] degrades over time because σ₋ reconstruction from the deterministic "−" ensemble loses coherence through interference nodes.

3. The forward ("+" ) channel reproduces standard Method 5 exactly, validating the algorithmic framework.

M5v3 resolves this by recognising that the backward sector's information content — the anti-diffusive quantum potential Q̃ and the osmotic divergence u' — can be extracted from the **same forward ensemble** through a different weight function (ln ρ) applied to the same GH probe points. The ill-posed backward heat equation is never solved; its curvature content is captured by a well-conditioned integral operation.

This is the central conceptual advance of v3: **the anti-diffusive channel requires not anti-diffusion of particles but anti-symmetric weighting of the density landscape.**

### 16.3 What Method 7 Confirmed

Despite its limitations, Method 7 provided crucial validation:

**The Q± sign correction.** Deriving Q± from scratch and verifying against the free Gaussian at t = 0 confirmed that Holland's published ∓ in eq. (2.4) is a sign error; the correct signs are ±.

**The trajectory phenomenology.** The "+" trajectories showed dissipative spreading with Bohmian-style non-crossing through interference zones. The "−" trajectories showed the predicted anti-dissipative contraction, with violent oscillations near collision nodes — exactly the concentrating behaviour predicted by Holland's backward Kolmogorov equation.

**The Bohm trajectories as geometric mean.** The standard Bohmian trajectories (drift v = (1/m)∂ₓS) are the arithmetic mean of the v₊ and v₋ velocity fields. Method 5's particles, which carry phase S and are pushed by v with √ρ-corrected osmotic drift, compute this mean trajectory without ever constructing either v₊ or v₋ explicitly.

---

## 17. The Full Schrödinger Bridge Interpretation

### 17.1 Holland's bi-HJ as an FBSDE Optimality System

The identification between Holland's framework and the Schrödinger bridge is now precise across all levels. Holland's bi-HJ equations are exactly the Hamilton–Jacobi–Bellman (HJB) equations of the forward and backward stochastic control problems that define the Schrödinger bridge.

The correspondence table (extending the one in FBSDE_SchrodingerBridge_Nelson_Holland.md §3.1):

    Schrödinger Bridge          Nelson's Mechanics         Holland's bi-HJ
    ────────────────────────    ───────────────────────    ──────────────────────
    φ̂ (forward potential)       R + S/ℏ                    σ₊/ℏ
    φ (backward potential)      R − S/ℏ                    −σ₋/ℏ
    ∇φ̂ (forward drift)          b = v + u                  v₊ = ∇σ₊/m
    −∇φ (backward drift)        b* = v − u                 v₋ = ∇σ₋/m
    ρ = exp(φ̂ − φ)              ρ = exp(2R/ℏ)              ρ = exp[(σ₊ − σ₋)/ℏ]
    Fisher information           Osmotic KE ½|u|²           Source terms in FPE±
    ν = ε/2                      ν = ℏ/(2m)                 ν = ℏ/(2m)

The bridge density factorises as ρ = φ̂ · φ, with each factor corresponding to one Schrödinger potential. In the time-symmetric case (S = 0), both factors reduce to √ρ, recovering the basic Method 5 selection weight.

### 17.2 The Three v3 Readouts as Bridge Components

The extended readout table of §10.2 now has a clean bridge interpretation:

**STEER (√ρ selection):** Implements the forward Schrödinger potential φ̂. The gradient ∂ₓφ̂ gives the osmotic drift u that enters the forward SDE. This is the "one half-iteration of Sinkhorn" interpretation: at each time step, the √ρ selection reweights the Brownian candidates by the forward potential, implementing the local marginal correction that the global IPFP algorithm does iteratively.

**WEIGH-forward (√ρ mean weight):** Accesses ∂²ₓφ̂/φ̂, the curvature of the forward Schrödinger potential. This is the quantum potential Q that enters the phase HJB equation — the backward component of the FBSDE.

**WEIGH-backward (ln ρ mean weight):** Accesses ∂²ₓ(ln ρ) = ∂²ₓ(ln φ̂ + ln φ), the combined curvature of both bridge potentials. Since Q gives ∂²ₓ(ln φ̂) separately, the difference yields ∂²ₓ(ln φ) — the curvature of the backward potential. The backward quantum potential Q̃ follows by algebraic reconstruction.

The three readouts thus provide a complete particle-level implementation of the FBSDE optimality system, with the forward FPE solved by particle propagation + selection and the backward HJB solved by the mean-weight readouts of the candidate cloud.

### 17.3 Anti-Diffusion Without Backward Simulation

The backward heat equation ∂ₜφ = −ν∇²φ is ill-posed as an initial-value problem — small perturbations grow exponentially. This is why direct simulation of the "−" channel (Method 7) was fragile.

But the information content of the backward equation — the curvature ∂²ₓ(ln φ) — is well-defined at every instant. It is encoded in the **current** density through the identity:

    ∂²ₓ(ln φ) = ∂²ₓ(ln ρ) − ∂²ₓ(ln φ̂) = (2/ν) u' − (1/ν²)(u² + νu') = u'/ν − u²/ν²

The M5v3 log-density readout computes (2/ν)u' = ∂²ₓ(ln ρ) directly. Combined with Q = −(ℏ²/2m)(g''/g), which gives (u² + νu')/ν², the backward curvature is fully determined.

The crucial insight: **the anti-diffusive equation's content is an instantaneous property of the density, not a time-evolved quantity.** The backward heat equation is ill-posed as a dynamical evolution, but its Laplacian (the only information needed for the coupling terms) is a well-conditioned local functional of ρ. The M5v3 algorithm extracts this functional through the GH quadrature of ln ρ, converting an ill-posed differential operation into a well-conditioned integral one.

This is the same mathematical trick that underlies the v2 WEIGH readout: the Gaussian mean-value property converts differentiation into averaging. What v3 adds is the recognition that applying this trick to **ln ρ** instead of **√ρ** gives access to the backward sector with no additional ill-conditioning.

### 17.4 The Sign of the Fisher Information

The Conforti–Pavon (2017) result places the relationship between Schrödinger bridges and quantum mechanics in a geometric framework. Both are Newton's second law on Wasserstein space W₂, with action functionals that differ only by the sign of the Fisher information term:

    Schrödinger bridge: A[ρ,v] = ∫∫ [½|v|²ρ + (ε²/8)|∇ ln ρ|²ρ] dx dt
    Quantum mechanics:  A[ρ,v] = ∫∫ [½|v|²ρ − (ν²/2)|∇ ln ρ|²ρ − Vρ] dx dt

The Fisher information ∫|∇ ln ρ|²ρ dx = 4∫|∇√ρ|² dx is the Dirichlet energy of √ρ.

For the Schrödinger bridge (positive Fisher information): the density has surface tension. Sharp features are penalised. The internal pressure opposes compression — the density resists being crumpled into interference fringes.

For quantum mechanics (negative Fisher information): the density has negative surface tension. Sharp features are energetically favourable. Interference fringes, tunnelling barriers, and all non-classical structure emerge spontaneously.

In both cases, the density traces a curve through W₂ with genuine inertia (the Benamou–Brenier kinetic energy ½∫|v|²ρ dx) and genuine internal forces (the functional derivative of the Fisher information term). The √ρ selection implements the internal force, and its sign (always positive — selection toward higher √ρ) is appropriate for quantum mechanics because the **dynamics** already account for the negative sign through the phase evolution (the Madelung/HJB equation).

---

## 18. Self-Consistency and Quantum Equilibrium

### 18.1 Method 5 Is Not Pilot-Wave Theory

In de Broglie–Bohm pilot-wave theory, the wavefunction ψ evolves according to the Schrödinger equation independently of the particle positions, and the particles are guided by the resulting velocity field v = ∇S/m. The density of particles ρ_particles need not equal |ψ|²; the "quantum equilibrium hypothesis" asserts that ρ_particles → |ψ|² under generic conditions (Valentini's H-theorem).

Method 5 is structurally different. There is no external wavefunction. The particles constitute the entire ontology:

- The density ρ is **defined as** the particle histogram (smoothed by KDE).
- The velocity field v is reconstructed from the particles' own phase values {Sᵢ} by local interpolation.
- The quantum potential Q is computed from the density estimated from the particles themselves.
- The osmotic drift u is enforced by √ρ selection on the actual particle density.

The system is therefore **always in equilibrium** in the Bohmian sense — ρ_particles = |ψ|² by construction, because "ψ" is defined by the particles. The relevant self-consistency question is different: **do the particle action values define a velocity field that is consistent with the density evolving according to the Madelung equations?**

### 18.2 The Self-Consistency Loop

The Method 5 dynamics involve a closed feedback loop:

    {Xᵢ, Sᵢ} → KDE → ρ(x) → √ρ(x) → selection → {X'ᵢ}
                                                              ↓
    {Sᵢ} → bin → S(x) → v(x) = ∂ₓS/m → classical push
                              ↓
                         ρ(x), v(x) → Q(x) → phase update → {S'ᵢ}

At each time step, the particles create the density, the density (through √ρ selection and GH mean weights) computes the osmotic drift and quantum potential, and these update the particles and their phases. The consistency condition is that after one step, the new (ρ', S') pair still satisfies the Madelung equations to O(dt²).

This is guaranteed by the proofs in v2 (Theorems 1 and 2): the √ρ selection induces exactly the osmotic drift u = ν∂ₓ ln ρ, and the mean weight gives exactly the quantum potential Q = −(ℏ²/2m)(∂²ₓ√ρ/√ρ), both to O(dt) per step. The error accumulates as a random walk, and with sufficient K (the number of GH quadrature points or stochastic candidates), the self-consistency is maintained to arbitrary precision over any finite time interval.

### 18.3 The (X, S) Representation

The choice to carry (X, S) per particle rather than (X, p) is motivated by several convergent considerations:

**Phase evolves deterministically along trajectories.** The action update dS = [½mv² − V − Q] dt involves no noise — the stochastic element of Method 5 enters only through the position update (selection step). In the (X, p) representation, momentum p = mv + mu acquires stochastic contributions from both the classical force and the osmotic velocity fluctuations, making both variables noisy.

**Phase requires fewer spatial derivatives.** The S-update needs only v(x) = ∂ₓS/m and Q(x) — one first derivative and one quantity available from the GH mean weight. The p-update would require both ∂ₓS and ∂ₓR (equivalently ∂ₓ ln ρ), the latter being the fragile log-density gradient that Method 5 was designed to avoid.

**Phase is the natural bridge variable.** In the Schrödinger bridge framework, the value function (backward potential) is S-like: it evolves by the HJB equation ∂ₜS + H(x, ∂ₓS) + Q = 0. The forward potential is R-like: it is the log-density, reconstructed from the particle histogram. The (S, R) = (S, (ℏ/2) ln ρ) representation maps directly onto Holland's (σ₊, σ₋) = (S + R, S − R).

---

## 19. Wasserstein Dynamics and the Newton Equation

### 19.1 The Density as a Particle in Wasserstein Space

In Otto's Riemannian geometry on Wasserstein space W₂, the density ρ(x, t) traces a curve through the infinite-dimensional manifold of probability distributions. The Wasserstein metric gives this space a Riemannian structure: the "distance" between two densities ρ₀ and ρ₁ is the optimal transport cost (the minimum work required to rearrange one into the other).

The kinetic energy of a "moving density" is:

    T = ½ ∫ |v(x)|² ρ(x) dx

This is the Benamou–Brenier formula. Every bit of probability ρ(x)dx at position x contributes its share to the total inertia. The density has genuine momentum ρv and genuine dynamical content.

### 19.2 Newton's Second Law on W₂

The Conforti–Pavon result shows that the Madelung equations are formally Hamilton's equations on Wasserstein space:

    Positions: ρ (the density)
    Momenta:   ρv (the momentum field)
    Hamiltonian: H[ρ, S] = ∫ [|∇S|²/(2m)]ρ dx + ∫ Vρ dx + (ℏ²/8m) ∫ |∇ρ|²/ρ dx

The last term is the Fisher information / quantum potential energy. The equations of motion are:

    ∂ₜρ = −∇·(ρ ∇S/m)
    ∂ₜS = −|∇S|²/(2m) − V + (ℏ²/2m)(∇²√ρ/√ρ)

Descending to particles, each tracer follows a second-order equation:

    m ẍᵢ = −∇V(xᵢ) + F_Q[{xⱼ}](xᵢ)

where the quantum force F_Q is the Wasserstein gradient of the Fisher information evaluated at xᵢ. This is structurally identical to classical N-body dynamics under a collective, density-dependent potential.

### 19.3 What Method 5 Computes

At each time step, Method 5 performs:

1. The **classical push** (v · dt) advances the density along its current Wasserstein momentum — applying the current velocity field to transport the distribution.

2. The **√ρ-weighted selection** applies the internal force — the functional derivative of the Fisher information — correcting the momentum so that the density follows the correct quantum trajectory in W₂ rather than the free-transport (classical) trajectory.

The two steps together implement a single step of Wasserstein Newton's law. The Wasserstein-geometric interpretation explains why the "old" density (evaluated before the classical push) is the correct selection weight: just as in a Verlet integrator, the force is evaluated at the current position, not the future position.

### 19.4 The Elastic Membrane Analogy

The Fisher information ∫|∇ ln ρ|²ρ dx = 4∫|∇√ρ|² dx is the Dirichlet energy of √ρ, which measures the "roughness" of the density. Thinking of ρ as an elastic membrane:

**Optimal transport** (no Fisher term): The membrane is perfectly floppy — it can be reshaped at zero internal cost. Only external forces (V) matter. Geodesics in W₂.

**Schrödinger bridge** (+Fisher): The membrane has surface tension. Deformations cost energy proportional to curvature. Sharp features are suppressed — the density relaxes toward smooth profiles.

**Quantum mechanics** (−Fisher): The membrane has negative surface tension. Sharp features are energetically favourable. Interference fringes spontaneously emerge when wave packets collide, and tunnelling through classically forbidden regions becomes possible because the negative internal energy can compensate the potential barrier.

In all three cases, the membrane has genuine inertia (kinetic energy ½∫|v|²ρ dx) and genuine internal forces (from δF/δρ). The √ρ selection couples the particle dynamics to the gradient structure of √ρ — the degree of freedom that carries the internal energy.

---

## 20. Related Literature: Derivative Estimation from Gaussian Perturbations

The mathematical technique underlying all three readouts — extracting derivative information from Gaussian-weighted function evaluations without explicit differentiation — has independent roots in several fields. We survey the principal traditions here and identify what is novel in the Method 5 synthesis.

### 20.1 The Mean Value Property and Harmonic Function Theory

The foundational identity E[f(x + η)] = f(x) + ½σ²∇²f(x) + O(σ⁴) is the infinitesimal version of the classical mean value property of harmonic functions: a function satisfying ∇²f = 0 is uniquely characterised by the property that its value at any point equals its average over any surrounding sphere (Evans, Partial Differential Equations, 2010, Ch. 2).

Kakutani (1944) made the decisive connection between the mean value property and Brownian motion: the solution to the Dirichlet problem for the Laplace equation at a point x equals the expected boundary value reached by a Brownian motion starting at x. This is the conceptual ancestor of all "walk on spheres" Monte Carlo PDE solvers.

The Method 5 WEIGH readout uses the inverse of this identity: given a known function g(x) = √ρ(x) (not harmonic, not solving a PDE), evaluate the Gaussian mean E[g(x + η)] and extract the Laplacian ∇²g from the deviation of the mean from the centre value. This reverses the Kakutani flow of information — from function values to differential operators, rather than from operators to boundary values.

### 20.2 Walk on Spheres and Monte Carlo PDE Solvers

Muller (1956) introduced the Walk on Spheres algorithm for solving the Dirichlet problem, exploiting the spherical mean value property to jump directly from one boundary sample to the next. Recent developments by Sawhney & Crane (2020, Monte Carlo Geometry Processing) and Sawhney, Seyb, Jarosz & Crane (2023, Walk on Stars) have extended this to general elliptic PDEs, including Poisson and screened Poisson equations, using the mean value property for non-harmonic problems.

The Method 5 connection is structural: both WoS and WEIGH extract Laplacian information from spherical (or Gaussian) averages. But WoS solves boundary value problems (unknown function, known operator), while WEIGH probes a known function (√ρ or ln ρ from KDE) to extract its unknown Laplacian. The mathematical identity is the same; the computational direction is reversed.

### 20.3 Zeroth-Order Optimisation

In the zeroth-order (derivative-free) optimisation literature, Nesterov & Spokoiny (2017, Random gradient-free minimization of convex functions) established the Gaussian smoothing framework: given an objective f(x), the gradient ∇f and Hessian ∇²f can be estimated from function-value queries at Gaussian perturbations:

    ∇f(x) ≈ (1/σ²) E[f(x + ση) · η]      (gradient estimator)
    ∇²f(x) ≈ (1/σ²) E[f(x + ση)(ηηᵀ − I)]  (Hessian estimator)

The mapping to Method 5 is:

    Optimisation                    Method 5
    ─────────────────────────      ──────────────────────────
    Objective f(x)                 √ρ(x) or ln ρ(x)
    Gradient estimator             STEER (selection → ∂ₓ ln √ρ)
    Hessian estimator              WEIGH (mean weight → ∂²ₓ√ρ/√ρ)
    Gaussian perturbations σ       Brownian noise σ√dt
    Perturbation dimension η       Candidate displacement ξ

The structural parallel is exact, but Method 5 uses selection (rank-order statistic) for the gradient rather than the linear estimator, and uses the scalar mean weight for the Laplacian rather than the tensor Hessian. These choices are specific to the quantum-mechanical application: selection preserves the diffusion coefficient, and only the Laplacian (trace of the Hessian) enters the quantum potential.

### 20.4 Stein's Method and Score Matching

Stein's identity (1972) states that for η ~ N(0, σ²I):

    E[f(x + η) · η] = σ² · E[∇f(x + η)]

This connects function-weighted perturbation averages to gradient expectations and underlies Hyvärinen's (2005) score matching: the score function ∇ ln p(x) of an unnormalised density can be estimated without knowing the normalisation constant, by matching the expected Laplacian ∇² ln p against a data-dependent statistic.

The Method 5 log-density readout (v3 Theorem 3) is closely related: it extracts ∂²ₓ(ln ρ) from the GH mean of ln ρ, which is equivalent to score-matching applied to the particle density. The connection is:

    ∂²ₓ(ln ρ) = ∂ₓ(score) = Stein operator applied to ρ

In the diffusion model literature (Song & Ermon 2019, Generative modeling by estimating gradients of the data distribution), the score ∇ ln p is estimated from noisy observations by denoising, and the Laplacian of the score controls the diffusion dynamics. M5v3's log-density readout provides the Laplacian of the score (= u' in our notation) directly, without a denoising step — the GH quadrature serves as an exact score-Laplacian estimator for smooth densities.

### 20.5 Diffusion Monte Carlo

The quantum chemistry community has long used diffusion Monte Carlo (DMC) methods (Anderson 1975) in which walkers diffuse and are selectively branched or killed based on the local potential energy. The quantum potential Q enters as part of the importance sampling weight, and its computation from fitted densities (Wyatt 2005) is notoriously unstable near nodes — the "node problem."

Method 5 eliminates this instability by never computing Q through spatial differentiation. The GH mean-weight approach extracts Q as an average property of the density landscape rather than a differential property. This is the same information (the Laplacian of √ρ divided by √ρ) but obtained through an integral operation that is well-conditioned even where the differential operation diverges.

### 20.6 What Is Novel in the Method 5 Synthesis

While individual pieces — derivative-free drift estimation, stochastic Laplacian extraction, harmonic mean-value properties — exist in the literatures surveyed above, their combination in Method 5 introduces several elements that appear to be new:

1. **The √ρ selection weight.** Neither ρ nor 1/ρ nor ln ρ, but specifically √ρ = exp(R/ℏ), chosen because it induces exactly the half-log-density gradient needed for the osmotic drift. This specific choice does not appear in the zeroth-order optimisation or score matching literatures.

2. **Dual readout from a single candidate cloud.** The recognition that selection (rank-order statistic → first derivative → osmotic drift → position) and mean weight (average statistic → second derivative → quantum potential → phase) are two independent, simultaneously available readouts from the same stochastic probe appears to be new. In zeroth-order optimisation, gradient and Hessian estimators are typically constructed from separate perturbation queries.

3. **Triple readout with the log-density channel.** The v3 addition of ln ρ as a third weight function, providing the osmotic divergence u' independently of Q, extends the dual-readout framework to access the full Holland bi-HJ structure. The combination (√ρ mean → Q, ln ρ mean → u', algebra → Q̃) is new.

4. **Physical interpretation as forward–backward coupling.** The identification of the steer readout with the forward Schrödinger potential and the weigh readout with the backward Schrödinger potential — connecting the triple-readout structure to Holland's bi-HJ theory and the FBSDE framework — is specific to the quantum-mechanical context and does not have a counterpart in the optimisation or score matching literatures.

5. **Elimination of all density differentiation from a quantum trajectory algorithm.** While individual pieces exist separately, their combination into a fully derivative-free quantum particle method — where the only remaining spatial derivative is ∂ₓS for the current velocity — appears to be new in the quantum trajectory literature.

---

## 21. Open Questions and Future Directions

### 21.1 The Ground State Challenge

The harmonic oscillator ground state remains the critical test case. It demands exact balance between the quantum and classical potentials over arbitrarily long times. The adaptive σ_kde controller (PI feedback maintaining dE/dt ≈ 0) achieves dramatic improvements for the free Gaussian and cat state but still hits ceilings for the ground state.

The v3 osmotic divergence readout provides a new diagnostic and potentially a new controller signal: if u'_measured ≠ −ω (the exact value), the density estimate is unreliable. The u' error could supplement or replace the energy error as a feedback signal for the adaptive bandwidth controller.

### 21.2 The ψ-KDE Pipeline

The ψ-KDE density estimator (NelsonMechanics_SchrodingerBridge_Swarmalator.md §§2–3) computes the wavefunction as a unified complex field ψ̂ = j_h/√n_h, where j_h is the complex current KDE and n_h is the standard density KDE. This approach naturally handles destructive interference at nodes (the complex current cancels) and has been validated as superior to separate density/phase estimation.

Integrating ψ-KDE with the v3 framework should improve the log-density readout: better density estimates → better ln ρ → more accurate u' → more accurate Q̃. The ground state, where the density is smooth and well-resolved, is the natural first test.

### 21.3 Symplectic Integration

The Wasserstein-Hamiltonian structure (§19) invites symplectic integration. A quantum Verlet algorithm would alternate half-steps of the quantum force (computed via GH mean weights) with full steps of the classical transport, preserving the symplectic structure of the Madelung equations on Wasserstein space. The tension between the deterministic requirements of symplectic integration and the stochastic character of the √ρ selection is an active area of investigation.

### 21.4 Higher Dimensions

All three v3 readouts (√ρ mean, ln ρ mean, algebraic reconstruction) generalise identically to d dimensions, replacing ∂²ₓ with the d-dimensional Laplacian ∇². The GH probe points become d-dimensional Gaussian samples, and the mean-weight formulae are unchanged. The mirror particle technique extends by reflecting along each coordinate axis independently.

The practical bottleneck in d > 3 is density estimation, not the M5v3 readouts themselves. The √ρ selection requires only pointwise ρ evaluation (not differentiation), which can use any density estimator: kernel methods, normalising flows, tree-based estimators, or tensor-network ansätze.

### 21.5 Multi-Particle Entanglement

For N particles in 1D (configuration space dimension D = N), the wavefunction ψ(x₁, ..., x_N) lives in a D-dimensional space. The Method 5 selection mechanism generalises coordinate-wise: K candidates per walker are drawn from a D-dimensional Gaussian, weighted by √ρ evaluated in the full configuration space. The induced osmotic drift is u_k = ν ∂_{x_k} ln ρ for each particle coordinate.

The entanglement structure enters through the density ρ(x₁, ..., x_N), which for entangled states is not a product of marginals. The selection weight √ρ then correlates the candidates across different particle coordinates, implementing the non-local osmotic coupling that distinguishes quantum mechanics from classical statistical mechanics.

---

## 22. Summary

Method 5 v3 provides a complete, derivative-free, particle-level implementation of Holland's bi-Hamilton–Jacobi system for quantum dynamics. The algorithm uses a single forward-evolving ensemble carrying (X, S) per particle, with three readouts from a shared Gauss–Hermite candidate cloud:

1. **√ρ selection** → forward osmotic drift u (v2, unchanged)
2. **√ρ mean weight** → forward quantum potential Q (v2, unchanged)
3. **ln ρ mean weight** → osmotic divergence u' (v3, new)
4. **Algebraic reconstruction** → backward quantum potential Q̃ = Q + ℏu' (v3, new)
5. **Mirror particles** → boundary-corrected KDE (v3, new)

The central conceptual advance is that the anti-diffusive (backward) information content of Holland's bi-HJ system is extracted through a **log-density weight** applied to the same GH quadrature nodes used for the forward readouts. The backward heat equation is never solved directly; its curvature content is captured by a well-conditioned integral operation that is machine-precision exact for Gaussian-like densities.

The algorithm connects to the Schrödinger bridge FBSDE framework: the √ρ selection implements the forward Schrödinger potential (osmotic drift), the √ρ mean weight accesses the forward HJB equation (quantum potential), and the ln ρ mean weight accesses the combined forward–backward curvature (osmotic divergence). Together, these three readouts provide particle-level access to all components of the Holland coupling without spatial differentiation of any field.

The density evolves as a genuine dynamical object on Wasserstein space, with collective momentum and inertia. The √ρ selection implements the internal force (the functional derivative of the Fisher information) that distinguishes quantum trajectories from classical optimal transport. The negative sign of the Fisher information in the quantum action — creating "negative surface tension" that favours sharp features — is responsible for all non-classical phenomena (interference, tunnelling, entanglement), and is faithfully reproduced by the self-consistent feedback loop between particle positions, density estimation, and phase evolution.

---

## 23. Complete References

### Quantum mechanics and stochastic mechanics

- Nelson, E. (1966). Derivation of the Schrödinger Equation from Newtonian Mechanics. *Phys. Rev.* 150, 1079.
- Nelson, E. (2012). Review of stochastic mechanics. *J. Phys.: Conf. Ser.* 361, 012011.
- Holland, P. (2021). Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory. arXiv:2111.09235.
- Hackebill, A. & Poirier, B. (2026). On Hydrodynamic Formulations of Quantum Mechanics and the Problem of Sparse Ontology. arXiv:2602.21106.
- Bohm, D. (1952). A suggested interpretation of quantum theory in terms of "hidden" variables. *Phys. Rev.* 85, 166–179.

### Schrödinger bridges, optimal transport, and FBSDEs

- Schrödinger, E. (1931). Über die Umkehrung der Naturgesetze. *Sitzungsber. Preuss. Akad. Wiss.*, 144–153.
- Zambrini, J.-C. (1986). Stochastic mechanics according to E. Schrödinger. *Phys. Rev. A* 33, 1532.
- Pavon, M. & Wakolbinger, A. (1991). On free energy, stochastic control, and Schrödinger processes. In *Modeling, Estimation and Control of Systems with Uncertainty*, 334–348.
- Pavon, M. (1995). Hamilton's principle in stochastic mechanics. *J. Math. Phys.* 36, 6774–6800.
- Beghi, A., Ferrante, A. & Pavon, M. (2001). How to steer a quantum system over a Schrödinger bridge. *Quantum Information Processing* 1, 183–206.
- Pavon, M. (2003). Quantum Schrödinger Bridges. In *Directions in Mathematical Systems Theory and Optimization*, LNCIS 286, 227–238.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2016). On the relation between optimal transport and Schrödinger bridges. *J. Optim. Theory Appl.* 174, 44–66.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge. *SIAM Review* 63, 249–313.
- Conforti, G. & Pavon, M. (2017). Extremal flows on Wasserstein space. arXiv:1712.02257.

### Mean value property, harmonic analysis, and Monte Carlo PDE solvers

- Kakutani, S. (1944). Two-dimensional Brownian motion and harmonic functions. *Proc. Imp. Acad.* 20, 706–714.
- Muller, M.E. (1956). Some continuous Monte Carlo methods for the Dirichlet problem. *Ann. Math. Stat.* 27, 569–589.
- Sawhney, R. & Crane, K. (2020). Monte Carlo geometry processing. *ACM Trans. Graph.* 39(4), Article 123.
- Sawhney, R., Seyb, D., Jarosz, W. & Crane, K. (2023). Walk on Stars. *ACM Trans. Graph.* 42(4), Article 80.
- Evans, L.C. (2010). *Partial Differential Equations.* 2nd ed.

### Zeroth-order optimisation and score matching

- Nesterov, Y. & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. *Found. Comput. Math.* 17, 527–566.
- Stein, C. (1972). A bound for the error in the normal approximation. *Proc. Sixth Berkeley Symp.* 2, 583–602.
- Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. *J. Mach. Learn. Res.* 6, 695–709.
- Song, Y. & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS 2019*. arXiv:1907.05600.
- Liu, Q., Lee, J. & Jordan, M. (2016). A kernelized Stein discrepancy for goodness-of-fit tests. *ICML 2016*. arXiv:1602.03253.
- Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural Comput.* 23, 1661–1674.

### Quantum trajectory methods

- Wyatt, R.E. (2005). *Quantum Dynamics with Trajectories: Introduction to Quantum Hydrodynamics.* Springer.
- Bittner, E.R. (2000). Quantum tunneling dynamics using hydrodynamic trajectories. *J. Chem. Phys.* 112, 9703.
- Anderson, J.B. (1975). A random-walk simulation of the Schrödinger equation: H₃⁺. *J. Chem. Phys.* 63, 1499.
- Lopreore, C.L. & Wyatt, R.E. (1999). Quantum wave packet dynamics with trajectories. *Phys. Rev. Lett.* 82, 5190.
- Trahan, C.J. & Wyatt, R.E. (2003). Quantum dynamics with trajectories: quantum interference and stability. *J. Chem. Phys.* 119, 7017.
- Kendrick, B.K. (2003). A new method for solving the quantum hydrodynamic equations of motion. *J. Chem. Phys.* 119, 5805.
---

### Project Knowledge Cross-References

- `Method5_Mathematical_Analysis_v2.md` — v2 dual-readout theory (Theorems 1–2), variance analysis, antithetic candidates (historical, not in repository)
- `Holland_Nelson_FokkerPlanck_Analysis.md` — Comparative analysis of Holland and Nelson, coupled SDE system, Fokker–Planck pair
- `FBSDE_SchrodingerBridge_Nelson_Holland.md` — Triangular relationship between FBSDEs, bridges, and bi-HJ
- `Method5_QA_Discussion.md` — Time symmetry, local Sinkhorn, Wasserstein dynamics
- `NelsonMechanics_SchrodingerBridge_Swarmalator.md` §§2–3 — ψ-KDE density estimator theory
- `HackebillPoirier2602_21106v1.pdf` — Sparse ontology analysis, CHV vs DHV classification
