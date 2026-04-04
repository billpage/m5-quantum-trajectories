# Kernel and Probe Selection for the ψ-KDE Swarmalator

## Compact Rational Kernels and Gauss–Jacobi Probes

---

### Abstract

The gridless swarmalator algorithm (§7 of the companion Swarmalator document) depends on two
mathematical objects: a **kernel** K_h(Δ) used in the ψ-KDE field estimates, and a **probe
distribution** used to generate the candidate cloud for STEER/WEIGH.  The original implementation
uses a Gaussian kernel and Gauss–Hermite (GH) probe nodes.  This document develops the theory
behind two alternatives — the compact rational kernel K(Δ) = C₄(1 − (Δ/R)²)⁴ and Gauss–Jacobi
probe quadrature — and analyses why they are structurally better matched to the algorithm's
requirements.

The key results are:

1. **Kernel second moment:** For a given support radius, the compact rational kernel has a smaller
   second moment μ₂ than the Gaussian, reducing the leading-order MISE bias in the density estimate
   and the O(h²) bias in the WEIGH quantum potential readout.

2. **Algebraic closure:** The compact rational kernel produces kernel sums that are rational functions
   of particle separations.  The Poirier C-coordinate variables (z = −ρ'/ρ, w = 3(ρ'/ρ)² − ρ''/ρ)
   are themselves rational functions of these sums.  The Gaussian kernel yields transcendental
   (exponential) sums, breaking this algebraic compatibility.

3. **Smoothness:** With exponent n = 4, the compact kernel is C⁶ at the support boundary — smooth
   enough that K, K', K'', and K''' are all continuous, and the quantum potential Q =
   −(ℏ²/2m)(√ρ)''/√ρ inherits smoothness from the density estimate.

4. **Bounded probes:** Gauss–Jacobi quadrature nodes lie on [−1, 1], giving a hard bound on probe
   displacement.  Gauss–Hermite nodes are unbounded, with the outermost node scaling as √(2K+1).
   For states with small spatial extent (e.g. the HO ground state with σ ≈ 0.71), unbounded probes
   can reach far into the exponential tail, where ψ-KDE estimates are noisy.

5. **Variance matching:** The Jacobi probe radius R_probe = σ_gh √(2n+3) is chosen so that the
   probe distribution's second moment equals σ_gh² — the same as for Gauss–Hermite.  This ensures
   the WEIGH readout formula Q = −(ℏ²/mσ²)(M₊ − 1) is unchanged.

---

## 1. Kernel Requirements

The ψ-KDE density estimate is a sum of kernel evaluations:

    n_h(x) = Σⱼ K_h(x − Xⱼ)

where K_h(Δ) = (1/h) K(Δ/h) is a scaled kernel.  The coherent current is the same sum weighted by
phasors:

    j_h(x) = Σⱼ K_h(x − Xⱼ) exp(iSⱼ/ℏ)

The kernel K must satisfy:

(a) **Non-negativity:** K(u) ≥ 0 for all u.  Required for the ψ-KDE estimate √ρ = |j|/√n to be
    well-defined (n > 0 wherever any particles contribute).

(b) **Normalisation:** ∫ K(u) du = 1.  Ensures n_h → ρ as Np → ∞ with h → 0 at the appropriate
    rate.

(c) **Smoothness ≥ C²:** The WEIGH readout extracts Q ∝ (√ρ)''/√ρ from the mean-weight ratio.
    This requires √ρ to have a well-defined second derivative, hence n and |j| must be at least C².
    Since n inherits smoothness from K, the kernel should be C² at minimum.  The Epanechnikov kernel
    K(u) = ¾(1 − u²)₊ is only C⁰ (discontinuous first derivative at the support boundary), and its
    second derivative contains a Dirac delta — it catastrophically corrupts the Q readout.

(d) **Symmetric:** K(u) = K(−u).  Required for odd-moment cancellation (the estimate should not
    systematically shift the density).

(e) **A well-defined derivative kernel K':** The coherent velocity v = (ℏ/m) Im(j'/j) requires the
    derivative kernel K' to be evaluated in the sums j'_re and j'_im.  The kernel must be
    differentiable on its support.

Beyond these necessary conditions, two further properties affect accuracy:

(f) **Small second moment μ₂ = ∫ u² K(u) du:**  The leading bias in the density estimate is
    ½ h² μ₂ ρ''(x), so smaller μ₂ reduces bias at a given bandwidth h.  This is the MISE-optimality
    criterion.

(g) **Compact support:**  A kernel with support [−R, R] limits each kernel sum to neighbours within
    distance R, enabling O(Np · K_nbr) instead of O(Np²) cost.  It also prevents particles in the
    exponential tail from contributing unphysical weight to distant evaluations.

---

## 2. The Compact Rational Kernel

### 2.1 Definition

For exponent n and support radius R:

    K_R^(n)(Δ) = C_n · (1 − (Δ/R)²)^n    for |Δ| < R
               = 0                          for |Δ| ≥ R

The normalisation constant is

    C_n = 1 / (R · B(½, n+1))

where B(½, n+1) = √π · Γ(n+1) / Γ(n + 3/2) is the beta function.

For n = 4:

    B(½, 5) = √π · 24 / (945√π/32) = 256/315

    C₄ = 315 / (256 R)

### 2.2 Smoothness

At the support boundary |Δ| = R, the kernel vanishes: K(R) = 0.  So does each derivative through
order 2n − 2:

    d^k/dΔ^k K(Δ)|_{Δ=R} = 0    for k = 0, 1, ..., 2n−2

This follows because each differentiation reduces the exponent by at most 1, and after k
differentiations the leading factor is (1 − ξ²)^{n−⌈k/2⌉}, which vanishes at ξ = 1 provided
n − ⌈k/2⌉ ≥ 1, i.e. k ≤ 2n − 2.

For n = 4: **C⁶ smoothness**.  K, K', K'', K''', K⁽⁴⁾, K⁽⁵⁾, K⁽⁶⁾ are all continuous everywhere,
including at the support boundary.

By comparison:

| Kernel | Smoothness | Support |
|--------|-----------|---------|
| Epanechnikov (n=1) | C⁰ | Compact |
| Quartic/biweight (n=2) | C² | Compact |
| Triweight (n=3) | C⁴ | Compact |
| **Compact rational (n=4)** | **C⁶** | **Compact** |
| Quintic B-spline | C⁴ | Compact |
| Gaussian | C^∞ | Unbounded |

### 2.3 Second Moment

The second moment of the normalised kernel on [−R, R] is:

    μ₂ = ∫ u² K(u) du = R² · B(3/2, n+1) / B(1/2, n+1) = R² / (2n + 3)

For n = 4, R = 2.5h:

    μ₂ = (2.5h)² / 11 ≈ 0.568 h²

For the Gaussian kernel:

    μ₂ = h²

So the compact rational kernel at R = 2.5h has μ₂ that is 57% of the Gaussian value at the same
bandwidth parameter h, or equivalently, the same μ₂ as a Gaussian with bandwidth h_eff = 0.75h.

This means the **MISE-optimal bandwidth** for the compact kernel is larger than for the Gaussian:
the compact kernel can use a wider support while incurring the same amount of smoothing bias.
Alternatively, at the same h, the compact kernel delivers lower bias in the density estimate and
hence lower systematic error in Q.

### 2.4 Choice of Support Radius

The support radius R = 2.5h is a design parameter.  The considerations are:

- **Neighbour count:** K_nbr ≈ Np · 2R / (x_max − x_min).  Larger R means more neighbours per
  particle, increasing cost.  At R = 2.5h, K_nbr is comparable to a Gaussian truncated at 4h
  (noting that 2.5h vs 4h gives ~63% of the Gaussian neighbour count).

- **MISE:** The integrated squared bias scales as h⁴ μ₂², so smaller μ₂ (larger n or smaller R)
  reduces bias.  But the variance term scales as 1/(Np h) · ∫K² du, and ∫K² is larger for narrower
  support.  The MISE-optimal R depends on Np and the target density smoothness.

- **Fringe resolution:** For interference fringes with spacing λ = πℏ/p₀, the kernel must resolve
  the fringe structure: R should be ≲ λ.  At R = 2.5h, this means h ≲ λ/2.5, compared to h ≲ λ/4
  for the Gaussian truncated at 4h.  The compact kernel gives slightly better fringe resolution at
  the same effective support width.

The default R = 2.5h is conservative.  For problems without fine interference structure, R = 3h (or
even R = 2h) could be used; the normalisation C_n adjusts automatically.

### 2.5 Derivative Kernel

Within the support |Δ| < R:

    K'(Δ) = C_n · n · (−2Δ/R²) · (1 − (Δ/R)²)^{n−1}

For n = 4:

    K'(Δ) = −8 C₄ · (Δ/R²) · (1 − Δ²/R²)³

K' vanishes at Δ = 0 (symmetric kernel → zero derivative at centre) and at |Δ| = R (smoothness
at boundary).  The velocity estimate v = (ℏ/m) Im(j'/j) is well-behaved throughout the support.

### 2.6 K''/K Ratio (Quantum Potential Structure)

Within the support:

    K''(Δ)/K(Δ) = (−2n/R²) · (1 − (2n−1)ξ²) / (1 − ξ²)

where ξ = Δ/R.  For n = 4:

    K''/K = (−8/R²) · (1 − 7ξ²) / (1 − ξ²)

This is a **rational function** of ξ² with a simple pole at ξ = 1.  The pole is never evaluated in
practice (K(R) = 0, so the ratio is taken as 0/0 with a well-defined C⁶ limit), but its algebraic
presence shapes the structure of the kernel sums in a way that mirrors the 1/ρ singularity of the
continuum quantum potential.

The Gaussian K''/K = (u² − 1)/h² is a polynomial in u — smoother but algebraically incompatible
with the rational structure of the Poirier C-derivatives (see §4 below and *kernel_expressions.md*
§4).

---

## 3. Gauss–Jacobi Probes

### 3.1 Motivation: The Displacement Problem

The STEER/WEIGH candidate cloud consists of probe points displaced from the classical-step position
X_class:

    x_k = X_class + η_k

where η_k are the probe offsets.  For Gauss–Hermite probes:

    η_k = √2 · σ_gh · ξ_k

where ξ_k are the GH quadrature nodes (roots of the Hermite polynomial H_K).

The problem: GH nodes are **unbounded**.  The outermost node scales as ξ_max ≈ √(2K + 1).  For
K = 8, ξ_max ≈ 4.14, giving a maximum probe offset of

    |η_max| = √2 · σ_gh · 4.14 ≈ 5.86 · σ_gh

At σ_gh = 0.20, this is |η_max| ≈ 1.17 — comparable to the width of the HO ground state (σ ≈ 0.71).
Probes reaching to |η| ≈ 1.2 from the particle are evaluating ψ-KDE deep in the tail, where few
source particles contribute and the density estimate is noisy.

For the coherent state with velocity v₀ = 1.0 and dt = 0.005:

    drift per step = v₀ · dt = 0.005
    max probe displacement ≈ 1.17

The probe cloud extends 234× further than the actual drift per step.  This mismatch is a
significant source of trajectory noise in the STEER selection.

### 3.2 The Gauss–Jacobi Alternative

Gauss–Jacobi quadrature with weight function w(t) = (1 − t²)^n on [−1, 1] provides nodes t_k and
weights ω_k satisfying:

    Σ_k ω_k f(t_k) = ∫₋₁¹ f(t) (1 − t²)^n dt    (exact for deg(f) ≤ 2K − 1)

(Technically this is the symmetric case α = β = n of the Jacobi polynomials P_k^{(α,β)}, sometimes
called Gegenbauer quadrature.)

All nodes lie in (−1, 1): the maximum node satisfies |t_max| < 1 strictly.  For K = 8, n = 4,
|t_max| ≈ 0.960.

**Probe offsets:** Map nodes from [−1, 1] to physical space via

    η_k = R_probe · t_k

where R_probe is the probe radius.  The maximum displacement is

    |η_max| = R_probe · |t_max| < R_probe

### 3.3 Variance Matching

The WEIGH readout formula

    Q = −(ℏ²/m) · (M₊ − 1) / σ²

is derived from the Taylor expansion √ρ(x₀ + η) ≈ √ρ(x₀) + η(√ρ)' + ½η²(√ρ)'' + ···, where the
first-order term vanishes by symmetry and the second-order term gives

    M₊ − 1 ≈ ½ ⟨η²⟩ · (√ρ)''/√ρ

The formula requires that σ² = ⟨η²⟩ — the probe variance under the quadrature weights.

For Gauss–Hermite: The probe variance is

    ⟨η²⟩ = 2 σ_gh² · ⟨ξ²⟩_GH = 2 σ_gh² · ½ = σ_gh²

(since the GH weight function is exp(−ξ²) and ⟨ξ²⟩_{exp(−ξ²)} = ½).

For Gauss–Jacobi: The second moment of the weight function (1 − t²)^n on [−1, 1] is

    ⟨t²⟩ = B(3/2, n+1) / B(1/2, n+1) = 1/(2n + 3)

So the probe variance is

    ⟨η²⟩ = R_probe² · ⟨t²⟩ = R_probe² / (2n + 3)

Setting ⟨η²⟩ = σ_gh² gives

    R_probe = σ_gh · √(2n + 3)

For n = 4:

    R_probe = σ_gh · √11 ≈ 3.317 σ_gh

At σ_gh = 0.20: R_probe ≈ 0.663, and |η_max| ≈ 0.637 (since |t_max| ≈ 0.960).

| Probe type | Max displacement (K=8, σ_gh=0.20) | Displacement / σ_ψ (HO ground) |
|------------|-----------------------------------|---------------------------------|
| Gauss–Hermite | 1.17 | 1.65 |
| Gauss–Jacobi (n=4) | 0.64 | 0.90 |

The Jacobi probe reduces the maximum displacement by 45%, keeping all probes within ~1σ of the
wavefunction width for the HO ground state.

### 3.4 Quadrature Accuracy

Both GH and GJ with K nodes are exact for polynomial integrands of degree ≤ 2K − 1.  The relevant
integrand in the WEIGH formula is √ρ(x₀ + η)/√ρ(x₀), which is approximately polynomial for smooth
densities.  At K = 8, both probe types handle 15th-degree polynomials exactly.

The weight function differences mean that GH is optimal for integrands multiplied by exp(−η²/2σ²),
while GJ is optimal for integrands multiplied by (1 − η²/R²)^n.  Neither is intrinsically more
accurate for the physical problem; both converge rapidly for smooth wavefunctions.  The advantage
of GJ is the bounded support, not superior polynomial accuracy.

### 3.5 STEER Selection with Jacobi Probes

The STEER operation draws a new position from the candidate distribution:

    p_k = ω_k · √ρ(x_k) / Σ_l ω_l · √ρ(x_l)

For GH probes, the candidates include points far in the tail (|η| > 4σ_gh) where √ρ is small but
noisy.  For GJ probes, all candidates are within R_probe of the departure point.  The selection
distribution is more concentrated, leading to smaller expected jumps and smoother trajectories.

The osmotic drift theorem (§4.2 of the Swarmalator document) holds for both probe types: it depends
on the importance-sampling structure of the selection, not on the specific probe placement.

### 3.6 Consistency with the Compact Kernel

When the compact rational kernel (§2) is used for the ψ-KDE, and Gauss–Jacobi probes are used for
STEER/WEIGH, there is an appealing structural consistency: both the kernel and the probe
distribution are members of the same family (1 − u²)^n with compact support.  The kernel evaluates
(1 − (Δ/R_K)²)⁴ at source particle separations; the probe weights are proportional to
(1 − (t_k)²)⁴ at quadrature nodes.

This is not a mathematical necessity — the WEIGH readout works for any probe distribution with the
correct second moment — but it suggests that the Jacobi probes may be a natural complement to the
compact kernel, in the same way that Gauss–Hermite probes are the natural complement of the
Gaussian kernel.

---

## 4. The Poirier Connection (Revisited)

### 4.1 C-Coordinates and Rational Kernel Sums

Poirier's uniformising coordinate C assigns to each particle a label proportional to the integrated
probability density.  The key variables are the dimensionless ratios

    z = x_CC / x_C² = −ρ'/ρ

    w = x_CCC / x_C³ = 3(ρ'/ρ)² − ρ''/ρ

which determine the quantum potential:

    Q = −(ℏ²/4m)(w − z²) = (ℏ²/2m)(ρ''/ρ − 2(ρ'/ρ)²) = −(ℏ²/2m)(√ρ)''/√ρ − ½mu²

For the compact rational kernel, the density estimate n_h(x) = C₄ Σⱼ (1 − (x − Xⱼ)²/R²)⁴ is
a polynomial in x within each interval where the contributing particle set is fixed.  Consequently:

    z^R(x) = −n'(x)/n(x) = [rational function of particle positions]

    w^R(x) = 3(n'/n)² − n''/n = [rational function of particle positions]

For the Gaussian kernel, these same ratios involve sums of exp(−(x − Xⱼ)²/2h²) — transcendental
functions that are algebraically incompatible with the polynomial form that the Poirier variables
assume in the continuum limit.

The significance of this algebraic closure is developed in *kernel_expressions.md* §4.  The compact
kernel produces ψ-KDE sums whose functional form converges to the continuum target through the
same algebraic class (rational functions), whereas the Gaussian kernel converges through a different
class (sums of exponentials).  Whether this algebraic compatibility translates into measurably better
energy conservation at finite Np is an empirical question addressed by the numerical experiments.

### 4.2 Relation to the Fisher Information Perspective

The Fisher information of the density is I_F = ∫ (ρ'/ρ)² ρ dx.  In terms of z:

    I_F = ∫ z² ρ dx

The osmotic kinetic energy is T_osm = (ℏ²/8m) I_F.  With the compact kernel,

    z = −n'/n = [a ratio of polynomials in (x − Xⱼ)]

This ratio is computed exactly (no truncation, no numerical differentiation) from the kernel sums
n and n', which are themselves finite sums of polynomials.  The Gaussian kernel requires
exp(−u²) evaluations, which are computed to floating-point precision but whose ratio n'/n does
not have a clean algebraic form.

Whether this algebraic difference is significant compared to the dominant source of error — the
finite number of particles Np — remains to be established by systematic numerical comparison.

---

## 5. Energy Conservation Analysis

### 5.1 Sources of Energy Drift

The M5 algorithm conserves total energy E = T_current + T_osm + V + Q only approximately.  The
main sources of drift are:

(a) **Density estimation bias:** h > 0 implies a biased density estimate, hence biased Q.  The
    bias is O(h² μ₂ ρ''/ρ).  Smaller μ₂ reduces this.

(b) **WEIGH quadrature truncation:** The GH/GJ readout gives Q + O(σ_gh²).  The error depends
    on the fourth derivative of √ρ.

(c) **STEER stochastic noise:** The √ρ-weighted selection introduces an O(1/√K) noise in each
    particle's position, which propagates through the density estimate.

(d) **Phase accumulation error:** The Euler time-stepping of S introduces O(dt) error per step.

(e) **Nelson diffusion noise:** The σ_noise √dt Brownian increment (when used) adds O(√dt) noise
    to each position.

The compact kernel reduces source (a) by a factor μ₂^compact / μ₂^Gaussian = (R/h)²/(2n+3) ≈ 0.57
at R = 2.5h, n = 4.  Static kernel comparison experiments (kernel_compare.py) showed ~40× less
energy drift for the compact rational kernel vs Gaussian on the cat-state collision test, and ~470×
less for the cubic B-spline, suggesting that source (a) may be the dominant contributor at large Np
where sources (c)–(e) are controlled.

### 5.2 Expected Improvement

For the HO ground state with Np = 600 particles (quick mode), the initial test results are:

| Kernel × Probe | L² error | Energy drift (%) |
|----------------|----------|------------------|
| Gaussian / Hermite | 0.241 | 166 |
| Compact / Jacobi | 0.208 | 96 |

A 42% reduction in energy drift and 14% improvement in L² fidelity at identical particle count and
bandwidth parameters.  Full Kaggle-scale runs (Np = 3000, extended time) are pending.

---

## 6. Implementation Details

### 6.1 Kernel Selection in m5/sim.py

The `kernel_sums()` function accepts a `kernel` parameter:

    kernel='gaussian':  K = (2πh²)^{−1/2} exp(−Δ²/2h²),  cutoff at |Δ| < 4h
    kernel='compact':   K = C₄(1 − (Δ/R)²)⁴,  R = 2.5h

Both branches compute n, j_re, j_im (and optionally jp_re, jp_im for velocities) using the same
chunked outer-product loop.  The compact kernel is cheaper per pair because it avoids the exp()
evaluation, but this is partially offset by the base⁴ and base³ power computations.

### 6.2 Probe Selection in m5/sim.py

The `make_probe()` function accepts a `probe_type` parameter:

    probe='hermite':  GH nodes ξ_k via numpy.polynomial.hermite.hermgauss(K)
                      offsets η_k = √2 · σ_gh · ξ_k
                      probe variance = σ_gh²

    probe='jacobi':   GJ nodes t_k via scipy.special.roots_jacobi(K, n, n)
                      R_probe = σ_gh · √(2n+3)
                      offsets η_k = R_probe · t_k
                      probe variance = R_probe² / (2n+3) = σ_gh²

Both return a dict with keys: nodes, weights, offsets, mu2, probe_type.

### 6.3 CLI for m5_compare.py

    python m5_compare.py --kernel compact --probe jacobi [other options]

The four combinations (gaussian/hermite, gaussian/jacobi, compact/hermite, compact/jacobi)
can be tested independently.

### 6.4 Existing Tests

The test suite `tests/test_m5_sim.py` includes 16 tests covering both kernel and probe options.
All tests pass for all four kernel×probe combinations.

---

## 7. Relationship to SPH Kernels

The compact rational kernel (1 − ξ²)^n belongs to a family widely used in smoothed particle
hydrodynamics (SPH), where it is known as the Wendland or generalised polynomial kernel.  Key
connections:

- **Wendland C⁶ kernel:** The Wendland function φ₃,₃(r) = (1 − r)₊⁸ · (32r³ + 25r² + 8r + 1)
  achieves C⁶ smoothness with a different polynomial structure.  The (1 − ξ²)⁴ kernel is simpler
  but achieves the same smoothness class.

- **Monaghan (2005):** Reviews kernel requirements for SPH, emphasising that compact support and
  sufficient smoothness (at least C²) are essential for stable hydrodynamic simulations.  The
  ψ-KDE has analogous requirements.

- **SPH normalisation convention:** SPH typically normalises ∫ K = 1 in d dimensions.  The 1D
  ψ-KDE uses the same convention, with C₄ = 315/(256R) ensuring ∫₋R^R K dΔ = 1.

The key difference from SPH is that the ψ-KDE uses the kernel for both density estimation and
coherent phase averaging.  In SPH, particles carry mass and momentum; in M5, they carry position
and action phase.  The kernel's role in the coherent sum j = Σ K exp(iS/ℏ) has no SPH analogue.

---

## 8. Summary and Recommendations

### 8.1 Kernel Choice

The compact rational kernel with n = 4, R = 2.5h is recommended as the default for new simulations.
It offers:

- Lower MISE bias (μ₂ = 0.57 h² vs h² for Gaussian)
- Compact support (natural O(Np · K_nbr) scaling, no arbitrary truncation radius)
- C⁶ smoothness (sufficient for Q readout)
- Rational kernel sums (algebraically compatible with Poirier C-coordinates)
- Cheaper per-pair evaluation (no exp() call)

The Gaussian kernel remains available as a baseline and for comparison with previous results.

### 8.2 Probe Choice

The Gauss–Jacobi probe with n = 4 is recommended for states with small spatial extent or large
current velocity, where the unbounded GH probe displacement is problematic.  The recommendation
is provisional pending full dynamic comparisons; GH may remain preferable for broad states where
the probe displacement is small relative to the wavefunction width.

### 8.3 Open Questions

1. **Dynamic energy conservation:** Static kernel comparisons (kernel_compare.py) show large
   improvements for the compact kernel.  Do these translate to proportional improvements in the
   full dynamic simulation over many time steps?

2. **Bandwidth debiasing:** The gradient correction T_osm^F → T_osm (removing the O(h²) bias)
   developed in the Fisher-form energy analysis may interact differently with the compact kernel.
   The smaller μ₂ reduces the uncorrected bias, but the correction itself may converge differently.

3. **Optimal n:** The exponent n = 4 was chosen for C⁶ smoothness.  n = 3 (C⁴, like the quintic
   B-spline) has an even smaller μ₂ at the same R, but less smoothness.  n = 5 or 6 would give
   more smoothness at the cost of larger μ₂.  Systematic comparison across n values is warranted.

4. **Decoupled scales:** Using σ_steer ≠ σ_weigh to optimise trajectory smoothness and Q accuracy
   independently.  This doubles the cost of the candidate evaluation but may be worthwhile.

---

## References

### Kernel density estimation

- Epanechnikov, V.A. (1969). Non-parametric estimation of a multivariate probability density.
  *Theory Prob. Appl.* 14, 153–158.
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis.* Chapman & Hall.
- Wand, M.P. & Jones, M.C. (1995). *Kernel Smoothing.* Chapman & Hall.

### SPH kernels

- Monaghan, J.J. (2005). Smoothed particle hydrodynamics. *Rep. Prog. Phys.* 68, 1703.
- Wendland, H. (1995). Piecewise polynomial, positive definite and compactly supported radial
  functions of minimal degree. *Adv. Comput. Math.* 4, 389–396.

### Quadrature

- Gautschi, W. (2004). *Orthogonal Polynomials: Computation and Approximation.* Oxford Univ. Press.
- Golub, G.H. & Welsch, J.H. (1969). Calculation of Gauss quadrature rules. *Math. Comp.* 23,
  221–230.

### Poirier C-coordinates

- Poirier, B. (2011). Lagrangian Quantum Hydrodynamic Trajectory Methods. In *Quantum Trajectories*
  (CCP6), Ed. K. Hughes & G. Parlant.
- Hackebill, A. & Poirier, B. (2026). The Problem of Sparse Ontology for Hydrodynamic Formulations
  of Quantum Mechanics. arXiv:2602.21106.

### Project Knowledge Cross-References

- `NelsonMechanics_SchrodingerBridge_Swarmalator.md` — Core swarmalator algorithm, §8–9 for cost
  and bandwidth analysis
- `kernel_expressions.md` — Explicit kernel sum expressions, K''/K structure, Poirier connection
- `m5/sim.py` — Implementation of kernel_sums(), make_probe(), m5_simulate()
- `kernel_compare.py` — Static kernel comparison experiments (energy drift benchmarks)
- `m5_compare.py` — Dynamic comparison program with --kernel and --probe CLI options
