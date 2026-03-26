# Complex Quantum Trajectories and the M5/Schrödinger Bridge Connection

## Yang & Han's Complex Mechanics in Context

---

## 1. What Yang & Han Do

Yang & Han (Found. Phys. 50, 960–976, 2020) start from the standard polar decomposition ψ = R_B · e^{iS_B/ℏ} but then take an unconventional step: they write ψ = e^{iW/ℏ} where W is a **complex-valued** Hamilton's principal function, turning Schrödinger's equation into a single **complex** Hamilton–Jacobi equation (the QHJE):

    ∂W/∂t + (1/2m)(∂W/∂x)² + V − (iℏ/2m)(∂²W/∂x²) = 0

This is *not* of classical HJ type — it retains a second-order derivative term (the last term, which contains the quantum potential). The key move is to **complexify the coordinate**: let x → z = x + iy, and solve the QHJE in the complex z-plane.

The complex velocity is:

    ż = (1/m)(∂W/∂z)

which is deterministic in the complex plane. To this they add a **complex Wiener noise** dw_z to get a complex stochastic differential equation:

    dz = v_c(z,t) dt + √(ℏ/m) dw_z

where v_c is derived from an **optimal guidance law** obtained by solving a Hamilton–Jacobi–Bellman equation — this is explicitly a stochastic optimal control formulation.

### The Three Probability Distributions

The central result is a **hierarchy of three probability distributions** obtained from the complex probability ρ_c(t, x, y):

| Distribution | How obtained | What it reproduces |
|---|---|---|
| ρ_c(t, x, 0) | Intersections of complex trajectories with real axis (y = 0) | Quantum probability \|ψ\|² |
| ρ_c(t, x) = ∫ρ_c(t,x,y) dy | Marginal over imaginary part (projection onto real axis) | Classical probability (no nodes) |
| ρ_c(t, x, y) | Full complex-plane density | Complex probability |

The **nodal issue** is resolved because: at a node of ψ, the quantum probability |ψ(x)|² = 0, but complex trajectories **never actually pass through the real axis at these points**. The zero density at y = 0 arises because the trajectories "orbit around" the node in the complex plane. Meanwhile, the projection (marginal) ρ_c(t,x) is nonzero everywhere, including at nodes, and approaches the classical distribution for large quantum numbers.

---

## 2. Holland's Footnote and Its Significance

Holland (2021), in §2 of his bi-HJ paper, explicitly acknowledges and then distances himself from this complex trajectory program. His critique has two prongs:

**First**, the complex QHJE is still second-order in spatial derivatives — it is "not of the classical type." Holland's bi-HJ system, by contrast, produces two *first-order* (genuine) Hamilton–Jacobi equations for σ±. This is a structural advantage: classical HJ theory (characteristics, Lagrangian mechanics, canonical transformations) applies directly to Holland's formulation.

**Second**, Holland flags the **probability current problem**: complexifying the coordinates creates difficulties in defining a locally conserved probability current in the complex plane. Chou & Wyatt (2008) showed that the naive extension of |ψ|² to complex space does not satisfy a standard continuity equation. Yang & Han partially address this by constructing ρ_c from the complex Fokker–Planck equation, but the interpretive status remains disputed.

Holland's implicit position is: *we don't need to complexify coordinates because the bi-HJ decomposition already provides the extra degree of freedom (the backward action σ₋) that the complex coordinate was introduced to supply.*

---

## 3. The Structural Parallels

Despite Holland's dismissal, the mathematical parallels between these approaches are striking and illuminate the M5 framework.

### 3.1 Decomposition of the Wave Function

All approaches decompose ψ into two coupled real (or one complex) fields:

| Approach | Decomposition | Fields |
|---|---|---|
| Standard (Madelung) | ψ = √ρ · e^{iS/ℏ} | (ρ, S) — density and phase |
| Holland bi-HJ | σ± = S ± (ℏ/2) ln ρ | (σ₊, σ₋) — forward and backward actions |
| Yang–Han complex | W = S_B + iR_B = S − (iℏ/2) ln ρ | (Re W, Im W) — real and imaginary action |
| Nelson stochastic | b± = v ± u = ∇σ±/m | (b₊, b₋) — forward and backward drifts |
| Schrödinger bridge | ρ = φ̂ · φ, with φ̂ = e^{σ₊/ℏ}, φ = e^{−σ₋/ℏ} | (φ̂, φ) — forward and backward potentials |

The crucial observation: **Holland's σ± and Yang–Han's (Re W, Im W) encode the same information**. Specifically:

    σ₊ = S + (ℏ/2) ln ρ = Re(W) − Im(W)
    σ₋ = S − (ℏ/2) ln ρ = Re(W) + Im(W)

so the complex action W and the Holland pair (σ₊, σ₋) are related by a simple linear transformation. The "extra dimension" that Yang–Han access by going to z = x + iy is mathematically equivalent to the "extra equation" that Holland accesses through his second HJ equation.

### 3.2 The Product Structure and Schrödinger Bridges

**This is the connection Bill noticed.** In the Schrödinger bridge framework:

    ρ(x, t) = φ̂(x, t) · φ(x, t)

where φ̂ = exp(σ₊/ℏ) satisfies a forward heat equation and φ = exp(−σ₋/ℏ) satisfies a backward heat equation. The density is a **product** of forward and backward potentials.

Yang & Han's construction has an analogous product structure, but viewed differently. Their complex probability ρ_c(t, x, y) lives on the full complex plane, and the quantum probability is obtained by **evaluating on a slice** (y = 0):

    |ψ(x)|² = ρ_c(t, x, 0)

Meanwhile, in polar form ψ = R · e^{iS/ℏ}, we have |ψ|² = R² = exp(2R_B/ℏ) where R_B is related to the imaginary part of the complex action. Writing this out:

    |ψ|² = exp[(σ₊ − σ₋)/ℏ] = exp(σ₊/ℏ) · exp(−σ₋/ℏ) = φ̂ · φ

So the "product" that Yang & Han work with — the density as a product arising from the real and imaginary parts of the complex action — is **exactly the Schrödinger bridge factorization** written in different variables. The "slice at y = 0" operation in Yang–Han corresponds to the "evaluate the product φ̂ · φ" operation in the bridge framework.

The classical limit is also illuminating:
- In Schrödinger bridges: as ε → 0, the bridge collapses to optimal mass transport, and ρ → δ-function on classical trajectories
- In Yang–Han: as ℏ → 0, complex trajectories collapse to the real axis, and ρ_c(t,x,y) → δ(y) · ρ_classical(x)
- Both limits eliminate the "quantum width" (the imaginary excursion / the osmotic velocity / the Fisher information term)

### 3.3 The Optimal Control Connection

Yang & Han derive their complex SD equation from an **optimal guidance law** via a Hamilton–Jacobi–Bellman equation. This is not incidental — it is the same mathematical structure as the Schrödinger bridge:

- **Schrödinger bridge**: minimize relative entropy D(Q||P) subject to marginal constraints → HJB equation for value function → optimal drift
- **Yang–Han**: minimize a cost functional for the complex trajectory → HJB equation in complex space → optimal guidance law
- **Holland**: bi-HJ equations *are* the HJB equations (forward and backward) of the bridge
- **Nelson**: forward and backward SDEs with drifts b± are the optimally controlled processes

The complex SD equation in Yang–Han is a **complexified version of Nelson's SDE**. Their drift velocity v_c = v_x + iv_y decomposes as:

    Re(v_c)|_{y=0} = (1/m) ∂S/∂x = v (current/Bohm velocity)
    Im(v_c)|_{y=0} = (ℏ/2m) ∂(ln ρ)/∂x = u (osmotic velocity)

On the real axis, the real and imaginary parts of the complex velocity **are** Nelson's current and osmotic velocities. The imaginary coordinate y carries the osmotic information that Holland encodes in σ₋ and that M5 extracts via √ρ selection.

---

## 4. Connections to M5

### 4.1 Node Crossing: Three Solutions to the Same Problem

The nodal issue — particles cannot cross or exist at nodes in deterministic Bohmian mechanics — has three resolutions that are mathematically related:

**Nelson/M5**: Stochastic diffusion allows particles to cross nodes. The Brownian noise term σ dW provides a mechanism for particles to "tunnel through" the zero-density region. M5's √ρ selection automatically reduces the density at nodes without requiring any special treatment.

**Yang–Han**: Particles move in the complex plane and never need to cross the real axis at nodes. The node is a singularity of the real guidance equation but is regular in the complex plane — trajectories orbit around the node at finite distance in the imaginary direction.

**Holland**: The two congruences q₊ and q₋ individually have non-conserved densities (each satisfies a Fokker–Planck equation with source term ±ν∇²ρ). Neither congruence density vanishes at nodes. The combined density ρ = exp[(σ₊ − σ₋)/ℏ] has nodes, but the individual trajectory families don't.

All three are resolving the same structural problem: **the osmotic velocity u = (ℏ/2m) ∂(ln ρ)/∂x diverges at nodes**. Nelson adds noise to regularize it, Yang–Han complexify the coordinate to avoid it, Holland decomposes into two flows neither of which individually encounters it.

M5's ψ-KDE estimator provides yet another resolution: at nodes, the complex current j_h undergoes destructive interference (opposing-phase contributions cancel), automatically producing |ψ̂| → 0 without requiring the osmotic velocity at all.

### 4.2 The "Two Point Sets" and M5's Dual Readout

Yang & Han's **two point sets** — intersections (quantum probability) vs. projections (classical probability) — have a suggestive parallel in M5v2/v3's dual readout architecture:

| Yang–Han | M5 | Mathematical role |
|---|---|---|
| Point set A (intersections with real axis) → |ψ|² | √ρ selection (STEER) | Forward Schrödinger potential φ̂ |
| Point set B (projections onto real axis) → classical ρ | ln ρ mean weight (backward readout) | Combined potential ln(φ̂ · φ) |
| ρ_c(t,x,y) (full complex density) | Full (X, S) particle state | Complete quantum state information |

The parallel is not exact, but the structural analogy is clear: both frameworks extract different physical content from the same underlying ensemble by applying different "readout" operations.

### 4.3 Complex Fokker–Planck and Holland's FP Pair

Yang & Han verify their probability distributions via a **complex Fokker–Planck equation**:

    ∂ρ_c/∂t + ∂(v_x ρ_c)/∂x + ∂(v_y ρ_c)/∂y = (ℏ/2m)[∂²ρ_c/∂x² + ∂²ρ_c/∂y²]

This is a 2D FP equation on the (x, y) plane. When restricted to the real axis (y = 0), it should reduce to something related to Holland's FP pair:

    ∂_t ρ + ∇·(ρ v₊) = +ν∇²ρ   (forward)
    ∂_t ρ + ∇·(ρ v₋) = −ν∇²ρ   (backward)

The sum gives the continuity equation ∂_t ρ + ∇·(ρv) = 0; the difference gives the osmotic equation. The complex FP equation encodes *both* Holland equations simultaneously in a single equation on the extended (complex) space, with the real direction carrying the continuity equation and the imaginary direction carrying the osmotic equation.

This is analogous to how M5v3 encodes both the forward and backward channels in a single forward-evolving ensemble: the √ρ selection implements the forward FP equation, and the ln ρ readout extracts the backward FP equation's content without solving it.

---

## 5. What M5 Gets That Complex Trajectories Don't (and Vice Versa)

### 5.1 Advantages of the M5/Nelson Approach

**Holland's probability current critique applies**: Defining a conserved, positive, normalizable probability on the complex plane is genuinely problematic. M5 works entirely on the real line with a well-defined particle histogram as the density. No interpretive gymnastics needed.

**Computational tractability**: M5 evolves N real-valued particle positions. Complex mechanics requires solving 2N-dimensional SDEs (real and imaginary parts for each particle). The dimensional doubling is a significant computational cost with no obvious physical payoff.

**The ψ-KDE already handles nodes**: M5's complex-current KDE achieves destructive interference at nodes automatically, without needing to go to complex coordinates. The "node problem" that motivates Yang–Han's complexification is already solved by a simpler mechanism.

**Self-consistent dynamics**: M5 particles constitute the density — there is no external pilot wave. Yang–Han still rely on ψ to define the guidance law, maintaining the standard pilot-wave ontological baggage.

### 5.2 What Complex Trajectories Offer

**Unified quantum-classical correspondence**: Yang–Han's demonstration that quantum and classical probabilities emerge from the *same* complex probability by different statistical operations (slice vs. marginal) is elegant and provides a clean picture of the correspondence principle. M5 doesn't have an equally clean story for the classical limit yet.

**Analytical power for energy eigenstates**: Complex trajectories can be computed analytically for many stationary problems (harmonic oscillator, hydrogen atom) where the complex velocity field has a known structure. This provides exact benchmarks.

**The imaginary direction as physical**: The idea that quantum particles actually explore an imaginary spatial dimension is speculative but provocative. It connects to complexified phase space in semiclassical mechanics (Maslov theory, complex WKB) and to the analytic continuation methods used in path integrals (Wick rotation, Picard–Lefschetz theory).

---

## 6. Synthesis: The Unified Picture

All four approaches — Nelson, Holland, Yang–Han, and M5 — can be understood as different "projections" of the Schrödinger bridge / FBSDE structure:

```
                    Schrödinger Bridge / FBSDE
                    ρ = φ̂ · φ,  HJB pair,  IPFP
                   /        |        \        \
                  /         |         \        \
          Nelson          Holland     Yang–Han    M5
     SDE on ℝ          bi-HJ on ℝ    SDE on ℂ    particles on ℝ
     b± = v ± u        σ± = S±½ℏlnρ  W = S+iR    (X, S) ensemble
     stochastic         deterministic  complex     √ρ selection
     noise is real      no noise       noise is    noise is real
                                       complex     (candidates)
```

The sign of the Fisher information distinguishes all of these from ordinary diffusion:
- **Schrödinger bridge** (entropy-regularized transport): +Fisher → positive surface tension, density smoothing
- **Quantum mechanics** (Nelson/Holland/Yang–Han/M5): −Fisher → negative surface tension, density sharpening (interference, nodes)

M5's unique contribution is implementing this structure **without a pilot wave** and **without explicit density differentiation**, using the √ρ selection mechanism to let the Fisher information term emerge self-consistently from particle statistics.

---

## 7. The "Product" Bill Noticed

The specific "product" structure in Yang–Han's discussion of quantum probability is indeed the Schrödinger bridge factorization in disguise. Here's the chain:

1. ψ = exp(iW/ℏ) where W = S_B + iR_B is complex
2. |ψ|² = exp(−2 Im(W)/ℏ) = exp(2R_B/ℏ) [since ψ = R_B e^{iS_B/ℏ} with R_B = ln(amplitude)]
3. But 2R_B/ℏ = (σ₊ − σ₋)/ℏ in Holland's notation
4. So |ψ|² = exp(σ₊/ℏ) · exp(−σ₋/ℏ) = φ̂ · φ

This is exactly ρ = φ̂ · φ, the Schrödinger bridge factorization.

Yang–Han's insight is that this product arises naturally when you view the real-axis quantum probability as a **cross-section** of a higher-dimensional (complex-plane) probability. The Schrödinger bridge insight is that this product arises from the **coupling of forward and backward optimal control problems**. Holland's insight is that this product arises from the **action difference** between two deterministic trajectory congruences.

M5 computes this product empirically: the particle histogram *is* ρ = φ̂ · φ, maintained self-consistently through the √ρ selection (which implements φ̂) and the phase evolution (which propagates the information about φ).

---

## References

- Yang, C.D. & Han, S.Y. (2020). "Trajectory Interpretation of Correspondence Principle: Solution of Nodal Issue." Found. Phys. 50, 960–976.
- Yang, C.D. & Han, S.Y. (2021). "Extending Quantum Probability from Real Axis to Complex Plane." Entropy 23(2), 210.
- Holland, P. (2021). "Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory." arXiv:2111.09235.
- Chou, C.C. & Wyatt, R.E. (2008). "Considerations on the probability density in complex space." Phys. Rev. A 78, 044101.
- Nelson, E. (1966). "Derivation of the Schrödinger Equation from Newtonian Mechanics." Phys. Rev. 150, 1079.
- Conforti, G. & Pavon, M. (2017). "Extremal flows on Wasserstein space." J. Math. Phys. 58, 093302.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). "Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schrödinger bridge." SIAM Review 63, 249–313.
- John, M.V. (2009). "Probability and Complex Quantum Trajectories." Ann. Phys. 324, 220–231.
