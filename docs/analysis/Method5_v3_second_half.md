
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

- `Method5_Mathematical_Analysis_v2.md` — v2 dual-readout theory (Theorems 1–2), variance analysis, antithetic candidates
- `Method5_Mathematical_Analysis_v3.md` — v3 first half: three weight functions, log-density approach, mirror particles, GH implementation, numerical validation
- `Holland_Nelson_FokkerPlanck_Analysis.md` — Comparative analysis of Holland and Nelson, coupled SDE system, Fokker–Planck pair
- `FBSDE_SchrodingerBridge_Nelson_Holland.md` — Triangular relationship between FBSDEs, bridges, and bi-HJ
- `Method5_QA_Discussion.md` — Time symmetry, local Sinkhorn, Wasserstein dynamics
- `NelsonMechanics_SchrodingerBridge_Swarmalator.md` §§2–3 — ψ-KDE density estimator theory
- `HackebillPoirier2602_21106v1.pdf` — Sparse ontology analysis, CHV vs DHV classification
