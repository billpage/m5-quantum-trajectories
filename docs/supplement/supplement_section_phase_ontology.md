## 24. The Ontology of Phase and Motion

### 24.1 The Velocity Reference Problem

In both the grid-based M5 algorithm and the quantum swarmalator reformulation, the current velocity appears as an apparently absolute quantity. The grid algorithm computes v(x) = (ℏ/m) ∂ₓS(x) — the gradient of a scalar field evaluated at a point. The swarmalator computes v_i = (ℏ/m) Im(j'_i / j_i), where j_i and j'_i are coherent kernel sums over the neighbourhood of particle i. Neither formula involves subtraction of any reference velocity. This raises a conceptual question: velocity with respect to what?

In classical Hamilton-Jacobi mechanics, the same apparent absoluteness holds: the HJ equation ∂ₜS + (∇S)²/(2m) + V = 0 defines S(x,t) as a scalar field, and v = ∇S/m appears to be an absolute velocity. The resolution is that S is not a Galilean scalar. Under a boost to a frame moving with velocity V, the action transforms as S'(x', t') = S(x' + Vt', t') + mVx' + ½mV²t', so that ∂_{x'}S'/m = ∂_xS/m + V. The velocity transforms correctly, but only because S transforms. The "reference frame" is encoded in the initial conditions of S — a spatially constant S means "at rest in this frame."

The swarmalator inherits this structure. The kernel K_h(X_i − X_j) depends only on relative positions and is Galilean invariant. The coherent sum j_i = Σ_j K_h(X_i − X_j) e^{iS_j/ℏ} changes under a boost only if the phases transform: S_j → S_j + mV·X_j + ½mV²t. This adds a linear phase ramp across the ensemble, shifting Im(j'/j) by mV/ℏ and hence v by V. The swarmalator velocity is thus relative to the frame encoded in the spatial pattern of phases {S_j}. The inter-particle phase structure is genuinely relational — only phase gradients enter the dynamical formula — but the global phase gradient encodes the boost velocity of the ensemble.

### 24.2 Phase Is Not an Internal State

The geometric formulation of Hamilton-Jacobi theory clarifies the status of S. The natural setting is the cotangent bundle T*Q, where the geometrically meaningful object is the momentum 1-form p = dS — the exterior derivative of the action — rather than S itself. The Hamilton-Jacobi equation becomes H(q, dS) + ∂ₜS = 0, and a solution defines a Lagrangian submanifold of T*Q on which the symplectic form ω = dq ∧ dp vanishes. The physical content lives entirely in dS (the 1-form) and in d²S = 0 (the irrotationality condition); S itself is a potential for this 1-form, determined only up to an additive constant.

The action S thus has three properties that distinguish it from an ordinary internal state variable (such as mass, charge, or temperature):

1. **Gauge freedom.** A constant shift S → S + c leaves all physics unchanged (U(1) symmetry of the wavefunction ψ = √ρ · e^{iS/ℏ}).

2. **Frame dependence.** Under a Galilean boost, S → S + mVx + ½mV²t. This is because S accumulates kinetic energy ½mv² along the trajectory, and kinetic energy is frame-dependent. By contrast, temperature — defined in the comoving frame as the variance of molecular velocities around the local mean — is a Galilean scalar. The action is not.

3. **Relational dynamics.** The swarmalator velocity formula j'/j depends only on the spatial variation of S across the neighbourhood: the differences S_j − S_i weighted by kernel proximity K_h(X_i − X_j). Adding a constant to all phases, or evaluating S_i in isolation, produces no dynamical consequence. The formula estimates the 1-form dS from a discrete sampling of the potential, using coherent kernel regression.

A quantity that is gauge-dependent, frame-dependent, and dynamically operative only through its inter-particle differences cannot honestly be called an "intrinsic property" of a particle in the way that mass or charge can. It is closer to a *connection variable* — a quantity whose value at a point is conventional, but whose variation between points encodes physics. The analogy to the electromagnetic vector potential A is precise: the field strength F = dA is gauge-invariant and physical, while A itself is gauge-dependent and unphysical, yet indispensable as a computational intermediary.

In the swarmalator, the coherent phase factors e^{i(S_j − S_i)/ℏ} play the role of *link variables* in lattice gauge theory: relational quantities defined on pairs of sites (particles), encoding how the gauge field (phase) parallel-transports between neighbours. The global phase e^{iS_i/ℏ} is the gauge choice at site i; the link variables are the physical content. The swarmalator velocity formula is a weighted sum over links — a kernel-regularised Wilson line — and the Wallstrom quantization condition ∮ dS = nℏ is a constraint on the holonomy (Wilson loop), not on any site variable.

### 24.3 Phase Is Constitutive of Motion

The geometric analysis of §24.2 might suggest that S is merely bookkeeping — a gauge artefact without ontological weight. But this conclusion is premature, and a closer examination of what "motion" means in a discrete-particle system reverses the judgement.

Consider a cellular automaton such as Conway's Game of Life. A "glider" is a pattern of cell states that cycles through a period-4 sequence, with the net effect that the pattern is displaced by one cell diagonally per cycle. The glider has no position as a primitive variable — its "position" is an emergent property of the pattern. And crucially, the glider's *motion* cannot be separated from its internal state cycling. If the internal cycling is frozen, there is no stationary glider — there is no glider at all. The entity's existence is constituted by its cycling; the "velocity" is the observable manifestation of how the cycling pattern shifts.

This example (which has obvious resonance with the swarmalator picture) suggests that the relationship between phase and motion may run deeper than "phase records motion." De Broglie's original intuition pointed in the same direction: a particle of mass m has an intrinsic frequency ω₀ = mc²/ℏ in its rest frame. Motion manifests as a spatial phase gradient — the de Broglie relation p = ℏk — which is the rate at which the internal clock phase falls out of synchronisation across space. In this picture, velocity is not a primitive that happens to generate phase accumulation; rather, the phase cycling is primitive, and velocity is the observable consequence of differential cycling across the ensemble.

A mechanical analogy makes this concrete. The orientation of a wheel on an automobile is a cyclic variable (phase). The odometer reading is the accumulated action — the integral of the wheel's cycling over the journey. One might argue that the odometer reading is "merely bookkeeping" — a record of history rather than an internal state. But the wheel orientation is physically real: it couples to the road surface, determines the instantaneous velocity, and its rate of change *is* the motion. The odometer is not a passive record; it is the integral of a dynamical variable whose current value is constitutive of the particle's interaction with its environment.

Similarly, the step counter on a pedometer integrates a cyclic internal variable (stride phase). The accumulated count is history, but the current phase of the stride cycle is constitutive of the walker's dynamics — it determines whether the left or right foot is in contact with the ground, the instantaneous force profile, and the coupling to the terrain.

In the swarmalator, each particle's phase e^{iS_i/ℏ} cycles at a rate determined by the local Lagrangian (½mv² − V − Q). The spatial variation of this cycling rate across the neighbourhood is the momentum field. The velocity of particle i is not "read off" from its phase and then applied as an external instruction; the phase cycling and the spatial displacement are two aspects of the same dynamical process. The formula v_i = (ℏ/m) Im(j'_i / j_i) extracts the local phase gradient — the differential cycling rate — and this IS the velocity, not a proxy for it.

### 24.4 Reconciling the Two Views

The tension between §24.2 (phase as gauge artefact) and §24.3 (phase as constitutive of motion) is resolved by recognising that these are statements about different aspects of the same variable:

**The absolute value** of S_i is gauge/frame-dependent and physically meaningless. This is the "connection" aspect — it depends on conventional choices (zero of energy, reference frame, initial phase assignment).

**The rate of change** dS_i/dt = ½mv² − V − Q is frame-dependent but dynamically meaningful — it is the Lagrangian evaluated along the trajectory, and it governs the local contribution to the phase pattern.

**The spatial pattern** {∂S/∂x} is the momentum 1-form, and its local value determines the particle's velocity. This is the "field strength" aspect — gauge-invariant, frame-covariant, and constitutive of motion.

The situation is precisely analogous to clock time. The absolute reading of a clock (14:37 vs. 2:37 PM vs. 18:37 UTC) is conventional. The *rate* of the clock is physically meaningful (and frame-dependent in relativity — time dilation). The *phase differences* between clocks are physical and frame-covariant (this is how GPS works). We do not say that time is "merely bookkeeping" simply because the zero of the time coordinate is arbitrary. The cycling is real; the label is conventional.

The swarmalator ensemble is thus best understood as a **network of coupled clocks on a dynamical lattice**: each particle is a clock (an entity with position and a cycling internal state), the velocity of each particle is determined by the phase relationships with its neighbours, and the collective behaviour — interference, tunnelling, nodal structure — emerges from the synchronisation and desynchronisation dynamics of the clock network. This is the quantum realisation of the swarmalator paradigm (O'Keeffe et al. 2017): mobile oscillators whose spatial velocity couples to the phases of neighbours. The distinction from classical swarmalators is that the quantum version uses coherent (complex) averaging, which permits destructive interference — classical swarmalators can synchronise and desynchronise, but they cannot cancel.

### 24.5 Levels of Ontological Description

The foregoing analysis suggests three possible levels of ontological commitment for the swarmalator, in increasing order of parsimony:

**Level 1 — Particles with internal phase.** Each particle carries {X_i, S_i} as independent degrees of freedom. The phase S_i is an internal state, albeit one with gauge and frame dependence. This is the computationally natural description and the one used in the algorithm. Its ontological weakness is that S_i appears as an intrinsic property when it functions relationally.

**Level 2 — Particles with relational momentum.** Each particle has position X_i, and there exists relational data p_{ij} ∝ (S_j − S_i)/(X_j − X_i) defined on pairs. No individual particle has a phase; the momentum 1-form is a property of the ensemble's configuration, estimated by kernel regression. The phases {S_i} are a convenient gauge-fixing that encodes this relational data on sites rather than on links. This is the geometrically honest description, corresponding to the cotangent bundle / lattice gauge theory picture of §24.2.

**Level 3 — Particles with trajectory history.** Each particle has only position X_i, and the phase differences between particles are entirely determined by the history of their positions — the accumulated actions along their respective trajectories through the potential landscape. In this view, S_i is not carried by the particle but is the integral ∫(½mv² − V − Q) dt along the particle's worldline. Two particles at the same position with different phases arrived by different routes. This is close to the Feynman path integral picture, where the phase of each path is its classical action.

The glider/clock arguments of §24.3 suggest that Level 3 is too deflationary. The accumulated action is not merely a record; its current spatial pattern is constitutive of the dynamics at the present moment. A particle's phase determines how it couples to its neighbours *now*, not merely where it has been. The distinction between Levels 1 and 2 is more subtle: Level 2 is geometrically cleaner, but Level 1 captures the constitutive role of phase cycling (the clock picture) more naturally.

In practice, the three levels are computationally equivalent — they produce identical dynamics. The choice between them is a matter of ontological taste, constrained by the requirement that the description be consistent with gauge freedom, Galilean covariance, and the relational character of the swarmalator velocity formula.

### 24.6 Connections to the Hamilton-Jacobi Geometric Framework

The invariant geometric formulation of Hamilton-Jacobi theory provides the mathematical infrastructure for the ontological distinctions drawn above. The key objects are:

**The tautological 1-form** θ = p_i dq^i on the cotangent bundle T*Q. This is defined intrinsically, without coordinates or metrics. A solution S(q,t) of the HJ equation defines a section q ↦ (q, dS(q)) of T*Q, and the momentum p = dS is the pullback of θ along this section.

**The symplectic 2-form** ω = −dθ = dq^i ∧ dp_i. This is boost-invariant and encodes all of Hamiltonian mechanics. The key point: ω is invariant under canonical transformations (including Galilean boosts), but θ is not. The non-invariance of S under boosts inherits from the non-invariance of θ, since S is the generating function whose exterior derivative gives the section.

**The Poincaré-Cartan 1-form** Θ = p_i dq^i − H dt on extended phase space. The HJ equation becomes dS = Θ restricted to the solution surface, and the full dynamical content lives in the 2-form dΘ = ω − dH ∧ dt, which is invariant. Hamilton's equations are the characteristics of dΘ = 0 on the Lagrangian submanifold defined by S.

**The irrotationality condition** dp = d²S = 0 is automatic from the nilpotency of the exterior derivative wherever S exists. On domains with nontrivial topology (e.g. encircling a nodal point), the closed 1-form p = dS need not be exact, and the obstruction is measured by the periods ∮_γ p = nℏ — the Wallstrom quantization condition. In the swarmalator, the ψ-KDE enforces quantized periods automatically through destructive interference in the coherent sum.

**Holland's bi-HJ structure** defines two sections q ↦ (q, dσ_±) of T*Q, where σ± = S ± R and R = (ℏ/2) ln ρ. The physical content decomposes cleanly:

- The sum dσ_+ + dσ_- = 2dS transforms covariantly under boosts (both momenta shift by mV).
- The difference dσ_+ − dσ_- = 2dR = ℏ d ln ρ is boost-invariant, since the density ρ is a Galilean scalar.
- The current velocity v = (dσ_+ + dσ_-)/(2m) transforms as a vector.
- The osmotic velocity u = (dσ_+ − dσ_-)/(2m) is invariant.

The exterior derivative formulation makes this manifest without any coordinate calculation: the differential operator d commutes with pullback along the boost map, so whether a quantity transforms or not is determined entirely by whether the underlying potential (S or R) transforms. Since S accumulates frame-dependent kinetic energy while R = (ℏ/2) ln ρ depends only on the frame-independent density, their transformation properties are fixed by their definitions.

### 24.7 Summary: The Swarmalator as a Coupled Clock Network

The velocity reference problem — "velocity with respect to what?" — admits a clean resolution once the geometric and ontological status of the phase is properly understood.

The phase S_i is not an internal state of particle i in the thermodynamic sense (like temperature), because it is gauge-dependent, frame-dependent, and enters the dynamics only through its spatial variation across the ensemble. It is best understood as a connection variable: a quantity whose absolute value is conventional but whose differential structure (the 1-form dS) encodes the physically meaningful momentum field.

At the same time, the phase is not merely a passive record of accumulated history. The cycling of phase — the continuous advance of S_i along each trajectory at rate dS/dt = ½mv² − V − Q — is constitutive of the particle's dynamics, in the same sense that the internal state cycling of a cellular automaton glider is constitutive of its motion, or the wheel rotation of an automobile is constitutive of its travel. The velocity of the swarmalator particle is not an independent primitive to which phase responds; it is the observable consequence of differential phase cycling across the ensemble.

The quantum swarmalator is therefore a system of coupled clocks: each particle is a clock (position + cycling phase), the velocity of each clock is determined by the coherent phase relationships with its neighbours, and the collective dynamics — including interference, tunnelling, and nodal structure — emerge from the synchronisation geometry of the clock network. The reference frame is encoded holographically in the global pattern of phases, which transforms correctly under Galilean boosts because the action (like kinetic energy) is frame-dependent. The physically meaningful content is entirely in the 1-form dS and the relational phase structure, both of which are accessible through the swarmalator's coherent kernel sums without reference to any external wavefunction or coordinate grid.
