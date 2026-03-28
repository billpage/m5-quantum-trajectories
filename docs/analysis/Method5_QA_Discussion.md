# Method 5: Time Symmetry, Schrödinger Bridges, and Wasserstein Dynamics

## A Q&A Discussion on the √ρ-Selection Algorithm and Its Deep Mathematical Connections

---

### Preamble

This document records a series of questions and answers exploring the conceptual and mathematical foundations of **Method 5**, the √ρ-selection particle algorithm for quantum dynamics. The discussion moves from the algorithm's time-reversal symmetry, through its connection to the Sinkhorn/IPFP algorithm for Schrödinger bridges, to its interpretation as Newton's second law on Wasserstein space. Throughout, the aim is to build intuition for *why* the simple mechanism of selecting Brownian candidates proportional to √ρ is sufficient to reproduce full quantum dynamics.

#### Key Project References

> **Note:** This discussion was written against early versions of the algorithm documents (`Method5_Mathematical_Analysis.md` v1 and `method5_selection.py`) that predate the current repository. Section references below (§3, §7, etc.) refer to those earlier documents. The current algorithm specification is `NelsonMechanics_SchrodingerBridge_Algorithm.md`; the proofs and derivations cited here have been consolidated there and in the companion supplement.

| Document | Description |
|----------|-------------|
| `Method5_Mathematical_Analysis.md` | Complete mathematical derivation and proof that √ρ-selection induces the correct osmotic drift (historical, not in repository; superseded by `NelsonMechanics_SchrodingerBridge_Algorithm.md`) |
| `method5_selection.py` | Python implementation comparing Method 5 against Method 4 (explicit osmotic drift) and the Schrödinger FFT reference (historical, not in repository; superseded by `m5psi_kde_catstate.py`) |
| `FBSDE_SchrodingerBridge_Nelson_Holland.md` | The triangular relationship between FBSDEs, Schrödinger bridges, Nelson's stochastic mechanics, and Holland's bi-HJ theory |
| `Holland_Nelson_FokkerPlanck_Analysis.md` | Detailed comparative analysis of Holland's and Nelson's formulations, focusing on the Fokker–Planck pair and coupled SDEs |
| `HollandEliminating2111_09235.pdf` | Holland (2021), "Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory" |
| `1966_ENelson_Derivation_of_SchrodEq_from_NewtMech.pdf` | Nelson (1966), the foundational paper on stochastic mechanics |
| `Nelson_2012_J__Phys___Conf__Ser__361_012011_2.pdf` | Nelson (2012), review of stochastic mechanics including his acknowledgement of its limitations |

---

## Q1. Is there an easy way to understand the time symmetry of the Method 5 algorithm? How is it related to the Schrödinger Bridge concept?

### The Forward–Backward Structure

In the Nelson/Holland framework, quantum dynamics involves two stochastic descriptions running simultaneously:

- **Forward process:** drift $v_+ = v + u$, governed by the forward Fokker–Planck equation with $+\nu\nabla^2\rho$
- **Backward process:** drift $v_- = v - u$, governed by the backward Kolmogorov equation with $-\nu\nabla^2\rho$

Under time reversal, these swap roles. In Holland's formulation, the exchange is non-standard: $\sigma_\pm'(x',t') = -\sigma_\mp(x,t)$, so each action function maps to the *negative of the other*, rather than simply reversing sign. The current velocity $v$ reverses (it tracks momentum), while the osmotic velocity $u$ remains invariant (it tracks density gradients). This is discussed in detail in `Holland_Nelson_FokkerPlanck_Analysis.md`, §5.1.

### Why √ρ Is the Time-Symmetric Weight

The key insight is that **√ρ is the geometric mean of the forward and backward descriptions**.

A pure forward solver would weight candidates by ρ. A pure backward solver would also use ρ (same marginal density). But their drifts differ — forward needs +u, backward needs −u. Method 5 uses √ρ, which induces exactly *half* the log-density gradient:

$$\sigma^2 \cdot \partial_x \ln \sqrt{\rho} = \sigma^2 \cdot \tfrac{1}{2}\partial_x \ln \rho = \nu \,\partial_x \ln \rho = u$$

This is precisely the osmotic component — the part that is symmetric under time reversal. The current velocity $v$ (which reverses under T) is handled explicitly in the classical step.

Method 5 thus naturally separates:
- **T-even part** (osmotic velocity $u$): handled by √ρ selection
- **T-odd part** (current velocity $v$): handled by the deterministic push

### The Schrödinger Bridge Connection

In the Schrödinger bridge framework, the action functional is:

$$A[\rho, v] = \int_0^T\!\int \left[\tfrac{1}{2}|v|^2\rho + \tfrac{\varepsilon^2}{8}|\nabla\ln\rho|^2\rho\right] dx\,dt$$

The second term is the **Fisher information** of ρ — the term that distinguishes the Schrödinger bridge from classical optimal transport (Benamou–Brenier). Method 5's √ρ selection **locally minimises the Fisher information cost**: by biasing candidates toward higher √ρ, the algorithm implements a local version of the Sinkhorn/IPFP reweighting at every time step.

The connection to the Hopf–Cole structure makes this precise. In the FBSDE framework, the forward drift is $b = \nabla\hat{\varphi}$ where $\hat{\varphi} = \ln(\sqrt{\rho_{\text{forward}}})$ — the forward Schrödinger potential is already a log of √ρ, not of ρ. Weighting by √ρ is equivalent to exponentiating the forward potential, which is exactly the correct half-bridge reweighting (see `Method5_Mathematical_Analysis.md`, §7 and `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §2.2–2.4).

### The Simple Mental Picture

The wavefunction $\psi = \sqrt{\rho} \cdot e^{iS/\hbar}$ has amplitude √ρ, not ρ. The Schrödinger bridge potentials $(\varphi, \hat{\varphi})$ factorise the density as $\rho = \varphi \cdot \hat{\varphi}$, with each potential contributing "√ρ's worth." Method 5 selects candidates using exactly one of these halves — the forward Schrödinger potential — which automatically builds in the correct time-symmetric osmotic correction without ever computing a log-density gradient.

#### References for Q1
- Holland, P. (2021). §§2, 4.2. `HollandEliminating2111_09235.pdf`
- Nelson, E. (1966). §§II–III. `1966_ENelson_Derivation_of_SchrodEq_from_NewtMech.pdf`
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). "Stochastic Control Liaisons." *SIAM Review* 63, 249–313.
- Conforti, G. & Pavon, M. (2017). "Extremal flows on Wasserstein space." arXiv:1712.02257.
- Project: `Method5_Mathematical_Analysis.md`, §§6–7
- Project: `Holland_Nelson_FokkerPlanck_Analysis.md`, §§5.1, 4.1–4.4
- Project: `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §§2.2–2.4, 4.1

---

## Q2. Can you explain the Sinkhorn/IPFP algorithm in simple terms using an example related to quantum dynamics?

### Schrödinger's Hot Gas Problem

The Sinkhorn/IPFP algorithm solves what Schrödinger posed in 1931: given a cloud of diffusing particles observed at two times with known densities ρ₀ and ρ₁, what is the *most likely* stochastic evolution connecting them?

The answer minimises the relative entropy $D(Q\|P)$ of the path measure $Q$ relative to the unconstrained diffusion $P$, subject to the marginal constraints $Q_0 = \rho_0$ and $Q_1 = \rho_1$. The solution is the **Schrödinger bridge** (see `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §2.1).

### The Matrix Analogy

Discretise space into bins. The unconstrained diffusion gives a **transition matrix** $M(i,j)$ — the probability that a particle starting in bin $i$ ends up in bin $j$ under pure Brownian motion. The problem is to find a new matrix $M^*$ that is as close to $M$ as possible (in relative entropy) while having the correct row marginals (matching ρ₀) and column marginals (matching ρ₁).

This is a matrix scaling problem, and Sinkhorn's algorithm solves it by alternating two operations:

**Step 1 — Forward pass (fix columns):** Multiply each column by a factor so column sums match ρ₁. This reweights endpoints to match the observed final density.

**Step 2 — Backward pass (fix rows):** Multiply each row by a factor so row sums match ρ₀. This reweights starting points to match the observed initial density.

Alternating these operations converges to $M^* = \text{diag}(\varphi) \cdot M \cdot \text{diag}(\hat{\varphi})$, where $\varphi$ and $\hat{\varphi}$ are the **Schrödinger potentials**. The bridge density satisfies $\rho = \varphi \cdot \hat{\varphi}$, with each potential contributing "half" the density — this is why √ρ appears naturally.

### A Quantum Example: Cat State Collision

Consider two Gaussian wave packets heading toward each other (the cat state collision that has been validated with Method 5). At $t = 0$, the density has two bumps. At $t = T$, an interference pattern forms.

- **Iteration 0:** Start with the pure diffusion kernel (no quantum potential). It just spreads the bumps — knows nothing about interference.
- **Iteration 1, forward pass:** Reweight the final-time density to match the interference pattern. Force particles to cluster at interference maxima and avoid nodes. The factor $\hat{\varphi}(x, T)$ is large at peaks, small at nodes.
- **Iteration 1, backward pass:** Propagate backward from the corrected final density to match the initial two-bump distribution.
- **Iterate** until convergence. The resulting bridge has drifts $v_+ = \nabla\ln\hat{\varphi}$ (forward) and $v_- = -\nabla\ln\varphi$ (backward), which are exactly Holland's $\sigma_\pm$ velocities.

### How Method 5 Does This Locally

The global Sinkhorn algorithm requires solving the full bridge problem iteratively over the entire time interval. Method 5 replaces this with a **local, one-step version**: at each time step dt, generate K Brownian candidates per particle and select one proportional to √ρ.

This is one "half-iteration" of Sinkhorn at the infinitesimal level. At small dt, one reweighting per step is sufficient — the correction is already small, and the local adjustments accumulate into the global bridge solution. As described in `Method5_Mathematical_Analysis.md`, §7:

> *"Method 5's per-step selection is a local, online version of this reweighting. Rather than globally solving the bridge problem, each particle locally corrects its trajectory by selecting among stochastic candidates to match the current density."*

### Why √ρ and Not ρ

In the Sinkhorn factorisation $\rho = \varphi \cdot \hat{\varphi}$, each potential scales like √ρ in the time-symmetric case. The forward reweighting uses $\hat{\varphi} \propto \sqrt{\rho}$; the backward uses $\varphi \propto \sqrt{\rho}$. Using the full ρ would apply *both* reweightings simultaneously, yielding drift $2u$ instead of $u$.

As proven in `Method5_Mathematical_Analysis.md`, §3: setting $w = \sqrt{\rho}$ induces drift $\sigma^2 \cdot \partial_x \ln\sqrt{\rho} = u$, while setting $w = \rho$ would give $\sigma^2 \cdot \partial_x \ln \rho = 2u$.

#### References for Q2
- Schrödinger, E. (1931). "Über die Umkehrung der Naturgesetze." *Sitzungsber. Preuss. Akad. Wiss.*, 144–153.
- Fortet, R. (1940). Résolution d'un système d'équations de M. Schrödinger. *J. Math. Pures Appl.* 9, 83–105.
- Sinkhorn, R. (1967). Diagonal equivalence to matrices with prescribed row and column sums. *Amer. Math. Monthly* 74, 402–405.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2021). "Stochastic Control Liaisons." *SIAM Review* 63, 249–313.
- Project: `Method5_Mathematical_Analysis.md`, §§3, 6, 7
- Project: `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §§2.1–2.3

---

## Q3. The algorithm uses the current density to weight candidates — it seems like we are "correcting toward the old density." Why does repeating these simple steps produce quantum dynamics?

### The Confusion

At each time step, Method 5:
1. Estimates the current density ρ(t)
2. Advances particles classically via the current velocity $v$
3. Applies the stochastic step: each particle jumps to a random nearby candidate, selected by weighting with √ρ evaluated at the **old** (pre-step) density

This appears to be "correcting toward the old density" — as if the distribution resists being changed by the particle dynamics.

### What the Selection Is Actually Doing

The √ρ selection is **not** restoring the old density. It is imposing a **constraint on how particles are allowed to diffuse**.

Without selection, every particle jumps to a random nearby location with equal probability in all directions — pure entropy-increasing diffusion. With √ρ selection, among the K candidate jumps, the algorithm prefers those landing where the current density is higher. Particles near the edge of a density peak are more likely to jump inward; particles near interference nodes (where √ρ → 0) are strongly repelled.

The net effect at each step is not "density stays the same." It is: *density changes, but diffusion contributes an inward drift of exactly* $u = \nu\partial_x\ln\rho$ *in addition to the random spreading.* The density absolutely evolves — the classical step transports it, diffusion broadens it, and the osmotic correction reshapes it — but all three effects combined produce exactly the change prescribed by the Schrödinger equation.

### An Analogy: Incompressible Fluid Flow

In the Navier–Stokes equations for an incompressible fluid, a pressure field adjusts at each instant to enforce $\nabla\cdot v = 0$. The pressure does not hold the fluid in place — it flows, swirls, and does complicated things — but the pressure constrains *how* it flows, preventing density changes.

The √ρ selection plays an analogous role. It constrains the *manner* in which diffusion reshapes the density, ensuring the reshaping is consistent with the quantum Fokker–Planck equation rather than the classical heat equation. The "resistance" is the **osmotic pressure** $\nu^2\partial_x\ln\rho$, the stochastic-mechanical counterpart of the quantum potential.

### Why Using the "Old" Density Gives the Correct "New" Density

**Practical answer:** This is explicit Euler time-stepping. Using ρ(t) to compute the correction at time $t$ introduces only $O(dt^2)$ errors, below the accuracy of the scheme. This is no different from using the current force to advance particles in a classical Verlet integrator.

**Deep answer:** The Schrödinger equation is **local in time**. The osmotic velocity $u(x,t) = \nu\partial_x\ln\rho(x,t)$ is determined entirely by the current density. The correct quantum drift at time $t$ depends only on where the density is *right now*. This is the Markov property of the Nelson process.

### Concrete Example: Free Gaussian Spreading

For a Gaussian wave packet at rest ($v = 0$, no potential):

- **Without √ρ selection:** Width grows as $\sigma(t) = \sqrt{\sigma_0^2 + 2\nu t}$ (heat equation spreading — too fast).
- **With √ρ selection:** The inward osmotic bias partially counteracts the outward diffusion. Width grows as $\sigma(t) = \sigma_0\sqrt{1 + \hbar^2 t^2 / 4m^2\sigma_0^4}$ — slower than heat diffusion, encoding the correct quantum uncertainty dynamics.

The density *does* change. The selection isn't restoring it — it's **moderating** how fast diffusion changes it.

### Why It Works: Self-Consistent Feedback

Particles create the density, and the density guides the particles. This circular causation is exactly the content of the nonlinear Schrödinger dynamics. Nelson's stochastic Newton equation (Nelson 1966, Eq. 30):

$$m\left[\frac{\partial v}{\partial t} + (v\cdot\nabla)v - (u\cdot\nabla)u - \nu\nabla^2 u\right] = -\nabla V$$

encodes this self-consistency. The osmotic velocity $u$ is a **density-dependent feedback** that constrains how the ensemble evolves — the density can only change in ways consistent with the Schrödinger equation. The √ρ selection is the minimal mechanism enforcing this at every time step.

#### References for Q3
- Nelson, E. (1966). §§II–IV. `1966_ENelson_Derivation_of_SchrodEq_from_NewtMech.pdf`
- Nelson, E. (2012). §§2–3. `Nelson_2012_J__Phys___Conf__Ser__361_012011_2.pdf`
- Project: `Method5_Mathematical_Analysis.md`, §§2–5
- Project: `method5_selection.py` (numerical validation across free Gaussian, harmonic oscillator, double well, and cat state test cases)
- Project: `Holland_Nelson_FokkerPlanck_Analysis.md`, §§3.3–3.4 (source terms and non-conservation along individual flows)

---

## Q4. Does the intuition of "collective inertia" or "resistance" of the ensemble connect to the idea of Wasserstein space? Can we think of the density as evolving as a whole with dynamic collective properties like inertia and momentum?

### Yes — This Becomes Literally True in Wasserstein Space

This is not merely a metaphor. Otto's geometric framework (circa 2001) equips the space of probability distributions with a Riemannian structure in which the density genuinely has position, velocity, momentum, and inertia.

### The Density as a Particle

In ordinary mechanics, a particle has position $q$, velocity $\dot{q}$, and mass $m$. Newton's law: $m\ddot{q} = -\nabla V$.

In the Wasserstein picture:

| Ordinary Mechanics | Wasserstein Mechanics |
|---|---|
| Position $q$ | Density $\rho(x) \in \mathcal{W}_2$ |
| Velocity $\dot{q}$ | Velocity field $v(x,t)$ transporting $\rho$ via $\partial_t\rho + \nabla\cdot(\rho v) = 0$ |
| Momentum $m\dot{q}$ | Momentum field $\rho v$ |
| Kinetic energy $\tfrac{1}{2}m|\dot{q}|^2$ | $T = \tfrac{1}{2}\int |v(x)|^2 \rho(x)\,dx$ (Benamou–Brenier) |
| Distance $|q_1 - q_2|$ | Wasserstein distance $W_2(\rho_0, \rho_1)$ = optimal transport cost |
| Mass $m$ | Distributed inertia: each element $\rho(x)\,dx$ contributes its share |

The Wasserstein metric makes $\mathcal{W}_2$ a Riemannian manifold. The "distance" between two densities is the minimum work to rearrange one into the other. The density-particle doesn't just have a position in this space; it has **genuine momentum** $\rho v$ with dynamical content.

### Newton's Law on Wasserstein Space

The Conforti–Pavon (2017) result, described in `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §3.4, shows that all three problems — optimal transport, Schrödinger bridges, and quantum mechanics — satisfy a **Newton-like second law** on Wasserstein space:

$$\nabla_t(\rho v) = -\rho\,\nabla\!\left(\frac{\delta F}{\delta\rho}\right)$$

where $\nabla_t$ is the covariant derivative along the curve in $\mathcal{W}_2$, and $F[\rho]$ is a functional that depends on which problem is being solved:

| Problem | Action Functional | Physical Interpretation |
|---|---|---|
| **Optimal Mass Transport** | $\int\!\int \tfrac{1}{2}|v|^2\rho\,dx\,dt$ | Free motion, no internal structure |
| **Schrödinger Bridge** | $\int\!\int \left[\tfrac{1}{2}|v|^2 + \tfrac{\varepsilon^2}{8}|\nabla\ln\rho|^2\right]\rho\,dx\,dt$ | Internal pressure from Fisher information (+) |
| **Quantum Mechanics (Madelung/Nelson)** | $\int\!\int \left[\tfrac{1}{2}|v|^2 - \tfrac{\nu^2}{2}|\nabla\ln\rho|^2 - V\right]\rho\,dx\,dt$ | Negative Fisher information = quantum potential |

The sign of the Fisher information term is the crucial structural distinction:

- **Schrödinger bridge (+):** Penalises density gradients (entropy production). Creates internal pressure — the density resists compression into sharp features.
- **Quantum mechanics (−):** Rewards density gradients (quantum potential). The density is energetically drawn *toward* sharp features — creating interference patterns, tunnelling barriers, and all non-classical structure.

### The Elastic Membrane Analogy

Think of ρ as an elastic membrane over the x-axis:

- **Optimal transport:** The membrane is perfectly floppy — can be reshaped at zero cost. Only external forces matter. Zero internal pressure.
- **Schrödinger bridge:** The membrane has **surface tension**. Deformations cost energy proportional to curvature. Sharp features are suppressed.
- **Quantum mechanics:** The membrane has **negative surface tension**. Sharp features are energetically favourable. Interference fringes spontaneously emerge when wave packets collide.

In all three cases, the membrane has genuine inertia ($\tfrac{1}{2}\int|v|^2\rho\,dx$) and genuine internal forces (from $\delta F/\delta\rho$).

### Connecting to the √ρ Selection

The Fisher information $\int|\nabla\ln\rho|^2\rho\,dx$ can be rewritten as $4\int|\nabla\sqrt{\rho}|^2\,dx$ — the **Dirichlet energy of √ρ**. This is why √ρ is the natural object. The internal energy of the density-particle in Wasserstein space is the integral of the squared gradient of √ρ.

When Method 5 weights candidates by √ρ, it is directly coupling the particle dynamics to the gradient structure of √ρ — the degree of freedom that carries the internal energy. Each time step of Method 5 implements a single step of Wasserstein Newton's law:

1. The **classical push** ($v \cdot dt$) advances the density-particle along its current Wasserstein momentum.
2. The **√ρ-weighted selection** applies the internal force — the functional derivative of the Fisher information — correcting the momentum so the density-particle follows the correct quantum trajectory in $\mathcal{W}_2$ rather than the free-transport (classical) trajectory.

### The "Resistance" Is Real

The intuition that the distribution "resists being changed" corresponds precisely to the density's internal energy (Fisher information) acting as a restoring force on the Wasserstein geodesic. The osmotic pressure $\nu^2\partial_x\ln\rho$ is the stochastic-mechanical expression of this internal force, and the √ρ selection is the algorithmic mechanism that implements it.

The density evolves as a whole through Wasserstein space, with collective momentum and inertia. The quantum potential is the internal stress tensor distinguishing its motion from classical optimal transport. Method 5 implements the force law that determines this trajectory, one time step at a time.

#### References for Q4
- Otto, F. (2001). "The geometry of dissipative evolution equations: the porous medium equation." *Comm. PDE* 26, 101–174.
- Benamou, J.-D. & Brenier, Y. (2000). "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem." *Numer. Math.* 84, 375–393.
- Conforti, G. & Pavon, M. (2017). "Extremal flows on Wasserstein space." arXiv:1712.02257.
- Chen, Y., Georgiou, T.T. & Pavon, M. (2016). "On the relation between optimal transport and Schrödinger bridges." *J. Optim. Theory Appl.* 169, 671–691.
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS Graduate Studies in Mathematics.
- Project: `FBSDE_SchrodingerBridge_Nelson_Holland.md`, §§2.3, 3.4, 4.1, 7 (Conforti–Pavon action table and triangular relationship)
- Project: `Holland_Nelson_FokkerPlanck_Analysis.md`, §6 (implications and open questions)

---

## Summary of Key Insights

### 1. Time Symmetry
Method 5 naturally separates the T-even (osmotic) and T-odd (current) components of the quantum drift. The √ρ weight sits at the geometric midpoint between the forward and backward Schrödinger potentials, implementing the correct time-symmetric osmotic correction without log-density differentiation.

### 2. Local Sinkhorn
The global Sinkhorn/IPFP algorithm alternates forward and backward marginal corrections to converge to the Schrödinger bridge. Method 5 performs a local, infinitesimal version of this reweighting at every time step — one half-iteration per dt is sufficient because the correction is already small in the continuous-time limit.

### 3. Self-Consistent Feedback
The √ρ selection does not restore the old density; it constrains *how* diffusion reshapes the density, enforcing the quantum Fokker–Planck equation rather than the classical heat equation. The density evolves self-consistently: particles create the density, and the density guides the particles.

### 4. Wasserstein Dynamics
The density evolves as a genuine dynamical object on Wasserstein space, with collective momentum and inertia. The √ρ selection implements the internal force (Fisher information gradient) that distinguishes quantum trajectories from classical optimal transport. The "resistance" of the ensemble is the Wasserstein-geometric expression of the quantum potential.

### 5. The Unifying Role of √ρ
The quantity √ρ appears because: (a) it is the amplitude of the wavefunction ψ = √ρ · e^{iS/ℏ}; (b) it is the natural variable for the Fisher information (Dirichlet energy of √ρ); (c) it corresponds to a single Schrödinger potential in the bridge factorisation ρ = φ · φ̂; and (d) it induces exactly the correct half-log-density gradient needed for the osmotic drift.

---

## Complete Literature References

1. Nelson, E. (1966). "Derivation of the Schrödinger Equation from Newtonian Mechanics." *Phys. Rev.* 150, 1079–1085.
2. Nelson, E. (2012). "Review of stochastic mechanics." *J. Phys.: Conf. Ser.* 361, 012011.
3. Holland, P. (2021). "Eliminating the wavefunction from quantum dynamics: the bi-Hamilton–Jacobi theory." arXiv:2111.09235.
4. Schrödinger, E. (1931). "Über die Umkehrung der Naturgesetze." *Sitzungsber. Preuss. Akad. Wiss.*, 144–153.
5. Zambrini, J.-C. (1986). "Stochastic mechanics according to E. Schrödinger." *Phys. Rev. A* 33, 1532.
6. Pavon, M. & Wakolbinger, A. (1991). "On free energy, stochastic control, and Schrödinger processes." In *Modeling, Estimation and Control of Systems with Uncertainty*, 334–348.
7. Pavon, M. (1995). "Hamilton's principle in stochastic mechanics." *J. Math. Phys.* 36, 6774–6800.
8. Chen, Y., Georgiou, T.T. & Pavon, M. (2016). "On the relation between optimal transport and Schrödinger bridges: A stochastic control viewpoint." *J. Optim. Theory Appl.* 169, 671–691.
9. Chen, Y., Georgiou, T.T. & Pavon, M. (2021). "Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge." *SIAM Review* 63, 249–313.
10. Conforti, G. & Pavon, M. (2017). "Extremal flows on Wasserstein space." arXiv:1712.02257.
11. Otto, F. (2001). "The geometry of dissipative evolution equations: the porous medium equation." *Comm. PDE* 26, 101–174.
12. Benamou, J.-D. & Brenier, Y. (2000). "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem." *Numer. Math.* 84, 375–393.
13. Villani, C. (2003). *Topics in Optimal Transportation*. AMS Graduate Studies in Mathematics.
14. Sinkhorn, R. (1967). "Diagonal equivalence to matrices with prescribed row and column sums." *Amer. Math. Monthly* 74, 402–405.
15. Léonard, C. (2012). "A survey of the Schrödinger problem and some of its connections with optimal transport." *J. Funct. Anal.* 262, 1879–1920.
16. Harchaoui, Z., Liu, L. & Pal, S. (2020). "Asymptotics of entropy-regularized optimal transport via chaos decomposition." arXiv:2011.08963.
17. Fortet, R. (1940). Résolution d'un système d'équations de M. Schrödinger. *J. Math. Pures Appl.* 9, 83–105.
