# Holland's biHamilton-Jacobi Theory and Nelson's Stochastic Mechanics: A Comparative Analysis

## Focusing on Fokker-Planck, Backward Kolmogorov, and Coupled SDEs

---

## 1. Overview: Two Decompositions of the SchrÃ¶dinger Equation

Both Holland and Nelson begin from the same starting point â€” the SchrÃ¶dinger equation:

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi$$

and both arrive at formulations involving **two velocity fields** that can be identified with one another. The crucial difference is ontological and structural: Nelson *adds* a stochastic mechanism to quantum mechanics, while Holland reformulates the quantum state *deterministically* as a pair of coupled Hamilton-Jacobi-like equations. Yet the formal kinematic skeleton â€” in particular, the velocity fields, their relationship to probability, and the Fokker-Planck structure â€” is shared.

### The Polar Decomposition (Madelung)

Writing $\psi = \sqrt{\rho}\,e^{iS/\hbar}$, one obtains:

- **Continuity equation:** $\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho\, \mathbf{v}) = 0$, where $\mathbf{v} = \nabla S / m$
- **Quantum Hamilton-Jacobi equation:** $\frac{\partial S}{\partial t} + \frac{(\nabla S)^2}{2m} + V + Q = 0$, where $Q = -\frac{\hbar^2}{2m\sqrt{\rho}}\nabla^2\sqrt{\rho}$

### Holland's Key Move: The $\sigma_\pm$ Decomposition

Holland introduces the two real functions:

$$\sigma_\pm = S \pm \frac{\hbar}{2}\ln\rho$$

so that $\psi = e^{(1+i)\sigma_+/2\hbar}\,e^{(-1+i)\sigma_-/2\hbar}$. The SchrÃ¶dinger equation then becomes **two coupled HJ-like equations**:

$$\frac{\partial \sigma_+}{\partial t} + \frac{1}{2m}(\nabla\sigma_+)^2 + Q_+ + V = 0$$

$$\frac{\partial \sigma_-}{\partial t} + \frac{1}{2m}(\nabla\sigma_-)^2 + Q_- + V = 0$$

where the quantum potentials are:

$$Q_\pm = \mp\frac{\hbar}{2m}\nabla^2\sigma_\mp - \frac{1}{4m}[\nabla(\sigma_+ - \sigma_-)]^2$$

### Nelson's Key Move: The Stochastic Decomposition

Nelson postulates that the particle follows an ItÃ´ stochastic differential equation:

$$dx(t) = b(x(t),t)\,dt + dw(t)$$

where $w$ is a Wiener process with $\mathbb{E}_t[dw^i\,dw^j] = \frac{\hbar}{m}\delta^{ij}\,dt$ (diffusion coefficient $\nu = \frac{\hbar}{2m}$). He then introduces:

- **Forward drift (mean forward velocity):** $b^i = Dx^i = v^i + u^i$
- **Backward drift (mean backward velocity):** $b_*^i = D_*x^i = v^i - u^i$
- **Current velocity:** $v^i = \frac{1}{2}(b^i + b_*^i) = \frac{1}{m}\nabla^i S$
- **Osmotic velocity:** $u^i = \frac{1}{2}(b^i - b_*^i) = \frac{\hbar}{2m}\frac{\nabla^i\rho}{\rho} = \frac{\hbar}{2m}\nabla^i\ln\rho$

---

## 2. The Precise Identification of Velocity Fields

The connection between the two frameworks is immediate. Holland's velocity fields are:

$$v_\pm^i = \frac{1}{m}\nabla^i\sigma_\pm = \frac{1}{m}\nabla^i S \pm \frac{\hbar}{2m}\nabla^i\ln\rho$$

which means, using Nelson's notation:

$$v_+^i = v^i + u^i = b^i \quad \text{(Nelson's forward drift)}$$

$$v_-^i = v^i - u^i = b_*^i \quad \text{(Nelson's backward drift)}$$

Holland himself notes this identification explicitly in Â§2 of his paper, acknowledging that "within [the stochastic] scheme, the velocities (2.5) are interpreted as forward and backward drift velocities, $u_i = v_+^i - v_-^i = (\hbar/m)\nabla_i\ln\rho$ is the osmotic velocity, the local mean $v_i = \frac{1}{2}(v_+^i + v_-^i)$ is the de Broglie-Bohm velocity, and equations (4.17) below are the corresponding Fokker-Planck equations."

The key quantities then relate as:

| Nelson's notation | Holland's notation | Expression |
|---|---|---|
| $b^i$ (forward drift) | $v_+^i$ | $\frac{1}{m}\nabla^i\sigma_+$ |
| $b_*^i$ (backward drift) | $v_-^i$ | $\frac{1}{m}\nabla^i\sigma_-$ |
| $v^i$ (current velocity) | $\frac{1}{2}(v_+^i + v_-^i)$ | $\frac{1}{m}\nabla^i S$ |
| $u^i$ (osmotic velocity) | $\frac{1}{2}(v_+^i - v_-^i)$ | $\frac{\hbar}{2m}\nabla^i\ln\rho$ |

And the probability density in Holland's notation is:

$$\rho = \exp[(\sigma_+ - \sigma_-)/\hbar]$$

---

## 3. The Fokker-Planck Equations and the Backward Kolmogorov Equation

This is the mathematical heart of the comparison. Both frameworks give rise to a **pair** of transport equations for $\rho$, which are naturally identified as a **forward Fokker-Planck** (FPE) and its **adjoint** (the backward Kolmogorov equation, BKE).

### 3.1 Holland's Fokker-Planck Pair (Equation 4.17)

Holland derives the following pair in Â§4.2 of his paper by setting $v_a^i = v_\pm^i$ in the general transport equation (4.9):

$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\,v_+^i) = +\frac{\hbar}{2m}\nabla^2\rho \tag{FPE+}$$

$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\,v_-^i) = -\frac{\hbar}{2m}\nabla^2\rho \tag{FPE-}$$

Let us unpack these. With $\nu = \hbar/(2m)$:

**Equation (FPE+): The Forward Fokker-Planck Equation**

$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\,v_+) = \nu\,\nabla^2\rho$$

Expanding the divergence and rearranging:

$$\frac{\partial\rho}{\partial t} = -\nabla\cdot(\rho\,v_+) + \nu\,\nabla^2\rho$$

This is the **forward Fokker-Planck equation** (also called the forward Kolmogorov equation) for a diffusion process with drift $v_+ = b$ (Nelson's forward drift) and diffusion coefficient $\nu$. In standard form:

$$\mathcal{L}\rho = \frac{\partial\rho}{\partial t} + \nabla\cdot(b\,\rho) - \nu\,\nabla^2\rho = 0$$

This governs the **forward-in-time** evolution of the probability density $\rho(x,t)$.

**Equation (FPE-): The Backward Kolmogorov Equation (Adjoint)**

$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\,v_-) = -\nu\,\nabla^2\rho$$

Rearranging:

$$\frac{\partial\rho}{\partial t} = -\nabla\cdot(\rho\,v_-) - \nu\,\nabla^2\rho$$

Now, expanding $\nabla\cdot(\rho\,v_-)$ = $v_-\cdot\nabla\rho + \rho\,\nabla\cdot v_-$ and using the osmotic equation $u = \nu\,\nabla\ln\rho$ (so $\nabla\cdot v_+ - \nabla\cdot v_- = 2\nabla\cdot u$), one can show this is equivalent to the **backward Kolmogorov equation** â€” the adjoint of the forward FPE â€” for the backward drift $v_- = b_*$.

The adjoint relationship is precisely what your Screenshot101 shows: if the forward operator is $\mathcal{L}\rho = \partial_t\rho + \partial_x(b\rho) - \nu\partial_{xx}\rho$, then the **adjoint operator** acting on a test function $f$ is:

$$\mathcal{L}^*f = -\partial_t f - b\,\partial_x f - \nu\,\partial_{xx}f = 0$$

In the Nelson/Holland context, the forward FPE uses drift $b = v_+$ with **positive** diffusion, and the adjoint (backward Kolmogorov) uses drift $b_* = v_-$ with the sign of diffusion **reversed**. The two equations are not independent â€” they are adjoints of one another, and their consistency is guaranteed by the SchrÃ¶dinger equation.

### 3.2 Nelson's Derivation of the Same Pair

In Nelson's framework, the forward SDE is:

$$dx = b(x,t)\,dt + dw, \qquad dw\sim\mathcal{N}(0,2\nu\,dt)$$

The Fokker-Planck equation for the transition density of this ItÃ´ process is:

$$\frac{\partial\rho}{\partial t} = -\nabla\cdot(b\,\rho) + \nu\,\nabla^2\rho$$

which is exactly Holland's (FPE+) with $b = v_+$.

By time-reversal symmetry, Nelson also considers the **backward** SDE:

$$dx = b_*(x,t)\,dt + dw_*, \qquad dw_*\sim\mathcal{N}(0,2\nu\,dt)$$

where $dw_*$ is a backward Wiener process. The associated backward Fokker-Planck (i.e., the backward Kolmogorov equation) is:

$$\frac{\partial\rho}{\partial t} = -\nabla\cdot(b_*\,\rho) - \nu\,\nabla^2\rho$$

which is exactly Holland's (FPE-) with $b_* = v_-$.

### 3.3 The Key Insight: Adding vs. Subtracting the Pair

**Adding** (FPE+) and (FPE-):

$$2\frac{\partial\rho}{\partial t} + \nabla\cdot[\rho(v_+ + v_-)] = 0$$

$$\Rightarrow\quad \frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\,v) = 0$$

This is the **continuity equation** â€” conservation of probability carried by the current velocity $v = \frac{1}{2}(v_+ + v_-) = \nabla S/m$. This is the first Madelung equation (1.2 in Holland's paper).

**Subtracting** (FPE+) from (FPE-):

$$\nabla\cdot[\rho(v_+ - v_-)] = 2\nu\,\nabla^2\rho$$

$$\Rightarrow\quad \nabla\cdot(\rho\,u) = \nu\,\nabla^2\rho$$

which, using $u = \nu\nabla\ln\rho$, is an identity â€” the **osmotic equation**. This is automatically satisfied and provides no new information; it defines the relationship between osmotic velocity and probability density.

### 3.4 The Source Terms and Non-Conservation Along Individual Flows

A critical result in Holland's paper (Â§4.2) is that **neither the $v_+$ nor the $v_-$ trajectories individually conserve probability**. Each Fokker-Planck equation has a source term ($\pm\nu\nabla^2\rho$). The probability density $\rho$ cannot be identified with the trajectory density of either congruence. Instead, $\rho$ is determined by the **difference in action functions**:

$$\rho = e^{(\sigma_+ - \sigma_-)/\hbar}$$

This is Holland's equation (2.7) and is deeply connected to the structure: probability arises from the relative "phase" between the two HJ flows, not from the bunching of trajectories in either flow alone.

---

## 4. Holland's Theory as Coupled Stochastic Differential Equations

Holland's theory is purely deterministic, but its formal structure maps directly onto a pair of coupled SDEs in the Nelson sense. The mapping is:

### 4.1 The Forward SDE (associated with $\sigma_+$)

$$dX_t = v_+(X_t, t)\,dt + \sqrt{2\nu}\,dW_t^{(+)}$$

where $v_+ = \frac{1}{m}\nabla\sigma_+$ and $\nu = \frac{\hbar}{2m}$, with $W^{(+)}$ a forward Wiener process.

### 4.2 The Backward SDE (associated with $\sigma_-$)

$$dX_t = v_-(X_t, t)\,dt + \sqrt{2\nu}\,dW_t^{(-)}$$

where $v_- = \frac{1}{m}\nabla\sigma_-$ and $W^{(-)}$ is a backward Wiener process.

### 4.3 The Coupling

These are not independent SDEs â€” they describe the **same** particle. The coupling comes through the fact that $v_+$ and $v_-$ are both determined by $\sigma_+$ and $\sigma_-$, which in turn satisfy the coupled bi-HJ system (2.2)â€“(2.3). In explicit Eulerian form, the coupling structure is:

**Coupled PDE system for the drift fields:**

$$\frac{\partial\sigma_+}{\partial t} + \frac{1}{2m}(\nabla\sigma_+)^2 - \frac{\hbar}{2m}\nabla^2\sigma_- - \frac{1}{4m}[\nabla(\sigma_+-\sigma_-)]^2 + V = 0$$

$$\frac{\partial\sigma_-}{\partial t} + \frac{1}{2m}(\nabla\sigma_-)^2 + \frac{\hbar}{2m}\nabla^2\sigma_+ - \frac{1}{4m}[\nabla(\sigma_+-\sigma_-)]^2 + V = 0$$

The forward SDE generates the forward Fokker-Planck (FPE+), and the backward SDE generates (FPE-). Both are consistent with the same $\rho(x,t)$.

### 4.4 An Equivalent Coupled System in $(v, u)$ Variables

Using $v = \frac{1}{2}(v_+ + v_-)$ (current velocity) and $u = \frac{1}{2}(v_+ - v_-)$ (osmotic velocity), the coupled SDE pair can also be written as a single SDE with two drift components:

$$dX_t = [v(X_t,t) + u(X_t,t)]\,dt + \sqrt{2\nu}\,dW_t \qquad\text{(forward)}$$

$$dX_t = [v(X_t,t) - u(X_t,t)]\,dt + \sqrt{2\nu}\,dW_t^* \qquad\text{(backward)}$$

with the constraint (from the osmotic equation):

$$u^i = \nu\,\frac{\nabla^i\rho}{\rho} = \nu\,\nabla^i\ln\rho$$

and the current equation $\partial_t\rho = -\nabla\cdot(v\rho)$ enforcing the overall probability conservation.

### 4.5 The Coupled Dynamics in Nelson's "Mean Acceleration" Form

Nelson's stochastic Newton equation provides the dynamical closure:

$$m\,a = -\nabla V$$

where the **mean acceleration** is:

$$a = \frac{1}{2}(DD_* + D_*D)x$$

In terms of $v$ and $u$, this becomes (Nelson 1966, Eq. 30):

$$m\left[\frac{\partial v}{\partial t} + (v\cdot\nabla)v - (u\cdot\nabla)u - \nu\nabla^2 u\right] = -\nabla V$$

This is equivalent to the quantum Hamilton-Jacobi equation and provides the dynamical content that couples the two SDEs. In Holland's formulation, this same content is encoded in the coupled acceleration equations (3.3)â€“(3.4).

### 4.6 Summary: The Coupled SDE System

The complete stochastic model of Holland's bi-HJ theory consists of:

**State:** the pair $(\sigma_+(x,t), \sigma_-(x,t))$ or equivalently $(S(x,t), R(x,t))$ where $R = \frac{\hbar}{2}\ln\rho$.

**SDEs (in ItÃ´ form, 1D for clarity):**

$$dX_t = \frac{1}{m}\frac{\partial\sigma_+}{\partial x}(X_t,t)\,dt + \sqrt{\frac{\hbar}{m}}\,dW_t \qquad\text{(forward process)}$$

$$dX_t = \frac{1}{m}\frac{\partial\sigma_-}{\partial x}(X_t,t)\,dt + \sqrt{\frac{\hbar}{m}}\,d\widetilde{W}_t \qquad\text{(backward process)}$$

**Field equations (coupling):**

$$\partial_t\sigma_+ + \frac{(\partial_x\sigma_+)^2}{2m} + Q_+ + V = 0$$

$$\partial_t\sigma_- + \frac{(\partial_x\sigma_-)^2}{2m} + Q_- + V = 0$$

where $Q_\pm = \mp\frac{\hbar}{2m}\partial_{xx}\sigma_\mp - \frac{1}{4m}[\partial_x(\sigma_+ - \sigma_-)]^2$.

**Transport equations (consistency):**

$$\partial_t\rho + \partial_x(\rho\,v_+) = \nu\,\partial_{xx}\rho \qquad\text{(forward Fokker-Planck)}$$

$$\partial_t\rho + \partial_x(\rho\,v_-) = -\nu\,\partial_{xx}\rho \qquad\text{(backward Kolmogorov)}$$

with $\rho = e^{(\sigma_+ - \sigma_-)/\hbar}$.

---

## 5. Deep Structural Comparison

### 5.1 Time Reversal

This is where the two theories diverge most strikingly.

**Nelson:** Under time reversal $t \to -t$, the forward and backward processes exchange roles: $b \leftrightarrow b_*$, $v \to -v$, $u \to +u$, $dW \to d\widetilde{W}$. The current velocity reverses (it tracks momentum), the osmotic velocity does not (it tracks density gradients). This is the standard T-symmetry of stochastic processes.

**Holland:** Under time reversal, the two action functions undergo a non-standard exchange:

$$\sigma_\pm'(x',t') = -\sigma_\mp(x,t), \qquad v_\pm'^i(x',t') = -v_\mp^i(x,t)$$

The velocities do **not** simply reverse sign individually. Instead, each maps to the **negative of the other**. Holland describes this as "effective T-covariance through the collective behaviour of elements that individually disobey T." This is a deeper symmetry than Nelson's framework reveals: it shows that time-reversal in quantum mechanics acts as a **swap** between the two HJ congruences, not merely a sign flip.

### 5.2 The Role of Probability

**Nelson:** $\rho = |\psi|^2$ is the probability density of the Markov process at time $t$. It is conserved along the current velocity via the continuity equation. The osmotic velocity encodes the gradient of $\rho$ and ensures the forward/backward Fokker-Planck pair is satisfied.

**Holland:** $\rho = e^{(\sigma_+ - \sigma_-)/\hbar}$ is determined by the **action difference**, not by trajectory bunching. The individual congruence densities $\rho_0\,J_\pm^{-1}$ do not reproduce $\rho$ (proved in Â§4.2). The peaks of $\rho$ occur where $v_+ = v_-$ (i.e., where osmotic velocity vanishes), not where trajectories cluster.

### 5.3 Determinism vs. Stochasticity

**Holland:** The bi-HJ trajectories are **deterministic**. The two congruences $q_\pm^i(q_{\pm 0}, t)$ are solutions to coupled ODEs (Newton-like second-order equations 3.3â€“3.4). There is no noise term. The Fokker-Planck structure emerges from the non-conservation of probability along each individual congruence (the "source" terms), but this is a feature of the decomposition, not of any physical randomness.

**Nelson:** The trajectories are **stochastic**. The Wiener process $dW$ represents genuine physical noise â€” a "subquantum" Brownian motion. The Fokker-Planck equations are *bona fide* transport equations for the probability of a random process. Nelson's 2012 review acknowledges that this leads to difficulties: stochastic mechanics gives wrong predictions for correlations at different times (the entanglement problem), and Nelson himself concluded that "stochastic mechanics is an approximation to a correct theory of quantum mechanics as emergent."

### 5.4 The Gaussian Example

For a free Gaussian wavefunction at rest:

$$\rho(x,t) = (2\pi\sigma^2)^{-1/2}e^{-x^2/2\sigma^2}, \qquad S(x,t) = \frac{\hbar\kappa t\,x^2}{4\sigma^2} - \frac{\hbar}{2}\arctan(\kappa t)$$

where $\sigma = \sigma_0(1+\kappa^2 t^2)^{1/2}$ and $\kappa = \hbar/(2m\sigma_0^2)$.

**Holland's bi-HJ trajectories:**

$$q_\pm(q_{\pm 0}, t) = q_{\pm 0}(1+\kappa^2 t^2)^{1/2}\,e^{\mp\arctan(\kappa t)}$$

The exponential factors $e^{\mp\arctan(\kappa t)}$ exhibit dissipative/anti-dissipative behaviour â€” one congruence contracts, the other expands. This is the hallmark of the forward/backward process asymmetry.

**Nelson's stochastic trajectories** would be sample paths of the SDE with drift $v_+ = \kappa t\,x/(1+\kappa^2 t^2) + \hbar x/(2m\sigma^2)$ and diffusion $\sqrt{\hbar/m}$, producing the same statistical distribution.

---

## 6. Implications and Open Questions

1. **Numerical methods:** Holland's coupled bi-HJ equations could be solved numerically as a system of SDEs (even though the underlying theory is deterministic). The forward/backward SDE pair provides a natural Monte Carlo sampling framework. This is closely related to the **forward-backward stochastic differential equation (FBSDE)** framework used in mathematical finance and optimal control.

2. **The FBSDE connection:** The structure $dX_t = b(X_t,t)dt + \sigma\,dW_t$ coupled with a backward equation for $\sigma_-$ or $Y_t$ is precisely the structure of an FBSDE system. The stochastic HJ equation (Holland's bi-HJ) plays the role of the backward component. Recent work by Pavon, Pal, and others on SchrÃ¶dinger bridges makes this connection explicit.

3. **Stochastic optimal control:** The SchrÃ¶dinger equation can be recast as a stochastic optimal control problem (Zambrini, 1986; Pavon & Wakolbinger, 1991). In that framework, $\sigma_+$ and $\sigma_-$ are the value functions of the forward and backward control problems, and the Fokker-Planck/backward Kolmogorov pair is the optimality system. Holland's bi-HJ equations are exactly the Hamilton-Jacobi-Bellman equations of this control problem.

4. **Nelson's own retreat:** In his 2012 review, Nelson acknowledged that stochastic mechanics fails for multi-time correlations and entanglement. He wrote: "The most natural explanation is that stochastic mechanics is an approximation to a correct theory of quantum mechanics as emergent." Holland's deterministic reformulation sidesteps these issues entirely, since it is an exact reformulation, not an additional physical hypothesis.

---

## 7. Summary Table

| Feature | Nelson (1966) | Holland (2021) |
|---------|--------------|----------------|
| **Nature of trajectories** | Stochastic (Markov/ItÃ´) | Deterministic (Lagrangian congruences) |
| **Forward velocity** $v_+$ | Forward drift $b$ of SDE | $\nabla\sigma_+/m$ |
| **Backward velocity** $v_-$ | Backward drift $b_*$ | $\nabla\sigma_-/m$ |
| **Diffusion coefficient** | $\nu = \hbar/(2m)$ | N/A (no noise), but $\nu$ appears in FPE source |
| **Fokker-Planck** | Forward FPE for forward SDE | Eq. (4.17+): $\partial_t\rho + \nabla\cdot(\rho v_+) = \nu\nabla^2\rho$ |
| **Backward Kolmogorov** | Adjoint FPE for backward SDE | Eq. (4.17-): $\partial_t\rho + \nabla\cdot(\rho v_-) = -\nu\nabla^2\rho$ |
| **Probability origin** | $\rho$ = density of Markov process | $\rho = e^{(\sigma_+-\sigma_-)/\hbar}$ (action difference) |
| **Time reversal** | $b \leftrightarrow b_*$, $v\to -v$ | $\sigma_\pm' = -\sigma_\mp$ (non-standard exchange) |
| **Dynamical equation** | Stochastic Newton: $ma = -\nabla V$ | Coupled bi-HJ: Eqs. (2.2)-(2.3) |
| **Conservation** | $\rho$ conserved along $v$ | $\rho$ NOT conserved along $v_+$ or $v_-$ individually |
| **Exactness** | Approximate (fails for entanglement) | Exact reformulation of SchrÃ¶dinger |

---

*This analysis draws on Holland (2021, arXiv:2111.09235), Nelson (Phys. Rev. 150, 1079, 1966), Nelson (J. Phys.: Conf. Ser. 361, 012011, 2012), and the general theory of forward-backward stochastic differential equations.*
