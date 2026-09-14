# Forward-Backward SDEs, Schrödinger Bridges, and the Nelson–Holland Connection

> A synthesis connecting three formalisms: forward-backward stochastic differential equations (FBSDEs, Pardoux–Peng / Ma–Protter–Yong), Schrödinger's bridge problem (the most-likely evolution conditional on a rare large-deviation event, entropic optimal transport), and the Nelson–Holland stochastic-mechanics picture of quantum dynamics. Traces how M5's forward-evolving ensemble with GH-quadrature readouts realizes the FBSDE/bridge structure without ever solving a backward equation directly.

## A Synthesis of Stochastic Control, Optimal Transport, and Quantum Mechanics

---

## 1. The FBSDE Framework: Origins and Structure

### 1.1 What is a Forward-Backward SDE?

A forward-backward stochastic differential equation (FBSDE) couples an ordinary forward Itô SDE with a *backward* SDE (BSDE) that runs from a terminal condition back in time. The general structure, formalized by Pardoux & Peng (1990) and Ma, Protter & Yong (1994), is:

**Forward SDE:**
$$dX_t = \mu(t, X_t, Y_t)\thinspace{}dt + \sigma(t, X_t, Y_t)\thinspace{}dW_t, \qquad X_0 = x_0$$

**Backward SDE:**
$$dY_t = -f(t, X_t, Y_t, Z_t)\thinspace{}dt + Z_t\thinspace{}dW_t, \qquad Y_T = g(X_T)$$

The forward process $X_t$ evolves from an initial condition into the future; the backward process $(Y_t, Z_t)$ is determined by a *terminal* condition $g(X_T)$ and propagates information backward. The auxiliary process $Z_t$ arises from the martingale representation theorem — it encodes how $Y_t$ responds to the noise driving $X_t$.

The coupling is the essential feature: the forward drift $\mu$ and diffusion $\sigma$ may depend on the backward solution $(Y, Z)$, and the backward driver $f$ depends on the forward state $X$. This creates a two-point boundary value problem in the stochastic setting, which is generically much harder than either component alone.

### 1.2 The Nonlinear Feynman–Kac Connection

The deep mathematical content of FBSDEs lies in their connection to nonlinear parabolic PDEs via the **nonlinear Feynman–Kac formula**. If $(X_t, Y_t, Z_t)$ solves the FBSDE, and we define $u(t,x) = Y_t$ when $X_t = x$, then under regularity conditions:

$$Y_t = u(t, X_t), \qquad Z_t = \sigma(t, X_t)\nabla_x u(t, X_t)$$

and $u$ satisfies a semilinear parabolic PDE — typically a **Hamilton–Jacobi–Bellman (HJB) equation**:

$$\partial_t u + \frac{1}{2}\sigma^2 \Delta u + \mu \cdot \nabla u + f(t, x, u, \sigma \nabla u) = 0, \qquad u(T,x) = g(x)$$

This is the stochastic counterpart of characteristics for PDEs: the forward SDE traces the characteristic curves, the backward SDE tracks the value function along them.

### 1.3 Applications in Finance and Control

In **mathematical finance**, the FBSDE framework provides the natural language for:

- **Option pricing and hedging:** $X_t$ is the underlying asset, $Y_t$ is the option price, $Z_t$ is the hedging portfolio. The backward equation propagates the terminal payoff $g(X_T)$ backward through time.
- **Recursive utility and stochastic differential utility:** The cost/utility is defined implicitly through the BSDE, generalizing classical dynamic programming.
- **Mean-field games:** When agents interact through the population distribution, the resulting McKean–Vlasov FBSDEs couple forward (Fokker–Planck for the population density) with backward (HJB for individual optimality).

In **stochastic optimal control**, the FBSDE encodes the Pontryagin stochastic maximum principle. For a controlled diffusion $dX_t = b(X_t, u_t)\thinspace{}dt + \sigma\thinspace{}dW_t$ minimizing $\mathbb{E}[\int_0^T L(X_t, u_t)\thinspace{}dt + g(X_T)]$, the optimality conditions yield:

- **Forward:** Fokker–Planck equation for the state density $\rho$
- **Backward:** Hamilton–Jacobi–Bellman equation for the value function $\varphi$
- **Coupling:** $u_t^* = \arg\min_u [b(x,u)\cdot\nabla\varphi + L(x,u)]$

The value function $\varphi$ satisfies the BSDE (or equivalently the HJB PDE), while the optimal state evolves forward under the optimized drift. This Fokker–Planck / HJB pair is the prototype of all FBSDE systems in control theory.

---

## 2. The Schrödinger Bridge Problem

### 2.1 Schrödinger's Hot Gas Gedankenexperiment (1931–32)

In 1931, Erwin Schrödinger posed the following question. Consider $N$ independent Brownian particles at thermal equilibrium. At time $t = 0$, observe their empirical distribution $\rho_0$. Let them evolve according to the equilibrium Langevin dynamics up to time $t = 1$, and observe the empirical distribution $\rho_1$. By the law of large numbers, $\rho_1$ should be close to the equilibrium measure $m$. But suppose we observe an "unexpected" configuration — $\rho_1$ significantly different from $m$. What is the **most likely** evolution of the particle system *conditional on this rare event*?

By Sanov's theorem (large deviations of empirical measures), the most likely evolution minimizes the **relative entropy** $D(Q\|P)$ of the path measure $Q$ with respect to the equilibrium measure $P$, subject to the marginal constraints $Q_0 = \rho_0$ and $Q_1 = \rho_1$:

$$\text{Minimize } D(Q\|P) \text{ over } Q \in \mathcal{P}(\rho_0, \rho_1)$$

The solution $Q^*$ is the **Schrödinger bridge** between $\rho_0$ and $\rho_1$.

Schrödinger himself noted (in the paper's title, "Über die Umkehrung der Naturgesetze") "Merkwürdige Analogien zur Quantenmechanik, die mir sehr des Hindenkens wert erscheinen" — remarkable analogies to quantum mechanics that seem well worth pondering.

### 2.2 The Schrödinger System

If the prior $P$ is Markovian (e.g., Brownian motion with diffusion $\nu = \varepsilon/2$), the Schrödinger bridge $Q^*$ is also Markovian (Jamison, 1974) and has a special structure. Its transition density factorizes as:

$$q^*(x, t \thinspace{}|\thinspace{} y, 0) = \hat{\varphi}(x,t) \cdot p(x,t\thinspace{}|\thinspace{}y,0) \cdot \varphi(y,0)$$

where $p$ is the prior transition kernel and $(\varphi, \hat{\varphi})$ are **Schrödinger potentials** satisfying the coupled integral equations (the **Schrödinger system**):

$$\hat{\varphi}(x,1) \int p(x,1\thinspace{}|\thinspace{}y,0)\thinspace{}\varphi(y,0)\thinspace{}\rho_0(y)\thinspace{}dy = \rho_1(x)$$
$$\varphi(y,0) \int p(x,1\thinspace{}|\thinspace{}y,0)\thinspace{}\hat{\varphi}(x,1)\thinspace{}\rho_1(x)\thinspace{}dx = \rho_0(y)$$

These can be solved iteratively (the **Sinkhorn algorithm** or iterative proportional fitting procedure, IPFP), which alternately normalizes rows and columns — a procedure that converges to the unique bridge under mild conditions (Fortet 1940, Beurling 1960, Föllmer 1988).

### 2.3 Fluid Dynamic Formulation: The Stochastic Control Problem

The pivotal modern development, recognized since the early 1990s by Zambrini (1986), Dai Pra (1991), and Pavon & Wakolbinger (1991), is that the Schrödinger bridge can be reformulated as a **stochastic optimal control problem**. When the prior is Wiener measure with diffusion $\varepsilon$, the bridge marginal flow solves:

$$\inf_{(\rho, v)} \int_0^1 \int_{\mathbb{R}^d} \left[\frac{1}{2}|v_t(x)|^2 + \frac{\varepsilon^2}{8}|\nabla\ln\rho_t(x)|^2\right]\rho_t(x)\thinspace{}dx\thinspace{}dt$$

subject to the **continuity equation** $\partial_t\rho + \nabla\cdot(v\rho) = 0$ and boundary conditions $\rho(0) = \rho_0$, $\rho(1) = \rho_1$.

The first term $\frac{1}{2}|v|^2\rho$ is kinetic energy — identical to the Benamou–Brenier formulation of optimal mass transport (OMT). The second term $\frac{\varepsilon^2}{8}|\nabla\ln\rho|^2\rho$ is the **Fisher information** of $\rho$. The Schrödinger bridge therefore differs from optimal transport by precisely this Fisher information penalty, and OMT is recovered in the zero-noise limit $\varepsilon \to 0$. This key observation was made explicit by Chen, Georgiou & Pavon (2016), resolving a question posed by Carlen in 2006.

### 2.4 The FBSDE Structure of the Schrödinger Bridge

The optimality conditions for the Schrödinger bridge control problem yield a coupled FBSDE system. Defining $\varphi = \ln\hat{\varphi}$ and $\hat{\varphi} = \ln\varphi$ as log-potentials, the optimal evolution satisfies:

**Forward Fokker–Planck:**
$$\partial_t\rho + \nabla\cdot(\rho\thinspace{}b) = \nu\thinspace{}\nabla^2\rho, \qquad b = v + \nu\nabla\ln\rho$$

**Backward HJB (for the value function $\varphi$):**
$$\partial_t\varphi + \frac{1}{2}|\nabla\varphi|^2 + \nu\thinspace{}\nabla^2\varphi = 0$$

**Forward HJB (for $\hat{\varphi}$):**
$$\partial_t\hat{\varphi} + \frac{1}{2}|\nabla\hat{\varphi}|^2 - \nu\thinspace{}\nabla^2\hat{\varphi} = 0$$

The density is $\rho = e^{\hat{\varphi} - \varphi}$ (up to normalization), and the optimal drifts are $b = \nabla\hat{\varphi}$ (forward) and $b_* = -\nabla\varphi$ (backward). This is a **forward-backward system**: the forward FPE propagates the density forward in time, while the backward HJB propagates the value function backward from the terminal condition.

The stochastic version takes the form of the coupled SDEs:

$$dX_t = \nabla\hat{\varphi}(X_t,t)\thinspace{}dt + \sqrt{2\nu}\thinspace{}dW_t \qquad\text{(forward)}$$
$$dX_t = -\nabla\varphi(X_t,t)\thinspace{}dt + \sqrt{2\nu}\thinspace{}dW_t^* \qquad\text{(backward)}$$

where $W^*$ is a backward Wiener process.

---

## 3. The Connection to Nelson's Stochastic Mechanics

### 3.1 Nelson's Framework as a Schrödinger Bridge

The structural parallel between the Schrödinger bridge FBSDE and Nelson's stochastic mechanics is not a coincidence — it is a precise mathematical identification, developed most thoroughly by Pavon and collaborators.

In Nelson's stochastic mechanics (1966), a quantum particle follows:

$$dx = b(x,t)\thinspace{}dt + dw, \qquad \mathbb{E}_t[dw^i dw^j] = \frac{\hbar}{m}\delta^{ij}\thinspace{}dt$$

with forward drift $b = v + u$ and backward drift $b_* = v - u$, where $v = \nabla S/m$ is the current velocity and $u = (\hbar/2m)\nabla\ln\rho$ is the osmotic velocity. The density $\rho = |\psi|^2$ satisfies:

- **Forward FPE:** $\partial_t\rho + \nabla\cdot(\rho\thinspace{}b) = \nu\nabla^2\rho$
- **Backward Kolmogorov:** $\partial_t\rho + \nabla\cdot(\rho\thinspace{}b_*) = -\nu\nabla^2\rho$

with $\nu = \hbar/(2m)$.

Now compare with the Schrödinger bridge system in §2.4. Making the identifications:

| Schrödinger Bridge | Nelson's Mechanics | Holland's bi-HJ |
|---|---|---|
| $\hat{\varphi}$ (forward potential) | $R + S/\hbar$ | $\sigma_+/\hbar$ |
| $\varphi$ (backward potential) | $R - S/\hbar$ | $-\sigma_-/\hbar$ |
| $\nabla\hat{\varphi}$ (forward drift) | $b = v + u$ | $v_+ = \nabla\sigma_+/m$ |
| $-\nabla\varphi$ (backward drift) | $b_* = v - u$ | $v_- = \nabla\sigma_-/m$ |
| $\rho = e^{\hat{\varphi}-\varphi}$ | $\rho = e^{2R/\hbar}$ | $\rho = e^{(\sigma_+-\sigma_-)/\hbar}$ |
| Fisher information | Osmotic kinetic energy $\frac{1}{2}|u|^2$ | Source terms in FPE± |
| $\nu = \varepsilon/2$ | $\nu = \hbar/(2m)$ | $\nu = \hbar/(2m)$ |

The identification is exact. Nelson's stochastic mechanics *is* a Schrödinger bridge problem where the diffusion coefficient is $\nu = \hbar/(2m)$, and the dynamics (the stochastic Newton equation $ma = -\nabla V$) selects the particular bridge that corresponds to the Schrödinger equation.

### 3.2 Pavon's Program: Quantum Schrödinger Bridges

Pavon has developed this connection systematically across several decades:

**Pavon & Wakolbinger (1991):** Established the link between Schrödinger processes and stochastic optimal control via free energy minimization. Showed that the Schrödinger bridge arises from minimizing relative entropy subject to marginal constraints, and that this is equivalent to a stochastic control problem whose optimality conditions yield the forward-backward FPE/HJB system.

**Pavon (1995), "Hamilton's principle in stochastic mechanics":** Developed a full variational principle within Nelson's stochastic mechanics. Showed that the stochastic Hamilton–Jacobi equation of Nelson (which is Holland's Ïƒ₊ equation) and the osmotic equation together arise from an action principle in which the kinetic energy includes both the current velocity contribution $\frac{1}{2}|v|^2$ and the osmotic contribution $\frac{1}{2}|u|^2$ — precisely the two terms in the Schrödinger bridge action.

**Pavon (2003), "Quantum Schrödinger Bridges":** Made the quantum mechanical connection fully explicit. Showed that wave-packet collapse in the von Neumann sense can be understood through a variational principle within Nelson's stochastic mechanics that parallels the Schrödinger bridge problem but with quantum-mechanical kinematics. Invoked Schrödinger's own 1931 observation about "remarkable analogies to quantum mechanics."

**Beghi, Ferrante & Pavon (2001), "How to steer a quantum system over a Schrödinger bridge":** Used Nelson's stochastic mechanics as the kinematic framework for a quantum steering problem. Given initial and final wavefunctions $\psi_0$ and $\psi_1$ (hence initial and final densities $|\psi_0|^2$ and $|\psi_1|^2$), the Nelson process of a reference quantum evolution serves as the prior in a Schrödinger bridge problem. The bridge solution yields the controlled quantum evolution achieving the transfer.

### 3.3 Chen, Georgiou & Pavon: The Unified Framework

The major synthesis came in a series of papers by Yongxin Chen, Tryphon Georgiou, and Michele Pavon (2015–2021), culminating in the comprehensive SIAM Review article "Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge" (2021). Their key contributions:

1. **Fluid dynamic formulation of Schrödinger bridges** (2016): The bridge flow minimizes kinetic energy *plus* Fisher information, with the continuity equation as constraint. This directly parallels the Benamou–Brenier formulation of optimal transport, with the Fisher information as the distinguishing term. This solves Carlen's 2006 question about the relationship between the two problems.

2. **Optimal transport with prior** (2016): Generalized OMT to allow a non-trivial prior Markovian evolution (not just Brownian motion). The Schrödinger bridge with general prior converges to "optimal transport with prior" in the zero-noise limit. This unifies the classical OMT and SBP within a single parametric family.

3. **Stochastic control viewpoint** (2016): Showed that the SBP can be reformulated as a stochastic control problem with atypical (two-point marginal) boundary conditions. The optimal drift is determined by a forward-backward system — a Fokker–Planck equation coupled with a Hamilton–Jacobi–Bellman equation — which is precisely an FBSDE.

4. **Connection to the Madelung fluid** (2021): In their SIAM Review paper, they explicitly discuss the connection to quantum mechanics through the Madelung hydrodynamic formulation, Nelson's stochastic mechanics, and Bohm's pilot-wave theory, situating all three as special cases or limits of the general Schrödinger bridge framework.

### 3.4 Conforti & Pavon: Extremal Flows on Wasserstein Space

The geometric unification was achieved in Conforti & Pavon (2017), "Extremal flows on Wasserstein space." They showed that the solution flows of OMT, SBP, and Nelson's stochastic mechanics can all be characterized as **critical points of action functionals on Wasserstein space** $`\mathcal{W}_2`$. The actions differ only in the presence or sign of a Fisher information functional:

| Problem | Action Functional |
|---|---|
| **Optimal Mass Transport** | $\int_0^1\int\frac{1}{2}|v|^2\rho\thinspace{}dx\thinspace{}dt$ |
| **Schrödinger Bridge** | $\int_0^1\int\left[\frac{1}{2}|v|^2 + \frac{\varepsilon^2}{8}|\nabla\ln\rho|^2\right]\rho\thinspace{}dx\thinspace{}dt$ |
| **Nelson / Madelung** | $\int_0^1\int\left[\frac{1}{2}|v|^2 - \frac{\nu^2}{2}|\nabla\ln\rho|^2 - V\right]\rho\thinspace{}dx\thinspace{}dt$ |

The sign flip on the Fisher information term between SBP (+) and the Madelung fluid (−) is the key structural distinction. The SBP penalizes density gradients (entropy production), while the quantum action *rewards* them (through the quantum potential $Q$, which can be written as $Q = -\frac{\nu^2}{2}|\nabla\ln\rho|^2 - \nu^2\nabla^2\ln\sqrt{\rho}$).

Conforti & Pavon proved that in all three cases, the critical flow satisfies a **Newton-like second law in Wasserstein space** — a covariant acceleration equation on the space of probability measures. This provides "a sort of fluid-dynamic reconciliation between Bohm's and Nelson's stochastic mechanics," as they put it.

---

## 4. The Connection to Holland's biHamilton-Jacobi Theory

### 4.1 Holland's Ïƒ± as Schrödinger Potentials

The connection to Holland (2021) is now immediate. Holland's decomposition $\sigma_\pm = S \pm (\hbar/2)\ln\rho$ defines velocity fields $v_\pm = \nabla\sigma_\pm/m$ that are precisely the forward and backward drifts of the Nelson process — or equivalently, the gradients of the Schrödinger bridge potentials.

Holland's coupled bi-HJ equations:
$$\partial_t\sigma_+ + \frac{1}{2m}(\nabla\sigma_+)^2 + Q_+ + V = 0$$
$$\partial_t\sigma_- + \frac{1}{2m}(\nabla\sigma_-)^2 + Q_- + V = 0$$

are the **Hamilton–Jacobi–Bellman equations** of the forward and backward stochastic control problems associated with the Schrödinger bridge. The functions $\sigma_+$ and $\sigma_-$ are value functions: $\sigma_+$ is the cost-to-go of the forward control problem, $\sigma_-$ the cost-to-go of the backward one.

Holland's Fokker–Planck pair (Eq. 4.17):
$$\partial_t\rho + \nabla\cdot(\rho\thinspace{}v_\pm) = \pm\nu\nabla^2\rho$$

is the **optimality system** — the forward FPE and backward Kolmogorov equation of the FBSDE. The source terms $\pm\nu\nabla^2\rho$ are not physical noise but the consequence of the control-theoretic decomposition: neither the forward nor the backward optimal drift individually conserves probability. Conservation is restored only through their combination (the continuity equation $\partial_t\rho + \nabla\cdot(\rho v) = 0$).

### 4.2 The Crucial Difference: Holland's Determinism

Holland's theory operates at the field level — the $\sigma_\pm$ are classical functions satisfying coupled PDEs, and his trajectories $q_\pm(q_{\pm 0}, t)$ are deterministic solutions to coupled ODEs. There is no physical noise. The Fokker–Planck structure emerges from the *geometry of the decomposition*, not from stochasticity.

In the Schrödinger bridge / FBSDE interpretation, this corresponds to working with the *PDE system* (the Fokker–Planck / HJB pair) rather than with the *stochastic* representation (the coupled SDEs). Both representations encode the same information — this is the content of the nonlinear Feynman–Kac theorem — but they carry different ontological commitments:

- **Nelson:** The SDEs are physically real; there is genuine subquantum noise.
- **Holland:** The PDE system (bi-HJ) is fundamental; trajectories are deterministic.
- **Schrödinger bridge / FBSDE:** The structure is mathematical; the interpretation depends on context.

### 4.3 Probability from Action Difference

In the Schrödinger bridge framework, the density $\rho = e^{\hat{\varphi}-\varphi}$ arises from the *potentials*, not from trajectory statistics. This matches Holland's $\rho = e^{(\sigma_+-\sigma_-)/\hbar}$ exactly: probability is determined by the **action difference** between the forward and backward value functions.

Holland proves (§4.2) that the trajectory densities of the individual congruences $\rho_0 J_\pm^{-1}$ do *not* reproduce $\rho$. This is a direct consequence of the source terms in the FPE pair. In the FBSDE language, this says that the forward process and backward process each have their own evolving densities, but neither coincides with the bridge density — only their coupled system does.

---

## 5. Soumik Pal and Entropic Optimal Transport

Soumik Pal (University of Washington) and collaborators have contributed to the rigorous mathematical theory of entropic optimal transport, which is the static (marginals-only) formulation of the Schrödinger bridge. Key contributions include:

- **Harchaoui, Liu & Pal (2020):** Studied the asymptotics of entropy-regularized optimal transport via chaos decomposition, providing rigorous convergence results for the Sinkhorn algorithm.
- **Pal (2019):** Analyzed the difference between entropic cost and the classical optimal transport cost, quantifying the effect of the Fisher information regularization.
- Pal's lecture slides "From portfolio theory to optimal transport and Schrödinger bridge in-between" explicitly connect the large-deviation foundation (Schrödinger's hot gas experiment, Sanov's theorem, $N \to \infty$ particle limits) with the Sinkhorn algorithm and the zero-noise limit to OMT.

This work complements the dynamic (FBSDE/stochastic control) approach of Pavon's school by establishing the measure-theoretic foundations and convergence properties of the static Schrödinger system.

Additionally, Ghosal, Nutz & Bernton (2021) established stability of Schrödinger bridges and entropic optimal transport solutions with respect to perturbations of marginals and cost functions, using a geometric notion of "cyclical invariance" that extends c-cyclical monotonicity from classical optimal transport.

---

## 6. Modern Applications: Generative Models and Machine Learning

The FBSDE structure of Schrödinger bridges has experienced an explosion of interest in machine learning, particularly for **generative modeling**:

**SB-FBSDE (Chen et al., 2021):** Used FBSDE theory to derive exact log-likelihood expressions for Schrödinger bridge generative models, connecting them rigorously to score-based generative models (SGMs). The key insight is that the optimality condition (the Schrödinger system) can be expressed through the nonlinear Feynman–Kac formula, linking the forward and backward score functions through the same FBSDE structure that appears in Nelson/Holland.

**Diffusion Schrödinger Bridges:** Multiple groups have developed iterative algorithms (based on IPFP/Sinkhorn) for learning the Schrödinger bridge between a noise distribution and a data distribution, providing an alternative to denoising diffusion models. The forward-backward structure is used directly in training.

The Nelson/Holland velocity decomposition $v = \frac{1}{2}(b + b_*)$ and $u = \frac{1}{2}(b - b_*)$ appears in these algorithms as the decomposition of the learned score into "transport" and "diffusion" components.

---

## 7. Summary: The Triangular Relationship

The three frameworks — Holland's bi-HJ theory, Nelson's stochastic mechanics, and the Schrödinger bridge / FBSDE formalism — share the same mathematical skeleton:

```
                    Schrödinger Bridge / FBSDE
                   (Pavon, Chen, Georgiou, Pal)
                         /              \
                        /                \
          Forward FPE + Backward HJB    Stochastic control
          Schrödinger potentials (Ï†,Ï†Ì‚)  Value functions
                      /                    \
                     /                      \
         Nelson's Stochastic              Holland's bi-HJ
            Mechanics (1966)              Theory (2021)
         Forward & backward SDE         Coupled HJ equations
         b = v+u, b* = v-u              Ïƒ± = S ± (â„/2)ln Ï
         Stochastic trajectories         Deterministic congruences
         Ï from Markov process           Ï from action difference
```

The unifying insight, developed over decades from Zambrini (1986) through Pavon & Wakolbinger (1991) to Chen, Georgiou & Pavon (2021), is that:

1. **Holland's Ïƒ± are Schrödinger bridge potentials** — value functions of the forward and backward optimal control problems.
2. **Holland's bi-HJ equations are Hamilton–Jacobi–Bellman equations** — the optimality conditions for the stochastic control reformulation of quantum mechanics.
3. **Holland's Fokker–Planck pair (Eq. 4.17) is the FBSDE optimality system** — the forward Kolmogorov and backward Kolmogorov equations that characterize the bridge.
4. **The difference between SBP and quantum mechanics** lies in the **sign of the Fisher information** in the action: positive for Schrödinger bridges (entropy regularization of transport), negative for quantum mechanics (the quantum potential).
5. **Optimal mass transport is the zero-noise limit** ($\hbar \to 0$ or $\varepsilon \to 0$) of both.

The Conforti–Pavon result that all three problems (OMT, SBP, Madelung/Nelson) are critical points of actions on Wasserstein space, differing only by the Fisher information term and its sign, is perhaps the most elegant expression of this unity.

---

## Key References

- **Nelson, E.** (1966). "Derivation of the Schrödinger Equation from Newtonian Mechanics." *Phys. Rev.* 150, 1079.
- **Nelson, E.** (2012). "Review of stochastic mechanics." *J. Phys.: Conf. Ser.* 361, 012011.
- **Holland, P.** (2021). "Eliminating the wavefunction from quantum dynamics: the biHamilton-Jacobi theory." arXiv:2111.09235.
- **Zambrini, J.-C.** (1986). "Stochastic mechanics according to E. Schrödinger." *Phys. Rev. A* 33, 1532.
- **Pavon, M. & Wakolbinger, A.** (1991). "On free energy, stochastic control, and Schrödinger processes." In *Modeling, Estimation and Control of Systems with Uncertainty*, pp. 334–348.
- **Pavon, M.** (1995). "Hamilton's principle in stochastic mechanics." *J. Math. Phys.* 36, 6774–6800.
- **Pavon, M.** (2003). "Quantum Schrödinger Bridges." In *Directions in Mathematical Systems Theory and Optimization*, pp. 227–238.
- **Chen, Y., Georgiou, T.T. & Pavon, M.** (2016). "On the relation between optimal transport and Schrödinger bridges: A stochastic control viewpoint." *J. Optim. Theory Appl.* 169, 671–691.
- **Chen, Y., Georgiou, T.T. & Pavon, M.** (2021). "Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge." *SIAM Review* 63, 249–313.
- **Conforti, G. & Pavon, M.** (2017). "Extremal flows on Wasserstein space." arXiv:1712.02257.
- **Beghi, A., Ferrante, A. & Pavon, M.** (2001). "How to steer a quantum system over a Schrödinger bridge." *Proc. MTNS 2000*.
- **Léonard, C.** (2012). "A survey of the Schrödinger problem and some of its connections with optimal transport." *J. Funct. Anal.* 262, 1879–1920.
- **Harchaoui, Z., Liu, L. & Pal, S.** (2020). "Asymptotics of entropy-regularized optimal transport via chaos decomposition." arXiv:2011.08963.
- **Ghosal, P., Nutz, M. & Bernton, E.** (2021). "Stability of Entropic Optimal Transport and Schrödinger Bridges." arXiv:2106.03670.
- **Ma, J., Protter, P. & Yong, J.** (1994). "Solving forward-backward stochastic differential equations explicitly." *Probab. Theory Relat. Fields* 98, 339–359.
- **Pardoux, E. & Peng, S.** (1990). "Adapted solution of a backward stochastic differential equation." *Systems & Control Letters* 14, 55–61.
