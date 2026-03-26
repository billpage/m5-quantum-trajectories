# Stochastic Langevin Dynamics and the Active Matter–Quantum Mechanics Connection

## Project Overview

This project explores two related threads connecting classical stochastic dynamics to quantum mechanics:

1. **A numerical simulation** that uses Langevin dynamics with a precomputed quantum force field to reproduce the probability density of a two-Gaussian wavepacket superposition evolving under free-particle Schrödinger dynamics.
2. **The te Vrugt et al. (2023) theoretical framework** [1], which derives a rigorous mathematical mapping from the microscopic Langevin equations of inertial active Brownian particles to an equation formally identical to the Schrödinger equation, via the Madelung hydrodynamic representation.

The core question motivating both is: *under what conditions can a classical stochastic or active-matter system reproduce quantum-mechanical behavior?*

---

## 1. The Numerical Simulation: Stochastic Reproduction of Quantum Densities

### 1.1 Setup

The simulation defines a quantum target system consisting of two counter-propagating Gaussian wavepackets:

$$\psi(x,t) = \frac{1}{\sqrt{2}}\left[\psi_+(x,t) + \psi_-(x,t)\right]$$

where each wavepacket $\psi_\pm$ is a minimum-uncertainty Gaussian centered at $\pm x_0$ with initial momentum $\pm p_0$. These evolve analytically under the free-particle Schrödinger equation with exact time-dependent complex width:

$$\sigma^2(t) = \sigma_0^2\left(1 + \frac{i\hbar t}{2m\sigma_0^2}\right)$$

As the two packets approach each other, they produce an interference pattern in $|\psi|^2$—the signature quantum effect that any classical reproduction must somehow capture.

### 1.2 The Stochastic Model

An ensemble of $N = 50{,}000$ classical particles evolves under Langevin equations:

$$m\dot{v}_i = -\gamma v_i + F_Q(x_i, t) + \xi_i(t)$$

$$\dot{x}_i = v_i$$

where:

- $F_Q(x,t) = m\left(\frac{\partial v_Q}{\partial t} + v_Q \frac{\partial v_Q}{\partial x}\right)$ is the **quantum force** derived from the Bohmian velocity field $v_Q = \frac{\hbar}{m}\text{Im}\left(\frac{\nabla\psi}{\psi}\right)$
- $\gamma = 0.5$ is the friction coefficient
- $\xi_i(t)$ is Gaussian white noise with strength $\sqrt{2\gamma m D_\text{eff}\, dt}$
- $D_\text{eff} = (\hbar/\sigma)(1+\gamma) = 3.75$

Particles are initialized by sampling from the quantum density $|\psi(x,0)|^2$ via inverse-CDF sampling.

### 1.3 Results and Limitations

The simulation achieves a time-averaged fidelity (Bhattacharyya overlap) of **0.939**, with fidelity degrading from 0.999 at $t=0$ to 0.865 at $t=1.0$.

**What works:** The stochastic ensemble tracks the broad envelope of the quantum density well at early times, including the initial bimodal structure and the approach of the two packets.

**What doesn't work well:**

- **Interference fringes at later times.** By $t = 0.5$–$1.0$, the quantum target shows sharp interference oscillations (3 peaks detected), while the stochastic ensemble produces a much broader, smoother distribution (11 noisy peaks detected). The ensemble cannot faithfully reproduce the fine interference structure.
- **The approach is circular.** The quantum force $F_Q$ is computed directly from the exact wavefunction $\psi(x,t)$, which is obtained by solving the Schrödinger equation analytically. The simulation does not *derive* quantum behavior from classical ingredients—it *imposes* quantum dynamics through a precomputed force field and then checks whether Langevin particles driven by that force reproduce the density. This is a test of sampling fidelity, not a demonstration of emergent quantum behavior.
- **The diffusion coefficient $D_\text{eff}$ is ad hoc.** The expression $D_\text{eff} = (\hbar/\sigma)(1+\gamma)$ is not derived from a physical argument; it is a tuning parameter. Different values of $\gamma$ give different fidelities, and the simulation effectively asks: for what noise level does Langevin dynamics best track a precomputed quantum trajectory?
- **Velocity field not compared.** The simulation plots the quantum velocity field $v_Q(x,t)$ but does not overlay the ensemble-averaged stochastic velocity, making it impossible to assess whether the particles actually follow Bohmian trajectories.

### 1.4 Connection to Bohmian Mechanics

The approach is loosely inspired by stochastic mechanics à la Nelson [2] and the de Broglie–Bohm pilot wave theory [3]. In Bohmian mechanics, particles follow deterministic trajectories guided by the velocity field $v = \nabla S/m$ where $S$ is the phase of $\psi = \sqrt{\rho}\, e^{iS/\hbar}$. Nelson's stochastic mechanics adds diffusion to this picture. The simulation sits in this tradition but does not carefully implement either framework—it uses the quantum force (which combines both convective and pressure-like terms) as a black-box input to standard Langevin dynamics.

---

## 2. The te Vrugt et al. Framework: From Active Brownian Particles to the Schrödinger Equation

### 2.1 Starting Point: Microscopic Langevin Equations

The paper [1] begins with the standard microscopic description of $N$ underdamped active Brownian particles (ABPs) in two dimensions:

$$\dot{\mathbf{r}}_i = \frac{\mathbf{p}_i}{m}, \qquad \dot{\mathbf{p}}_i = -\gamma \mathbf{p}_i - \nabla_{\mathbf{r}_i} U + m\gamma v_0 \hat{\mathbf{u}}_i + \boldsymbol{\eta}_i, \qquad \dot{\varphi}_i = \chi_i$$

where $\hat{\mathbf{u}}_i = (\cos\varphi_i, \sin\varphi_i)$ is the self-propulsion direction, $v_0$ is the swimming speed, $\gamma$ is friction, $U = U_1 + U_2$ includes external and interaction potentials, and $\boldsymbol{\eta}_i$, $\chi_i$ are translational and rotational noises.

These are *classical* Langevin equations for self-propelled particles—bacteria, Janus colloids, vibrated granular disks, etc.

### 2.2 Coarse-Graining: Active Model I+

Through a systematic coarse-graining procedure (interaction-expansion method, orientational expansion, quasi-stationary approximation for polarization), te Vrugt et al. derive a continuum field theory called **Active Model I+** (AMI+):

$$\dot{\rho} = -\nabla \cdot (\rho \mathbf{v}) + \frac{1}{2D_R}\nabla\cdot(v_\text{ld}(\rho)\,\nabla v_\text{ld}(\rho)\,\rho)$$

$$\dot{\mathbf{v}} + (\mathbf{v}\cdot\nabla)\mathbf{v} = -\frac{1}{m}\nabla\!\left(f'(\rho) - \kappa\nabla^2\rho + \lambda(\nabla\rho)^2 + U_1\right) - \gamma\mathbf{v} + \frac{v_\text{ld}(\rho)^2}{\gamma}\nabla^2\mathbf{v} + \frac{\xi}{m}(\nabla^2\rho)\nabla\rho$$

where $v_\text{ld}(\rho) = v_0 - A_1\rho/(\gamma m)$ is the density-dependent swimming speed. Key features:

- The density-dependent swimming speed acts as an **effective viscosity** $v_\text{ld}^2/\gamma$.
- **Thermodynamic vs. mechanical velocity fields differ** in the active case—a point with no analog in passive fluids.
- AMI+ contains the overdamped Active Model B+ (AMB+) as a limiting case ($\gamma \to \infty$).

### 2.3 The Mapping to the Schrödinger Equation

The central result proceeds from the simpler Active Model I (AMI), obtained by setting $v_\text{ld} \approx 0$ and $\xi = 0$:

$$\dot{\rho} = -\nabla\cdot(\rho\mathbf{v})$$

$$\dot{\mathbf{v}} + (\mathbf{v}\cdot\nabla)\mathbf{v} = -\frac{1}{m}\nabla\!\left(f'(\rho) - \kappa\nabla^2\rho + \lambda(\nabla\rho)^2 + U_1\right) - \gamma\mathbf{v}$$

Setting $f'(\rho) = 0$, $\gamma \approx 0$ (small friction, strong activity), defining $\rho_q = 2\rho$, and using $\rho \approx (\rho_0/2)\ln(\rho_q/\rho_0)$ for small deviations from a reference density $\rho_0$, together with the constraint $\kappa = -\lambda\rho_0$, the velocity equation becomes:

$$\dot{\mathbf{v}} + (\mathbf{v}\cdot\nabla)\mathbf{v} = \frac{1}{m}\nabla\!\left(\kappa\rho_0 \frac{\nabla^2\sqrt{\rho_q}}{\sqrt{\rho_q}} - U_1\right)$$

This uses the identity $\nabla^2\ln\theta + (\nabla\ln\theta)^2 = \nabla^2\theta/\theta$. Setting $\hbar^2/(2m) = \kappa\rho_0$, this is precisely the **Madelung equations** [4,5]:

$$\dot{\rho}_q = -\nabla\cdot(\rho_q \mathbf{v})$$

$$\dot{\mathbf{v}} + (\mathbf{v}\cdot\nabla)\mathbf{v} = \frac{1}{m}\nabla\!\left(\frac{\hbar^2}{2m}\frac{\nabla^2\sqrt{\rho_q}}{\sqrt{\rho_q}} - U_1\right)$$

If $\mathbf{v} = \nabla S/m$ is irrotational and satisfies the Wallstrom quantization condition $m\oint \mathbf{v}\cdot d\mathbf{l} = 2\pi n\hbar$, then the substitution $\psi = \sqrt{\rho_q}\, e^{iS/\hbar}$ yields:

$$i\hbar\dot{\psi} = -\frac{\hbar^2}{2m}\nabla^2\psi + U_1\psi$$

which is mathematically identical to the Schrödinger equation.

### 2.4 Generalizations

The paper identifies several extensions depending on which terms are retained:

| Active model terms | Resulting quantum equation |
|---|---|
| $f' = 0$, $\gamma = 0$ | Standard Schrödinger equation |
| $f'(\rho) = a\rho_q$ | Gross-Pitaevskii equation (BEC) |
| General $f'(\rho)$ | Nonlinear Schrödinger equation |
| $f'(\rho) \neq 0$, $\gamma \neq 0$ | Schrödinger-Langevin equation (dissipative QM) |

### 2.5 Correspondence Table

The mapping between active matter and quantum variables is:

| Active matter (AMI) | Quantum mechanics | Relation |
|---|---|---|
| Particle density $\rho$ | Probability density $\rho_q = |\psi|^2$ | $\rho_q = 2\rho$ |
| Velocity field $\mathbf{v}$ | Phase gradient $\nabla S/m$ | $\mathbf{v} = \nabla S/m$ |
| Parameters $\kappa$, $\lambda$ | Planck constant $\hbar$ | $\kappa\rho_0 = -\lambda\rho_0^2 = \hbar^2/(2m)$ |
| Chemical potential $\mu$ | Energy $E$ | $\mu = E$ |
| Gradient terms $-\kappa\nabla^2\rho + \lambda(\nabla\rho)^2$ | Quantum potential $\frac{\hbar^2}{2m}\frac{\nabla^2\sqrt{\rho_q}}{\sqrt{\rho_q}}$ | Via $\rho \approx \frac{\rho_0}{2}\ln(\rho_q/\rho_0)$ |
| Passive noninteracting limit | Classical limit $\hbar \to 0$ | Activity/interactions ↔ quantum effects |

### 2.6 Physical Significance and Caveats

**What the mapping is:**

- A formal mathematical equivalence between the hydrodynamic form of AMI (a classical active field theory) and the Madelung form of the Schrödinger equation.
- A tool for transferring intuition: the quantum potential $\propto \nabla^2\sqrt{\rho}/\sqrt{\rho}$ corresponds to gradient terms $-\kappa\nabla^2\rho + \lambda(\nabla\rho)^2$ in the chemical potential, which penalize sharp density interfaces. This gives a pattern-formation interpretation of quantum effects.

**What the mapping is not:**

- It is **not** a derivation of quantum mechanics from classical physics. The paper explicitly states that quantum physics is not a description of the dynamics of active classical particles.
- The physical interpretation differs fundamentally: in quantum mechanics, $|\psi|^2$ is the probability density of a single particle; in the active model, it is proportional to the number density of a classical many-body system.
- The mapping requires several approximations that may not hold simultaneously in a real active system: $f' = 0$ (no free energy contribution), $\gamma \approx 0$ (very weak friction), $\kappa = -\lambda\rho_0$ (specific parameter relation), small density deviations from $\rho_0$, and irrotational velocity field.

---

## 3. The Active Tunnel Effect

### 3.1 Setup

Setting $\mathbf{v} = 0$ and $f' = 0$ in AMI gives the static condition $\mu = -\kappa\partial_x^2\rho + \lambda(\partial_x\rho)^2 + U_1 = \text{const.}$ For a rectangular potential barrier $U_1(x) = V_0$ for $|x| \le L$ and $0$ otherwise, te Vrugt et al. find an analytical solution:

$$\rho(x) = \begin{cases} \ln(\cos(k(x+L)+\alpha)) + A & \text{for } x < -L \\ \ln(\cosh(\varkappa x)) + B & \text{for } |x| \le L \\ \ln(\cos(k(x-L)-\alpha)) + A & \text{for } x > L \end{cases}$$

with wavenumbers $k = \sqrt{\mu/\kappa}$ and $\varkappa = \sqrt{(V_0 - \mu)/\kappa}$. The density decays smoothly inside the barrier rather than dropping sharply—the active analog of quantum tunneling.

### 3.2 Robustness

Numerical continuation (varying the parameters $a$, $\kappa$, $\lambda$ away from the analytically solvable case) shows that the tunneling-like density profile persists over a range of parameter values. The effect becomes more pronounced for larger $\kappa$ (stronger gradient penalty) and positive $a$ (repulsive effective interactions). All computed solutions were linearly stable.

### 3.3 Physical Interpretation

The active tunnel effect means that interacting active particles at a sharp potential barrier (e.g., dielectric spheres illuminated by a laser with rectangular intensity profile) show a smoother density transition than passive noninteracting particles, which would show a discontinuous profile. The smoothing arises from the energetic cost of density gradients—the same mechanism that, in the quantum analog, gives rise to tunneling via the quantum potential.

---

## 4. Connection to Fuzzy Dark Matter

The Madelung equations appear in astrophysics as the hydrodynamic form of fuzzy dark matter (FDM) models [6], where ultralight scalar particles ($m \sim 10^{-22}$ eV) are described by a single macroscopic wavefunction. FDM was introduced to resolve small-scale structure problems in cold dark matter models. The quantum pressure term suppresses structure below the de Broglie wavelength.

The active matter mapping provides a pattern-formation interpretation: the gradient terms penalize interfaces and suppress small-scale structure, just as interfacial energy terms do in Cahn-Hilliard theory. An important difference is that gravity (Poisson equation with attractive sign) differs from electrostatics (repulsive for like charges), so the patterns in charged active matter and FDM will differ despite the formal similarity.

---

## 5. Open Questions and Critical Assessment

### Regarding the simulation

- The simulation does not close the loop: it extracts forces from quantum mechanics and feeds them to classical particles, rather than demonstrating emergent quantum behavior from purely classical active-matter ingredients. A more compelling test would start from the AMI equations with microscopically determined coefficients and check whether the resulting dynamics match Schrödinger predictions without ever computing $\psi$.
- The stochastic ensemble loses fidelity precisely where quantum interference is strongest—the regime where the quantum-classical distinction matters most.
- The noise model ($D_\text{eff}$) lacks a principled derivation connecting it to the te Vrugt et al. framework.

### Regarding the theoretical mapping

- The Wallstrom objection [7]: the Madelung equations are necessary but not sufficient for the Schrödinger equation. The quantization condition $m\oint \mathbf{v}\cdot d\mathbf{l} = 2\pi n\hbar$ must be imposed separately and has no obvious classical justification. The paper acknowledges this but does not resolve it.
- The limit $\gamma \to 0$ is subtle: setting $\gamma = 0$ exactly makes the microscopic model passive. The paper argues for "small but finite $\gamma$ with strong activity," but the precise regime where all approximations hold simultaneously is not quantified.
- The logarithmic approximation $\rho \approx (\rho_0/2)\ln(\rho_q/\rho_0)$ requires small density deviations, which may conflict with the large density contrasts seen in interference patterns or phase separation.

---

## References

[1] te Vrugt, M., Frohoff-Hülsmann, T., Heifetz, E., Thiele, U. & Wittkowski, R. "From a microscopic inertial active matter model to the Schrödinger equation." *Nature Communications* **14**, 1302 (2023). https://doi.org/10.1038/s41467-022-35635-1

[2] Nelson, E. "Derivation of the Schrödinger Equation from Newtonian Mechanics." *Physical Review* **150**, 1079–1085 (1966).

[3] Bohm, D. "A Suggested Interpretation of the Quantum Theory in Terms of 'Hidden' Variables. I." *Physical Review* **85**, 166–179 (1952).

[4] Madelung, E. "Quantentheorie in hydrodynamischer Form." *Zeitschrift für Physik* **40**, 322–326 (1927).

[5] Heifetz, E. & Cohen, E. "Toward a thermo-hydrodynamic like description of Schrödinger equation via the Madelung formulation and Fisher information." *Foundations of Physics* **45**, 1514–1525 (2015).

[6] Ferreira, E. G. M. "Ultra-light dark matter." *Astronomy and Astrophysics Review* **29**, 7 (2021).

[7] Wallstrom, T. C. "Inequivalence between the Schrödinger equation and the Madelung hydrodynamic equations." *Physical Review A* **49**, 1613–1617 (1994).

[8] Heifetz, E. & Plochotnikov, I. "Effective classical stochastic theory for quantum tunneling." *Physics Letters A* **384**, 126511 (2020).

[9] Tsekov, R., Heifetz, E. & Cohen, E. "Derivation of the local-mean stochastic quantum force." *Fluctuation and Noise Letters* **16**, 1750028 (2017).

[10] Wittkowski, R. et al. "Scalar φ⁴ field theory for active-particle phase separation." *Nature Communications* **5**, 4351 (2014).

[11] Cates, M. E. & Tailleur, J. "Motility-induced phase separation." *Annual Review of Condensed Matter Physics* **6**, 219–244 (2015).

[12] Kostin, M. D. "On the Schrödinger-Langevin equation." *Journal of Chemical Physics* **57**, 3589–3591 (1972).

[13] Mocz, P. & Succi, S. "Numerical solution of the nonlinear Schrödinger equation using smoothed-particle hydrodynamics." *Physical Review E* **91**, 053304 (2015).
