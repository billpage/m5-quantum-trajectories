# ψ-KDE Documentation Audit: Findings and Proposed Changes

## Scope

Documents reviewed for psi-KDE descriptions:

| Document | Role | psi-KDE content |
|---|---|---|
| `M5psi_KDE_Analysis.md` | Primary theory | §§2–9, §11: full derivation, CIC pseudocode, field extraction |
| `NelsonMechanics_SchrodingerBridge_Algorithm.md` | Main algorithm | §7.2 step 1b, §8 pseudocode (from Analysis §8) |
| `Adaptive_K_Strategy_M5psi_KDE.md` | Adaptive K | §1: references psi-KDE step; no implementation detail |
| `Adaptive_K_v2_DualReadout.md` | K requirements | References psi-KDE; no implementation detail |
| `Method5_Mathematical_Analysis.md` | Original M5 (v1) | Pre-psi-KDE; §2 describes histogram+smooth (superseded) |
| `Method5_QA_Discussion.md` | Q&A reference | References psi-KDE; Q5/Q6 discuss ψ-KDE theory |
| `m5psi_kde_catstate.py` | Primary implementation | `psi_kde_estimate()` — 3 independent copies |
| `m5v2_meanweight_Q.py` | v2 implementation | `psi_kde_estimate()` with mean-weight Q |
| `m5psi_kde_trajectories.py` | Trajectory code | `psi_kde_estimate()` — copy from catstate |
| `hybrid_m5v2.py` | Hybrid template | `psi_kde_estimate()` equivalent |

---

## Finding 1: Normalization Convention — Consistent but Undocumented

### The issue

The **theory** (M5psi_KDE_Analysis.md §2.2) defines:

$$n_h(x) = \frac{1}{N_p} \sum_{i=1}^{N_p} K_h(x - X_i)$$

where $K_h$ is a **normalized** Gaussian kernel ($\int K_h(u)\,du = 1$). With this definition, $n_h(x) \to \rho(x)$ as $N_p \to \infty$, where $\rho$ is a proper probability density ($\int \rho\,dx = 1$).

The **implementation** (all `.py` files) does:
```python
n_smooth = gaussian_filter1d(n_raw, sigma) / Np
```

Here `gaussian_filter1d` convolves with a discrete kernel that **sums** to 1 (not integrates). The CIC raw deposit gives counts per bin. After dividing by $N_p$, `n_smooth[k]` gives the **fraction of particles per bin**, not density per unit x. Specifically:

$$n_\text{smooth}[k] \approx \rho(x_k) \cdot dx$$

This is smaller than the theoretical $n_h(x_k)$ by a factor of $dx$.

### Why it doesn't break the dynamics

Since $j_\text{smooth}$ suffers the same factor of $dx$:

$$\hat\psi = \frac{j_\text{smooth}}{\sqrt{n_\text{smooth}}} = \frac{j_h \cdot dx}{\sqrt{n_h \cdot dx}} = \frac{j_h}{\sqrt{n_h}} \cdot \sqrt{dx} = \psi_\text{true} \cdot \sqrt{dx}$$

The explicit normalization step $\hat\psi \mathrel{/}= \sqrt{\sum|\hat\psi|^2 \cdot dx}$ then rescales $\hat\psi$ to have $\int|\hat\psi|^2\,dx = 1$, absorbing the $\sqrt{dx}$ factor. All derived fields ($v$, $Q$, $\sqrt\rho$ for selection) are ratios or derivatives of $\hat\psi$ that are scale-invariant, so the dynamics is unaffected.

### The documentation gap

The theoretical sections claim $n_h \to \rho$ (proper density), while the implementation gives $n_\text{smooth} \to \rho \cdot dx$ (fraction per bin). This is nowhere explained. A reader implementing from the theory would divide by $N_p$ and expect proper density, then be surprised when $\hat\psi$ is off by $\sqrt{dx}$. The normalization step (d) saves the implementation, but someone who omits it (e.g. for the inverse psi-KDE problem or for direct comparison against an exact $\psi$) will get wrong absolute values.

### Proposed changes

**In `M5psi_KDE_Analysis.md`**, add a note after §4.1 step 2:

> **Normalization convention.** The CIC + `gaussian_filter1d` implementation divides by $N_p$ (not $N_p \cdot dx$), so `n_smooth[k]` gives the fraction of particles per bin, not proper density per unit $x$. This differs from the theoretical $n_h(x)$ by a factor of $dx$: $n_\text{smooth}[k] \approx n_h(x_k) \cdot dx$. Since both `n_smooth` and `j_smooth` share this factor, the ratio $\hat\psi = j/\sqrt{n}$ is correct up to an overall scale of $\sqrt{dx}$, which is removed by the normalization step (d). All derived fields ($v$, $Q$, selection weights) are scale-invariant ratios, so the dynamics is unaffected by this convention.
>
> If absolute $\hat\psi$ values are needed (e.g. for direct comparison against an exact $\psi$ without the normalization step), divide by $N_p \cdot dx$ instead of $N_p$:
> ```
> n_smooth = gaussian_filter1d(n_raw, σ) / (Np * dx)
> j_smooth = gaussian_filter1d(j_raw, σ) / (Np * dx)
> ```

**In `NelsonMechanics_SchrodingerBridge_Algorithm.md`** §7.2 step 1b, expand the description to include this note.

**In `M5psi_KDE_Analysis.md` §8**, relabel step (d) from "Normalise (optional, periodic maintenance)" to "Normalise (required every step)" and add a sentence explaining that it compensates for the $\sqrt{dx}$ factor.

---

## Finding 2: Phase Assignment in `init_particles` — Known Bug Pattern

### The issue

All implementations of `init_particles` use:
```python
S0 = HBAR * np.unwrap(np.angle(psi0))
Sp = np.interp(X, X_GRID, S0)
```

This is the **phase assignment bug pattern** already documented in the project memory: `np.unwrap(np.angle(ψ))` on a grid fails in exponential tails where $|\psi| \approx 0$ produces floating-point noise that propagates into the bulk via unwrap.

### Impact

For initial states with well-separated wave packets (e.g. the cat state at $t = 0$), the tails between the packets have $|\psi| \approx 0$ and the grid phase there is pure noise. `np.unwrap` may propagate phase jumps from these noisy regions into the bulk, corrupting $S$ at particle positions sampled from the high-density regions.

### Proposed fix

Replace the grid-interpolation phase assignment with direct evaluation at particle positions:
```python
def init_particles(psi0, Np, x_grid, dx, hbar, seed=42):
    rng = np.random.default_rng(seed)
    rho0 = np.abs(psi0)**2
    cdf = np.cumsum(rho0) * dx; cdf /= cdf[-1]
    X = np.interp(rng.uniform(size=Np), cdf, x_grid)
    # Evaluate ψ at each particle position by interpolation,
    # then take angle THERE (not on the grid)
    psi_at = np.interp(X, x_grid, psi0.real) + \
             1j * np.interp(X, x_grid, psi0.imag)
    Sp = hbar * np.angle(psi_at)  # no unwrap needed
    return X, Sp
```

This avoids `np.unwrap` entirely by evaluating the phase locally at each particle's (high-density) position. No unwrapping is needed because particles sample from $\rho = |\psi|^2$ and will be concentrated where $|\psi|$ is large.

**Note**: This assigns $S \in (-\pi\hbar, \pi\hbar]$, not the globally unwrapped phase. For M5 dynamics this is correct because only $e^{iS/\hbar}$ enters the CIC deposit, and $e^{i\theta}$ is $2\pi$-periodic. The action update $dS = (\frac{1}{2}mv^2 - V - Q)\,dt$ accumulates the continuous-time phase correctly from any starting point.

### Where to apply

All files: `m5psi_kde_catstate.py`, `m5v2_meanweight_Q.py`, `m5psi_kde_trajectories.py`, `hybrid_m5v2.py`, and any new code. Also update the `init_particles` description in `M5psi_KDE_Analysis.md` §7.3 and `NelsonMechanics_SchrodingerBridge_Algorithm.md` (if an initialization section is added).

---

## Finding 3: No Documentation of Ensemble Initialization Best Practices

### The issue

All existing code uses naive random CDF-inversion sampling:
```python
X = np.interp(rng.uniform(size=Np), cdf, x_grid)
```

Our inverse psi-KDE investigation (this session) showed that **deterministic CDF quantile placement** reduces the variance component of the reconstruction error by 3–5× relative to random sampling, and the improvement is robust across wave function types including those with interference and nodes.

None of the documentation mentions this.

### Proposed addition

Add a new subsection to `M5psi_KDE_Analysis.md` (after §7.3 or as a new §12) and a paragraph to `NelsonMechanics_SchrodingerBridge_Algorithm.md`:

> ### Ensemble Initialization from a Known ψ
>
> Given a known wave function $\psi(x)$ (e.g. an exact initial condition), the particle ensemble $\{X_i, S_i\}$ should be constructed to minimize the psi-KDE reconstruction error $\|\hat\psi - \psi\|^2$. The error decomposes as bias (from kernel smoothing, controlled by $\sigma$) plus variance (from finite $N_p$, controlled by particle placement).
>
> **Recommended procedure** (in order of priority):
>
> 1. **Deterministic CDF quantiles.** Place particles at $X_i = F^{-1}((i - \tfrac{1}{2})/N_p)$ where $F(x) = \int_{-\infty}^x |\psi(x')|^2\,dx'$. This achieves $O(N_p^{-2})$ variance scaling (vs $O(N_p^{-1})$ for random sampling), a 3–5× improvement for typical $N_p$.
>
> 2. **Phase from local evaluation.** Assign $S_i = \hbar \cdot \arg(\psi(X_i))$ where $\psi(X_i)$ is evaluated by interpolating the real and imaginary parts of $\psi$ separately at $X_i$. Do **not** use `np.unwrap(np.angle(ψ))` on the grid — this propagates floating-point noise from low-density tails into the bulk.
>
> 3. **Optional: position optimization.** For states with interference structure, starting from the CDF quantile positions and running a few hundred iterations of L-BFGS-B to minimize $\|\hat\psi - \psi\|^2$ can remove 40–68% of the remaining variance. The optimizer naturally discovers node-avoidance and curvature-adapted placement. For smooth initial states (single Gaussians, well-separated packets) the improvement is marginal (~15%).
>
> For stochastic initialization (e.g. when the M5 diffusion requires random initial noise), **systematic sampling** (single random offset $u_0 \sim U(0, 1/N_p)$, then $u_i = u_0 + i/N_p$) provides nearly deterministic-quality results while preserving the stochastic character.

---

## Finding 4: `NelsonMechanics_SchrodingerBridge_Algorithm.md` §7.2 Step 1b — Insufficient Detail

### The issue

The main algorithm document describes the ψ-KDE step as:

```
1b. ψ-KDE DENSITY ESTIMATION
    CIC-deposit particle data (1, cos(S_k/ℏ), sin(S_k/ℏ)) onto grid
    Gaussian-convolve all three fields with bandwidth σ_kde
    Form ψ̂ = j_smooth / √n_smooth
    Extract: √ρ = |ψ̂|,  v(x) = (ℏ/m) Im(ψ̂* ∂ₓψ̂) / |ψ̂|²
    Store ln ρ(x) = 2 ln |ψ̂(x)| on grid (or at particle positions)
```

This is too terse for independent implementation. Missing:

- The CIC linear-interpolation deposit weights (left/right grid point, fractional offset α)
- The normalization convention (divide by Np after convolution)
- The normalization step ($\hat\psi \mathrel{/}= \|\hat\psi\|$)
- The quantum potential formula $Q = -(\hbar^2/2m)\,\partial_{xx}|\hat\psi|/|\hat\psi|$
- The ε-floor regularization for v and Q at nodes

### Proposed change

Expand step 1b to include the full pseudocode from `M5psi_KDE_Analysis.md` §8 (which is already in the document's later sections), or at minimum add a cross-reference: "See `M5psi_KDE_Analysis.md` §8 for the complete pseudocode."

---

## Finding 5: Terminology — "complex current" Clarification Present

The documents correctly note (M5psi_KDE_Analysis.md §10.2) that $j_h$ is **not** the quantum probability current $J = \rho v$ but rather the kernel estimate of $\rho \cdot e^{iS/\hbar} = |\psi| \cdot \psi$. The preferred terminology "phase-weighted kernel sum" or "coherent average" is established. No change needed, but this should be kept consistent in any new text.

---

## Finding 6: Code Duplication

`psi_kde_estimate()` is independently implemented in at least 4 files (`m5psi_kde_catstate.py`, `m5v2_meanweight_Q.py`, `m5psi_kde_trajectories.py`, `hybrid_m5v2.py`). Any normalization or bug fix must be applied in all copies. This is a maintenance concern but not a documentation issue.

---

## Summary of Proposed Changes

| # | Document | Section | Change type | Priority |
|---|---|---|---|---|
| 1a | `M5psi_KDE_Analysis.md` | After §4.1 | Add normalization convention note | High |
| 1b | `M5psi_KDE_Analysis.md` | §8 step (d) | Change "optional" to "required"; explain √dx | High |
| 1c | `NelsonMechanics_SchrodingerBridge_Algorithm.md` | §7.2 step 1b | Add normalization note | High |
| 2 | All `.py` files | `init_particles` | Fix phase assignment (avoid np.unwrap on grid) | High |
| 3 | `M5psi_KDE_Analysis.md` | New §12 | Add ensemble initialization best practices | Medium |
| 3b | `NelsonMechanics_SchrodingerBridge_Algorithm.md` | New subsection | Add initialization paragraph | Medium |
| 4 | `NelsonMechanics_SchrodingerBridge_Algorithm.md` | §7.2 step 1b | Expand or cross-reference full pseudocode | Medium |
