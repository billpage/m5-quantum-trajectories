# Kernel Expressions for the ψ-KDE Swarmalator
## Gaussian, Quintic B-spline, and Compact Rational

---

### Notation

Particle positions: $X_1, X_2, \ldots, X_{N_p}$, with phases $\phi_j = S_j/\hbar$.

Evaluation point: $x$ (could be any of the $X_i$, a GH candidate, or a grid point).

Displacement: $\Delta_j = x - X_j$

Normalised displacement: $u_j = \Delta_j / h$ where $h$ is the kernel bandwidth.

All kernel sums run over source particles $j = 1, \ldots, N_p$.

---

## 1. Kernel Definitions

### 1.1 Gaussian Kernel

$$K_h^{\rm G}(\Delta) = \frac{1}{h\sqrt{2\pi}} \exp\!\left(-\frac{\Delta^2}{2h^2}\right)$$

First derivative:

$$K_h^{\rm G\prime}(\Delta) = -\frac{\Delta}{h^2}\, K_h^{\rm G}(\Delta) = -\frac{\Delta}{h^3\sqrt{2\pi}} \exp\!\left(-\frac{\Delta^2}{2h^2}\right)$$

Properties:
- Support: $(-\infty, \infty)$, truncated at $|\Delta| < 4h$ in the code
- Smoothness: $C^\infty$
- Second moment: $\mu_2 = h^2$
- $K''/K = (u^2 - 1)/h^2$ — polynomial in $u$

### 1.2 Quintic B-spline Kernel

Define $q = |\Delta|/h$. The quintic B-spline (order 6, degree 5) is:

$$K_h^{\rm Q}(\Delta) = \frac{1}{120\,h} \times \begin{cases}
(3-q)^5 - 6(2-q)^5 + 15(1-q)^5 & 0 \le q < 1 \\[4pt]
(3-q)^5 - 6(2-q)^5 & 1 \le q < 2 \\[4pt]
(3-q)^5 & 2 \le q < 3 \\[4pt]
0 & q \ge 3
\end{cases}$$

First derivative ($s = \operatorname{sgn}(\Delta)$):

$$K_h^{\rm Q\prime}(\Delta) = \frac{s}{120\,h^2} \times \begin{cases}
-5(3-q)^4 + 30(2-q)^4 - 75(1-q)^4 & 0 \le q < 1 \\[4pt]
-5(3-q)^4 + 30(2-q)^4 & 1 \le q < 2 \\[4pt]
-5(3-q)^4 & 2 \le q < 3 \\[4pt]
0 & q \ge 3
\end{cases}$$

Properties:
- Support: $[-3h,\; 3h]$ (compact)
- Smoothness: $C^4$ (continuous through 4th derivative)
- $K''/K$ is piecewise rational (ratio of degree-3 polynomials in $q$)

### 1.3 Compact Rational Kernel

Define $R = 2.5\,h$ (support radius) and $\xi = \Delta/R$. With exponent $n = 4$:

$$K_h^{\rm R}(\Delta) = C_n \times \begin{cases}
\left(1 - \xi^2\right)^4 & |\Delta| < R \\[4pt]
0 & |\Delta| \ge R
\end{cases}$$

where the normalisation constant is

$$C_n = \frac{1}{R \cdot B\!\left(\tfrac{1}{2},\, n+1\right)} = \frac{\Gamma(n + \tfrac{3}{2})}{R\,\sqrt{\pi}\;\Gamma(n+1)}$$

For $n = 4$: $B(\tfrac{1}{2}, 5) = \sqrt{\pi}\;\Gamma(5)/\Gamma(\tfrac{11}{2}) = \sqrt{\pi}\cdot 24 / (945\sqrt{\pi}/32) = 256/315$, so $C_4 = 315/(256\,R)$.

First derivative (within support $|\Delta| < R$):

$$K_h^{\rm R\prime}(\Delta) = C_n \cdot n \cdot \frac{-2\Delta}{R^2}\left(1 - \xi^2\right)^{n-1} = -\frac{8\,C_4\,\Delta}{R^2}\left(1 - \frac{\Delta^2}{R^2}\right)^3$$

Properties:
- Support: $[-R,\; R] = [-2.5h,\; 2.5h]$ (compact)
- Smoothness: $C^{2n-2} = C^6$ for $n=4$
- $K''/K$ within support:

$$\frac{K''(\Delta)}{K(\Delta)} = \frac{-2n}{R^2}\;\frac{1 - (2n-1)\xi^2}{1 - \xi^2}$$

This is a **rational function** of $\xi^2 = \Delta^2/R^2$ with a simple pole at $|\Delta| = R$ (the support boundary). For $n=4$:

$$\frac{K''}{K} = \frac{-8}{R^2}\;\frac{1 - 7\xi^2}{1 - \xi^2}$$

---

## 2. The ψ-KDE Kernel Sums

For each evaluation point $x$, the code computes six sums (four without derivatives):

**Density KDE:**
$$n(x) = \sum_{j=1}^{N_p} K_h(x - X_j)$$

**Coherent complex current:**
$$j(x) = \sum_{j=1}^{N_p} K_h(x - X_j)\, e^{i\phi_j} = j_{\rm re}(x) + i\, j_{\rm im}(x)$$

where

$$j_{\rm re}(x) = \sum_j K_h(x - X_j) \cos\phi_j, \qquad j_{\rm im}(x) = \sum_j K_h(x - X_j) \sin\phi_j$$

**Derivative sums** (when needed for velocity):

$$j'_{\rm re}(x) = \sum_j K'_h(x - X_j) \cos\phi_j, \qquad j'_{\rm im}(x) = \sum_j K'_h(x - X_j) \sin\phi_j$$

Written out explicitly for each kernel:

### Gaussian:

$$n^{\rm G}(x) = \frac{1}{h\sqrt{2\pi}} \sum_j \exp\!\left(-\frac{(x-X_j)^2}{2h^2}\right)$$

$$j_{\rm re}^{\rm G}(x) = \frac{1}{h\sqrt{2\pi}} \sum_j \exp\!\left(-\frac{(x-X_j)^2}{2h^2}\right) \cos\!\left(\frac{S_j}{\hbar}\right)$$

$$j_{\rm im}^{\prime\,\rm G}(x) = -\frac{1}{h^3\sqrt{2\pi}} \sum_j (x-X_j)\, \exp\!\left(-\frac{(x-X_j)^2}{2h^2}\right) \sin\!\left(\frac{S_j}{\hbar}\right)$$

### Quintic B-spline:

$$n^{\rm Q}(x) = \frac{1}{120\,h} \sum_j W_5\!\left(\frac{|x - X_j|}{h}\right)$$

where $W_5(q)$ is the piecewise quintic defined in §1.2. Each term is a **polynomial** in $(x - X_j)/h$ within its piecewise region.

### Compact Rational:

$$n^{\rm R}(x) = C_4 \sum_{j:\,|x-X_j|<R} \left(1 - \frac{(x-X_j)^2}{R^2}\right)^4$$

This is a **sum of rational functions** — specifically, a sum of 8th-degree polynomials in $(x - X_j)$ divided by $R^8$. The sum is **exactly polynomial** in $x$ within any interval where the set of contributing particles doesn't change.

---

## 3. Derived Fields

### 3.1 Reconstructed wavefunction

$$\hat{\psi}(x) = \frac{j(x)}{\sqrt{n(x)}}, \qquad \hat{\rho}(x) = |\hat{\psi}|^2 = \frac{|j(x)|^2}{n(x)}$$

$$\sqrt{\hat\rho}(x) = \frac{|j(x)|}{\sqrt{n(x)}}, \qquad \ln\hat\rho(x) = 2\ln|j(x)| - \ln n(x)$$

### 3.2 Current velocity

$$v(x) = \frac{\hbar}{m}\,\operatorname{Im}\frac{j'(x)}{j(x)} = \frac{\hbar}{m}\;\frac{j'_{\rm im}\, j_{\rm re} - j'_{\rm re}\, j_{\rm im}}{j_{\rm re}^2 + j_{\rm im}^2}$$

### 3.3 Quantum potential (GH WEIGH)

$$Q_k = -\frac{\hbar^2}{m\,\sigma_{\rm gh}^2}\left(\frac{\sum_\alpha \omega_\alpha\, \sqrt{\hat\rho}(x_{k\alpha})}{\sqrt{\hat\rho}(x_{k0})} - 1\right)$$

where $x_{k\alpha} = X_k^{\rm class} + \sqrt{2}\,\sigma_{\rm gh}\,\xi_\alpha$ are the GH probe points and $x_{k0} = X_k^{\rm class}$ is the departure point.

---

## 4. The Poirier Connection: Kernel Sums as C-derivatives

Poirier's uniformizing coordinate $C_j = (j - \tfrac{1}{2})/N_p$ gives $\rho = 1/x_C$ where $x_C = \partial x/\partial C$. The dimensionless ratios are:

$$z = \frac{x_{CC}}{x_C^2} = -\frac{\rho'}{\rho} = -\frac{n'}{n}, \qquad w = \frac{x_{CCC}}{x_C^3} = 3\left(\frac{n'}{n}\right)^2 - \frac{n''}{n}$$

### For the Gaussian kernel:

$$z^{\rm G}(x) = -\frac{n^{\rm G\prime}(x)}{n^{\rm G}(x)} = \frac{\sum_j \frac{x-X_j}{h^2}\,e^{-(x-X_j)^2/2h^2}}{\sum_j e^{-(x-X_j)^2/2h^2}}$$

This is a **ratio of transcendental functions** of particle positions — the exponentials never simplify to rational form for finite $N_p$.

### For the compact rational kernel:

$$z^{\rm R}(x) = -\frac{n^{\rm R\prime}(x)}{n^{\rm R}(x)} = \frac{\sum_{j:\,|x-X_j|<R}\; \frac{8(x-X_j)}{R^2}\left(1 - \frac{(x-X_j)^2}{R^2}\right)^3}{\sum_{j:\,|x-X_j|<R}\; \left(1 - \frac{(x-X_j)^2}{R^2}\right)^4}$$

This is a **ratio of polynomials in $(x - X_j)$** — a genuine rational function of the particle positions. As $N_p \to \infty$, it converges to the continuum $-\rho'/\rho$, which is itself a rational function of C-derivatives (Poirier's $z$).

The algebraic closure — rational kernel sums yield rational Poirier variables — is the structural reason why the compact rational kernel may produce better energy conservation. The Gaussian approximates the same quantity, but through transcendental (exponential) functions that are algebraically incompatible with the rational form that Poirier's compatibility condition demands.

---

## 5. K″/K Structure Comparison

At the kernel peak ($\Delta = 0$):

| Kernel | $K''(0)/K(0)$ | Functional form of $K''/K$ |
|--------|---------------|---------------------------|
| Gaussian | $-1/h^2$ | $(u^2 - 1)/h^2$ — polynomial in $u$ |
| Quintic B-spline | $-20/(120h^2) \times \text{...}$ | piecewise polynomial / piecewise polynomial |
| Compact rational ($n=4$) | $-8/R^2$ | $-8(1 - 7\xi^2)/(R^2(1-\xi^2))$ — rational in $\xi^2$ |

The rational kernel's $K''/K$ has a **pole** at $\xi = 1$ (the support boundary), which is the discrete analogue of the $1/\rho$ singularity structure in the continuum quantum potential. This pole is never evaluated in practice (the kernel vanishes at the boundary), but its presence shapes the algebraic structure of the kernel sums throughout the interior.
