"""
m5_init.py — Particle ensemble initialization for M5 quantum trajectories.

Provides unified initialization of particle positions X and action phases S
from an initial wavefunction ψ₀, supporting:

  • Stochastic (inverse-CDF) sampling from |ψ₀|²
  • Deterministic (equally-spaced quantile) placement
  • 1-D, 2-D, and N-D grids
  • Optional ψ-KDE phase refinement to reduce initial reconstruction error

Typical usage (1-D stochastic)
------------------------------
    from m5_init import init_ensemble_1d

    x = np.linspace(-15, 15, 512, endpoint=False)
    psi0 = my_wavefunction(x)
    X, S = init_ensemble_1d(psi0, x, Np=4000)

Typical usage (2-D stochastic)
------------------------------
    from m5_init import init_ensemble_2d

    axes = [np.linspace(-8, 8, 256, endpoint=False)] * 2
    psi0_2d = my_wavefunction_2d(*np.meshgrid(*axes, indexing='ij'))
    Q, S = init_ensemble_2d(psi0_2d, axes, Np=4000)

Phase optimization
------------------
    X, S = init_ensemble_1d(psi0, x, Np=4000,
                            optimize_phase=True, h_kde=0.25)

    This adjusts the initial phases S_i so that the ψ-KDE reconstruction
    ψ̂ = j_h / √n_h more closely matches ψ₀ on the grid.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Core 1-D initialization
# ═══════════════════════════════════════════════════════════════════════════════

def init_ensemble_1d(psi0_vals, x_grid, Np, *,
                     hbar=1.0, seed=42,
                     method='stochastic',
                     psi0_func=None,
                     optimize_phase=False,
                     h_kde=None,
                     phase_opt_iters=5):
    """Sample Np particles from |ψ₀|² and assign phases S = ℏ·∠ψ₀.

    Parameters
    ----------
    psi0_vals : (Nx,) complex array
        Wavefunction evaluated on *x_grid*.
    x_grid : (Nx,) real array
        Uniformly-spaced grid (endpoint=False convention).
    Np : int
        Number of particles.
    hbar : float
        Reduced Planck constant (default 1.0).
    seed : int
        Random seed (used for both stochastic and deterministic modes
        where jitter is applied).
    method : {'stochastic', 'deterministic'}
        'stochastic'   — inverse-CDF sampling with uniform random quantiles.
        'deterministic' — equally-spaced quantiles (i + 0.5) / Np placed
                          via the CDF, giving a low-discrepancy ensemble.
    psi0_func : callable or None
        If provided, phases are assigned by evaluating psi0_func(X) directly
        (more accurate than Re/Im interpolation when ψ₀ has fine structure).
        Signature: psi0_func(x) → complex array, same shape as x.
    optimize_phase : bool
        If True, refine phases via ψ-KDE gradient descent (requires h_kde).
    h_kde : float or None
        KDE bandwidth for phase optimization.  Required when
        optimize_phase=True.  Ignored otherwise.
    phase_opt_iters : int
        Number of L-BFGS iterations for phase refinement (default 5).

    Returns
    -------
    X : (Np,) float64 array — particle positions
    S : (Np,) float64 array — action phases
    """
    psi0_vals = np.asarray(psi0_vals, dtype=np.complex128)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    Nx = len(x_grid)
    dx = (x_grid[-1] - x_grid[0]) / (Nx - 1) if Nx > 1 else 1.0
    # For endpoint=False grids the spacing is (xR - xL) / Nx, but
    # consecutive differences are more robust:
    if Nx > 1:
        dx = float(x_grid[1] - x_grid[0])

    # Build CDF from |ψ₀|²
    rho0 = np.abs(psi0_vals) ** 2
    rho0 /= rho0.sum() * dx          # normalise to density
    cdf = np.cumsum(rho0) * dx
    cdf /= cdf[-1]                    # ensure cdf[-1] == 1 exactly

    # --- Position sampling -------------------------------------------
    if method == 'stochastic':
        rng = np.random.default_rng(seed)
        quantiles = rng.uniform(size=Np)
    elif method == 'deterministic':
        quantiles = (np.arange(Np) + 0.5) / Np
    else:
        raise ValueError(f"Unknown method '{method}'; "
                         f"use 'stochastic' or 'deterministic'.")

    X = np.interp(quantiles, cdf, x_grid).astype(np.float64)

    # --- Phase assignment --------------------------------------------
    S = _assign_phase_1d(X, psi0_vals, x_grid, hbar, psi0_func)

    # --- Optional ψ-KDE phase refinement -----------------------------
    if optimize_phase:
        if h_kde is None:
            raise ValueError("h_kde must be provided when "
                             "optimize_phase=True.")
        S = _optimize_phase_1d(X, S, psi0_vals, x_grid, dx,
                               hbar, h_kde, phase_opt_iters)

    return X, S


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Core 2-D initialization
# ═══════════════════════════════════════════════════════════════════════════════

def init_ensemble_2d(psi0_2d, axes, Np, *,
                     hbar=1.0, seed=42,
                     method='stochastic',
                     jitter=True):
    """Sample Np particles from a 2-D |ψ₀|² via marginal + conditional CDF.

    Parameters
    ----------
    psi0_2d : (Nx, Ny) complex array
        Wavefunction on the 2-D grid defined by *axes*.
    axes : list of two 1-D arrays [x_axis, y_axis]
        Each axis is a uniformly-spaced grid (endpoint=False).
    Np : int
        Number of particles.
    hbar : float
    seed : int
    method : {'stochastic', 'deterministic'}
    jitter : bool
        Add sub-cell uniform jitter to deterministic placements
        (default True; has no effect for stochastic sampling).

    Returns
    -------
    Q : (Np, 2) float64 array — particle positions
    S : (Np,)   float64 array — action phases
    """
    psi0_2d = np.asarray(psi0_2d, dtype=np.complex128)
    x_ax = np.asarray(axes[0], dtype=np.float64)
    y_ax = np.asarray(axes[1], dtype=np.float64)
    Nx, Ny = psi0_2d.shape
    dx = float(x_ax[1] - x_ax[0]) if Nx > 1 else 1.0
    dy = float(y_ax[1] - y_ax[0]) if Ny > 1 else 1.0

    rng = np.random.default_rng(seed)

    rho0 = np.abs(psi0_2d) ** 2
    rho0 /= rho0.sum() * dx * dy

    # Marginal in x
    rho_x = rho0.sum(axis=1) * dy          # (Nx,)
    cdf_x = np.cumsum(rho_x)
    cdf_x /= cdf_x[-1]

    if method == 'stochastic':
        q_x = rng.uniform(size=Np)
    elif method == 'deterministic':
        q_x = (np.arange(Np) + 0.5) / Np
    else:
        raise ValueError(f"Unknown method '{method}'.")

    i_x = np.searchsorted(cdf_x, q_x)
    i_x = np.clip(i_x, 0, Nx - 1)

    if jitter or method == 'stochastic':
        x_smp = x_ax[i_x] + rng.uniform(-dx / 2, dx / 2, size=Np)
    else:
        x_smp = x_ax[i_x].copy()

    # Conditional  y | x
    y_smp = np.zeros(Np, dtype=np.float64)
    for j in np.unique(i_x):
        mask = (i_x == j)
        n_j = mask.sum()
        cond = rho0[j, :]
        s = cond.sum() * dy
        if s < 1e-30:
            y_smp[mask] = 0.0
            continue
        cdf_y = np.cumsum(cond / cond.sum())
        cdf_y /= cdf_y[-1]
        if method == 'stochastic':
            q_y = rng.uniform(size=n_j)
        else:
            q_y = (np.arange(n_j) + 0.5) / n_j
        iy = np.searchsorted(cdf_y, q_y)
        iy = np.clip(iy, 0, Ny - 1)
        if jitter or method == 'stochastic':
            y_smp[mask] = y_ax[iy] + rng.uniform(-dy / 2, dy / 2, size=n_j)
        else:
            y_smp[mask] = y_ax[iy]

    Q = np.stack([x_smp, y_smp], axis=1).astype(np.float64)

    # Clip to grid interior (avoid extrapolation artefacts)
    Q[:, 0] = np.clip(Q[:, 0], x_ax[0] + dx, x_ax[-1] - dx)
    Q[:, 1] = np.clip(Q[:, 1], y_ax[0] + dy, y_ax[-1] - dy)

    # Phase via bilinear interpolation of Re/Im
    S = _assign_phase_2d(Q, psi0_2d, x_ax, y_ax, hbar)

    return Q, S


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  N-D initialization  (chain of conditional CDFs)
# ═══════════════════════════════════════════════════════════════════════════════

def init_ensemble_nd(psi0_nd, axes, Np, *,
                     hbar=1.0, seed=42,
                     method='stochastic',
                     jitter=True):
    """Sample Np particles from an N-D |ψ₀|² via successive conditional CDFs.

    Parameters
    ----------
    psi0_nd : complex ndarray of shape (N0, N1, …, N_{D-1})
        Wavefunction on the tensor-product grid.
    axes : list of D 1-D arrays
        Grid along each dimension.
    Np : int
    hbar, seed, method, jitter : as for init_ensemble_2d.

    Returns
    -------
    Q : (Np, D) float64 — positions
    S : (Np,)   float64 — phases
    """
    psi0_nd = np.asarray(psi0_nd, dtype=np.complex128)
    D = psi0_nd.ndim
    axes = [np.asarray(a, dtype=np.float64) for a in axes]
    if len(axes) != D:
        raise ValueError(f"len(axes)={len(axes)} but psi0_nd.ndim={D}")

    dq = [float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes]
    rng = np.random.default_rng(seed)

    rho = np.abs(psi0_nd) ** 2
    vol = np.prod(dq)
    rho /= rho.sum() * vol

    Q = np.zeros((Np, D), dtype=np.float64)

    # Dimension-by-dimension sampling via conditional distributions.
    # We maintain a running set of chosen bin indices for dims already
    # sampled, and marginalise the remaining dims at each step.
    idx = np.zeros((Np, D), dtype=np.intp)

    for d in range(D):
        Nd = psi0_nd.shape[d]
        ax_d = axes[d]

        if d == 0:
            # Marginal over all dims except 0
            sum_axes = tuple(range(1, D))
            marginal = rho.sum(axis=sum_axes) if D > 1 else rho.copy()
            marginal *= np.prod([dq[k] for k in range(1, D)]) if D > 1 else 1.0

            cdf_d = np.cumsum(marginal)
            cdf_d /= cdf_d[-1]

            if method == 'stochastic':
                q = rng.uniform(size=Np)
            else:
                q = (np.arange(Np) + 0.5) / Np

            i_d = np.clip(np.searchsorted(cdf_d, q), 0, Nd - 1)
            idx[:, 0] = i_d

            if jitter or method == 'stochastic':
                Q[:, 0] = ax_d[i_d] + rng.uniform(-dq[0] / 2, dq[0] / 2,
                                                   size=Np)
            else:
                Q[:, 0] = ax_d[i_d]
        else:
            # Conditional on previously chosen bins for dims 0..d-1
            remaining_axes = tuple(range(d + 1, D))

            for j_combo in _unique_rows(idx[:, :d]):
                mask = np.all(idx[:, :d] == j_combo, axis=1)
                n_j = mask.sum()
                if n_j == 0:
                    continue

                # Slice rho at the fixed bin indices for dims 0..d-1
                slc = tuple(j_combo) + (slice(None),)
                if remaining_axes:
                    cond = rho[slc].sum(axis=tuple(range(1, D - d)))
                    cond *= np.prod([dq[k] for k in range(d + 1, D)])
                else:
                    cond = rho[slc]

                s = cond.sum() * dq[d]
                if s < 1e-30:
                    idx[mask, d] = Nd // 2
                    Q[mask, d] = ax_d[Nd // 2]
                    continue

                cdf_c = np.cumsum(cond / cond.sum())
                cdf_c /= cdf_c[-1]

                if method == 'stochastic':
                    q = rng.uniform(size=n_j)
                else:
                    q = (np.arange(n_j) + 0.5) / n_j

                i_d = np.clip(np.searchsorted(cdf_c, q), 0, Nd - 1)
                idx[mask, d] = i_d

                if jitter or method == 'stochastic':
                    Q[mask, d] = ax_d[i_d] + rng.uniform(
                        -dq[d] / 2, dq[d] / 2, size=n_j)
                else:
                    Q[mask, d] = ax_d[i_d]

    # Clip to grid interior
    for d in range(D):
        Q[:, d] = np.clip(Q[:, d],
                          axes[d][0] + dq[d],
                          axes[d][-1] - dq[d])

    # Phase via multilinear interpolation of Re/Im
    S = _assign_phase_nd(Q, psi0_nd, axes, hbar)

    return Q, S


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_phase_1d(X, psi0_vals, x_grid, hbar, psi0_func=None):
    """Compute S = ℏ·∠ψ₀ at particle positions."""
    if psi0_func is not None:
        psi_at = psi0_func(X)
    else:
        psi_at = (np.interp(X, x_grid, psi0_vals.real)
                  + 1j * np.interp(X, x_grid, psi0_vals.imag))
    return (hbar * np.angle(psi_at)).astype(np.float64)


def _assign_phase_2d(Q, psi0_2d, x_ax, y_ax, hbar):
    """Bilinear interpolation of Re/Im → S = ℏ·∠ψ₀."""
    from scipy.interpolate import RegularGridInterpolator
    interp_re = RegularGridInterpolator(
        (x_ax, y_ax), psi0_2d.real,
        method='linear', bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator(
        (x_ax, y_ax), psi0_2d.imag,
        method='linear', bounds_error=False, fill_value=0.0)
    psi_at = interp_re(Q) + 1j * interp_im(Q)
    return (hbar * np.angle(psi_at)).astype(np.float64)


def _assign_phase_nd(Q, psi0_nd, axes, hbar):
    """Multilinear interpolation of Re/Im → S = ℏ·∠ψ₀ in N-D."""
    from scipy.interpolate import RegularGridInterpolator
    interp_re = RegularGridInterpolator(
        tuple(axes), psi0_nd.real,
        method='linear', bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator(
        tuple(axes), psi0_nd.imag,
        method='linear', bounds_error=False, fill_value=0.0)
    psi_at = interp_re(Q) + 1j * interp_im(Q)
    return (hbar * np.angle(psi_at)).astype(np.float64)


def _unique_rows(arr):
    """Yield unique rows of a 2-D integer array."""
    # np.unique with axis=0 can be slow for large arrays with many
    # unique rows; a set-based approach is faster here.
    seen = set()
    for row in arr:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            yield np.array(key, dtype=arr.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ψ-KDE phase optimization
# ═══════════════════════════════════════════════════════════════════════════════

def _psi_kde_on_grid(X, S, x_grid, h, hbar):
    r"""Evaluate ψ-KDE reconstruction on *x_grid* from ensemble {X, S}.

    ψ̂(x) = \frac{1}{N h} \sum_i K_h(x - X_i) \exp(i S_i / ℏ)

    with Gaussian kernel K_h.  Returns complex array of shape (Nx,).
    """
    Np = len(X)
    dx_mat = x_grid[:, None] - X[None, :]          # (Nx, Np)
    K = np.exp(-0.5 * (dx_mat / h) ** 2) / (h * np.sqrt(2 * np.pi))
    phase = np.exp(1j * S[None, :] / hbar)         # (1, Np)
    psi_hat = (K * phase).sum(axis=1) / Np          # (Nx,)
    return psi_hat


def _optimize_phase_1d(X, S0, psi0_vals, x_grid, dx,
                       hbar, h_kde, max_iter):
    r"""Refine phases to minimise ‖ψ̂_KDE − ψ₀‖² on the grid.

    Uses L-BFGS-B from scipy.optimize.  The cost is

        L(S) = ∑_g |ψ̂(x_g; X, S) − ψ₀(x_g)|² dx

    with analytic gradient

        ∂L/∂S_k = (2 dx / N h ℏ) ∑_g Re[ (ψ̂(x_g) − ψ₀(x_g))^*
                   · i · K_h(x_g − X_k) exp(i S_k / ℏ) ]
    """
    from scipy.optimize import minimize as sp_minimize

    Np = len(X)
    Nx = len(x_grid)
    psi0 = np.asarray(psi0_vals, dtype=np.complex128)

    # Precompute kernel matrix  K[g, k] = K_h(x_g - X_k)
    dx_mat = x_grid[:, None] - X[None, :]
    K_mat = np.exp(-0.5 * (dx_mat / h_kde) ** 2) / (h_kde * np.sqrt(2 * np.pi))
    # shape: (Nx, Np)

    def cost_and_grad(S_flat):
        phase = np.exp(1j * S_flat / hbar)              # (Np,)
        psi_hat = (K_mat * phase[None, :]).sum(axis=1) / Np  # (Nx,)
        residual = psi_hat - psi0                        # (Nx,)

        cost = float(np.sum(np.abs(residual) ** 2) * dx)

        # Gradient  ∂L/∂S_k
        # factor = residual^* · i · K_mat[:, k] · exp(i S_k / ℏ) / (N ℏ)
        # summed over grid, times 2 dx (Re part)
        conj_res = residual.conj()                       # (Nx,)
        # (Nx, Np) * (Nx, 1) → (Nx, Np), sum over g → (Np,)
        inner = (K_mat * conj_res[:, None]).sum(axis=0)  # (Np,)
        grad = (2.0 * dx / (Np * hbar)) * np.real(1j * inner * phase)

        return cost, grad

    result = sp_minimize(cost_and_grad, S0, jac=True,
                         method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': 1e-14})
    return result.x.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Convenience wrappers for common wavefunctions
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, sigma, hbar=1.0):
    """Normalised 1-D Gaussian wave packet  ψ(x; x₀, p₀, σ)."""
    x = np.asarray(x)
    return ((2 * np.pi * sigma ** 2) ** (-0.25)
            * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)
                     + 1j * p0 * x / hbar))


def cat_state(x, x0=4.0, p0=3.0, sigma=0.7, hbar=1.0, norm_grid=None):
    """Symmetric cat state  ψ = ψ(−x₀,+p₀) + ψ(+x₀,−p₀), normalised.

    Parameters
    ----------
    x : array
        Evaluation points.
    norm_grid : (x_grid, dx) tuple, optional
        If provided, normalise using this uniform grid rather than
        inferring dx from *x* (necessary when *x* is not a uniform grid,
        e.g. particle positions).  If None, assumes *x* is uniform.
    """
    psi = gaussian_wp(x, -x0, +p0, sigma, hbar) + \
          gaussian_wp(x, +x0, -p0, sigma, hbar)
    if norm_grid is not None:
        x_g, dx_g = norm_grid
        psi_g = gaussian_wp(x_g, -x0, +p0, sigma, hbar) + \
                gaussian_wp(x_g, +x0, -p0, sigma, hbar)
        norm = np.sqrt(np.sum(np.abs(psi_g) ** 2) * dx_g)
    else:
        dx = float(x[1] - x[0]) if len(np.asarray(x)) > 1 else 1.0
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def ho_ground_state(x, omega=1.0, mass=1.0, hbar=1.0):
    """Harmonic oscillator ground state."""
    return ((omega * mass / (np.pi * hbar)) ** 0.25
            * np.exp(-omega * mass * np.asarray(x) ** 2 / (2 * hbar)))
