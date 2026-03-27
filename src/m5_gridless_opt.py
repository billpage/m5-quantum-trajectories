#!/usr/bin/env python3
"""
M5 Gridless Swarmalator — Optimized numpy/cupy Implementation
===============================================================

GPU-accelerated gridless quantum swarmalator algorithm.
All kernel sums are vectorized as batched matrix operations.

Architecture:
  - xp = cupy (GPU) or numpy (CPU), selected automatically
  - Kernel sums computed via broadcasting: (N_eval, N_src) matrices
  - Chunked to fit GPU memory when N_eval × N_src > chunk_limit
  - Shared GH candidate cloud for STEER + WEIGH + backward channel

Algorithm per step:
  1. Velocity:   v_i = (ℏ/m) Im(j'_i / j_i)   [coherent coupling]
  2. Advection:  X_class = X + v·dt
  3. Candidates:  GH nodes around X_class       [shared cloud]
  4. Evaluate:   ψ-KDE → √ρ, ln ρ at all candidates + departure
  5. STEER:      select candidate ∝ √ρ          [position update]
  6. WEIGH:      M₊ → Q                          [quantum potential]
  7. Backward:   ⟨ln ρ⟩ → u' → Q̃                [Holland bi-HJ]
  8. Phase:      S += (½mv² − V − Q)·dt

Usage:
  python m5_gridless_opt.py               # auto-detect GPU
  python m5_gridless_opt.py --cpu         # force CPU
  python m5_gridless_opt.py --gpu         # force GPU (error if unavailable)
"""

import sys, os, time, warnings, argparse
from m5_utils import output_path
from m5_fft_ref import schrodinger_fft_1d
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Backend selection: cupy (GPU) or numpy (CPU)
# ═══════════════════════════════════════════════════════════════════

def select_backend(force=None):
    """Return (xp, backend_name). force='cpu'|'gpu'|None."""
    if force == 'cpu':
        import numpy as np
        return np, 'numpy (CPU)'
    try:
        import cupy as cp
        # Quick test to ensure GPU is actually usable
        _ = cp.array([1.0])
        if force == 'gpu' or force is None:
            return cp, f'cupy (GPU: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()})'
    except Exception as e:
        if force == 'gpu':
            raise RuntimeError(f"GPU requested but cupy unavailable: {e}")
    import numpy as np
    return np, 'numpy (CPU)'


# ═══════════════════════════════════════════════════════════════════
# Physics constants
# ═══════════════════════════════════════════════════════════════════

HBAR = 1.0; MASS = 1.0; NU = HBAR / (2 * MASS)
SIG_NOISE = (HBAR / MASS) ** 0.5


# ═══════════════════════════════════════════════════════════════════
# Core kernel sums — fully vectorized
# ═══════════════════════════════════════════════════════════════════

def kernel_sums(eval_x, X_src, phi_src, h, xp, cutoff=4.0,
                need_deriv=False, chunk_size=2048):
    """
    Compute ψ-KDE kernel sums at evaluation points.

    eval_x:   (N_eval,)  — points where fields are needed
    X_src:    (N_src,)    — source particle positions
    phi_src:  (N_src,)    — source particle phases S/ℏ
    h:        scalar      — kernel bandwidth

    Returns dict with:
      n:       (N_eval,)  — density KDE  Σ K_h(x-X_j)
      j_re:    (N_eval,)  — Re Σ K_h(x-X_j) cos(φ_j)
      j_im:    (N_eval,)  — Im Σ K_h(x-X_j) sin(φ_j)
    If need_deriv:
      jp_re:   (N_eval,)  — Re Σ K'_h(x-X_j) cos(φ_j)
      jp_im:   (N_eval,)  — Im Σ K'_h(x-X_j) sin(φ_j)
    """
    N_eval = len(eval_x)
    N_src = len(X_src)

    cos_phi = xp.cos(phi_src)
    sin_phi = xp.sin(phi_src)

    n_out = xp.zeros(N_eval, dtype=xp.float64)
    j_re_out = xp.zeros(N_eval, dtype=xp.float64)
    j_im_out = xp.zeros(N_eval, dtype=xp.float64)
    if need_deriv:
        jp_re_out = xp.zeros(N_eval, dtype=xp.float64)
        jp_im_out = xp.zeros(N_eval, dtype=xp.float64)

    inv_h = 1.0 / h
    norm = inv_h / (2.0 * 3.141592653589793) ** 0.5
    radius = cutoff * h

    # Process in chunks to limit memory: (chunk, N_src) matrices
    for i0 in range(0, N_eval, chunk_size):
        i1 = min(i0 + chunk_size, N_eval)
        # (chunk, N_src)
        dx = eval_x[i0:i1, None] - X_src[None, :]

        # Cutoff mask — zero out contributions beyond radius
        mask = xp.abs(dx) < radius
        u = dx * inv_h               # normalised distance
        K = xp.where(mask, norm * xp.exp(-0.5 * u * u), 0.0)

        n_out[i0:i1] = K.sum(axis=1)
        j_re_out[i0:i1] = (K * cos_phi[None, :]).sum(axis=1)
        j_im_out[i0:i1] = (K * sin_phi[None, :]).sum(axis=1)

        if need_deriv:
            # K'_h(dx) = -(dx/h²) K_h(dx)
            Kp = xp.where(mask, -(dx / (h * h)) * K, 0.0)
            jp_re_out[i0:i1] = (Kp * cos_phi[None, :]).sum(axis=1)
            jp_im_out[i0:i1] = (Kp * sin_phi[None, :]).sum(axis=1)

    result = dict(n=n_out, j_re=j_re_out, j_im=j_im_out)
    if need_deriv:
        result['jp_re'] = jp_re_out
        result['jp_im'] = jp_im_out
    return result


def psi_kde_fields(ks, xp, eps_n=1e-30):
    """
    From raw kernel sums, compute √ρ, ln ρ, and optionally v.

    ks: dict from kernel_sums()
    Returns dict with sqrt_rho, ln_rho, and if derivatives present: v, u
    """
    n = xp.maximum(ks['n'], eps_n)
    j_abs = xp.sqrt(ks['j_re']**2 + ks['j_im']**2)
    sqrt_rho = j_abs / xp.sqrt(n)
    # ln ρ = 2 ln|j| - ln n, clamped for safety
    ln_j = xp.log(xp.maximum(j_abs, eps_n))
    ln_n = xp.log(n)
    ln_rho = 2.0 * ln_j - ln_n

    result = dict(sqrt_rho=sqrt_rho, ln_rho=ln_rho, n=n,
                  j_re=ks['j_re'], j_im=ks['j_im'], j_abs=j_abs)

    if 'jp_re' in ks:
        # v = (ℏ/m) Im(j'/j),  u = ν Re(j'/j)
        # j'/j = (jp_re + i jp_im) / (j_re + i j_im)
        # Im(a/b) = (a_im b_re - a_re b_im) / |b|²
        j2 = ks['j_re']**2 + ks['j_im']**2
        # Coherence threshold: set v=0 where |j|² < eps_coh * n²
        eps_coh = 1e-4
        coherent = j2 > eps_coh * n * n
        j2_safe = xp.where(coherent, j2, 1.0)

        im_ratio = (ks['jp_im'] * ks['j_re'] -
                     ks['jp_re'] * ks['j_im']) / j2_safe
        re_ratio = (ks['jp_re'] * ks['j_re'] +
                     ks['jp_im'] * ks['j_im']) / j2_safe

        result['v'] = xp.where(coherent, (HBAR / MASS) * im_ratio, 0.0)
        result['u'] = xp.where(coherent, NU * re_ratio, 0.0)

    return result


# ═══════════════════════════════════════════════════════════════════
# GH quadrature setup
# ═══════════════════════════════════════════════════════════════════

def gh_nodes_weights(K_gh):
    """Gauss-Hermite nodes and normalised weights (CPU, then transfer).

    Uses physicist's Hermite polynomials (weight exp(-t²), E[ξ²]=½).
    Combined with probe offsets η = √2·σ_gh·ξ, the probe variance is:
        E[η²] = 2·σ_gh²·E[ξ²] = 2·σ_gh²·½ = σ_gh²
    which matches the document formulas Q = -(ℏ²/(m·σ_gh²))·(M₊-1)
    and u' = ν·2·(L-L₀)/σ_gh².

    NOTE: the previous version used probabilist's hermegauss (E[ξ²]=1),
    giving E[η²] = 2σ_gh² and doubling Q and u'. Fix applied 2026-03-23.
    """
    import numpy as np
    from numpy.polynomial.hermite import hermgauss
    xi, omega = hermgauss(K_gh)
    omega /= omega.sum()
    return xi, omega


# ═══════════════════════════════════════════════════════════════════
# Schrödinger FFT reference solver (always CPU, via m5_fft_ref)
# ═══════════════════════════════════════════════════════════════════

def schrodinger_ref(psi0_grid, V_grid, x, T, Nt, save_every):
    """Split-operator FFT. Returns (psi_snaps, t_snaps) on CPU."""
    return schrodinger_fft_1d(psi0_grid, V_grid, x, T, Nt,
                              hbar=HBAR, mass=MASS, save_every=save_every)


# ═══════════════════════════════════════════════════════════════════
# Main M5 gridless swarmalator solver
# ═══════════════════════════════════════════════════════════════════

def m5_gridless(psi0_func, V_func, xL, xR, Nx, T, Nt,
                Np=3000, K_gh=6, sigma_gh=0.2, h_kde=0.25,
                seed=42, save_every=50, xp=None, chunk_size=2048):
    """
    Gridless M5 swarmalator with shared GH candidate cloud.

    Parameters
    ----------
    psi0_func : callable  — ψ₀(x), evaluated at arbitrary x
    V_func    : callable  — V(x), external potential
    xL, xR    : float     — domain bounds
    Nx        : int       — grid points (for reference/init only)
    T, Nt     : float,int — total time, number of steps
    Np        : int       — number of particles
    K_gh      : int       — GH quadrature order (shared STEER+WEIGH)
    sigma_gh  : float     — GH probe scale
    h_kde     : float     — kernel bandwidth (physical units)
    xp        : module    — numpy or cupy
    chunk_size: int       — max rows in kernel-sum matrices
    """
    if xp is None:
        import numpy
        xp = numpy

    dt = T / Nt
    h = h_kde

    # ── GH nodes (CPU then transfer) ─────────────────────────────
    xi_cpu, omega_cpu = gh_nodes_weights(K_gh)
    xi = xp.asarray(xi_cpu, dtype=xp.float64)
    omega = xp.asarray(omega_cpu, dtype=xp.float64)

    # ── Initialise particles from ψ₀ ─────────────────────────────
    import numpy as np
    x_grid = np.linspace(xL, xR, Nx, endpoint=False)
    dx = (xR - xL) / Nx
    psi0_grid = psi0_func(x_grid)
    rho0 = np.abs(psi0_grid)**2
    rho0 /= rho0.sum() * dx
    cdf = np.cumsum(rho0) * dx; cdf /= cdf[-1]

    rng = np.random.default_rng(seed)
    X_cpu = np.interp(rng.uniform(size=Np), cdf, x_grid)
    psi_at_X = psi0_func(X_cpu)
    S_cpu = HBAR * np.angle(psi_at_X)

    # Transfer to device
    X = xp.asarray(X_cpu, dtype=xp.float64)
    S = xp.asarray(S_cpu, dtype=xp.float64)

    # Precompute V on a grid for fast lookup (keep on device)
    V_grid_cpu = V_func(x_grid)
    V_grid_dev = xp.asarray(V_grid_cpu, dtype=xp.float64)
    x_grid_dev = xp.asarray(x_grid, dtype=xp.float64)

    # ── Storage ───────────────────────────────────────────────────
    idx_save = list(range(0, Nt + 1, save_every))
    X_hist = []; S_hist = []; t_hist = []
    si = 0

    def to_cpu(arr):
        return arr.get() if hasattr(arr, 'get') else arr

    if 0 in idx_save:
        X_hist.append(to_cpu(X.copy()))
        S_hist.append(to_cpu(S.copy()))
        t_hist.append(0.0)

    t0 = time.time()

    for n in range(1, Nt + 1):
        phi = S / HBAR  # phase array (Np,)

        # ══════════════ STEP 1: VELOCITY ══════════════════════════
        ks_vel = kernel_sums(X, X, phi, h, xp,
                             need_deriv=True, chunk_size=chunk_size)
        fields_vel = psi_kde_fields(ks_vel, xp)
        v_at = fields_vel['v']   # (Np,)

        # ══════════════ STEP 2: CLASSICAL ADVECTION ═══════════════
        X_class = X + v_at * dt
        X_class = xp.clip(X_class, xL + h, xR - h)

        # ══════════════ STEP 3: SHARED GH CANDIDATE CLOUD ════════
        # Candidates: x_jk = X_class_k + √2 · σ_gh · ξ_j
        # Shape: (Np, K_gh) → flattened to (Np*K_gh,) for batch eval
        # Plus departure points X_class (Np,)
        cand_offsets = (2.0 ** 0.5) * sigma_gh * xi  # (K_gh,)
        cands = X_class[:, None] + cand_offsets[None, :]  # (Np, K_gh)
        cands = xp.clip(cands, xL + h, xR - h)
        cands_flat = cands.reshape(-1)  # (Np*K_gh,)

        # Concatenate departure points for a single batch evaluation
        all_eval = xp.concatenate([cands_flat, X_class])  # (Np*K_gh + Np,)

        # ══════════════ STEP 4: ψ-KDE AT ALL POINTS ══════════════
        ks_all = kernel_sums(all_eval, X, phi, h, xp,
                             need_deriv=False, chunk_size=chunk_size)
        f_all = psi_kde_fields(ks_all, xp)

        # Split back
        sqr_cands = f_all['sqrt_rho'][:Np * K_gh].reshape(Np, K_gh)
        lnr_cands = f_all['ln_rho'][:Np * K_gh].reshape(Np, K_gh)
        sqr_depart = f_all['sqrt_rho'][Np * K_gh:]   # (Np,)
        lnr_depart = f_all['ln_rho'][Np * K_gh:]     # (Np,)

        # ══════════════ STEP 5: STEER (selection) ═════════════════
        # Weights: GH_omega * √ρ
        w_sel = omega[None, :] * xp.maximum(sqr_cands, 1e-30)  # (Np, K_gh)
        w_sum = w_sel.sum(axis=1, keepdims=True)
        probs = w_sel / xp.maximum(w_sum, 1e-30)

        # Categorical selection via cumulative sum + uniform draw
        cum = xp.cumsum(probs, axis=1)
        u_rand = xp.asarray(rng.uniform(size=Np), dtype=xp.float64)
        chosen = xp.clip((cum < u_rand[:, None]).sum(axis=1).astype(xp.int64),
                         0, K_gh - 1)

        # Gather selected positions
        idx_row = xp.arange(Np)
        X = cands[idx_row, chosen]

        # ══════════════ STEP 6: FORWARD WEIGH (Q) ════════════════
        # M₊ = Σ ω_j √ρ(x_j) / √ρ(x₀)  (already normalised ω)
        M_plus = (omega[None, :] * sqr_cands).sum(axis=1)
        ratio = M_plus / xp.maximum(sqr_depart, 1e-30)
        Q_at = -(HBAR**2 / (MASS * sigma_gh**2)) * (ratio - 1.0)

        # Clamp extreme Q from tail noise
        Q_med = xp.median(xp.abs(Q_at))
        Q_at = xp.clip(Q_at, -50.0 * Q_med - 1.0, 50.0 * Q_med + 1.0)

        # ══════════════ STEP 7: BACKWARD CHANNEL ═════════════════
        L_k = (omega[None, :] * lnr_cands).sum(axis=1)
        u_prime = NU * 2.0 * (L_k - lnr_depart) / (sigma_gh**2)
        Q_tilde = Q_at + HBAR * u_prime

        # ══════════════ STEP 8: PHASE UPDATE ══════════════════════
        # Interpolate V at new X positions (linear interp on device)
        fi = xp.clip((X - xL) / dx, 0.0, Nx - 1.001)
        lo = fi.astype(xp.int64)
        al = fi - lo
        hi = xp.minimum(lo + 1, Nx - 1)
        V_at = (1.0 - al) * V_grid_dev[lo] + al * V_grid_dev[hi]

        S = S + (0.5 * MASS * v_at**2 - V_at - Q_at) * dt
        # Phase reduction
        TWO_PI_HBAR = 2.0 * 3.141592653589793 * HBAR
        S = xp.mod(S + 0.5 * TWO_PI_HBAR, TWO_PI_HBAR) - 0.5 * TWO_PI_HBAR

        # ══════════════ SAVE ══════════════════════════════════════
        if n in idx_save:
            X_hist.append(to_cpu(X.copy()))
            S_hist.append(to_cpu(S.copy()))
            t_hist.append(n * dt)

        if n % max(1, Nt // 10) == 0:
            elapsed = time.time() - t0
            print(f"    step {n}/{Nt}  ({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t0
    return dict(
        X=np.array(X_hist), S=np.array(S_hist),
        t_save=np.array(t_hist), h_kde=h,
        time=total_time
    )


# ═══════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, s0):
    import numpy as np
    return ((2 * np.pi * s0**2) ** (-0.25) *
            np.exp(-(x - x0)**2 / (4 * s0**2) + 1j * p0 * x / HBAR))


def make_test_cases():
    """Return list of (tag, label, psi0_func, V_func, params_dict)."""
    import numpy as np

    cases = []

    # A: Free Gaussian
    def psi0_free(x):
        return gaussian_wp(x, x0=0, p0=1.5, s0=1.5)
    cases.append(('free_gauss', 'Free Gaussian', psi0_free,
                  lambda x: np.zeros_like(x),
                  dict(xL=-15, xR=15, Nx=512, T=2.0, Nt=2000,
                       Np=3000, K_gh=6, sigma_gh=0.20, h_kde=0.25,
                       save_every=50)))

    # B: Cat state
    x0c, p0c, s0c = 4.0, 3.0, 0.7
    def psi0_cat(x):
        psi = gaussian_wp(x, -x0c, +p0c, s0c) + gaussian_wp(x, +x0c, -p0c, s0c)
        # Normalise analytically is hard; do a rough norm
        return psi
    cases.append(('cat_state', 'Cat State', psi0_cat,
                  lambda x: np.zeros_like(x),
                  dict(xL=-15, xR=15, Nx=512, T=2.0, Nt=2000,
                       Np=4000, K_gh=6, sigma_gh=0.20, h_kde=0.22,
                       save_every=50)))

    # C: HO ground state
    omega = 1.0
    def psi0_ho(x):
        return ((omega * MASS / (np.pi * HBAR)) ** 0.25 *
                np.exp(-omega * MASS * x**2 / (2 * HBAR)))
    cases.append(('ho_ground', 'HO Ground State', psi0_ho,
                  lambda x: 0.5 * MASS * omega**2 * x**2,
                  dict(xL=-8, xR=8, Nx=256, T=3.0, Nt=6000,
                       Np=3000, K_gh=8, sigma_gh=0.15, h_kde=0.22,
                       save_every=100)))

    return cases


# ═══════════════════════════════════════════════════════════════════
# Plotting (always CPU / matplotlib)
# ═══════════════════════════════════════════════════════════════════

def plot_results(psi_ref, ts_ref, x_grid, dx, m5, title, fname, xp_mod):
    """3-panel comparison figure."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    rho_ref = np.abs(psi_ref)**2
    Ns = len(ts_ref)
    bw = 3.0
    edges = np.linspace(x_grid[0], x_grid[-1], len(x_grid) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"M5 Gridless Swarmalator — {title}  "
                 f"(Np={len(m5['X'][0])}, t={m5['time']:.1f}s)",
                 fontsize=13, weight='bold')

    # Panel 1: density
    ax = axes[0]
    snap_idx = [0, Ns // 3, 2 * Ns // 3, Ns - 1]
    colors = ['#2166ac', '#b2182b', '#4dac26', '#7570b3']
    for ci, si in enumerate(snap_idx):
        t_s = ts_ref[si]
        i_m5 = np.argmin(np.abs(m5['t_save'] - t_s))
        ax.plot(x_grid, rho_ref[si], '-', color=colors[ci], lw=2, alpha=0.5)
        h_m5, _ = np.histogram(m5['X'][i_m5], bins=edges, density=True)
        rho_m5 = gaussian_filter1d(
            np.interp(x_grid, 0.5 * (edges[:-1] + edges[1:]), h_m5), sigma=bw)
        ax.plot(x_grid, rho_m5, '--', color=colors[ci], lw=1.5,
                label=f't={t_s:.2f}')
    ax.set_xlim(x_grid[0] * 0.6, x_grid[-1] * 0.6)
    ax.set_xlabel('x'); ax.set_ylabel('ρ')
    ax.legend(fontsize=8); ax.set_title('Density: exact vs M5')

    # Panel 2: L² error
    ax = axes[1]
    errs = []
    for si in range(Ns):
        i_m5 = np.argmin(np.abs(m5['t_save'] - ts_ref[si]))
        h_m5, _ = np.histogram(m5['X'][i_m5], bins=edges, density=True)
        rho_m5 = gaussian_filter1d(
            np.interp(x_grid, 0.5 * (edges[:-1] + edges[1:]), h_m5), sigma=bw)
        errs.append(np.sqrt(np.sum((rho_m5 - rho_ref[si])**2) * dx))
    ax.semilogy(ts_ref, errs, '-', color='#d6604d', lw=1.5)
    ax.set_xlabel('t'); ax.set_ylabel('L² error')
    mean_err = np.mean(errs)
    ax.set_title(f'L² error (mean={mean_err:.5f})')
    ax.grid(True, alpha=0.3)

    # Panel 3: velocity cross-validation at mid-time
    ax = axes[2]
    si_mid = Ns // 2
    psi_mid = psi_ref[si_mid]
    rho_mid = np.abs(psi_mid)**2
    dpsi = (np.roll(psi_mid, -1) - np.roll(psi_mid, 1)) / (2 * dx)
    v_exact = ((HBAR / MASS) * np.imag(np.conj(psi_mid) * dpsi) /
               np.maximum(rho_mid, 1e-30))

    # Recompute swarmalator velocity at snapshot particles
    i_m5 = np.argmin(np.abs(m5['t_save'] - ts_ref[si_mid]))
    X_snap = m5['X'][i_m5]
    S_snap = m5['S'][i_m5]
    phi_snap = S_snap / HBAR

    X_dev = xp_mod.asarray(X_snap, dtype=xp_mod.float64)
    phi_dev = xp_mod.asarray(phi_snap, dtype=xp_mod.float64)
    ks_v = kernel_sums(X_dev, X_dev, phi_dev, m5['h_kde'], xp_mod,
                       need_deriv=True, chunk_size=2048)
    fv = psi_kde_fields(ks_v, xp_mod)
    v_m5 = fv['v']
    if hasattr(v_m5, 'get'):
        v_m5 = v_m5.get()

    mask = rho_mid > 0.005 * np.max(rho_mid)
    ax.plot(x_grid[mask], v_exact[mask], 'k-', lw=2, label='v exact')
    ax.scatter(X_snap, v_m5, s=1, alpha=0.3, color='#d6604d',
               label='v swarmalator')
    ax.set_xlabel('x'); ax.set_ylabel('v')
    ax.legend(fontsize=9)
    ax.set_title(f'Velocity at t={ts_ref[si_mid]:.2f}')
    ax.set_xlim(x_grid[0] * 0.6, x_grid[-1] * 0.6)

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return mean_err


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--gpu', action='store_true', help='Force GPU')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced params for quick testing')
    parser.add_argument('--test', type=str, default=None,
                        help='Run single test: free_gauss|cat_state|ho_ground')
    args = parser.parse_args()

    force = 'cpu' if args.cpu else ('gpu' if args.gpu else None)
    xp, backend_name = select_backend(force)

    print("=" * 65)
    print("  M5 Gridless Swarmalator — Optimized")
    print(f"  Backend: {backend_name}")
    print("=" * 65)

    cases = make_test_cases()
    if args.test:
        cases = [(t, l, p, v, d) for t, l, p, v, d in cases if t == args.test]
        if not cases:
            print(f"Unknown test: {args.test}")
            sys.exit(1)

    if args.quick:
        print("  [QUICK MODE: reduced parameters]")
        for _, _, _, _, d in cases:
            d['Np'] = min(d['Np'], 600)
            d['Nt'] = min(d['Nt'], 600)
            d['T'] = min(d['T'], 1.0)
            d['h_kde'] = max(d['h_kde'], 0.35)
            d['save_every'] = max(1, d['Nt'] // 15)

    results = {}

    for tag, label, psi0_func, V_func, params in cases:
        print(f"\n{'─' * 55}")
        print(f"  {label}")
        for k, v in params.items():
            print(f"    {k} = {v}")
        print(f"{'─' * 55}")

        xL = params['xL']; xR = params['xR']; Nx = params['Nx']
        dx = (xR - xL) / Nx
        x_grid = np.linspace(xL, xR, Nx, endpoint=False)

        # Normalise ψ₀ on grid
        psi0_grid = psi0_func(x_grid)
        norm = np.sqrt(np.sum(np.abs(psi0_grid)**2) * dx)
        psi0_func_normed = lambda x, _f=psi0_func, _n=norm: _f(x) / _n
        psi0_grid /= norm

        # FFT reference
        print("  FFT reference...", end=" ", flush=True)
        t0 = time.time()
        V_grid = V_func(x_grid)
        psi_ref, ts_ref = schrodinger_ref(
            psi0_grid, V_grid, x_grid, params['T'], params['Nt'],
            params['save_every'])
        print(f"({time.time() - t0:.1f}s)")

        # M5 gridless
        print("  M5 gridless swarmalator...", flush=True)
        m5 = m5_gridless(
            psi0_func_normed, V_func, xL, xR, Nx,
            params['T'], params['Nt'],
            Np=params['Np'], K_gh=params['K_gh'],
            sigma_gh=params['sigma_gh'], h_kde=params['h_kde'],
            save_every=params['save_every'], xp=xp,
            chunk_size=4096 if xp.__name__ == 'cupy' else 2048)
        print(f"  done ({m5['time']:.1f}s)")

        # Plot
        fname = output_path(f"m5_swarmalator_{tag}.png")
        mean_err = plot_results(
            psi_ref, ts_ref, x_grid, dx, m5, label, fname, xp)
        results[tag] = dict(err=mean_err, time=m5['time'])
        print(f"  L² = {mean_err:.5f},  time = {m5['time']:.1f}s")

    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  ({backend_name})")
    print(f"{'=' * 55}")
    for tag, r in results.items():
        print(f"  {tag:15s}  L²={r['err']:.5f}  time={r['time']:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
