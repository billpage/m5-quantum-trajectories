#!/usr/bin/env python3
"""
M5ψ-KDE:  Unified ψ-Space Kernel Density Estimation for Method 5
==================================================================

Core idea
---------
Replace the two independent smoothing operations of original M5
(histogram+smooth for ρ, bin-average+smooth for S) with a single
principled estimator:

    ψ̂(x) = j_h(x) / √n_h(x)

where:
    n_h(x) = (1/Np) Σ K_h(x - X_i)              — density KDE
    j_h(x) = (1/Np) Σ K_h(x - X_i) · e^{iS_i/ℏ} — complex current KDE

Both use the SAME Gaussian kernel bandwidth h = σ·dx.

Key properties:
  • Single smoothing — no double-smoothing, no bin-averaging
  • Nodes handled automatically via destructive interference in j_h
  • Single bandwidth parameter σ (replaces σ_ρ + σ_S)
  • No per-particle amplitude tracking needed
  • Same O(Np + Nx log Nx) cost as original M5

Implementation: CIC (cloud-in-cell) deposit + Gaussian convolution via
scipy's gaussian_filter1d.

Comparison
----------
We test three methods on the cat-state collision:
  (1) Original M5     (σ_ρ=4, σ_S=3, two independent smoothings)
  (2) M5ψ-prototype   (bin-average ψ + single smooth, carries amp_p)
  (3) M5ψ-KDE         (CIC deposit + single smooth, no amp_p)
against Schrödinger FFT reference.

Outputs
-------
  fig1  — 3-way density comparison at key times + L² error vs time
  fig2  — σ sweep: L² error vs bandwidth for M5ψ-KDE
  fig3  — Node diagnostics: ψ̂ components through interference region
  fig4  — Field comparison: v(x) and Q(x) from all three methods
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, warnings
from m5_utils import output_path
from m5_fft_ref import schrodinger_fft_1d
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════

class QP:
    hbar = 1.0; m = 1.0
    nu = hbar / (2*m)
    sigma_noise = np.sqrt(hbar / m)

class GP:
    xL = -15.; xR = 15.; Nx = 512; T = 3.0; Nt = 2000
    dx = (xR - xL) / Nx
    dt = T / Nt
    x  = np.linspace(xL, xR, Nx, endpoint=False)
    t_arr = np.linspace(0, T, Nt+1)

qp = QP(); gp = GP()


def Dx(f, dx):
    """Central difference derivative (periodic)."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2*dx)

def D2x(f, dx):
    """Central difference 2nd derivative (periodic)."""
    return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / dx**2


# ═══════════════════════════════════════════════════════════════════
# Initial condition — cat state
# ═══════════════════════════════════════════════════════════════════

def cat_state(x, x0=4.0, p0=3.0, s0=0.7, hbar=1.0):
    """Two counter-propagating Gaussian wave packets."""
    def wp(x, x0, p0):
        return (2*np.pi*s0**2)**(-.25) * np.exp(
            -(x - x0)**2 / (4*s0**2) + 1j*p0*x/hbar)
    psi = wp(x, -x0, +p0) + wp(x, +x0, -p0)
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * gp.dx)


# ═══════════════════════════════════════════════════════════════════
# Schrödinger FFT reference (via m5_fft_ref)
# ═══════════════════════════════════════════════════════════════════

def schrodinger_ref(psi0, save_every=10):
    """Split-step FFT Schrödinger solver (V=0 free case).
    Returns (rho_snaps, psi_snaps, t_snaps)."""
    V = np.zeros(gp.Nx)
    psi_h, t_save = schrodinger_fft_1d(psi0, V, gp.x, gp.T, gp.Nt,
                                        hbar=qp.hbar, mass=qp.m,
                                        save_every=save_every)
    return np.abs(psi_h)**2, psi_h, t_save


# ═══════════════════════════════════════════════════════════════════
# Particle initialisation (shared by all methods)
# ═══════════════════════════════════════════════════════════════════

def init_particles(psi0, Np, seed=42):
    """Sample Np positions from |ψ₀|² and assign phases from S₀."""
    rng = np.random.default_rng(seed)
    rho0 = np.abs(psi0)**2
    cdf = np.cumsum(rho0) * gp.dx; cdf /= cdf[-1]
    X  = np.interp(rng.uniform(size=Np), cdf, gp.x)
    S0 = qp.hbar * np.unwrap(np.angle(psi0))
    Sp = np.interp(X, gp.x, S0)
    return X, Sp


# ═══════════════════════════════════════════════════════════════════
# METHOD A:  Original M5 (two independent smoothings)
# ═══════════════════════════════════════════════════════════════════

def m5_original(psi0, Np=6000, K=48, seed=42, save_every=10,
                sigma_rho=4.0, sigma_S=3.0):
    """Original M5: histogram+smooth for ρ, bin-average+smooth for S."""
    rng = np.random.default_rng(seed)
    hbar, m, sig = qp.hbar, qp.m, qp.sigma_noise
    dx, dt, x, Nx = gp.dx, gp.dt, gp.x, gp.Nx

    X, Sp = init_particles(psi0, Np, seed)
    rho0 = np.abs(psi0)**2
    rho_est = rho0 / (np.sum(rho0)*dx)

    idx_save = list(range(0, gp.Nt+1, save_every))
    Ns = len(idx_save)
    X_h = np.zeros((Ns, Np)); rho_h = np.zeros((Ns, Nx))
    psi_h = np.zeros((Ns, Nx), dtype=complex)
    si = 0
    if 0 in idx_save:
        X_h[si] = X.copy(); rho_h[si] = rho_est.copy()
        psi_h[si] = psi0.copy(); si += 1

    for n in range(1, gp.Nt+1):
        # density: histogram + smooth
        h, _ = np.histogram(X, bins=Nx, range=(gp.xL, gp.xR))
        rho_est = gaussian_filter1d(h.astype(float), sigma=sigma_rho)
        rho_est = np.maximum(rho_est, 1e-30)
        rho_est /= np.sum(rho_est)*dx
        sqrt_rho = np.sqrt(rho_est)

        # phase: bin-average + smooth
        S_field = np.zeros(Nx); cnt = np.zeros(Nx)
        bi = np.clip(((X - gp.xL)/dx).astype(int), 0, Nx-1)
        np.add.at(S_field, bi, Sp); np.add.at(cnt, bi, 1.0)
        ok = cnt > 0
        S_field[ok] /= cnt[ok]
        if not np.all(ok):
            S_field[~ok] = np.interp(x[~ok], x[ok], S_field[ok])
        S_field = gaussian_filter1d(S_field, sigma=sigma_S)
        v_field = Dx(S_field, dx) / m
        v_at = np.interp(X, x, v_field)

        # √ρ selection
        X_class = X + v_at*dt
        noise = rng.normal(size=(Np, K)) * sig * np.sqrt(dt)
        cands = np.clip(X_class[:, None] + noise, gp.xL+dx, gp.xR-dx)
        fi = np.clip((cands - gp.xL)/dx, 0.0, Nx-1.001)
        lo = np.clip(fi.astype(np.int64), 0, Nx-2); hi = lo + 1
        al = np.clip(fi - lo, 0.0, 1.0)
        w  = np.maximum((1-al)*sqrt_rho[lo] + al*sqrt_rho[hi], 1e-30)
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        ck  = np.clip((cum < rng.uniform(size=Np)[:, None]).sum(axis=1), 0, K-1)
        X   = cands[np.arange(Np), ck]

        # action update
        sqr = np.sqrt(np.maximum(rho_est, 1e-30))
        Q_f = -(hbar**2/(2*m)) * D2x(sqr, dx) / np.maximum(sqr, 1e-30)
        Q_f = gaussian_filter1d(Q_f, sigma=2.0)
        Sp += (0.5*m*v_at**2 - np.interp(X, x, Q_f)) * dt

        if n in idx_save and si < Ns:
            X_h[si] = X.copy(); rho_h[si] = rho_est.copy()
            # reconstruct ψ-grid from independent estimates (for diagnostics)
            psi_h[si] = sqrt_rho * np.exp(1j * S_field / hbar)
            si += 1

    for j in range(si, Ns):
        X_h[j] = X.copy(); rho_h[j] = rho_est.copy()
        psi_h[j] = sqrt_rho * np.exp(1j * S_field / hbar)
    return dict(X=X_h, rho=rho_h, psi=psi_h, t_save=gp.t_arr[idx_save[:Ns]])


# ═══════════════════════════════════════════════════════════════════
# METHOD B:  M5ψ-prototype (bin-average ψ + single smooth)
# ═══════════════════════════════════════════════════════════════════

def m5psi_prototype(psi0, Np=6000, K=48, seed=42, save_every=10,
                    sigma_psi=2.5):
    """M5ψ prototype: bin-average complex ψ_p, single Gaussian smooth."""
    rng = np.random.default_rng(seed)
    hbar, m, sig = qp.hbar, qp.m, qp.sigma_noise
    dx, dt, x, Nx = gp.dx, gp.dt, gp.x, gp.Nx

    X, Sp = init_particles(psi0, Np, seed)
    rho0 = np.abs(psi0)**2
    amp_p = np.sqrt(np.maximum(np.interp(X, x, rho0), 1e-30))

    idx_save = list(range(0, gp.Nt+1, save_every))
    Ns = len(idx_save)
    X_h = np.zeros((Ns, Np)); rho_h = np.zeros((Ns, Nx))
    psi_h = np.zeros((Ns, Nx), dtype=complex)
    si = 0
    rho_est = rho0 / (np.sum(rho0)*dx)
    if 0 in idx_save:
        X_h[si] = X.copy(); rho_h[si] = rho_est.copy()
        psi_h[si] = psi0.copy(); si += 1
    eps = 1e-30

    for n in range(1, gp.Nt+1):
        # build ψ_grid: bin-average complex ψ_p
        psi_p = amp_p * np.exp(1j * Sp / hbar)
        psi_re = np.zeros(Nx); psi_im = np.zeros(Nx); cnt = np.zeros(Nx)
        bi = np.clip(((X - gp.xL)/dx).astype(int), 0, Nx-1)
        np.add.at(psi_re, bi, psi_p.real)
        np.add.at(psi_im, bi, psi_p.imag)
        np.add.at(cnt, bi, 1.0)
        ok = cnt > 0
        psi_re[ok] /= cnt[ok]; psi_im[ok] /= cnt[ok]
        if not np.all(ok):
            psi_re[~ok] = np.interp(x[~ok], x[ok], psi_re[ok])
            psi_im[~ok] = np.interp(x[~ok], x[ok], psi_im[ok])

        # single smooth
        psi_re = gaussian_filter1d(psi_re, sigma=sigma_psi)
        psi_im = gaussian_filter1d(psi_im, sigma=sigma_psi)
        psi_grid = psi_re + 1j * psi_im
        norm = np.sqrt(np.sum(np.abs(psi_grid)**2) * dx)
        psi_grid /= max(norm, eps)

        # derive fields
        rho_est  = np.abs(psi_grid)**2
        sqrt_rho = np.abs(psi_grid)
        dpsi     = Dx(psi_grid, dx)
        v_field  = (hbar/m) * np.imag(np.conj(psi_grid)*dpsi) / np.maximum(rho_est, eps)
        sqr      = sqrt_rho
        Q_f      = -(hbar**2/(2*m)) * D2x(sqr, dx) / np.maximum(sqr, eps)
        Q_f      = gaussian_filter1d(Q_f, sigma=1.5)
        div_v    = Dx(v_field, dx)

        # √ρ selection
        v_at = np.interp(X, x, v_field)
        X_class = X + v_at * dt
        noise = rng.normal(size=(Np, K)) * sig * np.sqrt(dt)
        cands = np.clip(X_class[:, None] + noise, gp.xL+dx, gp.xR-dx)
        fi = np.clip((cands - gp.xL)/dx, 0.0, Nx-1.001)
        lo = np.clip(fi.astype(np.int64), 0, Nx-2); hi = lo + 1
        al = np.clip(fi - lo, 0.0, 1.0)
        w  = np.maximum((1-al)*sqrt_rho[lo] + al*sqrt_rho[hi], eps)
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        ck  = np.clip((cum < rng.uniform(size=Np)[:, None]).sum(axis=1), 0, K-1)
        X   = cands[np.arange(Np), ck]

        # update per-particle state
        Q_at = np.interp(X, x, Q_f)
        Sp  += (0.5*m*v_at**2 - Q_at) * dt
        div_at = np.interp(X, x, div_v)
        amp_p *= np.exp(-0.5 * div_at * dt)
        amp_p  = np.maximum(amp_p, eps)
        amp_norm = np.sqrt(np.sum(amp_p**2) * dx / Np)
        amp_p /= max(amp_norm, eps)

        if n in idx_save and si < Ns:
            X_h[si] = X.copy(); rho_h[si] = rho_est.copy()
            psi_h[si] = psi_grid.copy(); si += 1

    for j in range(si, Ns):
        X_h[j] = X.copy(); rho_h[j] = rho_est.copy()
        psi_h[j] = psi_grid.copy()
    return dict(X=X_h, rho=rho_h, psi=psi_h, t_save=gp.t_arr[idx_save[:Ns]])


# ═══════════════════════════════════════════════════════════════════
# METHOD C:  M5ψ-KDE (CIC deposit + single smooth + j/√n)
# ═══════════════════════════════════════════════════════════════════

def psi_kde_estimate(X, Sp, sigma, hbar=1.0):
    """
    ψ-KDE field estimation from particle ensemble {X_i, S_i}.

    Steps:
      1. CIC deposit of density weights and phase-modulated weights
      2. Gaussian convolution (single bandwidth σ)
      3. Form ψ̂ = j / √n

    Parameters
    ----------
    X     : (Np,) particle positions
    Sp    : (Np,) particle action values
    sigma : smoothing bandwidth in grid cells

    Returns
    -------
    psi_grid  : (Nx,) complex — estimated wave function on grid
    n_smooth  : (Nx,) real — smoothed density (unnormalised KDE, ∝ ρ)
    j_smooth  : (Nx,) complex — smoothed complex current
    """
    dx, Nx, x = gp.dx, gp.Nx, gp.x
    Np = len(X)

    # ── Step 1: CIC deposit ──────────────────────────────────────
    # Each particle deposits to two nearest grid points with linear weights
    n_raw  = np.zeros(Nx)
    j_re   = np.zeros(Nx)
    j_im   = np.zeros(Nx)

    # Fractional grid index for each particle
    fi = (X - gp.xL) / dx
    fi = np.clip(fi, 0.0, Nx - 1.001)
    k_L = fi.astype(np.int64)          # left grid index
    alpha = fi - k_L                    # fractional offset, 0 ≤ α < 1
    k_R = np.minimum(k_L + 1, Nx - 1)  # right grid index

    # Phase factors
    phi = Sp / hbar
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Deposit density (unit weight per particle)
    np.add.at(n_raw, k_L, 1.0 - alpha)
    np.add.at(n_raw, k_R, alpha)

    # Deposit complex current (phase-weighted)
    np.add.at(j_re, k_L, (1.0 - alpha) * cos_phi)
    np.add.at(j_re, k_R, alpha * cos_phi)
    np.add.at(j_im, k_L, (1.0 - alpha) * sin_phi)
    np.add.at(j_im, k_R, alpha * sin_phi)

    # ── Step 2: Gaussian convolution (single bandwidth) ──────────
    n_smooth  = gaussian_filter1d(n_raw,  sigma=sigma) / Np
    j_re_smooth = gaussian_filter1d(j_re, sigma=sigma) / Np
    j_im_smooth = gaussian_filter1d(j_im, sigma=sigma) / Np
    j_smooth = j_re_smooth + 1j * j_im_smooth

    # ── Step 3: Form ψ̂ = j / √n ─────────────────────────────────
    eps_n = 1e-30
    sqrt_n = np.sqrt(np.maximum(n_smooth, eps_n))
    psi_grid = j_smooth / sqrt_n

    # Normalise so ∫|ψ̂|² dx = 1
    norm = np.sqrt(np.sum(np.abs(psi_grid)**2) * dx)
    if norm > eps_n:
        psi_grid /= norm

    return psi_grid, n_smooth, j_smooth


def m5psi_kde(psi0, Np=6000, K=48, seed=42, save_every=10, sigma=2.5):
    """
    M5ψ-KDE: ψ-space kernel density estimation.

    Per-particle state: (X_i, S_i) only — no amplitude tracking.
    Field estimation: CIC deposit → single Gaussian smooth → ψ̂ = j/√n.
    """
    rng = np.random.default_rng(seed)
    hbar, m, sig_noise = qp.hbar, qp.m, qp.sigma_noise
    dx, dt, x, Nx = gp.dx, gp.dt, gp.x, gp.Nx

    X, Sp = init_particles(psi0, Np, seed)

    idx_save = list(range(0, gp.Nt+1, save_every))
    Ns = len(idx_save)
    X_h   = np.zeros((Ns, Np))
    rho_h = np.zeros((Ns, Nx))
    psi_h = np.zeros((Ns, Nx), dtype=complex)
    v_h   = np.zeros((Ns, Nx))              # save v-field for diagnostics
    si = 0

    eps = 1e-30

    # Initial snapshot
    if 0 in idx_save:
        X_h[si] = X.copy()
        psi_grid, _, _ = psi_kde_estimate(X, Sp, sigma, hbar)
        rho_h[si]  = np.abs(psi_grid)**2
        psi_h[si]  = psi_grid.copy()
        v_h[si]    = np.zeros(Nx)
        si += 1

    for n in range(1, gp.Nt+1):
        # ── STEP 1: ψ-KDE field estimation ──────────────────────
        psi_grid, n_smooth, j_smooth = psi_kde_estimate(X, Sp, sigma, hbar)

        # Derived fields — all from the single smooth ψ̂
        rho_est  = np.abs(psi_grid)**2
        sqrt_rho = np.abs(psi_grid)

        # Current velocity: v = (ℏ/m) Im(ψ̂* ∂ψ̂) / |ψ̂|²
        dpsi    = Dx(psi_grid, dx)
        v_field = (hbar/m) * np.imag(np.conj(psi_grid) * dpsi) / np.maximum(rho_est, eps)

        # Quantum potential: Q = -(ℏ²/2m) ∂²|ψ̂| / |ψ̂|
        Q_f = -(hbar**2/(2*m)) * D2x(sqrt_rho, dx) / np.maximum(sqrt_rho, eps)
        Q_f = gaussian_filter1d(Q_f, sigma=1.5)   # light smooth on Q only

        # ── STEP 2: Current velocity advection ───────────────────
        v_at    = np.interp(X, x, v_field)
        X_class = X + v_at * dt

        # ── STEP 3: √ρ-weighted candidate selection ─────────────
        noise = rng.normal(size=(Np, K)) * sig_noise * np.sqrt(dt)
        cands = np.clip(X_class[:, None] + noise, gp.xL+dx, gp.xR-dx)

        # Interpolate √ρ = |ψ̂| at candidate positions
        fi = np.clip((cands - gp.xL)/dx, 0.0, Nx-1.001)
        lo = np.clip(fi.astype(np.int64), 0, Nx-2); hi = lo + 1
        al = np.clip(fi - lo, 0.0, 1.0)
        w  = np.maximum((1-al)*sqrt_rho[lo] + al*sqrt_rho[hi], eps)

        # Categorical selection proportional to weights
        w_norm = w / w.sum(axis=1, keepdims=True)
        cum = np.cumsum(w_norm, axis=1)
        ck  = np.clip((cum < rng.uniform(size=Np)[:, None]).sum(axis=1), 0, K-1)
        X   = cands[np.arange(Np), ck]

        # ── STEP 4: Action (phase) update ────────────────────────
        Q_at = np.interp(X, x, Q_f)
        Sp  += (0.5*m*v_at**2 - Q_at) * dt     # V=0 for free particle

        # Reduce phase to [-πℏ, πℏ] to prevent float precision loss
        Sp = np.remainder(Sp + np.pi*hbar, 2*np.pi*hbar) - np.pi*hbar

        # ── Save snapshot ────────────────────────────────────────
        if n in idx_save and si < Ns:
            X_h[si]   = X.copy()
            rho_h[si] = rho_est.copy()
            psi_h[si] = psi_grid.copy()
            v_h[si]   = v_field.copy()
            si += 1

    for j in range(si, Ns):
        X_h[j] = X.copy(); rho_h[j] = rho_est.copy()
        psi_h[j] = psi_grid.copy(); v_h[j] = v_field.copy()

    return dict(X=X_h, rho=rho_h, psi=psi_h, v=v_h,
                t_save=gp.t_arr[idx_save[:Ns]])


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def l2_error(rho_a, rho_b):
    return np.sqrt(np.sum((rho_a - rho_b)**2) * gp.dx)


def mean_l2(result, rho_ref, ts_ref):
    """Compute mean L² error across all saved time steps."""
    errs = []
    for i, t in enumerate(ts_ref):
        j = np.argmin(np.abs(result['t_save'] - t))
        errs.append(l2_error(result['rho'][j], rho_ref[i]))
    return np.mean(errs), np.array(errs)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Three-way density comparison + L² error
# ═══════════════════════════════════════════════════════════════════

def fig_comparison(rho_ref, psi_ref, ts_ref, m5, m5p, m5k, Np, K, sigma):
    """
    Row 0: density snapshots at t=0, t_collision, T
    Row 1: zoom on interference fringes at t_c; L² error vs time
    """
    x = gp.x
    t_c = 4.0 / 3.0   # collision time for x0=4, p0=3

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    # Time indices for snapshots
    snap_times = [0.0, t_c, gp.T * 0.95]
    snap_labels = [f't = {t:.2f}' for t in snap_times]

    colors = {'exact': 'k', 'm5': '#2166ac', 'm5p': '#7570b3', 'm5k': '#d6604d'}
    labels = {'exact': 'Exact (FFT)', 'm5': 'M5 original (σ_ρ=4,σ_S=3)',
              'm5p': 'M5ψ prototype (σ=2.5)', 'm5k': f'M5ψ-KDE (σ={sigma})'}

    # ── Row 0: density snapshots ─────────────────────────────────
    for col, (t_snap, lab) in enumerate(zip(snap_times, snap_labels)):
        ax = fig.add_subplot(gs[0, col])
        i_ref = np.argmin(np.abs(ts_ref - t_snap))

        ax.plot(x, rho_ref[i_ref], '-', color=colors['exact'],
                lw=2.5, label=labels['exact'], zorder=5)

        for key, res, ls, lw in [('m5', m5, '--', 1.5),
                                  ('m5p', m5p, ':', 1.8),
                                  ('m5k', m5k, '-', 1.8)]:
            i_m = np.argmin(np.abs(res['t_save'] - t_snap))
            ax.plot(x, res['rho'][i_m], ls, color=colors[key],
                    lw=lw, label=labels[key], zorder=3 if key=='m5' else 4)

        ax.set_title(lab, fontsize=12)
        ax.set_xlim(-8, 8); ax.set_xlabel('x')
        if col == 0:
            ax.set_ylabel('ρ(x,t)')
            ax.legend(fontsize=7, loc='upper left')

    # ── Row 1, left: zoom on fringes at t_c ──────────────────────
    ax = fig.add_subplot(gs[1, 0])
    i_ref = np.argmin(np.abs(ts_ref - t_c))
    ax.plot(x, rho_ref[i_ref], '-', color='k', lw=2.5, label='Exact')
    for key, res, ls, lw in [('m5', m5, '--', 1.5),
                              ('m5p', m5p, ':', 1.8),
                              ('m5k', m5k, '-', 1.8)]:
        i_m = np.argmin(np.abs(res['t_save'] - t_c))
        ax.plot(x, res['rho'][i_m], ls, color=colors[key], lw=lw, label=labels[key])
    ax.set_xlim(-3, 3); ax.set_xlabel('x'); ax.set_ylabel('ρ(x,t)')
    ax.set_title(f'Fringe zoom at t={t_c:.2f}', fontsize=11)
    ax.legend(fontsize=7)

    # ── Row 1, middle: fringe zoom — absolute error ──────────────
    ax = fig.add_subplot(gs[1, 1])
    for key, res, ls, lw in [('m5', m5, '--', 1.5),
                              ('m5p', m5p, ':', 1.8),
                              ('m5k', m5k, '-', 1.8)]:
        i_m = np.argmin(np.abs(res['t_save'] - t_c))
        err = np.abs(res['rho'][i_m] - rho_ref[i_ref])
        ax.plot(x, err, ls, color=colors[key], lw=lw, label=labels[key])
    ax.set_xlim(-3, 3); ax.set_xlabel('x'); ax.set_ylabel('|ρ - ρ_exact|')
    ax.set_title(f'Absolute error at t={t_c:.2f}', fontsize=11)
    ax.legend(fontsize=7)

    # ── Row 1, right: L² error vs time ───────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for key, res, ls, lw in [('m5', m5, '--', 1.5),
                              ('m5p', m5p, ':', 1.8),
                              ('m5k', m5k, '-', 1.8)]:
        mean_err, errs = mean_l2(res, rho_ref, ts_ref)
        # compute error at each ref time
        ts_plot = []
        es_plot = []
        for i, t in enumerate(ts_ref):
            j = np.argmin(np.abs(res['t_save'] - t))
            ts_plot.append(t)
            es_plot.append(l2_error(res['rho'][j], rho_ref[i]))
        ax.semilogy(ts_plot, es_plot, ls, color=colors[key], lw=lw,
                    label=f'{labels[key]}\n  (mean={mean_err:.5f})')

    ax.set_xlabel('t'); ax.set_ylabel('L² error')
    ax.set_title('L² density error vs time', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Cat State Collision — M5 vs M5ψ-proto vs M5ψ-KDE  (Np={Np}, K={K})',
                 fontsize=14, weight='bold')
    return fig


# ═══════════════════════════════════════════════════════════════════
# Figure 2: σ sweep for M5ψ-KDE
# ═══════════════════════════════════════════════════════════════════

def fig_sigma_sweep(rho_ref, ts_ref, sigma_list, results, err_m5, err_m5p, Np, K):
    """Mean L² error vs σ for M5ψ-KDE, with M5 and M5ψ-proto baselines."""
    mean_errs = []
    for res in results:
        me, _ = mean_l2(res, rho_ref, ts_ref)
        mean_errs.append(me)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: error vs σ
    ax = axes[0]
    ax.plot(sigma_list, mean_errs, 'o-', color='#d6604d', lw=2, ms=8,
            label='M5ψ-KDE')
    ax.axhline(err_m5, color='#2166ac', ls='--', lw=1.5,
               label=f'M5 original ({err_m5:.5f})')
    ax.axhline(err_m5p, color='#7570b3', ls=':', lw=1.5,
               label=f'M5ψ-proto ({err_m5p:.5f})')
    i_opt = int(np.argmin(mean_errs))
    ax.plot(sigma_list[i_opt], mean_errs[i_opt], '*', color='gold',
            ms=18, zorder=5)
    ax.annotate(f'σ*={sigma_list[i_opt]:.1f}\nerr={mean_errs[i_opt]:.5f}',
                xy=(sigma_list[i_opt], mean_errs[i_opt]),
                xytext=(sigma_list[i_opt]+0.5, mean_errs[i_opt]*1.2),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='k'))
    ax.set_xlabel('σ (grid cells)', fontsize=11)
    ax.set_ylabel('Mean L² error', fontsize=11)
    ax.set_title('M5ψ-KDE: accuracy vs bandwidth', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: fringes at t_c for best vs worst σ
    ax = axes[1]
    t_c = 4.0 / 3.0
    i_ref = np.argmin(np.abs(ts_ref - t_c))
    ax.plot(gp.x, rho_ref[i_ref], 'k-', lw=2.5, label='Exact')
    i_best = i_opt
    i_worst = int(np.argmax(mean_errs))
    for idx, lab_suffix, ls, clr in [(i_best, 'best', '-', '#d6604d'),
                                      (i_worst, 'worst', '--', '#999999')]:
        res = results[idx]
        i_m = np.argmin(np.abs(res['t_save'] - t_c))
        ax.plot(gp.x, res['rho'][i_m], ls, color=clr, lw=1.8,
                label=f'σ={sigma_list[idx]:.1f} ({lab_suffix})')
    ax.set_xlim(-3, 3); ax.set_xlabel('x'); ax.set_ylabel('ρ')
    ax.set_title(f'Fringes at t={t_c:.2f}: best vs worst σ', fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle(f'M5ψ-KDE bandwidth sweep  (Np={Np}, K={K})', fontsize=13, weight='bold')
    fig.tight_layout()
    return fig, sigma_list[i_opt], mean_errs[i_opt]


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Node diagnostics — ψ̂ components through nodes
# ═══════════════════════════════════════════════════════════════════

def fig_node_diagnostics(rho_ref, psi_ref, ts_ref, m5k, sigma):
    """
    At collision time, show:
      (a) Exact: Re(ψ), Im(ψ), |ψ|
      (b) M5ψ-KDE: Re(ψ̂), Im(ψ̂), |ψ̂|  — smooth through nodes
      (c) The raw KDE components n_h and |j_h| — showing destructive interference
      (d) Phase comparison: arg(ψ_exact) vs arg(ψ̂)
    """
    x = gp.x; dx = gp.dx
    t_c = 4.0 / 3.0
    i_ref = np.argmin(np.abs(ts_ref - t_c))
    i_kde = np.argmin(np.abs(m5k['t_save'] - t_c))

    psi_ex = psi_ref[i_ref]
    psi_k  = m5k['psi'][i_kde]

    # Recompute KDE components for display (using saved X, Sp at t_c)
    X_tc = m5k['X'][i_kde]
    # We need the Sp at this time — reconstruct from psi_grid phase
    # Actually, let's recompute j and n directly from the saved psi_grid
    # More informative: recompute from particles
    # But we don't have Sp saved... Let's compute n_h from just X
    n_raw = np.zeros(gp.Nx)
    fi = np.clip((X_tc - gp.xL)/dx, 0.0, gp.Nx-1.001)
    k_L = fi.astype(np.int64); alpha = fi - k_L
    k_R = np.minimum(k_L + 1, gp.Nx-1)
    np.add.at(n_raw, k_L, 1.0 - alpha)
    np.add.at(n_raw, k_R, alpha)
    Np = len(X_tc)
    n_smooth = gaussian_filter1d(n_raw, sigma=sigma) / Np

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Exact ψ
    ax = axes[0, 0]
    ax.plot(x, np.abs(psi_ex),   'k-',  lw=2.0, label='|ψ|')
    ax.plot(x, psi_ex.real,      '-',   color='#1b7837', lw=1.2, alpha=0.8, label='Re ψ')
    ax.plot(x, psi_ex.imag,      '--',  color='#762a83', lw=1.2, alpha=0.8, label='Im ψ')
    ax.axhline(0, color='gray', lw=0.5, ls='-')
    ax.set_xlim(-5, 5); ax.set_title('Exact ψ(x) at t_c', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('amplitude'); ax.legend(fontsize=9)

    # (b) M5ψ-KDE ψ̂
    ax = axes[0, 1]
    ax.plot(x, np.abs(psi_k),   '-',   color='#d6604d', lw=2.0, label='|ψ̂| (KDE)')
    ax.plot(x, np.abs(psi_ex),  'k--', lw=1.0, alpha=0.4, label='|ψ| exact')
    ax.plot(x, psi_k.real,      '-',   color='#1b7837', lw=1.0, alpha=0.7, label='Re ψ̂')
    ax.plot(x, psi_k.imag,      '-',   color='#762a83', lw=1.0, alpha=0.7, label='Im ψ̂')
    ax.axhline(0, color='gray', lw=0.5, ls='-')
    ax.set_xlim(-5, 5); ax.set_title(f'M5ψ-KDE ψ̂(x) at t_c  (σ={sigma})', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('amplitude'); ax.legend(fontsize=9)

    # (c) KDE components: n_h and |j_h| showing destructive interference
    ax = axes[1, 0]
    rho_ex = np.abs(psi_ex)**2
    # n_smooth ∝ ρ; scale for comparison
    n_scale = n_smooth / (np.sum(n_smooth)*dx) if np.sum(n_smooth) > 0 else n_smooth
    ax.plot(x, rho_ex, 'k-', lw=2.0, label='ρ exact')
    ax.plot(x, n_scale, '--', color='#e08214', lw=1.8,
            label='n_h (density KDE, ∝ ρ)')
    ax.plot(x, np.abs(psi_k)**2, '-', color='#d6604d', lw=1.8,
            label='|ψ̂|² = ρ from KDE')
    ax.set_xlim(-5, 5); ax.set_xlabel('x'); ax.set_ylabel('density')
    ax.set_title('Density: n_h sees particles, |ψ̂|² sees interference', fontsize=10)
    ax.legend(fontsize=8)

    # (d) Phase comparison
    ax = axes[1, 1]
    # Extract phases (only plot where amplitude is significant)
    mask_ex = np.abs(psi_ex) > 0.01 * np.max(np.abs(psi_ex))
    mask_k  = np.abs(psi_k)  > 0.01 * np.max(np.abs(psi_k))
    phase_ex = np.angle(psi_ex)
    phase_k  = np.angle(psi_k)
    # Plot as scattered points to avoid connecting across phase jumps
    ax.scatter(x[mask_ex], phase_ex[mask_ex], s=2, c='k', alpha=0.5, label='arg(ψ) exact')
    ax.scatter(x[mask_k],  phase_k[mask_k],  s=3, c='#d6604d', alpha=0.6, label='arg(ψ̂) KDE')
    ax.set_xlim(-5, 5); ax.set_ylim(-np.pi*1.1, np.pi*1.1)
    ax.set_xlabel('x'); ax.set_ylabel('phase (rad)')
    ax.set_title('Phase: exact vs KDE (where |ψ| > 1% max)', fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle(f'Node Diagnostics at Collision Time — M5ψ-KDE (σ={sigma})',
                 fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Figure 4: v(x) and Q(x) field comparison
# ═══════════════════════════════════════════════════════════════════

def fig_fields(rho_ref, psi_ref, ts_ref, m5, m5p, m5k, sigma):
    """Compare velocity and quantum potential fields at t_c."""
    x = gp.x; dx = gp.dx; hbar = qp.hbar; m = qp.m
    t_c = 4.0 / 3.0
    i_ref = np.argmin(np.abs(ts_ref - t_c))
    psi_ex = psi_ref[i_ref]
    rho_ex = rho_ref[i_ref]

    # Exact fields from ψ_ref
    dpsi_ex  = Dx(psi_ex, dx)
    v_exact  = (hbar/m) * np.imag(np.conj(psi_ex)*dpsi_ex) / np.maximum(rho_ex, 1e-30)
    sqr_ex   = np.sqrt(np.maximum(rho_ex, 1e-30))
    Q_exact  = -(hbar**2/(2*m)) * D2x(sqr_ex, dx) / np.maximum(sqr_ex, 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    colors = {'exact': 'k', 'm5': '#2166ac', 'm5p': '#7570b3', 'm5k': '#d6604d'}
    labels = {'exact': 'Exact', 'm5': 'M5 original',
              'm5p': 'M5ψ-proto', 'm5k': f'M5ψ-KDE (σ={sigma})'}

    # ── Current velocity v(x) ─────────────────────────────────────
    ax = axes[0]
    mask = rho_ex > 0.001 * np.max(rho_ex)
    ax.plot(x[mask], v_exact[mask], 'k-', lw=2.0, label=labels['exact'])

    for key, res, ls, lw in [('m5', m5, '--', 1.3),
                              ('m5p', m5p, ':', 1.5),
                              ('m5k', m5k, '-', 1.5)]:
        i_m = np.argmin(np.abs(res['t_save'] - t_c))
        psi_m = res['psi'][i_m]
        rho_m = np.abs(psi_m)**2
        dpsi_m = Dx(psi_m, dx)
        v_m = (hbar/m) * np.imag(np.conj(psi_m)*dpsi_m) / np.maximum(rho_m, 1e-30)
        mask_m = rho_m > 0.001 * np.max(rho_m)
        ax.plot(x[mask_m], v_m[mask_m], ls, color=colors[key], lw=lw, label=labels[key])

    ax.set_xlim(-5, 5); ax.set_ylim(-8, 8)
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('v(x)', fontsize=11)
    ax.set_title(f'Current velocity at t={t_c:.2f}', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Quantum potential Q(x) ────────────────────────────────────
    ax = axes[1]
    Q_smooth = gaussian_filter1d(Q_exact, sigma=1.5)
    ax.plot(x[mask], Q_smooth[mask], 'k-', lw=2.0, label=labels['exact'])

    for key, res, ls, lw in [('m5', m5, '--', 1.3),
                              ('m5p', m5p, ':', 1.5),
                              ('m5k', m5k, '-', 1.5)]:
        i_m = np.argmin(np.abs(res['t_save'] - t_c))
        psi_m = res['psi'][i_m]
        sqr_m = np.abs(psi_m)
        Q_m = -(hbar**2/(2*m)) * D2x(sqr_m, dx) / np.maximum(sqr_m, 1e-30)
        Q_m = gaussian_filter1d(Q_m, sigma=1.5)
        rho_m = np.abs(psi_m)**2
        mask_m = rho_m > 0.001 * np.max(rho_m)
        ax.plot(x[mask_m], Q_m[mask_m], ls, color=colors[key], lw=lw, label=labels[key])

    ax.set_xlim(-5, 5); ax.set_ylim(-10, 20)
    ax.set_xlabel('x', fontsize=11); ax.set_ylabel('Q(x)', fontsize=11)
    ax.set_title(f'Quantum potential at t={t_c:.2f}', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Field Comparison at Collision Time (Np={len(m5k["X"][0])}, K=48)',
                 fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  M5ψ-KDE: Unified ψ-Space Kernel Density Estimation             ║
║  Cat State Collision Test                                        ║
╚══════════════════════════════════════════════════════════════════╝
""")

    Np = 4000; K = 32
    sigma_kde = 2.5    # default KDE bandwidth (grid cells)
    se = 10            # save every N steps
    x  = gp.x
    psi0 = cat_state(x)

    # ── Schrödinger reference ────────────────────────────────────
    print("  [FFT]  Schrödinger reference… ", end="", flush=True)
    t0 = time.time()
    rho_ref, psi_ref, ts_ref = schrodinger_ref(psi0, save_every=se)
    print(f"done ({time.time()-t0:.1f}s)")

    # ── M5 original ──────────────────────────────────────────────
    print(f"  [M5]   original (σ_ρ=4, σ_S=3, Np={Np}, K={K})… ", end="", flush=True)
    t0 = time.time()
    m5 = m5_original(psi0, Np=Np, K=K, save_every=se)
    t_m5 = time.time() - t0
    err_m5, _ = mean_l2(m5, rho_ref, ts_ref)
    print(f"done ({t_m5:.1f}s, mean L²={err_m5:.5f})")

    # ── M5ψ prototype ────────────────────────────────────────────
    print(f"  [M5ψp] prototype (σ=2.5, Np={Np}, K={K})… ", end="", flush=True)
    t0 = time.time()
    m5p = m5psi_prototype(psi0, Np=Np, K=K, save_every=se, sigma_psi=2.5)
    t_m5p = time.time() - t0
    err_m5p, _ = mean_l2(m5p, rho_ref, ts_ref)
    print(f"done ({t_m5p:.1f}s, mean L²={err_m5p:.5f})")

    # ── M5ψ-KDE ─────────────────────────────────────────────────
    print(f"  [KDE]  M5ψ-KDE (σ={sigma_kde}, Np={Np}, K={K})… ", end="", flush=True)
    t0 = time.time()
    m5k = m5psi_kde(psi0, Np=Np, K=K, save_every=se, sigma=sigma_kde)
    t_kde = time.time() - t0
    err_kde, _ = mean_l2(m5k, rho_ref, ts_ref)
    print(f"done ({t_kde:.1f}s, mean L²={err_kde:.5f})")

    # ── Figure 1: Three-way comparison ───────────────────────────
    print("\n  Figure 1: 3-way comparison…", end="", flush=True)
    fig1 = fig_comparison(rho_ref, psi_ref, ts_ref, m5, m5p, m5k, Np, K, sigma_kde)
    fig1.savefig(output_path("fig1_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(" done")

    # ── Figure 3: Node diagnostics ───────────────────────────────
    print("  Figure 3: node diagnostics…", end="", flush=True)
    fig3 = fig_node_diagnostics(rho_ref, psi_ref, ts_ref, m5k, sigma_kde)
    fig3.savefig(output_path("fig3_node_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(" done")

    # ── Figure 4: Field comparison ───────────────────────────────
    print("  Figure 4: field comparison…", end="", flush=True)
    fig4 = fig_fields(rho_ref, psi_ref, ts_ref, m5, m5p, m5k, sigma_kde)
    fig4.savefig(output_path("fig4_fields.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(" done")

    # ── σ sweep ──────────────────────────────────────────────────
    sigma_list = [1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"\n  [σ sweep] testing σ ∈ {sigma_list}…")
    results_sweep = []
    for sig in sigma_list:
        print(f"    σ={sig}… ", end="", flush=True)
        t0 = time.time()
        res = m5psi_kde(psi0, Np=Np, K=K, save_every=se, sigma=sig)
        results_sweep.append(res)
        me, _ = mean_l2(res, rho_ref, ts_ref)
        print(f"({time.time()-t0:.1f}s, L²={me:.5f})")

    print("  Figure 2: σ sweep…", end="", flush=True)
    fig2, sig_opt, err_opt = fig_sigma_sweep(
        rho_ref, ts_ref, sigma_list, results_sweep, err_m5, err_m5p, Np, K)
    fig2.savefig(output_path("fig2_sigma_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f" done  →  optimal σ = {sig_opt}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Results Summary                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Method          │ Bandwidth(s)     │ Mean L²  │ Time            ║
║  ─────────────── │ ──────────────── │ ──────── │ ─────           ║
║  M5 original     │ σ_ρ=4, σ_S=3    │ {err_m5:.5f} │ {t_m5:5.1f}s           ║
║  M5ψ prototype   │ σ_ψ=2.5         │ {err_m5p:.5f} │ {t_m5p:5.1f}s           ║
║  M5ψ-KDE         │ σ={sigma_kde}            │ {err_kde:.5f} │ {t_kde:5.1f}s           ║
║  M5ψ-KDE optimal │ σ*={sig_opt}            │ {err_opt:.5f} │                 ║
║                                                                  ║
║  Key differences:                                                ║
║  • M5ψ-KDE uses CIC deposit (no bin-averaging, no empty bins)   ║
║  • Single Gaussian convolution (no double-smoothing)             ║
║  • ψ̂ = j_h/√n_h (automatic destructive interference at nodes)   ║
║  • No per-particle amplitude tracking                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
