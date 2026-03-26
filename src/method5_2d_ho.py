#!/usr/bin/env python3
"""
Method 5-ND: 2-D Quantum Harmonic Oscillator
=============================================

Demonstrates the generalised √ρ-selection algorithm in the
2-dimensional configuration space of a single quantum particle.

Physical system
---------------
  H = -(ℏ²/2m)(∂²/∂x² + ∂²/∂y²) + ½m(ωx²x² + ωy²y²)

Initial state: displaced Gaussian coherent state with initial momentum kick

  ψ₀(x,y) = ψ_x(x; x₀, px₀, σx) · ψ_y(y; y₀, py₀, σy)

For ωx ≠ ωy the two dimensions oscillate at incommensurate frequencies
(Lissajous orbit), producing interesting non-separable trajectories.

Outputs
-------
  1. m5_2d_comparison.png  — 2-D density snapshots: FFT exact vs M5-ND
  2. m5_2d_marginals.png   — Marginal densities ρ(x,t), ρ(y,t) spacetime maps
  3. m5_2d_trajectories.png — 2-D trajectory scatter at key times + full trails
  4. m5_2d_animation.mp4   — Animated density + live particle motion
"""

import sys, time, shutil, warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from dataclasses import dataclass
from typing import Optional, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# GPU backend (optional)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False

def get_xp(use_gpu: bool):
    if use_gpu and _CUPY_OK:
        return cp
    if use_gpu:
        print("  [WARNING] CuPy not found — using NumPy (CPU).")
    return np

def to_numpy(a, xp): return cp.asnumpy(a) if xp is not np else np.asarray(a)
def to_xp(a, xp):    return cp.asarray(a) if xp is not np else a


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsParams:
    hbar: float = 1.0
    m:    float = 1.0
    omx:  float = 1.0     # x-direction frequency
    omy:  float = 1.3     # y-direction frequency  (irrational ratio → Lissajous)
    @property
    def nu(self):          return self.hbar / (2.0 * self.m)
    @property
    def sigma_noise(self): return float(np.sqrt(self.hbar / self.m))

@dataclass
class GridParams:
    Ng:     int   = 96    # grid points per axis (2D grid: Ng×Ng)
    qL:     float = -7.0
    qR:     float =  7.0
    T:      float = 2 * np.pi     # one full ωx period
    Nt:     int   = 2000
    @property
    def dq(self):  return (self.qR - self.qL) / self.Ng
    @property
    def dt(self):  return self.T / self.Nt
    @property
    def x(self):   return np.linspace(self.qL, self.qR, self.Ng, endpoint=False)
    @property
    def kx(self):  return 2*np.pi * np.fft.fftfreq(self.Ng, self.dq)
    @property
    def t_arr(self): return np.linspace(0, self.T, self.Nt + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Optimised 2-D bilinear interpolation  (vectorised, no Python vertex loop)
# ─────────────────────────────────────────────────────────────────────────────

def _fi(Q_col, qL, dq, Ng):
    """Fractional index along one axis."""
    fi = (Q_col - qL) / dq
    return np.clip(fi, 0.0, Ng - 1.001)

def interp2d(field, Q, gp: GridParams):
    """
    Bilinear interpolation of a 2-D field (Ng×Ng) at particle positions
    Q (Np, 2).  Returns (Np,) array.
    """
    Ng = gp.Ng;  qL = gp.qL;  dq = gp.dq
    fi0 = _fi(Q[:, 0], qL, dq, Ng)
    fi1 = _fi(Q[:, 1], qL, dq, Ng)
    l0  = fi0.astype(np.int64);  a0 = fi0 - l0
    l1  = fi1.astype(np.int64);  a1 = fi1 - l1
    h0  = np.minimum(l0 + 1, Ng - 1)
    h1  = np.minimum(l1 + 1, Ng - 1)
    f   = field  # (Ng, Ng)
    return ((1-a0)*(1-a1)*f[l0, l1]
          + (  a0)*(1-a1)*f[h0, l1]
          + (1-a0)*(  a1)*f[l0, h1]
          + (  a0)*(  a1)*f[h0, h1])


def interp2d_batch(sqrt_rho, cands, gp: GridParams, xp):
    """
    Bilinear interp in xp-namespace (GPU-compatible).
    sqrt_rho : (Ng*Ng,) flat array
    cands    : (N, 2) array of 2-D positions
    Returns  : (N,) weights
    """
    Ng = gp.Ng;  qL = float(gp.qL);  dq = float(gp.dq)
    fi0 = xp.clip((cands[:, 0] - qL) / dq, 0.0, float(Ng) - 1.001)
    fi1 = xp.clip((cands[:, 1] - qL) / dq, 0.0, float(Ng) - 1.001)
    l0  = fi0.astype(xp.int64);  a0 = fi0 - l0
    l1  = fi1.astype(xp.int64);  a1 = fi1 - l1
    h0  = xp.minimum(l0 + 1, Ng - 1)
    h1  = xp.minimum(l1 + 1, Ng - 1)
    return ((1-a0)*(1-a1)*sqrt_rho[l0*Ng + l1]
          + (  a0)*(1-a1)*sqrt_rho[h0*Ng + l1]
          + (1-a0)*(  a1)*sqrt_rho[l0*Ng + h1]
          + (  a0)*(  a1)*sqrt_rho[h0*Ng + h1])


# ─────────────────────────────────────────────────────────────────────────────
# 2-D density / phase / quantum-potential estimation
# ─────────────────────────────────────────────────────────────────────────────

def density2d(Q, gp: GridParams, sigma: float = 4.0):
    """Estimate ρ(x,y) on Ng×Ng grid from particle scatter."""
    edges = [np.linspace(gp.qL, gp.qR, gp.Ng + 1)] * 2
    h, _  = np.histogramdd(Q, bins=edges)
    rho   = gaussian_filter(h.astype(np.float64), sigma=sigma)
    rho   = np.maximum(rho, 1e-30)
    rho  /= rho.sum() * gp.dq**2
    return rho


def phase_gradient2d(Q, S, gp: GridParams, sigma: float = 3.0):
    """
    Bin phase S at particle positions → smooth → compute ∇S.
    Returns v_field (Ng, Ng, 2): velocity field.
    """
    Ng  = gp.Ng;  dq = gp.dq
    edges = [np.linspace(gp.qL, gp.qR, Ng + 1)] * 2
    S_sum, _ = np.histogramdd(Q, bins=edges, weights=S)
    cnt,   _ = np.histogramdd(Q, bins=edges)
    cnt      = np.maximum(cnt, 1.0)
    S_grid   = gaussian_filter(S_sum / cnt, sigma=sigma)
    # Gradient along axis 0 (x) and axis 1 (y)
    dSdx = np.gradient(S_grid, dq, axis=0)
    dSdy = np.gradient(S_grid, dq, axis=1)
    return np.stack([dSdx, dSdy], axis=-1)   # (Ng, Ng, 2)


def quantum_potential2d(rho, gp: GridParams, pp: PhysicsParams, sigma: float = 2.0):
    """Q = -(ℏ²/2m) ∇²√ρ / √ρ  in 2-D."""
    dq = gp.dq
    sqr = np.sqrt(np.maximum(rho, 1e-30))
    # 2-D Laplacian via second gradients
    lap = (np.gradient(np.gradient(sqr, dq, axis=0), dq, axis=0)
         + np.gradient(np.gradient(sqr, dq, axis=1), dq, axis=1))
    Q_grid = -(pp.hbar**2 / (2.0*pp.m)) * lap / np.maximum(sqr, 1e-30)
    return gaussian_filter(Q_grid, sigma=sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Method 5  (2-D specialised, fast)
# ─────────────────────────────────────────────────────────────────────────────

def method5_2d(Q0, S0, V_func, pp: PhysicsParams, gp: GridParams,
               K=32, batch_size=2000, save_every=10,
               track_ids=None, use_gpu=False,
               smooth_rho=4.0, smooth_phase=3.0, smooth_Q=2.0,
               seed=42):
    """
    Method 5 √ρ-selection solver specialised for D=2.

    Q0         : (Np, 2)  initial positions
    S0         : (Np,)    initial phases
    V_func     : Q (Np,2) → (Np,)  potential
    track_ids  : list of particle indices for full trajectory storage
    """
    xp   = get_xp(use_gpu)
    rng  = np.random.default_rng(seed)
    Np   = Q0.shape[0]
    Ng   = gp.Ng
    dt   = gp.dt
    dq   = gp.dq
    sig  = pp.sigma_noise
    m    = pp.m
    hbar = pp.hbar

    Q  = Q0.copy().astype(np.float64)
    Sp = S0.copy().astype(np.float64)

    idx_save = list(range(0, gp.Nt + 1, save_every))
    Ns  = len(idx_save)
    Q_h   = np.zeros((Ns, Np, 2),   dtype=np.float32)
    rho_h = np.zeros((Ns, Ng, Ng),  dtype=np.float32)
    si    = 0

    if track_ids is not None:
        track_ids = np.asarray(track_ids, dtype=np.int64)
        traj_Q    = np.zeros((gp.Nt + 1, len(track_ids), 2), dtype=np.float32)
        traj_Q[0] = Q[track_ids].astype(np.float32)
    else:
        traj_Q = None

    rho_est = density2d(Q, gp, smooth_rho)

    if 0 in idx_save:
        Q_h[si]   = Q.astype(np.float32)
        rho_h[si] = rho_est.astype(np.float32)
        si += 1

    for n in range(1, gp.Nt + 1):

        # ── 1. Density ───────────────────────────────────────────────
        rho_est   = density2d(Q, gp, smooth_rho)
        sqrt_rho  = np.sqrt(rho_est)                # (Ng, Ng)

        # ── 2. Phase gradient → v field ──────────────────────────────
        gradS  = phase_gradient2d(Q, Sp, gp, smooth_phase)   # (Ng,Ng,2)
        vx_fld = gradS[:, :, 0] / m
        vy_fld = gradS[:, :, 1] / m

        vx_at  = interp2d(vx_fld, Q, gp)   # (Np,)
        vy_at  = interp2d(vy_fld, Q, gp)

        # ── 3. Classical push ─────────────────────────────────────────
        Qcx = Q[:, 0] + vx_at * dt
        Qcy = Q[:, 1] + vy_at * dt

        # ── 4-6. Batched √ρ-selection ─────────────────────────────────
        sqrt_flat_xp = to_xp(sqrt_rho.ravel(), xp)
        Q_new = np.empty_like(Q)
        n_batches = (Np + batch_size - 1) // batch_size

        for b in range(n_batches):
            sl = slice(b * batch_size, min((b+1)*batch_size, Np))
            Nb = sl.stop - sl.start

            # Generate K Brownian candidates per particle  (Nb, K, 2)
            noise = rng.standard_normal((Nb, K, 2)) * (sig * np.sqrt(dt))
            cx    = Qcx[sl, None] + noise[:, :, 0]  # (Nb, K)
            cy    = Qcy[sl, None] + noise[:, :, 1]

            # Clip to domain
            cx = np.clip(cx, gp.qL + dq, gp.qR - dq)
            cy = np.clip(cy, gp.qL + dq, gp.qR - dq)

            # Evaluate √ρ at each candidate via bilinear interp
            cands_flat = np.stack([cx.ravel(), cy.ravel()], axis=-1)  # (Nb*K, 2)
            cands_xp   = to_xp(cands_flat, xp)
            w_xp       = interp2d_batch(sqrt_flat_xp, cands_xp, gp, xp)  # (Nb*K,)
            w          = to_numpy(w_xp, xp).reshape(Nb, K)
            w          = np.maximum(w, 1e-30)

            # Normalise & categorical draw
            probs = w / w.sum(axis=1, keepdims=True)
            cum   = np.cumsum(probs, axis=1)
            u_draw = rng.uniform(size=(Nb, 1))
            chosen = np.clip((cum < u_draw).sum(axis=1), 0, K - 1)

            ib = np.arange(Nb)
            Q_new[sl, 0] = cx[ib, chosen]
            Q_new[sl, 1] = cy[ib, chosen]

        Q = Q_new

        # ── 7. Quantum potential ──────────────────────────────────────
        Q_grid = quantum_potential2d(rho_est, gp, pp, smooth_Q)
        Q_at   = interp2d(Q_grid, Q, gp)

        # ── 8. Phase update  dS = [½m|v|² - V - Q] dt ────────────────
        vx2 = interp2d(vx_fld, Q, gp)
        vy2 = interp2d(vy_fld, Q, gp)
        vsq = vx2**2 + vy2**2
        V_at = V_func(Q)
        Sp  += (0.5 * m * vsq - V_at - Q_at) * dt

        # ── 9. Store ──────────────────────────────────────────────────
        if traj_Q is not None:
            traj_Q[n] = Q[track_ids].astype(np.float32)

        if n in idx_save and si < Ns:
            Q_h[si]   = Q.astype(np.float32)
            rho_h[si] = rho_est.astype(np.float32)
            si += 1

    for j in range(si, Ns):
        Q_h[j]   = Q.astype(np.float32)
        rho_h[j] = rho_est.astype(np.float32)

    result = dict(Q=Q_h, rho=rho_h, t_save=gp.t_arr[idx_save[:Ns]])
    if traj_Q is not None:
        result['traj_Q'] = traj_Q
        result['traj_t'] = gp.t_arr
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Schrödinger FFT reference (split-operator)
# ─────────────────────────────────────────────────────────────────────────────

def schrodinger_2d(psi0, V_grid, pp: PhysicsParams, gp: GridParams, save_every=10):
    """
    Split-operator propagator for H = T_xy + V(x,y).
    psi0   : (Ng, Ng) complex initial wavefunction on grid
    V_grid : (Ng, Ng) real potential
    """
    hbar, m = pp.hbar, pp.m
    dt, dq  = gp.dt, gp.dq
    kx      = gp.kx
    KX, KY  = np.meshgrid(kx, kx, indexing='ij')   # (Ng, Ng)

    half_V = np.exp(-1j * V_grid * dt / (2 * hbar))
    T_k    = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))
    psi    = psi0.copy().astype(complex)

    idx_save = list(range(0, gp.Nt + 1, save_every))
    Ns   = len(idx_save)
    psi_h = np.zeros((Ns, gp.Ng, gp.Ng), dtype=complex)
    si    = 0
    if 0 in idx_save:
        psi_h[si] = psi.copy(); si += 1

    for n in range(1, gp.Nt + 1):
        psi = half_V * psi
        psi = np.fft.ifft2(T_k * np.fft.fft2(psi))
        psi = half_V * psi
        if n in idx_save and si < Ns:
            psi_h[si] = psi.copy(); si += 1

    for j in range(si, Ns):
        psi_h[j] = psi.copy()
    return psi_h, gp.t_arr[idx_save[:Ns]]


# ─────────────────────────────────────────────────────────────────────────────
# Initial state: displaced coherent state in 2-D
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_wp_2d(X, Y, x0, y0, px0, py0, sx, sy, hbar=1.):
    """
    ψ(x,y) = ψ_x(x; x0,px0,σx) · ψ_y(y; y0,py0,σy)
    """
    psi_x = ((2*np.pi*sx**2)**(-0.25)
             * np.exp(-(X - x0)**2 / (4*sx**2) + 1j*px0*X/hbar))
    psi_y = ((2*np.pi*sy**2)**(-0.25)
             * np.exp(-(Y - y0)**2 / (4*sy**2) + 1j*py0*Y/hbar))
    return psi_x * psi_y


def init_from_psi2d(psi0, gp: GridParams, pp: PhysicsParams, Np: int, seed: int = 42):
    """
    Sample Np positions from |ψ₀|² and assign initial phases.
    Returns Q0 (Np,2) and S0 (Np,).
    """
    rng   = np.random.default_rng(seed)
    x     = gp.x
    dq    = gp.dq
    rho0  = np.abs(psi0)**2
    rho0 /= rho0.sum() * dq**2   # normalise

    # Sample from 2-D density via marginal + conditional
    rho_x = rho0.sum(axis=1) * dq   # marginal in x  (Ng,)
    cdf_x = np.cumsum(rho_x);  cdf_x /= cdf_x[-1]
    i_x   = np.searchsorted(cdf_x, rng.uniform(size=Np))
    i_x   = np.clip(i_x, 0, gp.Ng - 1)
    x_smp = x[i_x] + rng.uniform(-dq/2, dq/2, size=Np)

    # Conditional y | x
    y_smp = np.zeros(Np)
    for j in np.unique(i_x):
        mask = i_x == j
        cond = rho0[j, :]
        s    = cond.sum() * dq
        if s < 1e-20:
            y_smp[mask] = 0.0
            continue
        cdf_y = np.cumsum(cond / cond.sum())
        y_smp[mask] = x[np.searchsorted(cdf_y, rng.uniform(size=mask.sum()))]

    Q0 = np.stack([x_smp, y_smp], axis=1).astype(np.float64)
    Q0 = np.clip(Q0, gp.qL + dq, gp.qR - dq)

    # Phase at each particle from grid
    S_grid = pp.hbar * np.unwrap(np.angle(psi0), axis=0)
    S_grid = np.unwrap(S_grid, axis=1)
    S0 = interp2d(S_grid, Q0, gp)

    return Q0, S0


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_RHO  = 'viridis'
_CMAP_TRAJ = 'plasma'


def _density_panel(ax, rho, gp, title, vmax=None, show_cb=True):
    """Draw a 2-D density heatmap on ax."""
    if vmax is None:
        vmax = np.percentile(rho, 99.5)
    im = ax.imshow(rho.T, origin='lower', aspect='equal',
                   extent=[gp.qL, gp.qR, gp.qL, gp.qR],
                   cmap=_CMAP_RHO, vmin=0, vmax=vmax,
                   interpolation='bilinear')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x', fontsize=9);  ax.set_ylabel('y', fontsize=9)
    if show_cb:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def fig_comparison(x, ts, rho_ref, m5, pp, gp, Np, K):
    """
    3×3 grid comparing exact density vs Method-5-ND snapshots at 6 times,
    plus L² error plot and info panel.
    """
    Ns       = len(ts)
    n_snap   = 6
    snap_idx = np.round(np.linspace(0, Ns-1, n_snap)).astype(int)
    dq       = gp.dq

    vmax = max(np.percentile(rho_ref, 99.8), 1e-10)

    fig, axes = plt.subplots(n_snap, 2, figsize=(10, 3*n_snap))
    for row, si in enumerate(snap_idx):
        t = ts[si]
        rho_ex  = rho_ref[si]
        rho5_raw = m5['rho'][si].astype(np.float64)
        # re-normalise
        rho5 = np.maximum(rho5_raw, 0)
        s5   = rho5.sum() * dq**2
        if s5 > 1e-10:
            rho5 /= s5

        _density_panel(axes[row, 0], rho_ex, gp,
                       f'Exact FFT  t={t:.2f}', vmax=vmax, show_cb=False)
        _density_panel(axes[row, 1], rho5, gp,
                       f'Method 5-ND  t={t:.2f}', vmax=vmax, show_cb=False)
        # Add L² error as text
        err = np.sqrt(np.sum((rho5 - rho_ex)**2) * dq**2)
        axes[row, 1].text(0.03, 0.95, f'L²={err:.4f}',
                          transform=axes[row,1].transAxes,
                          fontsize=8, color='white', va='top')

    fig.suptitle(f'2-D QHO: Exact FFT vs Method 5-ND  (ωx={pp.omx}, ωy={pp.omy},'
                 f'  Np={Np}, K={K})', fontsize=12, weight='bold')
    fig.tight_layout()
    return fig


def fig_marginals(x, ts, rho_ref, m5_rho, pp, gp):
    """
    Space-time maps of marginal densities ρ(x,t) and ρ(y,t)
    for both exact and Method-5-ND.
    """
    dq = gp.dq
    Ns = len(ts)

    # Marginals: integrate over the other axis
    def mx(rho_arr):   # (Ns, Ng, Ng) → (Ns, Ng)
        return rho_arr.sum(axis=2) * dq
    def my(rho_arr):
        return rho_arr.sum(axis=1) * dq

    mx_ref = mx(rho_ref);      my_ref = my(rho_ref)
    mx_m5  = mx(m5_rho);       my_m5  = my(m5_rho)

    # Re-normalise M5 marginals
    for i in range(Ns):
        s = mx_m5[i].sum() * dq
        if s > 1e-10:
            mx_m5[i] /= s
        s = my_m5[i].sum() * dq
        if s > 1e-10:
            my_m5[i] /= s

    vmax_x = max(np.percentile(mx_ref, 99.8), 1e-10)
    vmax_y = max(np.percentile(my_ref, 99.8), 1e-10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    T_g, X_g = np.meshgrid(ts, x)

    def _spacetime(ax, data, title, vmax):
        im = ax.pcolormesh(T_g, X_g, data.T, cmap='inferno',
                           vmin=0, vmax=vmax, shading='auto')
        ax.set_xlabel('t', fontsize=10);  ax.set_ylabel('position', fontsize=10)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _spacetime(axes[0,0], mx_ref, 'Exact  ρ(x,t)  (marginal over y)', vmax_x)
    _spacetime(axes[0,1], mx_m5,  'M5-ND  ρ(x,t)',                    vmax_x)
    _spacetime(axes[1,0], my_ref, 'Exact  ρ(y,t)  (marginal over x)', vmax_y)
    _spacetime(axes[1,1], my_m5,  'M5-ND  ρ(y,t)',                    vmax_y)

    fig.suptitle(f'2-D QHO marginal densities: ωx={pp.omx}, ωy={pp.omy}',
                 fontsize=12, weight='bold')
    fig.tight_layout()
    return fig


def fig_trajectories(m5, pp, gp, Np, K, x0, y0):
    """
    Three-panel trajectory figure:
      Left:  all tracked trajectories overlaid on colour-coded scatter
      Centre: zoomed collision region at collision time
      Right:  full trails of a few selected particles
    """
    traj_Q = m5['traj_Q']          # (Nt+1, N_track, 2)
    traj_t = m5['traj_t']          # (Nt+1,)
    N_track = traj_Q.shape[1]
    Nt_full = len(traj_t) - 1

    # Colour particles by their *initial angle* from the centre
    qx0 = traj_Q[0, :, 0]
    qy0 = traj_Q[0, :, 1]
    angle0 = np.arctan2(qy0 - y0, qx0 - x0)   # (N_track,)

    norm_c = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap_p = plt.cm.hsv

    t_thin = max(1, Nt_full // 800)

    fig = plt.figure(figsize=(18, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.30)

    # ─── Panel 1: full trajectory scatter ─────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(N_track):
        c = cmap_p(norm_c(angle0[i]))
        ax1.plot(traj_Q[::t_thin, i, 0], traj_Q[::t_thin, i, 1],
                 '-', color=c, lw=0.5, alpha=0.45)
    # Scatter at final time
    sc = ax1.scatter(traj_Q[-1, :, 0], traj_Q[-1, :, 1],
                     c=angle0, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                     s=8, zorder=5, alpha=0.8)
    ax1.set_xlim(gp.qL*0.7, gp.qR*0.7)
    ax1.set_ylim(gp.qL*0.7, gp.qR*0.7)
    ax1.set_xlabel('x', fontsize=10);  ax1.set_ylabel('y', fontsize=10)
    ax1.set_title(f'All {N_track} tracked trajectories\n(colour = initial angle)',
                  fontsize=10)
    ax1.set_aspect('equal')
    plt.colorbar(sc, ax=ax1, label='initial angle', fraction=0.046)

    # ─── Panel 2: position scatter at 5 snapshots ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    snap_fracs = [0, 0.25, 0.5, 0.75, 1.0]
    snap_idxs  = [int(f * Nt_full) for f in snap_fracs]
    cmap_time  = plt.cm.cool
    for k_s, si in enumerate(snap_idxs):
        t_val = traj_t[si]
        c_t   = cmap_time(k_s / max(len(snap_idxs)-1, 1))
        ax2.scatter(traj_Q[si, :, 0], traj_Q[si, :, 1],
                    color=c_t, s=5, alpha=0.6,
                    label=f't={t_val:.2f}')
    ax2.set_xlim(gp.qL*0.7, gp.qR*0.7)
    ax2.set_ylim(gp.qL*0.7, gp.qR*0.7)
    ax2.set_xlabel('x', fontsize=10);  ax2.set_ylabel('y', fontsize=10)
    ax2.set_title('Particle positions at 5 snapshots\n(dark → bright = early → late)',
                  fontsize=10)
    ax2.legend(fontsize=8, markerscale=3)
    ax2.set_aspect('equal')

    # ─── Panel 3: a few individual trails ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    n_trail = min(15, N_track)
    trail_ids = np.linspace(0, N_track-1, n_trail, dtype=int)
    lw_traj   = 1.2
    for i in trail_ids:
        c = cmap_p(norm_c(angle0[i]))
        ax3.plot(traj_Q[::t_thin, i, 0],
                 traj_Q[::t_thin, i, 1],
                 '-', color=c, lw=lw_traj, alpha=0.85)
        ax3.plot(*traj_Q[0,  i, :], 'o', color=c, ms=5, zorder=6)
        ax3.plot(*traj_Q[-1, i, :], 's', color=c, ms=5, zorder=6)
    ax3.set_xlim(gp.qL*0.7, gp.qR*0.7)
    ax3.set_ylim(gp.qL*0.7, gp.qR*0.7)
    ax3.set_xlabel('x', fontsize=10);  ax3.set_ylabel('y', fontsize=10)
    ax3.set_title(f'{n_trail} individual trails\n(● start, ■ end)', fontsize=10)
    ax3.set_aspect('equal')

    fig.suptitle(
        f'2-D QHO Trajectories  (ωx={pp.omx}, ωy={pp.omy}  |  '
        f'Np={Np}, K={K})',
        fontsize=12, weight='bold')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

def make_animation(m5, rho_ref, ts_ref, pp, gp, Np, K,
                   n_frames=120, trail_len=20,
                   out_path="/home/claude/m5_2d_animation.mp4"):
    """
    Animated MP4 with four panels:
      TL: Exact density (FFT)         TR: Method-5-ND density
      BL: M5 particles + density bg   BR: Trajectory trails (last trail_len steps)
    """
    traj_Q  = m5['traj_Q']    # (Nt+1, N_track, 2)
    traj_t  = m5['traj_t']    # (Nt+1,)
    N_track = traj_Q.shape[1]
    Nt_full = traj_Q.shape[0] - 1

    # Pick frames evenly from the full trajectory
    frame_idx = np.round(np.linspace(0, Nt_full, n_frames)).astype(int)

    # Map frame trajectory index → nearest saved snapshot index
    t_save   = m5['t_save']                     # (Ns,)
    ts_frame = traj_t[frame_idx]                # time at each animation frame

    def nearest_snap(t):
        return int(np.argmin(np.abs(t_save - t)))

    def nearest_ref(t):
        return int(np.argmin(np.abs(ts_ref - t)))

    # Colour particles by initial angle from density centroid
    qx0     = traj_Q[0, :, 0]
    qy0     = traj_Q[0, :, 1]
    cx0     = qx0.mean();  cy0 = qy0.mean()
    angle0  = np.arctan2(qy0 - cy0, qx0 - cx0)
    norm_c  = Normalize(vmin=-np.pi, vmax=np.pi)
    part_colors = plt.cm.hsv(norm_c(angle0))   # (N_track, 4)

    # Precompute shared vmax for density
    vmax = max(np.percentile(rho_ref, 99.8), 1e-10)

    # Extent for imshow
    ext = [gp.qL, gp.qR, gp.qL, gp.qR]

    fig = plt.figure(figsize=(14, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.05)
    ax_el = fig.add_subplot(gs[0, 0])   # Exact density
    ax_m5 = fig.add_subplot(gs[0, 1])   # M5 density
    ax_pt = fig.add_subplot(gs[1, 0])   # Particles on bg
    ax_tr = fig.add_subplot(gs[1, 1])   # Trajectory trails

    for ax in [ax_el, ax_m5, ax_pt, ax_tr]:
        ax.set_xlim(gp.qL*0.85, gp.qR*0.85)
        ax.set_ylim(gp.qL*0.85, gp.qR*0.85)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

    ax_el.set_xlabel('x', fontsize=9);  ax_el.set_ylabel('y', fontsize=9)
    ax_m5.set_xlabel('x', fontsize=9)
    ax_pt.set_xlabel('x', fontsize=9);  ax_pt.set_ylabel('y', fontsize=9)
    ax_tr.set_xlabel('x', fontsize=9)

    # Initial images
    im_el = ax_el.imshow(rho_ref[0].T, origin='lower', aspect='auto',
                          extent=ext, cmap=_CMAP_RHO, vmin=0, vmax=vmax,
                          interpolation='bilinear')
    im_m5 = ax_m5.imshow(m5['rho'][0].T, origin='lower', aspect='auto',
                          extent=ext, cmap=_CMAP_RHO, vmin=0, vmax=vmax,
                          interpolation='bilinear')
    im_pt = ax_pt.imshow(m5['rho'][0].T, origin='lower', aspect='auto',
                          extent=ext, cmap='Greys', vmin=0, vmax=vmax*0.7,
                          interpolation='bilinear', alpha=0.7)
    # Particle scatter (TL)
    sc_pt = ax_pt.scatter(traj_Q[0, :, 0], traj_Q[0, :, 1],
                           c=part_colors, s=4, alpha=0.6, linewidths=0)

    # Trail lines on ax_tr (one Line2D per tracked particle)
    trail_lines = []
    for i in range(N_track):
        ln, = ax_tr.plot([], [], '-',
                         color=part_colors[i], lw=0.6, alpha=0.65)
        trail_lines.append(ln)

    # Current-position dots on trail panel
    sc_tr = ax_tr.scatter(traj_Q[0, :, 0], traj_Q[0, :, 1],
                           c=part_colors, s=6, alpha=0.85,
                           linewidths=0, zorder=5)

    ax_el.set_title('Exact (FFT)', fontsize=10)
    ax_m5.set_title('Method 5-ND', fontsize=10)
    ax_pt.set_title('M5 particles on density', fontsize=10)
    ax_tr.set_title(f'Particle trails (last {trail_len} steps)', fontsize=10)

    time_text = fig.text(0.5, 0.97, '', ha='center', va='top',
                         fontsize=12, weight='bold')

    def _update(frame_num):
        fi = frame_idx[frame_num]
        t  = float(traj_t[fi])

        si_m5  = nearest_snap(t)
        si_ref = nearest_ref(t)

        # Density panels
        rho_e = rho_ref[si_ref].astype(np.float64)
        rho5  = m5['rho'][si_m5].astype(np.float64)
        # Normalise M5
        s5 = rho5.sum() * gp.dq**2
        if s5 > 1e-10:
            rho5 /= s5

        im_el.set_data(rho_e.T)
        im_m5.set_data(rho5.T)
        im_pt.set_data(rho5.T)

        # Particle positions
        pts = traj_Q[fi, :, :]             # (N_track, 2)
        sc_pt.set_offsets(pts)
        sc_tr.set_offsets(pts)

        # Trails: last trail_len trajectory steps
        t_lo = max(0, fi - trail_len)
        for i, ln in enumerate(trail_lines):
            ln.set_data(traj_Q[t_lo:fi+1, i, 0],
                        traj_Q[t_lo:fi+1, i, 1])

        time_text.set_text(f't = {t:.3f}  (T = {gp.T:.2f})')
        return [im_el, im_m5, im_pt, sc_pt, sc_tr, time_text] + trail_lines

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames,
        interval=80, blit=True)

    writer = animation.FFMpegWriter(fps=15, bitrate=1200,
                                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Animation saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    action='store_true')
    parser.add_argument('--Np',     type=int, default=5000)
    parser.add_argument('--K',      type=int, default=32)
    parser.add_argument('--batch',  type=int, default=2000)
    parser.add_argument('--Ntrack', type=int, default=200)
    parser.add_argument('--frames', type=int, default=120,
                        help='Animation frames')
    parser.add_argument('--trail',  type=int, default=25,
                        help='Trail length in animation (timesteps)')
    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Method 5-ND: 2-D Quantum Harmonic Oscillator                ║
    ║  √ρ-selection in 2-D configuration space                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    USE_GPU  = args.gpu
    Np       = args.Np
    K        = args.K
    BATCH    = args.batch
    N_TRACK  = args.Ntrack

    xp = get_xp(USE_GPU)
    backend = 'CuPy (GPU)' if xp is not np else 'NumPy (CPU)'

    # ── Physics ──────────────────────────────────────────────────────
    # Anisotropic 2-D harmonic oscillator
    # ωy/ωx = 1.3  → incommensurate → Lissajous pattern, no closing orbit
    pp = PhysicsParams(hbar=1., m=1., omx=1.0, omy=1.3)

    # Initial displaced Gaussian coherent state
    # Position offset (x₀, y₀), momentum (px₀, py₀)
    x0, y0   = 2.5, 1.5
    px0, py0 = 0.8, -0.6
    sx, sy   = 0.7, 0.7    # widths

    # Simulation domain
    gp = GridParams(Ng=96, qL=-7.0, qR=7.0, T=2*np.pi, Nt=2000)
    se = 4     # save every N steps → ~500 snapshots

    bar = "═" * 68
    print(f"  Backend  : {backend}")
    print(f"  Physics  : ωx={pp.omx}, ωy={pp.omy}  (Lissajous, ratio={pp.omy/pp.omx:.3f})")
    print(f"  Init     : (x₀,y₀)=({x0},{y0}),  (px₀,py₀)=({px0},{py0}),  σ=({sx},{sy})")
    print(f"  Grid     : Ng={gp.Ng}×{gp.Ng},  T={gp.T:.3f},  Nt={gp.Nt}")
    print(f"  Solver   : Np={Np},  K={K},  batch={BATCH},  N_track={N_TRACK}")

    x  = gp.x
    X, Y = np.meshgrid(x, x, indexing='ij')    # (Ng, Ng)

    # ── Initial wavefunction ─────────────────────────────────────────
    psi0 = gaussian_wp_2d(X, Y, x0, y0, px0, py0, sx, sy, pp.hbar)
    psi0 /= np.sqrt((np.abs(psi0)**2).sum() * gp.dq**2)

    # ── Potential on grid ─────────────────────────────────────────────
    V_grid = 0.5 * pp.m * (pp.omx**2 * X**2 + pp.omy**2 * Y**2)
    V_func = lambda Q: (0.5 * pp.m
                        * (pp.omx**2 * Q[:,0]**2 + pp.omy**2 * Q[:,1]**2))

    # ── Reference: 2-D Schrödinger FFT ──────────────────────────────
    t0 = time.time()
    print(f"\n  [1] 2-D Schrödinger FFT reference … ", end='', flush=True)
    psi_h, ts_ref = schrodinger_2d(psi0, V_grid, pp, gp, save_every=se)
    rho_ref = np.abs(psi_h)**2
    print(f"({time.time()-t0:.1f}s)  [{len(ts_ref)} snapshots]")

    # ── Method 5-ND in D=2 ───────────────────────────────────────────
    Q0, S0 = init_from_psi2d(psi0, gp, pp, Np, seed=42)

    # Tracked particle indices: spread across initial distribution
    rng_ref   = np.random.default_rng(42)
    Q0_ref, _ = init_from_psi2d(psi0, gp, pp, Np, seed=42)
    angles    = np.arctan2(Q0_ref[:,1] - y0, Q0_ref[:,0] - x0)
    sort_a    = np.argsort(angles)
    track_ids = sort_a[np.linspace(0, Np-1, N_TRACK, dtype=int)]

    t0 = time.time()
    print(f"  [5-ND] √ρ-selection D=2 (Np={Np}, K={K}, batch={BATCH},"
          f"  N_track={N_TRACK}) …", flush=True)
    m5 = method5_2d(
        Q0, S0, V_func, pp, gp,
        K=K, batch_size=BATCH, save_every=se,
        track_ids=track_ids, use_gpu=USE_GPU,
        smooth_rho=4.0, smooth_phase=3.0, smooth_Q=2.0,
        seed=42
    )
    m5['wall_time'] = time.time() - t0
    print(f"  → done in {m5['wall_time']:.1f}s")

    # ── Quick L² error ───────────────────────────────────────────────
    Ns  = len(ts_ref)
    dq  = gp.dq
    errs = []
    for i in range(Ns):
        rho5 = m5['rho'][i].astype(np.float64)
        s5   = rho5.sum() * dq**2
        if s5 > 1e-10:
            rho5 /= s5
        rho_e = rho_ref[i]
        errs.append(np.sqrt(np.sum((rho5 - rho_e)**2) * dq**2))
    print(f"\n  L² error vs exact FFT:")
    print(f"    mean = {np.mean(errs):.5f}   max = {np.max(errs):.5f}")

    # ── Static figures ────────────────────────────────────────────────
    print(f"\n  Generating figures …")

    fig1 = fig_comparison(x, ts_ref, rho_ref, m5, pp, gp, Np, K)
    fig1.savefig("/home/claude/m5_2d_comparison.png", dpi=140, bbox_inches='tight')
    plt.close(fig1)
    print("    ✓ m5_2d_comparison.png")

    fig2 = fig_marginals(x, ts_ref, rho_ref, m5['rho'], pp, gp)
    fig2.savefig("/home/claude/m5_2d_marginals.png", dpi=140, bbox_inches='tight')
    plt.close(fig2)
    print("    ✓ m5_2d_marginals.png")

    fig3 = fig_trajectories(m5, pp, gp, Np, K, x0, y0)
    fig3.savefig("/home/claude/m5_2d_trajectories.png", dpi=140, bbox_inches='tight')
    plt.close(fig3)
    print("    ✓ m5_2d_trajectories.png")

    # ── Animation ─────────────────────────────────────────────────────
    print(f"  Generating animation ({args.frames} frames) …")
    anim_path = make_animation(
        m5, rho_ref, ts_ref, pp, gp, Np, K,
        n_frames=args.frames, trail_len=args.trail,
        out_path="/home/claude/m5_2d_animation.mp4"
    )

    # ── Copy to outputs ───────────────────────────────────────────────
    outputs = ["m5_2d_comparison.png", "m5_2d_marginals.png",
               "m5_2d_trajectories.png", "m5_2d_animation.mp4"]
    for f in outputs:
        shutil.copy2(f"/home/claude/{f}", f"/mnt/user-data/outputs/{f}")
        print(f"    ✓ Saved: {f}")

    print(f"\n{bar}")
    print(f"  Done — 2-D QHO simulation complete.")
    print(f"  Wall time: {m5['wall_time']:.1f}s   L² mean: {np.mean(errs):.5f}")
    print(bar)


if __name__ == "__main__":
    main()
