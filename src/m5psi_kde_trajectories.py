#!/usr/bin/env python3
"""
M5ψ-KDE vs M5-original: Trajectory Comparison
===============================================

Produces trajectory figures comparable to method5_catstate.py, but using
the M5ψ-KDE algorithm (σ=2.0, optimal bandwidth from sweep) alongside
original M5 for direct side-by-side comparison.

Both methods track the same set of particles at every time step, allowing
pixel-for-pixel comparable visualisations of:
  1. Space-time heatmap + trajectories
  2. Zoom on collision region
  3. Density snapshots with node markers
  4. Particle histogram vs exact at collision
  5. Trajectory spaghetti on ρ-contours
  6. Left-right minimum approach distance

The crucial difference being evaluated:
  M5-original : histogram+smooth for ρ,  bin-average+smooth for S
  M5ψ-KDE     : CIC deposit → single Gaussian smooth → ψ̂ = j_h/√n_h

Node handling is the key: M5ψ-KDE's destructive interference in j_h
suppresses √ρ at nodes automatically, whereas M5-original sees positive ρ
from both branches simultaneously and must handle the node through osmotic
pressure alone.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import time, warnings
from m5_utils import output_path
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# Grid / physics constants  (match method5_catstate.py exactly)
# ═══════════════════════════════════════════════════════════════════

HBAR  = 1.0;  M = 1.0
NU    = HBAR / (2 * M)
SIGMA_NOISE = np.sqrt(HBAR / M)

XL = -15.;  XR = 15.;  NX = 512
T_TOTAL = 3.0;  NT = 3000
DX = (XR - XL) / NX
DT = T_TOTAL / NT
X_GRID = np.linspace(XL, XR, NX, endpoint=False)
K_GRID = 2 * np.pi * np.fft.fftfreq(NX, DX)
T_ARR  = np.linspace(0, T_TOTAL, NT + 1)

# Cat-state parameters
X0, P0, S0_WP = 4.0, 3.0, 0.7
T_COLLISION = X0 * M / P0   # ≈ 1.333

# Run parameters
NP      = 8000   # particles
K_CAND  = 48     # candidates per step
N_TRACK = 120    # trajectories to store at every step
SAVE_EVERY = 6   # snapshot frequency (matches method5_catstate.py)
SIGMA_KDE  = 2.0 # optimal bandwidth from sweep


# ═══════════════════════════════════════════════════════════════════
# Basic operators
# ═══════════════════════════════════════════════════════════════════

def Dx(f):
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * DX)

def D2x(f):
    return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / DX**2


# ═══════════════════════════════════════════════════════════════════
# Initial conditions
# ═══════════════════════════════════════════════════════════════════

def cat_state(x):
    def wp(x, x0, p0):
        return (2*np.pi*S0_WP**2)**(-.25) * np.exp(
            -(x - x0)**2 / (4*S0_WP**2) + 1j*p0*x/HBAR)
    psi = wp(x, -X0, +P0) + wp(x, +X0, -P0)
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * DX)


def init_particles(psi0, Np, seed=42):
    rng = np.random.default_rng(seed)
    rho0 = np.abs(psi0)**2
    cdf = np.cumsum(rho0) * DX;  cdf /= cdf[-1]
    X  = np.interp(rng.uniform(size=Np), cdf, X_GRID)
    S0 = HBAR * np.unwrap(np.angle(psi0))
    Sp = np.interp(X, X_GRID, S0)
    return X, Sp


# ═══════════════════════════════════════════════════════════════════
# Schrödinger FFT reference  (V = 0)
# ═══════════════════════════════════════════════════════════════════

def schrodinger_ref(psi0, save_every=SAVE_EVERY):
    half_V = np.ones(NX, dtype=complex)            # V = 0
    T_k    = np.exp(-1j * HBAR * K_GRID**2 * DT / (2*M))
    psi = psi0.copy().astype(complex)

    idx = list(range(0, NT+1, save_every))
    psi_h = np.zeros((len(idx), NX), dtype=complex)
    si = 0
    if 0 in idx: psi_h[si] = psi.copy(); si += 1

    for n in range(1, NT+1):
        psi = np.fft.ifft(T_k * np.fft.fft(psi))
        if n in idx and si < len(idx):
            psi_h[si] = psi.copy(); si += 1

    ts = T_ARR[idx[:len(idx)]]
    return np.abs(psi_h)**2, psi_h, ts


# ═══════════════════════════════════════════════════════════════════
# ψ-KDE field estimator  (the core M5ψ-KDE step)
# ═══════════════════════════════════════════════════════════════════

def psi_kde_estimate(X, Sp, sigma):
    """
    CIC deposit + Gaussian convolution → ψ̂ = j_h / √n_h.
    Returns (psi_grid, sqrt_rho, v_field, Q_field).
    """
    Np = len(X)
    n_raw = np.zeros(NX)
    j_re  = np.zeros(NX)
    j_im  = np.zeros(NX)

    fi    = np.clip((X - XL) / DX, 0.0, NX - 1.001)
    k_L   = fi.astype(np.int64)
    alpha = fi - k_L
    k_R   = np.minimum(k_L + 1, NX - 1)

    cos_phi = np.cos(Sp / HBAR)
    sin_phi = np.sin(Sp / HBAR)

    np.add.at(n_raw, k_L, 1.0 - alpha)
    np.add.at(n_raw, k_R, alpha)
    np.add.at(j_re, k_L, (1.0 - alpha) * cos_phi)
    np.add.at(j_re, k_R, alpha * cos_phi)
    np.add.at(j_im, k_L, (1.0 - alpha) * sin_phi)
    np.add.at(j_im, k_R, alpha * sin_phi)

    n_sm = gaussian_filter1d(n_raw, sigma=sigma) / Np
    j_sm = (gaussian_filter1d(j_re, sigma=sigma) +
            1j * gaussian_filter1d(j_im, sigma=sigma)) / Np

    eps_n = 1e-30
    psi_grid = j_sm / np.sqrt(np.maximum(n_sm, eps_n))
    norm = np.sqrt(np.sum(np.abs(psi_grid)**2) * DX)
    if norm > eps_n:
        psi_grid /= norm

    rho_est  = np.abs(psi_grid)**2
    sqrt_rho = np.abs(psi_grid)

    eps_psi = 1e-10
    dpsi    = Dx(psi_grid)
    v_field = (HBAR/M) * np.imag(np.conj(psi_grid) * dpsi) / np.maximum(rho_est, eps_psi)

    Q_field = -(HBAR**2/(2*M)) * D2x(sqrt_rho) / np.maximum(sqrt_rho, eps_psi)
    Q_field = gaussian_filter1d(Q_field, sigma=1.5)

    return psi_grid, sqrt_rho, v_field, Q_field


# ═══════════════════════════════════════════════════════════════════
# M5-original  (histogram + smooth ρ,  bin-average + smooth S)
# ═══════════════════════════════════════════════════════════════════

def run_m5_original(psi0, track_ids, seed=42,
                    sigma_rho=4.0, sigma_S=3.0):
    rng = np.random.default_rng(seed)
    X, Sp = init_particles(psi0, NP, seed)

    track_ids = np.array(track_ids)
    traj_X = np.zeros((NT + 1, len(track_ids)))
    traj_X[0] = X[track_ids]

    idx_save = list(range(0, NT+1, SAVE_EVERY))
    Ns = len(idx_save)
    X_h = np.zeros((Ns, NP)); rho_h = np.zeros((Ns, NX))
    si = 0
    rho_est = np.abs(psi0)**2;  rho_est /= np.sum(rho_est)*DX

    if 0 in idx_save:
        X_h[si] = X.copy(); rho_h[si] = rho_est.copy(); si += 1

    for n in range(1, NT+1):
        # density: histogram + smooth
        h, _ = np.histogram(X, bins=NX, range=(XL, XR))
        rho_est = gaussian_filter1d(h.astype(float), sigma=sigma_rho)
        rho_est = np.maximum(rho_est, 1e-30)
        rho_est /= np.sum(rho_est) * DX
        sqrt_rho = np.sqrt(rho_est)

        # phase: bin-average + smooth
        S_field = np.zeros(NX); cnt = np.zeros(NX)
        bi = np.clip(((X - XL) / DX).astype(int), 0, NX-1)
        np.add.at(S_field, bi, Sp); np.add.at(cnt, bi, 1.0)
        ok = cnt > 0
        S_field[ok] /= cnt[ok]
        if not np.all(ok):
            S_field[~ok] = np.interp(X_GRID[~ok], X_GRID[ok], S_field[ok])
        S_field = gaussian_filter1d(S_field, sigma=sigma_S)
        v_field = np.nan_to_num(Dx(S_field) / M, nan=0.0)

        v_at    = np.interp(X, X_GRID, v_field)
        X_class = X + v_at * DT

        # √ρ candidate selection
        noise = rng.normal(size=(NP, K_CAND)) * SIGMA_NOISE * np.sqrt(DT)
        cands = np.clip(X_class[:, None] + noise, XL+DX, XR-DX)
        fi = np.clip((cands - XL) / DX, 0.0, NX-1.001)
        lo = np.clip(fi.astype(np.int64), 0, NX-2); hi = lo + 1
        al = np.clip(fi - lo, 0.0, 1.0)
        w  = np.maximum((1-al)*sqrt_rho[lo] + al*sqrt_rho[hi], 1e-30)
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        ck  = np.clip((cum < rng.uniform(size=NP)[:, None]).sum(axis=1), 0, K_CAND-1)
        X   = cands[np.arange(NP), ck]

        # action update — guard against extreme Q near nodes
        sqr  = np.sqrt(np.maximum(rho_est, 1e-30))
        Q_f  = -(HBAR**2/(2*M)) * D2x(sqr) / np.maximum(sqr, 1e-8)
        Q_f  = gaussian_filter1d(Q_f, sigma=2.0)
        Q_f  = np.clip(Q_f, -50.0, 50.0)
        Sp  += (0.5*M*v_at**2 - np.interp(X, X_GRID, Q_f)) * DT
        Sp   = np.nan_to_num(Sp, nan=0.0, posinf=0.0, neginf=0.0)

        traj_X[n] = X[track_ids]

        if n in idx_save and si < Ns:
            X_h[si] = X.copy(); rho_h[si] = rho_est.copy(); si += 1

    return dict(X=X_h, rho=rho_h,
                traj_X=traj_X, traj_t=T_ARR,
                t_save=T_ARR[idx_save[:Ns]])


# ═══════════════════════════════════════════════════════════════════
# M5ψ-KDE  (CIC deposit + single Gaussian + j/√n)
# ═══════════════════════════════════════════════════════════════════

def run_m5psi_kde(psi0, track_ids, seed=42, sigma=SIGMA_KDE):
    rng = np.random.default_rng(seed)
    X, Sp = init_particles(psi0, NP, seed)

    track_ids = np.array(track_ids)
    traj_X = np.zeros((NT + 1, len(track_ids)))
    traj_X[0] = X[track_ids]

    idx_save = list(range(0, NT+1, SAVE_EVERY))
    Ns = len(idx_save)
    X_h = np.zeros((Ns, NP)); rho_h = np.zeros((Ns, NX))
    si = 0
    rho_est = np.abs(psi0)**2;  rho_est /= np.sum(rho_est)*DX

    if 0 in idx_save:
        X_h[si] = X.copy(); rho_h[si] = rho_est.copy(); si += 1

    for n in range(1, NT+1):
        # ψ-KDE field estimation
        _, sqrt_rho, v_field, Q_field = psi_kde_estimate(X, Sp, sigma)

        v_at    = np.interp(X, X_GRID, v_field)
        X_class = X + v_at * DT

        # √ρ candidate selection  (√ρ = |ψ̂| from ψ-KDE)
        noise = rng.normal(size=(NP, K_CAND)) * SIGMA_NOISE * np.sqrt(DT)
        cands = np.clip(X_class[:, None] + noise, XL+DX, XR-DX)
        fi = np.clip((cands - XL) / DX, 0.0, NX-1.001)
        lo = np.clip(fi.astype(np.int64), 0, NX-2); hi = lo + 1
        al = np.clip(fi - lo, 0.0, 1.0)
        w  = np.maximum((1-al)*sqrt_rho[lo] + al*sqrt_rho[hi], 1e-30)
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        ck  = np.clip((cum < rng.uniform(size=NP)[:, None]).sum(axis=1), 0, K_CAND-1)
        X   = cands[np.arange(NP), ck]

        # action update
        Sp += (0.5*M*v_at**2 - np.interp(X, X_GRID, Q_field)) * DT
        Sp  = np.remainder(Sp + np.pi*HBAR, 2*np.pi*HBAR) - np.pi*HBAR

        traj_X[n] = X[track_ids]
        rho_est = sqrt_rho**2

        if n in idx_save and si < Ns:
            X_h[si] = X.copy(); rho_h[si] = rho_est.copy(); si += 1

    return dict(X=X_h, rho=rho_h,
                traj_X=traj_X, traj_t=T_ARR,
                t_save=T_ARR[idx_save[:Ns]])


# ═══════════════════════════════════════════════════════════════════
# Shared figure builders
# ═══════════════════════════════════════════════════════════════════

def build_trajectory_figure(rho_ref, psi_ref, ts_ref, result,
                             method_label, color_left, color_right,
                             Np, K):
    """
    Produce the 6-panel trajectory figure (mirrors method5_catstate.py).

    Panels:
      (a) Space-time heatmap + trajectories          [row 0, cols 0-1]
      (b) Zoom: collision region                     [row 0, col  2  ]
      (c) Density snapshots at key times + nodes     [row 1, cols 0-1]
      (d) Particle histogram vs exact at collision   [row 1, col  2  ]
      (e) Trajectory spaghetti on ρ-contours         [row 2, cols 0-1]
      (f) Minimum left-right approach distance       [row 2, col  2  ]
    """
    x       = X_GRID
    traj_X  = result['traj_X']   # (NT+1, N_track)
    traj_t  = result['traj_t']   # (NT+1,)
    N_track = traj_X.shape[1]
    t_thin  = max(1, len(traj_t) // 600)

    # Colour by initial side
    x_init    = traj_X[0, :]
    left_ids  = np.where(x_init < 0)[0]
    right_ids = np.where(x_init >= 0)[0]
    n_show    = min(N_track, 60)
    n_L = min(n_show // 2, len(left_ids))
    n_R = min(n_show // 2, len(right_ids))
    show_left  = left_ids [np.linspace(0, len(left_ids) -1, n_L, dtype=int)]
    show_right = right_ids[np.linspace(0, len(right_ids)-1, n_R, dtype=int)]

    rho_img = rho_ref.copy()   # (Ns, Nx)

    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.33, wspace=0.30)

    # ── (a) Full space-time heatmap ──────────────────────────────
    ax_st = fig.add_subplot(gs[0, :2])
    ax_st.imshow(rho_img.T, origin='lower', aspect='auto',
                 extent=[0, T_TOTAL, XL, XR],
                 cmap='inferno', interpolation='bilinear',
                 vmin=0, vmax=np.percentile(rho_img, 99))
    for idx in show_left:
        ax_st.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                   '-', color=color_left,  lw=0.4, alpha=0.7)
    for idx in show_right:
        ax_st.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                   '-', color=color_right, lw=0.4, alpha=0.7)
    ax_st.axvline(T_COLLISION, color='#00ff88', ls='--', lw=1.0, alpha=0.8,
                  label=f'tᶜ ≈ {T_COLLISION:.2f}')
    ax_st.set_xlabel('t', fontsize=11); ax_st.set_ylabel('x', fontsize=11)
    ax_st.set_ylim(-8, 8)
    ax_st.set_title(f'ρ(x,t) with {method_label} trajectories  '
                    f'(blue = left packet, orange = right packet)', fontsize=11)
    ax_st.legend(fontsize=9, loc='upper right')

    # ── (b) Zoom on collision ────────────────────────────────────
    ax_zoom = fig.add_subplot(gs[0, 2])
    ax_zoom.imshow(rho_img.T, origin='lower', aspect='auto',
                   extent=[0, T_TOTAL, XL, XR],
                   cmap='inferno', interpolation='bilinear',
                   vmin=0, vmax=np.percentile(rho_img, 99))
    for idx in show_left:
        ax_zoom.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color=color_left,  lw=0.6, alpha=0.8)
    for idx in show_right:
        ax_zoom.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color=color_right, lw=0.6, alpha=0.8)
    ax_zoom.set_xlim(max(0, T_COLLISION - 0.8), min(T_TOTAL, T_COLLISION + 0.8))
    ax_zoom.set_ylim(-4.0, 4.0)
    ax_zoom.set_xlabel('t', fontsize=11); ax_zoom.set_ylabel('x', fontsize=11)
    ax_zoom.set_title('Zoom: collision region\n(trajectory non-crossing)', fontsize=11)

    # ── (c) Density snapshots + node markers ────────────────────
    snap_times = [0, T_COLLISION*0.5, T_COLLISION*0.85,
                  T_COLLISION, T_COLLISION*1.15, T_COLLISION*1.5]
    snap_times = [t for t in snap_times if t <= T_TOTAL]
    ax_snap = fig.add_subplot(gs[1, :2])
    cmap_s  = plt.cm.viridis
    col_s   = [cmap_s(i / max(len(snap_times)-1, 1)) for i in range(len(snap_times))]
    for i_s, t_s in enumerate(snap_times):
        i_fft = np.argmin(np.abs(ts_ref - t_s))
        rho_s = rho_ref[i_fft]
        ax_snap.plot(x, rho_s, '-', color=col_s[i_s], lw=1.8,
                     label=f't = {ts_ref[i_fft]:.2f}')
        if ts_ref[i_fft] > T_COLLISION * 0.6:
            rs = gaussian_filter1d(rho_s, sigma=2)
            thr = 0.005 * np.max(rs)
            lm  = np.where((rs[1:-1] < rs[:-2]) & (rs[1:-1] < rs[2:]) &
                           (rs[1:-1] < thr))[0] + 1
            xn  = x[lm]; xn = xn[(xn > -6) & (xn < 6)]
            if len(xn):
                ax_snap.plot(xn, np.zeros_like(xn), 'v',
                             color=col_s[i_s], ms=6, alpha=0.7)
    ax_snap.set_xlim(-7, 7)
    ax_snap.set_xlabel('x', fontsize=11); ax_snap.set_ylabel('ρ(x,t)', fontsize=11)
    ax_snap.set_title('Density evolution: interference fringes & node formation (▼ = nodes)',
                       fontsize=11)
    ax_snap.legend(fontsize=8, ncol=3, loc='upper right')

    # ── (d) Particle histogram vs exact at collision ─────────────
    ax_coll = fig.add_subplot(gs[1, 2])
    i_coll_ref = np.argmin(np.abs(ts_ref - T_COLLISION))
    i_coll_m   = np.argmin(np.abs(result['t_save'] - T_COLLISION))
    edges   = np.linspace(XL, XR, NX+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_coll.plot(x, rho_ref[i_coll_ref], 'k-', lw=2.5, label='Exact (FFT)')
    h_m, _ = np.histogram(result['X'][i_coll_m], bins=edges, density=True)
    rho_m  = gaussian_filter1d(np.interp(x, centers, h_m), sigma=3.5)
    ax_coll.plot(x, rho_m, '-', color='#d6604d', lw=1.8,
                 label=f'{method_label} (Np={Np}, K={K})')
    ax_coll.fill_between(x, rho_ref[i_coll_ref], alpha=0.12, color='gray')
    ax_coll.set_xlim(-6, 6)
    ax_coll.set_xlabel('x', fontsize=11); ax_coll.set_ylabel('ρ(x,t)', fontsize=11)
    ax_coll.set_title(f'Density at collision  t ≈ {ts_ref[i_coll_ref]:.2f}', fontsize=11)
    ax_coll.legend(fontsize=9)

    # ── (e) Spaghetti on ρ-contours ─────────────────────────────
    ax_traj = fig.add_subplot(gs[2, :2])
    T_grid, Xg = np.meshgrid(ts_ref, x)
    ax_traj.contourf(T_grid, Xg, rho_img.T, levels=30, cmap='Greys', alpha=0.35)
    n_detail = min(40, N_track)
    dl = show_left [:min(n_detail//2, len(show_left))]
    dr = show_right[:min(n_detail//2, len(show_right))]
    for idx in dl:
        ax_traj.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color=color_left,  lw=0.7, alpha=0.75)
    for idx in dr:
        ax_traj.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color=color_right, lw=0.7, alpha=0.75)
    ax_traj.axvline(T_COLLISION, color='#00aa44', ls='--', lw=1.2, alpha=0.7)
    ax_traj.set_ylim(-7, 7)
    ax_traj.set_xlabel('t', fontsize=11); ax_traj.set_ylabel('x', fontsize=11)
    ax_traj.set_title('Sample trajectories (blue = left, red = right) '
                       'over ρ(x,t) contours — non-crossing enforced by nodes',
                       fontsize=11)

    # ── (f) Left-right minimum approach distance ─────────────────
    ax_node = fig.add_subplot(gs[2, 2])
    ls_ = np.argsort(traj_X[0, show_left]);  dl_sorted = show_left [ls_]
    rs_ = np.argsort(traj_X[0, show_right]); dr_sorted = show_right[rs_]
    if len(dl_sorted) > 0 and len(dr_sorted) > 0:
        n_pairs = min(5, len(dl_sorted), len(dr_sorted))
        pair_d  = np.zeros((len(traj_t), n_pairs))
        for ip in range(n_pairs):
            il = dl_sorted[-(ip+1)]
            ir = dr_sorted[ip]
            pair_d[:, ip] = np.abs(traj_X[:, il] - traj_X[:, ir])
        for ip in range(n_pairs):
            ax_node.semilogy(traj_t[::t_thin*2], pair_d[::t_thin*2, ip],
                             '-', lw=1.0, alpha=0.7,
                             label=f'pair {ip+1}' if ip < 3 else None)
        ax_node.axvline(T_COLLISION, color='#00aa44', ls='--', lw=1.0,
                        alpha=0.7, label=f'tᶜ ≈ {T_COLLISION:.2f}')
    ax_node.set_xlabel('t', fontsize=11)
    ax_node.set_ylabel('|x_L − x_R|', fontsize=11)
    ax_node.set_title('Minimum approach distance\n(left-right pairs)', fontsize=11)
    ax_node.legend(fontsize=8); ax_node.grid(True, alpha=0.3)

    fig.suptitle(
        f'Cat State  (x₀=±{X0}, p₀=±{P0}, σ₀={S0_WP})  —  {method_label}  '
        f'(Np={Np}, K={K})',
        fontsize=14, weight='bold', y=0.98)
    return fig


def build_comparison_figure(rho_ref, ts_ref, m5_orig, m5_kde, Np, K,
                             t_m5, t_kde):
    """
    Side-by-side summary: density at collision + L² error trajectories.
    5 columns: t=0, t=t_c/2, t=t_c, t=1.5t_c, t=T  + error panel.
    """
    x = X_GRID
    edges   = np.linspace(XL, XR, NX+1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    snap_ts = [0.0, T_COLLISION/2, T_COLLISION, 1.5*T_COLLISION, T_TOTAL]
    snap_ts = [t for t in snap_ts if t <= T_TOTAL]

    fig, axes = plt.subplots(2, len(snap_ts)+1,
                             figsize=(5*(len(snap_ts)+1), 10))

    # ── Row 0: density snapshots ─────────────────────────────────
    for col, t_s in enumerate(snap_ts):
        ax = axes[0, col]
        i_ref = np.argmin(np.abs(ts_ref - t_s))
        i_m5  = np.argmin(np.abs(m5_orig['t_save'] - t_s))
        i_kde = np.argmin(np.abs(m5_kde['t_save']  - t_s))

        ax.plot(x, rho_ref[i_ref], 'k-', lw=2.2, label='Exact')

        h5, _ = np.histogram(m5_orig['X'][i_m5], bins=edges, density=True)
        r5    = gaussian_filter1d(np.interp(x, centers, h5), sigma=3.5)
        ax.plot(x, r5, '--', color='#2166ac', lw=1.6, label='M5-orig')

        hk, _ = np.histogram(m5_kde['X'][i_kde], bins=edges, density=True)
        rk    = gaussian_filter1d(np.interp(x, centers, hk), sigma=3.5)
        ax.plot(x, rk, '-', color='#d6604d', lw=1.6, label=f'M5ψ-KDE (σ={SIGMA_KDE})')

        ax.set_title(f't = {ts_ref[i_ref]:.2f}', fontsize=11)
        ax.set_xlim(-7, 7)
        if col == 0:
            ax.legend(fontsize=8); ax.set_ylabel('ρ(x,t)', fontsize=11)

    # ── L² error summary ─────────────────────────────────────────
    ax_e = axes[0, len(snap_ts)]
    err5 = []; errk = []
    for i, t in enumerate(ts_ref):
        i5  = np.argmin(np.abs(m5_orig['t_save'] - t))
        ik  = np.argmin(np.abs(m5_kde['t_save']  - t))
        h5, _ = np.histogram(m5_orig['X'][i5], bins=edges, density=True)
        r5 = gaussian_filter1d(np.interp(x, centers, h5), sigma=3.5)
        hk, _ = np.histogram(m5_kde['X'][ik], bins=edges, density=True)
        rk = gaussian_filter1d(np.interp(x, centers, hk), sigma=3.5)
        err5.append(np.sqrt(np.sum((r5 - rho_ref[i])**2) * DX))
        errk.append(np.sqrt(np.sum((rk - rho_ref[i])**2) * DX))
    err5 = np.array(err5); errk = np.array(errk)
    ax_e.semilogy(ts_ref, err5, '-', color='#2166ac', lw=1.5,
                  label=f'M5-orig  (mean={np.mean(err5):.5f})')
    ax_e.semilogy(ts_ref, errk, '-', color='#d6604d', lw=1.5,
                  label=f'M5ψ-KDE  (mean={np.mean(errk):.5f})')
    ax_e.set_xlabel('t', fontsize=11); ax_e.set_ylabel('‖ρ − ρ_ref‖₂', fontsize=11)
    ax_e.set_title('L² error vs time', fontsize=11)
    ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.3)

    # ── Row 1: zoom on interference fringes at t_c ───────────────
    for col in range(len(snap_ts)):
        ax = axes[1, col]
        t_s   = snap_ts[col]
        i_ref = np.argmin(np.abs(ts_ref - t_s))
        i_m5  = np.argmin(np.abs(m5_orig['t_save'] - t_s))
        i_kde = np.argmin(np.abs(m5_kde['t_save']  - t_s))

        ax.plot(x, rho_ref[i_ref], 'k-', lw=2.2)

        h5, _ = np.histogram(m5_orig['X'][i_m5], bins=edges, density=True)
        ax.plot(x, gaussian_filter1d(np.interp(x, centers, h5), sigma=3.5),
                '--', color='#2166ac', lw=1.5)
        hk, _ = np.histogram(m5_kde['X'][i_kde], bins=edges, density=True)
        ax.plot(x, gaussian_filter1d(np.interp(x, centers, hk), sigma=3.5),
                '-', color='#d6604d', lw=1.5)

        ax.set_xlim(-5, 5)
        ax.set_xlabel('x', fontsize=10)
        if col == 0: ax.set_ylabel('ρ(x,t)', fontsize=11)
        ax.set_title(f'Zoom  t={ts_ref[i_ref]:.2f}', fontsize=10)

    # ── Summary text ─────────────────────────────────────────────
    ax_txt = axes[1, len(snap_ts)]
    ax_txt.axis('off')
    summary = (
        f"M5-original:\n"
        f"  σ_ρ=4, σ_S=3 (two separate smoothings)\n"
        f"  mean L² = {np.mean(err5):.5f}\n"
        f"  max  L² = {np.max(err5):.5f}\n"
        f"  time  = {t_m5:.1f} s\n\n"
        f"M5ψ-KDE  (σ={SIGMA_KDE}):\n"
        f"  CIC + single Gaussian, ψ̂ = j/√n\n"
        f"  mean L² = {np.mean(errk):.5f}\n"
        f"  max  L² = {np.max(errk):.5f}\n"
        f"  time  = {t_kde:.1f} s\n\n"
        f"Ratio mean err (KDE/orig) = {np.mean(errk)/max(np.mean(err5),1e-12):.3f}\n"
        f"Ratio time     (KDE/orig) = {t_kde/max(t_m5,1e-9):.2f}×"
    )
    ax_txt.text(0.04, 0.96, summary, transform=ax_txt.transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle(
        f'Cat State Comparison — M5-original vs M5ψ-KDE  '
        f'(Np={Np}, K={K}, σ_KDE={SIGMA_KDE})',
        fontsize=14, weight='bold', y=1.00)
    fig.tight_layout()
    return fig, np.mean(err5), np.mean(errk)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  M5ψ-KDE vs M5-original: Cat State Trajectory Comparison        ║
╚══════════════════════════════════════════════════════════════════╝
""")
    psi0 = cat_state(X_GRID)

    # ── FFT reference ────────────────────────────────────────────
    print("  [FFT] Schrödinger reference… ", end="", flush=True)
    t0 = time.time()
    rho_ref, psi_ref, ts_ref = schrodinger_ref(psi0)
    print(f"done ({time.time()-t0:.1f}s)")

    # ── Choose tracked particle indices (same seed for both methods)
    rng0 = np.random.default_rng(42)
    rho0 = np.abs(psi0)**2
    cdf0 = np.cumsum(rho0)*DX; cdf0 /= cdf0[-1]
    X_init_ref = np.interp(rng0.uniform(size=NP), cdf0, X_GRID)
    sorted_ids = np.argsort(X_init_ref)
    track_ids  = sorted_ids[np.linspace(0, NP-1, N_TRACK, dtype=int)]
    print(f"  Tracking {N_TRACK} particles (same IDs for both methods)")

    # ── M5-original ──────────────────────────────────────────────
    print(f"  [M5]   original (σ_ρ=4, σ_S=3, Np={NP}, K={K_CAND}, "
          f"tracking {N_TRACK})… ", end="", flush=True)
    t0 = time.time()
    m5 = run_m5_original(psi0, track_ids)
    t_m5 = time.time() - t0
    print(f"done ({t_m5:.1f}s)")

    # ── M5ψ-KDE ─────────────────────────────────────────────────
    print(f"  [KDE]  M5ψ-KDE (σ={SIGMA_KDE}, Np={NP}, K={K_CAND}, "
          f"tracking {N_TRACK})… ", end="", flush=True)
    t0 = time.time()
    m5k = run_m5psi_kde(psi0, track_ids)
    t_kde = time.time() - t0
    print(f"done ({t_kde:.1f}s)")

    # ── Figure: M5-original trajectories ─────────────────────────
    print("\n  Figure 1: M5-original trajectory figure…", end="", flush=True)
    fig_m5 = build_trajectory_figure(
        rho_ref, psi_ref, ts_ref, m5,
        method_label='M5-original',
        color_left='#66ccff', color_right='#ff9966',
        Np=NP, K=K_CAND)
    fig_m5.savefig(output_path("traj_m5orig.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_m5)
    print(" done")

    # ── Figure: M5ψ-KDE trajectories ────────────────────────────
    print("  Figure 2: M5ψ-KDE trajectory figure…", end="", flush=True)
    fig_kde = build_trajectory_figure(
        rho_ref, psi_ref, ts_ref, m5k,
        method_label=f'M5ψ-KDE (σ={SIGMA_KDE})',
        color_left='#44bbff', color_right='#ff6633',
        Np=NP, K=K_CAND)
    fig_kde.savefig(output_path("traj_m5psi_kde.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_kde)
    print(" done")

    # ── Figure: Side-by-side comparison ─────────────────────────
    print("  Figure 3: comparison figure…", end="", flush=True)
    fig_cmp, mean5, mean_kde = build_comparison_figure(
        rho_ref, ts_ref, m5, m5k, NP, K_CAND, t_m5, t_kde)
    fig_cmp.savefig(output_path("traj_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print(" done")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  Results Summary                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Method       │ Bandwidth         │ Mean L²   │ Time            ║
║  ──────────── │ ─────────────     │ ──────── │ ────             ║
║  M5-original  │ σ_ρ=4, σ_S=3     │ {mean5:.5f} │ {t_m5:5.1f}s          ║
║  M5ψ-KDE      │ σ={SIGMA_KDE} (optimal)   │ {mean_kde:.5f} │ {t_kde:5.1f}s          ║
║                                                                  ║
║  Ratio mean L² (KDE/orig) = {mean_kde/max(mean5,1e-12):.3f}                       ║
║  Ratio time   (KDE/orig) = {t_kde/max(t_m5,1e-9):.2f}×                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
