#!/usr/bin/env python3
"""
Multi-Kernel Swarmalator Comparison
====================================

Compares Gaussian, quintic B-spline, and compact rational kernels
for energy conservation and trajectory quality in the M5 gridless
swarmalator algorithm.

Test cases: HO ground state, cat state collision
Outputs: trajectories, energy drift, density comparison

Based on m5_gridless_opt.py (project knowledge) with the Hermite
convention fix applied.
"""

import numpy as np
import time, sys, os
from m5_utils import output_path
from m5_fft_ref import schrodinger_fft_1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d

# ═══ Physics ═════════════════════════════════════════════════════════
HBAR = 1.0; MASS = 1.0; NU = HBAR / (2 * MASS)

# ═══ Kernel implementations ══════════════════════════════════════════

def kernel_gaussian(dx, h):
    """Gaussian kernel: K, K'. Returns (K, Kp) arrays."""
    inv_h = 1.0 / h
    norm = inv_h / np.sqrt(2 * np.pi)
    u = dx * inv_h
    u2 = u * u
    K = norm * np.exp(-0.5 * u2)
    Kp = -(u * inv_h) * K  # K' = -(dx/h²)K
    return K, Kp

def kernel_quintic_bspline(dx, h):
    """Quintic B-spline (C4, support [-3h, 3h])."""
    q = np.abs(dx) / h
    norm = 1.0 / (120.0 * h)
    K = np.zeros_like(dx)
    Kp = np.zeros_like(dx)
    
    m1 = q < 1; m2 = (q >= 1) & (q < 2); m3 = (q >= 2) & (q < 3)
    q1 = q[m1]; q2 = q[m2]; q3 = q[m3]
    
    K[m1] = norm * ((3-q1)**5 - 6*(2-q1)**5 + 15*(1-q1)**5)
    K[m2] = norm * ((3-q2)**5 - 6*(2-q2)**5)
    K[m3] = norm * ((3-q3)**5)
    
    s = np.sign(dx)
    Kp[m1] = norm/h * (-5*(3-q1)**4 + 30*(2-q1)**4 - 75*(1-q1)**4) * s[m1]
    Kp[m2] = norm/h * (-5*(3-q2)**4 + 30*(2-q2)**4) * s[m2]
    Kp[m3] = norm/h * (-5*(3-q3)**4) * s[m3]
    
    return K, Kp

def kernel_compact_rational(dx, h, n_pow=4):
    """Compact rational kernel: K(u) = C * max(0, 1 - (u/R)²)^n.
    
    Support: [-R, R] where R = 2.5*h (chosen to match Gaussian 2nd moment).
    n_pow=4 gives C6 smoothness.
    K''/K = rational function within support.
    """
    R = 2.5 * h  # support radius
    q2 = (dx / R)**2
    base = np.maximum(1.0 - q2, 0.0)
    
    # Normalization (analytic for (1-u²)^n on [-1,1]):
    # ∫_{-1}^{1} (1-u²)^n du = B(1/2, n+1) = √π Γ(n+1)/Γ(n+3/2)
    from scipy.special import gamma
    integral_unit = np.sqrt(np.pi) * gamma(n_pow + 1) / gamma(n_pow + 1.5)
    norm = 1.0 / (R * integral_unit)
    
    mask = np.abs(dx) < R
    K = np.where(mask, norm * base**n_pow, 0.0)
    # K' = n * (-2*dx/R²) * (1-(dx/R)²)^{n-1} * C
    Kp = np.where(mask & (base > 0),
                  norm * n_pow * (-2*dx/R**2) * base**(n_pow-1), 0.0)
    
    return K, Kp


KERNELS = {
    'gaussian':  (kernel_gaussian, 4.0),      # (func, cutoff in h-units)
    'quintic':   (kernel_quintic_bspline, 3.0),
    'rational':  (kernel_compact_rational, 2.5),  # cutoff in h-units (R=2.5h)
}

# ═══ Kernel sums (generic) ═══════════════════════════════════════════

def kernel_sums(eval_x, X_src, phi_src, h, kernel_name, need_deriv=False):
    """Compute ψ-KDE kernel sums using specified kernel."""
    kfunc, cutoff_h = KERNELS[kernel_name]
    N_eval = len(eval_x)
    N_src = len(X_src)
    
    cos_phi = np.cos(phi_src)
    sin_phi = np.sin(phi_src)
    
    n_out = np.zeros(N_eval)
    j_re_out = np.zeros(N_eval)
    j_im_out = np.zeros(N_eval)
    jp_re_out = np.zeros(N_eval) if need_deriv else None
    jp_im_out = np.zeros(N_eval) if need_deriv else None
    
    radius = cutoff_h * h
    chunk = 1024
    
    for i0 in range(0, N_eval, chunk):
        i1 = min(i0 + chunk, N_eval)
        dx = eval_x[i0:i1, None] - X_src[None, :]  # (chunk, N_src)
        
        mask = np.abs(dx) < radius
        K_vals = np.zeros_like(dx)
        Kp_vals = np.zeros_like(dx)
        
        # Evaluate kernel only where within cutoff
        active = mask.any(axis=0)  # which sources have any active eval points
        for j in range(N_src):
            if not active[j]:
                continue
            row_mask = mask[:, j]
            if not row_mask.any():
                continue
            k_out, kp_out = kfunc(dx[row_mask, j], h)
            K_vals[row_mask, j] = k_out
            if need_deriv:
                Kp_vals[row_mask, j] = kp_out
        
        # Actually, vectorize over full (chunk, N_src) for speed
        K_all, Kp_all = kfunc(dx, h)
        K_all = np.where(mask, K_all, 0.0)
        
        n_out[i0:i1] = K_all.sum(axis=1)
        j_re_out[i0:i1] = (K_all * cos_phi[None, :]).sum(axis=1)
        j_im_out[i0:i1] = (K_all * sin_phi[None, :]).sum(axis=1)
        
        if need_deriv:
            Kp_all = np.where(mask, Kp_all, 0.0)
            jp_re_out[i0:i1] = (Kp_all * cos_phi[None, :]).sum(axis=1)
            jp_im_out[i0:i1] = (Kp_all * sin_phi[None, :]).sum(axis=1)
    
    result = dict(n=n_out, j_re=j_re_out, j_im=j_im_out)
    if need_deriv:
        result['jp_re'] = jp_re_out
        result['jp_im'] = jp_im_out
    return result


def psi_kde_fields(ks, eps_n=1e-30):
    """From kernel sums, compute √ρ, ln ρ, v, u."""
    n = np.maximum(ks['n'], eps_n)
    j_abs = np.sqrt(ks['j_re']**2 + ks['j_im']**2)
    sqrt_rho = j_abs / np.sqrt(n)
    ln_j = np.log(np.maximum(j_abs, eps_n))
    ln_n = np.log(n)
    ln_rho = 2.0 * ln_j - ln_n
    
    result = dict(sqrt_rho=sqrt_rho, ln_rho=ln_rho, n=n, j_abs=j_abs)
    
    if 'jp_re' in ks:
        j2 = ks['j_re']**2 + ks['j_im']**2
        eps_coh = 1e-6
        coherent = j2 > eps_coh * n * n
        j2_safe = np.where(coherent, j2, 1.0)
        
        im_ratio = (ks['jp_im']*ks['j_re'] - ks['jp_re']*ks['j_im']) / j2_safe
        result['v'] = np.where(coherent, (HBAR/MASS) * im_ratio, 0.0)
    
    return result


# ═══ GH quadrature ═══════════════════════════════════════════════════

def gh_nodes_weights(K_gh):
    """Physicist's Hermite GH nodes and normalized weights."""
    from numpy.polynomial.hermite import hermgauss
    xi, omega = hermgauss(K_gh)
    omega /= omega.sum()
    return xi, omega


# ═══ FFT reference ═══════════════════════════════════════════════════

def schrodinger_ref(psi0, V, x, T, Nt, save_every):
    """Thin wrapper around m5_fft_ref.schrodinger_fft_1d."""
    return schrodinger_fft_1d(psi0, V, x, T, Nt,
                              hbar=HBAR, mass=MASS, save_every=save_every)


# ═══ Main swarmalator with kernel selection ══════════════════════════

def m5_swarmalator(psi0_func, V_func, xL, xR, Nx, T, Nt,
                   Np=200, K_gh=6, sigma_gh=0.20, h_kde=0.25,
                   kernel_name='gaussian', seed=42, save_every=50,
                   n_track=20):
    """Run M5 gridless swarmalator with specified kernel.
    
    Returns dict with X, S, t_save, E_history, trajectories.
    """
    dt = T / Nt
    h = h_kde
    
    # GH setup
    xi, omega = gh_nodes_weights(K_gh)
    
    # Initialize particles from ψ₀
    x_grid = np.linspace(xL, xR, Nx, endpoint=False)
    dx = (xR - xL) / Nx
    psi0_grid = psi0_func(x_grid)
    rho0 = np.abs(psi0_grid)**2
    rho0 /= rho0.sum() * dx
    cdf = np.cumsum(rho0) * dx; cdf /= cdf[-1]
    
    rng = np.random.default_rng(seed)
    X = np.interp(rng.uniform(size=Np), cdf, x_grid)
    psi_at_X = psi0_func(X)
    S = HBAR * np.angle(psi_at_X)
    
    # V on grid for interpolation
    V_grid = V_func(x_grid)
    
    # Tracking arrays
    idx_save = list(range(0, Nt+1, save_every))
    X_hist = []; S_hist = []; t_hist = []; E_hist = []
    
    # Select particles to track trajectories
    track_idx = np.linspace(0, Np-1, min(n_track, Np), dtype=int)
    traj_X = []
    traj_S = []
    traj_t = []
    
    def save_state(n):
        X_hist.append(X.copy()); S_hist.append(S.copy())
        t_hist.append(n * dt)
    
    def save_traj(n):
        traj_X.append(X[track_idx].copy())
        traj_S.append(S[track_idx].copy())
        traj_t.append(n * dt)
    
    if 0 in idx_save:
        save_state(0)
    save_traj(0)
    
    t0 = time.time()
    
    for n in range(1, Nt+1):
        phi = S / HBAR
        
        # STEP 1: VELOCITY
        ks_vel = kernel_sums(X, X, phi, h, kernel_name, need_deriv=True)
        fv = psi_kde_fields(ks_vel)
        v_at = fv['v']
        
        # STEP 2: CLASSICAL ADVECTION
        X_class = np.clip(X + v_at * dt, xL + h, xR - h)
        
        # STEP 3: GH CANDIDATE CLOUD
        cand_offsets = np.sqrt(2.0) * sigma_gh * xi
        cands = X_class[:, None] + cand_offsets[None, :]
        cands = np.clip(cands, xL + h, xR - h)
        cands_flat = cands.reshape(-1)
        
        all_eval = np.concatenate([cands_flat, X_class])
        
        # STEP 4: ψ-KDE AT ALL POINTS
        ks_all = kernel_sums(all_eval, X, phi, h, kernel_name, need_deriv=False)
        f_all = psi_kde_fields(ks_all)
        
        sqr_cands = f_all['sqrt_rho'][:Np*K_gh].reshape(Np, K_gh)
        lnr_cands = f_all['ln_rho'][:Np*K_gh].reshape(Np, K_gh)
        sqr_depart = f_all['sqrt_rho'][Np*K_gh:]
        lnr_depart = f_all['ln_rho'][Np*K_gh:]
        
        # STEP 5: STEER
        w_sel = omega[None, :] * np.maximum(sqr_cands, 1e-30)
        w_sum = w_sel.sum(axis=1, keepdims=True)
        probs = w_sel / np.maximum(w_sum, 1e-30)
        cum = np.cumsum(probs, axis=1)
        u_rand = rng.uniform(size=Np)
        chosen = np.clip((cum < u_rand[:, None]).sum(axis=1), 0, K_gh-1)
        X = cands[np.arange(Np), chosen]
        
        # STEP 6: FORWARD WEIGH (Q)
        M_plus = (omega[None, :] * sqr_cands).sum(axis=1)
        ratio = M_plus / np.maximum(sqr_depart, 1e-30)
        Q_at = -(HBAR**2 / (MASS * sigma_gh**2)) * (ratio - 1.0)
        Q_med = np.median(np.abs(Q_at))
        Q_at = np.clip(Q_at, -50*Q_med - 1, 50*Q_med + 1)
        
        # STEP 7: BACKWARD CHANNEL
        L_k = (omega[None, :] * lnr_cands).sum(axis=1)
        u_prime = NU * 2.0 * (L_k - lnr_depart) / sigma_gh**2
        
        # STEP 8: PHASE UPDATE
        fi = np.clip((X - xL) / dx, 0, Nx - 1.001)
        lo = fi.astype(int); al = fi - lo
        hi = np.minimum(lo + 1, Nx - 1)
        V_at = (1 - al) * V_grid[lo] + al * V_grid[hi]
        
        S = S + (0.5*MASS*v_at**2 - V_at - Q_at) * dt
        TWO_PI_HBAR = 2 * np.pi * HBAR
        S = np.mod(S + 0.5*TWO_PI_HBAR, TWO_PI_HBAR) - 0.5*TWO_PI_HBAR
        
        # ENERGY
        E_k = 0.5*MASS*v_at**2 + V_at + Q_at
        E_mean = np.mean(E_k)
        
        # SAVE
        if n in idx_save:
            save_state(n)
            E_hist.append(E_mean)
        
        # Save trajectory every few steps
        if n % max(1, save_every // 2) == 0 or n == Nt:
            save_traj(n)
        
        if n % max(1, Nt // 5) == 0:
            elapsed = time.time() - t0
            print(f"    [{kernel_name:10s}] step {n:5d}/{Nt}  "
                  f"E={E_mean:+.6f}  ({elapsed:.1f}s)", flush=True)
    
    total_time = time.time() - t0
    
    return dict(
        X=np.array(X_hist), S=np.array(S_hist),
        t_save=np.array(t_hist), E=np.array(E_hist),
        traj_X=np.array(traj_X), traj_S=np.array(traj_S),
        traj_t=np.array(traj_t),
        h_kde=h, kernel=kernel_name, time=total_time,
        track_idx=track_idx
    )


# ═══ Test cases ══════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, s0):
    return ((2*np.pi*s0**2)**(-0.25) *
            np.exp(-(x-x0)**2/(4*s0**2) + 1j*p0*x/HBAR))

# ═══ Plotting ════════════════════════════════════════════════════════

def plot_comparison(results, psi_ref, ts_ref, x_grid, dx,
                    case_label, fname, V_func):
    """6-panel comparison: trajectories, density, energy for each kernel."""
    Nk = len(results)
    fig, axes = plt.subplots(3, Nk, figsize=(5.5*Nk, 13))
    if Nk == 1:
        axes = axes[:, None]
    
    rho_ref = np.abs(psi_ref)**2
    
    colors_kern = {'gaussian': '#2166ac', 'quintic': '#4dac26', 'rational': '#d6604d'}
    
    for col, (kname, m5) in enumerate(results.items()):
        ck = colors_kern.get(kname, 'gray')
        
        # ── Row 0: Trajectories ──────────────────────────────────
        ax = axes[0, col]
        tX = m5['traj_X']; tS = m5['traj_S']; tt = m5['traj_t']
        n_traj = tX.shape[1]
        
        # Phase-colored trajectories
        cmap = plt.cm.twilight
        for i in range(n_traj):
            phases = tS[:, i] / HBAR
            phases_norm = (phases % (2*np.pi)) / (2*np.pi)
            
            pts = np.array([tt, tX[:, i]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, linewidths=1.8, alpha=0.7)
            lc.set_array(phases_norm[:-1])
            ax.add_collection(lc)
        
        ax.set_xlim(tt[0], tt[-1])
        xspan = np.percentile(np.abs(tX), 99) * 1.3
        ax.set_ylim(-xspan, xspan)
        ax.set_xlabel('t'); ax.set_ylabel('x')
        Np = m5['X'].shape[1]
        ax.set_title(f'{kname}\n(Np={Np}, h={m5["h_kde"]:.2f})',
                     fontsize=11, weight='bold', color=ck)
        ax.grid(True, alpha=0.15)
        
        # ── Row 1: Density comparison ────────────────────────────
        ax = axes[1, col]
        snap_frac = [0, 0.33, 0.67, 1.0]
        Ns = len(ts_ref)
        snap_colors = ['#2166ac', '#b2182b', '#4dac26', '#7570b3']
        edges = np.linspace(x_grid[0], x_grid[-1], len(x_grid)+1)
        
        for sf, sc in zip(snap_frac, snap_colors):
            si = min(int(sf * (Ns-1)), Ns-1)
            t_s = ts_ref[si]
            ax.plot(x_grid, rho_ref[si], '-', color=sc, lw=2, alpha=0.5)
            
            i_m5 = np.argmin(np.abs(m5['t_save'] - t_s))
            h_m5, _ = np.histogram(m5['X'][i_m5], bins=edges, density=True)
            rho_m5 = gaussian_filter1d(
                np.interp(x_grid, 0.5*(edges[:-1]+edges[1:]), h_m5), sigma=3)
            ax.plot(x_grid, rho_m5, '--', color=sc, lw=1.5,
                    label=f't={t_s:.2f}')
        
        xactive = x_grid[rho_ref[0] > 0.001*np.max(rho_ref[0])]
        if len(xactive) > 0:
            margin = (xactive[-1] - xactive[0]) * 0.3
            ax.set_xlim(xactive[0] - margin, xactive[-1] + margin)
        ax.set_xlabel('x'); ax.set_ylabel('ρ')
        ax.legend(fontsize=7)
        ax.set_title(f'Density: exact vs {kname}', fontsize=10)
        
        # ── Row 2: Energy ────────────────────────────────────────
        ax = axes[2, col]
        if len(m5['E']) > 0:
            t_e = m5['t_save'][1:len(m5['E'])+1]
            E = m5['E']
            E0 = E[0] if len(E) > 0 else 1.0
            ax.plot(t_e, E, '-', color=ck, lw=1.5)
            ax.axhline(E0, ls=':', color='gray', lw=0.8, alpha=0.5)
            
            E_rms = np.sqrt(np.mean((E - E0)**2))
            E_drift = abs(E[-1] - E[0]) if len(E) > 1 else 0
            ax.set_title(f'Energy: E₀={E0:.4f}, RMS={E_rms:.4f}\n'
                         f'drift={E_drift:.4f} ({100*E_drift/max(abs(E0),1e-10):.1f}%)',
                         fontsize=9)
        
        ax.set_xlabel('t'); ax.set_ylabel('⟨E⟩')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Kernel Comparison — {case_label}', fontsize=14, weight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


# ═══ Main ════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Multi-Kernel Swarmalator Comparison")
    print("  Kernels: Gaussian, Quintic B-spline, Compact Rational")
    print("=" * 65)
    
    kernel_names = ['gaussian', 'quintic', 'rational']
    
    # ══ TEST 1: HO Ground State ════════════════════════════════════
    print("\n" + "─" * 55)
    print("  TEST 1: Harmonic Oscillator Ground State")
    print("─" * 55)
    
    omega = 1.0
    sigma_ho = np.sqrt(HBAR / (2*MASS*omega))
    
    def psi0_ho(x):
        return ((omega*MASS/(np.pi*HBAR))**0.25 *
                np.exp(-omega*MASS*x**2/(2*HBAR)))
    V_ho = lambda x: 0.5*MASS*omega**2*x**2
    
    # Parameters
    Np = 150; T = 3.0; Nt = 1500
    K_gh = 4; sigma_gh = 0.15; h_kde = 0.25
    xL, xR, Nx = -6, 6, 256
    save_every = 30
    
    # FFT reference
    x_grid = np.linspace(xL, xR, Nx, endpoint=False)
    dx = (xR - xL) / Nx
    psi0_grid = psi0_ho(x_grid)
    norm = np.sqrt(np.sum(np.abs(psi0_grid)**2) * dx)
    psi0_normed = lambda x, _f=psi0_ho, _n=norm: _f(x) / _n
    psi0_grid /= norm
    V_grid = V_ho(x_grid)
    
    print("  FFT reference...", end=" ", flush=True)
    psi_ref, ts_ref = schrodinger_ref(psi0_grid, V_grid, x_grid, T, Nt, save_every)
    print("done")
    
    # Bandwidth for each kernel (tuned for fair comparison)
    h_map = {'gaussian': 0.25, 'quintic': 0.20, 'rational': 0.25}
    
    results_ho = {}
    for kname in kernel_names:
        print(f"\n  Running {kname}...")
        results_ho[kname] = m5_swarmalator(
            psi0_normed, V_ho, xL, xR, Nx, T, Nt,
            Np=Np, K_gh=K_gh, sigma_gh=sigma_gh,
            h_kde=h_map[kname], kernel_name=kname,
            save_every=save_every, n_track=15, seed=42)
    
    plot_comparison(results_ho, psi_ref, ts_ref, x_grid, dx,
                    f'HO Ground State (Np={Np})',
                    output_path('kernel_compare_ho.png'), V_ho)
    
    # ══ TEST 2: Cat State ══════════════════════════════════════════
    print("\n" + "─" * 55)
    print("  TEST 2: Cat State Collision")
    print("─" * 55)
    
    x0c, p0c, s0c = 4.0, 3.0, 0.7
    
    def psi0_cat(x):
        return gaussian_wp(x, -x0c, +p0c, s0c) + gaussian_wp(x, +x0c, -p0c, s0c)
    V_cat = lambda x: np.zeros_like(x)
    
    Np_cat = 300; T_cat = 1.6; Nt_cat = 1200
    xL_c, xR_c, Nx_c = -12, 12, 512
    save_every_cat = 24
    
    x_grid_c = np.linspace(xL_c, xR_c, Nx_c, endpoint=False)
    dx_c = (xR_c - xL_c) / Nx_c
    psi0_grid_c = psi0_cat(x_grid_c)
    norm_c = np.sqrt(np.sum(np.abs(psi0_grid_c)**2) * dx_c)
    psi0_cat_normed = lambda x, _f=psi0_cat, _n=norm_c: _f(x) / _n
    psi0_grid_c /= norm_c
    V_grid_c = V_cat(x_grid_c)
    
    print("  FFT reference...", end=" ", flush=True)
    psi_ref_c, ts_ref_c = schrodinger_ref(
        psi0_grid_c, V_grid_c, x_grid_c, T_cat, Nt_cat, save_every_cat)
    print("done")
    
    h_map_cat = {'gaussian': 0.25, 'quintic': 0.20, 'rational': 0.25}
    
    results_cat = {}
    for kname in kernel_names:
        print(f"\n  Running {kname}...")
        results_cat[kname] = m5_swarmalator(
            psi0_cat_normed, V_cat, xL_c, xR_c, Nx_c, T_cat, Nt_cat,
            Np=Np_cat, K_gh=4, sigma_gh=0.20,
            h_kde=h_map_cat[kname], kernel_name=kname,
            save_every=save_every_cat, n_track=20, seed=42)
    
    plot_comparison(results_cat, psi_ref_c, ts_ref_c, x_grid_c, dx_c,
                    f'Cat State Collision (Np={Np_cat})',
                    output_path('kernel_compare_cat.png'), V_cat)
    
    # ══ Summary ════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    
    for case_name, results in [("HO Ground State", results_ho),
                               ("Cat State", results_cat)]:
        print(f"\n  {case_name}:")
        for kname, m5 in results.items():
            E = m5['E']
            if len(E) > 1:
                E0 = E[0]
                E_rms = np.sqrt(np.mean((E - E0)**2))
                E_drift = abs(E[-1] - E[0])
                print(f"    {kname:12s}: E₀={E0:+.4f}, RMS={E_rms:.4f}, "
                      f"drift={E_drift:.4f} ({100*E_drift/max(abs(E0),1e-10):.1f}%), "
                      f"time={m5['time']:.1f}s")
    
    print(f"\n{'=' * 65}")

if __name__ == "__main__":
    main()
