#!/usr/bin/env python3
"""
M5 Gridless Swarmalator — Optimized numpy/cupy Implementation
===============================================================

Thin command-line driver for the gridless swarmalator algorithm.
All core physics (kernel sums, ψ-KDE fields, GH quadrature,
simulation loop) live in m5_sim.py; this script provides test
cases, plotting, and a CLI entry point.

Usage:
  python m5_gridless_opt.py               # auto-detect GPU
  python m5_gridless_opt.py --cpu         # force CPU
  python m5_gridless_opt.py --gpu         # force GPU (error if unavailable)
  python m5_gridless_opt.py --quick       # reduced params for quick test
  python m5_gridless_opt.py --test cat_state
"""

import sys, time, argparse
import numpy as np

from m5.utils import output_path
from m5.fft_ref import schrodinger_fft_1d
from m5.init import init_ensemble_1d, Ensemble, Units
from m5.sim import (m5_simulate, select_backend,
                    kernel_sums, psi_kde_fields)

import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# Physics constants
# ═══════════════════════════════════════════════════════════════════

HBAR = 1.0; MASS = 1.0


# ═══════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, s0):
    return ((2 * np.pi * s0**2) ** (-0.25) *
            np.exp(-(x - x0)**2 / (4 * s0**2) + 1j * p0 * x / HBAR))


def make_test_cases():
    """Return list of (tag, label, psi0_func, V_func, params_dict)."""
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
        return gaussian_wp(x, -x0c, +p0c, s0c) + gaussian_wp(x, +x0c, -p0c, s0c)
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
                 f"(Np={len(m5['X'][0])}, t={m5['wall_time']:.1f}s)",
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
    fv = psi_kde_fields(ks_v, xp_mod, hbar=HBAR, mass=MASS)
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
        psi_ref, ts_ref = schrodinger_fft_1d(
            psi0_grid, V_grid, x_grid, params['T'], params['Nt'],
            hbar=HBAR, mass=MASS, save_every=params['save_every'])
        print(f"({time.time() - t0:.1f}s)")

        # Create ensemble and run via m5_simulate
        ens = init_ensemble_1d(psi0_grid, x_grid, params['Np'],
                               mass=MASS, hbar=HBAR, seed=42,
                               psi0_func=psi0_func_normed)

        print("  M5 gridless swarmalator...", flush=True)
        m5 = m5_simulate(
            ens, V_func, params['T'], params['Nt'],
            mode='gridless', units=Units(hbar=HBAR), x_grid=x_grid,
            K_gh=params['K_gh'], sigma_gh=params['sigma_gh'],
            h_kde=params['h_kde'], save_every=params['save_every'],
            seed=42, backend=force,
            chunk_size=4096 if xp.__name__ == 'cupy' else 2048)
        print(f"  done ({m5['wall_time']:.1f}s)")

        # Plot
        fname = output_path(f"m5_swarmalator_{tag}.png")
        mean_err = plot_results(
            psi_ref, ts_ref, x_grid, dx, m5, label, fname, xp)
        results[tag] = dict(err=mean_err, time=m5['wall_time'])
        print(f"  L² = {mean_err:.5f},  time = {m5['wall_time']:.1f}s")

    print(f"\n{'=' * 55}")
    print(f"  SUMMARY  ({backend_name})")
    print(f"{'=' * 55}")
    for tag, r in results.items():
        print(f"  {tag:15s}  L²={r['err']:.5f}  time={r['time']:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
