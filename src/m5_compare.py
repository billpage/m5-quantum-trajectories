#!/usr/bin/env python3
"""
m5_compare.py — Grid vs Gridless (Swarmalator) Comparison Demo
===============================================================

Runs five standard quantum-mechanical test problems using both the
grid-based ψ-KDE algorithm and the gridless swarmalator, comparing
each against a split-operator FFT reference solution.

Test cases
----------
  free_gauss    — Translating Gaussian wave packet (V = 0)
  cat_state     — Counter-propagating cat state collision (V = 0)
  ho_ground     — Harmonic oscillator ground state (stationary)
  ho_coherent   — Displaced Gaussian oscillating in HO potential
  eckart        — Tunneling through a symmetric Eckart barrier

For each test case the program produces a 6-panel diagnostic figure:
  (a) Space-time heatmap + grid-mode trajectories
  (b) Space-time heatmap + gridless-mode trajectories
  (c) Density snapshots: exact (FFT) vs grid vs gridless
  (d) L² density error vs time for both methods
  (e) Energy conservation: ⟨Ĥ⟩ vs time for both methods
  (f) Summary metrics (wall time, mean L², energy drift)

Energy is computed via Approach B: reconstruct ψ on the reference
grid from particle snapshots (CIC + Gaussian smooth), then evaluate
⟨Ĥ⟩ = (ℏ²/2m)∫|ψ'|² dx + ∫ρV dx using the FFT kinetic operator.

Usage
-----
  python m5_compare.py                   # auto-detect GPU, all tests
  python m5_compare.py --cpu             # force CPU
  python m5_compare.py --quick           # reduced parameters
  python m5_compare.py --test cat_state  # single test case
  python m5_compare.py --test free_gauss --test ho_coherent  # multiple
"""

import sys
import os
import time
import argparse
import numpy as np

from m5.utils import output_path
from m5.fft_ref import schrodinger_fft_1d
from m5.init import init_ensemble_1d, Ensemble, Units
from m5.sim import (m5_simulate, select_backend,
                    _grid_psi_kde, kernel_sums, psi_kde_fields)

import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# Physical constants (atomic units)
# ═══════════════════════════════════════════════════════════════════════

HBAR = 1.0
MASS = 1.0
PI = np.pi


# ═══════════════════════════════════════════════════════════════════════
# Wavefunction helpers
# ═══════════════════════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, s0):
    """Normalised 1-D Gaussian wave packet."""
    return ((2 * PI * s0**2) ** (-0.25) *
            np.exp(-(x - x0)**2 / (4 * s0**2) + 1j * p0 * x / HBAR))


# ═══════════════════════════════════════════════════════════════════════
# Test case definitions
# ═══════════════════════════════════════════════════════════════════════
#
# Each case is a dict with keys:
#   tag, label, psi0_func, V_func,
#   xL, xR, Nx, T, Nt, Np, save_every,
#   grid_params  = {sigma_kde, K_cand, sigma_Q_smooth},
#   gridless_params = {K_gh, sigma_gh, h_kde},
#   extra  (optional: analytic_transmission, omega, etc.)
# ═══════════════════════════════════════════════════════════════════════

def make_test_cases():
    """Return list of test-case dicts."""
    cases = []

    # ── A: Free Gaussian ─────────────────────────────────────────────
    def psi0_free(x):
        return gaussian_wp(x, x0=0.0, p0=1.5, s0=1.5)

    cases.append(dict(
        tag='free_gauss', label='Free Gaussian',
        psi0_func=psi0_free,
        V_func=lambda x: np.zeros_like(x),
        xL=-15, xR=15, Nx=512, T=2.0, Nt=2000, Np=3000,
        save_every=50,
        grid_params=dict(sigma_kde=2.5, K_cand=48, sigma_Q_smooth=1.5),
        gridless_params=dict(K_gh=6, sigma_gh=0.20, h_kde=0.25),
    ))

    # ── B: Cat state ─────────────────────────────────────────────────
    x0c, p0c, s0c = 4.0, 3.0, 0.7

    def psi0_cat(x):
        return gaussian_wp(x, -x0c, +p0c, s0c) + \
               gaussian_wp(x, +x0c, -p0c, s0c)

    cases.append(dict(
        tag='cat_state', label='Cat State',
        psi0_func=psi0_cat,
        V_func=lambda x: np.zeros_like(x),
        xL=-15, xR=15, Nx=512, T=2.0, Nt=2000, Np=4000,
        save_every=50,
        grid_params=dict(sigma_kde=2.0, K_cand=48, sigma_Q_smooth=1.5),
        gridless_params=dict(K_gh=6, sigma_gh=0.20, h_kde=0.22),
    ))

    # ── C: HO ground state ──────────────────────────────────────────
    omega_ho = 1.0

    def psi0_ho_gs(x):
        return ((omega_ho * MASS / (PI * HBAR)) ** 0.25 *
                np.exp(-omega_ho * MASS * x**2 / (2 * HBAR)))

    def V_ho(x):
        return 0.5 * MASS * omega_ho**2 * x**2

    cases.append(dict(
        tag='ho_ground', label='HO Ground State',
        psi0_func=psi0_ho_gs,
        V_func=V_ho,
        xL=-8, xR=8, Nx=256, T=3.0, Nt=6000, Np=3000,
        save_every=100,
        grid_params=dict(sigma_kde=2.5, K_cand=48, sigma_Q_smooth=1.5),
        gridless_params=dict(K_gh=8, sigma_gh=0.15, h_kde=0.22),
        extra=dict(omega=omega_ho,
                   E_exact=0.5 * HBAR * omega_ho),
    ))

    # ── D: HO coherent state ────────────────────────────────────────
    # Displaced ground state: ψ(x,0) = ψ_gs(x - x_d) exp(i p_d x/ℏ)
    # with x_d, p_d chosen so that ⟨x⟩ oscillates with amplitude A.
    omega_co = 1.0
    A_co = 2.0       # oscillation amplitude

    def psi0_ho_coh(x):
        # Coherent state = displaced Gaussian; same width as ground state
        s_co = np.sqrt(HBAR / (2 * MASS * omega_co))
        x_d = A_co
        p_d = 0.0     # released from turning point → oscillates
        return gaussian_wp(x, x0=x_d, p0=p_d, s0=s_co)

    def V_ho_co(x):
        return 0.5 * MASS * omega_co**2 * x**2

    T_period = 2 * PI / omega_co    # one full period ≈ 6.28

    cases.append(dict(
        tag='ho_coherent', label='HO Coherent State',
        psi0_func=psi0_ho_coh,
        V_func=V_ho_co,
        xL=-8, xR=8, Nx=256, T=T_period, Nt=6000, Np=3000,
        save_every=100,
        grid_params=dict(sigma_kde=2.5, K_cand=48, sigma_Q_smooth=1.5),
        gridless_params=dict(K_gh=8, sigma_gh=0.15, h_kde=0.22),
        extra=dict(omega=omega_co, amplitude=A_co,
                   E_exact=0.5 * HBAR * omega_co +
                           0.5 * MASS * omega_co**2 * A_co**2),
    ))

    # ── E: Eckart barrier ───────────────────────────────────────────
    # V(x) = V0 / cosh²(x/a)
    # Exact transmission: T(E) = sinh²(πka)/[sinh²(πka) + cos²(…)]
    V0_eck = 1.0
    a_eck = 1.0
    E_inc = 0.8 * V0_eck   # sub-barrier → tunneling
    p0_eck = np.sqrt(2 * MASS * E_inc)
    s0_eck = 2.0            # broad enough for well-defined momentum

    def psi0_eck(x):
        return gaussian_wp(x, x0=-8.0, p0=p0_eck, s0=s0_eck)

    def V_eckart(x):
        return V0_eck / np.cosh(x / a_eck)**2

    # Analytic transmission coefficient
    k_eck = p0_eck / HBAR
    lam = 8 * MASS * V0_eck * a_eck**2 / HBAR**2
    if lam > 1:
        T_analytic = (np.sinh(PI * k_eck * a_eck)**2 /
                      (np.sinh(PI * k_eck * a_eck)**2 +
                       np.cos(0.5 * PI * np.sqrt(lam - 1))**2))
    else:
        T_analytic = (np.sinh(PI * k_eck * a_eck)**2 /
                      (np.sinh(PI * k_eck * a_eck)**2 +
                       np.cosh(0.5 * PI * np.sqrt(1 - lam))**2))

    cases.append(dict(
        tag='eckart', label='Eckart Barrier Tunneling',
        psi0_func=psi0_eck,
        V_func=V_eckart,
        xL=-20, xR=20, Nx=512, T=12.0, Nt=6000, Np=4000,
        save_every=100,
        grid_params=dict(sigma_kde=2.5, K_cand=48, sigma_Q_smooth=1.5),
        gridless_params=dict(K_gh=6, sigma_gh=0.20, h_kde=0.25),
        extra=dict(V0=V0_eck, a=a_eck, E_inc=E_inc,
                   T_analytic=T_analytic),
    ))

    return cases


# ═══════════════════════════════════════════════════════════════════════
# Energy diagnostics  (Approach B: grid reconstruction → FFT ⟨Ĥ⟩)
# ═══════════════════════════════════════════════════════════════════════

def energy_from_psi(psi_grid, V_grid, x_grid, hbar=HBAR, mass=MASS):
    """Compute ⟨Ĥ⟩ = E_kin + E_pot from ψ on a uniform grid.

    E_kin = (ℏ²/2m) ∫ |dψ/dx|² dx   (integration-by-parts form)
    E_pot = ∫ |ψ|² V dx

    Uses FFT derivative for E_kin.  Assumes periodic BC (split-operator
    convention).
    """
    Nx = len(x_grid)
    dx = float(x_grid[1] - x_grid[0])

    k = 2 * PI * np.fft.fftfreq(Nx, dx)
    dpsi = np.fft.ifft(1j * k * np.fft.fft(psi_grid))

    E_kin = (hbar**2 / (2 * mass)) * np.real(np.sum(np.abs(dpsi)**2) * dx)
    rho = np.abs(psi_grid)**2
    E_pot = np.real(np.sum(rho * V_grid) * dx)

    return E_kin, E_pot, E_kin + E_pot


def compute_energy_series(result, V_grid, x_grid, mode,
                          sigma_kde=2.5, hbar=HBAR, mass=MASS):
    """Compute energy at each snapshot for a simulation result.

    For grid mode: uses the stored ψ_grid snapshots directly.
    For gridless mode: reconstructs ψ on x_grid from (X, S) via
    CIC + Gaussian smooth (same algorithm as grid mode's ψ-KDE).

    Returns (t_save, E_kin, E_pot, E_total) arrays.
    """
    t_save = result['t_save']
    Ns = len(t_save)
    E_kin = np.zeros(Ns)
    E_pot = np.zeros(Ns)
    E_tot = np.zeros(Ns)

    x_dev = np.asarray(x_grid, dtype=np.float64)

    for si in range(Ns):
        if mode == 'grid' and 'psi' in result:
            psi_snap = result['psi'][si]
        else:
            # Reconstruct ψ on grid from particles
            X_snap = np.asarray(result['X'][si], dtype=np.float64)
            S_snap = np.asarray(result['S'][si], dtype=np.float64)
            psi_snap, _ = _grid_psi_kde(
                X_snap, S_snap, x_dev, sigma_kde, hbar, np)

        ek, ep, et = energy_from_psi(psi_snap, V_grid, x_grid,
                                     hbar=hbar, mass=mass)
        E_kin[si] = ek
        E_pot[si] = ep
        E_tot[si] = et

    return t_save, E_kin, E_pot, E_tot


# ═══════════════════════════════════════════════════════════════════════
# L² density error
# ═══════════════════════════════════════════════════════════════════════

def density_L2_error(result, rho_ref, ts_ref, x_grid, sigma_smooth=3.0):
    """L² error between M5 particle density and FFT reference.

    Returns (ts, errors) arrays aligned to the reference snapshots.
    """
    from scipy.ndimage import gaussian_filter1d

    dx = float(x_grid[1] - x_grid[0])
    Nx = len(x_grid)
    edges = np.linspace(x_grid[0], x_grid[-1] + dx, Nx + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    Ns_ref = len(ts_ref)
    errs = np.zeros(Ns_ref)

    for si in range(Ns_ref):
        i_m5 = np.argmin(np.abs(result['t_save'] - ts_ref[si]))
        h_m5, _ = np.histogram(result['X'][i_m5], bins=edges, density=True)
        rho_m5 = gaussian_filter1d(
            np.interp(x_grid, centres, h_m5), sigma=sigma_smooth)
        errs[si] = np.sqrt(np.sum((rho_m5 - rho_ref[si])**2) * dx)

    return ts_ref, errs


# ═══════════════════════════════════════════════════════════════════════
# Eckart barrier: transmission probability from ψ
# ═══════════════════════════════════════════════════════════════════════

def transmission_prob(psi_or_X, x_grid, mode='fft'):
    """Probability of finding particle at x > 0 (past the barrier).

    For 'fft': psi_or_X is a complex wavefunction on x_grid.
    For 'particles': psi_or_X is an array of particle positions.
    """
    if mode == 'fft':
        rho = np.abs(psi_or_X)**2
        dx = float(x_grid[1] - x_grid[0])
        mask = x_grid > 0
        return np.sum(rho[mask]) * dx
    else:
        X = psi_or_X
        return np.mean(X > 0)


# ═══════════════════════════════════════════════════════════════════════
# Figure builder
# ═══════════════════════════════════════════════════════════════════════

def build_figure(case, psi_ref, ts_ref, x_grid,
                 res_grid, res_gridless, E_grid, E_gridless,
                 errs_grid, errs_gridless):
    """Build the 6-panel comparison figure for one test case.

    Layout (3 rows × 2 cols):
      (a) heatmap + grid trajectories       (b) heatmap + gridless trajectories
      (c) density snapshots at key times     (d) L² error vs time
      (e) energy ⟨Ĥ⟩ vs time               (f) summary text
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_ref = np.abs(psi_ref)**2
    tag = case['tag']
    label = case['label']
    T = case['T']
    dx = float(x_grid[1] - x_grid[0])

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f'{label} — Grid ψ-KDE vs Gridless Swarmalator  '
                 f'(Np={case["Np"]})',
                 fontsize=14, weight='bold', y=0.98)

    # ── Shared: choose tracked trajectory subset ─────────────────────
    def plot_trajectories(ax, result, title, c_left, c_right):
        """Space-time heatmap with trajectory overlay."""
        ax.imshow(rho_ref.T, origin='lower', aspect='auto',
                  extent=[0, T, x_grid[0], x_grid[-1]],
                  cmap='inferno', interpolation='bilinear',
                  vmin=0, vmax=np.percentile(rho_ref, 99))

        if 'traj_X' in result and result['traj_X'] is not None:
            traj_X = result['traj_X']
            traj_t = result['traj_t']
            N_track = traj_X.shape[1]
            t_thin = max(1, len(traj_t) // 600)

            x_init = traj_X[0, :]
            left = np.where(x_init < np.median(x_init))[0]
            right = np.where(x_init >= np.median(x_init))[0]
            n_show = min(30, N_track // 2)

            for idx in left[np.linspace(0, len(left)-1,
                                        min(n_show, len(left)),
                                        dtype=int)]:
                ax.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                        '-', color=c_left, lw=0.4, alpha=0.65)
            for idx in right[np.linspace(0, len(right)-1,
                                         min(n_show, len(right)),
                                         dtype=int)]:
                ax.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                        '-', color=c_right, lw=0.4, alpha=0.65)

        ylim = max(6.0, 0.6 * (x_grid[-1] - x_grid[0]) / 2)
        mid = 0.5 * (x_grid[0] + x_grid[-1])
        ax.set_ylim(mid - ylim, mid + ylim)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_title(title, fontsize=11)

    # (a) Grid trajectories
    plot_trajectories(axes[0, 0], res_grid,
                      'Grid ψ-KDE trajectories',
                      '#66ccff', '#ff9966')

    # (b) Gridless trajectories
    plot_trajectories(axes[0, 1], res_gridless,
                      'Gridless swarmalator trajectories',
                      '#44bbff', '#ff6633')

    # (c) Density snapshots at key times
    ax_rho = axes[1, 0]
    Ns_ref = len(ts_ref)
    snap_idx = [0, Ns_ref // 3, 2 * Ns_ref // 3, Ns_ref - 1]
    colours = ['#2166ac', '#b2182b', '#4dac26', '#7570b3']
    from scipy.ndimage import gaussian_filter1d
    edges = np.linspace(x_grid[0], x_grid[-1] + dx, len(x_grid) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bw = 3.0

    for ci, si in enumerate(snap_idx):
        t_s = ts_ref[si]

        # Exact
        ax_rho.plot(x_grid, rho_ref[si], '-', color=colours[ci],
                    lw=2, alpha=0.5)

        # Grid
        ig = np.argmin(np.abs(res_grid['t_save'] - t_s))
        hg, _ = np.histogram(res_grid['X'][ig], bins=edges, density=True)
        rg = gaussian_filter1d(np.interp(x_grid, centres, hg), sigma=bw)
        ax_rho.plot(x_grid, rg, '--', color=colours[ci], lw=1.3)

        # Gridless
        il = np.argmin(np.abs(res_gridless['t_save'] - t_s))
        hl, _ = np.histogram(res_gridless['X'][il], bins=edges, density=True)
        rl = gaussian_filter1d(np.interp(x_grid, centres, hl), sigma=bw)
        ax_rho.plot(x_grid, rl, ':', color=colours[ci], lw=1.5,
                    label=f't={t_s:.2f}')

    xlim = 0.6 * (x_grid[-1] - x_grid[0]) / 2
    mid = 0.5 * (x_grid[0] + x_grid[-1])
    ax_rho.set_xlim(mid - xlim, mid + xlim)
    ax_rho.set_xlabel('x')
    ax_rho.set_ylabel('ρ(x,t)')
    ax_rho.set_title('Density: exact (solid), grid (dashed), gridless (dotted)',
                     fontsize=10)
    ax_rho.legend(fontsize=8)

    # (d) L² error vs time
    ax_err = axes[1, 1]
    ax_err.semilogy(ts_ref, errs_grid, '-', color='#2166ac', lw=1.5,
                    label=f'Grid  (mean={np.mean(errs_grid):.5f})')
    ax_err.semilogy(ts_ref, errs_gridless, '-', color='#d6604d', lw=1.5,
                    label=f'Gridless  (mean={np.mean(errs_gridless):.5f})')
    ax_err.set_xlabel('t')
    ax_err.set_ylabel('L² error')
    ax_err.set_title('L² density error vs time', fontsize=11)
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3)

    # (e) Energy conservation
    ax_E = axes[2, 0]
    t_g, Ek_g, Ep_g, Et_g = E_grid
    t_l, Ek_l, Ep_l, Et_l = E_gridless

    # Normalise to initial energy
    E0_g = Et_g[0] if abs(Et_g[0]) > 1e-12 else 1.0
    E0_l = Et_l[0] if abs(Et_l[0]) > 1e-12 else 1.0

    ax_E.plot(t_g, (Et_g - Et_g[0]) / abs(E0_g) * 100,
              '-', color='#2166ac', lw=1.5, label='Grid')
    ax_E.plot(t_l, (Et_l - Et_l[0]) / abs(E0_l) * 100,
              '-', color='#d6604d', lw=1.5, label='Gridless')

    # Exact energy line if available
    extra = case.get('extra', {})
    if 'E_exact' in extra:
        E_ex = extra['E_exact']
        ax_E.axhline((E_ex - Et_g[0]) / abs(E0_g) * 100,
                     ls=':', color='gray', alpha=0.5, label='Exact')

    ax_E.set_xlabel('t')
    ax_E.set_ylabel('ΔE / |E₀|  (%)')
    ax_E.set_title('Energy conservation', fontsize=11)
    ax_E.legend(fontsize=9)
    ax_E.grid(True, alpha=0.3)

    # (f) Summary text
    ax_txt = axes[2, 1]
    ax_txt.axis('off')

    drift_g = abs(Et_g[-1] - Et_g[0]) / max(abs(E0_g), 1e-12)
    drift_l = abs(Et_l[-1] - Et_l[0]) / max(abs(E0_l), 1e-12)

    # FFT reference energy at t=0 for bias assessment
    V_grid = case['V_func'](x_grid)
    _, _, E_fft0 = energy_from_psi(psi_ref[0], V_grid, x_grid)

    summary = (
        f"{label}\n"
        f"{'─' * 46}\n"
        f"  Np = {case['Np']},  Nt = {case['Nt']},  T = {case['T']:.2f}\n\n"
        f"  E(t=0):  FFT = {E_fft0:.4f}\n"
        f"    Grid recon = {Et_g[0]:.4f}  "
        f"(bias {(Et_g[0]-E_fft0)/max(abs(E_fft0),1e-12)*100:+.1f}%)\n"
        f"    GL   recon = {Et_l[0]:.4f}  "
        f"(bias {(Et_l[0]-E_fft0)/max(abs(E_fft0),1e-12)*100:+.1f}%)\n\n"
        f"  Method     │ mean L²   │ E drift  │  Time\n"
        f"  ────────── │ ───────── │ ──────── │ ─────\n"
        f"  Grid       │ {np.mean(errs_grid):9.5f} │ {drift_g:8.2e} │"
        f" {res_grid['wall_time']:5.1f}s\n"
        f"  Gridless   │ {np.mean(errs_gridless):9.5f} │ {drift_l:8.2e} │"
        f" {res_gridless['wall_time']:5.1f}s\n"
    )

    # Eckart barrier: transmission comparison
    if tag == 'eckart' and 'T_analytic' in extra:
        T_an = extra['T_analytic']
        # FFT reference
        psi_final_ref = psi_ref[-1]
        T_fft = transmission_prob(psi_final_ref, x_grid, mode='fft')
        # Particle methods
        ig_last = len(res_grid['t_save']) - 1
        il_last = len(res_gridless['t_save']) - 1
        T_grid_p = transmission_prob(res_grid['X'][ig_last], x_grid,
                                     mode='particles')
        T_gl_p = transmission_prob(res_gridless['X'][il_last], x_grid,
                                   mode='particles')
        summary += (
            f"\n  Eckart transmission (E/V₀ = {extra['E_inc']/extra['V0']:.2f}):\n"
            f"    Analytic  T = {T_an:.4f}\n"
            f"    FFT ref   T = {T_fft:.4f}\n"
            f"    Grid      T = {T_grid_p:.4f}\n"
            f"    Gridless  T = {T_gl_p:.4f}\n"
        )

    # Exact energy if available
    if 'E_exact' in extra:
        summary += f"\n  E_exact = {extra['E_exact']:.4f}\n"
        summary += f"  E₀ grid = {Et_g[0]:.4f},  E₀ gridless = {Et_l[0]:.4f}\n"

    ax_txt.text(0.04, 0.96, summary, transform=ax_txt.transAxes,
                fontsize=9.5, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Run one test case
# ═══════════════════════════════════════════════════════════════════════

def run_case(case, force_backend=None):
    """Run FFT reference + grid + gridless for one test case.

    Returns (psi_ref, ts_ref, res_grid, res_gridless, E_grid, E_gridless,
             errs_grid, errs_gridless).
    """
    tag = case['tag']
    label = case['label']
    psi0_func = case['psi0_func']
    V_func = case['V_func']

    xL, xR, Nx = case['xL'], case['xR'], case['Nx']
    T, Nt, Np = case['T'], case['Nt'], case['Np']
    save_every = case['save_every']

    dx = (xR - xL) / Nx
    x_grid = np.linspace(xL, xR, Nx, endpoint=False)

    # Normalise ψ₀ on grid
    psi0_grid = psi0_func(x_grid)
    norm = np.sqrt(np.sum(np.abs(psi0_grid)**2) * dx)
    psi0_grid /= norm
    psi0_func_normed = lambda x, _f=psi0_func, _n=norm: _f(x) / _n

    V_grid = V_func(x_grid)

    # ── FFT reference ────────────────────────────────────────────────
    print(f"    FFT reference...", end=" ", flush=True)
    t0 = time.time()
    psi_ref, ts_ref = schrodinger_fft_1d(
        psi0_grid, V_grid, x_grid, T, Nt,
        hbar=HBAR, mass=MASS, save_every=save_every)
    print(f"({time.time() - t0:.1f}s)")

    rho_ref = np.abs(psi_ref)**2

    # ── Create shared ensemble ───────────────────────────────────────
    ens = init_ensemble_1d(psi0_grid, x_grid, Np,
                           mass=MASS, hbar=HBAR, seed=42,
                           psi0_func=psi0_func_normed)

    # Trajectory tracking: evenly spaced among sorted initial positions
    N_track = min(80, Np)
    sorted_ids = np.argsort(ens.X)
    track_ids = sorted_ids[np.linspace(0, Np - 1, N_track, dtype=int)]

    # ── Grid mode ────────────────────────────────────────────────────
    gp = case['grid_params']
    print(f"    Grid mode (σ_kde={gp['sigma_kde']}, "
          f"K_cand={gp['K_cand']})...", flush=True)
    t0 = time.time()
    res_grid = m5_simulate(
        ens, V_func, T, Nt,
        mode='grid', units=Units(hbar=HBAR), x_grid=x_grid,
        sigma_kde=gp['sigma_kde'], K_cand=gp['K_cand'],
        sigma_Q_smooth=gp['sigma_Q_smooth'],
        save_every=save_every, seed=42,
        track_ids=track_ids, backend=force_backend)
    print(f"    done ({res_grid['wall_time']:.1f}s)")

    # ── Gridless mode ────────────────────────────────────────────────
    glp = case['gridless_params']
    print(f"    Gridless mode (K_gh={glp['K_gh']}, σ_gh={glp['sigma_gh']}, "
          f"h_kde={glp['h_kde']})...", flush=True)
    t0 = time.time()
    res_gridless = m5_simulate(
        ens, V_func, T, Nt,
        mode='gridless', units=Units(hbar=HBAR), x_grid=x_grid,
        K_gh=glp['K_gh'], sigma_gh=glp['sigma_gh'],
        h_kde=glp['h_kde'],
        save_every=save_every, seed=42,
        track_ids=track_ids, backend=force_backend)
    print(f"    done ({res_gridless['wall_time']:.1f}s)")

    # ── Energy diagnostics ───────────────────────────────────────────
    print(f"    Energy diagnostics...", end=" ", flush=True)
    t0 = time.time()
    E_grid = compute_energy_series(
        res_grid, V_grid, x_grid, mode='grid',
        sigma_kde=gp['sigma_kde'])
    E_gridless = compute_energy_series(
        res_gridless, V_grid, x_grid, mode='gridless',
        sigma_kde=gp['sigma_kde'])
    print(f"({time.time() - t0:.1f}s)")

    # ── L² density error ─────────────────────────────────────────────
    _, errs_grid = density_L2_error(res_grid, rho_ref, ts_ref, x_grid)
    _, errs_gridless = density_L2_error(
        res_gridless, rho_ref, ts_ref, x_grid)

    return (psi_ref, ts_ref, x_grid,
            res_grid, res_gridless,
            E_grid, E_gridless,
            errs_grid, errs_gridless)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='M5 Grid vs Gridless comparison demo')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--gpu', action='store_true', help='Force GPU')
    parser.add_argument('--quick', action='store_true',
                        help='Reduced parameters for quick testing')
    parser.add_argument('--test', type=str, action='append', default=None,
                        help='Run specific test(s): free_gauss, cat_state, '
                             'ho_ground, ho_coherent, eckart')
    args = parser.parse_args()

    force = 'cpu' if args.cpu else ('gpu' if args.gpu else None)
    xp, backend_name = select_backend(force)

    print("=" * 65)
    print("  M5 Grid vs Gridless Comparison")
    print(f"  Backend: {backend_name}")
    print("=" * 65)

    cases = make_test_cases()

    if args.test:
        tags = set(args.test)
        cases = [c for c in cases if c['tag'] in tags]
        if not cases:
            print(f"Unknown test(s): {args.test}")
            print(f"Available: free_gauss, cat_state, ho_ground, "
                  f"ho_coherent, eckart")
            sys.exit(1)

    if args.quick:
        print("  [QUICK MODE: reduced parameters]")
        for c in cases:
            c['Np'] = min(c['Np'], 600)
            c['Nt'] = min(c['Nt'], 1000)
            c['save_every'] = max(1, c['Nt'] // 20)
            # Widen gridless bandwidth for small Np
            c['gridless_params']['h_kde'] = max(
                c['gridless_params']['h_kde'], 0.35)
            # Note: T is NOT truncated — each test needs its full
            # propagation time for meaningful physics (e.g. Eckart
            # tunneling, HO coherent oscillation).

    all_results = {}

    for case in cases:
        tag = case['tag']
        label = case['label']
        print(f"\n{'─' * 55}")
        print(f"  {label}  [{tag}]")
        print(f"  Np={case['Np']}, Nt={case['Nt']}, T={case['T']:.2f}")
        print(f"{'─' * 55}")

        (psi_ref, ts_ref, x_grid,
         res_grid, res_gridless,
         E_grid, E_gridless,
         errs_grid, errs_gridless) = run_case(case, force_backend=force)

        # Build and save figure
        print(f"    Plotting...", end=" ", flush=True)
        fig = build_figure(case, psi_ref, ts_ref, x_grid,
                           res_grid, res_gridless,
                           E_grid, E_gridless,
                           errs_grid, errs_gridless)
        fname = output_path(f"m5_compare_{tag}.png")
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"saved {os.path.basename(fname)}")

        # Collect summary
        _, _, _, Et_g = E_grid
        _, _, _, Et_l = E_gridless
        E0_g = Et_g[0] if abs(Et_g[0]) > 1e-12 else 1.0
        E0_l = Et_l[0] if abs(Et_l[0]) > 1e-12 else 1.0

        all_results[tag] = dict(
            label=label,
            err_grid=np.mean(errs_grid),
            err_gridless=np.mean(errs_gridless),
            drift_grid=abs(Et_g[-1] - Et_g[0]) / abs(E0_g),
            drift_gridless=abs(Et_l[-1] - Et_l[0]) / abs(E0_l),
            time_grid=res_grid['wall_time'],
            time_gridless=res_gridless['wall_time'],
        )

        print(f"    Grid:     L²={all_results[tag]['err_grid']:.5f}  "
              f"ΔE={all_results[tag]['drift_grid']:.2e}  "
              f"t={all_results[tag]['time_grid']:.1f}s")
        print(f"    Gridless: L²={all_results[tag]['err_gridless']:.5f}  "
              f"ΔE={all_results[tag]['drift_gridless']:.2e}  "
              f"t={all_results[tag]['time_gridless']:.1f}s")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'═' * 85}")
    print(f"  SUMMARY  ({backend_name})")
    print(f"{'═' * 85}")
    print(f"  {'Case':<20s} │ {'Grid L²':>9s} {'GL L²':>9s} │ "
          f"{'Grid ΔE':>9s} {'GL ΔE':>9s} │ "
          f"{'Grid t':>6s} {'GL t':>6s}")
    print(f"  {'─'*20} │ {'─'*9} {'─'*9} │ "
          f"{'─'*9} {'─'*9} │ {'─'*6} {'─'*6}")
    for tag, r in all_results.items():
        name = r['label'][:20]
        print(f"  {name:<20s} │ "
              f"{r['err_grid']:9.5f} {r['err_gridless']:9.5f} │ "
              f"{r['drift_grid']:9.2e} {r['drift_gridless']:9.2e} │ "
              f"{r['time_grid']:5.1f}s {r['time_gridless']:5.1f}s")
    print(f"{'═' * 85}")


if __name__ == "__main__":
    main()
