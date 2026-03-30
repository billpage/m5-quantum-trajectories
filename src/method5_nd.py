#!/usr/bin/env python3
"""
Method 5 — N-Dimensional / Multi-Particle √ρ-Selection Solver
==============================================================

Generalises the 1-D √ρ-selection algorithm (Method 5) to an arbitrary
D-dimensional configuration space, covering both:

  • A single quantum particle in d spatial dimensions (D = d)
  • N distinguishable quantum particles in 1-D (D = N)
  • N particles in d dimensions (D = N × d)

The configuration-space walker **q** ∈ ℝ^D evolves as

    dq = [v(q,t) + u(q,t)] dt + σ dW_D

where σ = √(ℏ/m), v = ∇_q S / m is the D-dimensional current velocity,
and the osmotic correction u = ν ∇_q ln ρ is implemented implicitly through
the √ρ-selection mechanism (no log-gradient ever computed):

    generate K candidates q'_k = q_class + σ√dt ξ_k,  ξ_k ~ N(0, I_D)
    select with probability ∝ √ρ(q'_k)

Mathematical justification: the selection induces effective drift
    σ² ∇_q ln √ρ = ½(ℏ/m) ∇_q ln ρ = ν ∇_q ln ρ = u(q,t)
by the tilted-Gaussian / importance-sampling identity (dimension-agnostic).

GPU acceleration
----------------
Set USE_GPU=True (or pass --gpu flag).  If CuPy is installed the heavy inner
loops (candidate generation, √ρ interpolation, categorical draw) move to GPU.
All other bookkeeping stays on CPU/NumPy.

Batching
--------
Candidate generation and selection are processed in configurable particle
batches so the working set fits in GPU / RAM regardless of Np.

Testing
-------
Run with default arguments to reproduce the 1-D cat-state results from
method5_catstate.py and compare directly against the Schrödinger FFT
reference and the Method-4 explicit-osmotic solver.

Usage
-----
    python method5_nd.py [--gpu] [--Np 8000] [--K 48] [--batch 2000]
"""

import argparse, time, warnings, sys
from m5.utils import output_path
from m5.fft_ref import schrodinger_fft_1d as _schrodinger_fft_1d
warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Backend  (numpy / cupy)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False

def get_xp(use_gpu: bool):
    """Return the array namespace (cupy or numpy)."""
    if use_gpu and _CUPY_OK:
        return cp
    if use_gpu and not _CUPY_OK:
        print("  [WARNING] CuPy not found — falling back to NumPy (CPU).")
    return np

def to_numpy(arr, xp):
    """Move array to CPU numpy regardless of origin."""
    if xp is np:
        return np.asarray(arr)
    return cp.asnumpy(arr)

def to_xp(arr_np, xp):
    """Move numpy array to target backend."""
    if xp is np:
        return arr_np
    return cp.asarray(arr_np)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Configuration dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsParams:
    """Physical constants.  For N particles with equal mass m."""
    hbar: float = 1.0
    m:    float = 1.0

    @property
    def nu(self):          return self.hbar / (2.0 * self.m)
    @property
    def sigma_noise(self): return float(np.sqrt(self.hbar / self.m))


@dataclass
class GridParams:
    """
    Uniform Cartesian grid for D-dimensional configuration space.

    Each of the D axes shares the same bounds [qL, qR] and N_grid points.
    (For non-uniform masses or mixed coordinate ranges, override q_ranges.)

    Time grid: T total, Nt steps.
    """
    D:      int   = 1          # configuration-space dimension
    qL:     float = -15.0      # lower bound (same for all axes)
    qR:     float =  15.0      # upper bound
    Ng:     int   = 512        # grid points per axis
    T:      float = 3.0
    Nt:     int   = 3000

    # Optional per-axis overrides
    q_ranges: Optional[List[Tuple[float,float]]] = None  # [(qL,qR)]*D

    @property
    def dq(self):
        return (self.qR - self.qL) / self.Ng

    @property
    def dt(self):
        return self.T / self.Nt

    @property
    def axes(self):
        """Return list of 1-D grid vectors for each axis."""
        if self.q_ranges is None:
            return [np.linspace(self.qL, self.qR, self.Ng, endpoint=False)
                    for _ in range(self.D)]
        return [np.linspace(lo, hi, self.Ng, endpoint=False)
                for (lo, hi) in self.q_ranges]

    @property
    def t_arr(self):
        return np.linspace(0, self.T, self.Nt + 1)

    def dq_per_axis(self):
        if self.q_ranges is None:
            return [self.dq] * self.D
        return [(hi - lo) / self.Ng for (lo, hi) in self.q_ranges]

    def bounds(self):
        if self.q_ranges is None:
            return [(self.qL, self.qR)] * self.D
        return self.q_ranges


# ══════════════════════════════════════════════════════════════════════════════
# 2.  D-dimensional density / phase estimation on a regular grid
#     (all routines stay on CPU / NumPy — they operate on the summary grid,
#      not on every particle individually)
# ══════════════════════════════════════════════════════════════════════════════

def _grid_index(Q_np, gp: GridParams):
    """
    Map particle positions Q (Np, D) → grid indices (Np, D) as integer arrays.
    Returns (idx, lo, frac) where lo is the floor index and frac ∈ [0,1) is
    the fractional offset for linear interpolation.
    """
    axes  = gp.axes
    dqs   = gp.dq_per_axis()
    Ng    = gp.Ng
    lo    = np.zeros_like(Q_np, dtype=np.int64)
    frac  = np.zeros_like(Q_np, dtype=np.float64)
    for d in range(gp.D):
        fi = (Q_np[:, d] - axes[d][0]) / dqs[d]
        fi = np.clip(fi, 0.0, Ng - 1.001)
        lo[:, d]   = fi.astype(np.int64)
        frac[:, d] = fi - lo[:, d]
    return lo, frac


def interp_nd(grid_vals, Q_np, gp: GridParams):
    """
    N-linear interpolation of a D-dimensional scalar field grid_vals
    (shape Ng^D) at particle positions Q (Np, D).

    Uses vertex enumeration over 2^D corners.
    """
    Np   = Q_np.shape[0]
    Ng   = gp.Ng
    D    = gp.D
    lo, frac = _grid_index(Q_np, gp)   # (Np, D)

    result = np.zeros(Np, dtype=np.float64)

    # Iterate over 2^D vertices of the unit hypercube
    for vertex in range(1 << D):
        weight = np.ones(Np, dtype=np.float64)
        idx    = np.zeros((Np, D), dtype=np.int64)
        for d in range(D):
            if (vertex >> d) & 1:
                idx[:, d] = np.minimum(lo[:, d] + 1, Ng - 1)
                weight   *= frac[:, d]
            else:
                idx[:, d] = lo[:, d]
                weight   *= (1.0 - frac[:, d])
        # Flat index into grid_vals
        flat = np.zeros(Np, dtype=np.int64)
        stride = 1
        for d in range(D - 1, -1, -1):
            flat  += idx[:, d] * stride
            stride *= Ng
        result += weight * grid_vals.ravel()[flat]
    return result


def estimate_density(Q_np, gp: GridParams, smooth_sigma: float = 4.0):
    """
    Estimate the D-dimensional density ρ on a regular grid from particle
    positions Q (Np, D) via histogram + Gaussian smoothing.

    Returns grid array of shape (Ng,)*D, normalised so ∫ρ dq^D = 1.
    """
    D  = gp.D
    Ng = gp.Ng
    dqs = gp.dq_per_axis()
    axes = gp.axes

    # Build multi-dimensional histogram
    # np.histogramdd accepts (Np, D) data
    edges = [np.linspace(lo, hi, Ng + 1)
             for (lo, hi) in gp.bounds()]
    h, _ = np.histogramdd(Q_np, bins=edges)
    h = h.astype(np.float64)

    # Gaussian smooth with bandwidth sigma in grid units for each axis
    sigma_grid = [smooth_sigma / dq for dq in dqs]
    rho = gaussian_filter(h, sigma=sigma_grid)
    rho = np.maximum(rho, 1e-30)

    # Normalise  ∫ρ dq^D = 1
    cell_vol = float(np.prod(dqs))
    rho /= (rho.sum() * cell_vol)
    return rho


def estimate_phase_gradient(Q_np, S_np, gp: GridParams, smooth_sigma: float = 3.0):
    """
    Estimate S(q) and its gradient ∇_q S on the grid from scattered
    (q_i, S_i) data by binning and smoothing.

    Returns:
        S_grid  : (Ng,)*D array — smoothed phase field
        gradS   : D × (Ng,)*D array — gradient (list of D arrays)
    """
    D  = gp.D
    Ng = gp.Ng
    dqs = gp.dq_per_axis()

    edges = [np.linspace(lo, hi, Ng + 1)
             for (lo, hi) in gp.bounds()]

    # Bin S values: weighted sum / count
    S_sum, _ = np.histogramdd(Q_np, bins=edges, weights=S_np)
    cnt,   _ = np.histogramdd(Q_np, bins=edges)
    cnt    = np.maximum(cnt, 1.0)
    S_grid = S_sum / cnt

    # Inpaint empty cells by smoothing (zero weight empty cells contribute 0)
    # Use a larger smoothing sigma for stability
    sigma_grid = [smooth_sigma / dq for dq in dqs]
    S_grid = gaussian_filter(S_grid, sigma=sigma_grid)

    # Gradient via central differences (periodic wrap is fine for smooth fields)
    gradS = []
    for d in range(D):
        # np.gradient uses non-uniform spacing; we have uniform dq
        g = np.gradient(S_grid, dqs[d], axis=d)
        gradS.append(g)
    return S_grid, gradS


def compute_quantum_potential(rho_grid, gp: GridParams, pp: PhysicsParams,
                               smooth_sigma: float = 2.0):
    """
    Q = -(ℏ²/2m) ∇²_q √ρ / √ρ

    Compute on the D-dimensional grid.  Uses the sum of second differences
    along each axis.
    """
    dqs   = gp.dq_per_axis()
    hbar, m = pp.hbar, pp.m
    sqr  = np.sqrt(np.maximum(rho_grid, 1e-30))

    lap = np.zeros_like(sqr)
    for d in range(gp.D):
        # Second derivative along axis d via central difference
        lap += np.gradient(np.gradient(sqr, dqs[d], axis=d), dqs[d], axis=d)

    Q_grid = -(hbar**2 / (2.0 * m)) * lap / np.maximum(sqr, 1e-30)
    sigma_grid = [smooth_sigma / dq for dq in dqs]
    Q_grid = gaussian_filter(Q_grid, sigma=sigma_grid)
    return Q_grid


# ══════════════════════════════════════════════════════════════════════════════
# 3.  The core Method 5 ND solver
# ══════════════════════════════════════════════════════════════════════════════

def method5_nd(
    Q0_np,            # (Np, D)  — initial particle positions
    S0_np,            # (Np,)    — initial phase values
    V_func,           # callable: V(Q_np)  → (Np,) or broadcasts over grid
    pp: PhysicsParams,
    gp: GridParams,
    K:          int   = 32,
    batch_size: int   = 2000,
    save_every: int   = 10,
    track_ids         = None,   # list of particle indices to track every step
    use_gpu:    bool  = False,
    smooth_density:  float = 4.0,
    smooth_phase:    float = 3.0,
    smooth_Q:        float = 2.0,
    seed:       int   = 42,
):
    """
    Method 5: √ρ-selection FBSDE particle solver in D-dimensional
    configuration space.

    Parameters
    ----------
    Q0_np    : (Np, D)  initial configuration positions
    S0_np    : (Np,)    initial phase ψ = √ρ exp(iS/ℏ)
    V_func   : callable mapping (Np,D) → (Np,) potential energy
    pp       : PhysicsParams
    gp       : GridParams
    K        : number of Brownian candidates per particle per step
    batch_size : particles per GPU/RAM batch for selection step
    save_every : how often (in steps) to save snapshots
    track_ids  : list of particle indices whose full trajectory is stored
    use_gpu    : try to use CuPy for selection kernel
    smooth_*   : Gaussian smoothing bandwidths in physical units

    Returns
    -------
    dict with keys:
        Q        : (Ns, Np, D)  snapshots of particle positions
        rho      : (Ns,)*D+1    snapshots of density grid
        t_save   : (Ns,)        snapshot times
        traj_Q   : (Nt+1, N_track, D)  full trajectories if track_ids given
        traj_t   : (Nt+1,)             trajectory times
    """
    xp   = get_xp(use_gpu)
    rng  = np.random.default_rng(seed)

    Np, D = Q0_np.shape
    Ng    = gp.Ng
    dt    = gp.dt
    hbar, m, nu, sig = pp.hbar, pp.m, pp.nu, pp.sigma_noise
    dqs   = gp.dq_per_axis()
    axes  = gp.axes
    bnds  = gp.bounds()

    Q  = Q0_np.copy().astype(np.float64)
    Sp = S0_np.copy().astype(np.float64)

    idx_save = list(range(0, gp.Nt + 1, save_every))
    Ns = len(idx_save)
    Q_h   = np.zeros((Ns, Np, D), dtype=np.float32)
    rho_shape = (Ns,) + (Ng,) * D
    rho_h = np.zeros(rho_shape, dtype=np.float32)
    si    = 0

    if track_ids is not None:
        track_ids = np.array(track_ids, dtype=np.int64)
        traj_Q = np.zeros((gp.Nt + 1, len(track_ids), D), dtype=np.float32)
        traj_Q[0] = Q[track_ids].astype(np.float32)
    else:
        traj_Q = None

    # Initial density estimate
    rho_est = estimate_density(Q, gp, smooth_density)

    if 0 in idx_save:
        Q_h[si] = Q.astype(np.float32)
        rho_h[si] = rho_est.astype(np.float32)
        si += 1

    for n in range(1, gp.Nt + 1):

        # ── 1.  Density estimate ──────────────────────────────────
        rho_est  = estimate_density(Q, gp, smooth_density)
        sqrt_rho = np.sqrt(rho_est)  # (Ng,)*D — stays on CPU

        # ── 2.  Phase gradient → current velocity field ───────────
        _, gradS = estimate_phase_gradient(Q, Sp, gp, smooth_phase)
        # gradS: list of D arrays, each (Ng,)*D
        # Interpolate v_d at particle positions
        v_at = np.zeros((Np, D), dtype=np.float64)
        for d in range(D):
            v_field_d = gradS[d] / m
            v_at[:, d] = interp_nd(v_field_d, Q, gp)

        # ── 3.  Classical push  q_class = q + v dt ────────────────
        Q_class = Q + v_at * dt   # (Np, D)

        # ── 4-6.  Batched √ρ-selection ────────────────────────────
        # Transfer sqrt_rho to GPU once per step
        sqrt_rho_xp = to_xp(sqrt_rho.ravel(), xp)  # flat, stays until next step

        Q_new = np.zeros_like(Q)
        n_batches = (Np + batch_size - 1) // batch_size

        for b in range(n_batches):
            sl  = slice(b * batch_size, min((b + 1) * batch_size, Np))
            Nb  = sl.stop - sl.start

            # Generate candidates: (Nb, K, D)
            noise  = rng.standard_normal((Nb, K, D))   # CPU rng for reproducibility
            noise_xp = to_xp(noise, xp)
            Qc_xp  = to_xp(Q_class[sl], xp)           # (Nb, D)

            cands_xp = Qc_xp[:, None, :] + (sig * np.sqrt(dt)) * noise_xp
            # shape: (Nb, K, D)

            # Clip to domain
            for d in range(D):
                lo_d, hi_d = bnds[d]
                cands_xp[:, :, d] = xp.clip(cands_xp[:, :, d],
                                             lo_d + dqs[d],
                                             hi_d - dqs[d])

            # Evaluate √ρ at candidates via ND linear interpolation on GPU/CPU
            # We do this in the xp namespace for GPU speed
            # Flatten candidates to (Nb*K, D) for batch interpolation
            cands_flat = cands_xp.reshape(-1, D)   # (Nb*K, D)

            w_flat = _interp_nd_xp(sqrt_rho_xp, cands_flat, gp, xp)
            # shape: (Nb*K,)
            w = xp.maximum(w_flat, 1e-30).reshape(Nb, K)

            # Normalise and draw
            w_sum = w.sum(axis=1, keepdims=True)
            probs = w / w_sum                          # (Nb, K)
            cum   = xp.cumsum(probs, axis=1)           # (Nb, K)

            u_draw = to_xp(rng.uniform(size=(Nb, 1)), xp)
            chosen = (cum < u_draw).sum(axis=1)        # (Nb,)  ∈ [0, K)
            chosen = xp.clip(chosen, 0, K - 1)

            # Gather selected positions
            idx_b  = xp.arange(Nb, dtype=xp.int64)
            Q_sel_xp = cands_xp[idx_b, chosen, :]     # (Nb, D)
            Q_new[sl] = to_numpy(Q_sel_xp, xp)

        Q = Q_new

        # ── 7.  Quantum potential ──────────────────────────────────
        Q_grid = compute_quantum_potential(rho_est, gp, pp, smooth_Q)
        Q_at   = interp_nd(Q_grid, Q, gp)            # (Np,)

        # ── 8.  Phase update  dS = [½m|v|² - V - Q] dt ───────────
        v_at_new = np.zeros((Np, D), dtype=np.float64)
        for d in range(D):
            v_field_d = gradS[d] / m
            v_at_new[:, d] = interp_nd(v_field_d, Q, gp)

        vsq   = np.sum(v_at_new**2, axis=1)           # |v|²
        V_at  = V_func(Q)                             # (Np,)
        Sp   += (0.5 * m * vsq - V_at - Q_at) * dt

        # ── 9.  Trajectory tracking ────────────────────────────────
        if traj_Q is not None:
            traj_Q[n] = Q[track_ids].astype(np.float32)

        if n in idx_save and si < Ns:
            Q_h[si]   = Q.astype(np.float32)
            rho_h[si] = rho_est.astype(np.float32)
            si += 1

    # Fill any remaining save slots
    for j in range(si, Ns):
        Q_h[j]   = Q.astype(np.float32)
        rho_h[j] = rho_est.astype(np.float32)

    result = dict(Q=Q_h, rho=rho_h,
                  t_save=gp.t_arr[idx_save[:Ns]])
    if traj_Q is not None:
        result['traj_Q'] = traj_Q
        result['traj_t'] = gp.t_arr
    return result


def _interp_nd_xp(sqrt_rho_flat, Q_flat, gp: GridParams, xp):
    """
    N-linear interpolation of a D-dimensional field (stored as a flat 1-D
    array in C-order) at positions Q_flat (Np, D).
    Runs entirely in the xp namespace (numpy or cupy).
    """
    Np  = Q_flat.shape[0]
    D   = gp.D
    Ng  = gp.Ng

    # Compute per-axis fractional indices
    # axes and bounds stay as numpy scalars — fine since they're just numbers
    axes   = gp.axes
    dqs    = gp.dq_per_axis()

    lo_all   = xp.zeros((Np, D), dtype=xp.int64)
    frac_all = xp.zeros((Np, D), dtype=xp.float64)

    for d in range(D):
        fi = (Q_flat[:, d] - float(axes[d][0])) / float(dqs[d])
        fi = xp.clip(fi, 0.0, float(Ng) - 1.001)
        lo_all[:, d]   = fi.astype(xp.int64)
        frac_all[:, d] = fi - lo_all[:, d]

    result = xp.zeros(Np, dtype=xp.float64)

    for vertex in range(1 << D):
        weight = xp.ones(Np, dtype=xp.float64)
        flat   = xp.zeros(Np, dtype=xp.int64)
        stride = 1
        # Compute flat index and weight in reverse axis order (C layout)
        for d in range(D - 1, -1, -1):
            if (vertex >> d) & 1:
                idx_d  = xp.minimum(lo_all[:, d] + 1, Ng - 1)
                weight *= frac_all[:, d]
            else:
                idx_d  = lo_all[:, d]
                weight *= (1.0 - frac_all[:, d])
            flat  += idx_d * stride
            stride *= Ng
        result += weight * sqrt_rho_flat[flat]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Reference solvers  (1-D only, for validation)
# ══════════════════════════════════════════════════════════════════════════════

def schrodinger_fft_1d(psi0, V_func_1d, pp, gp, save_every=10):
    """Split-operator FFT reference solution in 1-D (via m5_fft_ref).
    V_func_1d: callable taking a 1-D array x, returning 1-D array V(x).
    """
    x = gp.axes[0]
    V_grid = V_func_1d(x)
    return _schrodinger_fft_1d(psi0, V_grid, x, gp.T, gp.Nt,
                               hbar=pp.hbar, mass=pp.m,
                               save_every=save_every)


def method4_1d(psi0, V_func, pp, gp, Np=10000, seed=42, save_every=10):
    """Method 4: explicit osmotic drift u = ν ∂_x ln ρ, for comparison."""
    from scipy.ndimage import gaussian_filter1d as gf1

    rng = np.random.default_rng(seed)
    hbar, m, nu, sig = pp.hbar, pp.m, pp.nu, pp.sigma_noise
    dx = gp.dq;  dt = gp.dt
    x  = gp.axes[0];  Ng = gp.Ng

    def Dx_1d(f): return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
    def D2x_1d(f): return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / dx**2

    rho0 = np.abs(psi0)**2
    cdf  = np.cumsum(rho0) * dx;  cdf /= cdf[-1]
    X    = np.interp(rng.uniform(size=Np), cdf, x)
    psi_at = np.interp(X, x, psi0.real) + 1j * np.interp(X, x, psi0.imag)
    Sp   = hbar * np.angle(psi_at)

    idx_save = list(range(0, gp.Nt + 1, save_every))
    Ns   = len(idx_save)
    X_h  = np.zeros((Ns, Np))
    rh_h = np.zeros((Ns, Ng))
    si   = 0
    rho_est = rho0 / (np.sum(rho0) * dx)

    if 0 in idx_save:
        X_h[si] = X.copy(); rh_h[si] = rho_est.copy(); si += 1

    bw = 4.0 * dx

    for n in range(1, gp.Nt + 1):
        h, _ = np.histogram(X, bins=Ng, range=(gp.qL, gp.qR))
        rho_est = gf1(h.astype(float), sigma=bw / dx)
        rho_est = np.maximum(rho_est, 1e-30)
        rho_est /= np.sum(rho_est) * dx

        ln_rho  = gf1(np.log(rho_est), sigma=2.0)
        u_field = nu * Dx_1d(ln_rho)

        S_field = np.zeros(Ng); cnt = np.zeros(Ng)
        bi = np.clip(((X - gp.qL) / dx).astype(int), 0, Ng - 1)
        np.add.at(S_field, bi, Sp)
        np.add.at(cnt, bi, 1.0)
        ok = cnt > 0
        S_field[ok] /= cnt[ok]
        if not np.all(ok):
            S_field[~ok] = np.interp(x[~ok], x[ok], S_field[ok])
        S_field = gf1(S_field, sigma=3.0)
        v_field = Dx_1d(S_field) / m

        vp    = v_field + u_field
        drift = np.interp(X, x, vp)
        v_at  = np.interp(X, x, v_field)
        V_at  = V_func(X)

        sqr   = np.sqrt(np.maximum(rho_est, 1e-30))
        Q_f   = -(hbar**2 / (2*m)) * D2x_1d(sqr) / np.maximum(sqr, 1e-30)
        Q_f   = gf1(Q_f, sigma=2.0)
        Q_at  = np.interp(X, x, Q_f)

        dW = rng.normal(size=Np) * np.sqrt(dt)
        X  = X + drift * dt + sig * dW
        X  = np.clip(X, gp.qL + dx, gp.qR - dx)
        Sp += (0.5 * m * v_at**2 - V_at - Q_at) * dt

        if n in idx_save and si < Ns:
            X_h[si] = X.copy(); rh_h[si] = rho_est.copy(); si += 1

    for j in range(si, Ns):
        X_h[j] = X.copy(); rh_h[j] = rho_est.copy()

    return dict(X=X_h, rho=rh_h, t_save=gp.t_arr[idx_save[:Ns]])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Helper: initialise from 1-D ψ₀ for D=1 test
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_wp(x, x0, p0, s0, hbar=1.):
    return ((2*np.pi*s0**2)**(-0.25)
            * np.exp(-(x-x0)**2 / (4*s0**2) + 1j*p0*x/hbar))


def cat_state_1d(x, x0, p0, s0, hbar=1.):
    return gaussian_wp(x, -x0, +p0, s0, hbar) + gaussian_wp(x, +x0, -p0, s0, hbar)


def init_from_psi1d(psi0, gp: GridParams, pp: PhysicsParams, Np: int, seed: int = 42):
    """
    Sample Np positions from |ψ₀|² and assign initial phases S₀ = ℏ·∠ψ₀.
    Returns Q0 (Np,1) and S0 (Np,).
    """
    rng  = np.random.default_rng(seed)
    x    = gp.axes[0]
    dx   = gp.dq
    rho0 = np.abs(psi0)**2
    cdf  = np.cumsum(rho0) * dx;  cdf /= cdf[-1]
    X    = np.interp(rng.uniform(size=Np), cdf, x)
    psi_at = np.interp(X, x, psi0.real) + 1j * np.interp(X, x, psi0.imag)
    Sp   = pp.hbar * np.angle(psi_at)
    return X[:, None].astype(np.float64), Sp.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Figures
# ══════════════════════════════════════════════════════════════════════════════

def _rho1d_from_result(result, gp, idx):
    """Extract 1-D density from ND result at snapshot idx (D=1 only)."""
    return result['rho'][idx, :].astype(np.float64)


def compare_figure_1d(x, ts, rho_ref, m4, m5_nd, gp, pp, label, Np, K):
    """Density comparison: FFT exact vs Method-4 vs Method-5 ND."""
    from scipy.ndimage import gaussian_filter1d as gf1
    Ns    = len(ts)
    snaps = [0, Ns//4, Ns//2, 3*Ns//4, Ns-1]
    dx    = gp.dq

    edges   = np.linspace(gp.qL, gp.qR, gp.Ng + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes_arr = plt.subplots(2, 3, figsize=(16, 9))

    def smooth_hist(X_arr, smooth=3.5):
        h, _ = np.histogram(X_arr, bins=edges, density=True)
        return gf1(np.interp(x, centers, h), sigma=smooth)

    for col, si in enumerate(snaps[:3]):
        ax = axes_arr[0, col]
        ax.plot(x, rho_ref[si], 'k-', lw=2, label='Schrödinger (exact)')
        ax.plot(x, smooth_hist(m4['X'][si]), '--', color='#2166ac', lw=1.5,
                label=f'M4: explicit u  (Np={Np})')
        rho5 = _rho1d_from_result(m5_nd, gp, si)
        ax.plot(x, rho5, '-.', color='#b2182b', lw=1.5,
                label=f'M5-ND: √ρ select  (Np={Np}, K={K})')
        ax.set_title(f't = {ts[si]:.2f}', fontsize=11)
        ax.set_xlim(-8, 8)
        if col == 0:
            ax.legend(fontsize=8)
            ax.set_ylabel('ρ(x,t)')

    ax_err = axes_arr[1, 0]
    err4   = np.zeros(Ns)
    err5   = np.zeros(Ns)
    for i in range(Ns):
        err4[i] = np.sqrt(np.sum((smooth_hist(m4['X'][i]) - rho_ref[i])**2) * dx)
        rho5    = _rho1d_from_result(m5_nd, gp, i)
        err5[i] = np.sqrt(np.sum((rho5 - rho_ref[i])**2) * dx)

    ax_err.semilogy(ts, err4, '-', color='#2166ac', lw=1.5,
                    label=f'M4 L² (mean={np.mean(err4):.4f})')
    ax_err.semilogy(ts, err5, '-', color='#b2182b', lw=1.5,
                    label=f'M5-ND L² (mean={np.mean(err5):.4f})')
    ax_err.set_xlabel('t');  ax_err.set_ylabel('‖ρ − ρ_ref‖₂')
    ax_err.set_title('L² density error vs time');  ax_err.legend(fontsize=9)

    ax2 = axes_arr[1, 1]
    for si in snaps[3:]:
        ax2.plot(x, rho_ref[si], 'k-', lw=2)
        ax2.plot(x, smooth_hist(m4['X'][si]), '--', color='#2166ac', lw=1.5)
        rho5 = _rho1d_from_result(m5_nd, gp, si)
        ax2.plot(x, rho5, '-.', color='#b2182b', lw=1.5)
    ax2.set_title(f'Late snapshots (t={ts[snaps[3]]:.2f}, {ts[snaps[4]]:.2f})')
    ax2.set_xlim(-8, 8)

    ax_txt = axes_arr[1, 2]
    ax_txt.axis('off')
    wt4 = m4.get('wall_time', 0);  wt5 = m5_nd.get('wall_time', 0)
    summary = (
        f"Method 4 (explicit osmotic):\n"
        f"  Np={Np}  mean L²={np.mean(err4):.5f}\n"
        f"  max  L²={np.max(err4):.5f}  t={wt4:.1f}s\n\n"
        f"Method 5-ND (√ρ selection):\n"
        f"  Np={Np}  K={K}\n"
        f"  mean L²={np.mean(err5):.5f}\n"
        f"  max  L²={np.max(err5):.5f}  t={wt5:.1f}s\n\n"
        f"Time ratio M5/M4 = {wt5/max(wt4,1e-6):.1f}×\n"
        f"Error ratio M5/M4 = {np.mean(err5)/max(np.mean(err4),1e-12):.2f}×"
    )
    ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f"Method 5-ND (√ρ selection) vs Method 4 — {label}",
                 fontsize=13, weight='bold')
    fig.tight_layout()
    return fig


def trajectory_figure_1d(x, ts_fft, rho_ref, m5_nd, gp, pp,
                          x0_cat, p0_cat, s0_cat, Np, K):
    """Trajectory visualisation (D=1 cat state): space-time heatmap + trajectories."""
    traj_Q = m5_nd['traj_Q']   # (Nt+1, N_track, 1)
    traj_t = m5_nd['traj_t']
    traj_X = traj_Q[:, :, 0]  # (Nt+1, N_track) — squeeze D=1

    N_track     = traj_X.shape[1]
    t_collision = x0_cat * pp.m / p0_cat
    Ns_fft      = len(ts_fft)

    x_init    = traj_X[0, :]
    left_ids  = np.where(x_init < 0)[0]
    right_ids = np.where(x_init >= 0)[0]
    t_thin    = max(1, len(traj_t) // 600)

    fig = plt.figure(figsize=(18, 13))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30)

    # ─── (a) Full space-time heatmap ────────────────────────────────
    ax_st = fig.add_subplot(gs[0, :2])
    rho_img = np.array(rho_ref)
    ax_st.imshow(rho_img.T, origin='lower', aspect='auto',
                 extent=[0, gp.T, gp.qL, gp.qR],
                 cmap='inferno', interpolation='bilinear',
                 vmin=0, vmax=np.percentile(rho_img, 99))

    n_show  = min(N_track, 60)
    n_L = min(n_show // 2, len(left_ids))
    n_R = min(n_show // 2, len(right_ids))
    show_L = left_ids [np.linspace(0, len(left_ids)  - 1, n_L, dtype=int)]
    show_R = right_ids[np.linspace(0, len(right_ids) - 1, n_R, dtype=int)]

    for idx in show_L:
        ax_st.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                   '-', color='#66ccff', lw=0.4, alpha=0.7)
    for idx in show_R:
        ax_st.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                   '-', color='#ff9966', lw=0.4, alpha=0.7)

    ax_st.axvline(t_collision, color='#00ff88', ls='--', lw=1.0, alpha=0.8,
                  label=f't_c≈{t_collision:.2f}')
    ax_st.set_xlabel('t', fontsize=11);  ax_st.set_ylabel('x', fontsize=11)
    ax_st.set_ylim(-8, 8)
    ax_st.set_title('ρ(x,t) + Method-5-ND trajectories  '
                    '(blue=left, orange=right)', fontsize=11)
    ax_st.legend(fontsize=9, loc='upper right')

    # ─── (b) Zoom on collision region ───────────────────────────────
    ax_zoom = fig.add_subplot(gs[0, 2])
    ax_zoom.imshow(rho_img.T, origin='lower', aspect='auto',
                   extent=[0, gp.T, gp.qL, gp.qR],
                   cmap='inferno', interpolation='bilinear',
                   vmin=0, vmax=np.percentile(rho_img, 99))
    for idx in show_L:
        ax_zoom.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color='#66ccff', lw=0.6, alpha=0.8)
    for idx in show_R:
        ax_zoom.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color='#ff9966', lw=0.6, alpha=0.8)
    t_lo = max(0, t_collision - 0.8);  t_hi = min(gp.T, t_collision + 0.8)
    ax_zoom.set_xlim(t_lo, t_hi);  ax_zoom.set_ylim(-4, 4)
    ax_zoom.set_xlabel('t', fontsize=11);  ax_zoom.set_ylabel('x', fontsize=11)
    ax_zoom.set_title('Zoom: collision\n(non-crossing)', fontsize=11)

    # ─── (c) Density snapshots with nodes ───────────────────────────
    snap_ts = [0, t_collision*0.5, t_collision*0.85,
               t_collision, t_collision*1.15, t_collision*1.5]
    snap_ts = [t for t in snap_ts if t <= gp.T]
    cmap_s  = plt.cm.viridis
    colors_s = [cmap_s(i / max(len(snap_ts)-1, 1)) for i in range(len(snap_ts))]

    ax_snap = fig.add_subplot(gs[1, :2])
    for i_s, t_s in enumerate(snap_ts):
        i_fft = np.argmin(np.abs(ts_fft - t_s))
        rho_s = rho_ref[i_fft]
        ax_snap.plot(x, rho_s, '-', color=colors_s[i_s], lw=1.8,
                     label=f't={ts_fft[i_fft]:.2f}')
        if ts_fft[i_fft] > t_collision * 0.6:
            from scipy.ndimage import gaussian_filter1d as gf1
            rs = gf1(rho_s, sigma=2)
            thr = 0.005 * rs.max()
            lm  = np.where((rs[1:-1] < rs[:-2]) & (rs[1:-1] < rs[2:]) &
                           (rs[1:-1] < thr))[0] + 1
            xn  = x[lm];  xn = xn[(xn > -6) & (xn < 6)]
            if len(xn):
                ax_snap.plot(xn, np.zeros_like(xn), 'v',
                             color=colors_s[i_s], ms=6, alpha=0.7)
    ax_snap.set_xlim(-7, 7);  ax_snap.set_xlabel('x', fontsize=11)
    ax_snap.set_ylabel('ρ(x,t)', fontsize=11)
    ax_snap.set_title('Density evolution: fringes & nodes  (▼=nodes)', fontsize=11)
    ax_snap.legend(fontsize=8, ncol=3, loc='upper right')

    # ─── (d) M5-ND density at collision ─────────────────────────────
    ax_coll = fig.add_subplot(gs[1, 2])
    from scipy.ndimage import gaussian_filter1d as gf1
    i_c_fft = np.argmin(np.abs(ts_fft - t_collision))
    i_c_nd  = np.argmin(np.abs(m5_nd['t_save'] - t_collision))
    ax_coll.plot(x, rho_ref[i_c_fft], 'k-', lw=2.5, label='Exact')
    rho5_c = _rho1d_from_result(m5_nd, gp, i_c_nd)
    ax_coll.plot(x, rho5_c, '-', color='#b2182b', lw=1.5,
                 label=f'M5-ND (Np={Np},K={K})')
    ax_coll.fill_between(x, rho_ref[i_c_fft], alpha=0.15, color='gray')
    ax_coll.set_xlim(-6, 6);  ax_coll.set_xlabel('x', fontsize=11)
    ax_coll.set_ylabel('ρ(x,t)', fontsize=11)
    ax_coll.set_title(f'Density at collision (t≈{ts_fft[i_c_fft]:.2f})', fontsize=11)
    ax_coll.legend(fontsize=9)

    # ─── (e) Trajectory "spaghetti" ─────────────────────────────────
    ax_traj = fig.add_subplot(gs[2, :2])
    T_g, X_g = np.meshgrid(ts_fft, x)
    ax_traj.contourf(T_g, X_g, rho_img.T, levels=30, cmap='Greys', alpha=0.35)
    for idx in show_L:
        ax_traj.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color='#0077cc', lw=0.7, alpha=0.75)
    for idx in show_R:
        ax_traj.plot(traj_t[::t_thin], traj_X[::t_thin, idx],
                     '-', color='#cc4400', lw=0.7, alpha=0.75)
    ax_traj.axvline(t_collision, color='#00aa44', ls='--', lw=1.2, alpha=0.7)
    ax_traj.set_ylim(-7, 7);  ax_traj.set_xlabel('t', fontsize=11)
    ax_traj.set_ylabel('x', fontsize=11)
    ax_traj.set_title('Sample trajectories over ρ contours — non-crossing', fontsize=11)

    # ─── (f) Minimum approach distances ─────────────────────────────
    ax_node = fig.add_subplot(gs[2, 2])
    left_s  = show_L[np.argsort(traj_X[0, show_L])]
    right_s = show_R[np.argsort(traj_X[0, show_R])]
    if len(left_s) > 0 and len(right_s) > 0:
        n_pairs = min(5, len(left_s), len(right_s))
        pair_d  = np.zeros((len(traj_t), n_pairs))
        for ip in range(n_pairs):
            il = left_s[-(ip+1)];  ir = right_s[ip]
            pair_d[:, ip] = np.abs(traj_X[:, il] - traj_X[:, ir])
        for ip in range(n_pairs):
            ax_node.semilogy(traj_t[::t_thin*2], pair_d[::t_thin*2, ip],
                             '-', lw=1.0, alpha=0.7,
                             label=f'pair {ip+1}' if ip < 3 else None)
        ax_node.axvline(t_collision, color='#00aa44', ls='--', lw=1.0, alpha=0.7,
                        label=f't_c≈{t_collision:.2f}')
        ax_node.set_xlabel('t', fontsize=11);  ax_node.set_ylabel('|x_L−x_R|', fontsize=11)
        ax_node.set_title('Min approach distance\n(left–right pairs)', fontsize=11)
        ax_node.legend(fontsize=8);  ax_node.grid(True, alpha=0.3)

    fig.suptitle(
        f'Cat State — Method 5-ND  (x₀=±{x0_cat}, p₀=±{p0_cat}, σ₀={s0_cat})'
        f'  Np={Np}, K={K}',
        fontsize=13, weight='bold', y=0.98)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main: 1-D cat-state validation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Method 5 ND solver — cat state test')
    parser.add_argument('--gpu',   action='store_true',  help='Use CuPy GPU backend')
    parser.add_argument('--Np',    type=int, default=8000, help='Number of particles')
    parser.add_argument('--K',     type=int, default=48,   help='Candidates per particle')
    parser.add_argument('--batch', type=int, default=2000, help='Batch size for selection')
    parser.add_argument('--Ntrack',type=int, default=120,  help='Trajectories to track')
    args = parser.parse_args()

    USE_GPU    = args.gpu
    Np         = args.Np
    K          = args.K
    BATCH      = args.batch
    N_track    = args.Ntrack

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  Method 5-ND: √ρ-Selection Solver — General Configuration    ║
    ║  Space  (D dimensions, multi-particle / multi-dimensional)    ║
    ║  TEST: 1-D cat state — reproduce method5_catstate.py output   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    xp = get_xp(USE_GPU)
    backend = 'CuPy (GPU)' if (xp is not np) else 'NumPy (CPU)'
    print(f"  Backend  : {backend}")
    print(f"  Np={Np}  K={K}  batch_size={BATCH}  N_track={N_track}")
    bar = "═" * 68

    # ── Physics & grid (identical to method5_catstate.py) ──────────
    pp = PhysicsParams(hbar=1., m=1.)

    x0_cat = 4.0;  p0_cat = 3.0;  s0_cat = 0.7
    t_collision = x0_cat * pp.m / p0_cat

    gp = GridParams(D=1, qL=-15., qR=15., Ng=512, T=3.0, Nt=3000)
    se = 6    # save_every

    x    = gp.axes[0]
    psi0 = cat_state_1d(x, x0_cat, p0_cat, s0_cat, pp.hbar)
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * gp.dq)

    V_free    = lambda Q: np.zeros(Q.shape[0])   # Q is (Np, 1) for ND solver
    V_free_1d = lambda x: np.zeros_like(x)        # x is 1-D grid for FFT/M4

    print(f"\n{bar}")
    print(f"  Cat State: x₀=±{x0_cat}  p₀=±{p0_cat}  σ₀={s0_cat}")
    print(f"  Expected collision: t_c ≈ {t_collision:.2f}")
    print(bar)

    # ── Reference: Schrödinger FFT ──────────────────────────────────
    t0 = time.time()
    print("  [1] Schrödinger FFT … ", end='', flush=True)
    psi_h, ts_fft = schrodinger_fft_1d(psi0, V_free_1d, pp, gp, save_every=se)
    rho_ref = np.abs(psi_h)**2
    print(f"({time.time()-t0:.1f}s)")

    # ── Method 4: explicit osmotic ──────────────────────────────────
    t0 = time.time()
    print(f"  [4] Explicit osmotic (Np={Np}) … ", end='', flush=True)
    m4 = method4_1d(psi0, V_free_1d, pp, gp, Np=Np, save_every=se)
    m4['wall_time'] = time.time() - t0
    print(f"({m4['wall_time']:.1f}s)")

    # ── Method 5-ND: generalised D=1 ────────────────────────────────
    # Choose tracked particle indices (same logic as catstate.py)
    rng_ref  = np.random.default_rng(42)
    rho0_1d  = np.abs(psi0)**2
    cdf0     = np.cumsum(rho0_1d) * gp.dq;  cdf0 /= cdf0[-1]
    X_ref    = np.interp(rng_ref.uniform(size=Np), cdf0, x)
    sorted_i = np.argsort(X_ref)
    track_ids = sorted_i[np.linspace(0, Np-1, N_track, dtype=int)]

    Q0, S0 = init_from_psi1d(psi0, gp, pp, Np, seed=42)

    t0 = time.time()
    print(f"  [5-ND] √ρ-selection D=1 (Np={Np}, K={K}, batch={BATCH},"
          f" track={N_track}) …")
    m5_nd = method5_nd(
        Q0, S0,
        V_func      = V_free,
        pp          = pp,
        gp          = gp,
        K           = K,
        batch_size  = BATCH,
        save_every  = se,
        track_ids   = track_ids,
        use_gpu     = USE_GPU,
        smooth_density = 4.0,
        smooth_phase   = 3.0,
        smooth_Q       = 2.0,
        seed        = 42,
    )
    m5_nd['wall_time'] = time.time() - t0
    print(f"  → done in {m5_nd['wall_time']:.1f}s")

    # ── Quick L² error summary ───────────────────────────────────────
    from scipy.ndimage import gaussian_filter1d as gf1
    Ns     = len(ts_fft)
    dx     = gp.dq
    edges  = np.linspace(gp.qL, gp.qR, gp.Ng + 1)
    cents  = 0.5 * (edges[:-1] + edges[1:])

    err4 = np.zeros(Ns);  err5 = np.zeros(Ns)
    for i in range(Ns):
        h4, _  = np.histogram(m4['X'][i], bins=edges, density=True)
        rho4   = gf1(np.interp(x, cents, h4), sigma=3.5)
        err4[i] = np.sqrt(np.sum((rho4 - rho_ref[i])**2) * dx)
        rho5   = _rho1d_from_result(m5_nd, gp, i)
        err5[i] = np.sqrt(np.sum((rho5 - rho_ref[i])**2) * dx)

    print(f"\n  L² error summary:")
    print(f"    Method 4   mean={np.mean(err4):.5f}  max={np.max(err4):.5f}")
    print(f"    Method 5-ND mean={np.mean(err5):.5f}  max={np.max(err5):.5f}")
    print(f"    Error ratio M5-ND / M4 = {np.mean(err5)/max(np.mean(err4),1e-12):.2f}×")
    print(f"    Time  ratio M5-ND / M4 = {m5_nd['wall_time']/max(m4['wall_time'],1e-6):.1f}×")

    # ── Figures ─────────────────────────────────────────────────────
    print(f"\n  Generating comparison figure …")
    fig1 = compare_figure_1d(
        x, ts_fft, rho_ref, m4, m5_nd, gp, pp,
        "D: Cat state (colliding Gaussians) — 1-D ND test", Np, K)
    fig1.savefig(output_path("m5nd_caseD_comparison.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    print(f"  Generating trajectory figure …")
    fig2 = trajectory_figure_1d(
        x, ts_fft, rho_ref, m5_nd, gp, pp,
        x0_cat, p0_cat, s0_cat, Np, K)
    fig2.savefig(output_path("m5nd_caseD_trajectories.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"\n{bar}")
    print("  Done — Method 5-ND cat-state validation complete.")
    print(bar)


if __name__ == "__main__":
    main()
