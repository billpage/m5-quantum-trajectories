"""
m5/fft_ref.py — Split-operator FFT reference solvers for the Schrödinger equation.

Provides exact (up to time-step error) propagation on a uniform grid via
the standard split-operator method:

    ψ(t+dt) = e^{-iV dt/2ℏ}  FFT⁻¹[ e^{-iℏk²dt/2m} FFT[ e^{-iV dt/2ℏ} ψ(t) ] ]

Both 1-D and 2-D solvers are provided.  All computation uses NumPy (CPU)
regardless of whether CuPy is available in the caller.

Usage
-----
    from m5.fft_ref import schrodinger_fft_1d, schrodinger_fft_2d

    psi_snaps, t_snaps = schrodinger_fft_1d(
        psi0, V_grid, x, T, Nt, hbar=1.0, mass=1.0, save_every=10
    )
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1-D split-operator FFT
# ═══════════════════════════════════════════════════════════════════════

def schrodinger_fft_1d(psi0, V_grid, x, T, Nt,
                       hbar=1.0, mass=1.0, save_every=10):
    """Split-operator FFT propagation of the 1-D Schrödinger equation.

    Parameters
    ----------
    psi0 : array (Nx,)
        Initial wavefunction on the grid *x*.
    V_grid : array (Nx,)
        Potential evaluated on the grid *x*.  Pass ``np.zeros_like(x)``
        for a free particle.
    x : array (Nx,)
        Uniform spatial grid (periodic boundary conditions assumed).
    T : float
        Total propagation time.
    Nt : int
        Number of time steps.
    hbar : float, optional
        Reduced Planck constant (default 1.0).
    mass : float, optional
        Particle mass (default 1.0).
    save_every : int, optional
        Save a snapshot every *save_every* time steps (default 10).

    Returns
    -------
    psi_snaps : ndarray (Ns, Nx), complex
        Wavefunction snapshots.
    t_snaps : ndarray (Ns,)
        Times corresponding to each snapshot.
    """
    Nx = len(x)
    dx = x[1] - x[0]
    dt = T / Nt

    k = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    half_V = np.exp(-1j * V_grid * dt / (2 * hbar))
    T_k    = np.exp(-1j * hbar * k**2 * dt / (2 * mass))

    psi = psi0.astype(np.complex128).copy()

    idx_save = list(range(0, Nt + 1, save_every))
    Ns = len(idx_save)
    psi_h = np.zeros((Ns, Nx), dtype=np.complex128)
    t_arr = np.linspace(0, T, Nt + 1)

    si = 0
    if 0 in idx_save:
        psi_h[si] = psi.copy(); si += 1

    for n in range(1, Nt + 1):
        psi = half_V * psi
        psi = np.fft.ifft(T_k * np.fft.fft(psi))
        psi = half_V * psi
        if n in idx_save and si < Ns:
            psi_h[si] = psi.copy(); si += 1

    # Safety: fill any remaining slots with the final state
    for j in range(si, Ns):
        psi_h[j] = psi.copy()

    return psi_h, t_arr[idx_save[:Ns]]


# ═══════════════════════════════════════════════════════════════════════
# 2-D split-operator FFT (square grid)
# ═══════════════════════════════════════════════════════════════════════

def schrodinger_fft_2d(psi0, V_grid, x, T, Nt,
                       hbar=1.0, mass=1.0, save_every=10):
    """Split-operator FFT propagation of the 2-D Schrödinger equation.

    Parameters
    ----------
    psi0 : array (Ng, Ng)
        Initial wavefunction on a square grid.
    V_grid : array (Ng, Ng)
        Potential on the same grid.
    x : array (Ng,)
        One axis of the square grid (same for both dimensions).
    T : float
        Total propagation time.
    Nt : int
        Number of time steps.
    hbar : float, optional
        Reduced Planck constant (default 1.0).
    mass : float, optional
        Particle mass (default 1.0).
    save_every : int, optional
        Save a snapshot every *save_every* time steps (default 10).

    Returns
    -------
    psi_snaps : ndarray (Ns, Ng, Ng), complex
        Wavefunction snapshots.
    t_snaps : ndarray (Ns,)
        Times corresponding to each snapshot.
    """
    Ng = len(x)
    dx = x[1] - x[0]
    dt = T / Nt

    kx = 2 * np.pi * np.fft.fftfreq(Ng, dx)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')

    half_V = np.exp(-1j * V_grid * dt / (2 * hbar))
    T_k    = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * mass))

    psi = psi0.astype(np.complex128).copy()

    idx_save = list(range(0, Nt + 1, save_every))
    Ns = len(idx_save)
    psi_h = np.zeros((Ns, Ng, Ng), dtype=np.complex128)
    t_arr = np.linspace(0, T, Nt + 1)

    si = 0
    if 0 in idx_save:
        psi_h[si] = psi.copy(); si += 1

    for n in range(1, Nt + 1):
        psi = half_V * psi
        psi = np.fft.ifft2(T_k * np.fft.fft2(psi))
        psi = half_V * psi
        if n in idx_save and si < Ns:
            psi_h[si] = psi.copy(); si += 1

    for j in range(si, Ns):
        psi_h[j] = psi.copy()

    return psi_h, t_arr[idx_save[:Ns]]
