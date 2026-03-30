"""
test_m5_fft_ref.py — Verify m5/fft_ref.py split-operator FFT reference solvers.

Tests:
  1. Snapshot bookkeeping: correct counts and times
  2. Norm conservation (1-D and 2-D): ‖ψ(t)‖² = ‖ψ₀‖² to machine precision
  3. Free-particle spreading: peak position and width match analytic solution
  4. HO ground state stationarity: |ψ(t)|² unchanged, global phase rotates
  5. HO coherent state: centre of mass oscillates classically
  6. Energy conservation: <H> is constant (up to split-operator O(dt²) error)
  7. Time reversal: forward + backward propagation recovers ψ₀
  8. 2-D norm conservation
  9. 2-D HO ground state stationarity
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from m5.fft_ref import schrodinger_fft_1d, schrodinger_fft_2d
from m5.init import gaussian_wp, ho_ground_state
from m5.utils import output_path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR = 1.0
MASS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_sq(psi, dx):
    """Compute ‖ψ‖² = ∫|ψ|² dx on a uniform grid."""
    return np.sum(np.abs(psi)**2) * dx


def _expectation_x(psi, x, dx):
    """Compute <x> = ∫ x|ψ|² dx."""
    return np.sum(x * np.abs(psi)**2) * dx


def _expectation_x2(psi, x, dx):
    """Compute <x²> = ∫ x²|ψ|² dx."""
    return np.sum(x**2 * np.abs(psi)**2) * dx


def _kinetic_energy_k(psi, k, dx, hbar=HBAR, mass=MASS):
    """Kinetic energy via momentum-space: <T> = ∫ (ℏk)²/(2m) |ψ̂(k)|² dk."""
    psi_k = np.fft.fft(psi) * dx                   # ψ̂(k) with correct units
    dk = k[1] - k[0] if len(k) > 1 else 1.0
    # For FFT grids dk = 2π/(N dx), and Parseval: ∫|ψ̂|²dk = ∫|ψ|²dx
    return np.real(np.sum(hbar**2 * k**2 / (2*mass) * np.abs(psi_k)**2) * dk / (2*np.pi))


def _total_energy_1d(psi, V, x, hbar=HBAR, mass=MASS):
    """Total energy <H> = <T> + <V> for a 1-D wavefunction."""
    dx = x[1] - x[0]
    Nx = len(x)
    k = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    T_val = _kinetic_energy_k(psi, k, dx, hbar, mass)
    V_val = np.real(np.sum(V * np.abs(psi)**2) * dx)
    return T_val + V_val


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Snapshot bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_bookkeeping():
    """Verify snapshot count and times are correct for various save_every."""
    Nx = 64
    x = np.linspace(-10, 10, Nx, endpoint=False)
    psi0 = gaussian_wp(x, 0, 0, 1.0)
    V = np.zeros(Nx)
    T, Nt = 1.0, 100

    for save_every in [1, 10, 25, 50, 100]:
        psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt,
                                                 save_every=save_every)
        expected_indices = list(range(0, Nt + 1, save_every))
        expected_Ns = len(expected_indices)
        expected_times = np.linspace(0, T, Nt + 1)[expected_indices]

        assert psi_snaps.shape[0] == expected_Ns, \
            f"save_every={save_every}: got {psi_snaps.shape[0]} snapshots, " \
            f"expected {expected_Ns}"
        assert len(t_snaps) == expected_Ns, \
            f"save_every={save_every}: got {len(t_snaps)} times, " \
            f"expected {expected_Ns}"
        np.testing.assert_allclose(t_snaps, expected_times, atol=1e-14,
            err_msg=f"save_every={save_every}: snapshot times incorrect")

        # First snapshot should be the initial state
        np.testing.assert_allclose(psi_snaps[0], psi0, atol=1e-14,
            err_msg=f"save_every={save_every}: first snapshot ≠ ψ₀")

    print("[PASS] Snapshot bookkeeping verified for all save_every values.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: 1-D norm conservation
# ─────────────────────────────────────────────────────────────────────────────

def test_1d_norm_conservation():
    """Split-operator FFT is unitary: ‖ψ(t)‖² should be constant."""
    Nx = 256
    x = np.linspace(-15, 15, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0

    # Test with non-trivial potential (HO) and moving wavepacket
    psi0 = gaussian_wp(x, x0=-2.0, p0=3.0, sigma=1.0)
    V = 0.5 * MASS * omega**2 * x**2

    T, Nt = 5.0, 2000
    psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=100)

    norm0 = _norm_sq(psi0, dx)
    norms = np.array([_norm_sq(psi_snaps[i], dx) for i in range(len(t_snaps))])
    max_drift = np.max(np.abs(norms - norm0))

    print(f"  1-D norm: initial={norm0:.12f}  max drift={max_drift:.2e}")
    assert max_drift < 1e-12, \
        f"Norm drift {max_drift:.2e} exceeds machine-precision threshold"
    print("[PASS] 1-D norm conservation verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Free-particle Gaussian spreading
# ─────────────────────────────────────────────────────────────────────────────

def test_free_particle_spreading():
    """Free Gaussian: peak moves at v=p₀/m, width broadens analytically."""
    Nx = 512
    x = np.linspace(-40, 40, Nx, endpoint=False)
    dx = x[1] - x[0]

    sigma0, p0, x0 = 2.0, 2.0, 0.0
    psi0 = gaussian_wp(x, x0, p0, sigma0)
    V = np.zeros(Nx)

    T = 5.0
    Nt = 4000
    psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=400)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_peaks_num = []
    sigmas_num = []

    for i, t in enumerate(t_snaps):
        psi = psi_snaps[i]
        rho = np.abs(psi)**2
        xmean = _expectation_x(psi, x, dx) / _norm_sq(psi, dx)
        x2mean = _expectation_x2(psi, x, dx) / _norm_sq(psi, dx)
        sigma_t = np.sqrt(x2mean - xmean**2)
        x_peaks_num.append(xmean)
        sigmas_num.append(sigma_t)

    x_peaks_num = np.array(x_peaks_num)
    sigmas_num = np.array(sigmas_num)

    # Analytic solutions
    v = p0 / MASS
    x_peaks_exact = x0 + v * t_snaps
    sigmas_exact = sigma0 * np.sqrt(1 + (HBAR * t_snaps / (2 * MASS * sigma0**2))**2)

    # Plot
    axes[0].plot(t_snaps, x_peaks_exact, 'k--', lw=1.5, label='exact ⟨x⟩')
    axes[0].plot(t_snaps, x_peaks_num, 'ro', ms=5, label='FFT ⟨x⟩')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('⟨x⟩')
    axes[0].set_title('Peak position'); axes[0].legend(fontsize=8)

    axes[1].plot(t_snaps, sigmas_exact, 'k--', lw=1.5, label='exact σ(t)')
    axes[1].plot(t_snaps, sigmas_num, 'ro', ms=5, label='FFT σ(t)')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('σ')
    axes[1].set_title('Width spreading'); axes[1].legend(fontsize=8)

    fig.suptitle('Free particle: Gaussian spreading', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path('test_fft_free_particle.png'), dpi=120)
    plt.close(fig)

    # Assertions
    err_x = np.max(np.abs(x_peaks_num - x_peaks_exact))
    err_sigma = np.max(np.abs(sigmas_num - sigmas_exact))
    print(f"  Free particle: ⟨x⟩ max error={err_x:.2e}  "
          f"σ(t) max error={err_sigma:.2e}")
    assert err_x < 1e-8, f"Peak position error {err_x:.2e} too large"
    assert err_sigma < 1e-6, f"Width error {err_sigma:.2e} too large"
    print("[PASS] Free-particle spreading matches analytic solution.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: HO ground state stationarity
# ─────────────────────────────────────────────────────────────────────────────

def test_ho_ground_state_stationary():
    """HO ground state: |ψ(t)|² constant, phase rotates at −ω/2."""
    Nx = 256
    x = np.linspace(-8, 8, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0

    psi0 = ho_ground_state(x, omega=omega, mass=MASS, hbar=HBAR)
    V = 0.5 * MASS * omega**2 * x**2

    T = 4 * np.pi     # two full oscillation periods
    Nt = 8000
    psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=400)

    rho0 = np.abs(psi0)**2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    density_errors = []
    phase_errors = []

    for i, t in enumerate(t_snaps):
        psi = psi_snaps[i]
        rho = np.abs(psi)**2

        # Density should be unchanged
        rho_err = np.max(np.abs(rho - rho0))
        density_errors.append(rho_err)

        # Global phase should be e^{-i E₀ t/ℏ} = e^{-i ω t/2}
        # Extract global phase from the ratio ψ(t)/ψ₀ at the peak
        if t > 0:
            # Use central region where ψ₀ is large to avoid numerical noise
            mask = np.abs(psi0) > 0.1 * np.max(np.abs(psi0))
            ratio = psi[mask] / psi0[mask]
            # Should be a constant phase factor
            expected_phase = -omega * t / 2
            measured_phase = np.angle(np.mean(ratio))
            # Wrap both to [-π, π] before comparing
            phase_diff = np.angle(np.exp(1j * (measured_phase - expected_phase)))
            phase_errors.append(np.abs(phase_diff))

    density_errors = np.array(density_errors)
    phase_errors = np.array(phase_errors)

    axes[0].semilogy(t_snaps, density_errors, 'b.-')
    axes[0].set_xlabel('t'); axes[0].set_ylabel('max |ρ(t) − ρ₀|')
    axes[0].set_title('Density stationarity')

    axes[1].semilogy(t_snaps[1:], phase_errors, 'r.-')
    axes[1].set_xlabel('t'); axes[1].set_ylabel('|phase error|')
    axes[1].set_title('Phase rotation −ωt/2')

    fig.suptitle('HO ground state stationarity', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path('test_fft_ho_stationary.png'), dpi=120)
    plt.close(fig)

    max_rho_err = np.max(density_errors)
    max_phase_err = np.max(phase_errors)
    print(f"  HO ground state: max density error={max_rho_err:.2e}  "
          f"max phase error={max_phase_err:.2e}")
    # Split-operator global error is O(dt²); with dt ≈ 1.6e-3 over 8000
    # steps the density error for an eigenstate is typically ~1e-7.
    assert max_rho_err < 1e-5, \
        f"Density drift {max_rho_err:.2e} too large for HO ground state"
    assert max_phase_err < 1e-5, \
        f"Phase rotation error {max_phase_err:.2e} too large"
    print("[PASS] HO ground state stationarity verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: HO coherent state oscillation
# ─────────────────────────────────────────────────────────────────────────────

def test_ho_coherent_state():
    """Coherent state in HO: ⟨x(t)⟩ = x₀ cos(ωt) + (p₀/mω) sin(ωt)."""
    Nx = 512
    x = np.linspace(-15, 15, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0

    # Coherent state = displaced ground state
    x0, p0, sigma0 = 3.0, 0.0, 1.0 / np.sqrt(omega * MASS / HBAR)
    psi0 = gaussian_wp(x, x0, p0, sigma0)
    V = 0.5 * MASS * omega**2 * x**2

    T_period = 2 * np.pi / omega
    T = 2 * T_period
    Nt = 8000
    psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=80)

    x_means = np.array([
        _expectation_x(psi_snaps[i], x, dx) / _norm_sq(psi_snaps[i], dx)
        for i in range(len(t_snaps))
    ])
    x_exact = x0 * np.cos(omega * t_snaps) + (p0 / (MASS * omega)) * np.sin(omega * t_snaps)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_snaps, x_exact, 'k--', lw=1.5, label='exact ⟨x⟩')
    ax.plot(t_snaps, x_means, 'ro', ms=3, label='FFT ⟨x⟩')
    ax.set_xlabel('t'); ax.set_ylabel('⟨x⟩')
    ax.set_title('Coherent state in HO: centre-of-mass oscillation')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path('test_fft_ho_coherent.png'), dpi=120)
    plt.close(fig)

    err = np.max(np.abs(x_means - x_exact))
    print(f"  Coherent state: ⟨x⟩ max error = {err:.2e}")
    assert err < 1e-4, f"Coherent state ⟨x⟩ error {err:.2e} too large"
    print("[PASS] HO coherent state oscillation verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Energy conservation
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_conservation():
    """Total energy <H> = <T> + <V> should be conserved for time-independent V."""
    Nx = 256
    x = np.linspace(-15, 15, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0

    psi0 = gaussian_wp(x, x0=-2.0, p0=2.0, sigma=1.0)
    V = 0.5 * MASS * omega**2 * x**2

    T, Nt = 4.0, 2000
    psi_snaps, t_snaps = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=100)

    energies = np.array([
        _total_energy_1d(psi_snaps[i], V, x) for i in range(len(t_snaps))
    ])

    E0 = energies[0]
    max_drift = np.max(np.abs(energies - E0))
    rel_drift = max_drift / np.abs(E0)

    print(f"  Energy: E₀={E0:.6f}  max drift={max_drift:.2e}  "
          f"relative={rel_drift:.2e}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_snaps, (energies - E0) / np.abs(E0), 'b.-')
    ax.set_xlabel('t'); ax.set_ylabel('(E(t) − E₀) / |E₀|')
    ax.set_title('Energy conservation (split-operator FFT)')
    ax.axhline(0, color='k', lw=0.5)
    fig.tight_layout()
    fig.savefig(output_path('test_fft_energy.png'), dpi=120)
    plt.close(fig)

    # Split-operator has O(dt²) error per step, O(dt²) accumulated for
    # energy.  With dt = 2e-3, expect ~few×10⁻⁶ relative drift.
    assert rel_drift < 1e-4, \
        f"Relative energy drift {rel_drift:.2e} too large"
    print("[PASS] Energy conservation verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Time reversal
# ─────────────────────────────────────────────────────────────────────────────

def test_time_reversal():
    """Propagate forward then backward: should recover ψ₀."""
    Nx = 256
    x = np.linspace(-10, 10, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0

    psi0 = gaussian_wp(x, x0=1.5, p0=2.0, sigma=0.8)
    V = 0.5 * MASS * omega**2 * x**2

    T, Nt = 3.0, 3000

    # Forward propagation: save only the final state
    psi_fwd, _ = schrodinger_fft_1d(psi0, V, x, T, Nt, save_every=Nt)
    psi_final = psi_fwd[-1]

    # Backward propagation: same potential but negative time step.
    # The split-operator with T → −T is equivalent to propagating backward.
    psi_bwd, _ = schrodinger_fft_1d(psi_final, V, x, T, Nt,
                                     save_every=Nt, hbar=-HBAR)
    psi_recovered = psi_bwd[-1]

    # Compare
    err = np.sqrt(np.sum(np.abs(psi_recovered - psi0)**2) * dx)
    print(f"  Time reversal: L² recovery error = {err:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, np.abs(psi0)**2, 'k-', lw=1.5, label='ψ₀')
    axes[0].plot(x, np.abs(psi_recovered)**2, 'r--', lw=1, label='recovered')
    axes[0].set_title('|ψ|²'); axes[0].legend(fontsize=8)

    axes[1].plot(x, psi0.real, 'k-', lw=1.5, label='Re(ψ₀)')
    axes[1].plot(x, psi_recovered.real, 'r--', lw=1, label='Re(recovered)')
    axes[1].set_title('Re(ψ)'); axes[1].legend(fontsize=8)

    fig.suptitle(f'Time reversal: L² error = {err:.2e}', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path('test_fft_time_reversal.png'), dpi=120)
    plt.close(fig)

    assert err < 1e-6, f"Time-reversal error {err:.2e} too large"
    print("[PASS] Time reversal verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: 2-D norm conservation
# ─────────────────────────────────────────────────────────────────────────────

def test_2d_norm_conservation():
    """2-D split-operator FFT preserves norm."""
    Ng = 64
    x = np.linspace(-8, 8, Ng, endpoint=False)
    dx = x[1] - x[0]
    X2, Y2 = np.meshgrid(x, x, indexing='ij')

    psi0 = gaussian_wp(X2, -1.0, 1.5, 1.0) * gaussian_wp(Y2, 0.5, -1.0, 0.8)
    norm0_2d = np.sum(np.abs(psi0)**2) * dx**2
    psi0 /= np.sqrt(norm0_2d)       # normalise

    omega = 1.0
    V = 0.5 * MASS * omega**2 * (X2**2 + Y2**2)

    T, Nt = 3.0, 1500
    psi_snaps, t_snaps = schrodinger_fft_2d(psi0, V, x, T, Nt, save_every=150)

    norms = np.array([np.sum(np.abs(psi_snaps[i])**2) * dx**2
                       for i in range(len(t_snaps))])
    max_drift = np.max(np.abs(norms - 1.0))

    print(f"  2-D norm: max drift = {max_drift:.2e}")
    assert max_drift < 1e-12, \
        f"2-D norm drift {max_drift:.2e} exceeds threshold"
    print("[PASS] 2-D norm conservation verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: 2-D HO ground state stationarity
# ─────────────────────────────────────────────────────────────────────────────

def test_2d_ho_stationary():
    """2-D isotropic HO ground state: |ψ(t)|² should be unchanged."""
    Ng = 64
    x = np.linspace(-8, 8, Ng, endpoint=False)
    dx = x[1] - x[0]
    X2, Y2 = np.meshgrid(x, x, indexing='ij')
    omega = 1.0

    # Separable ground state: ψ₀(x,y) = φ₀(x)·φ₀(y)
    psi0 = ho_ground_state(X2, omega, MASS, HBAR) * ho_ground_state(Y2, omega, MASS, HBAR)
    norm = np.sqrt(np.sum(np.abs(psi0)**2) * dx**2)
    psi0 /= norm

    V = 0.5 * MASS * omega**2 * (X2**2 + Y2**2)

    T, Nt = 4 * np.pi, 4000
    psi_snaps, t_snaps = schrodinger_fft_2d(psi0, V, x, T, Nt, save_every=400)

    rho0 = np.abs(psi0)**2
    max_rho_err = max(
        np.max(np.abs(np.abs(psi_snaps[i])**2 - rho0))
        for i in range(len(t_snaps))
    )

    print(f"  2-D HO ground state: max density error = {max_rho_err:.2e}")
    assert max_rho_err < 1e-5, \
        f"2-D HO density drift {max_rho_err:.2e} too large"
    print("[PASS] 2-D HO ground state stationarity verified.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("m5/fft_ref module tests")
    print("=" * 60)

    test_snapshot_bookkeeping()
    test_1d_norm_conservation()
    test_free_particle_spreading()
    test_ho_ground_state_stationary()
    test_ho_coherent_state()
    test_energy_conservation()
    test_time_reversal()
    test_2d_norm_conservation()
    test_2d_ho_stationary()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
