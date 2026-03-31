"""
test_m5_sim.py — Verify m5/sim.py simulation driver and helper functions.

Tests:
  1.  select_backend: CPU fallback returns numpy
  2.  _Dx: central-difference first derivative on sin → cos
  3.  _D2x: central-difference second derivative on sin → −sin
  4.  _gaussian_smooth: preserves integral, known analytic convolution
  5.  _linear_interp: exact for linear function, accurate for smooth
  6.  kernel_sums: single-particle Gaussian shape, derivative sign
  7.  psi_kde_fields: √ρ, ln ρ, v, u from known kernel sums
  8.  gh_nodes_weights: sum=1, symmetric nodes, E[ξ²]=½ (physicist)
  9.  _grid_psi_kde: reconstruct a simple Gaussian ψ from particles
  10. m5_simulate API: mode validation, required args, result keys
  11. Grid mode short run: HO ground state density approximately stable
  12. Gridless mode short run: free Gaussian centre drifts at p₀/m
  13. Snapshot bookkeeping: correct counts and times in both modes
  14. Trajectory tracking: track_ids records positions at every step
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from m5.sim import (select_backend, _Dx, _D2x, _gaussian_smooth,
                    _linear_interp, kernel_sums, psi_kde_fields,
                    gh_nodes_weights, _grid_psi_kde, m5_simulate)
from m5.init import (init_ensemble_1d, gaussian_wp, ho_ground_state,
                     Ensemble, Units)
from m5.utils import output_path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR = 1.0
MASS = 1.0
PI = np.pi


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: select_backend
# ─────────────────────────────────────────────────────────────────────────────

def test_select_backend():
    """CPU fallback should always return numpy."""
    xp, name = select_backend(force='cpu')
    assert xp is np, "force='cpu' should return numpy"
    assert 'numpy' in name.lower() or 'cpu' in name.lower(), \
        f"Backend name '{name}' should mention numpy or CPU"

    # Auto-detect should succeed (returns either numpy or cupy)
    xp_auto, name_auto = select_backend(force=None)
    assert hasattr(xp_auto, 'array'), "Auto backend must have .array()"

    # GPU request on CPU-only machine should either succeed or raise
    try:
        xp_gpu, _ = select_backend(force='gpu')
    except RuntimeError:
        pass  # expected if no GPU

    print("[PASS] select_backend verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: _Dx central-difference first derivative
# ─────────────────────────────────────────────────────────────────────────────

def test_Dx():
    """Central difference of sin(x) should approximate cos(x)."""
    Nx = 512
    L = 2 * PI
    x = np.linspace(0, L, Nx, endpoint=False)
    dx = x[1] - x[0]

    f = np.sin(x)
    df_num = _Dx(f, dx, np)
    df_exact = np.cos(x)

    err = np.max(np.abs(df_num - df_exact))
    print(f"  _Dx(sin) max error = {err:.2e}  (expect O(dx²) ≈ {dx**2:.2e})")
    assert err < 5 * dx**2, f"_Dx error {err:.2e} exceeds O(dx²) bound"

    # Also test on x² (non-periodic, so exclude boundary-wrapped points)
    f2 = x**2
    df2 = _Dx(f2, dx, np)
    # Interior points only (periodic BC wraps boundary)
    interior = slice(2, -2)
    err2 = np.max(np.abs(df2[interior] - 2 * x[interior]))
    print(f"  _Dx(x²) interior error = {err2:.2e}")
    assert err2 < 5 * dx**2, f"_Dx(x²) interior error {err2:.2e} too large"
    print("[PASS] _Dx central-difference verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: _D2x central-difference second derivative
# ─────────────────────────────────────────────────────────────────────────────

def test_D2x():
    """Central difference of sin(x) should approximate −sin(x)."""
    Nx = 512
    L = 2 * PI
    x = np.linspace(0, L, Nx, endpoint=False)
    dx = x[1] - x[0]

    f = np.sin(x)
    d2f_num = _D2x(f, dx, np)
    d2f_exact = -np.sin(x)

    err = np.max(np.abs(d2f_num - d2f_exact))
    print(f"  _D2x(sin) max error = {err:.2e}  (expect O(dx²) ≈ {dx**2:.2e})")
    assert err < 5 * dx**2, f"_D2x error {err:.2e} exceeds O(dx²) bound"
    print("[PASS] _D2x central-difference verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: _gaussian_smooth
# ─────────────────────────────────────────────────────────────────────────────

def test_gaussian_smooth():
    """Gaussian smoothing should preserve integral and match analytic result."""
    Nx = 512
    L = 40.0
    x = np.linspace(-L/2, L/2, Nx, endpoint=False)
    dx = x[1] - x[0]

    # Input: narrow Gaussian (σ₀ = 0.5)
    sigma0 = 0.5
    f = np.exp(-x**2 / (2 * sigma0**2))
    f /= np.sum(f) * dx   # normalise

    sigma_cells = 2.0     # smoothing bandwidth in grid cells
    sigma_phys = sigma_cells * dx
    f_smooth = _gaussian_smooth(f, sigma_cells, np)

    # Integral preservation
    int_orig = np.sum(f) * dx
    int_smooth = np.sum(f_smooth) * dx
    assert abs(int_smooth - int_orig) < 1e-12, \
        f"Integral changed: {int_orig:.12f} → {int_smooth:.12f}"

    # Analytic: Gaussian * Gaussian = Gaussian with σ² = σ₀² + σ_s²
    sigma_conv = np.sqrt(sigma0**2 + sigma_phys**2)
    f_analytic = np.exp(-x**2 / (2 * sigma_conv**2))
    f_analytic /= np.sum(f_analytic) * dx

    err = np.max(np.abs(f_smooth - f_analytic))
    print(f"  Gaussian smooth: integral preserved to {abs(int_smooth-int_orig):.2e}")
    print(f"  Convolution max error = {err:.2e}")
    assert err < 1e-4, f"Gaussian convolution error {err:.2e} too large"

    # Complex input: should handle real and imaginary independently
    fc = f + 1j * f * 0.5
    fc_smooth = _gaussian_smooth(fc, sigma_cells, np)
    assert np.iscomplexobj(fc_smooth), "Complex input should produce complex output"
    err_re = np.max(np.abs(fc_smooth.real - f_smooth))
    err_im = np.max(np.abs(fc_smooth.imag - 0.5 * f_smooth))
    assert err_re < 1e-14 and err_im < 1e-14, "Complex smoothing inconsistent"

    print("[PASS] _gaussian_smooth verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: _linear_interp
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_interp():
    """Linear interpolation should be exact for linear functions."""
    Nx = 64
    x_grid = np.linspace(-5, 5, Nx, endpoint=False)

    # Linear function: f(x) = 3x + 2
    field = 3.0 * x_grid + 2.0
    X_test = np.array([-4.3, -1.0, 0.0, 2.7, 4.5])
    f_interp = _linear_interp(X_test, x_grid, field, np)
    f_exact = 3.0 * X_test + 2.0

    err = np.max(np.abs(f_interp - f_exact))
    print(f"  Linear interp (linear func): max error = {err:.2e}")
    assert err < 1e-10, f"Linear interpolation of linear function error {err:.2e}"

    # Smooth function: f(x) = exp(-x²/2)
    field_gauss = np.exp(-x_grid**2 / 2.0)
    f_interp_g = _linear_interp(X_test, x_grid, field_gauss, np)
    f_exact_g = np.exp(-X_test**2 / 2.0)
    err_g = np.max(np.abs(f_interp_g - f_exact_g))
    dx = x_grid[1] - x_grid[0]
    print(f"  Linear interp (Gaussian): max error = {err_g:.2e}  "
          f"(expect O(dx²) ≈ {dx**2:.2e})")
    # For smooth functions, error is O(dx²·f″)
    assert err_g < 0.1, f"Gaussian interpolation error {err_g:.2e} too large"

    print("[PASS] _linear_interp verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: kernel_sums
# ─────────────────────────────────────────────────────────────────────────────

def test_kernel_sums():
    """
    Verify kernel_sums against analytic single-particle KDE.

    One particle at x=0 with phase φ=0:
      n(x) = K(x/h)/h,  j_re(x) = K(x/h)/h·cos(0) = n(x),  j_im=0
    where K is the standard Gaussian kernel.
    """
    h = 0.5
    cutoff = 4.0     # default cutoff in kernel_sums
    X_src = np.array([0.0])
    phi_src = np.array([0.0])

    # Stay well within cutoff radius (cutoff*h = 2.0)
    eval_x = np.linspace(-1.8, 1.8, 101)
    ks = kernel_sums(eval_x, X_src, phi_src, h, np,
                     need_deriv=True, cutoff=cutoff)

    # Analytic: n(x) = (1/h√2π) exp(-x²/2h²)
    inv_h = 1.0 / h
    norm = inv_h / np.sqrt(2 * PI)
    n_exact = norm * np.exp(-0.5 * (eval_x / h)**2)

    err_n = np.max(np.abs(ks['n'] - n_exact))
    err_jre = np.max(np.abs(ks['j_re'] - n_exact))  # cos(0)=1
    err_jim = np.max(np.abs(ks['j_im']))             # sin(0)=0

    print(f"  kernel_sums (1 particle): n error={err_n:.2e}  "
          f"j_re error={err_jre:.2e}  j_im max={err_jim:.2e}")
    assert err_n < 1e-12, f"n error {err_n:.2e}"
    assert err_jre < 1e-12, f"j_re error {err_jre:.2e}"
    assert err_jim < 1e-15, f"j_im should be zero, got max {err_jim:.2e}"

    # Derivative: n'(x) = -(x/h²) · n(x)
    np_exact = -(eval_x / h**2) * n_exact
    err_jp = np.max(np.abs(ks['jp_re'] - np_exact))
    print(f"  kernel_sums derivative: jp_re error={err_jp:.2e}")
    assert err_jp < 1e-10, f"jp_re error {err_jp:.2e}"

    # Test with non-zero phase: φ = π/2 → cos=0, sin=1
    phi_half_pi = np.array([PI / 2])
    ks2 = kernel_sums(eval_x, X_src, phi_half_pi, h, np, cutoff=cutoff)
    err_jre2 = np.max(np.abs(ks2['j_re']))  # should be ~0
    err_jim2 = np.max(np.abs(ks2['j_im'] - n_exact))  # should be n
    assert err_jre2 < 1e-12, f"j_re(φ=π/2) should be 0, got max {err_jre2:.2e}"
    assert err_jim2 < 1e-12, f"j_im(φ=π/2) error {err_jim2:.2e}"

    print("[PASS] kernel_sums verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: psi_kde_fields
# ─────────────────────────────────────────────────────────────────────────────

def test_psi_kde_fields():
    """
    From kernel_sums of a coherent ensemble, psi_kde_fields should
    give √ρ > 0, sensible ln ρ, and velocity v ≈ p₀/m.
    """
    # Ensemble: N particles with uniform phase p₀·x / ℏ (plane wave)
    Np = 500
    rng = np.random.default_rng(12345)
    X_src = rng.normal(0, 1.0, Np)
    p0 = 2.0
    phi_src = p0 * X_src / HBAR

    h = 0.4
    eval_x = np.linspace(-3, 3, 201)
    ks = kernel_sums(eval_x, X_src, phi_src, h, np, need_deriv=True)
    fields = psi_kde_fields(ks, np, hbar=HBAR, mass=MASS)

    # √ρ should be positive
    assert np.all(fields['sqrt_rho'] >= 0), "√ρ should be non-negative"

    # ln ρ should be finite in the bulk (where density is significant)
    bulk = fields['sqrt_rho'] > 0.01 * fields['sqrt_rho'].max()
    assert np.all(np.isfinite(fields['ln_rho'][bulk])), \
        "ln ρ should be finite in bulk"

    # Velocity should approximate p₀/m in the bulk
    v_expected = p0 / MASS
    v_bulk = fields['v'][bulk]
    v_median = np.median(v_bulk)
    print(f"  psi_kde_fields: median v in bulk = {v_median:.3f}  "
          f"(expected {v_expected:.3f})")
    assert abs(v_median - v_expected) < 0.5, \
        f"Median velocity {v_median:.3f} far from expected {v_expected:.3f}"

    # Osmotic velocity u should be present
    assert 'u' in fields, "Osmotic velocity u should be in fields"

    print("[PASS] psi_kde_fields verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: gh_nodes_weights
# ─────────────────────────────────────────────────────────────────────────────

def test_gh_nodes_weights():
    """
    Gauss–Hermite nodes and weights should satisfy:
      (a) weights sum to 1 (after normalisation)
      (b) nodes are symmetric about 0
      (c) E[ξ²] = ½  (physicist's convention, weight exp(−t²))
    """
    for K in [3, 6, 10, 20]:
        xi, omega = gh_nodes_weights(K)

        # (a) Weights sum to 1
        wsum = omega.sum()
        assert abs(wsum - 1.0) < 1e-14, \
            f"K={K}: weights sum to {wsum}, expected 1.0"

        # (b) Symmetric nodes
        xi_sorted = np.sort(xi)
        err_sym = np.max(np.abs(xi_sorted + xi_sorted[::-1]))
        assert err_sym < 1e-14, f"K={K}: nodes not symmetric, err={err_sym:.2e}"

        # (c) E[ξ²] = ½ for physicist's hermgauss
        # The normalised weights approximate ∫f(t)exp(-t²)dt / ∫exp(-t²)dt
        # so E[ξ²] = Σ ω_i ξ_i² should equal ½
        E_xi2 = np.sum(omega * xi**2)
        print(f"  K={K:2d}: Σω={wsum:.14f}  E[ξ²]={E_xi2:.10f}  "
              f"(expect 0.5)")
        assert abs(E_xi2 - 0.5) < 1e-10, \
            f"K={K}: E[ξ²]={E_xi2:.10f}, expected 0.5"

    print("[PASS] gh_nodes_weights verified for all K values.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: _grid_psi_kde (CIC deposit + Gaussian smooth)
# ─────────────────────────────────────────────────────────────────────────────

def test_grid_psi_kde():
    """
    CIC deposit + smooth of a Gaussian ensemble should reconstruct
    a Gaussian-like ψ with density ∝ |ψ₀|².
    """
    Nx = 256
    x_grid = np.linspace(-8, 8, Nx, endpoint=False)
    dx = x_grid[1] - x_grid[0]

    # Ensemble from HO ground state (real, positive ψ₀ → S=0)
    psi0 = ho_ground_state(x_grid)
    Np = 2000
    ens = init_ensemble_1d(psi0, x_grid, Np, method='deterministic')

    sigma_cells = 2.5
    psi_est, sqrt_rho = _grid_psi_kde(
        np.asarray(ens.X), np.asarray(ens.S),
        x_grid, sigma_cells, HBAR, np)

    # √ρ should be non-negative
    assert np.all(sqrt_rho >= 0), "√ρ from CIC deposit should be non-negative"

    # Reconstructed density should resemble |ψ₀|²
    rho_exact = np.abs(psi0)**2
    rho_exact /= np.sum(rho_exact) * dx
    rho_est = np.abs(psi_est)**2
    rho_est_norm = rho_est / (np.sum(rho_est) * dx + 1e-30)

    # L² error (allow for smoothing broadening)
    err = np.sqrt(np.sum((rho_est_norm - rho_exact)**2) * dx)
    print(f"  _grid_psi_kde: L² density error = {err:.4f}  (Np={Np})")
    assert err < 0.3, f"Grid ψ-KDE density error {err:.4f} too large"

    # Phase should be approximately zero (HO ground state is real)
    phase_est = np.angle(psi_est)
    bulk = np.abs(psi_est) > 0.1 * np.max(np.abs(psi_est))
    phase_err = np.max(np.abs(phase_est[bulk]))
    print(f"  Phase error in bulk = {phase_err:.4f}")
    assert phase_err < 0.5, f"Phase error {phase_err:.4f} too large for real ψ₀"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_grid, rho_exact, 'k-', lw=1.5, label='|ψ₀|²')
    axes[0].plot(x_grid, rho_est_norm, 'r--', lw=1, label='CIC+smooth')
    axes[0].set_title('Density reconstruction')
    axes[0].legend(fontsize=8)

    axes[1].plot(x_grid, phase_est, 'b-', lw=1)
    axes[1].set_title('Phase (should be ≈ 0)')
    axes[1].set_ylabel('∠ψ̂')

    fig.suptitle(f'_grid_psi_kde: HO ground state, Np={Np}', fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path('test_sim_grid_psi_kde.png'), dpi=120)
    plt.close(fig)

    print("[PASS] _grid_psi_kde verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: m5_simulate API validation
# ─────────────────────────────────────────────────────────────────────────────

def test_api_validation():
    """m5_simulate should reject invalid modes and missing x_grid for grid mode."""
    Nx = 64
    x = np.linspace(-6, 6, Nx, endpoint=False)
    psi0 = ho_ground_state(x)
    ens = init_ensemble_1d(psi0, x, 100)
    V_func = lambda xx: 0.5 * xx**2

    # Invalid mode should raise ValueError
    try:
        m5_simulate(ens, V_func, T=0.1, Nt=10, mode='invalid',
                    verbose=False)
        assert False, "Should have raised ValueError for invalid mode"
    except ValueError as e:
        assert 'invalid' in str(e).lower() or 'mode' in str(e).lower()

    # Grid mode without x_grid should raise ValueError
    try:
        m5_simulate(ens, V_func, T=0.1, Nt=10, mode='grid',
                    x_grid=None, verbose=False)
        assert False, "Should have raised ValueError for grid mode without x_grid"
    except ValueError as e:
        assert 'grid' in str(e).lower()

    print("[PASS] m5_simulate API validation verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Grid mode — HO ground state short run
# ─────────────────────────────────────────────────────────────────────────────

def test_grid_ho_ground_state():
    """
    Grid-mode simulation of HO ground state: density should remain
    approximately Gaussian and centred at origin.

    The ground state is a stationary state, so ⟨x⟩ ≈ 0 and the
    width should not change dramatically over a short run.
    Tolerances are relaxed because ψ-KDE is approximate.
    """
    Nx = 256
    x = np.linspace(-8, 8, Nx, endpoint=False)
    dx = x[1] - x[0]
    omega = 1.0
    Np = 1000
    T = 0.5          # short run: quarter period
    Nt = 200
    save_every = 100

    psi0 = ho_ground_state(x, omega=omega)
    ens = init_ensemble_1d(psi0, x, Np, method='deterministic')
    V_func = lambda xx: 0.5 * MASS * omega**2 * xx**2

    result = m5_simulate(ens, V_func, T=T, Nt=Nt, mode='grid',
                         x_grid=x, save_every=save_every,
                         sigma_kde=2.5, K_cand=24,
                         verbose=False, seed=42)

    # Check result dict keys
    assert 'X' in result and 'S' in result and 't_save' in result
    assert result['mode'] == 'grid'

    # Initial and final mean positions
    X0 = result['X'][0]
    X_final = result['X'][-1]
    x_mean_0 = np.mean(X0)
    x_mean_f = np.mean(X_final)

    print(f"  Grid HO ground: ⟨x⟩₀={x_mean_0:.4f}  "
          f"⟨x⟩_final={x_mean_f:.4f}")

    # For ground state, centre should stay near zero
    assert abs(x_mean_f) < 1.0, \
        f"Final ⟨x⟩ = {x_mean_f:.4f} drifted too far from origin"

    # Width should remain in reasonable range (σ₀ = 1/√ω for HO)
    sigma0 = 1.0 / np.sqrt(omega)
    sigma_f = np.std(X_final)
    ratio = sigma_f / sigma0
    print(f"  Width ratio σ_final/σ₀ = {ratio:.3f}")
    assert 0.5 < ratio < 2.0, \
        f"Width ratio {ratio:.3f} out of reasonable range"

    print("[PASS] Grid mode HO ground state short run verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: Gridless mode — free Gaussian drift
# ─────────────────────────────────────────────────────────────────────────────

def test_gridless_free_drift():
    """
    Gridless simulation of a free Gaussian wave packet (V=0):
    the ensemble centre of mass should drift at v = p₀/m.

    This tests the basic advection step (velocity from ψ-KDE)
    without complications from an external potential.
    """
    Nx = 256
    x = np.linspace(-15, 15, Nx, endpoint=False)
    dx = x[1] - x[0]

    sigma0, p0, x0 = 1.5, 2.0, 0.0
    Np = 600
    T = 0.5
    Nt = 200
    save_every = Nt   # only initial and final

    psi0 = gaussian_wp(x, x0, p0, sigma0)
    psi0_func = lambda xx: gaussian_wp(xx, x0, p0, sigma0,
                                        hbar=HBAR)
    ens = init_ensemble_1d(psi0, x, Np, method='deterministic',
                           psi0_func=psi0_func)
    V_func = lambda xx: np.zeros_like(xx)

    result = m5_simulate(ens, V_func, T=T, Nt=Nt, mode='gridless',
                         x_grid=x, save_every=save_every,
                         K_gh=6, sigma_gh=0.20, h_kde=0.35,
                         verbose=False, seed=42)

    assert result['mode'] == 'gridless'

    X0 = result['X'][0]
    X_final = result['X'][-1]
    x_mean_0 = np.mean(X0)
    x_mean_f = np.mean(X_final)

    v_expected = p0 / MASS
    x_expected = x0 + v_expected * T
    drift = x_mean_f - x_mean_0

    print(f"  Gridless free drift: ⟨x⟩₀={x_mean_0:.4f}  "
          f"⟨x⟩_final={x_mean_f:.4f}")
    print(f"  Expected drift = {v_expected * T:.4f}  "
          f"actual drift = {drift:.4f}")

    # Allow generous tolerance (ψ-KDE is approximate, small N)
    assert abs(drift - v_expected * T) < 1.0, \
        f"Drift error {abs(drift - v_expected * T):.4f} too large"

    # Direction should be correct at minimum
    assert drift > 0, "Free packet with p₀>0 should drift to the right"

    print("[PASS] Gridless free drift verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Snapshot bookkeeping (both modes)
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_bookkeeping():
    """Verify correct snapshot counts and times for various save_every."""
    Nx = 64
    x = np.linspace(-6, 6, Nx, endpoint=False)
    psi0 = ho_ground_state(x)
    Np = 100
    T, Nt = 0.5, 50
    V_func = lambda xx: 0.5 * xx**2

    for mode in ['grid', 'gridless']:
        for save_every in [10, 25, 50]:
            ens = init_ensemble_1d(psi0, x, Np, method='deterministic')
            kwargs = dict(x_grid=x, verbose=False, seed=42,
                          backend='cpu')
            if mode == 'gridless':
                kwargs.update(K_gh=4, sigma_gh=0.20, h_kde=0.35)

            result = m5_simulate(ens, V_func, T=T, Nt=Nt,
                                 mode=mode, save_every=save_every,
                                 **kwargs)

            expected_indices = list(range(0, Nt + 1, save_every))
            expected_Ns = len(expected_indices)
            expected_times = np.array([i * T / Nt for i in expected_indices])

            Ns_actual = len(result['t_save'])
            assert Ns_actual == expected_Ns, \
                f"{mode} save_every={save_every}: got {Ns_actual} snapshots, " \
                f"expected {expected_Ns}"

            np.testing.assert_allclose(
                result['t_save'], expected_times, atol=1e-10,
                err_msg=f"{mode} save_every={save_every}: times incorrect")

            # Particle arrays should have correct shapes
            assert result['X'].shape[0] == expected_Ns
            assert result['X'].shape[1] == Np
            assert result['S'].shape == result['X'].shape

    print("[PASS] Snapshot bookkeeping verified for both modes.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: Trajectory tracking
# ─────────────────────────────────────────────────────────────────────────────

def test_trajectory_tracking():
    """track_ids should record particle positions at every time step."""
    Nx = 64
    x = np.linspace(-6, 6, Nx, endpoint=False)
    psi0 = ho_ground_state(x)
    Np = 100
    T, Nt = 0.2, 20
    V_func = lambda xx: 0.5 * xx**2

    track_ids = [0, 10, 50, 99]

    for mode in ['grid', 'gridless']:
        ens = init_ensemble_1d(psi0, x, Np, method='deterministic')
        kwargs = dict(x_grid=x, verbose=False, seed=42, backend='cpu')
        if mode == 'gridless':
            kwargs.update(K_gh=4, sigma_gh=0.20, h_kde=0.35)

        result = m5_simulate(ens, V_func, T=T, Nt=Nt, mode=mode,
                             save_every=Nt, track_ids=track_ids,
                             **kwargs)

        assert 'traj_X' in result, \
            f"{mode}: traj_X missing from result"
        assert 'traj_t' in result, \
            f"{mode}: traj_t missing from result"

        tX = result['traj_X']
        tT = result['traj_t']

        # Shape: (Nt+1, len(track_ids))
        assert tX.shape == (Nt + 1, len(track_ids)), \
            f"{mode}: traj_X shape {tX.shape}, expected {(Nt+1, len(track_ids))}"
        assert len(tT) == Nt + 1, \
            f"{mode}: traj_t length {len(tT)}, expected {Nt+1}"

        # Initial positions should match ensemble
        X0 = np.asarray(ens.X)
        for j, pid in enumerate(track_ids):
            assert abs(tX[0, j] - X0[pid]) < 1e-12, \
                f"{mode}: initial traj[{pid}] mismatch"

        # All trajectory values should be finite
        assert np.all(np.isfinite(tX)), f"{mode}: non-finite trajectory values"

        print(f"  {mode}: trajectory tracking OK "
              f"(shape {tX.shape}, all finite)")

    print("[PASS] Trajectory tracking verified for both modes.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: Gridless without x_grid (V_func evaluated directly)
# ─────────────────────────────────────────────────────────────────────────────

def test_gridless_no_grid():
    """
    Gridless mode should work without x_grid, evaluating V_func
    directly at particle positions each step.
    """
    Nx = 128
    x = np.linspace(-8, 8, Nx, endpoint=False)
    psi0 = ho_ground_state(x)
    Np = 200
    T, Nt = 0.1, 20

    ens = init_ensemble_1d(psi0, x, Np, method='deterministic')
    V_func = lambda xx: 0.5 * xx**2

    result = m5_simulate(ens, V_func, T=T, Nt=Nt, mode='gridless',
                         x_grid=None,     # <-- no grid
                         K_gh=4, sigma_gh=0.20, h_kde=0.35,
                         save_every=Nt, verbose=False, seed=42)

    assert result['mode'] == 'gridless'
    assert np.all(np.isfinite(result['X']))
    assert np.all(np.isfinite(result['S']))
    print(f"  Gridless (no grid): final ⟨x⟩ = {np.mean(result['X'][-1]):.4f}")
    print("[PASS] Gridless without x_grid verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Grid mode result contains field history
# ─────────────────────────────────────────────────────────────────────────────

def test_grid_result_fields():
    """
    Grid mode should return psi, v_field, Q_field, Q_tilde arrays
    at snapshot times.
    """
    Nx = 64
    x = np.linspace(-6, 6, Nx, endpoint=False)
    psi0 = ho_ground_state(x)
    Np = 200
    T, Nt = 0.2, 20
    save_every = 10

    ens = init_ensemble_1d(psi0, x, Np, method='deterministic')
    V_func = lambda xx: 0.5 * xx**2

    result = m5_simulate(ens, V_func, T=T, Nt=Nt, mode='grid',
                         x_grid=x, save_every=save_every,
                         verbose=False, seed=42)

    expected_Ns = len(range(0, Nt + 1, save_every))

    for key in ['psi', 'v_field', 'Q_field', 'Q_tilde']:
        assert key in result, f"Grid result missing '{key}'"
        arr = result[key]
        assert arr.shape[0] == expected_Ns, \
            f"'{key}' has {arr.shape[0]} snapshots, expected {expected_Ns}"
        assert arr.shape[1] == Nx, \
            f"'{key}' grid size {arr.shape[1]}, expected {Nx}"

    # ψ should be complex
    assert np.iscomplexobj(result['psi']), "psi should be complex"

    # Q_field and v_field should be real and finite
    assert np.all(np.isfinite(result['v_field'])), "v_field has non-finite values"
    assert np.all(np.isfinite(result['Q_field'])), "Q_field has non-finite values"

    print("[PASS] Grid mode field history verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot: comparison of both modes
# ─────────────────────────────────────────────────────────────────────────────

def _comparison_plot():
    """
    Diagnostic plot comparing grid and gridless mode on a free
    Gaussian wave packet at several time snapshots.
    Not an assertion test — visual diagnostic only.
    """
    Nx = 256
    x = np.linspace(-15, 15, Nx, endpoint=False)
    dx = x[1] - x[0]

    sigma0, p0, x0 = 1.5, 1.5, 0.0
    Np = 800
    T = 1.0
    Nt = 400
    save_every = 100

    psi0 = gaussian_wp(x, x0, p0, sigma0)
    psi0_func = lambda xx: gaussian_wp(xx, x0, p0, sigma0)
    V_func = lambda xx: np.zeros_like(xx)

    ens_g = init_ensemble_1d(psi0, x, Np, method='deterministic',
                             psi0_func=psi0_func)
    ens_gl = init_ensemble_1d(psi0, x, Np, method='deterministic',
                              psi0_func=psi0_func)

    r_grid = m5_simulate(ens_g, V_func, T=T, Nt=Nt, mode='grid',
                         x_grid=x, save_every=save_every,
                         sigma_kde=2.5, K_cand=24,
                         verbose=False, seed=42)

    r_gl = m5_simulate(ens_gl, V_func, T=T, Nt=Nt, mode='gridless',
                       x_grid=x, save_every=save_every,
                       K_gh=6, sigma_gh=0.20, h_kde=0.35,
                       verbose=False, seed=42)

    # Build analytic ρ(x,t) for comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f'test_m5_sim: Grid vs Gridless (Np={Np}, T={T})',
                 fontsize=13, weight='bold')

    edges = np.linspace(x[0], x[-1], 201)
    centres = 0.5 * (edges[:-1] + edges[1:])

    for si in range(min(3, len(r_grid['t_save']))):
        t = r_grid['t_save'][si]

        # Analytic spreading
        v_grp = p0 / MASS
        sig_t = sigma0 * np.sqrt(1 + (HBAR * t / (2 * MASS * sigma0**2))**2)
        x_c = x0 + v_grp * t
        rho_exact = np.exp(-(x - x_c)**2 / (2 * sig_t**2))
        rho_exact /= np.sum(rho_exact) * dx

        for row, (res, label) in enumerate([(r_grid, 'Grid'),
                                             (r_gl, 'Gridless')]):
            ax = axes[row, si]
            X_snap = res['X'][si]
            h, _ = np.histogram(X_snap, bins=edges, density=True)
            ax.bar(centres, h, width=edges[1]-edges[0],
                   alpha=0.4, color='steelblue', label=f'{label} (N={Np})')
            ax.plot(x, rho_exact, 'k-', lw=1.5, label='exact')
            ax.set_title(f'{label}, t={t:.2f}', fontsize=10)
            ax.set_xlim(-12, 12)
            if si == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path('test_sim_comparison.png'), dpi=120)
    plt.close(fig)
    print("  Comparison plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("m5/sim module tests")
    print("=" * 60)

    test_select_backend()
    test_Dx()
    test_D2x()
    test_gaussian_smooth()
    test_linear_interp()
    test_kernel_sums()
    test_psi_kde_fields()
    test_gh_nodes_weights()
    test_grid_psi_kde()
    test_api_validation()
    test_grid_ho_ground_state()
    test_gridless_free_drift()
    test_snapshot_bookkeeping()
    test_trajectory_tracking()
    test_gridless_no_grid()
    test_grid_result_fields()

    print("\n" + "-" * 60)
    print("Generating diagnostic comparison plot ...")
    _comparison_plot()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
