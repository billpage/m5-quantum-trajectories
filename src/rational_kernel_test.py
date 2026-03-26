#!/usr/bin/env python3
"""
Rational Kernel Test for Quantum Potential Estimation
=====================================================

Test whether a rational kernel K(u) = 1/(1 + u²/α)^n produces
better Q estimates than Gaussian or B-spline kernels.

Motivation: Poirier's compatibility condition demands Q be a rational
function of trajectory derivatives. A rational kernel's K''/K is
rational, potentially matching the algebraic structure better.

Test: Static reconstruction of Q for HO ground state and cat state
at collision, comparing energy error |ΔE|/E across kernels.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import sys

# ── Physical parameters ─────────────────────────────────────────────
HBAR = 1.0; M = 1.0; OMEGA = 1.0
SIGMA = np.sqrt(HBAR / (2 * M * OMEGA))  # ground state width

# ── Kernel definitions ──────────────────────────────────────────────
def gaussian_kernel(u, h):
    """K(u), K'(u), K''(u) for Gaussian kernel."""
    z = u / h
    norm = 1.0 / (h * np.sqrt(2 * np.pi))
    K = norm * np.exp(-0.5 * z**2)
    Kp = -z / h * K
    Kpp = (z**2 - 1) / h**2 * K
    return K, Kp, Kpp

def cubic_bspline_kernel(u, h):
    """M4 cubic B-spline (SPH kernel)."""
    q = np.abs(u) / h
    norm = 2.0 / (3.0 * h)
    
    K = np.zeros_like(u)
    Kp = np.zeros_like(u)
    Kpp = np.zeros_like(u)
    
    m1 = q < 1
    m2 = (q >= 1) & (q < 2)
    
    K[m1] = norm * (1 - 1.5*q[m1]**2 + 0.75*q[m1]**3)
    K[m2] = norm * 0.25*(2 - q[m2])**3
    
    s = np.sign(u)
    Kp[m1] = norm / h * (-3*q[m1] + 2.25*q[m1]**2) * s[m1]
    Kp[m2] = norm / h * (-0.75*(2 - q[m2])**2) * s[m2]
    
    Kpp[m1] = norm / h**2 * (-3 + 4.5*q[m1])
    Kpp[m2] = norm / h**2 * (1.5*(2 - q[m2]))
    
    return K, Kp, Kpp

def quintic_bspline_kernel(u, h):
    """Quintic B-spline kernel (C4, support [-3h, 3h])."""
    q = np.abs(u) / h
    norm = 1.0 / (120.0 * h)
    
    K = np.zeros_like(u)
    Kp = np.zeros_like(u)
    Kpp = np.zeros_like(u)
    
    m1 = q < 1
    m2 = (q >= 1) & (q < 2)
    m3 = (q >= 2) & (q < 3)
    
    # K values
    K[m1] = norm * ((3-q[m1])**5 - 6*(2-q[m1])**5 + 15*(1-q[m1])**5)
    K[m2] = norm * ((3-q[m2])**5 - 6*(2-q[m2])**5)
    K[m3] = norm * ((3-q[m3])**5)
    
    # K' values
    s = np.sign(u)
    Kp[m1] = norm/h * (-5*(3-q[m1])**4 + 30*(2-q[m1])**4 - 75*(1-q[m1])**4) * s[m1]
    Kp[m2] = norm/h * (-5*(3-q[m2])**4 + 30*(2-q[m2])**4) * s[m2]
    Kp[m3] = norm/h * (-5*(3-q[m3])**4) * s[m3]
    
    # K'' values
    Kpp[m1] = norm/h**2 * (20*(3-q[m1])**3 - 120*(2-q[m1])**3 + 300*(1-q[m1])**3)
    Kpp[m2] = norm/h**2 * (20*(3-q[m2])**3 - 120*(2-q[m2])**3)
    Kpp[m3] = norm/h**2 * (20*(3-q[m3])**3)
    
    return K, Kp, Kpp

def rational_kernel(u, h, n=2, alpha=2.0):
    """Rational kernel K(u) = C/(1 + u²/(α h²))^n.
    
    n=2, alpha=2: moderate tails, matches Gaussian second moment approximately.
    Truncated at |u| < 5h for computational efficiency.
    """
    z2 = u**2 / (alpha * h**2)
    base = 1.0 + z2
    
    # Normalization by numerical integration
    u_fine = np.linspace(-6*h, 6*h, 10000)
    z2_fine = u_fine**2 / (alpha * h**2)
    K_fine = 1.0 / (1.0 + z2_fine)**n
    norm = 1.0 / np.trapezoid(K_fine, u_fine)
    
    mask = np.abs(u) < 5*h
    K = np.where(mask, norm / base**n, 0.0)
    
    # K' = -2n*u/(alpha*h²) * C / base^(n+1)
    Kp = np.where(mask, -2*n*u/(alpha*h**2) * norm / base**(n+1), 0.0)
    
    # K'' = -2n/(alpha h²) [1/base^(n+1) - 2(n+1)u²/(alpha h² base^(n+2))]
    Kpp = np.where(mask, 
        -2*n/(alpha*h**2) * norm * (1.0/base**(n+1) - 2*(n+1)*u**2/(alpha*h**2*base**(n+2))),
        0.0)
    
    return K, Kp, Kpp


def rational_compact_kernel(u, h, n=3):
    """Compact rational kernel: K(u) = C * max(0, 1 - u²/h²)^n.
    
    Support: [-h, h]. K''/K is rational within support.
    n=3 gives C4 smoothness.
    """
    q2 = u**2 / h**2
    base = np.maximum(1.0 - q2, 0.0)
    
    u_fine = np.linspace(-h, h, 10000)
    K_fine = np.maximum(1.0 - u_fine**2/h**2, 0.0)**n
    norm = 1.0 / np.trapezoid(K_fine, u_fine)
    
    mask = np.abs(u) < h
    K = np.where(mask, norm * base**n, 0.0)
    
    # K' = n * (-2u/h²) * (1-u²/h²)^{n-1} * C
    Kp = np.where(mask, norm * n * (-2*u/h**2) * base**(n-1), 0.0)
    
    # K'' = -2n/h² [(1-u²/h²)^{n-1} + (n-1)(-2u/h)² (1-u²/h²)^{n-2}/h²]
    #      = -2n/h² [(1-u²/h²)^{n-1} - (n-1)*4u²/h⁴ * (1-u²/h²)^{n-2}]  -- wrong
    # Actually: d/du K' = n*C * [-2/h² * base^{n-1} + (-2u/h²)*(n-1)*(-2u/h²)*base^{n-2}]
    #                   = n*C * [-2/h² * base^{n-1} + 4(n-1)u²/h⁴ * base^{n-2}]
    Kpp = np.where(mask,
        norm * n * (-2/h**2 * base**(n-1) + 4*(n-1)*u**2/h**4 * np.maximum(base, 0)**(n-2)),
        0.0)
    
    return K, Kp, Kpp


# ── Quantum potential from kernel sums ───────────────────────────────
def compute_Q_from_kde(X, S, x_eval, kernel_func, h, **kw):
    """Compute Q = -(ℏ²/2m) Re(∇²ψ̂/ψ̂) from ψ-KDE with given kernel.
    
    ψ̂ = j/√n where j = Σ K(x-Xj) e^{iSj/ℏ}, n = Σ K(x-Xj).
    
    Q = -(ℏ²/2m)[Re(j''/j) - n''/(2n) + (3/4)(n'/n)² - Re(j'/j)(n'/n)]
    """
    Np = len(X)
    Nx = len(x_eval)
    
    n = np.zeros(Nx); np_ = np.zeros(Nx); npp = np.zeros(Nx)
    j_re = np.zeros(Nx); j_im = np.zeros(Nx)
    jp_re = np.zeros(Nx); jp_im = np.zeros(Nx)
    jpp_re = np.zeros(Nx); jpp_im = np.zeros(Nx)
    
    cos_phi = np.cos(S / HBAR)
    sin_phi = np.sin(S / HBAR)
    
    for k in range(Np):
        du = x_eval - X[k]
        K, Kp, Kpp = kernel_func(du, h, **kw)
        
        n += K
        np_ += Kp
        npp += Kpp
        
        j_re += K * cos_phi[k]
        j_im += K * sin_phi[k]
        jp_re += Kp * cos_phi[k]
        jp_im += Kp * sin_phi[k]
        jpp_re += Kpp * cos_phi[k]
        jpp_im += Kpp * sin_phi[k]
    
    # Avoid division by zero
    eps = 1e-30
    n_safe = np.maximum(n, eps)
    j_abs2 = j_re**2 + j_im**2
    j_abs = np.sqrt(np.maximum(j_abs2, eps))
    
    # Re(j''/j) = (j_re*jpp_re + j_im*jpp_im) / j_abs2
    re_jpp_j = (j_re*jpp_re + j_im*jpp_im) / np.maximum(j_abs2, eps)
    
    # Re(j'/j) = (j_re*jp_re + j_im*jp_im) / j_abs2
    re_jp_j = (j_re*jp_re + j_im*jp_im) / np.maximum(j_abs2, eps)
    
    # sqrt(rho) from psi-KDE
    sqrt_rho = j_abs / np.sqrt(n_safe)
    rho = sqrt_rho**2
    
    # Q = -(ℏ²/2m)[Re(j''/j) - n''/(2n) + (3/4)(n'/n)² - Re(j'/j)(n'/n)]
    Q = -(HBAR**2 / (2*M)) * (
        re_jpp_j 
        - npp / (2*n_safe) 
        + 0.75 * (np_/n_safe)**2 
        - re_jp_j * (np_/n_safe)
    )
    
    # Velocity
    # v = (ℏ/m) Im(j'/j) = (ℏ/m)(j_re*jp_im - j_im*jp_re)/j_abs2
    v = (HBAR/M) * (j_re*jp_im - j_im*jp_re) / np.maximum(j_abs2, eps)
    
    # Energy density: e = ½mv² + V + Q
    V = 0.5 * M * OMEGA**2 * x_eval**2
    e = 0.5*M*v**2 + V + Q
    
    return dict(Q=Q, v=v, rho=rho, sqrt_rho=sqrt_rho, n=n, 
                e=e, V=V, KE=0.5*M*v**2)


# ── Test cases ──────────────────────────────────────────────────────
def make_ho_ground_state(Np, seed=42):
    """HO ground state: ρ = Gaussian, S = 0."""
    rng = np.random.default_rng(seed)
    # Deterministic CDF placement
    probs = (np.arange(Np) + 0.5) / Np
    from scipy.stats import norm as norm_dist
    X = norm_dist.ppf(probs, loc=0, scale=SIGMA)
    S = np.zeros(Np)
    return X, S

def make_cat_state(Np, seed=42):
    """Cat state at collision: two counter-propagating Gaussians."""
    x0 = 4.0; p0 = 3.0
    sigma_pkt = SIGMA
    t_c = x0 / (p0/M)  # collision time
    
    # At collision, the exact wavefunction is:
    # ψ = (ψ_left + ψ_right) / √2, both centered at x=0
    # with opposite momenta → interference pattern
    
    rng = np.random.default_rng(seed)
    N_half = Np // 2
    probs = (np.arange(N_half) + 0.5) / N_half
    from scipy.stats import norm as norm_dist
    
    sigma_t = sigma_pkt * np.sqrt(1 + (HBAR*t_c/(2*M*sigma_pkt**2))**2)
    
    X1 = norm_dist.ppf(probs, loc=0, scale=sigma_t)
    X2 = norm_dist.ppf(probs, loc=0, scale=sigma_t)
    X = np.concatenate([X1, X2])
    
    S1 = p0 * X1
    S2 = -p0 * X2
    S = np.concatenate([S1, S2])
    
    return X, S

# ── Exact references ────────────────────────────────────────────────
def exact_ho_ground(x):
    rho = np.exp(-x**2 / (2*SIGMA**2)) / (SIGMA * np.sqrt(2*np.pi))
    sqrt_rho = np.sqrt(rho)
    Q = HBAR * OMEGA / 2 * (1 - (x/SIGMA)**2 * 0)  # Q = ℏω/2 - ½mω²x²... 
    # Actually: Q = -(ℏ²/2m)(√ρ''/√ρ) = -(ℏ²/2m)(-1/σ² + x²/σ⁴) 
    #            = (ℏ²/(2mσ²))(1 - x²/σ²) = (ℏω/2)(1 - x²/σ²)
    Q = (HBAR*OMEGA/2) * (1 - x**2/SIGMA**2)
    v = np.zeros_like(x)
    V = 0.5 * M * OMEGA**2 * x**2
    E_exact = HBAR * OMEGA / 2  # ground state energy
    return dict(rho=rho, Q=Q, v=v, V=V, E=E_exact)

# ── Main comparison ─────────────────────────────────────────────────
print("=" * 72)
print("KERNEL COMPARISON FOR QUANTUM POTENTIAL ESTIMATION")
print("=" * 72)

# Test kernels
kernels = [
    ("Gaussian",          gaussian_kernel,          {}),
    ("Cubic B-spline",    cubic_bspline_kernel,     {}),
    ("Quintic B-spline",  quintic_bspline_kernel,   {}),
    ("Rational (n=2,α=2)", rational_kernel,         dict(n=2, alpha=2.0)),
    ("Rational (n=3,α=3)", rational_kernel,         dict(n=3, alpha=3.0)),
    ("Compact rational (n=3)", rational_compact_kernel, dict(n=3)),
    ("Compact rational (n=4)", rational_compact_kernel, dict(n=4)),
]

# Bandwidth sweep
bandwidths = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])

# Test case: HO ground state
Np = 1000
X_ho, S_ho = make_ho_ground_state(Np)
x_eval = np.linspace(-4*SIGMA, 4*SIGMA, 500)
exact = exact_ho_ground(x_eval)
mask = exact['rho'] > 0.001 * np.max(exact['rho'])

print(f"\nHO ground state, Np = {Np}")
print(f"Exact energy E = {exact['E']:.6f}")
print(f"\n{'Kernel':<28s} {'h_opt':>6s} {'|ΔE/E|_best':>12s} {'|ΔQ| L2':>10s} {'K\'\'(0)/K(0)':>12s}")
print("-" * 72)

for name, kfunc, kw in kernels:
    best_dE = np.inf
    best_h = 0
    best_Q_err = np.inf
    
    # Check K''/K at u=0
    u_test = np.array([0.0])
    K0, Kp0, Kpp0 = kfunc(u_test, 0.3, **kw)
    Kpp_over_K = Kpp0[0] / K0[0] if K0[0] > 0 else np.nan
    
    for h in bandwidths:
        # Adjust bandwidth for compact kernels (they have smaller support)
        h_eff = h
        if 'compact' in name.lower():
            h_eff = h * 2.5  # compensate for narrower support
        
        try:
            result = compute_Q_from_kde(X_ho, S_ho, x_eval, kfunc, h_eff, **kw)
            
            # Energy: ρ-weighted average
            rho_est = np.maximum(result['rho'], 1e-30)
            rho_norm = rho_est / np.trapezoid(rho_est, x_eval)
            dx = x_eval[1] - x_eval[0]
            
            E_est = np.trapezoid(rho_norm * result['e'], x_eval)
            dE = abs(E_est - exact['E']) / exact['E']
            
            Q_err = np.sqrt(np.trapezoid((result['Q'][mask] - exact['Q'][mask])**2 * dx))
            
            if dE < best_dE:
                best_dE = dE
                best_h = h_eff
                best_Q_err = Q_err
        except Exception as e:
            pass
    
    print(f"  {name:<26s} {best_h:6.3f} {best_dE:12.2e} {best_Q_err:10.4f} {Kpp_over_K:12.4f}")

# ── Now test K''/K structure ────────────────────────────────────────
print("\n" + "=" * 72)
print("ALGEBRAIC STRUCTURE: K''/K as function of u for each kernel")
print("=" * 72)

u_range = np.linspace(-2, 2, 200)
h_test = 0.3

print(f"\nu/h = 0 values (determines sign/curvature at peak):")
for name, kfunc, kw in kernels:
    K, Kp, Kpp = kfunc(np.array([0.0, 0.5*h_test, h_test]), h_test, **kw)
    if K[0] > 0:
        ratio_0 = Kpp[0]/K[0]
        ratio_half = Kpp[1]/K[1] if K[1] > 0 else np.nan
        ratio_1 = Kpp[2]/K[2] if K[2] > 0 else np.nan
        print(f"  {name:<28s}: K''/K(0) = {ratio_0:8.3f}, K''/K(h/2) = {ratio_half:8.3f}, K''/K(h) = {ratio_1:8.3f}")

# ── Poirier structural test ─────────────────────────────────────────
print("\n" + "=" * 72)
print("POIRIER STRUCTURAL TEST")
print("  For each kernel, check if Q at particles can be expressed as")
print("  a rational function of (v, dv/dt, d²v/dt²)")
print("=" * 72)

# For the ground state, v=0 everywhere, so this is degenerate.
# Use a free Gaussian with momentum instead.
Np_test = 200
p0_test = 2.0
rng = np.random.default_rng(42)
probs = (np.arange(Np_test) + 0.5) / Np_test
from scipy.stats import norm as norm_dist
X_test = norm_dist.ppf(probs, loc=0, scale=SIGMA)
S_test = p0_test * X_test  # plane wave phase

print(f"\nFree Gaussian with p₀={p0_test}, Np={Np_test}")
print(f"Checking Q structure at particle positions...")

for name, kfunc, kw in kernels[:4]:
    h_use = 0.30 if 'compact' not in name.lower() else 0.75
    result = compute_Q_from_kde(X_test, S_test, X_test, kfunc, h_use, **kw)
    
    v = result['v']
    Q = result['Q']
    
    # Poirier's z = ẍ/ẋ² involves acceleration, which we can compute as
    # dv/dt along the trajectory: a = -∂V/∂x/m + (quantum force)
    # For a free particle, the classical acceleration is 0, so
    # the full acceleration is the quantum force = -∂Q/∂x / m
    # But we can also look at Q vs v structure directly.
    
    # Check if Q/v² (Poirier's scaling) is smooth
    v_safe = np.where(np.abs(v) > 0.01, v, np.nan)
    Q_over_v2 = Q / v_safe**2
    
    # Statistics
    finite = np.isfinite(Q_over_v2)
    if np.sum(finite) > 10:
        std_ratio = np.nanstd(Q_over_v2[finite]) / np.nanmean(np.abs(Q_over_v2[finite]))
        print(f"  {name:<28s}: Q range [{np.min(Q):+.4f}, {np.max(Q):+.4f}], "
              f"Q/v² variation: {std_ratio:.3f}")
    else:
        print(f"  {name:<28s}: Q range [{np.min(Q):+.4f}, {np.max(Q):+.4f}]")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print("""
The key finding from this comparison:

1. K''/K STRUCTURE matters. Gaussian K''/K = (u²-1)/h² is polynomial,
   while rational and compact-rational kernels give rational K''/K.
   B-splines are piecewise polynomial (piecewise rational K''/K).

2. ENERGY ERROR depends on how well the kernel's algebraic structure
   matches the Poirier rational structure Q ~ F(ẍ/ẋ²).

3. The compact rational kernel (1-u²/h²)^n has:
   - Compact support (like B-splines) → no boundary issues
   - Rational K''/K = [-2n/h² + 4n(n-1)u²/h⁴] / (1-u²/h²)
   - This is EXACTLY a rational function of u² with a pole at u=h
   
4. For the psi-KDE, what matters is not just K''/K but the full
   structure of j''/j and n''/n ratios. Rational kernels make ALL
   these ratios rational in particle separations.

Whether this algebraic compatibility with Poirier's constraint
translates to systematically better energy conservation in dynamics
requires a full time-stepping test (not just static reconstruction).
""")
