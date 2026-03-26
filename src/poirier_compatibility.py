#!/usr/bin/env python3
"""
CORRECTED Poirier Higher-Order Analysis
========================================

KEY CORRECTION: Under t → λt, x_k → x_k/λ^k, so monomial
x1^a * x2^b * ... has time-scaling degree -(a+2b+3c+4d).

Poirier's Q is degree 0 (invariant under time scaling):
  a + 2b + 3c + 4d = 0

This is DIFFERENT from kinetic energy scaling (degree -2, sum=2).
The previous analysis used sum=2 which was WRONG.

In dimensionless variables z=x2/x1^2, w=x3/x1^3, u=x4/x1^4:
  Q = R(z, w, u)  [NO x1^2 prefactor]

A polynomial R of total degree d captures all meromorphic monomials 
with correct scaling at that degree.
"""

import sympy as sp
from sympy import (symbols, diff, Rational, S, expand, cancel, together,
                   fraction, solve, simplify)
import sys

x1, x2, x3, x4, x5, x6, x7, x8 = symbols('x1 x2 x3 x4 x5 x6 x7 x8')
z, w, u, s_var = symbols('z w u s')

def Dt(expr, order=1):
    xvars = [x1, x2, x3, x4, x5, x6, x7, x8]
    result = expr
    for _ in range(order):
        new = S(0)
        for i in range(len(xvars) - 1):
            new += xvars[i+1] * diff(result, xvars[i])
        result = new
    return result

def compat_eq(Q_expr, n):
    xvars = [x1, x2, x3, x4, x5, x6, x7, x8]
    Qk = [diff(Q_expr, xvars[k]) for k in range(n)]
    C = S(0)
    for k in range(n):
        C += x1 * ((-1)**k) * Dt(Qk[k], order=k+1)
        C += Qk[k] * xvars[k+1]
    return C

def check_sol(Q_expr, n, label=""):
    C = compat_eq(Q_expr, n)
    result = cancel(together(expand(C)))
    ok = (result == 0) or (simplify(result) == 0)
    print(f"  [{'PASS ✓' if ok else 'FAIL ✗'}] {label}")
    if not ok:
        # Show a few terms
        s = str(result)
        print(f"       residual: {s[:120]}{'...' if len(s)>120 else ''}")
    return ok


def polynomial_search(order, max_deg, only_new=False):
    """Search for polynomial R(z, w, u, s) solutions.
    
    order: max derivative order (3, 4, or 5)
    max_deg: max total degree of polynomial in dimensionless vars
    only_new: if True, require presence of highest-order variable
    """
    dim_var_map = {2: z, 3: w, 4: u, 5: s_var}
    dim_vars = [dim_var_map[k] for k in range(2, order+1)]
    
    # x-variable substitutions: z=x2/x1^2, w=x3/x1^3, u=x4/x1^4, s=x5/x1^5
    x_subs = {z: x2/x1**2, w: x3/x1**3, u: x4/x1**4, s_var: x5/x1**5}
    
    # Generate polynomial monomials
    def gen_monoms(vars_list, max_d):
        if not vars_list:
            return [S(1)]
        v = vars_list[0]
        rest = vars_list[1:]
        result = []
        for p in range(max_d + 1):
            for m in gen_monoms(rest, max_d - p):
                result.append(v**p * m)
        return result
    
    all_monoms = gen_monoms(dim_vars, max_deg)
    
    if only_new:
        # Keep only monomials involving the highest-order variable
        highest = dim_vars[-1]
        all_monoms = [m for m in all_monoms if highest in m.free_symbols]
    
    n_monoms = len(all_monoms)
    print(f"\n  Order {order}, poly degree ≤ {max_deg}: {n_monoms} monomials")
    if n_monoms <= 20:
        for i, m in enumerate(all_monoms):
            print(f"    c{i}: {m}")
    
    c_syms = symbols(f'c0:{n_monoms}')
    R_poly = sum(c * m for c, m in zip(c_syms, all_monoms))
    
    # Q = R(z, w, u, ...) = R(x2/x1^2, x3/x1^3, ...) — NO x1^2 prefactor!
    Q_poly = R_poly.subs(x_subs)
    Q_poly = expand(Q_poly)
    
    print(f"  Computing compatibility equation...")
    sys.stdout.flush()
    C = compat_eq(Q_poly, order)
    
    print(f"  Clearing denominators...")
    sys.stdout.flush()
    num, den = fraction(together(expand(C)))
    num = expand(num)
    
    xvars_all = [x1, x2, x3, x4, x5, x6, x7, x8]
    free_xvars = xvars_all[order:][:4]
    bound_xvars = xvars_all[:order]
    
    print(f"  Extracting scalar equations...")
    sys.stdout.flush()
    
    collection_order = list(reversed(free_xvars)) + list(reversed(bound_xvars))
    
    all_eqs = []
    def extract(expr, vars_left):
        if not vars_left:
            e = expand(expr)
            if e != 0 and any(c in e.free_symbols for c in c_syms):
                all_eqs.append(e)
            return
        v = vars_left[0]
        rest = vars_left[1:]
        for p in range(25):
            c = expand(expr).coeff(v, p)
            if c == 0:
                continue
            extract(c, rest)
    
    extract(num, collection_order)
    unique = list(set([eq for eq in all_eqs if eq != 0]))
    print(f"  {len(unique)} equations for {n_monoms} unknowns")
    sys.stdout.flush()
    
    print(f"  Solving...")
    sys.stdout.flush()
    sol = solve(unique, list(c_syms), dict=True)
    print(f"  Solutions: {len(sol)}")
    
    results = []
    if sol:
        for i, s_dict in enumerate(sol):
            R_sol = expand(R_poly.subs(s_dict))
            free = [c for c in c_syms if c not in s_dict]
            
            if R_sol == 0:
                print(f"\n  Solution {i+1}: R = 0 (trivial)")
            else:
                print(f"\n  Solution {i+1}: {len(free)} free parameter(s)")
                for fp in free:
                    basis_R = expand(R_sol).coeff(fp)
                    if basis_R != 0:
                        print(f"    R_{fp}({', '.join(str(v) for v in dim_vars)}) = {basis_R}")
                        
                        # Convert to Q and verify
                        Q_basis = expand(basis_R.subs(x_subs))
                        ok = check_sol(Q_basis, order, f"basis for {fp}")
                        results.append((fp, basis_R, Q_basis, ok))
    
    return results


# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("SANITY CHECKS (with correct scaling)")
print("=" * 72)

B = symbols('B')

# Poirier n=2 (standard quantum mechanics)
Q_P2 = B * (x2**2/x1**4 - Rational(2, 5)*x3/x1**3)
check_sol(Q_P2, 3, "Poirier n=2: B*(x2²/x1⁴ − (2/5)x3/x1³)")

# In dimensionless vars: R = B*(z² - 2w/5)
print(f"    In dimensionless form: R = B*(z² - 2w/5)")

# Check scaling
print(f"    x2²/x1⁴: a+2b+3c+4d = -4+4+0+0 = {-4+4}")
print(f"    x3/x1³:  a+2b+3c+4d = -3+0+3+0 = {-3+3}")

# Poirier n=3
Q_P3 = B * (x2**3/x1**6 - Rational(3, 7)*x2*x3/x1**5)
check_sol(Q_P3, 3, "Poirier n=3: B*(x2³/x1⁶ − (3/7)x2x3/x1⁵)")
print(f"    In dimensionless form: R = B*(z³ - 3zw/7)")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("3rd ORDER: R(z, w) polynomial search, degree ≤ 5")
print("  z = x2/x1², w = x3/x1³, Q = R(z, w)")
print("=" * 72)

res3 = polynomial_search(3, max_deg=5)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("4th ORDER: R(z, w, u) polynomial, degree ≤ 3 (all terms)")
print("  u = x4/x1⁴, Q = R(z, w, u)")
print("=" * 72)

res4a = polynomial_search(4, max_deg=3, only_new=False)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("4th ORDER: R(z, w, u), degree ≤ 4, u-involving terms only")
print("=" * 72)

res4b = polynomial_search(4, max_deg=4, only_new=True)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("4th ORDER: R(z, w, u), degree ≤ 6, u-involving terms only")
print("=" * 72)

res4c = polynomial_search(4, max_deg=6, only_new=True)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("5th ORDER: R(z, w, u, s), degree ≤ 2, s-involving terms only")
print("  s = x5/x1⁵")
print("=" * 72)

res5a = polynomial_search(5, max_deg=2, only_new=True)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("5th ORDER: R(z, w, u, s), degree ≤ 3, s-involving terms only")
print("=" * 72)

res5b = polynomial_search(5, max_deg=3, only_new=True)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print("""
CORRECTION: Previous analysis used scaling a+2b+3c+4d=2 (kinetic energy).
The correct constraint is a+2b+3c+4d=0 (Q is time-scale invariant).
Equivalently, Q = R(z, w, u, ...) with NO x1^2 prefactor.

Results:
""")

if res3:
    print(f"  3rd order: {len(res3)} independent basis element(s) found")
    for fp, basis_R, _, ok in res3:
        print(f"    R = {basis_R}  [verified: {ok}]")
else:
    print("  3rd order: no solutions (unexpected!)")

n4 = len(res4a) + len(res4b) + len(res4c) 
if any(r for r in [res4a, res4b, res4c]):
    genuinely_4th = [r for res in [res4a, res4b, res4c] for r in res 
                     if u in r[1].free_symbols]
    print(f"  4th order: {len(genuinely_4th)} genuinely 4th-order solution(s)")
    for fp, basis_R, _, ok in genuinely_4th:
        print(f"    R = {basis_R}  [verified: {ok}]")
    if not genuinely_4th:
        print("  4th order: only 3rd-order solutions survive (no new u-terms)")
else:
    print("  4th order: NO solutions (Q=0 only)")

if any(r for r in [res5a, res5b]):
    genuinely_5th = [r for res in [res5a, res5b] for r in res
                     if s_var in r[1].free_symbols]
    print(f"  5th order: {len(genuinely_5th)} genuinely 5th-order solution(s)")
else:
    print("  5th order: NO solutions (Q=0 only)")
