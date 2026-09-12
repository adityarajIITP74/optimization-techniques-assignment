"""
Offline verification harness.

This script is NOT part of the solvers -- it exists only to prove that the
hand-written Big-M simplex agrees with a trusted reference implementation.

Checks performed
----------------
1. Known textbook optima for both case studies in big_m_simplex.py
2. Infeasible and unbounded detection
3. 60 randomly generated LPPs with mixed <=, >=, = constraints, compared
   against scipy.optimize.linprog (HiGHS)
4. Known optima for all three transportation case studies

SciPy is required to run THIS file only.  Neither solver imports it.

    pip install scipy
    python3 verify.py
"""

import random
from fractions import Fraction

from big_m_simplex import BigMSimplex, case_study_1, case_study_2
from transportation import (TransportationProblem, case_study_main,
                            case_study, unbalanced_case)

passed = failed = 0


def check(label, got, want):
    global passed, failed
    ok = got == want
    passed, failed = passed + ok, failed + (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} got={got}  want={want}")


print("=" * 78)
print(" VERIFICATION SUITE")
print("=" * 78)

# ---------------------------------------------------------------- 1. known LPPs
print("\n1. Known textbook optima (Big-M)")
r = case_study_1()
r.verbose = False
res = r.solve()
check("Case 1  min Z = 4x1 + x2", res["z"], Fraction(17, 5))
check("Case 1  x1", res["x"]["x1"], Fraction(2, 5))
check("Case 1  x2", res["x"]["x2"], Fraction(9, 5))

r = case_study_2()
r.verbose = False
res = r.solve()
check("Case 2  max Z = 5x1 + 4x2", res["z"], Fraction(21))

# ------------------------------------------------------- 2. degenerate outcomes
print("\n2. Infeasible / unbounded detection")
res = BigMSimplex(c=[1, 1], A=[[1, 1], [1, 1]], signs=[">=", "<="],
                  b=[10, 2], sense="min", verbose=False).solve()
check("x1+x2>=10 and x1+x2<=2", res["status"], "infeasible")

res = BigMSimplex(c=[1, 1], A=[[1, -1]], signs=[">="],
                  b=[1], sense="max", verbose=False).solve()
check("max x1+x2 s.t. x1-x2>=1", res["status"], "unbounded")

# ------------------------------------------------------- 3. random vs SciPy
print("\n3. Random LPPs vs scipy.optimize.linprog")
try:
    from scipy.optimize import linprog
except ImportError:
    print("  [SKIP] scipy not installed -- pip install scipy to run this block")
else:
    random.seed(1)
    mismatches = 0
    for t in range(60):
        m, n = random.randint(2, 4), random.randint(2, 4)
        A = [[random.randint(1, 9) for _ in range(n)] for _ in range(m)]
        b = [random.randint(5, 40) for _ in range(m)]
        c = [random.randint(1, 9) for _ in range(n)]
        signs = [random.choice(["<=", ">=", "="]) for _ in range(m)]

        mine = BigMSimplex(c, A, signs, b, sense="min", verbose=False).solve()

        Aub = [r for r, s in zip(A, signs) if s == "<="]
        bub = [x for x, s in zip(b, signs) if s == "<="]
        Aub += [[-v for v in r] for r, s in zip(A, signs) if s == ">="]
        bub += [-x for x, s in zip(b, signs) if s == ">="]
        Aeq = [r for r, s in zip(A, signs) if s == "="]
        beq = [x for x, s in zip(b, signs) if s == "="]
        ref = linprog(c, Aub or None, bub or None, Aeq or None, beq or None)

        if ref.status == 0 and mine["status"] == "optimal":
            if abs(float(mine["z"]) - ref.fun) > 1e-6:
                mismatches += 1
                print(f"    mismatch on trial {t}: {float(mine['z'])} vs {ref.fun}")
        elif ref.status == 2 and mine["status"] != "infeasible":
            mismatches += 1
            print(f"    feasibility mismatch on trial {t}: {mine['status']}")
    check("60 random mixed-constraint LPPs", mismatches, 0)

# ------------------------------------------------- 4. transportation optima
print("\n4. Known transportation optima (VAM + MODI)")
p = case_study_main()
p.verbose = False
p.vam()
check("Case 1  VAM initial cost", p.total_cost(), Fraction(779))
p.modi()
check("Case 1  MODI optimal cost", p.total_cost(), Fraction(743))

p = case_study()
p.verbose = False
p.vam()
p.modi()
check("Case 2  optimal cost", p.total_cost(), Fraction(2850))

p = unbalanced_case()
p.verbose = False
p.vam()
p.modi()
check("Case 3  optimal cost (unbalanced)", p.total_cost(), Fraction(2424))

# supply / demand feasibility of the final plan
for i in range(p.m):
    assert sum(q for (a, b_), q in p.alloc.items() if a == i) == p.supply[i]
for j in range(p.n):
    assert sum(q for (a, b_), q in p.alloc.items() if b_ == j) == p.demand[j]
check("Case 3  row/column totals satisfied", True, True)

print("\n" + "=" * 78)
print(f" {passed} passed, {failed} failed")
print("=" * 78)
raise SystemExit(1 if failed else 0)
