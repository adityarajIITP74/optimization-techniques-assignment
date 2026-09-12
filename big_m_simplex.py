"""
=============================================================================
 BIG-M (CHARNES' PENALTY) SIMPLEX METHOD  --  implemented from scratch
=============================================================================

 Solves a Linear Programming Problem (LPP) of the form

        Optimise   Z = c1*x1 + c2*x2 + ... + cn*xn
        subject to  a_i1*x1 + ... + a_in*xn   {<= , = , >=}   b_i
                    x_j >= 0

 Method
 ------
 1. Convert the LPP to STANDARD FORM:
        '<='  ->  + slack       (s >= 0)
        '>='  ->  - surplus + artificial
        '='   ->  + artificial
 2. Each artificial variable is penalised in the objective with a very large
    cost M (Big-M).  M is kept SYMBOLIC here: every cost / reduced cost is
    stored as the pair  (constant_part, M_part)  and compared
    lexicographically on the M part first.  This is exact -- it avoids the
    round-off problems you get when M is replaced by a big number like 1e9.
 3. Ordinary simplex iterations (min form): entering variable = most negative
    reduced cost, leaving variable = minimum-ratio test, then pivot.
 4. Termination:
        - all reduced costs >= 0           -> optimal
        - entering column has no positive  -> unbounded
        - optimal but some artificial      -> infeasible (no feasible region)
          still basic at a positive level

 All arithmetic uses fractions.Fraction => exact rational answers, no
 floating-point drift.

 Author : <your name>
 Course : Optimization Techniques -- Assignment 1
=============================================================================
"""

from fractions import Fraction
from typing import List, Tuple, Optional

# --------------------------------------------------------------------------
# A tiny "number with an M in it".  Value = const + coeff_M * M , M -> +inf
# --------------------------------------------------------------------------


class BigMNum:
    """Represents  a + b*M  where M is an arbitrarily large positive number."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fraction(a)   # constant part
        self.b = Fraction(b)   # coefficient of M

    # ---- arithmetic -------------------------------------------------------
    def __add__(self, o):
        o = _as_bigm(o)
        return BigMNum(self.a + o.a, self.b + o.b)

    def __sub__(self, o):
        o = _as_bigm(o)
        return BigMNum(self.a - o.a, self.b - o.b)

    def __mul__(self, k):                       # multiply by a plain scalar
        k = Fraction(k)
        return BigMNum(self.a * k, self.b * k)

    __rmul__ = __mul__

    def __neg__(self):
        return BigMNum(-self.a, -self.b)

    # ---- comparison : M dominates the constant part ------------------------
    def _key(self):
        return (self.b, self.a)

    def __lt__(self, o):
        return self._key() < _as_bigm(o)._key()

    def __gt__(self, o):
        return self._key() > _as_bigm(o)._key()

    def __eq__(self, o):
        return self._key() == _as_bigm(o)._key()

    def is_zero(self):
        return self.a == 0 and self.b == 0

    def is_negative(self):
        return self._key() < (0, 0)

    # ---- printing ----------------------------------------------------------
    def __str__(self):
        if self.b == 0:
            return _fs(self.a)
        coef = _fs(self.b)
        if "/" in coef:
            coef = f"({coef})"
        m = ("" if self.b == 1 else "-" if self.b == -1 else coef) + "M"
        if self.b == -1:
            m = "-M"
        if self.a == 0:
            return m
        return f"{m} {'+' if self.a > 0 else '-'} {_fs(abs(self.a))}"

    __repr__ = __str__


def _as_bigm(x):
    return x if isinstance(x, BigMNum) else BigMNum(x)


def _fs(f: Fraction) -> str:
    """Pretty-print a Fraction: 7 -> '7',  17/5 -> '17/5 (3.4)'."""
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _num(f: Fraction) -> str:
    f = Fraction(f)
    return str(f.numerator) if f.denominator == 1 else f"{float(f):.4f}"


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


class BigMSimplex:
    """
    Parameters
    ----------
    c      : objective coefficients            [c1, c2, ..., cn]
    A      : constraint coefficient matrix     [[a11,...,a1n], ...]
    signs  : one of '<=', '>=', '=' per row    ['<=', '=', ...]
    b      : right-hand sides                  [b1, ..., bm]
    sense  : 'max' or 'min'
    names  : optional names for the decision variables
    """

    def __init__(self, c, A, signs, b, sense="max", names=None, verbose=True):
        self.n = len(c)
        self.m = len(A)
        self.sense = sense.lower()
        self.verbose = verbose
        self.var_names = names or [f"x{j+1}" for j in range(self.n)]

        self.orig_c = [Fraction(v) for v in c]
        self.A = [[Fraction(v) for v in row] for row in A]
        self.b = [Fraction(v) for v in b]
        self.signs = list(signs)

        # --- make every RHS non-negative (multiply the row by -1, flip sign)
        for i in range(self.m):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i] = [-v for v in self.A[i]]
                self.signs[i] = {"<=": ">=", ">=": "<=", "=": "="}[self.signs[i]]

        # --- internally we always MINIMISE.  max Z  ==  min (-Z)
        self.min_c = [v if self.sense == "min" else -v for v in self.orig_c]

        self._build_standard_form()

    # ---------------------------------------------------------------- setup
    def _build_standard_form(self):
        """Add slack / surplus / artificial columns and build the tableau."""
        self.names = list(self.var_names)
        self.cost: List[BigMNum] = [BigMNum(v) for v in self.min_c]
        self.artificials: List[int] = []
        cols = [row[:] for row in self.A]          # working copy

        def add_col(name, cost, rows_with_one, coef=1):
            idx = len(self.names)
            self.names.append(name)
            self.cost.append(cost)
            for i in range(self.m):
                cols[i].append(Fraction(coef) if i in rows_with_one else Fraction(0))
            return idx

        basis = [None] * self.m
        s = su = art = 0

        for i, sign in enumerate(self.signs):
            if sign == "<=":
                s += 1
                basis[i] = add_col(f"s{s}", BigMNum(0), {i}, +1)
            elif sign == ">=":
                su += 1
                add_col(f"su{su}", BigMNum(0), {i}, -1)        # surplus
                art += 1
                k = add_col(f"A{art}", BigMNum(0, 1), {i}, +1)  # artificial, cost +M
                self.artificials.append(k)
                basis[i] = k
            elif sign == "=":
                art += 1
                k = add_col(f"A{art}", BigMNum(0, 1), {i}, +1)
                self.artificials.append(k)
                basis[i] = k
            else:
                raise ValueError(f"Unknown constraint sign: {sign}")

        self.T = cols                # m x (total vars) tableau body
        self.rhs = self.b[:]         # current RHS column
        self.basis = basis
        self.total = len(self.names)

    # ------------------------------------------------------------ reporting
    def _print_tableau(self, it, reduced, entering=None, leaving=None):
        w = 14
        head = "Basis".ljust(7) + "".join(n.rjust(w) for n in self.names) + "RHS".rjust(w)
        print("\n" + "-" * len(head))
        print(f" ITERATION {it}")
        print("-" * len(head))
        print(head)
        for i in range(self.m):
            row = self.names[self.basis[i]].ljust(7)
            row += "".join(_num(v).rjust(w) for v in self.T[i])
            row += _num(self.rhs[i]).rjust(w)
            print(row)
        print("zj-cj".ljust(7) + "".join(str(r).rjust(w) for r in reduced))
        if entering is not None:
            print(f"  -> entering : {self.names[entering]}"
                  f"   |   leaving : {self.names[self.basis[leaving]]}")

    # ------------------------------------------------------------- solving
    def _reduced_costs(self) -> List[BigMNum]:
        """cbar_j = c_j - sum_i cB_i * a_ij   (minimisation form)."""
        cb = [self.cost[self.basis[i]] for i in range(self.m)]
        out = []
        for j in range(self.total):
            z = BigMNum(0)
            for i in range(self.m):
                z = z + cb[i] * self.T[i][j]
            out.append(self.cost[j] - z)
        return out

    def solve(self, max_iter=200):
        if self.verbose:
            self._banner()

        it = 0
        while it < max_iter:
            reduced = self._reduced_costs()

            # ---------- optimality test : all reduced costs >= 0 ------------
            entering, best = None, BigMNum(0)
            for j in range(self.total):
                if reduced[j] < best:
                    best, entering = reduced[j], j

            if entering is None:
                if self.verbose:
                    self._print_tableau(it, reduced)
                    print("\n  All (zj - cj) >= 0  ->  OPTIMAL TABLEAU REACHED.")
                return self._finish()

            # ---------- minimum ratio test ---------------------------------
            leaving, best_ratio = None, None
            for i in range(self.m):
                if self.T[i][entering] > 0:
                    ratio = self.rhs[i] / self.T[i][entering]
                    if best_ratio is None or ratio < best_ratio:
                        best_ratio, leaving = ratio, i

            if leaving is None:
                if self.verbose:
                    self._print_tableau(it, reduced, entering, 0)
                return {"status": "unbounded"}

            if self.verbose:
                self._print_tableau(it, reduced, entering, leaving)

            self._pivot(leaving, entering)
            it += 1

        return {"status": "iteration limit reached"}

    def _pivot(self, r, c):
        p = self.T[r][c]
        self.T[r] = [v / p for v in self.T[r]]
        self.rhs[r] /= p
        for i in range(self.m):
            if i != r and self.T[i][c] != 0:
                f = self.T[i][c]
                self.T[i] = [a - f * b for a, b in zip(self.T[i], self.T[r])]
                self.rhs[i] -= f * self.rhs[r]
        self.basis[r] = c

    # ------------------------------------------------------------- results
    def _finish(self):
        # infeasible if an artificial variable is basic at a positive value
        for i in range(self.m):
            if self.basis[i] in self.artificials and self.rhs[i] > 0:
                return {"status": "infeasible"}

        x = [Fraction(0)] * self.total
        for i in range(self.m):
            x[self.basis[i]] = self.rhs[i]

        z_min = sum(self.cost[j].a * x[j] for j in range(self.total))
        z = z_min if self.sense == "min" else -z_min

        # alternate optima: a non-basic variable with zero reduced cost
        reduced = self._reduced_costs()
        alt = any(j not in self.basis and reduced[j].is_zero()
                  and j < self.n for j in range(self.n))

        return {
            "status": "optimal",
            "x": {self.var_names[j]: x[j] for j in range(self.n)},
            "all_vars": {self.names[j]: x[j] for j in range(self.total)},
            "z": z,
            "alternate_optima": alt,
        }

    # -------------------------------------------------------------- pretty
    def _banner(self):
        print("=" * 78)
        print(" BIG-M SIMPLEX METHOD")
        print("=" * 78)
        obj = "  ".join(
            f"{'+ ' if v >= 0 and j else ''}{_fs(v)} {self.var_names[j]}"
            for j, v in enumerate(self.orig_c))
        print(f"\n {self.sense.upper()}  Z = {obj}")
        print(" subject to")
        for i in range(self.m):
            lhs = "  ".join(
                f"{'+ ' if v >= 0 and j else ''}{_fs(v)} {self.var_names[j]}"
                for j, v in enumerate(self.A[i]))
            print(f"      {lhs}  {self.signs[i]}  {_fs(self.b[i])}")
        print(f"      {', '.join(self.var_names)} >= 0")
        print("\n Standard form variables:", ", ".join(self.names))
        print(" Artificial variables    :",
              ", ".join(self.names[k] for k in self.artificials) or "none")

    @staticmethod
    def report(res, title="RESULT"):
        print("\n" + "=" * 78)
        print(f" {title}")
        print("=" * 78)
        if res["status"] != "optimal":
            print(f" Status: {res['status'].upper()}")
            if res["status"] == "infeasible":
                print(" An artificial variable stayed in the basis at a positive")
                print(" level -> the constraint set has no feasible point.")
            return
        print(" Status: OPTIMAL\n")
        print(" Optimal decision variables")
        for k, v in res["x"].items():
            print(f"     {k:>4} = {_fs(v):<10} ({float(v):.4f})")
        print("\n Slack / surplus / artificial values")
        for k, v in res["all_vars"].items():
            if k not in res["x"]:
                print(f"     {k:>4} = {_fs(v):<10} ({float(v):.4f})")
        z = res["z"]
        print(f"\n Optimal objective value  Z* = {_fs(z)}  ({float(z):.4f})")
        if res["alternate_optima"]:
            print(" Note: a non-basic variable has zero (zj-cj) ->"
                  " alternate optimal solutions exist.")
        print("=" * 78)


# --------------------------------------------------------------------------
# CASE STUDY
# --------------------------------------------------------------------------

def case_study_1():
    """
    Classic textbook Big-M problem (Taha, 'Operations Research').
    Needs a surplus + artificial for the '>=' row and an artificial for the
    '=' row, so the Big-M machinery is genuinely exercised.

        Minimise  Z = 4 x1 + x2
        s.t.      3 x1 +   x2  =  3
                  4 x1 + 3 x2 >=  6
                    x1 + 2 x2 <=  4
                  x1, x2 >= 0

    Known optimum:  x1 = 2/5, x2 = 9/5, Z = 17/5 = 3.4
    """
    return BigMSimplex(
        c=[4, 1],
        A=[[3, 1],
           [4, 3],
           [1, 2]],
        signs=["=", ">=", "<="],
        b=[3, 6, 4],
        sense="min",
        names=["x1", "x2"],
    )


def case_study_2():
    """
    A production / product-mix problem (maximisation with a mixed set of
    constraints, so artificial variables are again required).

        A workshop makes two products P1 and P2.
        Profit  : Rs 5 per P1, Rs 4 per P2
        Machine : 6 P1 + 4 P2 <= 24 hours
        Labour  : 1 P1 + 2 P2 <=  6 hours
        Contract: P1 + P2     >=  2 units must be produced

        Maximise Z = 5 x1 + 4 x2
    """
    return BigMSimplex(
        c=[5, 4],
        A=[[6, 4],
           [1, 2],
           [1, 1]],
        signs=["<=", "<=", ">="],
        b=[24, 6, 2],
        sense="max",
        names=["x1", "x2"],
    )


def main():
    s1 = case_study_1()
    BigMSimplex.report(s1.solve(), "CASE STUDY 1 -- MINIMISATION")

    print("\n\n")

    s2 = case_study_2()
    BigMSimplex.report(s2.solve(), "CASE STUDY 2 -- MAXIMISATION (PRODUCT MIX)")


if __name__ == "__main__":
    main()
