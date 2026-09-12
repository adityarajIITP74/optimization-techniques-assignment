"""
=============================================================================
 TRANSPORTATION PROBLEM
   (i)  VAM  -- Vogel's Approximation Method      (initial BFS)
   (ii) MODI -- Modified Distribution / u-v Method (optimality test + improve)
=============================================================================

 Problem
 -------
 m sources with supplies a_i, n destinations with demands b_j, unit shipping
 cost c_ij.  Find x_ij >= 0 minimising  sum(c_ij * x_ij)  subject to
        sum_j x_ij = a_i        sum_i x_ij = b_j

 Pipeline implemented here
 -------------------------
 1. BALANCE      : if sum(supply) != sum(demand) a dummy row/column with zero
                   cost is added.
 2. VAM          : repeatedly compute the row/column PENALTY (difference of
                   the two smallest remaining costs), pick the line with the
                   largest penalty, and allocate as much as possible to its
                   cheapest cell.  Gives a much better starting solution than
                   North-West Corner or Least-Cost.
 3. DEGENERACY   : a BFS must occupy exactly m+n-1 cells.  If VAM leaves
                   fewer, near-zero (epsilon) allocations are inserted in the
                   cheapest cells that do not close a loop.
 4. MODI         : solve  u_i + v_j = c_ij  on the occupied cells (u_1 = 0),
                   then the opportunity cost of every empty cell is
                        d_ij = c_ij - (u_i + v_j)
                   - all d_ij >= 0  -> current solution is OPTIMAL
                   - otherwise take the most negative d_ij, trace the unique
                     closed loop through occupied cells, and shift theta =
                     min allocation on the '-' corners around that loop.
 5. Repeat step 4 until optimal.  Report the shipment plan and the minimum
    total cost, plus a note if alternate optima exist (some d_ij == 0).

 The two methods can also be run INDEPENDENTLY (see `run_mode` below), as the
 assignment asks for "one method at a time".

 Author : Aditya Raj
 Course : Optimization Techniques -- Assignment 1
=============================================================================
"""

from fractions import Fraction
from typing import List, Tuple, Dict, Optional

EPS = Fraction(0)          # value stored in a degenerate (epsilon) cell
INF = float("inf")


# --------------------------------------------------------------------------
class TransportationProblem:

    def __init__(self, cost, supply, demand,
                 source_names=None, dest_names=None, verbose=True):
        self.cost = [[Fraction(v) for v in row] for row in cost]
        self.supply = [Fraction(v) for v in supply]
        self.demand = [Fraction(v) for v in demand]
        self.src = source_names or [f"S{i+1}" for i in range(len(supply))]
        self.dst = dest_names or [f"D{j+1}" for j in range(len(demand))]
        self.verbose = verbose
        self.dummy_note = None
        self._balance()
        self.m, self.n = len(self.supply), len(self.demand)
        self.alloc: Dict[Tuple[int, int], Fraction] = {}
        self.basic: set = set()

    # ------------------------------------------------------------- balancing
    def _balance(self):
        ts, td = sum(self.supply), sum(self.demand)
        if ts == td:
            return
        if ts < td:                      # need a dummy SOURCE
            self.cost.append([Fraction(0)] * len(self.demand))
            self.supply.append(td - ts)
            self.src.append("Dummy")
            self.dummy_note = (f"Unbalanced: supply {ts} < demand {td}. "
                               f"Dummy source of {td-ts} units added "
                               f"(zero cost = unmet demand).")
        else:                            # need a dummy DESTINATION
            for row in self.cost:
                row.append(Fraction(0))
            self.demand.append(ts - td)
            self.dst.append("Dummy")
            self.dummy_note = (f"Unbalanced: supply {ts} > demand {td}. "
                               f"Dummy destination of {ts-td} units added "
                               f"(zero cost = unshipped stock).")

    # =====================================================================
    # STEP 1 : VOGEL'S APPROXIMATION METHOD
    # =====================================================================
    def vam(self):
        if self.verbose:
            print("\n" + "=" * 72)
            print(" STEP 1 -- VOGEL'S APPROXIMATION METHOD (initial BFS)")
            print("=" * 72)

        sup = self.supply[:]
        dem = self.demand[:]
        rows_open = set(range(self.m))
        cols_open = set(range(self.n))
        alloc: Dict[Tuple[int, int], Fraction] = {}
        step = 0

        while rows_open and cols_open:
            # ---- penalties ------------------------------------------------
            pens = []   # (penalty, kind, index)
            for i in rows_open:
                cs = sorted(self.cost[i][j] for j in cols_open)
                pens.append(((cs[1] - cs[0]) if len(cs) > 1 else cs[0], "row", i))
            for j in cols_open:
                cs = sorted(self.cost[i][j] for i in rows_open)
                pens.append(((cs[1] - cs[0]) if len(cs) > 1 else cs[0], "col", j))

            best = max(p[0] for p in pens)
            # tie-break: among the largest penalties pick the one whose
            # cheapest cell allows the biggest allocation
            cands = [p for p in pens if p[0] == best]
            choice = None
            for _, kind, idx in cands:
                if kind == "row":
                    j = min(cols_open, key=lambda j: self.cost[idx][j])
                    cell = (idx, j)
                else:
                    i = min(rows_open, key=lambda i: self.cost[i][idx])
                    cell = (i, idx)
                q = min(sup[cell[0]], dem[cell[1]])
                if choice is None or q > choice[0]:
                    choice = (q, cell, kind, idx, best)

            q, (i, j), kind, idx, pen = choice
            alloc[(i, j)] = alloc.get((i, j), Fraction(0)) + q
            sup[i] -= q
            dem[j] -= q
            step += 1

            if self.verbose:
                who = f"{self.src[idx]} (row)" if kind == "row" else f"{self.dst[idx]} (col)"
                print(f"  {step:>2}. max penalty = {_f(pen)} on {who:<14}"
                      f" -> allocate {_f(q):>6} to ({self.src[i]}, {self.dst[j]})"
                      f"  cost {_f(self.cost[i][j])}")

            # close the exhausted line (only one, to avoid losing basic cells)
            if sup[i] == 0 and dem[j] == 0:
                if len(rows_open) > 1:
                    rows_open.discard(i)
                else:
                    cols_open.discard(j)
            elif sup[i] == 0:
                rows_open.discard(i)
            else:
                cols_open.discard(j)

        self.alloc = alloc
        self.basic = set(alloc.keys())
        self._fix_degeneracy()

        if self.verbose:
            self.show_table("Initial basic feasible solution (VAM)")
            print(f"  Initial transportation cost = {_f(self.total_cost())}")
        return self.alloc

    # =====================================================================
    # degeneracy handling : we need exactly m + n - 1 occupied cells
    # =====================================================================
    def _creates_loop(self, cell, cells):
        return self._find_loop(cell, cells | {cell}) is not None

    def _fix_degeneracy(self):
        need = self.m + self.n - 1
        if len(self.basic) >= need:
            return
        empties = sorted(
            ((self.cost[i][j], i, j)
             for i in range(self.m) for j in range(self.n)
             if (i, j) not in self.basic))
        for c, i, j in empties:
            if len(self.basic) >= need:
                break
            if not self._creates_loop((i, j), self.basic):
                self.basic.add((i, j))
                self.alloc[(i, j)] = EPS
                if self.verbose:
                    print(f"  [degeneracy] epsilon allocation placed at "
                          f"({self.src[i]}, {self.dst[j]}) to reach "
                          f"m+n-1 = {need} occupied cells")

    # =====================================================================
    # closed-loop search : alternate row moves and column moves
    # =====================================================================
    def _find_loop(self, start, cells) -> Optional[List[Tuple[int, int]]]:
        cells = set(cells)

        def rec(path, axis):
            # axis 0 -> next move stays in the same ROW (changes column)
            # axis 1 -> next move stays in the same COLUMN (changes row)
            last = path[-1]
            if len(path) >= 4 and len(path) % 2 == 0:
                if (axis == 0 and last[0] == start[0]) or \
                   (axis == 1 and last[1] == start[1]):
                    return path
            for c in cells:
                if c in path:
                    continue
                if axis == 0 and c[0] == last[0]:
                    r = rec(path + [c], 1)
                    if r:
                        return r
                elif axis == 1 and c[1] == last[1]:
                    r = rec(path + [c], 0)
                    if r:
                        return r
            return None

        return rec([start], 0) or rec([start], 1)

    # =====================================================================
    # STEP 2 : MODI (u-v) METHOD
    # =====================================================================
    def modi(self, max_iter=100):
        if self.verbose:
            print("\n" + "=" * 72)
            print(" STEP 2 -- MODI (MODIFIED DISTRIBUTION) METHOD")
            print("=" * 72)

        for it in range(1, max_iter + 1):
            u, v = self._potentials()

            # opportunity costs of the unoccupied cells
            d = {}
            for i in range(self.m):
                for j in range(self.n):
                    if (i, j) not in self.basic:
                        d[(i, j)] = self.cost[i][j] - (u[i] + v[j])

            if self.verbose:
                print(f"\n --- MODI iteration {it} ---")
                print("  u =", "  ".join(f"{self.src[i]}:{_f(u[i])}"
                                         for i in range(self.m)))
                print("  v =", "  ".join(f"{self.dst[j]}:{_f(v[j])}"
                                         for j in range(self.n)))
                neg = {k: x for k, x in d.items() if x < 0}
                print("  opportunity costs d_ij = c_ij - (u_i + v_j):")
                print("    " + "  ".join(
                    f"({self.src[i]},{self.dst[j]})={_f(x)}"
                    for (i, j), x in sorted(d.items())))
                if not neg:
                    print("  -> every d_ij >= 0 : solution is OPTIMAL")

            if not d or min(d.values()) >= 0:
                self.alternate = any(x == 0 for x in d.values())
                return self.alloc

            enter = min(d, key=lambda k: d[k])
            loop = self._find_loop(enter, self.basic | {enter})
            if loop is None:
                raise RuntimeError("no closed loop found - basis is degenerate")

            minus = loop[1::2]                       # '-' corners
            theta = min(self.alloc[c] for c in minus)
            leaving = min(minus, key=lambda c: self.alloc[c])

            if self.verbose:
                print(f"  most negative d = {_f(d[enter])} at "
                      f"({self.src[enter[0]]}, {self.dst[enter[1]]})  -> entering cell")
                print("  closed loop: " + " -> ".join(
                    f"{'+' if k % 2 == 0 else '-'}({self.src[i]},{self.dst[j]})"
                    for k, (i, j) in enumerate(loop)))
                print(f"  theta = {_f(theta)}  (leaving cell "
                      f"({self.src[leaving[0]]}, {self.dst[leaving[1]]}))")

            for k, c in enumerate(loop):
                self.alloc[c] = self.alloc.get(c, Fraction(0)) + \
                    (theta if k % 2 == 0 else -theta)
            self.basic.add(enter)
            self.basic.discard(leaving)
            self.alloc.pop(leaving, None)

            if self.verbose:
                self.show_table(f"Revised allocation after iteration {it}")
                print(f"  Cost now = {_f(self.total_cost())}")

        raise RuntimeError("MODI did not converge")

    def _potentials(self):
        u = [None] * self.m
        v = [None] * self.n
        u[0] = Fraction(0)
        changed = True
        while changed:
            changed = False
            for (i, j) in self.basic:
                if u[i] is not None and v[j] is None:
                    v[j] = self.cost[i][j] - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = self.cost[i][j] - v[j]
                    changed = True
        # any component left unreached (shouldn't happen after degeneracy fix)
        u = [x if x is not None else Fraction(0) for x in u]
        v = [x if x is not None else Fraction(0) for x in v]
        return u, v

    # ------------------------------------------------------------- reporting
    def total_cost(self):
        return sum(self.cost[i][j] * q for (i, j), q in self.alloc.items())

    def show_table(self, title=""):
        w = 10
        print(f"\n  {title}")
        print("  " + "Source".ljust(9) + "".join(d.rjust(w) for d in self.dst)
              + "Supply".rjust(w))
        for i in range(self.m):
            line = "  " + self.src[i].ljust(9)
            for j in range(self.n):
                q = self.alloc.get((i, j))
                if q is None:
                    txt = f"[{_f(self.cost[i][j])}]"
                elif q == 0:
                    txt = f"e({_f(self.cost[i][j])})"
                else:
                    txt = f"{_f(q)}({_f(self.cost[i][j])})"
                line += txt.rjust(w)
            line += _f(self.supply[i]).rjust(w)
            print(line)
        print("  " + "Demand".ljust(9)
              + "".join(_f(d).rjust(w) for d in self.demand))
        print("   legend: qty(cost), [cost] = empty cell, e = epsilon cell")

    def report(self):
        print("\n" + "=" * 72)
        print(" FINAL RESULT")
        print("=" * 72)
        if self.dummy_note:
            print(" " + self.dummy_note + "\n")
        print(" Optimal shipment plan")
        for (i, j), q in sorted(self.alloc.items()):
            if q > 0:
                print(f"     {self.src[i]:>6} -> {self.dst[j]:<6} : "
                      f"{_f(q):>7} units @ {_f(self.cost[i][j])} = "
                      f"{_f(q * self.cost[i][j])}")
        print(f"\n Minimum total transportation cost = {_f(self.total_cost())}")
        if getattr(self, "alternate", False):
            print(" Note: some d_ij = 0 -> alternate optimal shipment plans exist.")
        print("=" * 72)

    def describe(self):
        print("=" * 72)
        print(" TRANSPORTATION PROBLEM -- COST MATRIX")
        print("=" * 72)
        w = 9
        print("  " + "".ljust(9) + "".join(d.rjust(w) for d in self.dst)
              + "Supply".rjust(w))
        for i in range(self.m):
            print("  " + self.src[i].ljust(9)
                  + "".join(_f(c).rjust(w) for c in self.cost[i])
                  + _f(self.supply[i]).rjust(w))
        print("  " + "Demand".ljust(9)
              + "".join(_f(d).rjust(w) for d in self.demand))
        if self.dummy_note:
            print("\n  " + self.dummy_note)


def _f(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# --------------------------------------------------------------------------
# CASE STUDY
# --------------------------------------------------------------------------

def case_study_main():
    """
    Featured case study -- a well-known textbook transportation problem where
    the VAM starting solution is NOT optimal, so the MODI improvement loop is
    genuinely exercised.

        Three factories supply four distribution centres.
        supply = (7, 9, 18)     demand = (5, 8, 7, 14)

    VAM initial cost = 779   ->   MODI optimal cost = 743
    """
    return TransportationProblem(
        cost=[[19, 30, 50, 10],
              [70, 30, 40, 60],
              [40,  8, 70, 20]],
        supply=[7, 9, 18],
        demand=[5, 8, 7, 14],
        source_names=["F1", "F2", "F3"],
        dest_names=["DC1", "DC2", "DC3", "DC4"],
    )


def case_study():
    """
    Classic balanced transportation problem.
    Three plants ship to four warehouses.
        supply = (300, 400, 500)   demand = (250, 350, 400, 200)
    Known minimum cost = 2850.
    """
    return TransportationProblem(
        cost=[[3, 1, 7, 4],
              [2, 6, 5, 9],
              [8, 3, 3, 2]],
        supply=[300, 400, 500],
        demand=[250, 350, 400, 200],
        source_names=["Plant1", "Plant2", "Plant3"],
        dest_names=["W1", "W2", "W3", "W4"],
    )


def unbalanced_case():
    """Supply (76, 82, 77) > demand (72, 102, 41) -> dummy destination."""
    return TransportationProblem(
        cost=[[4, 8, 8],
              [16, 24, 16],
              [8, 16, 24]],
        supply=[76, 82, 77],
        demand=[72, 102, 41],
        source_names=["A", "B", "C"],
        dest_names=["X", "Y", "Z"],
    )


def run_mode(problem, mode="both"):
    """
    mode = 'vam'  -> only the initial BFS by Vogel's Approximation Method
    mode = 'modi' -> BFS (needed as a starting point) + MODI optimisation
    mode = 'both' -> full pipeline with commentary
    """
    problem.describe()
    problem.vam()
    if mode in ("modi", "both"):
        problem.modi()
    problem.report()
    return problem


def main():
    print("\n########## CASE STUDY 1 : VAM SUB-OPTIMAL, MODI IMPROVES IT ##########")
    run_mode(case_study_main(), mode="both")

    print("\n\n########## CASE STUDY 2 : BALANCED, VAM ALREADY OPTIMAL ##########")
    run_mode(case_study(), mode="both")

    print("\n\n########## CASE STUDY 3 : UNBALANCED (DUMMY COLUMN) ##########")
    run_mode(unbalanced_case(), mode="both")


if __name__ == "__main__":
    main()
