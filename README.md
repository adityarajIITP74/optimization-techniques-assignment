# Optimization Techniques — Assignment 1

> Big-M Simplex Method and the Transportation Problem (VAM + MODI), implemented
> from scratch in pure Python.

Two constrained-optimization solvers written from scratch in pure Python
(standard library only — no PuLP, no SciPy, no OR-Tools).

| File | Method | Problem solved |
|---|---|---|
| `big_m_simplex.py` | Big-M (Charnes' penalty) Simplex | Linear Programming Problem |
| `transportation.py` | VAM (initial BFS) + MODI (optimality) | Transportation Problem |

Run them with:

```bash
python3 big_m_simplex.py
python3 transportation.py
```

Both print every intermediate tableau / iteration, so the console output can
be pasted straight into the report as working.

## Repository layout

```
.
├── big_m_simplex.py                  Big-M simplex solver
├── transportation.py                 VAM + MODI solver
├── verify.py                         correctness test suite (needs scipy)
├── run_all.sh                        regenerates everything in outputs/
├── outputs/
│   ├── big_m_simplex_output.txt      full tableau-by-tableau run
│   ├── transportation_output.txt     full VAM + MODI run
│   └── verification_output.txt       test results (12/12 passing)
├── README.md
└── LICENSE
```

To regenerate all captured output:

```bash
./run_all.sh
```

---

## 1. Big-M Simplex Method

### What it does

Takes an LPP with any mix of `<=`, `>=` and `=` constraints and either sense
(`max` / `min`), converts it to standard form, and solves it.

| Constraint | Added variables |
|---|---|
| `<=` | `+ slack` |
| `>=` | `− surplus  + artificial` |
| `=`  | `+ artificial` |

Every artificial variable carries a penalty cost **M** in the objective, so the
simplex is forced to drive it out of the basis.

### Design point worth noting

**M is kept symbolic, not numeric.** Each cost and reduced cost is stored as a
pair `(constant, coefficient of M)` and compared lexicographically on the M
part first. Substituting a big number like `1e9` for M is the common shortcut,
but it causes catastrophic cancellation and can pick the wrong pivot. All
arithmetic uses `fractions.Fraction`, so results are exact rationals
(`Z* = 17/5`, not `3.4000000000000004`).

### Termination rules implemented

* all `(zⱼ − cⱼ) ≥ 0` → **optimal**
* entering column has no positive entry → **unbounded**
* optimal reached but an artificial is still basic at a positive value →
  **infeasible**
* a non-basic variable with zero reduced cost → **alternate optima** flagged

### Case studies included

**Case 1 (minimisation, textbook — Taha):**

```
Min  Z = 4x₁ + x₂
s.t. 3x₁ +  x₂  =  3
     4x₁ + 3x₂ >=  6
      x₁ + 2x₂ <=  4
```
Optimum: `x₁ = 2/5, x₂ = 9/5, Z* = 17/5 = 3.4`

**Case 2 (maximisation, product mix):** profit Rs 5/Rs 4 per unit, machine and
labour limits, plus a minimum-contract `>=` constraint.
Optimum: `x₁ = 3, x₂ = 3/2, Z* = 21`

### Solving your own LPP

```python
from big_m_simplex import BigMSimplex

s = BigMSimplex(
    c=[4, 1],
    A=[[3, 1], [4, 3], [1, 2]],
    signs=["=", ">=", "<="],
    b=[3, 6, 4],
    sense="min",
)
BigMSimplex.report(s.solve())
```

Negative right-hand sides are normalised automatically (row multiplied by −1
and the inequality flipped).

---

## 2. Transportation Problem — VAM + MODI

### Pipeline

1. **Balance** — if `Σsupply ≠ Σdemand`, a dummy row or column with zero cost
   is inserted automatically.
2. **VAM** — repeatedly compute each row/column penalty (difference between the
   two smallest remaining costs), pick the largest penalty, and allocate as
   much as possible to the cheapest cell in that line. Ties are broken toward
   the larger allocation.
3. **Degeneracy check** — a basic feasible solution must occupy exactly
   `m + n − 1` cells. If VAM leaves fewer, epsilon (zero-quantity) allocations
   are placed in the cheapest cells that do not close a loop.
4. **MODI** — set `u₁ = 0`, solve `uᵢ + vⱼ = cᵢⱼ` over occupied cells, then
   compute the opportunity cost of every empty cell:

   ```
   dᵢⱼ = cᵢⱼ − (uᵢ + vⱼ)
   ```

   * all `dᵢⱼ ≥ 0` → optimal
   * otherwise take the most negative `dᵢⱼ`, trace the unique closed loop
     through occupied cells, and shift `θ = min allocation on the “−” corners`

5. Repeat until optimal. Alternate optima are flagged when some `dᵢⱼ = 0`.

The closed-loop search is a depth-first search that strictly alternates
horizontal and vertical moves and must return to the entering cell on an even
number of steps — this is what guarantees the loop is valid.

### Running one method at a time

The assignment asks for VAM and MODI to be usable separately:

```python
run_mode(problem, mode="vam")    # initial BFS only
run_mode(problem, mode="modi")   # BFS + MODI optimisation
run_mode(problem, mode="both")   # full pipeline with commentary
```

### Case studies included

| Case | Setup | Result |
|---|---|---|
| 1 | supply (7, 9, 18), demand (5, 8, 7, 14) | VAM gives **779**, MODI improves to **743** |
| 2 | supply (300, 400, 500), demand (250, 350, 400, 200) | VAM already optimal at **2850** |
| 3 | supply (76, 82, 77) > demand (72, 102, 41) | dummy column added, optimal **2424** |

Case 1 is the important one for the report — it is the case where MODI actually
performs an improvement iteration, with the closed loop
`+(F2,DC2) → −(F2,DC4) → +(F3,DC4) → −(F3,DC2)` and `θ = 2`.

### Solving your own transportation problem

```python
from transportation import TransportationProblem, run_mode

p = TransportationProblem(
    cost=[[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20]],
    supply=[7, 9, 18],
    demand=[5, 8, 7, 14],
)
run_mode(p, mode="both")
```

---

## Verification

`verify.py` runs 12 checks — all passing (see
[`outputs/verification_output.txt`](outputs/verification_output.txt)):

* known textbook optima for both LPP case studies, including the exact
  fractional values `x₁ = 2/5, x₂ = 9/5, Z* = 17/5`
* infeasible and unbounded detection
* **60 randomly generated LPPs** with mixed `<=`, `>=`, `=` constraints
  cross-checked against `scipy.optimize.linprog` — 60/60 objective values match
* known optima for all three transportation case studies, plus a check that the
  final shipment plan satisfies every supply and demand total exactly

```bash
pip install scipy     # needed for the random cross-check block only
python3 verify.py
```

SciPy is used **only** here. Neither solver imports it at runtime — if SciPy is
absent, that one block is skipped and the remaining checks still run.

---

## Requirements

Python 3.8+ for the solvers — no third-party packages.
SciPy is optional and only used by `verify.py`.

## License

MIT — see [LICENSE](LICENSE).
