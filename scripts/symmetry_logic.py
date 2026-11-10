"""Symmetry checks written as plain finch_logic programs, planned by
NON_RECURSIVE_SCHEDULER (default factorizer, loop orderer and formatter, full
dense statistics, compiled to Julia), for comparison with the hand-written
FinchNotation kernels in symmetry_ntn.py.

    PYTHONPATH=src python scripts/symmetry_logic.py [dataset ...]
"""

from __future__ import annotations

import sys
import time

import finch as ft
import numpy as np
from finch import ffuncs
from finch.autoschedule.default_schedulers import NON_RECURSIVE_SCHEDULER
from finch.finch_logic import (
    Aggregate,
    Field,
    HardAlias,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from symmetry_ntn import csr_tensor  # noqa: E402

I64 = np.int64
i, j = Field("i"), Field("j")
ZERO = Literal(I64(0))


def total(expr, *fields):
    return Aggregate(Literal(ffuncs.add), ZERO, expr, fields)


def as_int(expr):
    return MapJoin(Literal(ffuncs.astype(ft.int64)), (expr,))


def scalar_plan(name, expr):
    out = HardAlias(name)
    return Plan((Query(Table(out, ()), expr), Produces((out,))))


def mismatches(A):
    """sum_ij [A[i,j] != A[j,i]]: 0 iff symmetric."""
    a_ij, a_ji = Table(Literal(A), (i, j)), Table(Literal(A), (j, i))
    return scalar_plan("mismatches", total(as_int(MapJoin(Literal(ffuncs.ne), (a_ij, a_ji))), i, j))


def reciprocal(A):
    """sum_ij A[i,j] * A[j,i]: nnz iff the 0/1 pattern is symmetric."""
    a_ij, a_ji = Table(Literal(A), (i, j)), Table(Literal(A), (j, i))
    return scalar_plan("reciprocal", total(MapJoin(Literal(ffuncs.mul), (a_ij, a_ji)), i, j))


def residual(A, u, v):
    """sum_ij A[i,j] * (u[i] v[j] - v[i] u[j]): 0 if symmetric (randomized)."""
    U, V = Literal(u), Literal(v)
    mul, sub = Literal(ffuncs.mul), Literal(ffuncs.sub)
    term = MapJoin(
        sub,
        (
            MapJoin(mul, (Table(U, (i,)), Table(V, (j,)))),
            MapJoin(mul, (Table(V, (i,)), Table(U, (j,)))),
        ),
    )
    return scalar_plan(
        "residual", total(MapJoin(mul, (Table(Literal(A), (i, j)), term)), i, j)
    )


def degree_mismatch(A):
    """The number of vertices whose row and column counts differ."""
    rows, cols, out = HardAlias("rows"), HardAlias("cols"), HardAlias("degree_mismatch")
    return Plan(
        (
            Query(Table(rows, (i,)), total(Table(Literal(A), (i, j)), j)),
            Query(Table(cols, (j,)), total(Table(Literal(A), (i, j)), i)),
            Query(
                Table(out, ()),
                total(
                    as_int(MapJoin(Literal(ffuncs.ne), (Table(rows, (i,)), Table(cols, (i,))))),
                    i,
                ),
            ),
            Produces((out,)),
        )
    )


def run(prgm, reps=3):
    t0 = time.perf_counter()
    (result,) = NON_RECURSIVE_SCHEDULER(prgm)
    first = time.perf_counter() - t0
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        (result,) = NON_RECURSIVE_SCHEDULER(prgm)
        best = min(best, time.perf_counter() - t)
    return int(np.asarray(ft.to_numpy(result)).ravel()[0]), first, best


def check(name, rows, cols, n, rng):
    A = csr_tensor(rows, cols, n)
    nnz = len(np.unique(rows.astype(I64) * n + cols.astype(I64)))
    u = ft.asarray(rng.integers(-(2**62), 2**62, n, dtype=I64))
    v = ft.asarray(rng.integers(-(2**62), 2**62, n, dtype=I64))
    print(f"== {name}: n={n}, nnz={nnz}", flush=True)
    for label, prgm, verdict in (
        ("mismatches", mismatches(A), lambda x: x == 0),
        ("reciprocal", reciprocal(A), lambda x: x == nnz),
        ("residual", residual(A, u, v), lambda x: x == 0),
        ("degree_mismatch", degree_mismatch(A), lambda x: x == 0),
    ):
        try:
            value, first, best = run(prgm)
            print(f"  {label:16s} {value:>22}  {'symmetric' if verdict(value) else 'NOT symmetric':13s}"
                  f"  first {first*1e3:9.1f} ms  then {best*1e3:9.2f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:16s} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)


def main(argv):
    rng = np.random.default_rng(0)
    n = 6
    sym = [(0, 1), (1, 0), (1, 2), (2, 1), (3, 5), (5, 3), (4, 4)]
    for name, edges in {
        "small symmetric": sym,
        "small, one reverse edge missing": sym[1:],
        "small directed cycle": [(0, 1), (1, 2), (2, 0)],
    }.items():
        r, c = (np.array(x) for x in zip(*edges, strict=True))
        check(name, r, c, n, rng)
    from finchtensorbench import gcare

    for ds in argv:
        g = gcare.load_graph(ds)
        r, c = g.matrices["E0"]
        check(ds, np.asarray(r), np.asarray(c), g.num_vertices, rng)


if __name__ == "__main__":
    main(sys.argv[1:])
