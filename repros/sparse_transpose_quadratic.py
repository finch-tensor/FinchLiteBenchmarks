"""Repro: transposing a sparse matrix costs O(n^2) instead of O(nnz).

Status: FIXED on finch-tensor-lite ``kbd/workspace-loop-orderer`` @ 00869086
("Loop copies in the order of their argument"): copies now walk their source
and scatter into a ``Dense(SparseHash)`` output, e.g.
``A_3[i_5, i_6] = A_7[follow(i_6), walk(i_5)]``, and every case below scales
linearly (<= 2.1x per doubling).  Kept as a regression test; the text below
describes the bug as found.

Base: finch-tensor-lite branch ``kbd/workspace-loop-orderer`` @ 8f51a7d2
(Finch.jl 2.1.x).  Same behaviour on ``main`` @ d53438cb.

Problem
-------
Any program that needs a sparse matrix in the other storage order --
``permute_dims(A, (1, 0))``, or reading CSR column-first as in
``einsum("ij,kj->", A, A)`` -- materializes a transposed copy, and that copy is
lowered as a *gather* over the whole index space:

    fuse(i_1):
        fuse(i_0):
            A_T[i_1, i_0] = A[follow(i_0), follow(i_1)]

Both loops run over the full extent ``n`` and every iteration does a random
``follow`` lookup into the sparse source, so the kernel is O(n^2) although
``A`` has O(n) nonzeros.  It is not a format problem: the input is CSR
(``Dense(SparseList(Element))``) and the chosen output formats are sparse too.

Measured on ``main`` @ d53438cb with 2 nonzeros per row (execution of a cached
plan; compilation excluded), time grows ~4x per doubling of n:

    permute_dims, COMPILE_JULIA_GALLEY   n=5k 0.145s  10k 0.62s  20k 2.89s
    einsum ij,kj->, COMPILE_JULIA_GALLEY n=5k 0.105s  10k 0.54s  20k 2.60s  40k 10.0s
    einsum ij,kj->, COMPILE_JULIA        n=10k 2.26s  20k 8.63s

Impact: G-CARE AIDS subgraph-counting queries (n = 254,156, ~550k edges) take
~11 minutes, nearly all of it in these transpose kernels.

Where it comes from (src/finch/autoschedule/loop_orderer/loop_ordering.py)
-------------------------------------------------------------------------
* ``concordize`` (line 44, called at line 317) makes every discordant Table
  read a transposed copy of its tensor.
* ``with_copy_loop_orders`` (line 148, called at line 325) gives each copy the
  loop order of its *left-hand side* ("A copy may read its argument in another
  order, representing a transpose").  For a transpose that is precisely the
  order the source cannot be iterated in: neither source level can be walked,
  the protocolizer can only pick ``follow`` for both, and nothing sparse
  bounds either loop.  ``permute_dims`` goes through the same copy path.

Goal
----
A sparse transpose / discordant copy should cost O(nnz + n), e.g. by looping
in the *source's* storage order (walking its stored entries) and writing out of
order into an output level that accepts unordered inserts (hash / COO /
bytemap, converted afterwards if needed), or by a dedicated transpose lowering.
Results must stay exact and existing tests must pass.

Acceptance
----------
``python repros/sparse_transpose_quadratic.py`` exits 0: every case matches
scipy and the time ratio per doubling of n stays below 3 (it is ~4-5 today;
linear would be ~2).  ``--plan`` prints the program and formats handed to the
compiler for the smallest case, to check that the transpose kernel now walks
the source.

Run with finch-tensor[julia] installed; the first compute pays about a minute
of Julia start-up and JIT, which is excluded from the timings.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse as sps

import finch as fl
from finch.autoschedule.compiler import CompilerFormLowerer
from finch.autoschedule.default_schedulers import COMPILE_JULIA, COMPILE_JULIA_GALLEY

SIZES = (5_000, 10_000, 20_000)
PER_ROW = 2
MAX_RATIO = 3.0  # quadratic: ~4 per doubling; linear: ~2


def random_csr(n: int, per_row: int = PER_ROW, seed: int = 0):
    """An n x n 0/1 matrix as a finch CSR tensor, plus the scipy original."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), per_row)
    cols = rng.integers(0, n, n * per_row)
    m = sps.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    m.sum_duplicates()
    m.data[:] = 1.0
    m.sort_indices()
    values = fl.ElementLevel(
        fl.element(np.float64(0), fl.ftype(np.float64), fl.ftype(np.intp), fl.NumpyBufferFType),
        fl.NumpyBuffer(m.data.astype(np.float64)),
    )
    tensor = fl.FiberTensor(
        fl.DenseLevel(
            fl.SparseListLevel(
                values,
                np.intp(n),
                fl.NumpyBuffer(m.indptr.astype(np.intp)),
                fl.NumpyBuffer(m.indices.astype(np.intp)),
            ),
            np.intp(n),
        )
    )
    return tensor, m


def _transpose(A, m):
    """Materialize A^T; check it against scipy."""
    out = fl.permute_dims(fl.defer(A), (1, 0))

    def check(result, ctx):
        # Format-independent fingerprint (the result may be CSR or hash-based):
        # sum_ij R[i,j] u[i] v[j] must equal u^T (A^T) v.
        rng = np.random.default_rng(1)
        u, v = rng.random(m.shape[0]), rng.random(m.shape[1])
        prog = fl.einsum("ij,i,j->", fl.defer(result), fl.defer(u), fl.defer(v))
        got = fl.compute(prog, ctx=ctx)
        got = float(np.asarray(fl.to_numpy(got)).ravel()[0])
        want = float(u @ (m.T @ v))
        assert np.isclose(got, want, rtol=1e-9), f"transpose is wrong: {got} != {want}"

    return out, check


def _reversed_edge(A, m):
    """sum_{i,j,k} A[i,j] * A[k,j]: the second read is column-first on CSR."""
    a = fl.defer(A)
    out = fl.einsum("ij,kj->", a, a)

    def check(result, ctx):
        got = float(np.asarray(fl.to_numpy(result)).ravel()[0])
        want = float((m @ m.T).sum())
        assert got == want, f"wrong count: {got} != {want}"

    return out, check


CASES = {"permute_dims(A, (1, 0))": _transpose, 'einsum("ij,kj->", A, A)': _reversed_edge}
SCHEDULERS = {"COMPILE_JULIA_GALLEY": COMPILE_JULIA_GALLEY, "COMPILE_JULIA": COMPILE_JULIA}


def scaling_check() -> int:
    failed = False
    for case_name, case in CASES.items():
        for sched_name, scheduler in SCHEDULERS.items():
            times = []
            try:
                for n in SIZES:
                    A, m = random_csr(n)
                    program, check = case(A, m)
                    fl.compute(program, ctx=scheduler)  # plan + compile; cached after
                    start = time.perf_counter()
                    result = fl.compute(program, ctx=scheduler)  # execution only
                    times.append(time.perf_counter() - start)
                    check(result, scheduler)
            except Exception as error:  # report and keep checking the others
                failed = True
                print(f"[ERROR] {case_name:26} {sched_name:21} {type(error).__name__}: {error}")
                continue
            ratios = [b / a for a, b in zip(times, times[1:], strict=False)]
            worst = max(ratios)
            status = "FAIL" if worst >= MAX_RATIO else "ok"
            failed |= worst >= MAX_RATIO
            timings = "  ".join(f"n={n}: {t:.3f}s" for n, t in zip(SIZES, times, strict=True))
            print(
                f"[{status}] {case_name:26} {sched_name:21} {timings}  "
                f"ratio/doubling {' '.join(f'{r:.2f}' for r in ratios)}",
                flush=True,
            )
    if failed:
        print(f"FAIL: superlinear scaling (ratio >= {MAX_RATIO}): sparse transpose is O(n^2)")
        return 1
    print("OK: transpose cost scales linearly")
    return 0


def print_plan() -> int:
    """Print the program and formats each case hands to the compiler (n=1000)."""
    original = CompilerFormLowerer.lower

    def lower(self, prgm, bindings, *args):
        print(prgm)
        for alias, ftype in bindings.items():
            print(f"  {alias}: {ftype}")
        return original(self, prgm, bindings, *args)

    CompilerFormLowerer.lower = lower
    try:
        for case_name, case in CASES.items():
            A, m = random_csr(1_000)
            print(f"==== {case_name} under COMPILE_JULIA_GALLEY ====")
            program, check = case(A, m)
            check(fl.compute(program, ctx=COMPILE_JULIA_GALLEY), COMPILE_JULIA_GALLEY)
    finally:
        CompilerFormLowerer.lower = original
    return 0


if __name__ == "__main__":
    sys.exit(print_plan() if "--plan" in sys.argv else scaling_check())
