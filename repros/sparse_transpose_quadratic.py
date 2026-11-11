"""Check that transposing a sparse matrix costs O(nnz), not O(n^2).

Any program that needs a sparse matrix in the other storage order --
``permute_dims(A, (1, 0))``, or reading CSR column-first as in
``einsum("ij,kj->", A, A)`` -- materializes a transposed copy.  That copy
should walk the source's stored entries and scatter into an output that
accepts unordered inserts, e.g. ``A_3[i_5, i_6] = A_7[follow(i_6), walk(i_5)]``
into ``Dense(SparseHash)``.  Lowered as a gather over the whole index space
instead (``follow`` on both loops), it is O(n^2): about 4x slower per doubling
of n, which on G-CARE AIDS (n = 254,156) dominates every query.

Each case is run under ``COMPILE_JULIA_GALLEY`` and ``COMPILE_JULIA`` on random
CSR matrices with 2 nonzeros per row at n = 5k, 10k and 20k, timing the
execution of a cached plan.  The script exits 0 when every result matches
scipy and the time ratio per doubling of n stays below 3 (linear is ~2).

    python repros/sparse_transpose_quadratic.py         # scaling check
    python repros/sparse_transpose_quadratic.py --plan  # print each case's plan

``--plan`` prints the program and formats handed to the compiler at n = 1000,
to see how the transpose is lowered.  Needs finch-tensor[julia]; the first
compute pays about a minute of Julia start-up, which is not timed.
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
