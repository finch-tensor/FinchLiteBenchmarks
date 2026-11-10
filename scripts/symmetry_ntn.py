"""Prototype FinchNotation kernels that check whether a square sparse matrix is
symmetric, compiled with finch-tensor's Julia backend.

Each takes the matrix in CSR (Dense over SparseList), which can only be walked
in its stored order, so none of them reads A[j, i] out of order (which would
need a transposed copy, the very thing we want to avoid):

* ``asym_residual``: one in-order pass computing
  r = sum_ij A[i,j] * (u[i]*v[j] - v[i]*u[j]) for random int64 vectors u, v.
  A symmetric A gives r == 0 exactly (the terms of (i,j) and (j,i) cancel, also
  under int64 wraparound); an asymmetric one gives r == 0 only by an unlikely
  cancellation, so r != 0 proves asymmetry and r == 0 is strong evidence.
* ``degree_mismatch``: row and column counts in one pass (the columns
  scattered into a dense vector), then the number of vertices whose counts
  differ.  Deterministic, but only a necessary condition.
* ``reciprocal_count``: exact; the number of stored (i, j) whose (j, i) is also
  stored, which equals nnz iff the pattern is symmetric.  It looks up H[j, i]
  in a random-access copy H (Dense over SparseHash), so it exercises an
  out-of-order (non-concordant) read with Follow protocols.

    PYTHONPATH=src python scripts/symmetry_ntn.py [dataset ...]
"""

from __future__ import annotations

import sys
import time

import finch as ft
import finch.finch_notation as ntn
import numpy as np
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    element,
    ffuncs,
    ftype,
)
from finch.compile.lower import make_extent
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_assembly import AssemblyKernelFType
from finch.tensor import BufferizedNDArray

I64 = np.int64
EXACT_MAX_N = 50_000  # larger Dense(SparseHash) copies need too much memory


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def csr_tensor(rows: np.ndarray, cols: np.ndarray, n: int) -> FiberTensor:
    """The n x n 0/1 matrix with entries at (rows, cols), as Dense(SparseList)."""
    key = np.unique(rows.astype(I64) * n + cols.astype(I64))
    rows, cols = key // n, key % n
    ptr = np.zeros(n + 1, dtype=np.intp)
    np.add.at(ptr, rows + 1, 1)
    np.cumsum(ptr, out=ptr)
    elem = element(I64(0), ftype(I64), ftype(np.intp), NumpyBufferFType)
    return FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem, NumpyBuffer(np.ones(len(cols), dtype=I64))),
                np.intp(n),
                NumpyBuffer(ptr),
                NumpyBuffer(cols.astype(np.intp)),
            ),
            np.intp(n),
        )
    )


def empty_hash(n: int) -> FiberTensor:
    """An empty n x n Dense(SparseHash) tensor, which supports random access;
    ``build_hash`` fills it."""
    return ft.fiber_tensor(
        ft.dense(
            ft.sparse_hash(element(I64(0), ftype(I64), ftype(np.intp)), ftype(np.intp)),
            ftype(np.intp),
        )
    ).construct((n, n))


def scalar(value=0) -> BufferizedNDArray:
    return BufferizedNDArray.from_numpy(np.array(value, dtype=I64))


# ---------------------------------------------------------------------------
# Notation helpers
# ---------------------------------------------------------------------------


def var(name, t):
    return ntn.Variable(name, t)


def lit(x):
    return ntn.Literal(x)


def call(op, *args):
    return ntn.Call(lit(op), tuple(args))


def read(slot, *idxs, protocols=()):
    return ntn.Unwrap(ntn.Access(slot, ntn.Read(tuple(protocols)), tuple(idxs)))


def extent(stop):
    return call(make_extent, lit(np.intp(0)), stop)


def function(name, args, body_stmts, outputs):
    """A notation function unpacking *args*, running *body_stmts* on their slots,
    and returning *outputs* (a subset of args)."""
    slots = {a: ntn.Slot(a.name + "_", a.result_type) for a in args}
    result = call(ffuncs.make_tuple, *outputs)
    body = ntn.Block(
        (
            *(ntn.Unpack(slots[a], a) for a in args),
            *body_stmts(slots),
            *(ntn.Repack(slots[a], a) for a in args),
            ntn.Return(result),
        )
    )
    kernel_t = AssemblyKernelFType(
        name, tuple(a.result_type for a in args), result.result_type
    )
    return ntn.Function(var(name, kernel_t), tuple(args), body)


ADD = ffuncs.add


def accumulate(slot, value, *idxs):
    return ntn.Increment(ntn.Access(slot, ntn.Update(lit(ADD)), tuple(idxs)), value)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def asym_residual_fn(a_t, vec_t, out_t):
    A, u, v, out = var("A", a_t), var("u", vec_t), var("v", vec_t), var("out", out_t)
    i, j, n = var("i", ftype(np.intp)), var("j", ftype(np.intp)), var("n", ftype(np.intp))

    def body(s):
        term = call(
            ffuncs.mul,
            read(s[A], i, j),
            call(
                ffuncs.sub,
                call(ffuncs.mul, read(s[u], i), read(s[v], j)),
                call(ffuncs.mul, read(s[v], i), read(s[u], j)),
            ),
        )
        return (
            ntn.Assign(n, ntn.Dimension(s[A], lit(0))),
            ntn.Declare(s[out], lit(I64(0)), lit(ADD), ()),
            ntn.Loop(i, extent(n), ntn.Loop(j, extent(n), accumulate(s[out], term))),
            ntn.Freeze(s[out], lit(ADD)),
        )

    return function("asym_residual", (A, u, v, out), body, (out,))


def degree_mismatch_fn(a_t, vec_t, out_t):
    A, r, c, out = var("A", a_t), var("r", vec_t), var("c", vec_t), var("out", out_t)
    i, j, n = var("i", ftype(np.intp)), var("j", ftype(np.intp)), var("n", ftype(np.intp))

    def body(s):
        a_ij = read(s[A], i, j)
        return (
            ntn.Assign(n, ntn.Dimension(s[A], lit(0))),
            ntn.Declare(s[r], lit(I64(0)), lit(ADD), (n,)),
            ntn.Declare(s[c], lit(I64(0)), lit(ADD), (n,)),
            ntn.Loop(
                i,
                extent(n),
                ntn.Loop(
                    j,
                    extent(n),
                    ntn.Block((accumulate(s[r], a_ij, i), accumulate(s[c], a_ij, j))),
                ),
            ),
            ntn.Freeze(s[r], lit(ADD)),
            ntn.Freeze(s[c], lit(ADD)),
            ntn.Declare(s[out], lit(I64(0)), lit(ADD), ()),
            ntn.Loop(
                i,
                extent(n),
                accumulate(
                    s[out],
                    call(ffuncs.astype(ft.int64), call(ffuncs.ne, read(s[r], i), read(s[c], i))),
                ),
            ),
            ntn.Freeze(s[out], lit(ADD)),
        )

    return function("degree_mismatch", (A, r, c, out), body, (r, c, out))


def build_hash_fn(a_t, h_t):
    """H[i, j] = A[i, j]: copy CSR A into the random-access H, in order."""
    A, H = var("A", a_t), var("H", h_t)
    i, j, n = var("i", ftype(np.intp)), var("j", ftype(np.intp)), var("n", ftype(np.intp))
    over = ffuncs.overwrite

    def body(s):
        return (
            ntn.Assign(n, ntn.Dimension(s[A], lit(0))),
            ntn.Declare(s[H], lit(I64(0)), lit(over), (n, n)),
            ntn.Loop(
                i,
                extent(n),
                ntn.Loop(
                    j,
                    extent(n),
                    ntn.Increment(
                        ntn.Access(s[H], ntn.Update(lit(over)), (i, j)), read(s[A], i, j)
                    ),
                ),
            ),
            ntn.Freeze(s[H], lit(over)),
        )

    return function("build_hash", (A, H), body, (H,))


def reciprocal_count_fn(a_t, h_t, out_t):
    A, H, out = var("A", a_t), var("H", h_t), var("out", out_t)
    i, j, n = var("i", ftype(np.intp)), var("j", ftype(np.intp)), var("n", ftype(np.intp))

    def body(s):
        term = call(
            ffuncs.mul,
            read(s[A], i, j),
            read(s[H], j, i, protocols=(ntn.Follow(), ntn.Follow())),
        )
        return (
            ntn.Assign(n, ntn.Dimension(s[A], lit(0))),
            ntn.Declare(s[out], lit(I64(0)), lit(ADD), ()),
            ntn.Loop(i, extent(n), ntn.Loop(j, extent(n), accumulate(s[out], term))),
            ntn.Freeze(s[out], lit(ADD)),
        )

    return function("reciprocal_count", (A, H, out), body, (out,))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def timed(f, *args, reps=3):
    f(*args)  # compile
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        out = f(*args)
        times.append(time.perf_counter() - t)
    return out, min(times)


def check(name, rows, cols, n, rng):
    A = csr_tensor(rows, cols, n)
    nnz = len(np.unique(rows.astype(I64) * n + cols.astype(I64)))
    u = ft.asarray(rng.integers(-(2**62), 2**62, n, dtype=I64))
    v = ft.asarray(rng.integers(-(2**62), 2**62, n, dtype=I64))
    vec_t, out_t = u.ftype, scalar().ftype
    print(f"== {name}: n={n}, nnz={nnz}")

    lib = FinchJLCompiler()(
        ntn.Module(
            (asym_residual_fn(A.ftype, vec_t, out_t), degree_mismatch_fn(A.ftype, vec_t, out_t))
        )
    )
    (res,), t = timed(lib.asym_residual, A, u, v, scalar())
    print(f"  asym_residual    r = {int(res.to_numpy())}  ->  "
          f"{'symmetric (w.h.p.)' if int(res.to_numpy()) == 0 else 'NOT symmetric'}   {t*1e3:8.2f} ms")
    zeros = ft.asarray(np.zeros(n, dtype=I64))
    (_, _, mism), t = timed(lib.degree_mismatch, A, zeros, ft.asarray(np.zeros(n, dtype=I64)), scalar())
    print(f"  degree_mismatch  {int(mism.to_numpy())} vertices with in-degree != out-degree"
          f"   {t*1e3:8.2f} ms")

    if n > EXACT_MAX_N:
        print(f"  reciprocal_count skipped: n > {EXACT_MAX_N} (finch-tensor's "
              "Dense(SparseHash) copy took over 23 GB for dblp's 317k rows)")
        return
    try:
        h_t = empty_hash(n).ftype
        lib2 = FinchJLCompiler()(
            ntn.Module((build_hash_fn(A.ftype, h_t), reciprocal_count_fn(A.ftype, h_t, out_t)))
        )
        (H,), t_copy = timed(lambda: lib2.build_hash(A, empty_hash(n)), reps=1)
        (cnt,), t = timed(lib2.reciprocal_count, A, H, scalar())
        print(f"  reciprocal_count {int(cnt.to_numpy())} of {nnz} entries reciprocated -> "
              f"{'symmetric' if int(cnt.to_numpy()) == nnz else 'NOT symmetric'}"
              f"   {t*1e3:8.2f} ms  (+{t_copy*1e3:.0f} ms for the SparseHash copy)")
    except Exception as e:  # noqa: BLE001
        print(f"  reciprocal_count failed: {type(e).__name__}: {str(e)[:300]}")


def scipy_check(rows, cols, n):
    import scipy.sparse as sps

    A = sps.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    t = time.perf_counter()
    same = (A != A.T).nnz == 0
    return same, time.perf_counter() - t


def main(argv):
    rng = np.random.default_rng(0)
    # Small cases: symmetric, one missing reverse edge, and a directed cycle.
    n = 6
    sym = [(0, 1), (1, 0), (1, 2), (2, 1), (3, 5), (5, 3), (4, 4)]
    cases = {
        "small symmetric": sym,
        "small, one reverse edge missing": sym[1:],
        "small directed cycle (degrees equal)": [(0, 1), (1, 2), (2, 0)],
    }
    for name, edges in cases.items():
        r, c = (np.array(x) for x in zip(*edges, strict=True))
        check(name, r, c, n, rng)
    from finchtensorbench import gcare

    for ds in argv:
        g = gcare.load_graph(ds)
        r, c = g.matrices["E0"]
        check(ds, np.asarray(r), np.asarray(c), g.num_vertices, rng)
        same, t = scipy_check(np.asarray(r), np.asarray(c), g.num_vertices)
        print(f"  scipy (A != A.T).nnz == 0: {same}   {t*1e3:8.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1:])
