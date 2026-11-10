"""Prototype: a counting-sort transpose of a CSR matrix in FinchNotation.

CSR A (Dense over SparseList) -> CSR T = A^T, as scipy's csc_tocsr does it:

1. count:   cnt[j] += 1 for every stored (i, j)                (column degrees)
2. scan:    ptr[j] = sum_{k<j} cnt[k], ptr[n] = nnz             (row starts of T)
3. scatter: for i in order, for j in row i of A:
               idx[ptr[j] + seen[j]] = i;  val[...] = A[i,j];  seen[j] += 1

Steps 2 and 3 index tensors by computed positions and step 3 reads a counter
while bumping it, which tensor-level notation may not allow, so ``probe()``
first tries each construct on its own.

    PYTHONPATH=src python scripts/transpose_ntn.py [dataset ...]
"""

from __future__ import annotations

import sys
import time
import traceback

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


def scalar(value=0) -> BufferizedNDArray:
    return BufferizedNDArray.from_numpy(np.array(value, dtype=I64))


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


IDX = ftype(np.intp)
OVER = ffuncs.overwrite


def store(slot, index, value):
    return ntn.Increment(ntn.Access(slot, ntn.Update(lit(OVER)), (index,)), value)


def probe_kernels(vec_t, a_t):
    """Small kernels, each using one construct the transpose needs."""
    x, y = var("x", vec_t), var("y", vec_t)
    j, n, acc, pos = var("j", IDX), var("n", IDX), var("acc", ftype(I64)), var("pos", IDX)

    # (a) write at a computed index: y[j + 1] = x[j]
    def shifted(s):
        return (
            ntn.Assign(n, ntn.Dimension(s[x], lit(0))),
            ntn.Declare(s[y], lit(I64(0)), lit(OVER), (call(ffuncs.add, n, lit(np.intp(1))),)),
            ntn.Loop(j, extent(n), store(s[y], call(ffuncs.add, j, lit(np.intp(1))), read(s[x], j))),
            ntn.Freeze(s[y], lit(OVER)),
        )

    # (b) a loop-carried scalar: y[j] = sum_{k<j} x[k]  (exclusive scan)
    def scan(s):
        return (
            ntn.Assign(n, ntn.Dimension(s[x], lit(0))),
            ntn.Assign(acc, lit(I64(0))),
            ntn.Declare(s[y], lit(I64(0)), lit(OVER), (n,)),
            ntn.Loop(
                j,
                extent(n),
                ntn.Block((store(s[y], j, acc), ntn.Assign(acc, call(ffuncs.add, acc, read(s[x], j))))),
            ),
            ntn.Freeze(s[y], lit(OVER)),
        )

    # (c) index by a value read from a tensor: y[x[j]] = j  (inverse permutation)
    def gather_index(s):
        return (
            ntn.Assign(n, ntn.Dimension(s[x], lit(0))),
            ntn.Declare(s[y], lit(I64(0)), lit(OVER), (n,)),
            ntn.Loop(
                j,
                extent(n),
                ntn.Block(
                    (
                        ntn.Assign(pos, call(ffuncs.astype(ft.intp), read(s[x], j))),
                        store(s[y], pos, call(ffuncs.astype(ft.int64), j)),
                    )
                ),
            ),
            ntn.Freeze(s[y], lit(OVER)),
        )

    # (d) read a tensor while updating it: y[j] = y[j] + x[j]
    def read_while_updating(s):
        return (
            ntn.Assign(n, ntn.Dimension(s[x], lit(0))),
            ntn.Declare(s[y], lit(I64(0)), lit(OVER), (n,)),
            ntn.Loop(
                j, extent(n), store(s[y], j, call(ffuncs.add, read(s[y], j), read(s[x], j)))
            ),
            ntn.Freeze(s[y], lit(OVER)),
        )

    return {
        "computed index y[j+1]": function("shifted", (x, y), shifted, (y,)),
        "loop-carried scalar (scan)": function("scan", (x, y), scan, (y,)),
        "index read from a tensor y[x[j]]": function("gather_index", (x, y), gather_index, (y,)),
        "read while updating y[j] += ...": function("read_while_updating", (x, y), read_while_updating, (y,)),
    }


def probe():
    x = ft.asarray(np.array([2, 0, 3, 1], dtype=I64))
    expected = {
        "computed index y[j+1]": [0, 2, 0, 3, 1],
        "loop-carried scalar (scan)": [0, 2, 2, 5],
        "index read from a tensor y[x[j]]": [1, 3, 0, 2],
        "read while updating y[j] += ...": [2, 0, 3, 1],
    }
    sizes = {"computed index y[j+1]": 5}
    for label, fn in probe_kernels(x.ftype, None).items():
        y = ft.asarray(np.zeros(sizes.get(label, 4), dtype=I64))
        try:
            lib = FinchJLCompiler()(ntn.Module((fn,)))
            (out,) = getattr(lib, fn.name.name)(x, y)
            got = list(np.asarray(out.to_numpy()))
            ok = got == expected[label]
            print(f"  {label:36s} {'OK' if ok else 'WRONG'}  {got}")
        except Exception as e:  # noqa: BLE001
            msg = "".join(traceback.format_exception_only(e)).strip().splitlines()
            print(f"  {label:36s} FAILED  {msg[0][:160]}")
            for line in msg[1:3]:
                print(f"  {'':36s}         {line[:160]}")


if __name__ == "__main__":
    print("probing notation constructs:")
    probe()
