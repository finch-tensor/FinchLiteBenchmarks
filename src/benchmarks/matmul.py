import numpy as np
import scipy.sparse as sp

import finchlite as fl


def benchmark_matmul(ctx, A, B):
    A_lazy = fl.lazy(fl.asarray(A))
    B_lazy = fl.lazy(fl.asarray(B))
    C_lazy = fl.matmul(A_lazy, B_lazy)
    return fl.compute(C_lazy, ctx=ctx)


def dg_matmul_dense_small():
    rng = np.random.default_rng(0)
    A = rng.random((32, 32), dtype=np.float32)
    B = rng.random((32, 32), dtype=np.float32)
    return (A, B)


def dg_matmul_dense_large():
    rng = np.random.default_rng(0)
    A = rng.random((4096, 4096), dtype=np.float32)
    B = rng.random((4096, 4096), dtype=np.float32)
    return (A, B)


def dg_matmul_sparse_small():
    rng = np.random.default_rng(0)
    A = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    B = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    return (A, B)


def dg_matmul_sparse_large():
    rng = np.random.default_rng(0)
    A = sp.random(
        4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng
    ).toarray()
    B = sp.random(
        4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng
    ).toarray()
    return (A, B)
