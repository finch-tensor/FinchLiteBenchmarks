from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

import finchlite as fl
from finchlite import ffuncs
from finchlite.autoschedule import Aggregate, Field, Literal, MapJoin, Table
from finchlite.autoschedule.tensor_stats import StatsInterpreter


def benchmark_matmul(stats_factory, *matrices):
    n = len(matrices)
    indices = [Field(f"i{k}") for k in range(n + 1)]
    fl_matrices = [fl.asarray(m) for m in matrices]

    node = Aggregate(
        Literal(ffuncs.add),
        Literal(0.0),
        MapJoin(
            Literal(ffuncs.mul),
            (
                Table(Literal(fl_matrices[0]), (indices[0], indices[1])),
                Table(Literal(fl_matrices[1]), (indices[1], indices[2])),
            ),
        ),
        (indices[1],),
    )
    for k in range(2, n):
        node = Aggregate(
            Literal(ffuncs.add),
            Literal(0.0),
            MapJoin(
                Literal(ffuncs.mul),
                (node, Table(Literal(fl_matrices[k]), (indices[k], indices[k + 1]))),
            ),
            (indices[k],),
        )

    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()

# ───────────────────────────────  matmul benchmark ────────────────────────────────

def matmul_dense_small():
    """
    A: dense matrix
    B: dense matrix

    Result: small dense matrix
    """
    rng = np.random.default_rng(0)
    A = rng.random((32, 32), dtype=np.float32)
    B = rng.random((32, 32), dtype=np.float32)
    return (A, B)


def matmul_sparse_small():
    """
    A: sparse matrix
    B: sparse matrix

    Result: small sparse matrix
    """
    rng = np.random.default_rng(0)
    A = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    B = sp.random(32, 32, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    return (A, B)


def matmul_dense_large():
    """
    A: dense matrix
    B: dense matrix

    Result: large dense matrix
    """
    rng = np.random.default_rng(0)
    A = rng.random((4096, 4096), dtype=np.float32)
    B = rng.random((4096, 4096), dtype=np.float32)
    return (A, B)


def matmul_sparse_large():
    """
    A: sparse matrix
    B: sparse matrix

    Result: large sparse matrix
    """
    rng = np.random.default_rng(0)
    A = sp.random(4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng).toarray()
    B = sp.random(4096, 4096, density=0.01, format="coo", dtype=np.float32, random_state=rng).toarray()
    return (A, B)


def matmul_dense_horizontal_clash():
    """
    A: dense horizontal vector (1, 64)
    B: dense matrix (64, 64)

    Result: dense horizontal vector
    """
    A = np.random.random((1, 64), dtype=np.float32)
    B = np.random.random((64, 64), dtype=np.float32)
    return (A, B)


def matmul_sparse_horizontal_clash():
    """
    A: sparse horizontal vector (1, 64)
    B: sparse matrix (64, 64)

    Result: sparse horizontal vector (1, 64)
    """
    rng = np.random.default_rng(0)
    A = sp.random(1, 64, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    B = sp.random(64, 64, density=0.1, format="coo", dtype=np.float32, random_state=rng).toarray()
    return (A, B)


def matmul_dense_column_row():
    """
    A: dense vertical vector
    B: dense horizontal vector

    Result: dense matrix
    """
    rng = np.random.default_rng(42)
    A = rng.random((64, 1))
    B = rng.random((1, 64))
    return (A, B)

def matmul_sparse_column_row():
    """
    A: sparse vertical vector
    B: sparse horizontal vector

    Result: sparse matrix
    """
    rng = np.random.default_rng(42)
    A = sp.random(64, 1, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    B = sp.random(1, 64, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    return (A, B)


def matmul_dense_col():
    """
    A: dense column on the left matrix
    B: dense matrix

    Result: dense matrix
    """
    A = np.zeros((64, 64))
    A[:, 0] = np.random.random(64)
    B = np.random.random((64, 64))
    return (A, B)


def matmul_sparse_col():
    """
    A: sparse column on the left matrix
    B: sparse matrix

    Result: sparse matrix
    """
    rng = np.random.default_rng(42)
    A = np.zeros((64, 64))
    A[:, 0] = sp.random(64, 1, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray().ravel()
    B = sp.random(64, 64, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    return (A, B)


def matmul_sparse_single_col():
    """
    A: dense matrix
    B: onn dense column on the left matrix

    Result: one single column on the left matrix
    """
    A = np.random.random((64, 64))
    B = np.zeros((64, 64))
    B[:, 0] = np.random.random(64)
    return (A, B)


def matmul_sparse_single_col_double():
    """
    A: sparse matrix
    B: sparse column

    Result: single sparse column matrix
    """
    rng = np.random.default_rng(42)
    A = sp.random(64, 64, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    B = np.zeros((64, 64))
    B[:, 0] = sp.random(64, 1, density=0.5, format="coo", dtype=np.float64, random_state=rng).toarray().ravel()
    return (A, B)


def matmul_sparse_outer_clash():
    rng = np.random.default_rng(42)
    A = sp.random(256, 8, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    B = sp.random(8, 256, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    return (A, B)


def matmul_sparse_inner_clash():
    rng = np.random.default_rng(42)
    A = sp.random(8, 256, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    B = sp.random(256, 8, density=0.1, format="coo", dtype=np.float64, random_state=rng).toarray()
    return (A, B)


def matmul_diagonal():
    A = np.eye(50, dtype=np.float64)
    B = np.eye(50, dtype=np.float64)
    return (A, B)


def matmul_tridiagonal():
    A = (np.eye(50, k=0) + np.eye(50, k=1) + np.eye(50, k=-1)).astype(np.float64)
    B = A.copy()
    return (A, B)


def matmul_banded():
    bw = 5
    r, c = np.indices((50, 50))
    A = (np.abs(r - c) <= bw).astype(np.float64)
    B = A.copy()
    return (A, B)


def matmul_upper_triangular():
    A = np.triu(np.ones((50, 50), dtype=np.float64))
    B = A.copy()
    return (A, B)


def matmul_lower_triangular():
    A = np.tril(np.ones((50, 50), dtype=np.float64))
    B = A.copy()
    return (A, B)


def matmul_triangular_clash():
    rng = np.random.default_rng(42)
    A = np.tril(rng.random((64, 64)))
    B = np.triu(rng.random((64, 64)))
    return (A, B)


def matmul_vertical_horizontal():
    A = np.zeros((50, 50), dtype=np.float64)
    A[:, ::5] = 1.0
    B = np.zeros((50, 50), dtype=np.float64)
    B[::5, :] = 1.0
    return (A, B)


def matmul_vertical_stripe():
    A = np.zeros((50, 50), dtype=np.float64)
    A[:, ::5] = 1.0
    B = np.zeros((50, 50), dtype=np.float64)
    B[:, ::5] = 1.0
    return (A, B)


def matmul_horizontal_stripe():
    A = np.zeros((50, 50), dtype=np.float64)
    A[::5, :] = 1.0
    B = np.zeros((50, 50), dtype=np.float64)
    B[::5, :] = 1.0
    return (A, B)


def matmul_mod32():
    A = np.zeros((64, 64), dtype=np.float64)
    A[::32, ::32] = 1.0
    B = np.zeros((64, 64), dtype=np.float64)
    B[::32, ::32] = 1.0
    return (A, B)


def matmul_diagonal_anti_clash():
    A = np.diag(np.random.random(64))
    B = np.fliplr(np.diag(np.random.random(64)))
    return (A, B)


def matmul_anti_anti_clash():
    A = np.fliplr(np.diag(np.random.random(64)))
    B = np.fliplr(np.diag(np.random.random(64)))
    return (A, B)


def chain_matmul_dense_10():
    rng = np.random.default_rng(0)
    return tuple(rng.random((32, 32), dtype=np.float32) for _ in range(10))


def chain_matmul_sparse_5():
    rng = np.random.default_rng(42)
    return tuple(sp.random(32, 32, density=0.1, random_state=rng).toarray() for _ in range(5))


def chain_matmul_sparse_10():
    rng = np.random.default_rng(42)
    return tuple(sp.random(32, 32, density=0.1, random_state=rng).toarray() for _ in range(10))


def chain_matmul_diagonal_10():
    return tuple(np.diag(np.random.random(32)) for _ in range(10))


def chain_matmul_diagonal_anti_10():
    return tuple(
        np.diag(np.random.random(64)) if i % 2 == 0 else np.fliplr(np.diag(np.random.random(64)))
        for i in range(10)
    )


def chain_matmul_mod32_10():
    matrices = []
    for _ in range(10):
        A = np.zeros((64, 64), dtype=np.float64)
        A[::32, ::32] = 1.0
        matrices.append(A)
    return tuple(matrices)


def chain_matmul_vertical_stripe_10():
    matrices = []
    for i in range(10):
        A = np.zeros((50, 50), dtype=np.float64)
        if i % 2 == 0:
            A[:, ::5] = 1.0
        else:
            A[::5, :] = 1.0
        matrices.append(A)
    return tuple(matrices)


def chain_matmul_triangular_clash_10():
    rng = np.random.default_rng(42)
    return tuple(
        np.tril(rng.random((64, 64))) if i % 2 == 0 else np.triu(rng.random((64, 64)))
        for i in range(10)
    )
