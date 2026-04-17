import os

import numpy as np
import ssgetpy
from scipy.io import mmread
from scipy.sparse import random

import finchlite as fl


def benchmark_cg(ctx, A_bench, b_bench, x_bench, rel_tol=1e-8, abs_tol=1e-20, max_iters=10_000):
    A = fl.lazy(fl.asarray(np.asarray(A_bench)))
    b = fl.asarray(np.asarray(b_bench).flatten())
    x = fl.asarray(np.asarray(x_bench).flatten())

    bb = float(fl.compute(fl.vecdot(fl.lazy(b), fl.lazy(b)), ctx=ctx)[()])
    tolerance = max(rel_tol * np.sqrt(bb), abs_tol)
    tol_sq = tolerance * tolerance

    r = fl.asarray(np.asarray(b) - np.asarray(fl.compute(fl.matmul(A, fl.lazy(fl.asarray(np.asarray(x)[:, None]))), ctx=ctx)).flatten())
    p = r
    it = 0
    rr = float(fl.compute(fl.vecdot(fl.lazy(r), fl.lazy(r)), ctx=ctx)[()])

    if rr >= tol_sq:
        while it < max_iters:
            Ap = fl.asarray(np.asarray(fl.compute(fl.matmul(A, fl.lazy(fl.asarray(np.asarray(p)[:, None]))), ctx=ctx)).flatten())
            alpha = rr / float(fl.compute(fl.vecdot(fl.lazy(r), fl.lazy(Ap)), ctx=ctx)[()])
            x = fl.asarray(np.asarray(x) + alpha * np.asarray(p))
            r = fl.asarray(np.asarray(r) - alpha * np.asarray(Ap))
            new_rr = float(fl.compute(fl.vecdot(fl.lazy(r), fl.lazy(r)), ctx=ctx)[()])

            it += 1

            if new_rr < tol_sq:
                break

            beta = new_rr / rr
            p = fl.asarray(np.asarray(r) + beta * np.asarray(p))
            rr = new_rr

    return np.asarray(x)


def generate_cg_data(source, has_b_file=False):
    matrices = ssgetpy.search(name=source)
    if not matrices:
        raise ValueError(f"No matrix found with name '{source}'")
    matrix = matrices[0]
    (path, archive) = matrix.download(extract=True)
    matrix_path = os.path.join(path, matrix.name + ".mtx")
    if matrix_path and os.path.exists(matrix_path):
        A = mmread(matrix_path)
    else:
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
    rng = np.random.default_rng(0)
    A = A.tocoo()

    if has_b_file:
        matrix_path = os.path.join(path, matrix.name + "_b.mtx")
        if matrix_path and os.path.exists(matrix_path):
            b = mmread(matrix_path)
        else:
            raise FileNotFoundError(f"Matrix file not found at {matrix_path}")
        if not isinstance(b, np.ndarray):
            b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        b = b.flatten()
    else:
        x = random(A.shape[1], 1, density=0.1, format="coo", dtype=np.float64, random_state=rng)
        b = A @ x
        b = b.toarray().flatten()
    x = np.zeros(A.shape[1])

    return (fl.asarray(A.toarray()), fl.asarray(b), fl.asarray(x))


def dg_cg_sparse_1():
    return generate_cg_data("mesh3em5")


def dg_cg_sparse_2():
    return generate_cg_data("bcsstm02")


def dg_cg_sparse_3():
    return generate_cg_data("fv1")


def dg_cg_sparse_4():
    return generate_cg_data("Muu")


def dg_cg_sparse_5():
    return generate_cg_data("Chem97ZtZ")


def dg_cg_sparse_6():
    return generate_cg_data("Dubcova1")


def dg_cg_sparse_7():
    return generate_cg_data("t3dl_e")


def dg_cg_sparse_8():
    return generate_cg_data("bcsstk09")
