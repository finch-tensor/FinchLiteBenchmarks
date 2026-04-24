from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

import finchlite as fl
from finchlite import ffunc
from finchlite.autoschedule import Aggregate, Field, Literal, MapJoin, Table
from finchlite.autoschedule.tensor_stats import StatsInterpreter


def benchmark_matmul(stats_factory, A, B):
    i, k, j = Field("i"), Field("k"), Field("j")
    A_fl = fl.asarray(A)
    B_fl = fl.asarray(B)
    node = Aggregate(
        Literal(ffunc.add), Literal(0.0),
        MapJoin(Literal(ffunc.mul), (Table(Literal(A_fl), (i, k)), Table(Literal(B_fl), (k, j)))),
        (k,),
    )
    interp = StatsInterpreter(stats_factory=stats_factory)
    result = interp(node, OrderedDict())
    return result.estimate_non_fill_values()


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
