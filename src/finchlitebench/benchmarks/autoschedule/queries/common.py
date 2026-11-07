"""Random input data shared by the query files."""

import numpy as np

import finch as fl

from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase


DENSITIES = {"dense": 0.9, "sparse": 0.1}


def density(case):
    """Fraction of nonzeros: the case's own density, else its pattern's default."""
    return DENSITIES.get(case.pattern) if case.density is None else case.density


def _sparse_tensor(array):
    """Compress a numpy array into a sparse Finch tensor."""
    nonzero = array != 0
    counts = np.count_nonzero(nonzero, axis=-1).reshape(-1)
    ptr = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts)))
    indices = np.ascontiguousarray(np.nonzero(nonzero)[-1], dtype=np.int64)
    level = fl.SparseListLevel(
        fl.ElementLevel(
            fl.element(0.0, element_type=fl.float64, position_type=fl.int64),
            fl.NumpyBuffer(array[nonzero]),
        ),
        np.int64(array.shape[-1]),
        ptr=fl.NumpyBuffer(ptr),
        idx=fl.NumpyBuffer(indices),
    )
    for size in reversed(array.shape[:-1]):
        level = fl.DenseLevel(level, np.int64(size))
    return fl.FiberTensor(level)


def random_tensors(case, *shapes):
    """One random 0/1 tensor per shape, stored sparse for "sparse" cases."""
    rng = np.random.default_rng(case.seed)
    arrays = tuple(
        (rng.random(shape) < density(case)).astype(np.float64) for shape in shapes
    )
    if case.pattern == "sparse":
        return tuple(_sparse_tensor(array) for array in arrays)
    return arrays


def labelled_graph(case):
    """Random graph (E0, E1, V, P): two edge labels, type-0 mask, pinned vertex."""
    n = case.shape[0]
    rng = np.random.default_rng(case.seed)
    upper = np.triu(rng.random((n, n)) < density(case), k=1)
    label = rng.integers(2, size=(n, n))
    E0 = upper & (label == 0)
    E1 = upper & (label == 1)
    E0, E1 = ((E | E.T).astype(np.float64) for E in (E0, E1))
    V = (rng.integers(2, size=n) == 0).astype(np.float64)
    P = np.zeros(n)
    P[rng.integers(n)] = 1.0
    arrays = (E0, E1, V, P)
    if case.pattern == "sparse":
        return tuple(_sparse_tensor(array) for array in arrays)
    return arrays


def graph_cases(
    sizes,
    patterns=("dense", "sparse"),
    degree=None,
    end_to_end_size=None,
    end_to_end_seed=0,
):
    """Cases "n{size}-{pattern}", plus an optional small dense "e2e" graph."""
    cases = tuple(
        BenchmarkCase(
            f"n{n}-{pattern}",
            (n,),
            pattern,
            density=degree / n if degree else None,
        )
        for n in sizes
        for pattern in patterns
    )
    if end_to_end_size is None:
        return cases
    return cases + (
        BenchmarkCase("e2e", (end_to_end_size,), "dense", seed=end_to_end_seed),
    )
