"""Shared input helpers for the autoscheduler query files."""

import numpy as np

import finch as fl

from ..spec import BenchmarkCase, cases_from_shapes


DENSITIES = {"dense": 0.9, "sparse": 0.1}


def _sparse_tensor(array):
    # Compressed fibers avoid Julia COO failures on rectangular and 3D inputs.
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
    """Build random 0/1 tensors using the case seed and density pattern."""
    rng = np.random.default_rng(case.seed)
    density = DENSITIES[case.pattern]
    arrays = tuple((rng.random(shape) < density).astype(np.float64) for shape in shapes)
    return (
        tuple(_sparse_tensor(array) for array in arrays)
        if case.pattern == "sparse"
        else arrays
    )


def random_graph(case, edge_count):
    """One symmetric 0/1 adjacency matrix without self loops, once per query edge.

    Subgraph matching joins the same data graph once for every query edge, so
    every einsum operand is the same matrix.
    """
    n = case.shape[0]
    rng = np.random.default_rng(case.seed)
    upper = np.triu(rng.random((n, n)) < DENSITIES[case.pattern], k=1)
    graph = (upper | upper.T).astype(np.float64)
    if case.pattern == "sparse":
        graph = _sparse_tensor(graph)
    return (graph,) * edge_count


def graph_cases(sizes, end_to_end_size, end_to_end_seed=0):
    """Dense and sparse graphs per size, plus one dense ``e2e`` graph.

    The default plan loops over every query vertex at once, so end-to-end time
    grows like n ** vertices; ``end_to_end_size`` keeps that bounded per query.
    """
    return cases_from_shapes({f"n{n}": (n,) for n in sizes}) + (
        BenchmarkCase("e2e", (end_to_end_size,), "dense", seed=end_to_end_seed),
    )
