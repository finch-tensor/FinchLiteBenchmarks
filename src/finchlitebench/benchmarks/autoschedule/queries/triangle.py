import finch as fl
import numpy as np

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase, make_spec


def triangle_query(A, B, C):
    return fl.einsum("ij,jk,ki->", A, B, C)


def triangle_inputs(case):
    n = case.shape[0]
    if case.pattern != "skewed":
        return random_tensors(case, (n, n), (n, n), (n, n))

    rng = np.random.default_rng(case.seed)
    A = (rng.random((n, n)) < 0.9).astype(np.float64)
    B = np.zeros((n, n), dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    B[:, : min(4, n)] = 1.0
    C[: min(4, n), :] = 1.0
    return A, B, C


CASES = tuple(
    BenchmarkCase(f"{pattern}-{n}", (n, n), pattern)
    for pattern in ("skewed", "dense", "sparse")
    for n in (80, 320)
)

SPEC = make_spec(
    "triangle",
    triangle_query,
    triangle_inputs,
    CASES,
    end_to_end_cases=("skewed-80",),
)
