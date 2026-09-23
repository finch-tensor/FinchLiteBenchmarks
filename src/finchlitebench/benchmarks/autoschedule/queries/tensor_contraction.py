"""Tensor contraction benchmark: ``R[i,l] = sum_jk A[i,j,k] B[j,k,l]``."""

from pathlib import Path
import sys

import numpy as np

from finch import ffuncs
from finch.autoschedule import Aggregate, Field, Literal, MapJoin, Table

if __package__ and "." in __package__:
    from ..initialization import BenchmarkCase, BenchmarkSpec, run_cli
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from initialization import BenchmarkCase, BenchmarkSpec, run_cli


def contraction_query(A, B):
    i, j, k, l = (Field(name) for name in ("i", "j", "k", "l"))
    join = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Literal(A), (i, j, k)),
            Table(Literal(B), (j, k, l)),
        ),
    )
    return Aggregate(Literal(ffuncs.add), Literal(0.0), join, (j, k))


def contraction_inputs(case):
    i, j, k, l = case.shape
    rng = np.random.default_rng(case.seed)
    density = {"dense": 0.9, "sparse": 0.1}[case.pattern]
    return tuple(
        (rng.random(shape) < density).astype(np.float64)
        for shape in ((i, j, k), (j, k, l))
    )


SHAPES = {
    "balanced": (24, 24, 24, 24),
    "narrow-j": (64, 4, 128, 64),
    "narrow-k": (64, 128, 4, 64),
}
CASES = tuple(
    BenchmarkCase(f"{name}-{pattern}", shape, pattern)
    for name, shape in SHAPES.items()
    for pattern in ("dense", "sparse")
)

SPEC = BenchmarkSpec(
    name="tensor_contraction",
    module="finchlitebench.benchmarks.autoschedule.queries.tensor_contraction",
    query=contraction_query,
    inputs=contraction_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=tuple(
        case.name for case in CASES if case.pattern == "sparse"
    ),
    results_dir=Path(__file__).resolve().parents[1] / "results",
)


if __name__ == "__main__":
    run_cli(SPEC)
