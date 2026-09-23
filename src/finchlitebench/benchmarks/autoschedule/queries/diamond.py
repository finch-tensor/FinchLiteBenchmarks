"""Diamond join benchmark.

The query is ``R[i] = sum_jkl A[i,j] B[j,k] C[j,l] D[k,l]``.
"""

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


def diamond_query(A, B, C, D):
    i, j, k, l = (Field(name) for name in ("i", "j", "k", "l"))
    join = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Literal(A), (i, j)),
            Table(Literal(B), (j, k)),
            Table(Literal(C), (j, l)),
            Table(Literal(D), (k, l)),
        ),
    )
    return Aggregate(Literal(ffuncs.add), Literal(0.0), join, (j, k, l))


def diamond_inputs(case):
    i, j, k, l = case.shape
    rng = np.random.default_rng(case.seed)
    density = {"dense": 0.9, "sparse": 0.1}[case.pattern]
    return tuple(
        (rng.random(shape) < density).astype(np.float64)
        for shape in ((i, j), (j, k), (j, l), (k, l))
    )


SHAPES = {
    "balanced": (24, 24, 24, 24),
    "narrow-k": (32, 32, 4, 64),
    "narrow-l": (32, 32, 64, 4),
}
CASES = tuple(
    BenchmarkCase(f"{name}-{pattern}", shape, pattern)
    for name, shape in SHAPES.items()
    for pattern in ("dense", "sparse")
)

SPEC = BenchmarkSpec(
    name="diamond",
    module="finchlitebench.benchmarks.autoschedule.queries.diamond",
    query=diamond_query,
    inputs=diamond_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=tuple(
        case.name for case in CASES if case.pattern == "sparse"
    ),
    results_dir=Path(__file__).resolve().parents[1] / "results",
)


if __name__ == "__main__":
    run_cli(SPEC)
