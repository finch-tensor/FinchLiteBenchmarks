"""Star join benchmark: ``R[j,k,l] = sum_i A[i,j] B[i,k] C[i,l]``."""

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


def star_query(A, B, C):
    i, j, k, l = (Field(name) for name in ("i", "j", "k", "l"))
    join = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Literal(A), (i, j)),
            Table(Literal(B), (i, k)),
            Table(Literal(C), (i, l)),
        ),
    )
    return Aggregate(Literal(ffuncs.add), Literal(0.0), join, (i,))


def star_inputs(case):
    i, j, k, l = case.shape
    rng = np.random.default_rng(case.seed)
    density = {"dense": 0.9, "sparse": 0.1}[case.pattern]
    return tuple(
        (rng.random(shape) < density).astype(np.float64)
        for shape in ((i, j), (i, k), (i, l))
    )


SHAPES = {
    "balanced": (32, 32, 32, 32),
    "narrow-center": (4, 64, 64, 64),
    "wide-center": (128, 16, 16, 16),
}
CASES = tuple(
    BenchmarkCase(f"{name}-{pattern}", shape, pattern)
    for name, shape in SHAPES.items()
    for pattern in ("dense", "sparse")
)

SPEC = BenchmarkSpec(
    name="star",
    module="finchlitebench.benchmarks.autoschedule.queries.star",
    query=star_query,
    inputs=star_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("balanced-sparse", "wide-center-sparse"),
    results_dir=Path(__file__).resolve().parents[1] / "results",
)


if __name__ == "__main__":
    run_cli(SPEC)
