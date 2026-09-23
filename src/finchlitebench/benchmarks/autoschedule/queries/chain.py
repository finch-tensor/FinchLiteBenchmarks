"""Three-table chain benchmark.

The query is ``R[i,l] = sum_jk A[i,j] * B[j,k] * C[k,l]``.
It tests both join association and loop ordering without a cycle.
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


def chain_query(A, B, C):
    """Build ``R[i,l] = sum_jk A[i,j] * B[j,k] * C[k,l]``."""
    i, j, k, l = (Field(name) for name in ("i", "j", "k", "l"))
    join = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Literal(A), (i, j)),
            Table(Literal(B), (j, k)),
            Table(Literal(C), (k, l)),
        ),
    )
    return Aggregate(Literal(ffuncs.add), Literal(0.0), join, (j, k))


def chain_inputs(case):
    """Build A(i,j), B(j,k), and C(k,l) for one case."""
    i, j, k, l = case.shape
    rng = np.random.default_rng(case.seed)
    density = {"dense": 0.9, "sparse": 0.1}[case.pattern]
    return tuple(
        (rng.random(shape) < density).astype(np.float64)
        for shape in ((i, j), (j, k), (k, l))
    )


# The two rectangular cases put the narrow dimension on opposite sides of B.
SHAPES = {
    "balanced": (32, 32, 32, 32),
    "left-bottleneck": (64, 4, 128, 64),
    "right-bottleneck": (64, 128, 4, 64),
}
CASES = tuple(
    BenchmarkCase(f"{name}-{pattern}", shape, pattern)
    for name, shape in SHAPES.items()
    for pattern in ("dense", "sparse")
)

SPEC = BenchmarkSpec(
    name="chain",
    module="finchlitebench.benchmarks.autoschedule.queries.chain",
    query=chain_query,
    inputs=chain_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=(
        "balanced-sparse",
        "left-bottleneck-sparse",
        "right-bottleneck-sparse",
    ),
    results_dir=Path(__file__).resolve().parents[1] / "results",
)


if __name__ == "__main__":
    run_cli(SPEC)
