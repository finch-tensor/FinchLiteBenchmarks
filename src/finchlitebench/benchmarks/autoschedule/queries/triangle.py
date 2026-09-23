"""Cyclic three-table join benchmark.

Copy this file for another benchmark and replace only the query, input builder,
and case list.  Timing and result handling live in ``initialization.py``.
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


def triangle_query(A, B, C):
    """Build ``sum_ijk A[i,j] * B[j,k] * C[k,i]``."""
    i, j, k = Field("i"), Field("j"), Field("k")
    join = MapJoin(
        Literal(ffuncs.mul),
        (
            Table(Literal(A), (i, j)),
            Table(Literal(B), (j, k)),
            Table(Literal(C), (k, i)),
        ),
    )
    return Aggregate(Literal(ffuncs.add), Literal(0.0), join, (i, j, k))


def triangle_inputs(case):
    """Build the three tensors for one case."""
    n = case.shape[0]
    rng = np.random.default_rng(case.seed)
    if case.pattern == "skewed":
        A = (rng.random((n, n)) < 0.9).astype(np.float64)
        B = np.zeros((n, n), dtype=np.float64)
        C = np.zeros((n, n), dtype=np.float64)
        B[:, : min(4, n)] = 1.0
        C[: min(4, n), :] = 1.0
        return A, B, C

    density = {"dense": 0.9, "sparse": 0.1}[case.pattern]
    return tuple(
        (rng.random((n, n)) < density).astype(np.float64) for _ in range(3)
    )


CASES = tuple(
    BenchmarkCase(f"{pattern}-{n}", (n, n), pattern)
    for pattern in ("skewed", "dense", "sparse")
    for n in (16, 64)
)

SPEC = BenchmarkSpec(
    name="triangle",
    module="finchlitebench.benchmarks.autoschedule.queries.triangle",
    query=triangle_query,
    inputs=triangle_inputs,
    cases=CASES,
    scheduler_cases=tuple(case.name for case in CASES),
    end_to_end_cases=("skewed-16",),
    results_dir=Path(__file__).resolve().parents[1] / "results",
)


if __name__ == "__main__":
    run_cli(SPEC)
