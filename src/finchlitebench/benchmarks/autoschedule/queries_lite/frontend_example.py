"""Example query file: copy this to add a benchmark."""

import finch as fl
from finch.autoschedule.default_schedulers import COMPILE_JULIA_GALLEY
from finch.autoschedule.tensor_stats import UniformStatsFactory

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase, make_spec


def frontend_example_query(A, B, C):
    """The query: any Finch frontend expression on the lazy input tensors."""
    return fl.sum((A @ B) * C)


def frontend_example_inputs(case):
    """The query's inputs for one case, in argument order: three n x n 0/1 matrices."""
    n = case.shape[0]
    return random_tensors(case, (n, n), (n, n), (n, n))


SPEC = make_spec(
    "frontend_example_lite",  # benchmark name, as passed to initialization.py
    frontend_example_query,
    frontend_example_inputs,
    (
        # BenchmarkCase(name, shape, pattern, seed=0, density=None)
        # shape: sizes read by the inputs function; here (n,).
        # pattern: "dense" = numpy, 90% nonzero; "sparse" = Finch tensors, 10%.
        BenchmarkCase("dense-4", (4,), "dense"),
        BenchmarkCase("sparse-8", (8,), "sparse", density=0.05),
    ),
    end_to_end_cases=("sparse-8",),  # cases that also run end-to-end; default is all
    stats=UniformStatsFactory,  # stats factory the planners use
    pipeline=COMPILE_JULIA_GALLEY,  # supplies the formatter and backend
    lite=True,  # fewer warmups and runs
)
