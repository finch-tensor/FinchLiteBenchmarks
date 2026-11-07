from finchlitebench.benchmarks.autoschedule.queries.triangle import (
    triangle_inputs,
    triangle_query,
)
from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase, make_spec


CASES = (
    BenchmarkCase("dense-4", (4, 4), "dense"),
    BenchmarkCase("sparse-8", (8, 8), "sparse"),
    BenchmarkCase("skewed-8", (8, 8), "skewed"),
)

SPEC = make_spec(
    "triangle_lite",
    triangle_query,
    triangle_inputs,
    CASES,
    end_to_end_cases=("skewed-8",),
    lite=True,
)
