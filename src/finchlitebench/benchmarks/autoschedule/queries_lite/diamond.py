from finchlitebench.benchmarks.autoschedule.queries.diamond import (
    diamond_inputs,
    diamond_query,
)
from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase, make_spec


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-k-sparse", (8, 8, 2, 12), "sparse"),
)

SPEC = make_spec(
    "diamond_lite",
    diamond_query,
    diamond_inputs,
    CASES,
    end_to_end_cases=("narrow-k-sparse",),
    lite=True,
)
