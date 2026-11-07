from finchlitebench.benchmarks.autoschedule.queries.tensor_contraction import (
    contraction_inputs,
    contraction_query,
)
from finchlitebench.benchmarks.autoschedule.spec import BenchmarkCase, make_spec


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-j-sparse", (8, 2, 12, 8), "sparse"),
)

SPEC = make_spec(
    "tensor_contraction_lite",
    contraction_query,
    contraction_inputs,
    CASES,
    end_to_end_cases=("narrow-j-sparse",),
    lite=True,
)
