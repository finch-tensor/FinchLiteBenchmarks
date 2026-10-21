"""Minute-scale version of the chain benchmark."""

from ..queries.chain import chain_inputs, chain_query
from ..spec import BenchmarkCase, make_spec


CASES = (
    BenchmarkCase("balanced-dense", (8, 8, 8, 8), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("bottleneck-sparse", (8, 2, 16, 8), "sparse"),
)

SPEC = make_spec(
    "chain_lite",
    chain_query,
    chain_inputs,
    CASES,
    end_to_end_cases=("bottleneck-sparse",),
    lite=True,
)
