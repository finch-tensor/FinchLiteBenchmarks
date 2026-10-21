"""Minute-scale version of the star benchmark."""

from ..queries.star import star_inputs, star_query
from ..spec import BenchmarkCase, make_spec


CASES = (
    BenchmarkCase("balanced-dense", (6, 6, 6, 6), "dense"),
    BenchmarkCase("balanced-sparse", (8, 8, 8, 8), "sparse"),
    BenchmarkCase("narrow-center-sparse", (2, 12, 12, 12), "sparse"),
)

SPEC = make_spec(
    "star_lite",
    star_query,
    star_inputs,
    CASES,
    end_to_end_cases=("narrow-center-sparse",),
    lite=True,
)
