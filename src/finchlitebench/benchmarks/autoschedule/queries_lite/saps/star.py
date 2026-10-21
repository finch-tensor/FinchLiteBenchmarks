"""Minute-scale version of SAPS Star_4."""

from ...queries.common import graph_cases
from ...queries.saps.star import star_4_inputs, star_4_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_star_4_lite",
    star_4_query,
    star_4_inputs,
    graph_cases((8,), end_to_end_size=5),
    end_to_end_cases=("e2e",),
    lite=True,
)
