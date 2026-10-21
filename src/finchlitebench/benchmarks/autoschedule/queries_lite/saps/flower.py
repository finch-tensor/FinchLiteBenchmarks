"""Minute-scale version of SAPS Flower_8."""

from ...queries.common import graph_cases
from ...queries.saps.flower import flower_8_inputs, flower_8_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_flower_8_lite",
    flower_8_query,
    flower_8_inputs,
    # n=4 takes minutes per run; seed 2 gives a 3-vertex graph with a triangle,
    # so the petal can match and the count is nonzero.
    graph_cases((8,), end_to_end_size=3, end_to_end_seed=2),
    end_to_end_cases=("e2e",),
    lite=True,
)
