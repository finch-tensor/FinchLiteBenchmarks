"""Minute-scale version of SAPS Petal_6."""

from ...queries.common import graph_cases
from ...queries.saps.petal import petal_6_inputs, petal_6_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_petal_6_lite",
    petal_6_query,
    petal_6_inputs,
    graph_cases((8,), end_to_end_size=5),
    end_to_end_cases=("e2e",),
    lite=True,
)
