"""Minute-scale version of SAPS Graph_7."""

from ...queries.common import graph_cases
from ...queries.saps.graph import graph_7_inputs, graph_7_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_graph_7_lite",
    graph_7_query,
    graph_7_inputs,
    graph_cases((8,), end_to_end_size=4),
    end_to_end_cases=("e2e",),
    lite=True,
)
