"""Minute-scale version of SAPS Clique_6 (K4)."""

from ...queries.common import graph_cases
from ...queries.saps.clique import clique_6_inputs, clique_6_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_clique_6_lite",
    clique_6_query,
    clique_6_inputs,
    graph_cases((8,), end_to_end_size=8),
    end_to_end_cases=("e2e",),
    lite=True,
)
