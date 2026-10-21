"""Minute-scale version of SAPS Tree_5."""

from ...queries.common import graph_cases
from ...queries.saps.tree import tree_5_inputs, tree_5_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_tree_5_lite",
    tree_5_query,
    tree_5_inputs,
    graph_cases((8,), end_to_end_size=4),
    end_to_end_cases=("e2e",),
    lite=True,
)
