"""SAPS Tree_5: acyclic query a-b, a-c, b-d, b-e, c-f (branches at a and b)."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def tree_5_query(G1, G2, G3, G4, G5):
    return fl.einsum("ab,ac,bd,be,cf->", G1, G2, G3, G4, G5)


def tree_5_inputs(case):
    return random_graph(case, 5)


CASES = graph_cases((16, 32), end_to_end_size=6)

SPEC = make_spec(
    "saps_tree_5",
    tree_5_query,
    tree_5_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
