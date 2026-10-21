"""SAPS Clique_6: the complete graph K4 on a, b, c, d (6 edges)."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def clique_6_query(G1, G2, G3, G4, G5, G6):
    return fl.einsum("ab,ac,ad,bc,bd,cd->", G1, G2, G3, G4, G5, G6)


def clique_6_inputs(case):
    return random_graph(case, 6)


CASES = graph_cases((16, 32), end_to_end_size=12)

SPEC = make_spec(
    "saps_clique_6",
    clique_6_query,
    clique_6_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
