"""SAPS Star_4: center a connected to leaves b, c, d, e."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def star_4_query(G1, G2, G3, G4):
    return fl.einsum("ab,ac,ad,ae->", G1, G2, G3, G4)


def star_4_inputs(case):
    return random_graph(case, 4)


CASES = graph_cases((16, 32), end_to_end_size=8)

SPEC = make_spec(
    "saps_star_4",
    star_4_query,
    star_4_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
