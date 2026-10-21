"""SAPS Petal_6: a and b joined by three vertex-disjoint paths a-c-b, a-d-b, a-e-b."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def petal_6_query(G1, G2, G3, G4, G5, G6):
    return fl.einsum("ac,cb,ad,db,ae,eb->", G1, G2, G3, G4, G5, G6)


def petal_6_inputs(case):
    return random_graph(case, 6)


CASES = graph_cases((16, 32), end_to_end_size=8)

SPEC = make_spec(
    "saps_petal_6",
    petal_6_query,
    petal_6_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
