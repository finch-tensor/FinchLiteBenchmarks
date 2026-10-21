"""SAPS Graph_7: triangle a-b-c sharing edge b-c with square b-c-d-e, plus tail e-f."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def graph_7_query(G1, G2, G3, G4, G5, G6, G7):
    return fl.einsum("ab,bc,ca,cd,de,eb,ef->", G1, G2, G3, G4, G5, G6, G7)


def graph_7_inputs(case):
    return random_graph(case, 7)


CASES = graph_cases((16, 32), end_to_end_size=6)

SPEC = make_spec(
    "saps_graph_7",
    graph_7_query,
    graph_7_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
