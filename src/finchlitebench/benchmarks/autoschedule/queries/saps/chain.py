"""SAPS Chain_4: a path of 4 edges, a-b-c-d-e."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def chain_4_query(G1, G2, G3, G4):
    return fl.einsum("ab,bc,cd,de->", G1, G2, G3, G4)


def chain_4_inputs(case):
    return random_graph(case, 4)


CASES = graph_cases((16, 32), end_to_end_size=8)

SPEC = make_spec(
    "saps_chain_4",
    chain_4_query,
    chain_4_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
