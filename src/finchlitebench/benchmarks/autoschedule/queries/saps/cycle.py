"""SAPS Cycle_5: a cycle of 5 edges, a-b-c-d-e-a."""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def cycle_5_query(G1, G2, G3, G4, G5):
    return fl.einsum("ab,bc,cd,de,ea->", G1, G2, G3, G4, G5)


def cycle_5_inputs(case):
    return random_graph(case, 5)


CASES = graph_cases((16, 32), end_to_end_size=8)

SPEC = make_spec(
    "saps_cycle_5",
    cycle_5_query,
    cycle_5_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
