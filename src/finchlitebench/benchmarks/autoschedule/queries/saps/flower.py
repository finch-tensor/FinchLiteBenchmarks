"""SAPS Flower_8: center a with three attachments.

* chain: a-b-c
* tree: a-d, d-e, d-f
* petal: a and h joined by a-h and a-g-h
"""

import finch as fl

from ...spec import make_spec
from ..common import graph_cases, random_graph


def flower_8_query(G1, G2, G3, G4, G5, G6, G7, G8):
    return fl.einsum(
        "ab,bc,ad,de,df,ah,ag,gh->", G1, G2, G3, G4, G5, G6, G7, G8
    )


def flower_8_inputs(case):
    return random_graph(case, 8)


CASES = graph_cases((16, 32), end_to_end_size=4)

SPEC = make_spec(
    "saps_flower_8",
    flower_8_query,
    flower_8_inputs,
    CASES,
    end_to_end_cases=("e2e",),
)
