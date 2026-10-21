"""Minute-scale version of SAPS Chain_4."""

from ...queries.common import graph_cases
from ...queries.saps.chain import chain_4_inputs, chain_4_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_chain_4_lite",
    chain_4_query,
    chain_4_inputs,
    graph_cases((8,), end_to_end_size=5),
    end_to_end_cases=("e2e",),
    lite=True,
)
