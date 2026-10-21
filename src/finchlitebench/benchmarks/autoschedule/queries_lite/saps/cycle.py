"""Minute-scale version of SAPS Cycle_5."""

from ...queries.common import graph_cases
from ...queries.saps.cycle import cycle_5_inputs, cycle_5_query
from ...spec import make_spec


SPEC = make_spec(
    "saps_cycle_5_lite",
    cycle_5_query,
    cycle_5_inputs,
    graph_cases((8,), end_to_end_size=5),
    end_to_end_cases=("e2e",),
    lite=True,
)
