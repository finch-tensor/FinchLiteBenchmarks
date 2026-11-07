from finchlitebench.benchmarks.autoschedule.queries.common import (
    graph_cases,
    labelled_graph,
)
from finchlitebench.benchmarks.autoschedule.queries.saps.flower import flower_8_query
from finchlitebench.benchmarks.autoschedule.spec import make_spec


SPEC = make_spec(
    "saps_flower_8_lite",
    flower_8_query,
    labelled_graph,
    graph_cases((8,), end_to_end_size=3, end_to_end_seed=2),
    end_to_end_cases=("e2e",),
    lite=True,
)
