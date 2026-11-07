from finchlitebench.benchmarks.autoschedule.queries.common import (
    graph_cases,
    labelled_graph,
)
from finchlitebench.benchmarks.autoschedule.queries.saps.graph import graph_7_query
from finchlitebench.benchmarks.autoschedule.spec import make_spec


SPEC = make_spec(
    "saps_graph_7_lite",
    graph_7_query,
    labelled_graph,
    graph_cases((8,), end_to_end_size=4),
    end_to_end_cases=("e2e",),
    lite=True,
)
