from finchlitebench.benchmarks.autoschedule.queries.common import (
    graph_cases,
    labelled_graph,
)
from finchlitebench.benchmarks.autoschedule.queries.saps.petal import petal_6_query
from finchlitebench.benchmarks.autoschedule.spec import make_spec


SPEC = make_spec(
    "saps_petal_6_lite",
    petal_6_query,
    labelled_graph,
    graph_cases((8,), end_to_end_size=5),
    end_to_end_cases=("e2e",),
    lite=True,
)
