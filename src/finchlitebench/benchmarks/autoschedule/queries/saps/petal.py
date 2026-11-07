import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import (
    graph_cases,
    labelled_graph,
)
from finchlitebench.benchmarks.autoschedule.spec import make_spec


def petal_6_query(E0, E1, V, P):
    return fl.einsum("ac,cb,ad,db,ae,eb,a,b->", E0, E1, E0, E1, E0, E1, V, P)


CASES = (
    graph_cases((40,), ("dense",))
    + graph_cases((1000,), ("sparse",), degree=10)
)

SPEC = make_spec(
    "saps_petal_6",
    petal_6_query,
    labelled_graph,
    CASES,
    end_to_end_cases=("n1000-sparse",),
)
