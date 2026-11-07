import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import (
    graph_cases,
    labelled_graph,
)
from finchlitebench.benchmarks.autoschedule.spec import make_spec


def star_4_query(E0, E1, V, P):
    return fl.einsum("ab,ac,ad,ae,a,e->", E0, E1, E0, E1, V, P)


CASES = (
    graph_cases((40,), ("dense",))
    + graph_cases((1000,), ("sparse",), degree=10)
)

SPEC = make_spec(
    "saps_star_4",
    star_4_query,
    labelled_graph,
    CASES,
    end_to_end_cases=("n1000-sparse",),
)
