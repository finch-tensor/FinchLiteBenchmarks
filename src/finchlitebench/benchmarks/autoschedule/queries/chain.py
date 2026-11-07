import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import cases_from_shapes, make_spec


def chain_query(A, B, C):
    return fl.einsum("ij,jk,kl->il", A, B, C)


def chain_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (j, k), (k, l))


CASES = cases_from_shapes(
    {
        "balanced": (160, 160, 160, 160),
        "left-bottleneck": (320, 20, 640, 320),
        "right-bottleneck": (320, 640, 20, 320),
    }
)

SPEC = make_spec(
    "chain",
    chain_query,
    chain_inputs,
    CASES,
    end_to_end_cases=(
        "balanced-sparse",
        "left-bottleneck-sparse",
        "right-bottleneck-sparse",
    ),
)
