import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import cases_from_shapes, make_spec


def diamond_query(A, B, C, D):
    return fl.einsum("ij,jk,jl,kl->i", A, B, C, D)


def diamond_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (j, k), (j, l), (k, l))


CASES = cases_from_shapes(
    {
        "balanced": (120, 120, 120, 120),
        "narrow-k": (160, 160, 20, 320),
        "narrow-l": (160, 160, 320, 20),
    }
)

SPEC = make_spec(
    "diamond",
    diamond_query,
    diamond_inputs,
    CASES,
    end_to_end_cases=tuple(case.name for case in CASES if case.pattern == "sparse"),
)
