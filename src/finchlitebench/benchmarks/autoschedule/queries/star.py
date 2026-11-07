import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import cases_from_shapes, make_spec


def star_query(A, B, C):
    return fl.einsum("ij,ik,il->jkl", A, B, C)


def star_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j), (i, k), (i, l))


CASES = cases_from_shapes(
    {
        "balanced": (160, 160, 160, 160),
        "narrow-center": (20, 320, 320, 320),
        "wide-center": (640, 80, 80, 80),
    }
)

SPEC = make_spec(
    "star",
    star_query,
    star_inputs,
    CASES,
    end_to_end_cases=("balanced-sparse", "wide-center-sparse"),
)
