import finch as fl

from finchlitebench.benchmarks.autoschedule.queries.common import random_tensors
from finchlitebench.benchmarks.autoschedule.spec import cases_from_shapes, make_spec


def contraction_query(A, B):
    return fl.einsum("ijk,jkl->il", A, B)


def contraction_inputs(case):
    i, j, k, l = case.shape
    return random_tensors(case, (i, j, k), (j, k, l))


CASES = cases_from_shapes(
    {
        "balanced": (120, 120, 120, 120),
        "narrow-j": (320, 20, 640, 320),
        "narrow-k": (320, 640, 20, 320),
    }
)

SPEC = make_spec(
    "tensor_contraction",
    contraction_query,
    contraction_inputs,
    CASES,
    end_to_end_cases=tuple(case.name for case in CASES if case.pattern == "sparse"),
)
